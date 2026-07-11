# SPDX-License-Identifier: Apache-2.0

"""Independent durable ledger for the Memory helpfulness model run.

The ledger commits all 384 preregistered logical call slots before execution.
It is deliberately separate from the user-facing Memory Service SQLite store:
experiment evidence and user memory have different schemas, lifetimes, and
failure semantics.  This module provides the schema, strict snapshot loader,
and ordered run-root calculation.  The single-cursor runner is layered above
it in a later stage.  Hashes detect accidental corruption and bind exported
artifacts; without an externally published signature or MAC, they do not
protect against an adversary who can rewrite the database and recompute hashes.
The inode checks fail closed on ordinary path replacement during one session;
they do not defend against malicious ABA replacement or restoration of an old,
otherwise valid database.  Preventing rollback needs an external monotonic
anchor or an idempotent remote request registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    AuditedModelCallExecutionV2,
    AuditedModelCallReceiptV2,
    CallPlanV2,
    RunEnvelopeV2,
    audited_model_call_execution_v2_from_artifacts,
    audited_model_call_receipt_v2_bytes,
    audited_model_call_receipt_v2_from_bytes,
    infbridge_run_envelope_v2_bytes,
    infbridge_run_envelope_v2_from_bytes,
    infbridge_run_envelope_v2_sha256,
    validate_infbridge_run_envelope_v2,
)

from areal.v2.inference_service.client_trace import (
    generation_physical_trace_bytes,
    generation_response_evidence_bytes,
)
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

__all__ = [
    "LedgerArtifactCommitmentV1",
    "RunLedgerError",
    "RunLedgerReceiptSlotV1",
    "RunLedgerReceiptSnapshotV1",
    "RunLedgerSlotSnapshotV1",
    "RunLedgerSnapshotV1",
    "RunLedgerSessionV1",
    "initialize_run_ledger",
    "load_run_ledger",
    "load_run_ledger_receipt_snapshot_v1",
]

LedgerSlotState = Literal[
    "PLANNED",
    "STARTED",
    "SUCCEEDED",
    "ATTRITION",
    "INDETERMINATE",
]
LedgerRunStatus = Literal["OPEN", "SEALED"]


class RunLedgerError(RuntimeError):
    """Closed reason for rejecting a run-ledger operation."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("ledger error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class RunLedgerSlotSnapshotV1:
    plan: CallPlanV2
    plan_sha256: str
    state: LedgerSlotState
    attempt_count: int
    receipt_bytes: bytes | None
    trace_bytes: bytes | None
    response_evidence_bytes: bytes | None
    decoded_response_utf8: bytes | None
    terminal_reason: str | None
    leaf_sha256: str


@dataclass(frozen=True, slots=True)
class RunLedgerSnapshotV1:
    schema_version: int
    ledger_policy: str
    run_id: str
    manifest_sha256: str
    run_envelope_sha256: str
    call_count: int
    status: LedgerRunStatus
    seal_kind: str | None
    stored_run_root_sha256: str | None
    computed_run_root_sha256: str
    slots: tuple[RunLedgerSlotSnapshotV1, ...]


@dataclass(frozen=True, slots=True)
class LedgerArtifactCommitmentV1:
    """Content-blind size and digest for one persisted ledger artifact."""

    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RunLedgerReceiptSlotV1:
    """One terminal slot projected without returning response content."""

    plan: CallPlanV2
    plan_sha256: str
    state: Literal["SUCCEEDED", "ATTRITION"]
    attempt_count: int
    canonical_receipt: AuditedModelCallReceiptV2 | None
    receipt_commitment: LedgerArtifactCommitmentV1 | None
    trace_commitment: LedgerArtifactCommitmentV1 | None
    response_evidence_commitment: LedgerArtifactCommitmentV1 | None
    decoded_response_commitment: LedgerArtifactCommitmentV1 | None
    terminal_reason: str | None
    ledger_leaf_sha256: str


@dataclass(frozen=True, slots=True)
class RunLedgerReceiptSnapshotV1:
    """Answer-free validation report, not a capability for later trust."""

    schema_version: int
    projection_policy: str
    ledger_policy: str
    run_id: str
    manifest_sha256: str
    run_envelope_sha256: str
    call_count: int
    seal_kind: Literal["complete", "complete_with_attrition"]
    run_root_sha256: str
    slots: tuple[RunLedgerReceiptSlotV1, ...]


_APPLICATION_ID = 0x41524C31  # ASCII "ARL1"
_SCHEMA_VERSION = 1
_LEDGER_POLICY = "single-cursor-prewrite-v1"
_RECEIPT_PROJECTION_POLICY = "sealed-answer-content-blind-receipts-v1"
_MIN_SQLITE_VERSION = (3, 11, 0)
_BUSY_TIMEOUT_MS = 5_000
_CALL_COUNT = helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 8_388_608
_MAX_ENVELOPE_BYTES = 8_388_608
_MAX_PLAN_BYTES = 16_384
_MAX_RECEIPT_BYTES = 16_384
_MAX_TRACE_BYTES = 8_388_608
_MAX_RESPONSE_EVIDENCE_BYTES = 8_388_608
_MAX_DECODED_RESPONSE_BYTES = 1_048_576
_MAX_TOTAL_SLOT_ARTIFACT_BYTES = 134_217_728

_RUN_ID_DOMAIN = b"areal-memory-run-ledger-id-v1\0"
_PLAN_DOMAIN = b"areal-memory-run-ledger-plan-v1\0"
_LEAF_DOMAIN = b"areal-memory-run-ledger-leaf-v1\0"
_ROOT_HEADER_DOMAIN = b"areal-memory-run-ledger-root-header-v1\0"
_ROOT_FOLD_DOMAIN = b"areal-memory-run-ledger-root-fold-v1\0"

_ATTRITION_REASONS = frozenset(
    {
        "attempt_limit",
        "decode_failure",
        "decoder_mismatch",
        "generation_failure",
        "receipt_mismatch",
        "response_evidence",
        "run_envelope",
        "runtime_mismatch",
        "trace_mismatch",
        "unsupported_backend",
    }
)
_INDETERMINATE_REASONS = frozenset(
    {
        "artifact_validation_failure",
        "commit_ambiguous",
        "orphan_started",
        "runner_cancelled",
        "started_record_lost",
        "terminal_persistence_failure",
        "unexpected_failure",
    }
)
_SEAL_KINDS = frozenset(
    {
        "complete",
        "complete_with_attrition",
        "indeterminate",
    }
)

_SCHEMA_DDL = (
    """CREATE TABLE run_ledger_schema_metadata (
    singleton INTEGER NOT NULL PRIMARY KEY
        CHECK (typeof(singleton) = 'integer') CHECK (singleton = 1),
    schema_spec_sha256 TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_spec_sha256) = 'text')
        CHECK (length(schema_spec_sha256) = 64),
    schema_catalog_sha256 TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_catalog_sha256) = 'text')
        CHECK (length(schema_catalog_sha256) = 64)
)""",
    """CREATE TABLE run_ledger_header (
    singleton INTEGER NOT NULL PRIMARY KEY
        CHECK (typeof(singleton) = 'integer') CHECK (singleton = 1),
    ledger_policy TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(ledger_policy) = 'text'),
    run_id TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(run_id) = 'text') CHECK (length(run_id) = 64),
    manifest_sha256 TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(manifest_sha256) = 'text')
        CHECK (length(manifest_sha256) = 64),
    manifest_bytes BLOB NOT NULL CHECK (typeof(manifest_bytes) = 'blob'),
    run_envelope_sha256 TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(run_envelope_sha256) = 'text')
        CHECK (length(run_envelope_sha256) = 64),
    run_envelope_bytes BLOB NOT NULL
        CHECK (typeof(run_envelope_bytes) = 'blob'),
    call_count INTEGER NOT NULL CHECK (typeof(call_count) = 'integer')
        CHECK (call_count = 384),
    status TEXT NOT NULL COLLATE BINARY CHECK (typeof(status) = 'text')
        CHECK (status IN ('OPEN','SEALED')),
    seal_kind TEXT COLLATE BINARY
        CHECK (seal_kind IS NULL OR typeof(seal_kind) = 'text')
        CHECK (seal_kind IS NULL OR seal_kind IN
            ('complete','complete_with_attrition','indeterminate')),
    run_root_sha256 TEXT COLLATE BINARY
        CHECK (run_root_sha256 IS NULL OR typeof(run_root_sha256) = 'text')
        CHECK (run_root_sha256 IS NULL OR length(run_root_sha256) = 64),
    CHECK ((status = 'OPEN' AND seal_kind IS NULL AND run_root_sha256 IS NULL)
        OR (status = 'SEALED' AND seal_kind IS NOT NULL
            AND run_root_sha256 IS NOT NULL))
)""",
    """CREATE TABLE run_ledger_slots (
    slot_index INTEGER NOT NULL PRIMARY KEY CHECK (typeof(slot_index) = 'integer')
        CHECK (slot_index BETWEEN 0 AND 383),
    case_index INTEGER NOT NULL CHECK (typeof(case_index) = 'integer')
        CHECK (case_index BETWEEN 0 AND 63),
    arm TEXT NOT NULL COLLATE BINARY CHECK (typeof(arm) = 'text')
        CHECK (arm IN ('current_release','raw_history','memory_off',
            'target_masked','stale_release','oracle')),
    request_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(request_id) = 'text'),
    plan_bytes BLOB NOT NULL CHECK (typeof(plan_bytes) = 'blob'),
    plan_sha256 TEXT NOT NULL COLLATE BINARY CHECK (typeof(plan_sha256) = 'text')
        CHECK (length(plan_sha256) = 64),
    state TEXT NOT NULL COLLATE BINARY CHECK (typeof(state) = 'text')
        CHECK (state IN
            ('PLANNED','STARTED','SUCCEEDED','ATTRITION','INDETERMINATE')),
    attempt_count INTEGER NOT NULL CHECK (typeof(attempt_count) = 'integer')
        CHECK (attempt_count IN (0,1)),
    receipt_bytes BLOB
        CHECK (receipt_bytes IS NULL OR typeof(receipt_bytes) = 'blob'),
    trace_bytes BLOB CHECK (trace_bytes IS NULL OR typeof(trace_bytes) = 'blob'),
    response_evidence_bytes BLOB
        CHECK (response_evidence_bytes IS NULL
            OR typeof(response_evidence_bytes) = 'blob'),
    decoded_response_utf8 BLOB
        CHECK (decoded_response_utf8 IS NULL
            OR typeof(decoded_response_utf8) = 'blob'),
    terminal_reason TEXT COLLATE BINARY
        CHECK (terminal_reason IS NULL OR typeof(terminal_reason) = 'text')
        CHECK (terminal_reason IS NULL OR length(terminal_reason) BETWEEN 1 AND 128),
    leaf_sha256 TEXT NOT NULL COLLATE BINARY CHECK (typeof(leaf_sha256) = 'text')
        CHECK (length(leaf_sha256) = 64),
    UNIQUE (case_index, arm),
    UNIQUE (request_id),
    CHECK ((state = 'PLANNED' AND attempt_count = 0
            AND receipt_bytes IS NULL AND trace_bytes IS NULL
            AND response_evidence_bytes IS NULL AND decoded_response_utf8 IS NULL
            AND terminal_reason IS NULL)
        OR (state = 'STARTED' AND attempt_count = 1
            AND receipt_bytes IS NULL AND trace_bytes IS NULL
            AND response_evidence_bytes IS NULL AND decoded_response_utf8 IS NULL
            AND terminal_reason IS NULL)
        OR (state = 'SUCCEEDED' AND attempt_count = 1
            AND receipt_bytes IS NOT NULL AND trace_bytes IS NOT NULL
            AND response_evidence_bytes IS NOT NULL
            AND decoded_response_utf8 IS NOT NULL AND terminal_reason IS NULL)
        OR (state IN ('ATTRITION','INDETERMINATE') AND attempt_count = 1
            AND receipt_bytes IS NULL AND trace_bytes IS NULL
            AND response_evidence_bytes IS NULL AND decoded_response_utf8 IS NULL
            AND terminal_reason IS NOT NULL))
)""",
)

_REQUIRED_TABLES = frozenset(
    {
        "run_ledger_schema_metadata",
        "run_ledger_header",
        "run_ledger_slots",
    }
)
_EXPECTED_CATALOG_ROWS = tuple(
    sorted(
        (
            (
                "table",
                "run_ledger_schema_metadata",
                "run_ledger_schema_metadata",
                _SCHEMA_DDL[0],
            ),
            ("table", "run_ledger_header", "run_ledger_header", _SCHEMA_DDL[1]),
            ("table", "run_ledger_slots", "run_ledger_slots", _SCHEMA_DDL[2]),
        ),
        key=lambda row: (row[0], row[1], row[2], row[3]),
    )
)
_CATALOG_SQL = """SELECT type, name, tbl_name, sql
FROM main.sqlite_master
WHERE sql IS NOT NULL AND substr(name, 1, 7) <> 'sqlite_'
ORDER BY type COLLATE BINARY, name COLLATE BINARY,
         tbl_name COLLATE BINARY, sql COLLATE BINARY"""
_HEADER_SELECT = """SELECT singleton, ledger_policy, run_id,
manifest_sha256, manifest_bytes, run_envelope_sha256, run_envelope_bytes,
call_count, status, seal_kind, run_root_sha256
FROM run_ledger_header"""
_HEADER_LENGTH_SELECT = """SELECT length(manifest_bytes),
length(run_envelope_bytes) FROM run_ledger_header"""
_SLOT_SELECT = """SELECT slot_index, case_index, arm, request_id,
plan_bytes, plan_sha256, state, attempt_count, receipt_bytes, trace_bytes,
response_evidence_bytes, decoded_response_utf8, terminal_reason, leaf_sha256
FROM run_ledger_slots ORDER BY slot_index"""
_SLOT_SELECT_BY_INDEX = """SELECT slot_index, case_index, arm, request_id,
plan_bytes, plan_sha256, state, attempt_count, receipt_bytes, trace_bytes,
response_evidence_bytes, decoded_response_utf8, terminal_reason, leaf_sha256
FROM run_ledger_slots WHERE slot_index = ?"""
_SLOT_LENGTH_SELECT = """SELECT slot_index, length(plan_bytes),
coalesce(length(receipt_bytes), 0), coalesce(length(trace_bytes), 0),
coalesce(length(response_evidence_bytes), 0),
coalesce(length(decoded_response_utf8), 0)
FROM run_ledger_slots ORDER BY slot_index"""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _schema_spec_sha256() -> str:
    return _sha256(_canonical_json_bytes(list(_SCHEMA_DDL)))


_SCHEMA_SPEC_SHA256 = "7c747c9c213695739a8d28f65a3df19afba4a680433e4a553baaae4ac741cb55"
_EXPECTED_CATALOG_SHA256 = (
    "c7cf712c96fb609412c676b932e82d38c564f7de335d2f1c5cc675619ee23cdc"
)


def _plan_value(plan: CallPlanV2) -> dict[str, object]:
    return {
        "kind": "areal-memory-run-ledger-call-plan-v1",
        "slot_index": plan.slot_index,
        "case_index": plan.case_index,
        "arm": plan.arm,
        "request_id": plan.request_id,
        "input_token_ids_sha256": plan.input_token_ids_sha256,
        "input_token_count": plan.input_token_count,
        "expected_endpoint": plan.expected_endpoint,
        "expected_method": plan.expected_method,
        "first_prepared_request_json_sha256": (plan.first_prepared_request_json_sha256),
    }


def _plan_bytes(plan: CallPlanV2) -> bytes:
    return _canonical_json_bytes(_plan_value(plan))


def _plan_sha256(plan_bytes: bytes) -> str:
    return _sha256(_PLAN_DOMAIN + plan_bytes)


def _artifact_commitment(value: bytes | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {"byte_count": len(value), "sha256": _sha256(value)}


def _slot_leaf_sha256(
    *,
    slot_index: int,
    plan_sha256: str,
    state: str,
    attempt_count: int,
    receipt_bytes: bytes | None,
    trace_bytes: bytes | None,
    response_evidence_bytes: bytes | None,
    decoded_response_utf8: bytes | None,
    terminal_reason: str | None,
) -> str:
    value = {
        "attempt_count": attempt_count,
        "decoded_response_utf8": _artifact_commitment(decoded_response_utf8),
        "plan_sha256": plan_sha256,
        "receipt": _artifact_commitment(receipt_bytes),
        "response_evidence": _artifact_commitment(response_evidence_bytes),
        "slot_index": slot_index,
        "state": state,
        "terminal_reason": terminal_reason,
        "trace": _artifact_commitment(trace_bytes),
    }
    return _sha256(_LEAF_DOMAIN + _canonical_json_bytes(value))


def _run_root_sha256(
    *,
    run_id: str,
    manifest_sha256: str,
    run_envelope_sha256: str,
    status: str,
    seal_kind: str | None,
    slots: tuple[RunLedgerSlotSnapshotV1, ...],
) -> str:
    header = {
        "call_count": _CALL_COUNT,
        "ledger_policy": _LEDGER_POLICY,
        "manifest_sha256": manifest_sha256,
        "run_envelope_sha256": run_envelope_sha256,
        "run_id": run_id,
        "schema_version": _SCHEMA_VERSION,
        "seal_kind": seal_kind,
        "status": status,
    }
    root = hashlib.sha256(_ROOT_HEADER_DOMAIN + _canonical_json_bytes(header)).digest()
    for slot in slots:
        root = hashlib.sha256(
            _ROOT_FOLD_DOMAIN
            + root
            + slot.plan.slot_index.to_bytes(8, "big", signed=False)
            + bytes.fromhex(slot.leaf_sha256)
        ).digest()
    return root.hex()


@dataclass(frozen=True, slots=True)
class _RunIdentity:
    run_id: str
    manifest_bytes: bytes
    manifest_sha256: str
    envelope_bytes: bytes
    envelope_sha256: str


def _run_identity(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> _RunIdentity:
    try:
        manifest_bytes = helpfulness.model_run_manifest_bytes(manifest, tokenizer)
        manifest_sha256 = _sha256(manifest_bytes)
        envelope_bytes = infbridge_run_envelope_v2_bytes(envelope)
        infbridge_run_envelope_v2_from_bytes(envelope_bytes)
        validate_infbridge_run_envelope_v2(manifest, tokenizer, envelope)
        envelope_sha256 = infbridge_run_envelope_v2_sha256(envelope)
    except Exception as error:
        raise RunLedgerError("ledger_identity") from error
    if (
        manifest_sha256 != envelope.manifest_sha256
        or len(envelope.call_plans) != _CALL_COUNT
        or len(manifest_bytes) > _MAX_MANIFEST_BYTES
        or len(envelope_bytes) > _MAX_ENVELOPE_BYTES
        or envelope.evidence_policy.max_response_json_bytes_per_attempt > 1_048_576
        or envelope.evidence_policy.max_response_evidence_bytes_per_call
        > _MAX_RESPONSE_EVIDENCE_BYTES
    ):
        raise RunLedgerError("ledger_identity")
    run_id = _sha256(
        _RUN_ID_DOMAIN + bytes.fromhex(manifest_sha256) + bytes.fromhex(envelope_sha256)
    )
    return _RunIdentity(
        run_id=run_id,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        envelope_bytes=envelope_bytes,
        envelope_sha256=envelope_sha256,
    )


def _snapshot_database_path(database_path: str | os.PathLike[str]) -> str:
    try:
        raw_path = os.fspath(database_path)
    except TypeError as error:
        raise RunLedgerError("ledger_path") from error
    if not isinstance(raw_path, str):
        raise RunLedgerError("ledger_path")
    path = str.__str__(raw_path)
    if type(path) is not str:
        try:
            path = path.encode("utf-8", "strict").decode("utf-8", "strict")
        except UnicodeError as error:
            raise RunLedgerError("ledger_path") from error
    if (
        not path.strip()
        or "\x00" in path
        or path == ":memory:"
        or path.startswith("//")
    ):
        raise RunLedgerError("ledger_path")
    try:
        path.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise RunLedgerError("ledger_path") from error
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    canonical_parent = os.path.realpath(parent)
    return os.path.join(canonical_parent, os.path.basename(absolute))


def _require_private_regular_database(path: str) -> tuple[int, int]:
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError as error:
        raise RunLedgerError("ledger_persistence") from error
    except OSError as error:
        raise RunLedgerError("ledger_persistence") from error
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_mode & 0o077
        or (hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid())
    ):
        raise RunLedgerError("ledger_path")
    return file_stat.st_dev, file_stat.st_ino


def _require_expected_database(
    path: str,
    expected_file_identity: tuple[int, int],
) -> tuple[int, int]:
    try:
        observed = _require_private_regular_database(path)
    except RunLedgerError as error:
        raise RunLedgerError("ledger_path") from error
    if observed != expected_file_identity:
        raise RunLedgerError("ledger_path")
    return observed


def _fsync_parent_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(os.path.dirname(path), flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise RunLedgerError("ledger_persistence") from error


def _create_private_database_file(path: str) -> tuple[int, int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        _require_private_regular_database(path)
        raise RunLedgerError("ledger_exists") from error
    except OSError as error:
        raise RunLedgerError("ledger_persistence") from error
    try:
        file_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
            or file_stat.st_mode & 0o077
            or (hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid())
        ):
            raise RunLedgerError("ledger_path")
        os.fsync(file_descriptor)
        identity = (file_stat.st_dev, file_stat.st_ino)
    finally:
        os.close(file_descriptor)
    _fsync_parent_directory(path)
    _require_expected_database(path, identity)
    return identity


def _strict_text_factory(value: bytes) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RunLedgerError("ledger_corruption") from error


def _sqlite_ledger_error(error: sqlite3.Error) -> RunLedgerError:
    error_code = getattr(error, "sqlite_errorcode", None)
    base_code = error_code & 0xFF if type(error_code) is int else None
    corruption_codes = {
        getattr(sqlite3, "SQLITE_CORRUPT", -1),
        getattr(sqlite3, "SQLITE_NOTADB", -1),
    }
    reason = (
        "ledger_corruption" if base_code in corruption_codes else "ledger_persistence"
    )
    return RunLedgerError(reason)


def _read_integer_pragma(cursor: sqlite3.Cursor, name: str) -> int:
    row = cursor.execute(f"PRAGMA {name}").fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int:
        raise RunLedgerError("ledger_schema")
    return row[0]


def _require_pragma(cursor: sqlite3.Cursor, name: str, expected: int) -> None:
    if _read_integer_pragma(cursor, name) != expected:
        raise RunLedgerError("ledger_schema")


def _require_delete_journal(cursor: sqlite3.Cursor) -> None:
    journal = cursor.execute("PRAGMA journal_mode").fetchone()
    if journal != ("delete",):
        raise RunLedgerError("ledger_schema")


def _acquire_read_lock(cursor: sqlite3.Cursor) -> None:
    row = cursor.execute("SELECT count(*) FROM sqlite_master").fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int:
        raise RunLedgerError("ledger_schema")


def _connect(
    path: str,
    *,
    expected_file_identity: tuple[int, int] | None = None,
) -> sqlite3.Connection:
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        raise RunLedgerError("ledger_schema")
    observed_file_identity = (
        _require_private_regular_database(path)
        if expected_file_identity is None
        else _require_expected_database(path, expected_file_identity)
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{Path(path).as_uri()}?mode=rw",
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
            uri=True,
        )
        _require_expected_database(path, observed_file_identity)
        connection.text_factory = _strict_text_factory
        cursor = connection.cursor()
        _require_delete_journal(cursor)
        cursor.execute("PRAGMA foreign_keys = ON")
        _require_pragma(cursor, "foreign_keys", 1)
        cursor.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        _require_pragma(cursor, "busy_timeout", _BUSY_TIMEOUT_MS)
        cursor.execute("PRAGMA synchronous = EXTRA")
        _require_pragma(cursor, "synchronous", 3)
        cursor.execute("PRAGMA fullfsync = ON")
        _require_pragma(cursor, "fullfsync", 1)
        return connection
    except BaseException as error:
        try:
            if connection is not None:
                connection.close()
        except BaseException as close_error:
            error.add_note(
                "close failed without replacing the connection error: "
                f"{type(close_error).__name__}: {close_error}"
            )
        if isinstance(error, RunLedgerError):
            raise
        if isinstance(error, sqlite3.Error):
            raise _sqlite_ledger_error(error) from error
        raise


def _catalog_rows(cursor: sqlite3.Cursor) -> tuple[tuple[str, str, str, str], ...]:
    try:
        rows = cursor.execute(_CATALOG_SQL).fetchall()
    except sqlite3.Error as error:
        raise _sqlite_ledger_error(error) from error
    result: list[tuple[str, str, str, str]] = []
    for row in rows:
        if len(row) != 4 or any(type(value) is not str for value in row):
            raise RunLedgerError("ledger_schema")
        result.append((row[0], row[1], row[2], row[3]))
    return tuple(result)


def _catalog_sha256(cursor: sqlite3.Cursor) -> str:
    return _sha256(_canonical_json_bytes(_catalog_rows(cursor)))


def _initialize_schema_locked(cursor: sqlite3.Cursor) -> None:
    if (
        _schema_spec_sha256() != _SCHEMA_SPEC_SHA256
        or _sha256(_canonical_json_bytes(_EXPECTED_CATALOG_ROWS))
        != _EXPECTED_CATALOG_SHA256
    ):
        raise RunLedgerError("ledger_schema")
    for statement in _SCHEMA_DDL:
        cursor.execute(statement)
    catalog_sha256 = _catalog_sha256(cursor)
    if (
        _catalog_rows(cursor) != _EXPECTED_CATALOG_ROWS
        or catalog_sha256 != _EXPECTED_CATALOG_SHA256
    ):
        raise RunLedgerError("ledger_schema")
    cursor.execute(
        "INSERT INTO run_ledger_schema_metadata VALUES (?, ?, ?)",
        (1, _SCHEMA_SPEC_SHA256, catalog_sha256),
    )
    cursor.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
    cursor.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _validate_schema_locked(
    cursor: sqlite3.Cursor,
    *,
    full_integrity: bool = True,
) -> None:
    if (
        _read_integer_pragma(cursor, "application_id") != _APPLICATION_ID
        or _read_integer_pragma(cursor, "user_version") != _SCHEMA_VERSION
        or _schema_spec_sha256() != _SCHEMA_SPEC_SHA256
        or _sha256(_canonical_json_bytes(_EXPECTED_CATALOG_ROWS))
        != _EXPECTED_CATALOG_SHA256
    ):
        raise RunLedgerError("ledger_schema")
    catalog_rows = _catalog_rows(cursor)
    tables = {name for kind, name, _table, _sql in catalog_rows if kind == "table"}
    if (
        tables != _REQUIRED_TABLES
        or catalog_rows != _EXPECTED_CATALOG_ROWS
        or len(catalog_rows) != len(_REQUIRED_TABLES)
    ):
        raise RunLedgerError("ledger_schema")
    metadata = cursor.execute(
        "SELECT singleton, schema_spec_sha256, schema_catalog_sha256 "
        "FROM run_ledger_schema_metadata"
    ).fetchall()
    if len(metadata) != 1 or metadata[0] != (
        1,
        _SCHEMA_SPEC_SHA256,
        _EXPECTED_CATALOG_SHA256,
    ):
        raise RunLedgerError("ledger_schema")
    if full_integrity:
        if cursor.execute("PRAGMA foreign_key_check").fetchall():
            raise RunLedgerError("ledger_corruption")
        if cursor.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise RunLedgerError("ledger_corruption")


def _initialize_or_validate_schema_locked(
    cursor: sqlite3.Cursor,
    *,
    prelock_page_count: int,
) -> bool:
    application_id = _read_integer_pragma(cursor, "application_id")
    user_version = _read_integer_pragma(cursor, "user_version")
    catalog = _catalog_rows(cursor)
    if application_id == _APPLICATION_ID and user_version == _SCHEMA_VERSION:
        _validate_schema_locked(cursor)
        return False
    if (
        prelock_page_count == 0
        and application_id == 0
        and user_version == 0
        and not catalog
    ):
        _initialize_schema_locked(cursor)
        _validate_schema_locked(cursor)
        return True
    raise RunLedgerError("ledger_schema")


def _insert_run_locked(
    cursor: sqlite3.Cursor,
    identity: _RunIdentity,
    envelope: RunEnvelopeV2,
) -> None:
    cursor.execute(
        "INSERT INTO run_ledger_header VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            _LEDGER_POLICY,
            identity.run_id,
            identity.manifest_sha256,
            sqlite3.Binary(identity.manifest_bytes),
            identity.envelope_sha256,
            sqlite3.Binary(identity.envelope_bytes),
            _CALL_COUNT,
            "OPEN",
            None,
            None,
        ),
    )
    for plan in envelope.call_plans:
        plan_bytes = _plan_bytes(plan)
        plan_sha256 = _plan_sha256(plan_bytes)
        leaf_sha256 = _slot_leaf_sha256(
            slot_index=plan.slot_index,
            plan_sha256=plan_sha256,
            state="PLANNED",
            attempt_count=0,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason=None,
        )
        cursor.execute(
            "INSERT INTO run_ledger_slots VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.slot_index,
                plan.case_index,
                plan.arm,
                plan.request_id,
                sqlite3.Binary(plan_bytes),
                plan_sha256,
                "PLANNED",
                0,
                None,
                None,
                None,
                None,
                None,
                leaf_sha256,
            ),
        )


def _require_header_identity(
    row: tuple[object, ...],
    identity: _RunIdentity,
) -> tuple[str, str | None, str | None]:
    if len(row) != 11:
        raise RunLedgerError("ledger_corruption")
    (
        singleton,
        ledger_policy,
        run_id,
        manifest_sha256,
        manifest_bytes,
        envelope_sha256,
        envelope_bytes,
        call_count,
        status,
        seal_kind,
        stored_root,
    ) = row
    if (
        singleton != 1
        or ledger_policy != _LEDGER_POLICY
        or run_id != identity.run_id
        or manifest_sha256 != identity.manifest_sha256
        or type(manifest_bytes) is not bytes
        or manifest_bytes != identity.manifest_bytes
        or envelope_sha256 != identity.envelope_sha256
        or type(envelope_bytes) is not bytes
        or envelope_bytes != identity.envelope_bytes
        or call_count != _CALL_COUNT
    ):
        raise RunLedgerError("ledger_identity")
    if (
        type(status) is not str
        or status not in ("OPEN", "SEALED")
        or (seal_kind is not None and seal_kind not in _SEAL_KINDS)
        or (stored_root is not None and not _is_sha256(stored_root))
        or (status == "OPEN" and (seal_kind is not None or stored_root is not None))
        or (status == "SEALED" and (seal_kind is None or stored_root is None))
    ):
        raise RunLedgerError("ledger_corruption")
    return status, seal_kind, stored_root  # type: ignore[return-value]


def _optional_blob(value: object) -> bytes | None:
    if value is None:
        return None
    if type(value) is not bytes:
        raise RunLedgerError("ledger_corruption")
    return value


def _load_slot(
    row: tuple[object, ...],
    expected_plan: CallPlanV2,
) -> RunLedgerSlotSnapshotV1:
    if len(row) != 14:
        raise RunLedgerError("ledger_corruption")
    (
        slot_index,
        case_index,
        arm,
        request_id,
        stored_plan_bytes,
        stored_plan_sha256,
        state,
        attempt_count,
        receipt_value,
        trace_value,
        response_evidence_value,
        decoded_response_value,
        terminal_reason,
        stored_leaf_sha256,
    ) = row
    expected_plan_bytes = _plan_bytes(expected_plan)
    expected_plan_sha256 = _plan_sha256(expected_plan_bytes)
    if (
        type(slot_index) is not int
        or slot_index != expected_plan.slot_index
        or type(case_index) is not int
        or case_index != expected_plan.case_index
        or arm != expected_plan.arm
        or request_id != expected_plan.request_id
        or type(stored_plan_bytes) is not bytes
        or stored_plan_bytes != expected_plan_bytes
        or stored_plan_sha256 != expected_plan_sha256
        or type(state) is not str
        or state
        not in ("PLANNED", "STARTED", "SUCCEEDED", "ATTRITION", "INDETERMINATE")
        or type(attempt_count) is not int
        or type(terminal_reason) not in (str, type(None))
        or not _is_sha256(stored_leaf_sha256)
    ):
        raise RunLedgerError("ledger_corruption")
    receipt_bytes = _optional_blob(receipt_value)
    trace_bytes = _optional_blob(trace_value)
    response_evidence_bytes = _optional_blob(response_evidence_value)
    decoded_response_utf8 = _optional_blob(decoded_response_value)
    no_artifacts = (
        receipt_bytes is None
        and trace_bytes is None
        and response_evidence_bytes is None
        and decoded_response_utf8 is None
    )
    if (
        (
            state == "PLANNED"
            and (attempt_count != 0 or not no_artifacts or terminal_reason is not None)
        )
        or (
            state == "STARTED"
            and (attempt_count != 1 or not no_artifacts or terminal_reason is not None)
        )
        or (
            state == "SUCCEEDED"
            and (
                attempt_count != 1
                or receipt_bytes is None
                or trace_bytes is None
                or response_evidence_bytes is None
                or decoded_response_utf8 is None
                or terminal_reason is not None
            )
        )
        or (
            state == "ATTRITION"
            and (
                attempt_count != 1
                or not no_artifacts
                or terminal_reason not in _ATTRITION_REASONS
            )
        )
        or (
            state == "INDETERMINATE"
            and (
                attempt_count != 1
                or not no_artifacts
                or terminal_reason not in _INDETERMINATE_REASONS
            )
        )
    ):
        raise RunLedgerError("ledger_corruption")
    leaf_sha256 = _slot_leaf_sha256(
        slot_index=slot_index,
        plan_sha256=expected_plan_sha256,
        state=state,
        attempt_count=attempt_count,
        receipt_bytes=receipt_bytes,
        trace_bytes=trace_bytes,
        response_evidence_bytes=response_evidence_bytes,
        decoded_response_utf8=decoded_response_utf8,
        terminal_reason=terminal_reason,
    )
    if leaf_sha256 != stored_leaf_sha256:
        raise RunLedgerError("ledger_corruption")
    return RunLedgerSlotSnapshotV1(
        plan=expected_plan,
        plan_sha256=expected_plan_sha256,
        state=state,  # type: ignore[arg-type]
        attempt_count=attempt_count,
        receipt_bytes=receipt_bytes,
        trace_bytes=trace_bytes,
        response_evidence_bytes=response_evidence_bytes,
        decoded_response_utf8=decoded_response_utf8,
        terminal_reason=terminal_reason,
        leaf_sha256=leaf_sha256,
    )


def _validate_slot_sequence(
    status: str,
    seal_kind: str | None,
    slots: tuple[RunLedgerSlotSnapshotV1, ...],
) -> None:
    suffix_started = False
    started_count = 0
    indeterminate_count = 0
    for slot in slots:
        if not suffix_started and slot.state in ("SUCCEEDED", "ATTRITION"):
            continue
        if not suffix_started:
            suffix_started = True
            if slot.state == "STARTED":
                started_count += 1
            elif slot.state == "INDETERMINATE":
                indeterminate_count += 1
            elif slot.state != "PLANNED":
                raise RunLedgerError("ledger_corruption")
            continue
        if slot.state != "PLANNED":
            raise RunLedgerError("ledger_corruption")
    if status == "OPEN":
        if seal_kind is not None or started_count > 1 or indeterminate_count != 0:
            raise RunLedgerError("ledger_corruption")
        return
    if started_count != 0:
        raise RunLedgerError("ledger_corruption")
    states = tuple(slot.state for slot in slots)
    if seal_kind == "complete":
        valid = all(state == "SUCCEEDED" for state in states)
    elif seal_kind == "complete_with_attrition":
        valid = all(state in ("SUCCEEDED", "ATTRITION") for state in states) and any(
            state == "ATTRITION" for state in states
        )
    elif seal_kind == "indeterminate":
        valid = indeterminate_count == 1
    else:
        valid = False
    if not valid:
        raise RunLedgerError("ledger_corruption")


def _validate_blob_budgets_locked(cursor: sqlite3.Cursor) -> None:
    header_lengths = cursor.execute(_HEADER_LENGTH_SELECT).fetchall()
    if (
        len(header_lengths) != 1
        or len(header_lengths[0]) != 2
        or any(type(value) is not int for value in header_lengths[0])
        or header_lengths[0][0] not in range(1, _MAX_MANIFEST_BYTES + 1)
        or header_lengths[0][1] not in range(1, _MAX_ENVELOPE_BYTES + 1)
    ):
        raise RunLedgerError("ledger_corruption")
    length_rows = cursor.execute(_SLOT_LENGTH_SELECT).fetchall()
    if len(length_rows) != _CALL_COUNT:
        raise RunLedgerError("ledger_corruption")
    total_bytes = 0
    limits = (
        _MAX_PLAN_BYTES,
        _MAX_RECEIPT_BYTES,
        _MAX_TRACE_BYTES,
        _MAX_RESPONSE_EVIDENCE_BYTES,
        _MAX_DECODED_RESPONSE_BYTES,
    )
    for expected_index, row in enumerate(length_rows):
        if (
            len(row) != 6
            or type(row[0]) is not int
            or row[0] != expected_index
            or any(type(value) is not int for value in row[1:])
            or row[1] <= 0
            or any(value < 0 or value > limit for value, limit in zip(row[1:], limits))
        ):
            raise RunLedgerError("ledger_corruption")
        total_bytes += sum(row[1:])
        if total_bytes > _MAX_TOTAL_SLOT_ARTIFACT_BYTES:
            raise RunLedgerError("ledger_corruption")


def _load_snapshot_locked(
    cursor: sqlite3.Cursor,
    identity: _RunIdentity,
    envelope: RunEnvelopeV2,
) -> RunLedgerSnapshotV1:
    _validate_blob_budgets_locked(cursor)
    header_rows = cursor.execute(_HEADER_SELECT).fetchall()
    if len(header_rows) != 1:
        raise RunLedgerError("ledger_corruption")
    status, seal_kind, stored_root = _require_header_identity(
        header_rows[0],
        identity,
    )
    slots_list: list[RunLedgerSlotSnapshotV1] = []
    for index, row in enumerate(cursor.execute(_SLOT_SELECT)):
        if index >= _CALL_COUNT:
            raise RunLedgerError("ledger_corruption")
        slots_list.append(_load_slot(row, envelope.call_plans[index]))
    if len(slots_list) != _CALL_COUNT:
        raise RunLedgerError("ledger_corruption")
    slots = tuple(slots_list)
    _validate_slot_sequence(status, seal_kind, slots)
    computed_root = _run_root_sha256(
        run_id=identity.run_id,
        manifest_sha256=identity.manifest_sha256,
        run_envelope_sha256=identity.envelope_sha256,
        status=status,
        seal_kind=seal_kind,
        slots=slots,
    )
    if stored_root is not None and stored_root != computed_root:
        raise RunLedgerError("ledger_corruption")
    return RunLedgerSnapshotV1(
        schema_version=_SCHEMA_VERSION,
        ledger_policy=_LEDGER_POLICY,
        run_id=identity.run_id,
        manifest_sha256=identity.manifest_sha256,
        run_envelope_sha256=identity.envelope_sha256,
        call_count=_CALL_COUNT,
        status=status,  # type: ignore[arg-type]
        seal_kind=seal_kind,
        stored_run_root_sha256=stored_root,
        computed_run_root_sha256=computed_root,
        slots=slots,
    )


def _replay_snapshot_success_artifacts(
    snapshot: RunLedgerSnapshotV1,
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> None:
    """Preserve the full loader's semantic replay and answer decoding."""

    for slot in snapshot.slots:
        if slot.state != "SUCCEEDED":
            continue
        try:
            execution = audited_model_call_execution_v2_from_artifacts(
                manifest,
                tokenizer,
                envelope,
                receipt_bytes=slot.receipt_bytes,  # type: ignore[arg-type]
                trace_bytes=slot.trace_bytes,  # type: ignore[arg-type]
                response_evidence_bytes=slot.response_evidence_bytes,  # type: ignore[arg-type]
                decoded_response_utf8=slot.decoded_response_utf8,  # type: ignore[arg-type]
                backend=SGLangBridgeBackend(),
            )
            receipt = execution.receipt
            if (
                receipt.slot_index != slot.plan.slot_index
                or receipt.case_index != slot.plan.case_index
                or receipt.arm != slot.plan.arm
                or receipt.request_id != slot.plan.request_id
            ):
                raise RunLedgerError("ledger_corruption")
        except Exception as error:
            raise RunLedgerError("ledger_corruption") from error


def _rollback_with_note(
    cursor: sqlite3.Cursor | None,
    primary_error: BaseException,
) -> None:
    if cursor is None:
        return
    try:
        cursor.execute("ROLLBACK")
    except BaseException as rollback_error:
        primary_error.add_note(
            "ROLLBACK failed without replacing the primary error: "
            f"{type(rollback_error).__name__}: {rollback_error}"
        )


def _close_with_note(
    connection: sqlite3.Connection | None,
    primary_error: BaseException | None,
) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except BaseException as close_error:
        if primary_error is None:
            if isinstance(close_error, sqlite3.Error):
                raise _sqlite_ledger_error(close_error) from close_error
            raise
        primary_error.add_note(
            "close failed without replacing the primary error: "
            f"{type(close_error).__name__}: {close_error}"
        )


def _initialize_run_ledger_snapshot(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> tuple[RunLedgerSnapshotV1, tuple[int, int]]:
    """Exclusively create a new file with all 384 PLANNED slots.

    Resumption must call :func:`load_run_ledger`; an existing path is never
    guessed to be a new run, even when the file is empty or damaged.
    """

    path = _snapshot_database_path(database_path)
    identity = _run_identity(manifest, tokenizer, envelope)
    expected_file_identity = _create_private_database_file(path)
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    transaction_active = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(
            path,
            expected_file_identity=expected_file_identity,
        )
        cursor = connection.cursor()
        prelock_page_count = _read_integer_pragma(cursor, "page_count")
        cursor.execute("BEGIN EXCLUSIVE")
        transaction_active = True
        _require_delete_journal(cursor)
        schema_created = _initialize_or_validate_schema_locked(
            cursor,
            prelock_page_count=prelock_page_count,
        )
        if not schema_created:
            raise RunLedgerError("ledger_schema")
        if cursor.execute("SELECT count(*) FROM run_ledger_header").fetchone() != (0,):
            raise RunLedgerError("ledger_corruption")
        _insert_run_locked(cursor, identity, envelope)
        cursor.execute("COMMIT")
        transaction_active = False
        _require_expected_database(path, expected_file_identity)
    except BaseException as error:
        primary_error = error
        if transaction_active:
            _rollback_with_note(cursor, error)
        if isinstance(error, RunLedgerError):
            raise
        if isinstance(error, sqlite3.Error):
            raise _sqlite_ledger_error(error) from error
        raise
    finally:
        _close_with_note(connection, primary_error)
    snapshot = _load_run_ledger_snapshot(
        path,
        manifest,
        tokenizer,
        envelope,
        identity=identity,
        expected_file_identity=expected_file_identity,
    )
    return snapshot, expected_file_identity, identity


def initialize_run_ledger(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> RunLedgerSnapshotV1:
    """Exclusively create and return a validated 384-slot ledger."""

    snapshot, _file_identity, _identity = _initialize_run_ledger_snapshot(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    return snapshot


def _load_run_ledger_snapshot(
    path: str,
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    *,
    identity: _RunIdentity,
    expected_file_identity: tuple[int, int],
    replay_success_artifacts: bool = True,
) -> RunLedgerSnapshotV1:
    """Load one snapshot while pinning the database path to one inode."""

    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    transaction_active = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(
            path,
            expected_file_identity=expected_file_identity,
        )
        cursor = connection.cursor()
        cursor.execute("BEGIN")
        transaction_active = True
        _acquire_read_lock(cursor)
        _require_delete_journal(cursor)
        _validate_schema_locked(cursor)
        snapshot = _load_snapshot_locked(
            cursor,
            identity,
            envelope,
        )
        if replay_success_artifacts:
            _replay_snapshot_success_artifacts(
                snapshot,
                manifest,
                tokenizer,
                envelope,
            )
        cursor.execute("COMMIT")
        transaction_active = False
        _require_expected_database(path, expected_file_identity)
        return snapshot
    except BaseException as error:
        primary_error = error
        if transaction_active:
            _rollback_with_note(cursor, error)
        if isinstance(error, RunLedgerError):
            raise
        if isinstance(error, sqlite3.Error):
            raise _sqlite_ledger_error(error) from error
        raise
    finally:
        _close_with_note(connection, primary_error)


def load_run_ledger(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> RunLedgerSnapshotV1:
    """Load one transactionally consistent snapshot and replay all evidence."""

    path = _snapshot_database_path(database_path)
    file_identity = _require_private_regular_database(path)
    identity = _run_identity(manifest, tokenizer, envelope)
    return _load_run_ledger_snapshot(
        path,
        manifest,
        tokenizer,
        envelope,
        identity=identity,
        expected_file_identity=file_identity,
    )


def _receipt_artifact_commitment(
    value: bytes,
) -> LedgerArtifactCommitmentV1:
    return LedgerArtifactCommitmentV1(
        byte_count=len(value),
        sha256=_sha256(value),
    )


def _project_receipt_slot_v1(
    snapshot: RunLedgerSnapshotV1,
    slot: RunLedgerSlotSnapshotV1,
) -> RunLedgerReceiptSlotV1:
    if slot.state == "ATTRITION":
        if (
            slot.attempt_count != 1
            or slot.receipt_bytes is not None
            or slot.trace_bytes is not None
            or slot.response_evidence_bytes is not None
            or slot.decoded_response_utf8 is not None
            or slot.terminal_reason not in _ATTRITION_REASONS
        ):
            raise RunLedgerError("ledger_corruption")
        return RunLedgerReceiptSlotV1(
            plan=slot.plan,
            plan_sha256=slot.plan_sha256,
            state="ATTRITION",
            attempt_count=slot.attempt_count,
            canonical_receipt=None,
            receipt_commitment=None,
            trace_commitment=None,
            response_evidence_commitment=None,
            decoded_response_commitment=None,
            terminal_reason=slot.terminal_reason,
            ledger_leaf_sha256=slot.leaf_sha256,
        )
    if (
        slot.state != "SUCCEEDED"
        or slot.attempt_count != 1
        or slot.receipt_bytes is None
        or slot.trace_bytes is None
        or slot.response_evidence_bytes is None
        or slot.decoded_response_utf8 is None
        or slot.terminal_reason is not None
    ):
        raise RunLedgerError("ledger_state")
    try:
        receipt = audited_model_call_receipt_v2_from_bytes(slot.receipt_bytes)
    except Exception as error:
        raise RunLedgerError("ledger_corruption") from error
    plan = slot.plan
    if (
        receipt.manifest_sha256 != snapshot.manifest_sha256
        or receipt.run_envelope_sha256 != snapshot.run_envelope_sha256
        or receipt.slot_index != plan.slot_index
        or receipt.case_index != plan.case_index
        or receipt.arm != plan.arm
        or receipt.request_id != plan.request_id
    ):
        raise RunLedgerError("ledger_corruption")
    receipt_commitment = _receipt_artifact_commitment(slot.receipt_bytes)
    trace_commitment = _receipt_artifact_commitment(slot.trace_bytes)
    response_evidence_commitment = _receipt_artifact_commitment(
        slot.response_evidence_bytes
    )
    decoded_response_commitment = _receipt_artifact_commitment(
        slot.decoded_response_utf8
    )
    if (
        receipt.generation_trace_sha256 != trace_commitment.sha256
        or receipt.generation_response_evidence_sha256
        != response_evidence_commitment.sha256
        or receipt.generation_response_evidence_byte_count
        != response_evidence_commitment.byte_count
        or receipt.decoded_response_utf8_sha256 != decoded_response_commitment.sha256
        or receipt.decoded_response_utf8_bytes != decoded_response_commitment.byte_count
    ):
        raise RunLedgerError("ledger_corruption")
    return RunLedgerReceiptSlotV1(
        plan=plan,
        plan_sha256=slot.plan_sha256,
        state="SUCCEEDED",
        attempt_count=slot.attempt_count,
        canonical_receipt=receipt,
        receipt_commitment=receipt_commitment,
        trace_commitment=trace_commitment,
        response_evidence_commitment=response_evidence_commitment,
        decoded_response_commitment=decoded_response_commitment,
        terminal_reason=None,
        ledger_leaf_sha256=slot.leaf_sha256,
    )


def _project_run_ledger_receipts_v1(
    snapshot: RunLedgerSnapshotV1,
) -> RunLedgerReceiptSnapshotV1:
    if (
        snapshot.status != "SEALED"
        or snapshot.seal_kind not in ("complete", "complete_with_attrition")
        or snapshot.stored_run_root_sha256 is None
        or snapshot.stored_run_root_sha256 != snapshot.computed_run_root_sha256
    ):
        raise RunLedgerError("ledger_state")
    slots = tuple(_project_receipt_slot_v1(snapshot, slot) for slot in snapshot.slots)
    if len(slots) != _CALL_COUNT:
        raise RunLedgerError("ledger_corruption")
    return RunLedgerReceiptSnapshotV1(
        schema_version=1,
        projection_policy=_RECEIPT_PROJECTION_POLICY,
        ledger_policy=snapshot.ledger_policy,
        run_id=snapshot.run_id,
        manifest_sha256=snapshot.manifest_sha256,
        run_envelope_sha256=snapshot.run_envelope_sha256,
        call_count=snapshot.call_count,
        seal_kind=snapshot.seal_kind,  # type: ignore[arg-type]
        run_root_sha256=snapshot.stored_run_root_sha256,
        slots=slots,
    )


def load_run_ledger_receipt_snapshot_v1(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> RunLedgerReceiptSnapshotV1:
    """Load sealed receipt commitments without parsing or decoding answers.

    This projection still verifies SQLite structure, all 384 slot leaves, the
    sealed run root, canonical audited receipts, and their raw artifact hashes.
    It deliberately does not parse physical traces or response evidence, invoke
    the tokenizer decoder, return response bytes, or claim semantic replay.
    Hashes and lengths are commitments, not confidentiality against guessing.
    ``load_run_ledger`` remains the stronger full-replay API.
    Later public joins must reload the database rather than trust this
    constructible report as proof.
    """

    path = _snapshot_database_path(database_path)
    file_identity = _require_private_regular_database(path)
    identity = _run_identity(manifest, tokenizer, envelope)
    snapshot = _load_run_ledger_snapshot(
        path,
        manifest,
        tokenizer,
        envelope,
        identity=identity,
        expected_file_identity=file_identity,
        replay_success_artifacts=False,
    )
    return _project_run_ledger_receipts_v1(snapshot)


def _sql_blob(value: bytes | None) -> sqlite3.Binary | None:
    return None if value is None else sqlite3.Binary(value)


def _snapshot_artifact_bytes(snapshot: RunLedgerSnapshotV1) -> int:
    total = 0
    for slot in snapshot.slots:
        total += len(_plan_bytes(slot.plan))
        for value in (
            slot.receipt_bytes,
            slot.trace_bytes,
            slot.response_evidence_bytes,
            slot.decoded_response_utf8,
        ):
            if value is not None:
                total += len(value)
    return total


class RunLedgerSessionV1:
    """Trusted single-writer state transitions for one validated run ledger.

    This class relies on the runner holding the companion process lock for its
    whole lifetime.  SQLite transactions still validate the exact schema,
    current header, target row, and transition readback on every mutation.
    """

    def __init__(
        self,
        path: str,
        manifest: helpfulness.ModelRunManifest,
        tokenizer: AuditedDecoderTokenizer,
        envelope: RunEnvelopeV2,
        identity: _RunIdentity,
        file_identity: tuple[int, int],
        snapshot: RunLedgerSnapshotV1,
    ) -> None:
        self._path = path
        self._manifest = manifest
        self._tokenizer = tokenizer
        self._envelope = envelope
        self._identity = identity
        self._file_identity = file_identity
        self._snapshot = snapshot

    @classmethod
    def create(
        cls,
        database_path: str | os.PathLike[str],
        manifest: helpfulness.ModelRunManifest,
        tokenizer: AuditedDecoderTokenizer,
        envelope: RunEnvelopeV2,
    ) -> RunLedgerSessionV1:
        path = _snapshot_database_path(database_path)
        snapshot, file_identity, identity = _initialize_run_ledger_snapshot(
            path,
            manifest,
            tokenizer,
            envelope,
        )
        return cls(
            path,
            manifest,
            tokenizer,
            envelope,
            identity,
            file_identity,
            snapshot,
        )

    @classmethod
    def resume(
        cls,
        database_path: str | os.PathLike[str],
        manifest: helpfulness.ModelRunManifest,
        tokenizer: AuditedDecoderTokenizer,
        envelope: RunEnvelopeV2,
    ) -> RunLedgerSessionV1:
        path = _snapshot_database_path(database_path)
        file_identity = _require_private_regular_database(path)
        identity = _run_identity(manifest, tokenizer, envelope)
        snapshot = _load_run_ledger_snapshot(
            path,
            manifest,
            tokenizer,
            envelope,
            identity=identity,
            expected_file_identity=file_identity,
        )
        return cls(
            path,
            manifest,
            tokenizer,
            envelope,
            identity,
            file_identity,
            snapshot,
        )

    @property
    def snapshot(self) -> RunLedgerSnapshotV1:
        return self._snapshot

    @property
    def database_file_identity(self) -> tuple[int, int]:
        """Pinned ``(device, inode)`` used by the local run coordinator."""

        return self._file_identity

    def refresh(self) -> RunLedgerSnapshotV1:
        _require_expected_database(self._path, self._file_identity)
        snapshot = _load_run_ledger_snapshot(
            self._path,
            self._manifest,
            self._tokenizer,
            self._envelope,
            identity=self._identity,
            expected_file_identity=self._file_identity,
        )
        _require_expected_database(self._path, self._file_identity)
        self._snapshot = snapshot
        return snapshot

    def _next_snapshot(
        self,
        *,
        changed_slot: RunLedgerSlotSnapshotV1 | None,
        status: LedgerRunStatus,
        seal_kind: str | None,
    ) -> RunLedgerSnapshotV1:
        slots = list(self._snapshot.slots)
        if changed_slot is not None:
            slots[changed_slot.plan.slot_index] = changed_slot
        frozen_slots = tuple(slots)
        _validate_slot_sequence(status, seal_kind, frozen_slots)
        root = _run_root_sha256(
            run_id=self._identity.run_id,
            manifest_sha256=self._identity.manifest_sha256,
            run_envelope_sha256=self._identity.envelope_sha256,
            status=status,
            seal_kind=seal_kind,
            slots=frozen_slots,
        )
        snapshot = replace(
            self._snapshot,
            status=status,
            seal_kind=seal_kind,
            stored_run_root_sha256=root if status == "SEALED" else None,
            computed_run_root_sha256=root,
            slots=frozen_slots,
        )
        if _snapshot_artifact_bytes(snapshot) > _MAX_TOTAL_SLOT_ARTIFACT_BYTES:
            raise RunLedgerError("ledger_artifact")
        return snapshot

    def _commit_snapshot(
        self,
        next_snapshot: RunLedgerSnapshotV1,
        *,
        changed_slot_index: int | None,
    ) -> RunLedgerSnapshotV1:
        connection: sqlite3.Connection | None = None
        cursor: sqlite3.Cursor | None = None
        transaction_active = False
        primary_error: BaseException | None = None
        try:
            connection = _connect(
                self._path,
                expected_file_identity=self._file_identity,
            )
            cursor = connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            transaction_active = True
            _require_delete_journal(cursor)
            _validate_schema_locked(
                cursor,
                full_integrity=next_snapshot.status == "SEALED",
            )
            header_rows = cursor.execute(_HEADER_SELECT).fetchall()
            if len(header_rows) != 1:
                raise RunLedgerError("ledger_corruption")
            current_header = _require_header_identity(
                header_rows[0],
                self._identity,
            )
            if current_header != (
                self._snapshot.status,
                self._snapshot.seal_kind,
                self._snapshot.stored_run_root_sha256,
            ):
                raise RunLedgerError("ledger_state")
            if changed_slot_index is not None:
                if type(
                    changed_slot_index
                ) is not int or changed_slot_index not in range(_CALL_COUNT):
                    raise RunLedgerError("ledger_state")
                current_rows = cursor.execute(
                    _SLOT_SELECT_BY_INDEX,
                    (changed_slot_index,),
                ).fetchall()
                if len(current_rows) != 1:
                    raise RunLedgerError("ledger_corruption")
                current_slot = _load_slot(
                    current_rows[0],
                    self._envelope.call_plans[changed_slot_index],
                )
                if current_slot != self._snapshot.slots[changed_slot_index]:
                    raise RunLedgerError("ledger_state")
                next_slot = next_snapshot.slots[changed_slot_index]
                update = cursor.execute(
                    "UPDATE run_ledger_slots SET state = ?, attempt_count = ?, "
                    "receipt_bytes = ?, trace_bytes = ?, response_evidence_bytes = ?, "
                    "decoded_response_utf8 = ?, terminal_reason = ?, leaf_sha256 = ? "
                    "WHERE slot_index = ? AND state = ? AND leaf_sha256 = ?",
                    (
                        next_slot.state,
                        next_slot.attempt_count,
                        _sql_blob(next_slot.receipt_bytes),
                        _sql_blob(next_slot.trace_bytes),
                        _sql_blob(next_slot.response_evidence_bytes),
                        _sql_blob(next_slot.decoded_response_utf8),
                        next_slot.terminal_reason,
                        next_slot.leaf_sha256,
                        changed_slot_index,
                        current_slot.state,
                        current_slot.leaf_sha256,
                    ),
                )
                if update.rowcount != 1:
                    raise RunLedgerError("ledger_state")
            if next_snapshot.status == "SEALED":
                _validate_blob_budgets_locked(cursor)
                current_rows = cursor.execute(_SLOT_SELECT).fetchall()
                if len(current_rows) != _CALL_COUNT:
                    raise RunLedgerError("ledger_corruption")
                current_slots = tuple(
                    _load_slot(row, self._envelope.call_plans[index])
                    for index, row in enumerate(current_rows)
                )
                if current_slots != next_snapshot.slots:
                    raise RunLedgerError("ledger_state")
            if (
                next_snapshot.status,
                next_snapshot.seal_kind,
                next_snapshot.stored_run_root_sha256,
            ) != current_header:
                header_update = cursor.execute(
                    "UPDATE run_ledger_header SET status = ?, seal_kind = ?, "
                    "run_root_sha256 = ? WHERE singleton = 1 AND status = ?",
                    (
                        next_snapshot.status,
                        next_snapshot.seal_kind,
                        next_snapshot.stored_run_root_sha256,
                        self._snapshot.status,
                    ),
                )
                if header_update.rowcount != 1:
                    raise RunLedgerError("ledger_state")
            if changed_slot_index is not None:
                readback_rows = cursor.execute(
                    _SLOT_SELECT_BY_INDEX,
                    (changed_slot_index,),
                ).fetchall()
                if (
                    len(readback_rows) != 1
                    or _load_slot(
                        readback_rows[0],
                        self._envelope.call_plans[changed_slot_index],
                    )
                    != next_snapshot.slots[changed_slot_index]
                ):
                    raise RunLedgerError("ledger_corruption")
            header_readback = cursor.execute(_HEADER_SELECT).fetchall()
            if len(header_readback) != 1 or _require_header_identity(
                header_readback[0],
                self._identity,
            ) != (
                next_snapshot.status,
                next_snapshot.seal_kind,
                next_snapshot.stored_run_root_sha256,
            ):
                raise RunLedgerError("ledger_corruption")
            cursor.execute("COMMIT")
            transaction_active = False
            _require_expected_database(self._path, self._file_identity)
            self._snapshot = next_snapshot
            return next_snapshot
        except BaseException as error:
            primary_error = error
            if transaction_active:
                _rollback_with_note(cursor, error)
            if isinstance(error, RunLedgerError):
                raise
            if isinstance(error, sqlite3.Error):
                raise _sqlite_ledger_error(error) from error
            raise
        finally:
            _close_with_note(connection, primary_error)

    def mark_started(self, slot_index: int) -> RunLedgerSnapshotV1:
        if self._snapshot.status != "OPEN" or type(slot_index) is not int:
            raise RunLedgerError("ledger_state")
        planned_indexes = tuple(
            slot.plan.slot_index
            for slot in self._snapshot.slots
            if slot.state == "PLANNED"
        )
        if (
            not planned_indexes
            or planned_indexes[0] != slot_index
            or any(slot.state == "STARTED" for slot in self._snapshot.slots)
        ):
            raise RunLedgerError("ledger_state")
        current = self._snapshot.slots[slot_index]
        leaf = _slot_leaf_sha256(
            slot_index=slot_index,
            plan_sha256=current.plan_sha256,
            state="STARTED",
            attempt_count=1,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason=None,
        )
        changed = replace(
            current,
            state="STARTED",
            attempt_count=1,
            leaf_sha256=leaf,
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=changed,
                status="OPEN",
                seal_kind=None,
            ),
            changed_slot_index=slot_index,
        )

    def record_success(
        self,
        slot_index: int,
        execution: AuditedModelCallExecutionV2,
    ) -> RunLedgerSnapshotV1:
        if (
            self._snapshot.status != "OPEN"
            or type(slot_index) is not int
            or slot_index not in range(_CALL_COUNT)
            or self._snapshot.slots[slot_index].state != "STARTED"
            or type(execution) is not AuditedModelCallExecutionV2
        ):
            raise RunLedgerError("ledger_state")
        try:
            receipt_bytes = audited_model_call_receipt_v2_bytes(execution.receipt)
            trace_bytes = generation_physical_trace_bytes(execution.trace)
            response_evidence_bytes = generation_response_evidence_bytes(
                execution.response_evidence
            )
            decoded_response_utf8 = execution.response.encode("utf-8", errors="strict")
            reloaded = audited_model_call_execution_v2_from_artifacts(
                self._manifest,
                self._tokenizer,
                self._envelope,
                receipt_bytes=receipt_bytes,
                trace_bytes=trace_bytes,
                response_evidence_bytes=response_evidence_bytes,
                decoded_response_utf8=decoded_response_utf8,
                backend=SGLangBridgeBackend(),
            )
        except Exception as error:
            raise RunLedgerError("ledger_artifact") from error
        current = self._snapshot.slots[slot_index]
        receipt = reloaded.receipt
        if (
            reloaded != execution
            or receipt.slot_index != slot_index
            or receipt.case_index != current.plan.case_index
            or receipt.arm != current.plan.arm
            or receipt.request_id != current.plan.request_id
            or len(receipt_bytes) > _MAX_RECEIPT_BYTES
            or len(trace_bytes) > _MAX_TRACE_BYTES
            or len(response_evidence_bytes) > _MAX_RESPONSE_EVIDENCE_BYTES
            or len(decoded_response_utf8) > _MAX_DECODED_RESPONSE_BYTES
        ):
            raise RunLedgerError("ledger_artifact")
        leaf = _slot_leaf_sha256(
            slot_index=slot_index,
            plan_sha256=current.plan_sha256,
            state="SUCCEEDED",
            attempt_count=1,
            receipt_bytes=receipt_bytes,
            trace_bytes=trace_bytes,
            response_evidence_bytes=response_evidence_bytes,
            decoded_response_utf8=decoded_response_utf8,
            terminal_reason=None,
        )
        changed = replace(
            current,
            state="SUCCEEDED",
            receipt_bytes=receipt_bytes,
            trace_bytes=trace_bytes,
            response_evidence_bytes=response_evidence_bytes,
            decoded_response_utf8=decoded_response_utf8,
            leaf_sha256=leaf,
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=changed,
                status="OPEN",
                seal_kind=None,
            ),
            changed_slot_index=slot_index,
        )

    def record_attrition(
        self,
        slot_index: int,
        reason: str,
    ) -> RunLedgerSnapshotV1:
        if (
            self._snapshot.status != "OPEN"
            or type(slot_index) is not int
            or slot_index not in range(_CALL_COUNT)
            or self._snapshot.slots[slot_index].state != "STARTED"
            or reason not in _ATTRITION_REASONS
        ):
            raise RunLedgerError("ledger_state")
        current = self._snapshot.slots[slot_index]
        leaf = _slot_leaf_sha256(
            slot_index=slot_index,
            plan_sha256=current.plan_sha256,
            state="ATTRITION",
            attempt_count=1,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason=reason,
        )
        changed = replace(
            current,
            state="ATTRITION",
            attempt_count=1,
            terminal_reason=reason,
            leaf_sha256=leaf,
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=changed,
                status="OPEN",
                seal_kind=None,
            ),
            changed_slot_index=slot_index,
        )

    def seal_indeterminate(
        self,
        slot_index: int,
        reason: str,
    ) -> RunLedgerSnapshotV1:
        if (
            self._snapshot.status != "OPEN"
            or type(slot_index) is not int
            or slot_index not in range(_CALL_COUNT)
            or self._snapshot.slots[slot_index].state != "STARTED"
            or reason not in _INDETERMINATE_REASONS
        ):
            raise RunLedgerError("ledger_state")
        current = self._snapshot.slots[slot_index]
        leaf = _slot_leaf_sha256(
            slot_index=slot_index,
            plan_sha256=current.plan_sha256,
            state="INDETERMINATE",
            attempt_count=1,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason=reason,
        )
        changed = replace(
            current,
            state="INDETERMINATE",
            attempt_count=1,
            terminal_reason=reason,
            leaf_sha256=leaf,
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=changed,
                status="SEALED",
                seal_kind="indeterminate",
            ),
            changed_slot_index=slot_index,
        )

    def seal_after_lost_started(self, slot_index: int) -> RunLedgerSnapshotV1:
        """Fail closed if a post-call readback has regressed to PLANNED."""

        planned_indexes = tuple(
            slot.plan.slot_index
            for slot in self._snapshot.slots
            if slot.state == "PLANNED"
        )
        if (
            self._snapshot.status != "OPEN"
            or type(slot_index) is not int
            or slot_index not in range(_CALL_COUNT)
            or self._snapshot.slots[slot_index].state != "PLANNED"
            or not planned_indexes
            or planned_indexes[0] != slot_index
            or any(slot.state == "STARTED" for slot in self._snapshot.slots)
        ):
            raise RunLedgerError("ledger_state")
        current = self._snapshot.slots[slot_index]
        leaf = _slot_leaf_sha256(
            slot_index=slot_index,
            plan_sha256=current.plan_sha256,
            state="INDETERMINATE",
            attempt_count=1,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason="started_record_lost",
        )
        changed = replace(
            current,
            state="INDETERMINATE",
            attempt_count=1,
            terminal_reason="started_record_lost",
            leaf_sha256=leaf,
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=changed,
                status="SEALED",
                seal_kind="indeterminate",
            ),
            changed_slot_index=slot_index,
        )

    def seal_complete(self) -> RunLedgerSnapshotV1:
        if self._snapshot.status != "OPEN" or any(
            slot.state not in ("SUCCEEDED", "ATTRITION")
            for slot in self._snapshot.slots
        ):
            raise RunLedgerError("ledger_state")
        seal_kind = (
            "complete_with_attrition"
            if any(slot.state == "ATTRITION" for slot in self._snapshot.slots)
            else "complete"
        )
        return self._commit_snapshot(
            self._next_snapshot(
                changed_slot=None,
                status="SEALED",
                seal_kind=seal_kind,
            ),
            changed_slot_index=None,
        )
