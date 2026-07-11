# SPDX-License-Identifier: Apache-2.0

"""Execute release-bound future queries and validate them before scoring.

The local-update experiment must not hand policy labels or scorer truth to its
future consumer.  This adapter therefore replaces logical row numbers with
random opaque tokens, mixes their wire order, executes every query in one
isolated child, and validates the complete release/read/render/consume trace
before restoring the parent-owned logical order.

This is an honest-runner causal boundary, not a sandbox against malicious
Python.  The parent owns the database paths, release assignments, token map,
and eventual scorer truth.  The child receives only an explicit release and a
query.  Stable pre/post database identity and bytes detect ordinary drift.  A
privileged swap-and-restore race remains outside this honest-runner boundary;
hashes are integrity commitments, not signatures or a malicious-admin sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import stat
from dataclasses import dataclass, replace
from pathlib import Path

from examples.memory_service import scoped_codebook_eval as harness

from areal.v2.memory_service import MemoryScope
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "FutureReleaseQueryV1",
    "FutureReleaseValidationError",
    "VerifiedFutureAnswerV1",
    "execute_verified_release_batch_v1",
]


_SHA256_HEX_LENGTH = 64
_RENDERER_VERSION = "memory-codebook/v1"
_CONSUMER_VERSION = "scripted-last-occurrence/v1"
_EXECUTION_TOKEN_DOMAIN = b"areal-memory-local-update-execution-token-v1\0"


class FutureReleaseValidationError(RuntimeError):
    """Stable, answer-free reason for rejecting a future execution."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("future validation reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class FutureReleaseQueryV1:
    """One parent-owned logical assignment; policy and truth stay elsewhere."""

    logical_index: int
    database_path: str
    scope: MemoryScope
    release_id: str
    query: str


@dataclass(frozen=True, slots=True)
class VerifiedFutureAnswerV1:
    """A fully acknowledged child response restored to parent logical order."""

    logical_index: int
    database_sha256: str
    scope_sha256: str
    release_id: str
    release_content_sha256: str
    query_sha256: str
    response: str
    response_sha256: str
    execution_token_sha256: str


@dataclass(frozen=True, slots=True)
class _DatabaseReceipt:
    path: str
    device: int
    inode: int
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ExpectedItem:
    logical: FutureReleaseQueryV1
    database: _DatabaseReceipt
    request: harness.FutureChildRequest
    expected_treatment: harness.ResolvedTreatment
    expected_audit: tuple[harness.ReadAuditEvent, ...]
    expected_rendered: harness.RenderedContext


@dataclass(frozen=True, slots=True)
class _SpawnedBatch:
    response: harness.FutureBatchResponse
    expected_pid: int


def _fail(reason: str, error: BaseException | None = None) -> None:
    failure = FutureReleaseValidationError(reason)
    if error is None:
        raise failure
    raise failure from error


def _scope_sha256(scope: MemoryScope) -> str:
    if type(scope) is not MemoryScope:
        _fail("closed_schema")
    value = {
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
        "tenant_id": scope.tenant_id,
    }
    return hashlib.sha256(
        b"areal-memory-local-update-future-scope-v1\0"
        + json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _database_receipt(path_text: str) -> _DatabaseReceipt:
    if type(path_text) is not str or not path_text:
        _fail("database_invalid")
    try:
        path = Path(path_text)
        link_state = path.lstat()
        if stat.S_ISLNK(link_state.st_mode) or not stat.S_ISREG(link_state.st_mode):
            _fail("database_invalid")
        resolved = path.resolve(strict=True)
        state = resolved.stat()
        if not stat.S_ISREG(state.st_mode):
            _fail("database_invalid")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        final_state = resolved.stat()
    except FutureReleaseValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        _fail("database_invalid", error)
    if (state.st_dev, state.st_ino, state.st_size) != (
        final_state.st_dev,
        final_state.st_ino,
        final_state.st_size,
    ):
        _fail("database_changed")
    return _DatabaseReceipt(
        path=str(resolved),
        device=state.st_dev,
        inode=state.st_ino,
        size=state.st_size,
        sha256=digest.hexdigest(),
    )


def _validate_logical_items(
    items: tuple[FutureReleaseQueryV1, ...],
) -> tuple[_DatabaseReceipt, ...]:
    if type(items) is not tuple or not items:
        _fail("closed_schema")
    logical_indexes: set[int] = set()
    receipts_by_path: dict[str, _DatabaseReceipt] = {}
    ordered_receipts: list[_DatabaseReceipt] = []
    for item in items:
        if (
            type(item) is not FutureReleaseQueryV1
            or type(item.logical_index) is not int
            or item.logical_index < 0
            or item.logical_index > 2**63 - 1
            or item.logical_index in logical_indexes
            or type(item.database_path) is not str
            or type(item.scope) is not MemoryScope
            or type(item.release_id) is not str
            or not item.release_id.startswith("rel_")
            or type(item.query) is not str
        ):
            _fail("closed_schema")
        try:
            item.query.encode("ascii", errors="strict")
        except UnicodeEncodeError as error:
            _fail("closed_schema", error)
        logical_indexes.add(item.logical_index)
        receipt = _database_receipt(item.database_path)
        existing = receipts_by_path.setdefault(receipt.path, receipt)
        if existing != receipt:
            _fail("database_changed")
        ordered_receipts.append(existing)
    return tuple(ordered_receipts)


def _sample_execution_seed() -> bytes:
    return secrets.token_bytes(32)


def _derive_opaque_tokens(count: int, seed: bytes) -> tuple[int, ...]:
    if type(seed) is not bytes or len(seed) != 32:
        _fail("opaque_seed")
    tokens = tuple(
        (1 << 127)
        | int.from_bytes(
            hashlib.sha256(
                _EXECUTION_TOKEN_DOMAIN + seed + index.to_bytes(8, "big")
            ).digest()[:16],
            "big",
        )
        for index in range(count)
    )
    if len(set(tokens)) != len(tokens):
        _fail("opaque_token")
    return tokens


def _expected_item(
    logical: FutureReleaseQueryV1,
    database: _DatabaseReceipt,
    token: int,
) -> _ExpectedItem:
    try:
        session_id, run_id = harness._opaque_future_identity(token)
        request = harness.FutureChildRequest(
            execution_index=token,
            database_path=database.path,
            scope=logical.scope,
            source=harness.WireSourceSpec(
                source_kind="release",
                release_id=logical.release_id,
                cutoff=None,
                allowed_evidence_kinds=(),
                oracle_entries=(),
            ),
            query=logical.query,
            future_session_id=session_id,
            future_run_id=run_id,
            renderer_version=_RENDERER_VERSION,
            consumer_version=_CONSUMER_VERSION,
        )
        assignment = harness.ReleaseSourceAssignment(
            scope=logical.scope,
            release_id=logical.release_id,
        )
        treatment, audit = harness._expected_release_source(
            SQLiteMemoryStore(database.path),
            assignment,
        )
        rendered = harness.render_context(treatment.entries)
        # Exercise the exact child-side wire/query checks before any subprocess
        # is launched.  Errors are deliberately collapsed to an answer-free code.
        harness._validate_future_request_item(request)
    except Exception as error:
        _fail("assignment_invalid", error)
    return _ExpectedItem(
        logical=logical,
        database=database,
        request=request,
        expected_treatment=treatment,
        expected_audit=audit,
        expected_rendered=rendered,
    )


def _retoken_expected_item(
    expected: _ExpectedItem,
    token: int,
) -> _ExpectedItem:
    try:
        session_id, run_id = harness._opaque_future_identity(token)
        request = replace(
            expected.request,
            execution_index=token,
            future_session_id=session_id,
            future_run_id=run_id,
        )
        harness._validate_future_request_item(request)
    except Exception as error:
        _fail("assignment_invalid", error)
    return replace(expected, request=request)


def _spawn_future_batch(
    request: harness.FutureBatchRequest,
    timeout_seconds: float,
) -> _SpawnedBatch:
    """Spawn the trusted child and retain its PID outside response wire."""

    completed = harness.run_isolated_child_raw(
        request,
        role="future-child",
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        _fail("child_execution")
    try:
        response = harness.wire_loads(completed.stdout)
        if harness.wire_dumps(response) != completed.stdout:
            _fail("child_response")
    except FutureReleaseValidationError:
        raise
    except Exception as error:
        _fail("child_response", error)
    if type(response) is harness.ChildFailureResponse:
        _fail(response.reason)
    if type(response) is not harness.FutureBatchResponse:
        _fail("child_response")
    try:
        harness._validate_child_process_response(
            response,
            expected_pid=completed.pid,
        )
    except Exception as error:
        _fail("process_isolation", error)
    return _SpawnedBatch(response=response, expected_pid=completed.pid)


def _validate_state_receipt(
    receipt: harness.ItemStateReceipt,
    expected: _ExpectedItem,
    generation_index: int,
    all_component_ids: set[str],
) -> None:
    if type(receipt) is not harness.ItemStateReceipt:
        _fail("state_reuse")
    identities = (
        receipt.store_instance_id,
        receipt.reader_instance_id,
        receipt.resolver_instance_id,
        receipt.renderer_instance_id,
        receipt.consumer_instance_id,
        receipt.audit_instance_id,
        receipt.logical_session_instance_id,
        receipt.history_instance_id,
    )
    if (
        receipt.execution_index != expected.request.execution_index
        or receipt.generation_index != generation_index
        or receipt.logical_session_id != expected.request.future_session_id
        or receipt.logical_run_id != expected.request.future_run_id
        or receipt.history_length != 0
        or len(set(identities)) != len(identities)
        or any(
            type(identity) is not str
            or len(identity) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in identity)
            for identity in identities
        )
        or all_component_ids.intersection(identities)
    ):
        _fail("state_reuse")
    all_component_ids.update(identities)


def _validate_observation(
    observation: harness.FutureExecutionObservation,
    expected: _ExpectedItem,
    response: harness.FutureBatchResponse,
) -> VerifiedFutureAnswerV1:
    request = expected.request
    treatment = expected.expected_treatment
    rendered = expected.expected_rendered
    query_bytes = request.query.encode("ascii")
    query_sha256 = hashlib.sha256(query_bytes).hexdigest()
    if (
        type(observation) is not harness.FutureExecutionObservation
        or observation.execution_index != request.execution_index
        or observation.source_kind != "release"
        or observation.scope != request.scope
        or observation.release_id != request.source.release_id
        or observation.future_session_id != request.future_session_id
        or observation.future_run_id != request.future_run_id
        or observation.future_pid != response.pid
        or observation.future_process_instance_id != response.process_instance_id
        or observation.model_call_receipt is not None
        or observation.rendered_context_token_count is not None
        or observation.history_length != 0
        or observation.query_sha256 != query_sha256
        or observation.eligible_ids != treatment.eligible_ids
        or observation.retrieved_ids != treatment.retrieved_ids
        or observation.returned_ids != treatment.returned_ids
        or observation.source_evidence_ids != treatment.source_evidence_ids
        or observation.entries != rendered.entry_receipts
        or observation.reader_audit != expected.expected_audit
        or observation.rendered_context_sha256
        != hashlib.sha256(rendered.bytes).hexdigest()
        or observation.rendered_context_utf8_bytes != len(rendered.bytes)
    ):
        _fail("assignment_mismatch")
    consumer = observation.consumer_input_receipt
    expected_consumer = harness.consume_scripted(
        query_bytes,
        rendered.bytes,
        history=(),
    )
    if (
        type(consumer) is not harness.ConsumerInputReceipt
        or consumer.received_context_sha256 != observation.rendered_context_sha256
        or consumer.received_context_utf8_bytes
        != observation.rendered_context_utf8_bytes
        or consumer.received_query_sha256 != query_sha256
        or consumer.received_history_length != 0
        or consumer != expected_consumer.input_receipt
        or observation.response != expected_consumer.response
    ):
        _fail("consumer_receipt")
    try:
        # Re-rendering catches forged byte offsets/content receipts even if a
        # test double changes matching fields together.
        resolved = tuple(
            harness.ResolvedEntry(
                slot=entry.slot,
                key=entry.key,
                value=entry.value,
                source_kind=entry.source_kind,
                revision_id=entry.revision_id,
                candidate_id=entry.candidate_id,
                evidence_ids=entry.evidence_ids,
            )
            for entry in observation.entries
        )
        rerendered = harness.render_context(resolved)
    except (AttributeError, TypeError, UnicodeError, ValueError) as error:
        _fail("render_receipt", error)
    if rerendered != rendered:
        _fail("render_receipt")
    if type(observation.response) is not str:
        _fail("child_response")
    try:
        response_bytes = observation.response.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        _fail("child_response", error)
    return VerifiedFutureAnswerV1(
        logical_index=expected.logical.logical_index,
        database_sha256=expected.database.sha256,
        scope_sha256=_scope_sha256(expected.logical.scope),
        release_id=expected.logical.release_id,
        release_content_sha256=expected.expected_audit[0].returned_content_hashes[0],
        query_sha256=query_sha256,
        response=observation.response,
        response_sha256=hashlib.sha256(response_bytes).hexdigest(),
        execution_token_sha256=hashlib.sha256(
            request.execution_index.to_bytes(16, "big")
        ).hexdigest(),
    )


def execute_verified_release_batch_v1(
    items: tuple[FutureReleaseQueryV1, ...],
    *,
    timeout_seconds: float = 120.0,
) -> tuple[VerifiedFutureAnswerV1, ...]:
    """Run and fully acknowledge all future items before returning any answer.

    The function is all-or-nothing: missing, duplicated, foreign, or tampered
    observations raise before the opaque-to-logical map is used to construct a
    result.  Callers must score only the returned verified tuple.
    """

    if (
        type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or timeout_seconds > 3600
        or (type(timeout_seconds) is float and not math.isfinite(timeout_seconds))
    ):
        _fail("closed_schema")
    database_receipts = _validate_logical_items(items)
    # Validate every query, release, graph, render, and database before an
    # opaque-token factory (which may be externally observable) is consumed.
    preflight = tuple(
        _expected_item(item, database, (1 << 127) + index)
        for index, (item, database) in enumerate(
            zip(items, database_receipts, strict=True)
        )
    )
    try:
        execution_seed = _sample_execution_seed()
    except Exception as error:
        _fail("opaque_seed", error)
    tokens = _derive_opaque_tokens(len(items), execution_seed)
    del execution_seed
    expected_items = tuple(
        _retoken_expected_item(expected, token)
        for expected, token in zip(preflight, tokens, strict=True)
    )

    # Token order is unrelated to case/policy order and hides adjacency from
    # the honest child.  The parent retains the only logical mapping.
    wire_items = tuple(
        sorted(expected_items, key=lambda item: item.request.execution_index)
    )
    request = harness.FutureBatchRequest(
        items=tuple(item.request for item in wire_items)
    )
    try:
        spawned = _spawn_future_batch(request, float(timeout_seconds))
    except FutureReleaseValidationError:
        raise
    except Exception as error:
        _fail("child_execution", error)
    if type(spawned) is not _SpawnedBatch:
        _fail("child_response")
    response = spawned.response
    if type(response) is not harness.FutureBatchResponse:
        _fail("child_response")
    try:
        harness._validate_child_process_response(
            response,
            expected_pid=spawned.expected_pid,
        )
    except Exception as error:
        _fail("process_isolation", error)
    if type(response.foreign_probes) is not tuple:
        _fail("child_response")
    if response.foreign_probes:
        _fail("foreign_probe")
    if (
        type(response.observations) is not tuple
        or type(response.state_receipts) is not tuple
        or len(response.observations) != len(wire_items)
        or len(response.state_receipts) != len(wire_items)
        or tuple(item.execution_index for item in response.observations)
        != tuple(item.request.execution_index for item in wire_items)
        or tuple(item.execution_index for item in response.state_receipts)
        != tuple(item.request.execution_index for item in wire_items)
    ):
        _fail("assignment_mismatch")

    all_component_ids: set[str] = set()
    verified_by_logical_index: dict[int, VerifiedFutureAnswerV1] = {}
    for generation_index, (expected, observation, receipt) in enumerate(
        zip(
            wire_items,
            response.observations,
            response.state_receipts,
            strict=True,
        )
    ):
        _validate_state_receipt(
            receipt,
            expected,
            generation_index,
            all_component_ids,
        )
        verified = _validate_observation(observation, expected, response)
        if verified.logical_index in verified_by_logical_index:
            _fail("assignment_mismatch")
        verified_by_logical_index[verified.logical_index] = verified

    # Ensure the child did not change or replace a database during execution.
    for before in {receipt.path: receipt for receipt in database_receipts}.values():
        after = _database_receipt(before.path)
        if after != before:
            _fail("database_changed")
    expected_logical = tuple(item.logical_index for item in items)
    if set(verified_by_logical_index) != set(expected_logical):
        _fail("assignment_mismatch")
    return tuple(verified_by_logical_index[index] for index in expected_logical)
