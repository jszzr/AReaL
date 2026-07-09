# SPDX-License-Identifier: Apache-2.0

"""Deterministic scoped-codebook helpfulness evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from weakref import WeakKeyDictionary

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryCandidate,
    MemoryRelease,
    MemoryRevision,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

SCHEMA_VERSION = 1
CASE_SEED = "areal-memory-helpfulness-v1-20260708"
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
MASKED_VALUE = "XXXXX"
UNKNOWN = "UNKNOWN"
_RENDER_HEADER = (
    b"[memory-codebook/v1]\n[mask=XXXXX means unavailable; answer UNKNOWN]\n"
)
_KEY_PATTERN = rb"project-[abcdefghjklmnpqrstuvwxyz23456789]{6}"
_VALUE_PATTERN = rb"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}"
_QUERY_PATTERN = re.compile(
    rb"What is the current code for ("
    + _KEY_PATTERN
    + rb")\? Reply with exactly the code or UNKNOWN\."
)
_ENTRY_LINE_PATTERN = re.compile(
    rb"(?P<slot>[0-9]{2})\t(?P<key>"
    + _KEY_PATTERN
    + rb")\t(?P<value>"
    + _VALUE_PATTERN
    + rb")\n"
)
_FACT_PATTERN = re.compile(
    rb"(?P<key>" + _KEY_PATTERN + rb") = (?P<value>" + _VALUE_PATTERN + rb")"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CodebookEntry:
    """One generated key/value assignment."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class CodebookCase:
    """Private manifest for one deterministic scoped-codebook case."""

    schema_version: int
    seed: str
    case_id: str
    subject_id: str
    case_index: int
    target_slot: int
    target_key: str
    old_value: str
    current_value: str
    masked_value: str
    shared_entries: tuple[CodebookEntry, ...]
    padding_entry: CodebookEntry


@dataclass(frozen=True, slots=True)
class ResolvedEntry:
    """One ordered resolver output before canonical rendering."""

    slot: int
    key: str
    value: str
    source_kind: str
    revision_id: str | None = None
    candidate_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntryReceipt:
    """Provenance and exact rendered byte range for one entry line."""

    slot: int
    key: str
    value: str
    source_kind: str
    revision_id: str | None
    candidate_id: str | None
    evidence_ids: tuple[str, ...]
    content_sha256: str
    rendered_start: int
    rendered_end: int


@dataclass(frozen=True, slots=True)
class RenderedContext:
    """Canonical context bytes and evaluator-owned entry receipts."""

    bytes: bytes
    entry_receipts: tuple[EntryReceipt, ...]


@dataclass(frozen=True, slots=True)
class ConsumerInputReceipt:
    """Hashes measured from the bytes actually received by a consumer."""

    received_context_sha256: str
    received_context_utf8_bytes: int
    received_query_sha256: str


@dataclass(frozen=True, slots=True)
class ConsumerResult:
    """Unscored scripted response plus its input-boundary receipt."""

    response: str
    input_receipt: ConsumerInputReceipt


@dataclass(frozen=True, slots=True)
class CaptureReferences:
    """Deterministic evidence addresses and capture-time policy boundary."""

    local_scope: MemoryScope
    foreign_scope: MemoryScope
    case_base: datetime
    raw_history_cutoff: datetime
    capture_session_ids: tuple[str, str, str]
    old_evidence_ids: tuple[str, ...]
    current_evidence_id: str
    control_evidence_ids: tuple[str, str]
    foreign_evidence_id: str


@dataclass(frozen=True, slots=True)
class RevisionReferences:
    """Stable revision addresses needed to verify one case graph."""

    target_old_revision_id: str
    target_current_revision_id: str
    shared_revision_ids: tuple[str, ...]
    padding_revision_id: str
    target_masked_revision_id: str
    foreign_target_revision_id: str


@dataclass(frozen=True, slots=True)
class ReleaseAssignments:
    """Exact local treatment releases plus the foreign leakage sentinel."""

    stale_release_id: str
    current_release_id: str
    masked_release_id: str
    empty_release_id: str
    foreign_sentinel_release_id: str


@dataclass(frozen=True, slots=True)
class CaseDatabaseReferences:
    """Portable references for a fully persisted deterministic case graph."""

    capture: CaptureReferences
    revisions: RevisionReferences
    releases: ReleaseAssignments


@dataclass(frozen=True, slots=True)
class ReadAuditEvent:
    """One evaluator-owned record of an allowed or denied capability call."""

    operation: str
    requested_scope: MemoryScope
    requested_ids: tuple[str, ...]
    allowed: bool
    returned_record_ids: tuple[str, ...]
    returned_content_hashes: tuple[str, ...]


class ReadAuditSink:
    """Mutable audit buffer held outside resolver-owned return values."""

    __slots__ = ("__events",)

    def __init__(self) -> None:
        self.__events: list[ReadAuditEvent] = []

    def _record(self, event: ReadAuditEvent) -> None:
        self.__events.append(event)

    def snapshot(self) -> tuple[ReadAuditEvent, ...]:
        return tuple(self.__events)


@dataclass(frozen=True, slots=True)
class ReleaseSourceAssignment:
    scope: MemoryScope
    release_id: str


@dataclass(frozen=True, slots=True)
class RawSourceAssignment:
    scope: MemoryScope
    cutoff: datetime


@dataclass(frozen=True, slots=True)
class OracleSourceAssignment:
    scope: MemoryScope
    entries: tuple[ResolvedEntry, ...]


@dataclass(frozen=True, slots=True)
class OracleEntryBatch:
    """Structural scope plus explicit entries returned by the only oracle call."""

    scope: MemoryScope
    entries: tuple[ResolvedEntry, ...]


@dataclass(frozen=True, slots=True)
class ResolvedTreatment:
    """Unrendered entries plus resolver-reported retrieval provenance."""

    source_kind: str
    scope: MemoryScope
    release_id: str | None
    eligible_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    entries: tuple[ResolvedEntry, ...]


@dataclass(frozen=True, slots=True)
class ModelCallReceipt:
    """Hashes captured at the evaluator-owned local model-call boundary."""

    submitted_prompt_sha256: str
    submitted_prompt_context_start: int
    submitted_prompt_context_end: int
    submitted_prompt_context_sha256: str
    submitted_input_token_ids_sha256: str
    submitted_input_token_count: int


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    """Child-side source and consumption facts without scorer-owned truth."""

    execution_index: int
    source_kind: str
    scope: MemoryScope
    capture_session_ids: tuple[str, str, str]
    future_session_id: str
    future_run_id: str
    capture_pid: int
    future_pid: int
    capture_process_instance_id: str
    future_process_instance_id: str
    release_id: str | None
    eligible_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    entries: tuple[EntryReceipt, ...]
    reader_audit: tuple[ReadAuditEvent, ...]
    rendered_context_sha256: str
    rendered_context_utf8_bytes: int
    rendered_context_token_count: int | None
    consumer_input_receipt: ConsumerInputReceipt
    model_call_receipt: ModelCallReceipt | None
    query_sha256: str
    history_length: int
    response: str


@dataclass(frozen=True, slots=True)
class ParentSourceContract:
    """Parent-generated source/audit/render facts never supplied by the child."""

    eligible_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    entries: tuple[EntryReceipt, ...]
    reader_audit: tuple[ReadAuditEvent, ...]
    rendered_context_sha256: str
    rendered_context_utf8_bytes: int


@dataclass(frozen=True, slots=True)
class ParentScheduleItem:
    """Private parent-owned case truth joined by opaque execution index."""

    execution_index: int
    case: CodebookCase
    case_manifest_sha256: str
    arm: str
    source_kind: str
    scope: MemoryScope
    release_id: str | None
    capture_session_ids: tuple[str, str, str]
    query_sha256: str
    expected_response: str
    expected_source: ParentSourceContract


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    """Parent-validated exposure plus scorer-owned case outcome."""

    schema_version: int
    case_id: str
    case_manifest_sha256: str
    execution_index: int
    arm: str
    source_kind: str
    scope: MemoryScope
    capture_session_ids: tuple[str, str, str]
    future_session_id: str
    future_run_id: str
    capture_pid: int
    future_pid: int
    capture_process_instance_id: str
    future_process_instance_id: str
    release_id: str | None
    eligible_revision_ids: tuple[str, ...]
    retrieved_revision_ids: tuple[str, ...]
    returned_revision_ids: tuple[str, ...]
    injected_revision_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    entries: tuple[EntryReceipt, ...]
    reader_audit: tuple[ReadAuditEvent, ...]
    rendered_context_sha256: str
    rendered_context_utf8_bytes: int
    rendered_context_token_count: int | None
    received_context_sha256: str
    received_context_utf8_bytes: int
    received_query_sha256: str
    submitted_prompt_sha256: str | None
    submitted_prompt_context_start: int | None
    submitted_prompt_context_end: int | None
    submitted_prompt_context_sha256: str | None
    submitted_input_token_ids_sha256: str | None
    submitted_input_token_count: int | None
    query_sha256: str
    history_length: int
    response: str
    normalized_response: str
    expected_response: str
    utility: int
    abstained: bool
    followed_injected_value: bool


class CapabilityInterfaceError(AttributeError):
    """A resolver requested an operation absent from its sealed capability."""


class UnauthorizedReadError(PermissionError):
    """An exposed point-read requested an ID outside the sealed graph."""


class TreatmentValidationError(ValueError):
    """A stable pre-registered reason for rejecting source evidence."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ObservationValidationError(ValueError):
    """A stable reason for rejecting exposure before parent scoring."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _token(
    seed: str,
    case_index: int,
    label: str,
    item_index: int,
    attempt: int,
    length: int,
) -> str:
    material = f"{seed}|{case_index}|{label}|{item_index}|{attempt}".encode()
    digest = hashlib.sha256(material).digest()
    return "".join(ALPHABET[digest[index] & 31] for index in range(length))


def _unique_key(
    *,
    case_index: int,
    label: str,
    item_index: int,
    used: set[str],
) -> str:
    attempt = 0
    while True:
        key = f"project-{_token(CASE_SEED, case_index, label, item_index, attempt, 6).lower()}"
        if key not in used:
            used.add(key)
            return key
        attempt += 1


def _unique_value(
    *,
    case_index: int,
    label: str,
    item_index: int,
    used: set[str],
) -> str:
    attempt = 0
    while True:
        value = _token(CASE_SEED, case_index, label, item_index, attempt, 5)
        if value not in used and value not in {UNKNOWN, MASKED_VALUE}:
            used.add(value)
            return value
        attempt += 1


def generate_case(case_index: int) -> CodebookCase:
    """Generate one frozen case without sharing collision attempts across fields."""

    if type(case_index) is not int or case_index < 0:
        raise ValueError("case_index must be a non-negative integer")

    used_keys: set[str] = set()
    used_values: set[str] = set()
    target_key = _unique_key(
        case_index=case_index,
        label="target",
        item_index=0,
        used=used_keys,
    )
    old_value = _unique_value(
        case_index=case_index,
        label="old",
        item_index=0,
        used=used_values,
    )
    current_value = _unique_value(
        case_index=case_index,
        label="current",
        item_index=0,
        used=used_values,
    )
    shared_keys = tuple(
        _unique_key(
            case_index=case_index,
            label="shared",
            item_index=item_index,
            used=used_keys,
        )
        for item_index in range(4)
    )
    shared_values = tuple(
        _unique_value(
            case_index=case_index,
            label="shared-value",
            item_index=item_index,
            used=used_values,
        )
        for item_index in range(4)
    )
    padding_key = _unique_key(
        case_index=case_index,
        label="padding",
        item_index=0,
        used=used_keys,
    )
    padding_value = _unique_value(
        case_index=case_index,
        label="padding-value",
        item_index=0,
        used=used_values,
    )
    return CodebookCase(
        schema_version=SCHEMA_VERSION,
        seed=CASE_SEED,
        case_id=f"nonce-{case_index:03d}",
        subject_id=f"nonce-subject-{case_index:03d}",
        case_index=case_index,
        target_slot=case_index % 5,
        target_key=target_key,
        old_value=old_value,
        current_value=current_value,
        masked_value=MASKED_VALUE,
        shared_entries=tuple(
            CodebookEntry(key=key, value=value)
            for key, value in zip(shared_keys, shared_values, strict=True)
        ),
        padding_entry=CodebookEntry(key=padding_key, value=padding_value),
    )


def _case_manifest(case: CodebookCase) -> dict[str, Any]:
    return {
        "schema_version": case.schema_version,
        "seed": case.seed,
        "case_id": case.case_id,
        "subject_id": case.subject_id,
        "case_index": case.case_index,
        "target_slot": case.target_slot,
        "target_key": case.target_key,
        "old_value": case.old_value,
        "current_value": case.current_value,
        "masked_value": case.masked_value,
        "shared_entries": [
            {"key": entry.key, "value": entry.value} for entry in case.shared_entries
        ],
        "padding_entry": {
            "key": case.padding_entry.key,
            "value": case.padding_entry.value,
        },
    }


def case_manifest_bytes(case: CodebookCase) -> bytes:
    """Return the compact, key-sorted UTF-8 case manifest."""

    return json.dumps(
        _case_manifest(case),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def case_manifest_sha256(case: CodebookCase) -> str:
    return hashlib.sha256(case_manifest_bytes(case)).hexdigest()


def parse_fact(payload: str) -> CodebookEntry:
    """Parse exactly one generated ``key = value`` evidence payload."""

    try:
        payload_bytes = payload.encode("ascii")
    except (AttributeError, UnicodeEncodeError) as error:
        raise ValueError("fact must use the exact ASCII key = value grammar") from error
    match = _FACT_PATTERN.fullmatch(payload_bytes)
    if match is None:
        raise ValueError("fact must use the exact ASCII key = value grammar")
    return CodebookEntry(
        key=match.group("key").decode("ascii"),
        value=match.group("value").decode("ascii"),
    )


def render_context(entries: tuple[ResolvedEntry, ...]) -> RenderedContext:
    """Render ordered entries and derive full-line half-open byte receipts."""

    if not entries:
        return RenderedContext(bytes=b"", entry_receipts=())

    chunks = [_RENDER_HEADER]
    receipts: list[EntryReceipt] = []
    offset = len(_RENDER_HEADER)
    for entry in entries:
        if type(entry.slot) is not int or not 0 <= entry.slot <= 99:
            raise ValueError("entry slot must be an integer from 0 through 99")
        try:
            line = f"{entry.slot:02d}\t{entry.key}\t{entry.value}\n".encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("rendered entries must be ASCII") from error
        if _ENTRY_LINE_PATTERN.fullmatch(line) is None:
            raise ValueError("entry does not match the frozen renderer grammar")
        end = offset + len(line)
        receipts.append(
            EntryReceipt(
                slot=entry.slot,
                key=entry.key,
                value=entry.value,
                source_kind=entry.source_kind,
                revision_id=entry.revision_id,
                candidate_id=entry.candidate_id,
                evidence_ids=entry.evidence_ids,
                content_sha256=hashlib.sha256(
                    f"{entry.key}\t{entry.value}".encode()
                ).hexdigest(),
                rendered_start=offset,
                rendered_end=end,
            )
        )
        chunks.append(line)
        offset = end
    return RenderedContext(bytes=b"".join(chunks), entry_receipts=tuple(receipts))


def _query_key(query: bytes) -> str:
    match = _QUERY_PATTERN.fullmatch(query)
    if match is None:
        raise ValueError("query does not match the frozen scoped-codebook grammar")
    return match.group(1).decode("ascii")


def _parse_rendered_context(context: bytes) -> tuple[tuple[str, str], ...]:
    if not context:
        return ()
    if not context.startswith(_RENDER_HEADER):
        raise ValueError("context does not start with the frozen renderer headers")
    body = context[len(_RENDER_HEADER) :]
    if not body:
        raise ValueError("non-empty context must contain at least one entry")
    parsed: list[tuple[str, str]] = []
    offset = 0
    while offset < len(body):
        match = _ENTRY_LINE_PATTERN.match(body, offset)
        if match is None:
            raise ValueError("context entry does not match the frozen renderer grammar")
        parsed.append(
            (
                match.group("key").decode("ascii"),
                match.group("value").decode("ascii"),
            )
        )
        offset = match.end()
    return tuple(parsed)


def consume_scripted(query: bytes, rendered_context: bytes) -> ConsumerResult:
    """Consume only supplied bytes, using last non-masked target occurrence."""

    receipt = ConsumerInputReceipt(
        received_context_sha256=hashlib.sha256(rendered_context).hexdigest(),
        received_context_utf8_bytes=len(rendered_context),
        received_query_sha256=hashlib.sha256(query).hexdigest(),
    )
    target_key = _query_key(query)
    response = UNKNOWN
    for key, value in _parse_rendered_context(rendered_context):
        if key == target_key and value != MASKED_VALUE:
            response = value
    return ConsumerResult(response=response, input_receipt=receipt)


def normalize_response(response: str) -> str:
    """Apply the frozen Unicode-NFKC, strip, and uppercase normalization."""

    return unicodedata.normalize("NFKC", response).strip().upper()


def abstained(normalized_response: str) -> bool:
    return normalized_response == UNKNOWN


def utility(normalized_response: str, *, current_value: str) -> int:
    """Score one already-normalized response against private case truth."""

    if normalized_response == current_value:
        return 1
    if abstained(normalized_response):
        return 0
    return -1


def _target_and_shared_slots(
    case: CodebookCase,
    *,
    target_value: str,
) -> tuple[tuple[int, CodebookEntry], ...]:
    entries = [
        (
            case.target_slot,
            CodebookEntry(key=case.target_key, value=target_value),
        )
    ]
    shared_slots = iter(slot for slot in range(5) if slot != case.target_slot)
    entries.extend(
        (slot, entry)
        for slot, entry in zip(
            shared_slots,
            case.shared_entries,
            strict=True,
        )
    )
    return tuple(sorted(entries, key=lambda item: item[0]))


def _append_capture_record(
    store: SQLiteMemoryStore,
    *,
    scope: MemoryScope,
    session_id: str,
    run_id: str,
    sequence_no: int,
    kind: EvidenceKind,
    entry: CodebookEntry,
    observed_at: datetime,
    idempotency_key: str,
) -> EvidenceRecord:
    return store.append(
        EvidenceEvent(
            scope=scope,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            kind=kind,
            payload=f"{entry.key} = {entry.value}",
            observed_at=observed_at,
            idempotency_key=idempotency_key,
        )
    )


def _append_grounded_revision(
    store: SQLiteMemoryStore,
    *,
    case_id: str,
    owner: str,
    role: str,
    slot: int,
    evidence: EvidenceRecord,
    operation: RevisionOperation,
    parent_revision_id: str | None = None,
) -> MemoryRevision:
    parse_fact(evidence.event.payload)
    candidate = store.append_candidate(
        CandidateProposal(
            scope=evidence.event.scope,
            content=evidence.event.payload,
            evidence_ids=(evidence.evidence_id,),
            idempotency_key=(f"{case_id}-candidate-{owner}-{role}-{slot:02d}"),
        )
    )
    return store.append_revision(
        RevisionProposal(
            scope=evidence.event.scope,
            candidate_id=candidate.candidate_id,
            operation=operation,
            parent_revision_id=parent_revision_id,
            idempotency_key=(f"{case_id}-revision-{owner}-{role}-{slot:02d}"),
        )
    )


def _manifest_revision_ids(
    case: CodebookCase,
    *,
    target_revision_id: str,
    shared_revision_ids: tuple[str, ...],
    padding_revision_id: str,
) -> tuple[str, ...]:
    shared = iter(shared_revision_ids)
    return tuple(
        target_revision_id if slot == case.target_slot else next(shared)
        for slot in range(5)
    ) + (padding_revision_id,)


def build_case_database(
    case: CodebookCase,
    database_path: str | os.PathLike[str],
) -> CaseDatabaseReferences:
    """Persist one exact local graph and its answer-bearing foreign sentinel."""

    return _build_case_database(case, database_path)


def _build_case_database(
    case: CodebookCase,
    database_path: str | os.PathLike[str],
) -> CaseDatabaseReferences:
    if type(case) is not CodebookCase:
        raise TypeError("case must be a CodebookCase")

    store = SQLiteMemoryStore(database_path)
    local_scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="scoped-codebook-v1",
        subject_id=case.subject_id,
    )
    foreign_scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="scoped-codebook-v1",
        subject_id=f"{case.subject_id}-foreign",
    )
    case_base = datetime(2026, 7, 8, tzinfo=UTC) + timedelta(days=case.case_index)
    raw_history_cutoff = case_base + timedelta(seconds=90)
    capture_session_ids = (
        f"{case.case_id}-capture-old",
        f"{case.case_id}-capture-new",
        f"{case.case_id}-capture-control",
    )

    old_records_by_slot: dict[int, EvidenceRecord] = {}
    for slot, entry in _target_and_shared_slots(case, target_value=case.old_value):
        old_records_by_slot[slot] = _append_capture_record(
            store,
            scope=local_scope,
            session_id=capture_session_ids[0],
            run_id=f"{case.case_id}-run-old",
            sequence_no=slot,
            kind=EvidenceKind.USER_MESSAGE,
            entry=entry,
            observed_at=case_base + timedelta(seconds=slot),
            idempotency_key=f"{case.case_id}-evidence-old-{slot:02d}",
        )
    current_record = _append_capture_record(
        store,
        scope=local_scope,
        session_id=capture_session_ids[1],
        run_id=f"{case.case_id}-run-new",
        sequence_no=0,
        kind=EvidenceKind.FEEDBACK,
        entry=CodebookEntry(key=case.target_key, value=case.current_value),
        observed_at=case_base + timedelta(seconds=60),
        idempotency_key=(f"{case.case_id}-evidence-new-{case.target_slot:02d}"),
    )
    padding_record = _append_capture_record(
        store,
        scope=local_scope,
        session_id=capture_session_ids[2],
        run_id=f"{case.case_id}-run-control",
        sequence_no=0,
        kind=EvidenceKind.ENVIRONMENT,
        entry=case.padding_entry,
        observed_at=case_base + timedelta(seconds=120),
        idempotency_key=f"{case.case_id}-evidence-control-05",
    )
    masked_record = _append_capture_record(
        store,
        scope=local_scope,
        session_id=capture_session_ids[2],
        run_id=f"{case.case_id}-run-control",
        sequence_no=1,
        kind=EvidenceKind.ENVIRONMENT,
        entry=CodebookEntry(key=case.target_key, value=case.masked_value),
        observed_at=case_base + timedelta(seconds=121),
        idempotency_key=(f"{case.case_id}-evidence-control-{case.target_slot:02d}"),
    )
    foreign_record = _append_capture_record(
        store,
        scope=foreign_scope,
        session_id=f"{case.case_id}-capture-foreign",
        run_id=f"{case.case_id}-run-foreign",
        sequence_no=0,
        kind=EvidenceKind.ENVIRONMENT,
        entry=CodebookEntry(key=case.target_key, value=case.current_value),
        observed_at=case_base + timedelta(seconds=180),
        idempotency_key=(
            f"{case.case_id}-evidence-foreign-target-current-{case.target_slot:02d}"
        ),
    )

    target_old_revision = _append_grounded_revision(
        store,
        case_id=case.case_id,
        owner="local",
        role="target-old",
        slot=case.target_slot,
        evidence=old_records_by_slot[case.target_slot],
        operation=RevisionOperation.ADD,
    )
    target_current_revision = _append_grounded_revision(
        store,
        case_id=case.case_id,
        owner="local",
        role="target-current",
        slot=case.target_slot,
        evidence=current_record,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=target_old_revision.revision_id,
    )
    shared_slots = tuple(slot for slot in range(5) if slot != case.target_slot)
    shared_revisions = tuple(
        _append_grounded_revision(
            store,
            case_id=case.case_id,
            owner="local",
            role="shared",
            slot=slot,
            evidence=old_records_by_slot[slot],
            operation=RevisionOperation.ADD,
        )
        for slot in shared_slots
    )
    padding_revision = _append_grounded_revision(
        store,
        case_id=case.case_id,
        owner="local",
        role="padding",
        slot=5,
        evidence=padding_record,
        operation=RevisionOperation.ADD,
    )
    target_masked_revision = _append_grounded_revision(
        store,
        case_id=case.case_id,
        owner="local",
        role="target-masked",
        slot=case.target_slot,
        evidence=masked_record,
        operation=RevisionOperation.ADD,
    )
    foreign_target_revision = _append_grounded_revision(
        store,
        case_id=case.case_id,
        owner="foreign",
        role="target-current",
        slot=case.target_slot,
        evidence=foreign_record,
        operation=RevisionOperation.ADD,
    )

    shared_revision_ids = tuple(revision.revision_id for revision in shared_revisions)
    stale_manifest = ReleaseManifest(
        scope=local_scope,
        revision_ids=_manifest_revision_ids(
            case,
            target_revision_id=target_old_revision.revision_id,
            shared_revision_ids=shared_revision_ids,
            padding_revision_id=padding_revision.revision_id,
        ),
    )
    current_manifest = ReleaseManifest(
        scope=local_scope,
        revision_ids=_manifest_revision_ids(
            case,
            target_revision_id=target_current_revision.revision_id,
            shared_revision_ids=shared_revision_ids,
            padding_revision_id=padding_revision.revision_id,
        ),
    )
    masked_manifest = ReleaseManifest(
        scope=local_scope,
        revision_ids=_manifest_revision_ids(
            case,
            target_revision_id=target_masked_revision.revision_id,
            shared_revision_ids=shared_revision_ids,
            padding_revision_id=padding_revision.revision_id,
        ),
    )
    stale_release = store.append_release(
        stale_manifest,
        idempotency_key=f"{case.case_id}-release-local-stale",
    )
    current_release = store.append_release(
        current_manifest,
        idempotency_key=f"{case.case_id}-release-local-current",
    )
    masked_release = store.append_release(
        masked_manifest,
        idempotency_key=f"{case.case_id}-release-local-masked",
    )
    empty_release = store.append_release(
        ReleaseManifest(scope=local_scope, revision_ids=()),
        idempotency_key=f"{case.case_id}-release-local-empty",
    )
    foreign_release = store.append_release(
        ReleaseManifest(
            scope=foreign_scope,
            revision_ids=(foreign_target_revision.revision_id,),
        ),
        idempotency_key=f"{case.case_id}-release-foreign-sentinel",
    )

    return CaseDatabaseReferences(
        capture=CaptureReferences(
            local_scope=local_scope,
            foreign_scope=foreign_scope,
            case_base=case_base,
            raw_history_cutoff=raw_history_cutoff,
            capture_session_ids=capture_session_ids,
            old_evidence_ids=tuple(
                old_records_by_slot[slot].evidence_id for slot in range(5)
            ),
            current_evidence_id=current_record.evidence_id,
            control_evidence_ids=(
                padding_record.evidence_id,
                masked_record.evidence_id,
            ),
            foreign_evidence_id=foreign_record.evidence_id,
        ),
        revisions=RevisionReferences(
            target_old_revision_id=target_old_revision.revision_id,
            target_current_revision_id=target_current_revision.revision_id,
            shared_revision_ids=shared_revision_ids,
            padding_revision_id=padding_revision.revision_id,
            target_masked_revision_id=target_masked_revision.revision_id,
            foreign_target_revision_id=foreign_target_revision.revision_id,
        ),
        releases=ReleaseAssignments(
            stale_release_id=stale_release.release_id,
            current_release_id=current_release.release_id,
            masked_release_id=masked_release.release_id,
            empty_release_id=empty_release.release_id,
            foreign_sentinel_release_id=foreign_release.release_id,
        ),
    )


def _semantic_entry_hash(entry: ResolvedEntry) -> str:
    return hashlib.sha256(f"{entry.key}\t{entry.value}".encode()).hexdigest()


def _allowed_audit_event(
    *,
    operation: str,
    scope: MemoryScope,
    requested_ids: tuple[str, ...],
    returned_record_ids: tuple[str, ...],
    returned_content_hashes: tuple[str, ...],
) -> ReadAuditEvent:
    return ReadAuditEvent(
        operation=operation,
        requested_scope=scope,
        requested_ids=requested_ids,
        allowed=True,
        returned_record_ids=returned_record_ids,
        returned_content_hashes=returned_content_hashes,
    )


def _denied_audit_event(
    *,
    operation: str,
    scope: MemoryScope,
    requested_id: str,
) -> ReadAuditEvent:
    return ReadAuditEvent(
        operation=operation,
        requested_scope=scope,
        requested_ids=(requested_id,),
        allowed=False,
        returned_record_ids=(),
        returned_content_hashes=(),
    )


class ReleaseReadCapability:
    """Sealed point-reader for one assigned release's reachable graph."""

    __slots__ = (
        "__assignment",
        "__audit",
        "__candidate_ids",
        "__revision_ids",
        "__store",
    )

    def __init__(
        self,
        store: SQLiteMemoryStore,
        assignment: ReleaseSourceAssignment,
        audit: ReadAuditSink,
    ) -> None:
        if type(store) is not SQLiteMemoryStore:
            raise TypeError("store must be a SQLiteMemoryStore")
        if type(assignment) is not ReleaseSourceAssignment:
            raise TypeError("assignment must be a ReleaseSourceAssignment")
        if type(audit) is not ReadAuditSink:
            raise TypeError("audit must be a ReadAuditSink")
        self.__store = store
        self.__assignment = assignment
        self.__audit = audit
        self.__revision_ids: set[str] = set()
        self.__candidate_ids: set[str] = set()

    def get_assigned_release(self) -> MemoryRelease:
        release = self.__store.get_release(
            self.__assignment.scope,
            self.__assignment.release_id,
        )
        self.__revision_ids = set(release.manifest.revision_ids)
        self.__audit._record(
            _allowed_audit_event(
                operation="get_assigned_release",
                scope=self.__assignment.scope,
                requested_ids=(self.__assignment.release_id,),
                returned_record_ids=(release.release_id,),
                returned_content_hashes=(release.content_hash,),
            )
        )
        return release

    def get_revision(self, revision_id: str) -> MemoryRevision:
        if revision_id not in self.__revision_ids:
            self.__audit._record(
                _denied_audit_event(
                    operation="get_revision",
                    scope=self.__assignment.scope,
                    requested_id=revision_id,
                )
            )
            raise UnauthorizedReadError("revision is outside the assigned release")
        revision = self.__store.get_revision(self.__assignment.scope, revision_id)
        self.__candidate_ids.add(revision.proposal.candidate_id)
        self.__audit._record(
            _allowed_audit_event(
                operation="get_revision",
                scope=self.__assignment.scope,
                requested_ids=(revision_id,),
                returned_record_ids=(revision.revision_id,),
                returned_content_hashes=(revision.content_hash,),
            )
        )
        return revision

    def get_candidate(self, candidate_id: str) -> MemoryCandidate:
        if candidate_id not in self.__candidate_ids:
            self.__audit._record(
                _denied_audit_event(
                    operation="get_candidate",
                    scope=self.__assignment.scope,
                    requested_id=candidate_id,
                )
            )
            raise UnauthorizedReadError("candidate is outside the assigned release")
        candidate = self.__store.get_candidate(self.__assignment.scope, candidate_id)
        self.__audit._record(
            _allowed_audit_event(
                operation="get_candidate",
                scope=self.__assignment.scope,
                requested_ids=(candidate_id,),
                returned_record_ids=(candidate.candidate_id,),
                returned_content_hashes=(candidate.content_hash,),
            )
        )
        return candidate


class RawEvidenceReadCapability:
    """Sealed list-reader for the frozen evidence-kind and cutoff policy."""

    __slots__ = ("__assignment", "__audit", "__store")

    def __init__(
        self,
        store: SQLiteMemoryStore,
        assignment: RawSourceAssignment,
        audit: ReadAuditSink,
    ) -> None:
        if type(store) is not SQLiteMemoryStore:
            raise TypeError("store must be a SQLiteMemoryStore")
        if type(assignment) is not RawSourceAssignment:
            raise TypeError("assignment must be a RawSourceAssignment")
        if type(audit) is not ReadAuditSink:
            raise TypeError("audit must be a ReadAuditSink")
        self.__store = store
        self.__assignment = assignment
        self.__audit = audit

    def list_eligible_evidence(self) -> tuple[EvidenceRecord, ...]:
        records = tuple(
            sorted(
                (
                    record
                    for record in self.__store.list(self.__assignment.scope)
                    if record.event.kind
                    in {EvidenceKind.USER_MESSAGE, EvidenceKind.FEEDBACK}
                    and record.event.observed_at <= self.__assignment.cutoff
                ),
                key=lambda record: (
                    record.event.observed_at,
                    record.event.sequence_no,
                    record.evidence_id,
                ),
            )
        )
        self.__audit._record(
            _allowed_audit_event(
                operation="list_eligible_evidence",
                scope=self.__assignment.scope,
                requested_ids=(),
                returned_record_ids=tuple(record.evidence_id for record in records),
                returned_content_hashes=tuple(
                    record.content_hash for record in records
                ),
            )
        )
        return records


class OracleEntryCapability:
    """Store-free capability for explicit evaluator-owned oracle entries."""

    __slots__ = ("__assignment", "__audit")

    def __init__(
        self,
        assignment: OracleSourceAssignment,
        audit: ReadAuditSink,
    ) -> None:
        if type(assignment) is not OracleSourceAssignment:
            raise TypeError("assignment must be an OracleSourceAssignment")
        if type(audit) is not ReadAuditSink:
            raise TypeError("audit must be a ReadAuditSink")
        self.__assignment = assignment
        self.__audit = audit

    def entries(self) -> OracleEntryBatch:
        batch = OracleEntryBatch(
            scope=self.__assignment.scope,
            entries=self.__assignment.entries,
        )
        self.__audit._record(
            _allowed_audit_event(
                operation="entries",
                scope=self.__assignment.scope,
                requested_ids=(),
                returned_record_ids=(),
                returned_content_hashes=tuple(
                    _semantic_entry_hash(entry) for entry in batch.entries
                ),
            )
        )
        return batch


class _CapabilityBoundary:
    """Resolver-facing allowlist with no target capability in instance state."""

    __slots__ = ("__weakref__",)

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            target = _BOUNDARY_TARGETS[self]
            allowed = _BOUNDARY_ALLOWED_METHODS[self]
        except KeyError as error:
            raise AttributeError("unbound capability boundary") from error
        if name not in allowed:
            raise CapabilityInterfaceError(
                f"source capability does not expose {name!r}"
            )
        return getattr(target, name)


_BOUNDARY_TARGETS: WeakKeyDictionary[_CapabilityBoundary, object] = WeakKeyDictionary()
_BOUNDARY_ALLOWED_METHODS: WeakKeyDictionary[_CapabilityBoundary, frozenset[str]] = (
    WeakKeyDictionary()
)


def _resolver_boundary(capability: object) -> _CapabilityBoundary:
    target = _unwrap_capability(capability)
    if type(target) is ReleaseReadCapability:
        allowed = frozenset({"get_assigned_release", "get_revision", "get_candidate"})
    elif type(target) is RawEvidenceReadCapability:
        allowed = frozenset({"list_eligible_evidence"})
    elif type(target) is OracleEntryCapability:
        allowed = frozenset({"entries"})
    else:
        raise TypeError("unsupported source capability")
    boundary = _CapabilityBoundary()
    _BOUNDARY_TARGETS[boundary] = target
    _BOUNDARY_ALLOWED_METHODS[boundary] = allowed
    return boundary


def _unwrap_capability(capability: object) -> object:
    if type(capability) is not _CapabilityBoundary:
        return capability
    try:
        return _BOUNDARY_TARGETS[capability]
    except KeyError as error:
        raise AttributeError("unbound capability boundary") from error


def _resolve_raw_records(
    records: tuple[EvidenceRecord, ...],
) -> tuple[ResolvedEntry, ...]:
    slots_by_key: dict[str, int] = {}
    entries: list[ResolvedEntry] = []
    for record in records:
        parsed = parse_fact(record.event.payload)
        slot = slots_by_key.setdefault(parsed.key, record.event.sequence_no)
        entries.append(
            ResolvedEntry(
                slot=slot,
                key=parsed.key,
                value=parsed.value,
                source_kind="raw_evidence",
                evidence_ids=(record.evidence_id,),
            )
        )
    return tuple(entries)


def resolve_treatment(
    capability: ReleaseReadCapability
    | RawEvidenceReadCapability
    | OracleEntryCapability
    | _CapabilityBoundary,
) -> ResolvedTreatment:
    """Run the frozen resolver against one least-privilege capability."""

    capability = _unwrap_capability(capability)
    if type(capability) is ReleaseReadCapability:
        release = capability.get_assigned_release()
        entries: list[ResolvedEntry] = []
        source_evidence_ids: list[str] = []
        for slot, revision_id in enumerate(release.manifest.revision_ids):
            revision = capability.get_revision(revision_id)
            candidate = capability.get_candidate(revision.proposal.candidate_id)
            parsed = parse_fact(candidate.proposal.content)
            entries.append(
                ResolvedEntry(
                    slot=slot,
                    key=parsed.key,
                    value=parsed.value,
                    source_kind="release",
                    revision_id=revision.revision_id,
                    candidate_id=candidate.candidate_id,
                    evidence_ids=candidate.proposal.evidence_ids,
                )
            )
            source_evidence_ids.extend(candidate.proposal.evidence_ids)
        revision_ids = release.manifest.revision_ids
        return ResolvedTreatment(
            source_kind="release",
            scope=release.manifest.scope,
            release_id=release.release_id,
            eligible_ids=revision_ids,
            retrieved_ids=revision_ids,
            returned_ids=revision_ids,
            source_evidence_ids=tuple(source_evidence_ids),
            entries=tuple(entries),
        )
    if type(capability) is RawEvidenceReadCapability:
        records = capability.list_eligible_evidence()
        evidence_ids = tuple(record.evidence_id for record in records)
        entries = _resolve_raw_records(records)
        scope = records[0].event.scope if records else None
        if scope is None:
            raise TreatmentValidationError("source_or_audit_mismatch")
        return ResolvedTreatment(
            source_kind="raw_evidence",
            scope=scope,
            release_id=None,
            eligible_ids=evidence_ids,
            retrieved_ids=evidence_ids,
            returned_ids=evidence_ids,
            source_evidence_ids=evidence_ids,
            entries=entries,
        )
    if type(capability) is OracleEntryCapability:
        batch = capability.entries()
        if not batch.entries:
            raise TreatmentValidationError("source_or_audit_mismatch")
        return ResolvedTreatment(
            source_kind="oracle",
            scope=batch.scope,
            release_id=None,
            eligible_ids=(),
            retrieved_ids=(),
            returned_ids=(),
            source_evidence_ids=(),
            entries=batch.entries,
        )
    raise TypeError("unsupported source capability")


def _expected_release_source(
    store: SQLiteMemoryStore,
    assignment: ReleaseSourceAssignment,
) -> tuple[ResolvedTreatment, tuple[ReadAuditEvent, ...]]:
    release = store.get_release(assignment.scope, assignment.release_id)
    audit = [
        _allowed_audit_event(
            operation="get_assigned_release",
            scope=assignment.scope,
            requested_ids=(assignment.release_id,),
            returned_record_ids=(release.release_id,),
            returned_content_hashes=(release.content_hash,),
        )
    ]
    entries: list[ResolvedEntry] = []
    evidence_ids: list[str] = []
    for slot, revision_id in enumerate(release.manifest.revision_ids):
        revision = store.get_revision(assignment.scope, revision_id)
        candidate = store.get_candidate(
            assignment.scope,
            revision.proposal.candidate_id,
        )
        audit.extend(
            (
                _allowed_audit_event(
                    operation="get_revision",
                    scope=assignment.scope,
                    requested_ids=(revision_id,),
                    returned_record_ids=(revision.revision_id,),
                    returned_content_hashes=(revision.content_hash,),
                ),
                _allowed_audit_event(
                    operation="get_candidate",
                    scope=assignment.scope,
                    requested_ids=(candidate.candidate_id,),
                    returned_record_ids=(candidate.candidate_id,),
                    returned_content_hashes=(candidate.content_hash,),
                ),
            )
        )
        parsed = parse_fact(candidate.proposal.content)
        if not candidate.proposal.evidence_ids:
            raise TreatmentValidationError("provenance_mismatch")
        for evidence_id in candidate.proposal.evidence_ids:
            evidence = store.get(assignment.scope, evidence_id)
            if evidence.event.payload != candidate.proposal.content:
                raise TreatmentValidationError("provenance_mismatch")
        entries.append(
            ResolvedEntry(
                slot=slot,
                key=parsed.key,
                value=parsed.value,
                source_kind="release",
                revision_id=revision.revision_id,
                candidate_id=candidate.candidate_id,
                evidence_ids=candidate.proposal.evidence_ids,
            )
        )
        evidence_ids.extend(candidate.proposal.evidence_ids)
    revision_ids = release.manifest.revision_ids
    return (
        ResolvedTreatment(
            source_kind="release",
            scope=assignment.scope,
            release_id=assignment.release_id,
            eligible_ids=revision_ids,
            retrieved_ids=revision_ids,
            returned_ids=revision_ids,
            source_evidence_ids=tuple(evidence_ids),
            entries=tuple(entries),
        ),
        tuple(audit),
    )


def _expected_raw_source(
    store: SQLiteMemoryStore,
    assignment: RawSourceAssignment,
) -> tuple[ResolvedTreatment, tuple[ReadAuditEvent, ...]]:
    records = tuple(
        sorted(
            (
                record
                for record in store.list(assignment.scope)
                if record.event.kind
                in {EvidenceKind.USER_MESSAGE, EvidenceKind.FEEDBACK}
                and record.event.observed_at <= assignment.cutoff
            ),
            key=lambda record: (
                record.event.observed_at,
                record.event.sequence_no,
                record.evidence_id,
            ),
        )
    )
    evidence_ids = tuple(record.evidence_id for record in records)
    slots_by_key: dict[str, int] = {}
    expected_entries: list[ResolvedEntry] = []
    for record in records:
        parsed = parse_fact(record.event.payload)
        slot = slots_by_key.setdefault(parsed.key, record.event.sequence_no)
        expected_entries.append(
            ResolvedEntry(
                slot=slot,
                key=parsed.key,
                value=parsed.value,
                source_kind="raw_evidence",
                evidence_ids=(record.evidence_id,),
            )
        )
    expected = ResolvedTreatment(
        source_kind="raw_evidence",
        scope=assignment.scope,
        release_id=None,
        eligible_ids=evidence_ids,
        retrieved_ids=evidence_ids,
        returned_ids=evidence_ids,
        source_evidence_ids=evidence_ids,
        entries=tuple(expected_entries),
    )
    audit = (
        _allowed_audit_event(
            operation="list_eligible_evidence",
            scope=assignment.scope,
            requested_ids=(),
            returned_record_ids=evidence_ids,
            returned_content_hashes=tuple(record.content_hash for record in records),
        ),
    )
    return expected, audit


def _expected_oracle_source(
    assignment: OracleSourceAssignment,
) -> tuple[ResolvedTreatment, tuple[ReadAuditEvent, ...]]:
    if any(
        entry.source_kind != "oracle"
        or entry.revision_id is not None
        or entry.candidate_id is not None
        or entry.evidence_ids
        for entry in assignment.entries
    ):
        raise TreatmentValidationError("provenance_mismatch")
    expected = ResolvedTreatment(
        source_kind="oracle",
        scope=assignment.scope,
        release_id=None,
        eligible_ids=(),
        retrieved_ids=(),
        returned_ids=(),
        source_evidence_ids=(),
        entries=assignment.entries,
    )
    audit = (
        _allowed_audit_event(
            operation="entries",
            scope=assignment.scope,
            requested_ids=(),
            returned_record_ids=(),
            returned_content_hashes=tuple(
                _semantic_entry_hash(entry) for entry in assignment.entries
            ),
        ),
    )
    return expected, audit


def validate_resolved_treatment(
    database_path: str | os.PathLike[str],
    assignment: ReleaseSourceAssignment | RawSourceAssignment | OracleSourceAssignment,
    treatment: ResolvedTreatment,
    reader_audit: tuple[ReadAuditEvent, ...],
) -> None:
    """Independently reopen the source and reject provenance or audit drift."""

    if type(treatment) is not ResolvedTreatment:
        raise TreatmentValidationError("provenance_mismatch")
    if type(assignment) is ReleaseSourceAssignment:
        if (
            treatment.source_kind != "release"
            or treatment.scope != assignment.scope
            or treatment.release_id != assignment.release_id
        ):
            raise TreatmentValidationError("source_or_audit_mismatch")
        expected, expected_audit = _expected_release_source(
            SQLiteMemoryStore(database_path),
            assignment,
        )
    elif type(assignment) is RawSourceAssignment:
        if (
            treatment.source_kind != "raw_evidence"
            or treatment.scope != assignment.scope
            or treatment.release_id is not None
        ):
            raise TreatmentValidationError("source_or_audit_mismatch")
        expected, expected_audit = _expected_raw_source(
            SQLiteMemoryStore(database_path),
            assignment,
        )
    elif type(assignment) is OracleSourceAssignment:
        if (
            treatment.source_kind != "oracle"
            or treatment.scope != assignment.scope
            or treatment.release_id is not None
        ):
            raise TreatmentValidationError("source_or_audit_mismatch")
        expected, expected_audit = _expected_oracle_source(assignment)
    else:
        raise TypeError("unsupported source assignment")

    if len(reader_audit) > len(expected_audit):
        raise TreatmentValidationError("extra_read")
    if reader_audit != expected_audit:
        raise TreatmentValidationError("source_or_audit_mismatch")
    if treatment != expected:
        raise TreatmentValidationError("provenance_mismatch")


def resolve_and_validate(
    resolver: Callable[[object], object],
    capability: object,
    *,
    database_path: str | os.PathLike[str],
    assignment: ReleaseSourceAssignment | RawSourceAssignment | OracleSourceAssignment,
    audit_sink: ReadAuditSink,
) -> ResolvedTreatment:
    """Classify capability failures, then independently validate resolver output."""

    try:
        treatment = resolver(_resolver_boundary(capability))
    except UnauthorizedReadError as error:
        raise TreatmentValidationError("unauthorized_read") from error
    except CapabilityInterfaceError as error:
        raise TreatmentValidationError("capability_interface_violation") from error
    if type(treatment) is not ResolvedTreatment:
        raise TreatmentValidationError("provenance_mismatch")
    validate_resolved_treatment(
        database_path,
        assignment,
        treatment,
        audit_sink.snapshot(),
    )
    return treatment


def make_model_call_receipt(
    *,
    submitted_prompt: bytes,
    context_start: int,
    context_end: int,
    input_token_ids: tuple[int, ...],
) -> ModelCallReceipt:
    """Hash the exact prompt slice and compact integer token-ID sequence."""

    if type(submitted_prompt) is not bytes:
        raise TypeError("submitted_prompt must be bytes")
    if type(context_start) is not int or type(context_end) is not int:
        raise TypeError("context offsets must be integers")
    if not 0 <= context_start <= context_end <= len(submitted_prompt):
        raise ValueError("context offsets must select a submitted prompt slice")
    if type(input_token_ids) is not tuple or any(
        type(token_id) is not int for token_id in input_token_ids
    ):
        raise TypeError("input_token_ids must be a tuple of integers")
    token_bytes = json.dumps(
        list(input_token_ids),
        separators=(",", ":"),
    ).encode()
    return ModelCallReceipt(
        submitted_prompt_sha256=hashlib.sha256(submitted_prompt).hexdigest(),
        submitted_prompt_context_start=context_start,
        submitted_prompt_context_end=context_end,
        submitted_prompt_context_sha256=hashlib.sha256(
            submitted_prompt[context_start:context_end]
        ).hexdigest(),
        submitted_input_token_ids_sha256=hashlib.sha256(token_bytes).hexdigest(),
        submitted_input_token_count=len(input_token_ids),
    )


def make_execution_observation(
    *,
    execution_index: int,
    treatment: ResolvedTreatment,
    reader_audit: tuple[ReadAuditEvent, ...],
    rendered_context: RenderedContext,
    query: bytes,
    consumer_result: ConsumerResult,
    model_call_receipt: ModelCallReceipt | None,
    capture_session_ids: tuple[str, str, str],
    future_session_id: str,
    future_run_id: str,
    capture_pid: int,
    future_pid: int,
    capture_process_instance_id: str,
    future_process_instance_id: str,
    history_length: int,
) -> ExecutionObservation:
    """Create a child-side observation without acknowledging or scoring exposure."""

    return ExecutionObservation(
        execution_index=execution_index,
        source_kind=treatment.source_kind,
        scope=treatment.scope,
        capture_session_ids=capture_session_ids,
        future_session_id=future_session_id,
        future_run_id=future_run_id,
        capture_pid=capture_pid,
        future_pid=future_pid,
        capture_process_instance_id=capture_process_instance_id,
        future_process_instance_id=future_process_instance_id,
        release_id=treatment.release_id,
        eligible_ids=treatment.eligible_ids,
        retrieved_ids=treatment.retrieved_ids,
        returned_ids=treatment.returned_ids,
        source_evidence_ids=treatment.source_evidence_ids,
        entries=rendered_context.entry_receipts,
        reader_audit=reader_audit,
        rendered_context_sha256=hashlib.sha256(rendered_context.bytes).hexdigest(),
        rendered_context_utf8_bytes=len(rendered_context.bytes),
        rendered_context_token_count=None,
        consumer_input_receipt=consumer_result.input_receipt,
        model_call_receipt=model_call_receipt,
        query_sha256=hashlib.sha256(query).hexdigest(),
        history_length=history_length,
        response=consumer_result.response,
    )


def _case_query_bytes(case: CodebookCase) -> bytes:
    return (
        f"What is the current code for {case.target_key}? "
        "Reply with exactly the code or UNKNOWN."
    ).encode()


def _parent_slot_entries(
    case: CodebookCase,
    *,
    target_value: str,
) -> tuple[tuple[int, CodebookEntry], ...]:
    shared = iter(case.shared_entries)
    rows: list[tuple[int, CodebookEntry]] = []
    for slot in range(5):
        if slot == case.target_slot:
            entry = CodebookEntry(case.target_key, target_value)
        else:
            entry = next(shared)
        rows.append((slot, entry))
    return tuple(rows)


def _parent_capture_catalog(
    case: CodebookCase,
    references: CaseDatabaseReferences,
) -> dict[tuple[str, int], tuple[str, str, EvidenceEvent]]:
    capture = references.capture
    expected_base = datetime(2026, 7, 8, tzinfo=UTC) + timedelta(days=case.case_index)
    if (
        capture.local_scope
        != MemoryScope("memory-eval", "scoped-codebook-v1", case.subject_id)
        or capture.case_base != expected_base
        or capture.raw_history_cutoff != expected_base + timedelta(seconds=90)
        or capture.capture_session_ids
        != (
            f"{case.case_id}-capture-old",
            f"{case.case_id}-capture-new",
            f"{case.case_id}-capture-control",
        )
    ):
        raise ValueError("capture references do not match parent case truth")

    catalog: dict[tuple[str, int], tuple[str, str, EvidenceEvent]] = {}

    def record(
        role: str,
        slot: int,
        event: EvidenceEvent,
        expected_id: str,
    ) -> None:
        content_hash = hashlib.sha256(event.canonical_bytes()).hexdigest()
        evidence_id = f"evd_{content_hash[:24]}"
        if evidence_id != expected_id:
            raise ValueError("capture evidence address does not match parent truth")
        catalog[(role, slot)] = (evidence_id, content_hash, event)

    for slot, entry in _parent_slot_entries(case, target_value=case.old_value):
        record(
            "old",
            slot,
            EvidenceEvent(
                scope=capture.local_scope,
                session_id=f"{case.case_id}-capture-old",
                run_id=f"{case.case_id}-run-old",
                sequence_no=slot,
                kind=EvidenceKind.USER_MESSAGE,
                payload=f"{entry.key} = {entry.value}",
                observed_at=expected_base + timedelta(seconds=slot),
                idempotency_key=f"{case.case_id}-evidence-old-{slot:02d}",
            ),
            capture.old_evidence_ids[slot],
        )
    record(
        "current",
        case.target_slot,
        EvidenceEvent(
            scope=capture.local_scope,
            session_id=f"{case.case_id}-capture-new",
            run_id=f"{case.case_id}-run-new",
            sequence_no=0,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{case.target_key} = {case.current_value}",
            observed_at=expected_base + timedelta(seconds=60),
            idempotency_key=(f"{case.case_id}-evidence-new-{case.target_slot:02d}"),
        ),
        capture.current_evidence_id,
    )
    record(
        "padding",
        5,
        EvidenceEvent(
            scope=capture.local_scope,
            session_id=f"{case.case_id}-capture-control",
            run_id=f"{case.case_id}-run-control",
            sequence_no=0,
            kind=EvidenceKind.ENVIRONMENT,
            payload=f"{case.padding_entry.key} = {case.padding_entry.value}",
            observed_at=expected_base + timedelta(seconds=120),
            idempotency_key=f"{case.case_id}-evidence-control-05",
        ),
        capture.control_evidence_ids[0],
    )
    record(
        "masked",
        case.target_slot,
        EvidenceEvent(
            scope=capture.local_scope,
            session_id=f"{case.case_id}-capture-control",
            run_id=f"{case.case_id}-run-control",
            sequence_no=1,
            kind=EvidenceKind.ENVIRONMENT,
            payload=f"{case.target_key} = {case.masked_value}",
            observed_at=expected_base + timedelta(seconds=121),
            idempotency_key=(f"{case.case_id}-evidence-control-{case.target_slot:02d}"),
        ),
        capture.control_evidence_ids[1],
    )
    return catalog


def _parent_graph_entry(
    *,
    case: CodebookCase,
    scope: MemoryScope,
    slot: int,
    role: str,
    content: str,
    evidence_id: str,
    expected_revision_id: str,
    operation: RevisionOperation,
    parent_revision_id: str | None,
) -> tuple[ResolvedEntry, str, str]:
    candidate_proposal = CandidateProposal(
        scope=scope,
        content=content,
        evidence_ids=(evidence_id,),
        idempotency_key=(f"{case.case_id}-candidate-local-{role}-{slot:02d}"),
    )
    candidate_hash = hashlib.sha256(candidate_proposal.canonical_bytes()).hexdigest()
    candidate_id = f"cand_{candidate_hash[:24]}"
    revision_proposal = RevisionProposal(
        scope=scope,
        candidate_id=candidate_id,
        operation=operation,
        parent_revision_id=parent_revision_id,
        idempotency_key=(f"{case.case_id}-revision-local-{role}-{slot:02d}"),
    )
    revision_hash = hashlib.sha256(revision_proposal.canonical_bytes()).hexdigest()
    revision_id = f"rev_{revision_hash[:24]}"
    if revision_id != expected_revision_id:
        raise ValueError("revision address does not match parent case truth")
    parsed = parse_fact(content)
    return (
        ResolvedEntry(
            slot=slot,
            key=parsed.key,
            value=parsed.value,
            source_kind="release",
            revision_id=revision_id,
            candidate_id=candidate_id,
            evidence_ids=(evidence_id,),
        ),
        revision_hash,
        candidate_hash,
    )


def _parent_release_source_contract(
    case: CodebookCase,
    references: CaseDatabaseReferences,
    *,
    arm: str,
    release_id: str,
) -> ParentSourceContract:
    catalog = _parent_capture_catalog(case, references)
    scope = references.capture.local_scope
    revisions = references.revisions
    rows: list[tuple[ResolvedEntry, str, str]] = []
    if arm != "memory_off":
        if arm == "current_release":
            target_role = "target-current"
            target_catalog_role = "current"
            target_value = case.current_value
            target_revision_id = revisions.target_current_revision_id
            target_operation = RevisionOperation.SUPERSEDE
            target_parent = revisions.target_old_revision_id
        elif arm == "stale_release":
            target_role = "target-old"
            target_catalog_role = "old"
            target_value = case.old_value
            target_revision_id = revisions.target_old_revision_id
            target_operation = RevisionOperation.ADD
            target_parent = None
        elif arm == "target_masked":
            target_role = "target-masked"
            target_catalog_role = "masked"
            target_value = case.masked_value
            target_revision_id = revisions.target_masked_revision_id
            target_operation = RevisionOperation.ADD
            target_parent = None
        else:
            raise ValueError("release arm is not frozen")
        shared_revision_ids = iter(revisions.shared_revision_ids)
        for slot, entry in _parent_slot_entries(case, target_value=target_value):
            if slot == case.target_slot:
                role = target_role
                evidence_id = catalog[(target_catalog_role, slot)][0]
                expected_revision_id = target_revision_id
                operation = target_operation
                parent_revision_id = target_parent
            else:
                role = "shared"
                evidence_id = catalog[("old", slot)][0]
                expected_revision_id = next(shared_revision_ids)
                operation = RevisionOperation.ADD
                parent_revision_id = None
            rows.append(
                _parent_graph_entry(
                    case=case,
                    scope=scope,
                    slot=slot,
                    role=role,
                    content=f"{entry.key} = {entry.value}",
                    evidence_id=evidence_id,
                    expected_revision_id=expected_revision_id,
                    operation=operation,
                    parent_revision_id=parent_revision_id,
                )
            )
        rows.append(
            _parent_graph_entry(
                case=case,
                scope=scope,
                slot=5,
                role="padding",
                content=f"{case.padding_entry.key} = {case.padding_entry.value}",
                evidence_id=catalog[("padding", 5)][0],
                expected_revision_id=revisions.padding_revision_id,
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
            )
        )

    revision_ids = tuple(row[0].revision_id for row in rows)
    if any(revision_id is None for revision_id in revision_ids):
        raise AssertionError("parent release rows require revision IDs")
    typed_revision_ids = tuple(
        revision_id for revision_id in revision_ids if revision_id is not None
    )
    manifest = ReleaseManifest(scope=scope, revision_ids=typed_revision_ids)
    release_hash = hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    if f"rel_{release_hash[:24]}" != release_id:
        raise ValueError("release address does not match parent case truth")
    audit: list[ReadAuditEvent] = [
        _allowed_audit_event(
            operation="get_assigned_release",
            scope=scope,
            requested_ids=(release_id,),
            returned_record_ids=(release_id,),
            returned_content_hashes=(release_hash,),
        )
    ]
    evidence_ids: list[str] = []
    for entry, revision_hash, candidate_hash in rows:
        assert entry.revision_id is not None
        assert entry.candidate_id is not None
        audit.extend(
            (
                _allowed_audit_event(
                    operation="get_revision",
                    scope=scope,
                    requested_ids=(entry.revision_id,),
                    returned_record_ids=(entry.revision_id,),
                    returned_content_hashes=(revision_hash,),
                ),
                _allowed_audit_event(
                    operation="get_candidate",
                    scope=scope,
                    requested_ids=(entry.candidate_id,),
                    returned_record_ids=(entry.candidate_id,),
                    returned_content_hashes=(candidate_hash,),
                ),
            )
        )
        evidence_ids.extend(entry.evidence_ids)
    rendered = render_context(tuple(row[0] for row in rows))
    return ParentSourceContract(
        eligible_ids=typed_revision_ids,
        retrieved_ids=typed_revision_ids,
        returned_ids=typed_revision_ids,
        source_evidence_ids=tuple(evidence_ids),
        entries=rendered.entry_receipts,
        reader_audit=tuple(audit),
        rendered_context_sha256=hashlib.sha256(rendered.bytes).hexdigest(),
        rendered_context_utf8_bytes=len(rendered.bytes),
    )


def _parent_raw_source_contract(
    case: CodebookCase,
    references: CaseDatabaseReferences,
) -> ParentSourceContract:
    catalog = _parent_capture_catalog(case, references)
    records = tuple(catalog[("old", slot)] for slot in range(5)) + (
        catalog[("current", case.target_slot)],
    )
    entries: list[ResolvedEntry] = []
    slots_by_key: dict[str, int] = {}
    for evidence_id, _content_hash, event in records:
        parsed = parse_fact(event.payload)
        slot = slots_by_key.setdefault(parsed.key, event.sequence_no)
        entries.append(
            ResolvedEntry(
                slot=slot,
                key=parsed.key,
                value=parsed.value,
                source_kind="raw_evidence",
                evidence_ids=(evidence_id,),
            )
        )
    evidence_ids = tuple(record[0] for record in records)
    audit = (
        _allowed_audit_event(
            operation="list_eligible_evidence",
            scope=references.capture.local_scope,
            requested_ids=(),
            returned_record_ids=evidence_ids,
            returned_content_hashes=tuple(record[1] for record in records),
        ),
    )
    rendered = render_context(tuple(entries))
    return ParentSourceContract(
        eligible_ids=evidence_ids,
        retrieved_ids=evidence_ids,
        returned_ids=evidence_ids,
        source_evidence_ids=evidence_ids,
        entries=rendered.entry_receipts,
        reader_audit=audit,
        rendered_context_sha256=hashlib.sha256(rendered.bytes).hexdigest(),
        rendered_context_utf8_bytes=len(rendered.bytes),
    )


def _parent_oracle_source_contract(
    case: CodebookCase,
    references: CaseDatabaseReferences,
) -> ParentSourceContract:
    entries = tuple(
        ResolvedEntry(
            slot=slot,
            key=entry.key,
            value=entry.value,
            source_kind="oracle",
        )
        for slot, entry in _parent_slot_entries(
            case,
            target_value=case.current_value,
        )
    ) + (
        ResolvedEntry(
            slot=5,
            key=case.padding_entry.key,
            value=case.padding_entry.value,
            source_kind="oracle",
        ),
    )
    audit = (
        _allowed_audit_event(
            operation="entries",
            scope=references.capture.local_scope,
            requested_ids=(),
            returned_record_ids=(),
            returned_content_hashes=tuple(
                _semantic_entry_hash(entry) for entry in entries
            ),
        ),
    )
    rendered = render_context(entries)
    return ParentSourceContract(
        eligible_ids=(),
        retrieved_ids=(),
        returned_ids=(),
        source_evidence_ids=(),
        entries=rendered.entry_receipts,
        reader_audit=audit,
        rendered_context_sha256=hashlib.sha256(rendered.bytes).hexdigest(),
        rendered_context_utf8_bytes=len(rendered.bytes),
    )


def make_parent_schedule_item(
    *,
    execution_index: int,
    case: CodebookCase,
    references: CaseDatabaseReferences,
    arm: str,
) -> ParentScheduleItem:
    """Create one private schedule row without exposing it to child execution."""

    source_kind: str
    release_id: str | None
    expected_response: str
    expected_source: ParentSourceContract
    if arm == "current_release":
        source_kind = "release"
        release_id = references.releases.current_release_id
        expected_response = case.current_value
        expected_source = _parent_release_source_contract(
            case,
            references,
            arm=arm,
            release_id=release_id,
        )
    elif arm == "raw_history":
        source_kind = "raw_evidence"
        release_id = None
        expected_response = case.current_value
        expected_source = _parent_raw_source_contract(case, references)
    elif arm == "memory_off":
        source_kind = "release"
        release_id = references.releases.empty_release_id
        expected_response = UNKNOWN
        expected_source = _parent_release_source_contract(
            case,
            references,
            arm=arm,
            release_id=release_id,
        )
    elif arm == "target_masked":
        source_kind = "release"
        release_id = references.releases.masked_release_id
        expected_response = UNKNOWN
        expected_source = _parent_release_source_contract(
            case,
            references,
            arm=arm,
            release_id=release_id,
        )
    elif arm == "stale_release":
        source_kind = "release"
        release_id = references.releases.stale_release_id
        expected_response = case.old_value
        expected_source = _parent_release_source_contract(
            case,
            references,
            arm=arm,
            release_id=release_id,
        )
    elif arm == "oracle":
        source_kind = "oracle"
        release_id = None
        expected_response = case.current_value
        expected_source = _parent_oracle_source_contract(case, references)
    else:
        raise ValueError("arm is not part of the frozen six-arm schedule")
    return ParentScheduleItem(
        execution_index=execution_index,
        case=case,
        case_manifest_sha256=case_manifest_sha256(case),
        arm=arm,
        source_kind=source_kind,
        scope=references.capture.local_scope,
        release_id=release_id,
        capture_session_ids=references.capture.capture_session_ids,
        query_sha256=hashlib.sha256(_case_query_bytes(case)).hexdigest(),
        expected_response=expected_response,
        expected_source=expected_source,
    )


def _validate_receipts_and_acknowledge(
    observation: ExecutionObservation,
    schedule: ParentScheduleItem,
) -> tuple[str, ...]:
    receipt = observation.consumer_input_receipt
    if (
        receipt.received_context_sha256 != observation.rendered_context_sha256
        or receipt.received_context_utf8_bytes
        != observation.rendered_context_utf8_bytes
    ):
        raise ObservationValidationError("received_context_mismatch")
    if (
        observation.query_sha256 != schedule.query_sha256
        or receipt.received_query_sha256 != observation.query_sha256
    ):
        raise ObservationValidationError("received_query_mismatch")

    model_receipt = observation.model_call_receipt
    if model_receipt is not None:
        if (
            model_receipt.submitted_prompt_context_start < 0
            or model_receipt.submitted_prompt_context_end
            < model_receipt.submitted_prompt_context_start
            or model_receipt.submitted_prompt_context_end
            - model_receipt.submitted_prompt_context_start
            != observation.rendered_context_utf8_bytes
            or model_receipt.submitted_prompt_context_sha256
            != observation.rendered_context_sha256
        ):
            raise ObservationValidationError("call_boundary_context_mismatch")
        if (
            _SHA256_PATTERN.fullmatch(model_receipt.submitted_prompt_sha256) is None
            or _SHA256_PATTERN.fullmatch(model_receipt.submitted_prompt_context_sha256)
            is None
            or _SHA256_PATTERN.fullmatch(model_receipt.submitted_input_token_ids_sha256)
            is None
            or model_receipt.submitted_input_token_count < 0
        ):
            raise ObservationValidationError("call_boundary_token_mismatch")

    last_end = 0
    for entry in observation.entries:
        if (
            entry.rendered_start < last_end
            or entry.rendered_start < 0
            or entry.rendered_end <= entry.rendered_start
            or entry.rendered_end > observation.rendered_context_utf8_bytes
            or entry.content_sha256
            != hashlib.sha256(f"{entry.key}\t{entry.value}".encode()).hexdigest()
        ):
            raise ObservationValidationError("injection_provenance_mismatch")
        last_end = entry.rendered_end

    if observation.source_kind == "release":
        injected_revision_ids: list[str] = []
        evidence_ids: list[str] = []
        for entry in observation.entries:
            if entry.revision_id is None or entry.candidate_id is None:
                raise ObservationValidationError("injection_provenance_mismatch")
            injected_revision_ids.append(entry.revision_id)
            evidence_ids.extend(entry.evidence_ids)
        if (
            tuple(injected_revision_ids) != observation.returned_ids
            or tuple(evidence_ids) != observation.source_evidence_ids
        ):
            raise ObservationValidationError("injection_provenance_mismatch")
        return tuple(injected_revision_ids)
    if observation.source_kind == "raw_evidence":
        evidence_ids = []
        for entry in observation.entries:
            if (
                entry.revision_id is not None
                or entry.candidate_id is not None
                or not entry.evidence_ids
            ):
                raise ObservationValidationError("injection_provenance_mismatch")
            evidence_ids.extend(entry.evidence_ids)
        if (
            tuple(evidence_ids) != observation.returned_ids
            or tuple(evidence_ids) != observation.source_evidence_ids
        ):
            raise ObservationValidationError("injection_provenance_mismatch")
        return ()
    if observation.source_kind == "oracle":
        if (
            observation.returned_ids
            or observation.source_evidence_ids
            or any(
                entry.revision_id is not None
                or entry.candidate_id is not None
                or entry.evidence_ids
                for entry in observation.entries
            )
        ):
            raise ObservationValidationError("injection_provenance_mismatch")
        return ()
    raise ObservationValidationError("assignment_mismatch")


def _validate_parent_source_contract(
    observation: ExecutionObservation,
    schedule: ParentScheduleItem,
) -> None:
    expected = schedule.expected_source
    if observation.reader_audit != expected.reader_audit:
        raise ObservationValidationError("source_or_audit_mismatch")
    if (
        observation.eligible_ids != expected.eligible_ids
        or observation.retrieved_ids != expected.retrieved_ids
        or observation.returned_ids != expected.returned_ids
        or observation.source_evidence_ids != expected.source_evidence_ids
    ):
        raise ObservationValidationError("provenance_mismatch")
    if observation.entries != expected.entries:
        raise ObservationValidationError("injection_provenance_mismatch")
    if (
        observation.rendered_context_sha256 != expected.rendered_context_sha256
        or observation.rendered_context_utf8_bytes
        != expected.rendered_context_utf8_bytes
    ):
        raise ObservationValidationError("received_context_mismatch")


def _trace_from_observation(
    observation: ExecutionObservation,
    schedule: ParentScheduleItem,
    *,
    injected_revision_ids: tuple[str, ...],
    normalized_response: str,
) -> EvaluationTrace:
    model = observation.model_call_receipt
    release_ids = (
        observation.eligible_ids if observation.source_kind == "release" else ()
    )
    retrieved_ids = (
        observation.retrieved_ids if observation.source_kind == "release" else ()
    )
    returned_ids = (
        observation.returned_ids if observation.source_kind == "release" else ()
    )
    followed = any(
        entry.key == schedule.case.target_key
        and entry.value != MASKED_VALUE
        and entry.value == normalized_response
        for entry in observation.entries
    )
    return EvaluationTrace(
        schema_version=SCHEMA_VERSION,
        case_id=schedule.case.case_id,
        case_manifest_sha256=schedule.case_manifest_sha256,
        execution_index=observation.execution_index,
        arm=schedule.arm,
        source_kind=observation.source_kind,
        scope=observation.scope,
        capture_session_ids=observation.capture_session_ids,
        future_session_id=observation.future_session_id,
        future_run_id=observation.future_run_id,
        capture_pid=observation.capture_pid,
        future_pid=observation.future_pid,
        capture_process_instance_id=observation.capture_process_instance_id,
        future_process_instance_id=observation.future_process_instance_id,
        release_id=observation.release_id,
        eligible_revision_ids=release_ids,
        retrieved_revision_ids=retrieved_ids,
        returned_revision_ids=returned_ids,
        injected_revision_ids=injected_revision_ids,
        source_evidence_ids=observation.source_evidence_ids,
        entries=observation.entries,
        reader_audit=observation.reader_audit,
        rendered_context_sha256=observation.rendered_context_sha256,
        rendered_context_utf8_bytes=observation.rendered_context_utf8_bytes,
        rendered_context_token_count=observation.rendered_context_token_count,
        received_context_sha256=(
            observation.consumer_input_receipt.received_context_sha256
        ),
        received_context_utf8_bytes=(
            observation.consumer_input_receipt.received_context_utf8_bytes
        ),
        received_query_sha256=(
            observation.consumer_input_receipt.received_query_sha256
        ),
        submitted_prompt_sha256=(
            None if model is None else model.submitted_prompt_sha256
        ),
        submitted_prompt_context_start=(
            None if model is None else model.submitted_prompt_context_start
        ),
        submitted_prompt_context_end=(
            None if model is None else model.submitted_prompt_context_end
        ),
        submitted_prompt_context_sha256=(
            None if model is None else model.submitted_prompt_context_sha256
        ),
        submitted_input_token_ids_sha256=(
            None if model is None else model.submitted_input_token_ids_sha256
        ),
        submitted_input_token_count=(
            None if model is None else model.submitted_input_token_count
        ),
        query_sha256=observation.query_sha256,
        history_length=observation.history_length,
        response=observation.response,
        normalized_response=normalized_response,
        expected_response=schedule.expected_response,
        utility=utility(
            normalized_response,
            current_value=schedule.case.current_value,
        ),
        abstained=abstained(normalized_response),
        followed_injected_value=followed,
    )


def parent_join_and_score(
    observations: tuple[ExecutionObservation, ...],
    schedule: tuple[ParentScheduleItem, ...],
    *,
    enforce_scripted_outcomes: bool,
) -> tuple[EvaluationTrace, ...]:
    """Validate exposure, privately join case truth, then normalize and score."""

    expected_indexes = tuple(item.execution_index for item in schedule)
    observed_indexes = tuple(item.execution_index for item in observations)
    if (
        len(set(expected_indexes)) != len(expected_indexes)
        or len(set(observed_indexes)) != len(observed_indexes)
        or set(observed_indexes) != set(expected_indexes)
    ):
        raise ObservationValidationError("execution_index_mismatch")
    observations_by_index = {
        observation.execution_index: observation for observation in observations
    }
    acknowledged: list[
        tuple[ExecutionObservation, ParentScheduleItem, tuple[str, ...]]
    ] = []
    for scheduled in schedule:
        observation = observations_by_index[scheduled.execution_index]
        if (
            observation.source_kind != scheduled.source_kind
            or observation.scope != scheduled.scope
            or observation.release_id != scheduled.release_id
            or observation.capture_session_ids != scheduled.capture_session_ids
        ):
            raise ObservationValidationError("assignment_mismatch")
        _validate_parent_source_contract(observation, scheduled)
        injected_revision_ids = _validate_receipts_and_acknowledge(
            observation,
            scheduled,
        )
        acknowledged.append((observation, scheduled, injected_revision_ids))

    normalized_rows: list[
        tuple[
            ExecutionObservation,
            ParentScheduleItem,
            tuple[str, ...],
            str,
        ]
    ] = []
    for observation, scheduled, injected_revision_ids in acknowledged:
        normalized = normalize_response(observation.response)
        if enforce_scripted_outcomes and normalized != scheduled.expected_response:
            reason = (
                "raw_order_failure"
                if scheduled.arm == "raw_history"
                and normalized == scheduled.case.old_value
                else "strict_outcome_failure"
            )
            raise ObservationValidationError(reason)
        normalized_rows.append(
            (observation, scheduled, injected_revision_ids, normalized)
        )

    traces: list[EvaluationTrace] = []
    for observation, scheduled, injected_revision_ids, normalized in normalized_rows:
        traces.append(
            _trace_from_observation(
                observation,
                scheduled,
                injected_revision_ids=injected_revision_ids,
                normalized_response=normalized,
            )
        )
    return tuple(traces)
