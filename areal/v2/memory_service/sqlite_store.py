# SPDX-License-Identifier: Apache-2.0

"""File-backed SQLite implementation of immutable Memory Service history."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from areal.v2.memory_service._sqlite_backend import (
    _application_revision_binding_hash,
    _evidence_ingest_binding_hash,
    _initialize_database,
    _read_transaction,
    _record_storage_hash,
    _release_binding_hash,
    _snapshot_binding_hash,
    _snapshot_database_path,
    _validate_ingest_orders_locked,
    _write_transaction,
)
from areal.v2.memory_service.application_types import (
    AppliedMemoryUpdateV1,
    MemoryApplicationProposal,
    MemoryApplicationRootV1,
    MemoryApplicationUpdateProposal,
    MemoryApplicationV1,
)
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    EvidenceSnapshotConflictError,
    EvidenceSnapshotNotFoundError,
    MemoryApplicationConflictError,
    MemoryApplicationNotFoundError,
    MemoryApplicationRootConflictError,
    MemoryApplicationRootNotFoundError,
    MemoryApplicationStaleSnapshotError,
    MemoryPersistenceCorruptionError,
    ReleaseConflictError,
    ReleaseNotFoundError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.snapshot_types import (
    EVIDENCE_SNAPSHOT_ORDERING_POLICY,
    EvidenceSnapshot,
    EvidenceSnapshotMember,
    EvidenceSnapshotSpec,
)
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
    _validate_string,
)

_EVIDENCE_SELECT = """SELECT evidence_id, canonical, content_hash,
       created_at, storage_hash, session_id, run_id, sequence_no,
       kind, payload, observed_at, idempotency_key
FROM memory_evidence
WHERE scope_id = ? AND evidence_id = ?"""

_EVIDENCE_ROW_TYPES = (
    str,
    bytes,
    str,
    str,
    str,
    str,
    str,
    int,
    str,
    str,
    str,
    str,
)

_EVIDENCE_SNAPSHOT_SELECT = """SELECT snapshot_id, canonical, content_hash,
       created_at, storage_hash, allowed_kinds_canonical, cutoff_utc,
       evidence_high_watermark, ordering_policy, member_count
FROM memory_evidence_snapshots
WHERE scope_id = ? AND snapshot_id = ?"""

_EVIDENCE_SNAPSHOT_ROW_TYPES = (
    str,
    bytes,
    str,
    str,
    str,
    bytes,
    str,
    int,
    str,
    int,
)

_CANDIDATE_SELECT = """SELECT candidate_id, canonical, content_hash,
       created_at, storage_hash, content, idempotency_key
FROM memory_candidates
WHERE scope_id = ? AND candidate_id = ?"""

_CANDIDATE_ROW_TYPES = (str, bytes, str, str, str, str, str)

_REVISION_SELECT = """SELECT revision_id, canonical, content_hash,
       created_at, storage_hash, candidate_id, memory_id, generation,
       operation, parent_revision_id, idempotency_key
FROM memory_revisions
WHERE scope_id = ? AND revision_id = ?"""

_REVISION_ROW_TYPES = (
    (str,),
    (bytes,),
    (str,),
    (str,),
    (str,),
    (str,),
    (str,),
    (int,),
    (str,),
    (str, type(None)),
    (str,),
)

_RELEASE_SELECT = """SELECT release_id, canonical, content_hash,
       created_at, storage_hash
FROM memory_releases
WHERE scope_id = ? AND release_id = ?"""

_RELEASE_ROW_TYPES = (str, bytes, str, str, str)

_APPLICATION_ROOT_SELECT = """SELECT root_id, canonical, content_hash,
       created_at, storage_hash, release_id, release_content_hash
FROM memory_application_roots
WHERE scope_id = ?"""

_APPLICATION_SELECT = """SELECT application_order, application_id, canonical,
       content_hash, created_at, storage_hash, source_snapshot_id,
       source_snapshot_content_hash, source_evidence_high_watermark,
       base_release_id, base_release_content_hash, result_release_id,
       result_release_content_hash, projector_id, projector_version_hash,
       policy_id, policy_version_hash, policy_input_hash, decision_hash,
       policy_context, update_count, idempotency_key
FROM memory_applications
WHERE scope_id = ? AND application_id = ?"""

_APPLICATION_ROW_TYPES = (
    int,
    str,
    bytes,
    str,
    str,
    str,
    str,
    str,
    int,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    str,
)

_APPLICATION_CHILD_KEY_DOMAIN = b"areal-memory-application-child-key-v1\0"


def _application_transaction_fault_hook(_stage: str) -> None:
    """Private deterministic failure seam used by atomicity tests."""


@dataclass(frozen=True, slots=True)
class _ApplicationState:
    scope_by_id: dict[int, MemoryScope]
    evidence_by_address: dict[tuple[int, str], EvidenceRecord]
    ingest_order_by_address: dict[tuple[int, str], int]
    snapshot_by_address: dict[tuple[int, str], EvidenceSnapshot]
    candidate_by_address: dict[tuple[int, str], MemoryCandidate]
    revision_by_address: dict[tuple[int, str], MemoryRevision]
    release_by_address: dict[tuple[int, str], MemoryRelease]
    release_by_alias: dict[tuple[int, str], MemoryRelease]
    revisions_by_release: dict[tuple[int, str], tuple[MemoryRevision, ...]]
    root_by_scope_id: dict[int, MemoryApplicationRootV1]
    root_by_release: dict[tuple[int, str], MemoryApplicationRootV1]
    application_by_address: dict[tuple[int, str], MemoryApplicationV1]
    application_by_idempotency: dict[tuple[int, str], MemoryApplicationV1]
    application_by_result_release: dict[tuple[int, str], MemoryApplicationV1]
    application_by_revision: dict[tuple[int, str], MemoryApplicationV1]

_MAX_SCOPE_ID = 2**63 - 1
_MAX_GENERATION = 2**63 - 1


def _require_scope_id(value: object, message: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_SCOPE_ID:
        raise MemoryPersistenceCorruptionError(message)
    return value


def _load_scope_index(cursor: sqlite3.Cursor) -> dict[int, MemoryScope]:
    rows = cursor.execute(
        "SELECT scope_id, tenant_id, namespace, subject_id FROM memory_scopes"
    ).fetchall()
    scope_by_id: dict[int, MemoryScope] = {}
    scope_ids_by_identity: dict[MemoryScope, int] = {}
    for row in rows:
        if len(row) != 4:
            raise MemoryPersistenceCorruptionError(
                "memory scope row does not contain exactly four values"
            )
        scope_id = _require_scope_id(
            row[0],
            "memory scope identifier is not a positive signed 64-bit integer",
        )
        stored_identity = row[1:]
        if not all(type(value) is str for value in stored_identity):
            raise MemoryPersistenceCorruptionError(
                "memory scope identity values must be text"
            )
        try:
            stored_scope = MemoryScope(
                tenant_id=stored_identity[0],
                namespace=stored_identity[1],
                subject_id=stored_identity[2],
            )
        except (TypeError, ValueError) as error:
            raise MemoryPersistenceCorruptionError(
                "stored memory scope identity failed validation"
            ) from error
        if scope_id in scope_by_id:
            raise MemoryPersistenceCorruptionError(
                "memory scope identifier appears in multiple rows"
            )
        if stored_scope in scope_ids_by_identity:
            raise MemoryPersistenceCorruptionError(
                "memory scope identity matches multiple rows"
            )
        scope_by_id[scope_id] = stored_scope
        scope_ids_by_identity[stored_scope] = scope_id
    return scope_by_id


def _find_scope_id(cursor: sqlite3.Cursor, scope: MemoryScope) -> int | None:
    for scope_id, stored_scope in _load_scope_index(cursor).items():
        if stored_scope == scope:
            return scope_id
    return None


def _ensure_scope_id(cursor: sqlite3.Cursor, scope: MemoryScope) -> int:
    scope_id = _find_scope_id(cursor, scope)
    if scope_id is not None:
        return scope_id
    cursor.execute(
        "INSERT INTO memory_scopes (tenant_id, namespace, subject_id) VALUES (?, ?, ?)",
        (scope.tenant_id, scope.namespace, scope.subject_id),
    )
    scope_id = _require_scope_id(
        cursor.lastrowid,
        "memory scope insert did not return a positive signed 64-bit identifier",
    )
    persisted_scope_id = _find_scope_id(cursor, scope)
    if persisted_scope_id != scope_id:
        raise MemoryPersistenceCorruptionError(
            "memory scope insert did not round-trip its identifier"
        )
    return scope_id


def _load_evidence(
    cursor: sqlite3.Cursor,
    scope: MemoryScope,
    scope_id: int,
    evidence_id: str,
) -> EvidenceRecord | None:
    _require_scope_id(
        scope_id,
        "evidence lookup received an invalid positive signed 64-bit scope ID",
    )
    row = cursor.execute(_EVIDENCE_SELECT, (scope_id, evidence_id)).fetchone()
    if row is None:
        return None
    try:
        if len(row) != len(_EVIDENCE_ROW_TYPES):
            raise ValueError("evidence row has the wrong field count")
        for index, (value, expected_type) in enumerate(
            zip(row, _EVIDENCE_ROW_TYPES, strict=True)
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"evidence row field {index} has the wrong storage class"
                )
        (
            stored_evidence_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            session_id,
            run_id,
            sequence_no,
            kind_text,
            payload,
            observed_at_text,
            idempotency_key,
        ) = row
        if stored_evidence_id != evidence_id:
            raise ValueError("loaded evidence ID differs from requested ID")
        event = EvidenceEvent(
            scope=scope,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            kind=EvidenceKind(kind_text),
            payload=payload,
            observed_at=datetime.fromisoformat(observed_at_text),
            idempotency_key=idempotency_key,
        )
        record = EvidenceRecord(
            evidence_id=stored_evidence_id,
            event=event,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if event.observed_at.isoformat() != observed_at_text:
            raise ValueError("observed_at is not exact UTC isoformat text")
        if record.created_at.isoformat() != created_at_text:
            raise ValueError("created_at is not exact UTC isoformat text")
        if event.canonical_bytes() != canonical:
            raise ValueError("canonical evidence bytes disagree with projections")
        calculated_hash = sha256(canonical).hexdigest()
        if content_hash != calculated_hash:
            raise ValueError("evidence content hash disagrees with canonical bytes")
        calculated_id = f"evd_{calculated_hash[:24]}"
        if stored_evidence_id != calculated_id:
            raise ValueError("evidence ID disagrees with its content hash")
        calculated_storage_hash = _record_storage_hash(
            record_kind="evidence",
            scope=scope,
            record_id=stored_evidence_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
        )
        if storage_hash != calculated_storage_hash:
            raise ValueError("evidence storage hash disagrees with stored metadata")
        return record
    except MemoryPersistenceCorruptionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored evidence row failed integrity validation"
        ) from error


def _load_evidence_index(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
) -> dict[tuple[int, str], EvidenceRecord]:
    rows = cursor.execute(
        "SELECT scope_id, evidence_id FROM memory_evidence",
    ).fetchall()
    records_by_address: dict[tuple[int, str], EvidenceRecord] = {}
    for row in rows:
        if len(row) != 2:
            raise MemoryPersistenceCorruptionError(
                "evidence address row does not contain exactly two values"
            )
        stored_scope_id = _require_scope_id(
            row[0],
            "evidence address contains an invalid positive signed 64-bit scope ID",
        )
        evidence_id = row[1]
        if type(evidence_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "evidence address contains a non-text identifier"
            )
        address = (stored_scope_id, evidence_id)
        if address in records_by_address:
            raise MemoryPersistenceCorruptionError(
                "evidence address appears multiple times"
            )
        stored_scope = scope_by_id.get(stored_scope_id)
        if stored_scope is None:
            raise MemoryPersistenceCorruptionError(
                "evidence address refers to a missing scope"
            )
        record = _load_evidence(
            cursor,
            stored_scope,
            stored_scope_id,
            evidence_id,
        )
        if record is None:
            raise MemoryPersistenceCorruptionError(
                "evidence address refers to a missing row"
            )
        records_by_address[address] = record
    return records_by_address


def _load_scope_evidence(
    cursor: sqlite3.Cursor,
    scope: MemoryScope,
    scope_id: int | None,
) -> tuple[EvidenceRecord, ...]:
    if scope_id is not None:
        _require_scope_id(
            scope_id,
            "evidence snapshot received an invalid positive signed 64-bit scope ID",
        )
    scope_by_id = _load_scope_index(cursor)
    if scope_id is not None and scope_by_id.get(scope_id) != scope:
        raise MemoryPersistenceCorruptionError(
            "evidence snapshot scope does not match the stored scope index"
        )
    records_by_address = _load_evidence_index(cursor, scope_by_id)
    if scope_id is None:
        return ()
    return tuple(
        record
        for (stored_scope_id, _evidence_id), record in records_by_address.items()
        if stored_scope_id == scope_id
    )


def _evidence_sort_key(
    record: EvidenceRecord,
) -> tuple[str, str, int, datetime, str]:
    return (
        record.event.session_id,
        record.event.run_id,
        record.event.sequence_no,
        record.event.observed_at,
        record.evidence_id,
    )


def _evidence_snapshot_sort_key(
    record: EvidenceRecord,
) -> tuple[datetime, int, str]:
    return (
        record.event.observed_at,
        record.event.sequence_no,
        record.evidence_id,
    )


def _allowed_kinds_canonical_bytes(
    allowed_kinds: tuple[EvidenceKind, ...],
) -> bytes:
    return json.dumps(
        [kind.value for kind in allowed_kinds],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_ingest_order_index(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
    evidence_by_address: dict[tuple[int, str], EvidenceRecord],
) -> dict[tuple[int, str], int]:
    rows = cursor.execute(
        "SELECT ingest_order, scope_id, evidence_id, binding_hash "
        "FROM memory_evidence_ingest_orders ORDER BY ingest_order"
    ).fetchall()
    ingest_order_by_address: dict[tuple[int, str], int] = {}
    observed_orders: list[int] = []
    for row in rows:
        if (
            len(row) != 4
            or type(row[0]) is not int
            or not 0 <= row[0] <= _MAX_SCOPE_ID
            or type(row[1]) is not int
            or type(row[2]) is not str
            or type(row[3]) is not str
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence ingest-order row failed integrity validation"
            )
        ingest_order, scope_id, evidence_id, binding_hash = row
        scope = scope_by_id.get(scope_id)
        address = (scope_id, evidence_id)
        if (
            scope is None
            or address not in evidence_by_address
            or address in ingest_order_by_address
            or binding_hash
            != _evidence_ingest_binding_hash(
                scope=scope,
                evidence_id=evidence_id,
                ingest_order=ingest_order,
            )
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence ingest-order row failed integrity validation"
            )
        ingest_order_by_address[address] = ingest_order
        observed_orders.append(ingest_order)
    if (
        set(ingest_order_by_address) != set(evidence_by_address)
        or tuple(observed_orders) != tuple(range(len(observed_orders)))
    ):
        raise MemoryPersistenceCorruptionError(
            "evidence ingest-order mapping is incomplete or non-contiguous"
        )
    return ingest_order_by_address


def _snapshot_members_from_state(
    *,
    spec: EvidenceSnapshotSpec,
    scope_id: int,
    evidence_high_watermark: int,
    evidence_by_address: dict[tuple[int, str], EvidenceRecord],
    ingest_order_by_address: dict[tuple[int, str], int],
) -> tuple[EvidenceSnapshotMember, ...]:
    selected = tuple(
        record
        for address, record in evidence_by_address.items()
        if address[0] == scope_id
        and ingest_order_by_address[address] <= evidence_high_watermark
        and record.event.kind in spec.allowed_kinds
        and record.event.observed_at <= spec.cutoff
    )
    ordered = tuple(sorted(selected, key=_evidence_snapshot_sort_key))
    return tuple(
        EvidenceSnapshotMember(
            evidence_id=record.evidence_id,
            evidence_content_hash=record.content_hash,
            ingest_order=ingest_order_by_address[(scope_id, record.evidence_id)],
        )
        for record in ordered
    )


def _load_evidence_snapshot_addresses(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
) -> tuple[tuple[int, str], ...]:
    rows = cursor.execute(
        "SELECT scope_id, snapshot_id FROM memory_evidence_snapshots"
    ).fetchall()
    addresses: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if len(row) != 2:
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot address row does not contain exactly two values"
            )
        scope_id = _require_scope_id(
            row[0],
            "evidence snapshot address contains an invalid scope ID",
        )
        snapshot_id = row[1]
        if type(snapshot_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot address contains a non-text identifier"
            )
        address = (scope_id, snapshot_id)
        if scope_id not in scope_by_id or address in seen:
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot address failed integrity validation"
            )
        seen.add(address)
        addresses.append(address)
    return tuple(addresses)


def _load_evidence_snapshot(
    cursor: sqlite3.Cursor,
    *,
    scope: MemoryScope,
    scope_id: int,
    snapshot_id: str,
    evidence_by_address: dict[tuple[int, str], EvidenceRecord],
    ingest_order_by_address: dict[tuple[int, str], int],
) -> EvidenceSnapshot | None:
    row = cursor.execute(
        _EVIDENCE_SNAPSHOT_SELECT,
        (scope_id, snapshot_id),
    ).fetchone()
    if row is None:
        return None
    try:
        if len(row) != len(_EVIDENCE_SNAPSHOT_ROW_TYPES):
            raise ValueError("evidence snapshot row has the wrong field count")
        for index, (value, expected_type) in enumerate(
            zip(row, _EVIDENCE_SNAPSHOT_ROW_TYPES, strict=True)
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"evidence snapshot field {index} has the wrong storage class"
                )
        (
            stored_snapshot_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            allowed_kinds_canonical,
            cutoff_utc,
            evidence_high_watermark,
            ordering_policy,
            member_count,
        ) = row
        if stored_snapshot_id != snapshot_id:
            raise ValueError("loaded evidence snapshot ID differs from requested ID")
        if evidence_high_watermark > len(ingest_order_by_address) - 1:
            raise ValueError("evidence snapshot watermark exceeds current ingestion")
        decoded_kinds = json.loads(allowed_kinds_canonical.decode("utf-8"))
        if type(decoded_kinds) is not list or any(
            type(value) is not str for value in decoded_kinds
        ):
            raise ValueError("allowed evidence kinds are not a JSON string list")
        allowed_kinds = tuple(EvidenceKind(value) for value in decoded_kinds)
        spec = EvidenceSnapshotSpec(
            scope=scope,
            allowed_kinds=allowed_kinds,
            cutoff=datetime.fromisoformat(cutoff_utc),
        )
        if (
            _allowed_kinds_canonical_bytes(spec.allowed_kinds)
            != allowed_kinds_canonical
            or spec.cutoff.isoformat() != cutoff_utc
        ):
            raise ValueError("evidence snapshot projections are not canonical")
        members = _snapshot_members_from_state(
            spec=spec,
            scope_id=scope_id,
            evidence_high_watermark=evidence_high_watermark,
            evidence_by_address=evidence_by_address,
            ingest_order_by_address=ingest_order_by_address,
        )
        if member_count != len(members):
            raise ValueError("evidence snapshot member count is inconsistent")
        snapshot = EvidenceSnapshot(
            snapshot_id=stored_snapshot_id,
            spec=spec,
            evidence_high_watermark=evidence_high_watermark,
            ordering_policy=ordering_policy,
            members=members,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if snapshot.created_at.isoformat() != created_at_text:
            raise ValueError("snapshot created_at is not exact UTC isoformat text")
        if snapshot.canonical_bytes() != canonical:
            raise ValueError("canonical snapshot bytes disagree with projections")
        calculated_hash = sha256(canonical).hexdigest()
        if content_hash != calculated_hash:
            raise ValueError("snapshot content hash disagrees with canonical bytes")
        if stored_snapshot_id != f"esnap_{calculated_hash[:24]}":
            raise ValueError("snapshot ID disagrees with its content hash")
        calculated_storage_hash = _record_storage_hash(
            record_kind="evidence_snapshot",
            scope=scope,
            record_id=stored_snapshot_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
        )
        if storage_hash != calculated_storage_hash:
            raise ValueError("snapshot storage hash disagrees with stored metadata")
        return snapshot
    except MemoryPersistenceCorruptionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored evidence snapshot row failed integrity validation"
        ) from error


def _load_evidence_snapshot_aliases(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
    snapshot_by_address: dict[tuple[int, str], EvidenceSnapshot],
) -> dict[tuple[int, str], EvidenceSnapshot]:
    rows = cursor.execute(
        "SELECT scope_id, idempotency_key, snapshot_id, binding_hash "
        "FROM memory_evidence_snapshot_aliases"
    ).fetchall()
    snapshot_by_alias: dict[tuple[int, str], EvidenceSnapshot] = {}
    for row in rows:
        if (
            len(row) != 4
            or type(row[0]) is not int
            or any(type(value) is not str for value in row[1:])
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot alias row failed integrity validation"
            )
        scope_id, idempotency_key, snapshot_id, binding_hash = row
        scope = scope_by_id.get(scope_id)
        try:
            idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        except (TypeError, ValueError) as error:
            raise MemoryPersistenceCorruptionError(
                "stored evidence snapshot alias failed integrity validation"
            ) from error
        snapshot = snapshot_by_address.get((scope_id, snapshot_id))
        alias_address = (scope_id, idempotency_key)
        if (
            scope is None
            or snapshot is None
            or alias_address in snapshot_by_alias
            or binding_hash
            != _snapshot_binding_hash(
                scope=scope,
                idempotency_key=idempotency_key,
                snapshot_id=snapshot_id,
            )
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot alias failed integrity validation"
            )
        snapshot_by_alias[alias_address] = snapshot
    aliased_snapshot_addresses = {
        (scope_id, snapshot.snapshot_id)
        for (scope_id, _idempotency_key), snapshot in snapshot_by_alias.items()
    }
    if aliased_snapshot_addresses != set(snapshot_by_address):
        raise MemoryPersistenceCorruptionError(
            "evidence snapshot exists without an idempotency alias"
        )
    return snapshot_by_alias


def _load_evidence_snapshot_state(
    cursor: sqlite3.Cursor,
) -> tuple[
    dict[int, MemoryScope],
    dict[tuple[int, str], EvidenceRecord],
    dict[tuple[int, str], int],
    dict[tuple[int, str], EvidenceSnapshot],
    dict[tuple[int, str], EvidenceSnapshot],
]:
    scope_by_id = _load_scope_index(cursor)
    evidence_by_address = _load_evidence_index(cursor, scope_by_id)
    ingest_order_by_address = _load_ingest_order_index(
        cursor,
        scope_by_id,
        evidence_by_address,
    )
    snapshot_addresses = _load_evidence_snapshot_addresses(cursor, scope_by_id)
    snapshot_by_address: dict[tuple[int, str], EvidenceSnapshot] = {}
    for scope_id, snapshot_id in snapshot_addresses:
        snapshot = _load_evidence_snapshot(
            cursor,
            scope=scope_by_id[scope_id],
            scope_id=scope_id,
            snapshot_id=snapshot_id,
            evidence_by_address=evidence_by_address,
            ingest_order_by_address=ingest_order_by_address,
        )
        if snapshot is None:
            raise MemoryPersistenceCorruptionError(
                "evidence snapshot address refers to a missing row"
            )
        snapshot_by_address[(scope_id, snapshot_id)] = snapshot
    snapshot_by_alias = _load_evidence_snapshot_aliases(
        cursor,
        scope_by_id,
        snapshot_by_address,
    )
    return (
        scope_by_id,
        evidence_by_address,
        ingest_order_by_address,
        snapshot_by_address,
        snapshot_by_alias,
    )


def _find_scope_id_in_index(
    scope_by_id: dict[int, MemoryScope],
    scope: MemoryScope,
) -> int | None:
    for scope_id, stored_scope in scope_by_id.items():
        if stored_scope == scope:
            return scope_id
    return None


def _load_candidate_addresses(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
) -> tuple[tuple[int, str], ...]:
    rows = cursor.execute(
        "SELECT scope_id, candidate_id FROM memory_candidates"
    ).fetchall()
    addresses: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if len(row) != 2:
            raise MemoryPersistenceCorruptionError(
                "candidate address row does not contain exactly two values"
            )
        scope_id = _require_scope_id(
            row[0],
            "candidate address contains an invalid positive signed 64-bit scope ID",
        )
        candidate_id = row[1]
        if type(candidate_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "candidate address contains a non-text identifier"
            )
        if scope_id not in scope_by_id:
            raise MemoryPersistenceCorruptionError(
                "candidate address refers to a missing scope"
            )
        address = (scope_id, candidate_id)
        if address in seen:
            raise MemoryPersistenceCorruptionError(
                "candidate address appears multiple times"
            )
        seen.add(address)
        addresses.append(address)
    return tuple(addresses)


def _load_candidate_edges(
    cursor: sqlite3.Cursor,
    candidate_addresses: tuple[tuple[int, str], ...],
    evidence_by_address: dict[tuple[int, str], EvidenceRecord],
) -> dict[tuple[int, str], tuple[str, ...]]:
    candidate_address_set = set(candidate_addresses)
    rows = cursor.execute(
        "SELECT scope_id, candidate_id, position, evidence_id "
        "FROM memory_candidate_evidence"
    ).fetchall()
    edges_by_candidate: dict[tuple[int, str], list[tuple[int, str]]] = {
        address: [] for address in candidate_addresses
    }
    seen_positions: set[tuple[int, str, int]] = set()
    seen_evidence: set[tuple[int, str, str]] = set()
    for row in rows:
        if len(row) != 4:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence row does not contain exactly four values"
            )
        scope_id = _require_scope_id(
            row[0],
            "candidate evidence contains an invalid positive signed 64-bit scope ID",
        )
        candidate_id, position, evidence_id = row[1:]
        if type(candidate_id) is not str or type(evidence_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence identifiers must be text"
            )
        if type(position) is not int or not 0 <= position <= _MAX_SCOPE_ID:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence position is not a non-negative signed 64-bit integer"
            )
        candidate_address = (scope_id, candidate_id)
        if candidate_address not in candidate_address_set:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence refers to a missing candidate"
            )
        if (scope_id, evidence_id) not in evidence_by_address:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence refers to missing same-scope evidence"
            )
        position_address = (scope_id, candidate_id, position)
        if position_address in seen_positions:
            raise MemoryPersistenceCorruptionError(
                "candidate evidence position appears multiple times"
            )
        evidence_address = (scope_id, candidate_id, evidence_id)
        if evidence_address in seen_evidence:
            raise MemoryPersistenceCorruptionError(
                "candidate contains the same evidence multiple times"
            )
        seen_positions.add(position_address)
        seen_evidence.add(evidence_address)
        edges_by_candidate[candidate_address].append((position, evidence_id))

    ordered_evidence_by_candidate: dict[tuple[int, str], tuple[str, ...]] = {}
    for address, edges in edges_by_candidate.items():
        ordered = sorted(edges, key=lambda item: item[0])
        positions = tuple(position for position, _evidence_id in ordered)
        if positions != tuple(range(len(ordered))):
            raise MemoryPersistenceCorruptionError(
                "candidate evidence positions are not contiguous from zero"
            )
        ordered_evidence_by_candidate[address] = tuple(
            evidence_id for _position, evidence_id in ordered
        )
    return ordered_evidence_by_candidate


def _load_candidate(
    cursor: sqlite3.Cursor,
    scope: MemoryScope,
    scope_id: int,
    candidate_id: str,
    evidence_ids: tuple[str, ...],
) -> MemoryCandidate | None:
    _require_scope_id(
        scope_id,
        "candidate lookup received an invalid positive signed 64-bit scope ID",
    )
    row = cursor.execute(_CANDIDATE_SELECT, (scope_id, candidate_id)).fetchone()
    if row is None:
        return None
    try:
        if len(row) != len(_CANDIDATE_ROW_TYPES):
            raise ValueError("candidate row has the wrong field count")
        for index, (value, expected_type) in enumerate(
            zip(row, _CANDIDATE_ROW_TYPES, strict=True)
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"candidate row field {index} has the wrong storage class"
                )
        (
            stored_candidate_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            content,
            idempotency_key,
        ) = row
        if stored_candidate_id != candidate_id:
            raise ValueError("loaded candidate ID differs from requested ID")
        proposal = CandidateProposal(
            scope=scope,
            content=content,
            evidence_ids=evidence_ids,
            idempotency_key=idempotency_key,
        )
        candidate = MemoryCandidate(
            candidate_id=stored_candidate_id,
            proposal=proposal,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if candidate.created_at.isoformat() != created_at_text:
            raise ValueError("candidate created_at is not exact UTC isoformat text")
        if proposal.canonical_bytes() != canonical:
            raise ValueError("canonical candidate bytes disagree with projections")
        calculated_hash = sha256(canonical).hexdigest()
        if content_hash != calculated_hash:
            raise ValueError("candidate content hash disagrees with canonical bytes")
        calculated_id = f"cand_{calculated_hash[:24]}"
        if stored_candidate_id != calculated_id:
            raise ValueError("candidate ID disagrees with its content hash")
        calculated_storage_hash = _record_storage_hash(
            record_kind="candidate",
            scope=scope,
            record_id=stored_candidate_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
        )
        if storage_hash != calculated_storage_hash:
            raise ValueError("candidate storage hash disagrees with stored metadata")
        return candidate
    except MemoryPersistenceCorruptionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored candidate row failed integrity validation"
        ) from error


def _load_candidate_snapshot(
    cursor: sqlite3.Cursor,
) -> tuple[
    dict[int, MemoryScope],
    dict[tuple[int, str], EvidenceRecord],
    dict[tuple[int, str], MemoryCandidate],
    dict[tuple[int, str], MemoryCandidate],
]:
    scope_by_id = _load_scope_index(cursor)
    evidence_by_address = _load_evidence_index(cursor, scope_by_id)
    candidate_addresses = _load_candidate_addresses(cursor, scope_by_id)
    evidence_ids_by_candidate = _load_candidate_edges(
        cursor,
        candidate_addresses,
        evidence_by_address,
    )
    candidate_by_address: dict[tuple[int, str], MemoryCandidate] = {}
    candidate_by_idempotency: dict[tuple[int, str], MemoryCandidate] = {}
    for address in candidate_addresses:
        scope_id, candidate_id = address
        candidate = _load_candidate(
            cursor,
            scope_by_id[scope_id],
            scope_id,
            candidate_id,
            evidence_ids_by_candidate[address],
        )
        if candidate is None:
            raise MemoryPersistenceCorruptionError(
                "candidate address refers to a missing row"
            )
        candidate_by_address[address] = candidate
        idempotency_address = (scope_id, candidate.proposal.idempotency_key)
        if idempotency_address in candidate_by_idempotency:
            raise MemoryPersistenceCorruptionError(
                "candidate idempotency key appears multiple times in one scope"
            )
        candidate_by_idempotency[idempotency_address] = candidate
    return (
        scope_by_id,
        evidence_by_address,
        candidate_by_address,
        candidate_by_idempotency,
    )


def _load_revision_addresses(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
) -> tuple[tuple[int, str], ...]:
    rows = cursor.execute(
        "SELECT scope_id, revision_id FROM memory_revisions"
    ).fetchall()
    addresses: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if len(row) != 2:
            raise MemoryPersistenceCorruptionError(
                "revision address row does not contain exactly two values"
            )
        scope_id = _require_scope_id(
            row[0],
            "revision address contains an invalid positive signed 64-bit scope ID",
        )
        revision_id = row[1]
        if type(revision_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "revision address contains a non-text identifier"
            )
        if scope_id not in scope_by_id:
            raise MemoryPersistenceCorruptionError(
                "revision address refers to a missing scope"
            )
        address = (scope_id, revision_id)
        if address in seen:
            raise MemoryPersistenceCorruptionError(
                "revision address appears multiple times"
            )
        seen.add(address)
        addresses.append(address)
    return tuple(addresses)


def _load_revision(
    cursor: sqlite3.Cursor,
    scope: MemoryScope,
    scope_id: int,
    revision_id: str,
) -> MemoryRevision | None:
    _require_scope_id(
        scope_id,
        "revision lookup received an invalid positive signed 64-bit scope ID",
    )
    row = cursor.execute(_REVISION_SELECT, (scope_id, revision_id)).fetchone()
    if row is None:
        return None
    try:
        if len(row) != len(_REVISION_ROW_TYPES):
            raise ValueError("revision row has the wrong field count")
        for index, (value, expected_types) in enumerate(
            zip(row, _REVISION_ROW_TYPES, strict=True)
        ):
            if not any(
                type(value) is expected_type for expected_type in expected_types
            ):
                raise TypeError(
                    f"revision row field {index} has the wrong storage class"
                )
        (
            stored_revision_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            candidate_id,
            memory_id,
            generation,
            operation_text,
            parent_revision_id,
            idempotency_key,
        ) = row
        if stored_revision_id != revision_id:
            raise ValueError("loaded revision ID differs from requested ID")
        proposal = RevisionProposal(
            scope=scope,
            candidate_id=candidate_id,
            operation=RevisionOperation(operation_text),
            parent_revision_id=parent_revision_id,
            idempotency_key=idempotency_key,
        )
        revision = MemoryRevision(
            revision_id=stored_revision_id,
            memory_id=memory_id,
            generation=generation,
            proposal=proposal,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if revision.created_at.isoformat() != created_at_text:
            raise ValueError("revision created_at is not exact UTC isoformat text")
        if proposal.canonical_bytes() != canonical:
            raise ValueError("canonical revision bytes disagree with projections")
        calculated_hash = sha256(canonical).hexdigest()
        if content_hash != calculated_hash:
            raise ValueError("revision content hash disagrees with canonical bytes")
        calculated_id = f"rev_{calculated_hash[:24]}"
        if stored_revision_id != calculated_id:
            raise ValueError("revision ID disagrees with its content hash")
        calculated_storage_hash = _record_storage_hash(
            record_kind="revision",
            scope=scope,
            record_id=stored_revision_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
            memory_id=memory_id,
            generation=generation,
        )
        if storage_hash != calculated_storage_hash:
            raise ValueError("revision storage hash disagrees with stored metadata")
        return revision
    except MemoryPersistenceCorruptionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored revision row failed integrity validation"
        ) from error


def _validate_revision_topology(
    revision_by_address: dict[tuple[int, str], MemoryRevision],
    parent_by_address: dict[
        tuple[int, str],
        tuple[int, str] | None,
    ],
) -> None:
    validated: set[tuple[int, str]] = set()
    for start in revision_by_address:
        if start in validated:
            continue
        trail: list[tuple[int, str]] = []
        positions: dict[tuple[int, str], int] = {}
        current = start
        while current not in validated:
            if current in positions:
                raise MemoryPersistenceCorruptionError(
                    "revision parent graph contains a cycle"
                )
            positions[current] = len(trail)
            trail.append(current)
            parent = parent_by_address[current]
            if parent is None:
                break
            current = parent

        for address in reversed(trail):
            revision = revision_by_address[address]
            parent_address = parent_by_address[address]
            if parent_address is None:
                if revision.generation != 0:
                    raise MemoryPersistenceCorruptionError(
                        "ADD revision generation is not zero"
                    )
                if revision.memory_id != f"mem_{revision.content_hash[:24]}":
                    raise MemoryPersistenceCorruptionError(
                        "ADD revision memory ID disagrees with its content hash"
                    )
            else:
                parent = revision_by_address[parent_address]
                if parent.generation == _MAX_GENERATION:
                    raise MemoryPersistenceCorruptionError(
                        "child revision follows a parent at maximum generation"
                    )
                if revision.memory_id != parent.memory_id:
                    raise MemoryPersistenceCorruptionError(
                        "child revision memory ID differs from parent memory ID"
                    )
                if revision.generation != parent.generation + 1:
                    raise MemoryPersistenceCorruptionError(
                        "child revision generation is not exactly parent generation plus one"
                    )
            validated.add(address)


def _derive_revision_lineage(
    proposal: RevisionProposal,
    content_hash: str,
    parent: MemoryRevision | None,
) -> tuple[str, int]:
    if proposal.operation is RevisionOperation.ADD:
        if parent is not None:
            raise ValueError("ADD revision lineage must not have a parent")
        return f"mem_{content_hash[:24]}", 0
    if parent is None:
        raise ValueError("non-ADD revision lineage requires a parent")
    if parent.generation == _MAX_GENERATION:
        raise RevisionConflictError("revision generation exceeds the signed-64 range")
    return parent.memory_id, parent.generation + 1


def _load_revision_snapshot(
    cursor: sqlite3.Cursor,
) -> tuple[
    dict[int, MemoryScope],
    dict[tuple[int, str], MemoryCandidate],
    dict[tuple[int, str], MemoryRevision],
    dict[tuple[int, str], MemoryRevision],
    dict[tuple[int, str], MemoryRevision],
]:
    (
        scope_by_id,
        _evidence_by_address,
        candidate_by_address,
        _candidate_by_idempotency,
    ) = _load_candidate_snapshot(cursor)
    addresses = _load_revision_addresses(cursor, scope_by_id)
    revision_by_address: dict[tuple[int, str], MemoryRevision] = {}
    for address in addresses:
        scope_id, revision_id = address
        revision = _load_revision(
            cursor,
            scope_by_id[scope_id],
            scope_id,
            revision_id,
        )
        if revision is None:
            raise MemoryPersistenceCorruptionError(
                "revision address refers to a missing row"
            )
        revision_by_address[address] = revision

    revision_by_idempotency: dict[tuple[int, str], MemoryRevision] = {}
    revision_by_candidate: dict[tuple[int, str], MemoryRevision] = {}
    parent_by_address: dict[
        tuple[int, str],
        tuple[int, str] | None,
    ] = {}
    for address in addresses:
        scope_id, _revision_id = address
        revision = revision_by_address[address]
        candidate_address = (scope_id, revision.proposal.candidate_id)
        if candidate_address not in candidate_by_address:
            raise MemoryPersistenceCorruptionError(
                "revision refers to a missing same-scope candidate"
            )
        idempotency_address = (scope_id, revision.proposal.idempotency_key)
        if idempotency_address in revision_by_idempotency:
            raise MemoryPersistenceCorruptionError(
                "revision idempotency key appears multiple times in one scope"
            )
        if candidate_address in revision_by_candidate:
            raise MemoryPersistenceCorruptionError(
                "candidate backs multiple revisions in one scope"
            )
        revision_by_idempotency[idempotency_address] = revision
        revision_by_candidate[candidate_address] = revision

        parent_revision_id = revision.proposal.parent_revision_id
        parent_address = (
            None if parent_revision_id is None else (scope_id, parent_revision_id)
        )
        if parent_address is not None and parent_address not in revision_by_address:
            raise MemoryPersistenceCorruptionError(
                "revision refers to a missing same-scope parent"
            )
        parent_by_address[address] = parent_address

    _validate_revision_topology(revision_by_address, parent_by_address)
    return (
        scope_by_id,
        candidate_by_address,
        revision_by_address,
        revision_by_idempotency,
        revision_by_candidate,
    )


def _load_release_addresses(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
) -> tuple[tuple[int, str], ...]:
    rows = cursor.execute("SELECT scope_id, release_id FROM memory_releases").fetchall()
    addresses: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if len(row) != 2:
            raise MemoryPersistenceCorruptionError(
                "release address row does not contain exactly two values"
            )
        scope_id = _require_scope_id(
            row[0],
            "release address contains an invalid positive signed 64-bit scope ID",
        )
        release_id = row[1]
        if type(release_id) is not str:
            raise MemoryPersistenceCorruptionError(
                "release address contains a non-text identifier"
            )
        if scope_id not in scope_by_id:
            raise MemoryPersistenceCorruptionError(
                "release address refers to a missing scope"
            )
        address = (scope_id, release_id)
        if address in seen:
            raise MemoryPersistenceCorruptionError(
                "release address appears multiple times"
            )
        seen.add(address)
        addresses.append(address)
    return tuple(addresses)


def _load_release_members(
    cursor: sqlite3.Cursor,
    release_addresses: tuple[tuple[int, str], ...],
    revision_by_address: dict[tuple[int, str], MemoryRevision],
) -> dict[tuple[int, str], tuple[MemoryRevision, ...]]:
    release_address_set = set(release_addresses)
    rows = cursor.execute(
        "SELECT scope_id, release_id, position, revision_id, memory_id "
        "FROM memory_release_revisions"
    ).fetchall()
    edges_by_release: dict[tuple[int, str], list[tuple[int, MemoryRevision]]] = {
        address: [] for address in release_addresses
    }
    seen_positions: set[tuple[int, str, int]] = set()
    seen_revisions: set[tuple[int, str, str]] = set()
    seen_memories: set[tuple[int, str, str]] = set()
    for row in rows:
        if len(row) != 5:
            raise MemoryPersistenceCorruptionError(
                "release member row does not contain exactly five values"
            )
        scope_id = _require_scope_id(
            row[0],
            "release member contains an invalid positive signed 64-bit scope ID",
        )
        release_id, position, revision_id, memory_id = row[1:]
        if not all(
            type(value) is str for value in (release_id, revision_id, memory_id)
        ):
            raise MemoryPersistenceCorruptionError(
                "release member identifiers must be text"
            )
        if type(position) is not int or not 0 <= position <= _MAX_SCOPE_ID:
            raise MemoryPersistenceCorruptionError(
                "release member position is not a non-negative signed 64-bit integer"
            )
        release_address = (scope_id, release_id)
        if release_address not in release_address_set:
            raise MemoryPersistenceCorruptionError(
                "release member refers to a missing release"
            )
        revision = revision_by_address.get((scope_id, revision_id))
        if revision is None:
            raise MemoryPersistenceCorruptionError(
                "release member refers to a missing same-scope revision"
            )
        if revision.memory_id != memory_id:
            raise MemoryPersistenceCorruptionError(
                "release member memory ID differs from its revision"
            )
        position_address = (scope_id, release_id, position)
        revision_address = (scope_id, release_id, revision_id)
        memory_address = (scope_id, release_id, memory_id)
        if position_address in seen_positions:
            raise MemoryPersistenceCorruptionError(
                "release member position appears multiple times"
            )
        if revision_address in seen_revisions:
            raise MemoryPersistenceCorruptionError(
                "release contains the same revision multiple times"
            )
        if memory_address in seen_memories:
            raise MemoryPersistenceCorruptionError(
                "release contains the same memory multiple times"
            )
        seen_positions.add(position_address)
        seen_revisions.add(revision_address)
        seen_memories.add(memory_address)
        edges_by_release[release_address].append((position, revision))
    revisions_by_release: dict[tuple[int, str], tuple[MemoryRevision, ...]] = {}
    for address, edges in edges_by_release.items():
        ordered = sorted(edges, key=lambda item: item[0])
        positions = tuple(position for position, _revision in ordered)
        if positions != tuple(range(len(ordered))):
            raise MemoryPersistenceCorruptionError(
                "release member positions are not contiguous from zero"
            )
        revisions_by_release[address] = tuple(
            revision for _position, revision in ordered
        )
    return revisions_by_release


def _load_release(
    cursor: sqlite3.Cursor,
    scope: MemoryScope,
    scope_id: int,
    release_id: str,
    revisions: tuple[MemoryRevision, ...],
) -> MemoryRelease | None:
    _require_scope_id(
        scope_id,
        "release lookup received an invalid positive signed 64-bit scope ID",
    )
    row = cursor.execute(_RELEASE_SELECT, (scope_id, release_id)).fetchone()
    if row is None:
        return None
    try:
        if len(row) != len(_RELEASE_ROW_TYPES):
            raise ValueError("release row has the wrong field count")
        for index, (value, expected_type) in enumerate(
            zip(row, _RELEASE_ROW_TYPES, strict=True)
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"release row field {index} has the wrong storage class"
                )
        (
            stored_release_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
        ) = row
        if stored_release_id != release_id:
            raise ValueError("loaded release ID differs from requested ID")
        manifest = ReleaseManifest(
            scope=scope,
            revision_ids=tuple(revision.revision_id for revision in revisions),
        )
        release = MemoryRelease(
            release_id=stored_release_id,
            manifest=manifest,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if release.created_at.isoformat() != created_at_text:
            raise ValueError("release created_at is not exact UTC isoformat text")
        if manifest.canonical_bytes() != canonical:
            raise ValueError("canonical release bytes disagree with projections")
        calculated_hash = sha256(canonical).hexdigest()
        if content_hash != calculated_hash:
            raise ValueError("release content hash disagrees with canonical bytes")
        calculated_id = f"rel_{calculated_hash[:24]}"
        if stored_release_id != calculated_id:
            raise ValueError("release ID disagrees with its content hash")
        calculated_storage_hash = _record_storage_hash(
            record_kind="release",
            scope=scope,
            record_id=stored_release_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
        )
        if storage_hash != calculated_storage_hash:
            raise ValueError("release storage hash disagrees with stored metadata")
        return release
    except MemoryPersistenceCorruptionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored release row failed integrity validation"
        ) from error


def _load_release_aliases(
    cursor: sqlite3.Cursor,
    scope_by_id: dict[int, MemoryScope],
    release_by_address: dict[tuple[int, str], MemoryRelease],
) -> dict[tuple[int, str], MemoryRelease]:
    rows = cursor.execute(
        "SELECT scope_id, idempotency_key, release_id, binding_hash "
        "FROM memory_release_aliases"
    ).fetchall()
    release_by_alias: dict[tuple[int, str], MemoryRelease] = {}
    for row in rows:
        if len(row) != 4:
            raise MemoryPersistenceCorruptionError(
                "release alias row does not contain exactly four values"
            )
        scope_id = _require_scope_id(
            row[0],
            "release alias contains an invalid positive signed 64-bit scope ID",
        )
        idempotency_key, release_id, binding_hash = row[1:]
        if not all(
            type(value) is str for value in (idempotency_key, release_id, binding_hash)
        ):
            raise MemoryPersistenceCorruptionError("release alias values must be text")
        scope = scope_by_id.get(scope_id)
        if scope is None:
            raise MemoryPersistenceCorruptionError(
                "release alias refers to a missing scope"
            )
        try:
            idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        except (TypeError, ValueError) as error:
            raise MemoryPersistenceCorruptionError(
                "stored release alias failed integrity validation"
            ) from error
        release = release_by_address.get((scope_id, release_id))
        if release is None:
            raise MemoryPersistenceCorruptionError(
                "release alias refers to a missing same-scope release"
            )
        alias_address = (scope_id, idempotency_key)
        if alias_address in release_by_alias:
            raise MemoryPersistenceCorruptionError(
                "release idempotency key appears multiple times in one scope"
            )
        expected_binding_hash = _release_binding_hash(
            scope=scope,
            idempotency_key=idempotency_key,
            release_id=release_id,
        )
        if binding_hash != expected_binding_hash:
            raise MemoryPersistenceCorruptionError(
                "release alias binding hash disagrees with stored metadata"
            )
        release_by_alias[alias_address] = release
    return release_by_alias


def _load_release_snapshot(
    cursor: sqlite3.Cursor,
) -> tuple[
    dict[int, MemoryScope],
    dict[tuple[int, str], MemoryRevision],
    dict[tuple[int, str], MemoryRelease],
    dict[tuple[int, str], MemoryRelease],
    dict[tuple[int, str], tuple[MemoryRevision, ...]],
]:
    (
        scope_by_id,
        _candidate_by_address,
        revision_by_address,
        _revision_by_idempotency,
        _revision_by_candidate,
    ) = _load_revision_snapshot(cursor)
    release_addresses = _load_release_addresses(cursor, scope_by_id)
    revisions_by_release = _load_release_members(
        cursor,
        release_addresses,
        revision_by_address,
    )
    release_by_address: dict[tuple[int, str], MemoryRelease] = {}
    for address in release_addresses:
        scope_id, release_id = address
        release = _load_release(
            cursor,
            scope_by_id[scope_id],
            scope_id,
            release_id,
            revisions_by_release[address],
        )
        if release is None:
            raise MemoryPersistenceCorruptionError(
                "release address refers to a missing row"
            )
        release_by_address[address] = release
    release_by_alias = _load_release_aliases(
        cursor,
        scope_by_id,
        release_by_address,
    )
    aliased_release_addresses = {
        (scope_id, release.release_id)
        for (scope_id, _idempotency_key), release in release_by_alias.items()
    }
    if aliased_release_addresses != set(release_by_address):
        raise MemoryPersistenceCorruptionError(
            "release exists without an idempotency alias"
        )
    return (
        scope_by_id,
        revision_by_address,
        release_by_address,
        release_by_alias,
        revisions_by_release,
    )


def _load_application_roots(
    cursor: sqlite3.Cursor,
    *,
    scope_by_id: dict[int, MemoryScope],
    release_by_address: dict[tuple[int, str], MemoryRelease],
) -> tuple[
    dict[int, MemoryApplicationRootV1],
    dict[tuple[int, str], MemoryApplicationRootV1],
]:
    rows = cursor.execute(
        "SELECT scope_id FROM memory_application_roots ORDER BY scope_id"
    ).fetchall()
    by_scope_id: dict[int, MemoryApplicationRootV1] = {}
    by_release: dict[tuple[int, str], MemoryApplicationRootV1] = {}
    for address_row in rows:
        if len(address_row) != 1:
            raise MemoryPersistenceCorruptionError(
                "application root address has the wrong field count"
            )
        scope_id = _require_scope_id(
            address_row[0],
            "application root has an invalid scope ID",
        )
        scope = scope_by_id.get(scope_id)
        row = cursor.execute(_APPLICATION_ROOT_SELECT, (scope_id,)).fetchone()
        if (
            scope is None
            or row is None
            or len(row) != 7
            or not all(
                type(value) is expected
                for value, expected in zip(
                    row,
                    (str, bytes, str, str, str, str, str),
                    strict=True,
                )
            )
        ):
            raise MemoryPersistenceCorruptionError(
                "stored application root has invalid fields"
            )
        (
            root_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            release_id,
            release_content_hash,
        ) = row
        release = release_by_address.get((scope_id, release_id))
        try:
            root = MemoryApplicationRootV1(
                scope=scope,
                release_id=release_id,
                release_content_sha256=release_content_hash,
                root_id=root_id,
                content_hash=content_hash,
                created_at=datetime.fromisoformat(created_at_text),
            )
            if (
                root.created_at.isoformat() != created_at_text
                or root.canonical_bytes() != canonical
                or release is None
                or release.content_hash != release_content_hash
                or storage_hash
                != _record_storage_hash(
                    record_kind="memory_application_root",
                    scope=scope,
                    record_id=root_id,
                    content_hash=content_hash,
                    created_at_text=created_at_text,
                )
            ):
                raise ValueError("application root commitment mismatch")
        except (TypeError, ValueError, OverflowError) as error:
            raise MemoryPersistenceCorruptionError(
                "stored application root failed integrity validation"
            ) from error
        release_address = (scope_id, release_id)
        if scope_id in by_scope_id or release_address in by_release:
            raise MemoryPersistenceCorruptionError(
                "application root is not unique in its scope"
            )
        by_scope_id[scope_id] = root
        by_release[release_address] = root
    return by_scope_id, by_release


def _load_application_edges(
    cursor: sqlite3.Cursor,
    *,
    scope_by_id: dict[int, MemoryScope],
    revision_by_address: dict[tuple[int, str], MemoryRevision],
) -> dict[tuple[int, str], tuple[MemoryRevision, ...]]:
    rows = cursor.execute(
        "SELECT scope_id, application_id, ordinal, revision_id, "
        "revision_content_hash, binding_hash "
        "FROM memory_application_revisions "
        "ORDER BY scope_id, application_id COLLATE BINARY, ordinal"
    ).fetchall()
    grouped: dict[tuple[int, str], list[MemoryRevision]] = {}
    seen_revisions: set[tuple[int, str]] = set()
    for row in rows:
        if (
            len(row) != 6
            or type(row[0]) is not int
            or type(row[1]) is not str
            or type(row[2]) is not int
            or any(type(value) is not str for value in row[3:])
        ):
            raise MemoryPersistenceCorruptionError(
                "application revision edge has invalid storage classes"
            )
        scope_id = _require_scope_id(
            row[0],
            "application revision edge has an invalid scope ID",
        )
        application_id, ordinal, revision_id, revision_hash, binding_hash = row[1:]
        scope = scope_by_id.get(scope_id)
        revision = revision_by_address.get((scope_id, revision_id))
        address = (scope_id, application_id)
        items = grouped.setdefault(address, [])
        revision_address = (scope_id, revision_id)
        if (
            scope is None
            or type(ordinal) is not int
            or ordinal != len(items)
            or revision is None
            or revision.content_hash != revision_hash
            or revision_address in seen_revisions
            or binding_hash
            != _application_revision_binding_hash(
                scope=scope,
                application_id=application_id,
                ordinal=ordinal,
                revision_id=revision_id,
                revision_content_hash=revision_hash,
            )
        ):
            raise MemoryPersistenceCorruptionError(
                "application revision edge failed integrity validation"
            )
        seen_revisions.add(revision_address)
        items.append(revision)
    return {address: tuple(items) for address, items in grouped.items()}


def _application_child_idempotency_key(
    proposal: MemoryApplicationProposal,
    *,
    record_kind: str,
    ordinal: int | None = None,
) -> str:
    if record_kind not in {"candidate", "revision", "release"}:
        raise ValueError("unsupported application child record kind")
    if record_kind == "release":
        if ordinal is not None:
            raise ValueError("release application key must not have an ordinal")
    elif type(ordinal) is not int or ordinal < 0:
        raise ValueError("application child ordinal must be non-negative")
    payload = json.dumps(
        {
            "ordinal": ordinal,
            "proposal_sha256": sha256(proposal.canonical_bytes()).hexdigest(),
            "record_kind": record_kind,
            "schema_version": 1,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return (
        f"memory-application-{record_kind}-"
        + sha256(_APPLICATION_CHILD_KEY_DOMAIN + payload).hexdigest()
    )


def _validate_application_release_transform(
    application: MemoryApplicationV1,
    *,
    base_revisions: tuple[MemoryRevision, ...],
    result_revisions: tuple[MemoryRevision, ...],
) -> None:
    base_ids = tuple(item.revision_id for item in base_revisions)
    expected = list(base_ids)
    base_position = {revision_id: index for index, revision_id in enumerate(base_ids)}
    add_count = 0
    for update in application.applied_updates:
        if update.operation is RevisionOperation.SUPERSEDE:
            assert update.parent_revision_id is not None
            position = base_position.get(update.parent_revision_id)
            if position is None or update.release_position != position:
                raise ValueError("SUPERSEDE does not replace an exact base tip")
            expected[position] = update.revision_id
        elif update.operation is RevisionOperation.ADD:
            expected_position = len(base_ids) + add_count
            if update.release_position != expected_position:
                raise ValueError("ADD does not append in application order")
            expected.append(update.revision_id)
            add_count += 1
        else:
            raise ValueError("application contains an unsupported operation")
    result_ids = tuple(item.revision_id for item in result_revisions)
    if tuple(expected) != result_ids or application.result_revision_ids != result_ids:
        raise ValueError("result release is not the exact declared base transform")


def _load_application(
    cursor: sqlite3.Cursor,
    *,
    scope_id: int,
    application_id: str,
    scope_by_id: dict[int, MemoryScope],
    candidate_by_address: dict[tuple[int, str], MemoryCandidate],
    revision_by_address: dict[tuple[int, str], MemoryRevision],
    release_by_address: dict[tuple[int, str], MemoryRelease],
    release_by_alias: dict[tuple[int, str], MemoryRelease],
    revisions_by_release: dict[tuple[int, str], tuple[MemoryRevision, ...]],
    snapshot_by_address: dict[tuple[int, str], EvidenceSnapshot],
    application_edges: dict[tuple[int, str], tuple[MemoryRevision, ...]],
) -> MemoryApplicationV1:
    row = cursor.execute(_APPLICATION_SELECT, (scope_id, application_id)).fetchone()
    if row is None:
        raise MemoryApplicationNotFoundError(
            f"application {application_id!r} was not found"
        )
    try:
        if len(row) != len(_APPLICATION_ROW_TYPES) or any(
            type(value) is not expected
            for value, expected in zip(row, _APPLICATION_ROW_TYPES, strict=True)
        ):
            raise TypeError("application row has invalid storage classes")
        (
            application_order,
            stored_application_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            source_snapshot_id,
            source_snapshot_hash,
            source_high_watermark,
            base_release_id,
            base_release_hash,
            result_release_id,
            result_release_hash,
            projector_id,
            projector_version_hash,
            policy_id,
            policy_version_hash,
            policy_input_hash,
            decision_hash,
            policy_context,
            update_count,
            idempotency_key,
        ) = row
        if stored_application_id != application_id:
            raise ValueError("application ID differs from requested ID")
        scope = scope_by_id[scope_id]
        snapshot = snapshot_by_address[(scope_id, source_snapshot_id)]
        base_release = release_by_address[(scope_id, base_release_id)]
        result_release = release_by_address[(scope_id, result_release_id)]
        base_revisions = revisions_by_release[(scope_id, base_release_id)]
        result_revisions = revisions_by_release[(scope_id, result_release_id)]
        edge_revisions = application_edges[(scope_id, application_id)]
        if len(edge_revisions) != update_count:
            raise ValueError("application update count disagrees with edges")

        member_by_id = {member.evidence_id: member for member in snapshot.members}
        update_proposals: list[MemoryApplicationUpdateProposal] = []
        applied_updates: list[AppliedMemoryUpdateV1] = []
        result_position = {
            revision.revision_id: position
            for position, revision in enumerate(result_revisions)
        }
        for ordinal, revision in enumerate(edge_revisions):
            candidate = candidate_by_address[(scope_id, revision.proposal.candidate_id)]
            grounding = tuple(
                member_by_id[evidence_id]
                for evidence_id in candidate.proposal.evidence_ids
            )
            update_proposals.append(
                MemoryApplicationUpdateProposal(
                    content=candidate.proposal.content,
                    evidence_ids=candidate.proposal.evidence_ids,
                    operation=revision.proposal.operation,
                    parent_revision_id=revision.proposal.parent_revision_id,
                )
            )
            applied_updates.append(
                AppliedMemoryUpdateV1(
                    ordinal=ordinal,
                    release_position=result_position[revision.revision_id],
                    operation=revision.proposal.operation,
                    grounding=grounding,
                    candidate_id=candidate.candidate_id,
                    candidate_content_sha256=candidate.content_hash,
                    revision_id=revision.revision_id,
                    revision_content_sha256=revision.content_hash,
                    memory_id=revision.memory_id,
                    generation=revision.generation,
                    parent_revision_id=revision.proposal.parent_revision_id,
                )
            )
        proposal = MemoryApplicationProposal(
            scope=scope,
            source_snapshot_id=source_snapshot_id,
            source_base_release_id=base_release_id,
            projector_id=projector_id,
            projector_version_sha256=projector_version_hash,
            policy_id=policy_id,
            policy_version_sha256=policy_version_hash,
            policy_input_sha256=policy_input_hash,
            decision_sha256=decision_hash,
            policy_context=policy_context,
            updates=tuple(update_proposals),
            idempotency_key=idempotency_key,
        )
        for ordinal, (revision, update) in enumerate(
            zip(edge_revisions, applied_updates, strict=True)
        ):
            candidate = candidate_by_address[(scope_id, update.candidate_id)]
            if (
                candidate.proposal.idempotency_key
                != _application_child_idempotency_key(
                    proposal,
                    record_kind="candidate",
                    ordinal=ordinal,
                )
                or revision.proposal.idempotency_key
                != _application_child_idempotency_key(
                    proposal,
                    record_kind="revision",
                    ordinal=ordinal,
                )
            ):
                raise ValueError("application child idempotency binding mismatch")
        release_alias = release_by_alias.get(
            (
                scope_id,
                _application_child_idempotency_key(
                    proposal,
                    record_kind="release",
                ),
            )
        )
        if release_alias != result_release:
            raise ValueError("application result release alias is missing")
        application = MemoryApplicationV1(
            proposal=proposal,
            source_snapshot_content_sha256=source_snapshot_hash,
            source_evidence_high_watermark=source_high_watermark,
            base_release_content_sha256=base_release_hash,
            result_release_id=result_release_id,
            result_release_content_sha256=result_release_hash,
            result_revision_ids=result_release.manifest.revision_ids,
            applied_updates=tuple(applied_updates),
            application_order=application_order,
            application_id=stored_application_id,
            content_hash=content_hash,
            created_at=datetime.fromisoformat(created_at_text),
        )
        if (
            application.created_at.isoformat() != created_at_text
            or application.canonical_bytes() != canonical
            or snapshot.content_hash != source_snapshot_hash
            or snapshot.evidence_high_watermark != source_high_watermark
            or base_release.content_hash != base_release_hash
            or result_release.content_hash != result_release_hash
            or base_release_id == result_release_id
            or storage_hash
            != _record_storage_hash(
                record_kind="memory_application",
                scope=scope,
                record_id=application_id,
                content_hash=content_hash,
                created_at_text=created_at_text,
            )
        ):
            raise ValueError("application commitment mismatch")
        _validate_application_release_transform(
            application,
            base_revisions=base_revisions,
            result_revisions=result_revisions,
        )
        return application
    except MemoryApplicationNotFoundError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise MemoryPersistenceCorruptionError(
            "stored memory application failed integrity validation"
        ) from error


def _load_application_state(cursor: sqlite3.Cursor) -> _ApplicationState:
    (
        scope_by_id,
        evidence_by_address,
        ingest_order_by_address,
        snapshot_by_address,
        _snapshot_by_alias,
    ) = _load_evidence_snapshot_state(cursor)
    (
        candidate_scope_by_id,
        candidate_evidence_by_address,
        candidate_by_address,
        _candidate_by_idempotency,
    ) = _load_candidate_snapshot(cursor)
    (
        release_scope_by_id,
        revision_by_address,
        release_by_address,
        release_by_alias,
        revisions_by_release,
    ) = _load_release_snapshot(cursor)
    if (
        candidate_scope_by_id != scope_by_id
        or release_scope_by_id != scope_by_id
        or candidate_evidence_by_address != evidence_by_address
    ):
        raise MemoryPersistenceCorruptionError(
            "application dependencies disagree on durable scope or evidence state"
        )
    root_by_scope_id, root_by_release = _load_application_roots(
        cursor,
        scope_by_id=scope_by_id,
        release_by_address=release_by_address,
    )
    application_edges = _load_application_edges(
        cursor,
        scope_by_id=scope_by_id,
        revision_by_address=revision_by_address,
    )
    address_rows = cursor.execute(
        "SELECT application_order, scope_id, application_id "
        "FROM memory_applications ORDER BY application_order"
    ).fetchall()
    if any(
        len(row) != 3
        or type(row[0]) is not int
        or type(row[1]) is not int
        or type(row[2]) is not str
        for row in address_rows
    ):
        raise MemoryPersistenceCorruptionError(
            "application address rows have invalid storage classes"
        )
    if tuple(row[0] for row in address_rows) != tuple(range(len(address_rows))):
        raise MemoryPersistenceCorruptionError(
            "application_order is not globally contiguous"
        )
    application_addresses = {
        (_require_scope_id(row[1], "application has an invalid scope ID"), row[2])
        for row in address_rows
    }
    if len(application_addresses) != len(address_rows) or set(application_edges) != (
        application_addresses
    ):
        raise MemoryPersistenceCorruptionError(
            "application rows and revision edges are not one-to-one"
        )

    application_by_address: dict[tuple[int, str], MemoryApplicationV1] = {}
    application_by_idempotency: dict[tuple[int, str], MemoryApplicationV1] = {}
    application_by_result_release: dict[tuple[int, str], MemoryApplicationV1] = {}
    application_by_revision: dict[tuple[int, str], MemoryApplicationV1] = {}
    head_release_by_scope = {
        scope_id: root.release_id for scope_id, root in root_by_scope_id.items()
    }
    for expected_order, (_stored_order, scope_id, application_id) in enumerate(
        address_rows
    ):
        if scope_id not in scope_by_id:
            raise MemoryPersistenceCorruptionError(
                "application refers to a missing scope"
            )
        application = _load_application(
            cursor,
            scope_id=scope_id,
            application_id=application_id,
            scope_by_id=scope_by_id,
            candidate_by_address=candidate_by_address,
            revision_by_address=revision_by_address,
            release_by_address=release_by_address,
            release_by_alias=release_by_alias,
            revisions_by_release=revisions_by_release,
            snapshot_by_address=snapshot_by_address,
            application_edges=application_edges,
        )
        if application.application_order != expected_order:
            raise MemoryPersistenceCorruptionError(
                "application row order disagrees with canonical order"
            )
        address = (scope_id, application.application_id)
        idempotency_address = (scope_id, application.proposal.idempotency_key)
        result_address = (scope_id, application.result_release_id)
        if (
            address in application_by_address
            or idempotency_address in application_by_idempotency
            or result_address in application_by_result_release
            or result_address in root_by_release
        ):
            raise MemoryPersistenceCorruptionError(
                "application identity or result release is not unique"
            )
        source_base_address = (
            scope_id,
            application.proposal.source_base_release_id,
        )
        predecessor = application_by_result_release.get(source_base_address)
        if source_base_address not in root_by_release and (
            predecessor is None
            or predecessor.application_order >= application.application_order
        ):
            raise MemoryPersistenceCorruptionError(
                "application source release is not a prior committed lineage node"
            )
        if head_release_by_scope.get(scope_id) != (
            application.proposal.source_base_release_id
        ):
            raise MemoryPersistenceCorruptionError(
                "application does not extend the scope's canonical head"
            )
        application_by_address[address] = application
        application_by_idempotency[idempotency_address] = application
        application_by_result_release[result_address] = application
        head_release_by_scope[scope_id] = application.result_release_id
        for update in application.applied_updates:
            revision_address = (scope_id, update.revision_id)
            if revision_address in application_by_revision:
                raise MemoryPersistenceCorruptionError(
                    "revision is attributed to multiple applications"
                )
            application_by_revision[revision_address] = application

    return _ApplicationState(
        scope_by_id=scope_by_id,
        evidence_by_address=evidence_by_address,
        ingest_order_by_address=ingest_order_by_address,
        snapshot_by_address=snapshot_by_address,
        candidate_by_address=candidate_by_address,
        revision_by_address=revision_by_address,
        release_by_address=release_by_address,
        release_by_alias=release_by_alias,
        revisions_by_release=revisions_by_release,
        root_by_scope_id=root_by_scope_id,
        root_by_release=root_by_release,
        application_by_address=application_by_address,
        application_by_idempotency=application_by_idempotency,
        application_by_result_release=application_by_result_release,
        application_by_revision=application_by_revision,
    )


def _append_candidate_locked(
    cursor: sqlite3.Cursor,
    proposal: CandidateProposal,
) -> MemoryCandidate:
    """Append one candidate using the caller's active writer transaction."""

    canonical = proposal.canonical_bytes()
    content_hash = sha256(canonical).hexdigest()
    candidate_id = f"cand_{content_hash[:24]}"
    (
        scope_by_id,
        evidence_by_address,
        candidate_by_address,
        candidate_by_idempotency,
    ) = _load_candidate_snapshot(cursor)
    scope_id = _find_scope_id_in_index(scope_by_id, proposal.scope)
    existing = (
        None
        if scope_id is None
        else candidate_by_idempotency.get((scope_id, proposal.idempotency_key))
    )
    if existing is not None:
        if existing.proposal.canonical_bytes() == canonical:
            return existing
        raise CandidateConflictError(
            "scoped candidate idempotency key already refers to different content"
        )

    for evidence_id in proposal.evidence_ids:
        if scope_id is None or (scope_id, evidence_id) not in evidence_by_address:
            raise EvidenceNotFoundError(f"evidence {evidence_id!r} was not found")
    assert scope_id is not None

    existing = candidate_by_address.get((scope_id, candidate_id))
    if existing is not None:
        if existing.proposal.canonical_bytes() == canonical:
            return existing
        raise CandidateConflictError(f"candidate ID collision for {candidate_id!r}")

    created_at = datetime.now(UTC)
    created_at_text = created_at.isoformat()
    storage_hash = _record_storage_hash(
        record_kind="candidate",
        scope=proposal.scope,
        record_id=candidate_id,
        content_hash=content_hash,
        created_at_text=created_at_text,
    )
    cursor.execute(
        """INSERT INTO memory_candidates (
    scope_id, candidate_id, canonical, content_hash, created_at,
    storage_hash, content, idempotency_key
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scope_id,
            candidate_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            proposal.content,
            proposal.idempotency_key,
        ),
    )
    for position, evidence_id in enumerate(proposal.evidence_ids):
        cursor.execute(
            """INSERT INTO memory_candidate_evidence (
    scope_id, candidate_id, position, evidence_id
) VALUES (?, ?, ?, ?)""",
            (scope_id, candidate_id, position, evidence_id),
        )

    (
        _scope_by_id,
        _evidence_by_address,
        inserted_candidates,
        _candidate_by_idempotency,
    ) = _load_candidate_snapshot(cursor)
    inserted = inserted_candidates.get((scope_id, candidate_id))
    if inserted is None:
        raise MemoryPersistenceCorruptionError(
            "inserted candidate graph could not be reloaded"
        )
    return inserted


def _append_revision_locked(
    cursor: sqlite3.Cursor,
    proposal: RevisionProposal,
) -> MemoryRevision:
    """Append one revision using the caller's active writer transaction."""

    canonical = proposal.canonical_bytes()
    content_hash = sha256(canonical).hexdigest()
    revision_id = f"rev_{content_hash[:24]}"
    (
        scope_by_id,
        candidate_by_address,
        revision_by_address,
        revision_by_idempotency,
        revision_by_candidate,
    ) = _load_revision_snapshot(cursor)
    scope_id = _find_scope_id_in_index(scope_by_id, proposal.scope)
    existing = (
        None
        if scope_id is None
        else revision_by_idempotency.get((scope_id, proposal.idempotency_key))
    )
    if existing is not None:
        if existing.proposal.canonical_bytes() == canonical:
            return existing
        raise RevisionConflictError(
            "scoped revision idempotency key already refers to different content"
        )

    existing = (
        None
        if scope_id is None
        else revision_by_address.get((scope_id, revision_id))
    )
    if existing is not None:
        if existing.proposal.canonical_bytes() == canonical:
            return existing
        raise RevisionConflictError(f"revision ID collision for {revision_id!r}")

    candidate = (
        None
        if scope_id is None
        else candidate_by_address.get((scope_id, proposal.candidate_id))
    )
    if candidate is None:
        raise CandidateNotFoundError(
            f"candidate {proposal.candidate_id!r} was not found"
        )
    assert scope_id is not None
    candidate_address = (scope_id, proposal.candidate_id)
    if candidate_address in revision_by_candidate:
        raise RevisionConflictError(
            f"candidate {proposal.candidate_id!r} already backs a revision"
        )

    parent: MemoryRevision | None = None
    if proposal.operation is not RevisionOperation.ADD:
        assert proposal.parent_revision_id is not None
        parent = revision_by_address.get((scope_id, proposal.parent_revision_id))
        if parent is None:
            raise RevisionNotFoundError(
                f"revision {proposal.parent_revision_id!r} was not found"
            )
    memory_id, generation = _derive_revision_lineage(
        proposal,
        content_hash,
        parent,
    )
    created_at = datetime.now(UTC)
    created_at_text = created_at.isoformat()
    storage_hash = _record_storage_hash(
        record_kind="revision",
        scope=proposal.scope,
        record_id=revision_id,
        content_hash=content_hash,
        created_at_text=created_at_text,
        memory_id=memory_id,
        generation=generation,
    )
    expected = MemoryRevision(
        revision_id=revision_id,
        memory_id=memory_id,
        generation=generation,
        proposal=proposal,
        content_hash=content_hash,
        created_at=created_at,
    )
    cursor.execute(
        """INSERT INTO memory_revisions (
    scope_id, revision_id, canonical, content_hash, created_at,
    storage_hash, candidate_id, memory_id, generation, operation,
    parent_revision_id, idempotency_key
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scope_id,
            revision_id,
            canonical,
            content_hash,
            created_at_text,
            storage_hash,
            proposal.candidate_id,
            memory_id,
            generation,
            proposal.operation.value,
            proposal.parent_revision_id,
            proposal.idempotency_key,
        ),
    )
    (
        _scope_by_id,
        _candidate_by_address,
        inserted_revisions,
        _revision_by_idempotency,
        _revision_by_candidate,
    ) = _load_revision_snapshot(cursor)
    inserted = inserted_revisions.get((scope_id, revision_id))
    if inserted is None:
        raise MemoryPersistenceCorruptionError(
            "inserted revision graph could not be reloaded"
        )
    if inserted != expected:
        raise MemoryPersistenceCorruptionError(
            "inserted revision did not round-trip exactly"
        )
    return inserted


def _append_release_locked(
    cursor: sqlite3.Cursor,
    manifest: ReleaseManifest,
    *,
    idempotency_key: str,
) -> MemoryRelease:
    """Append one release using the caller's active writer transaction."""

    canonical = manifest.canonical_bytes()
    content_hash = sha256(canonical).hexdigest()
    release_id = f"rel_{content_hash[:24]}"
    (
        scope_by_id,
        revision_by_address,
        release_by_address,
        release_by_alias,
        _revisions_by_release,
    ) = _load_release_snapshot(cursor)
    scope_id = _find_scope_id_in_index(scope_by_id, manifest.scope)
    existing = (
        None
        if scope_id is None
        else release_by_alias.get((scope_id, idempotency_key))
    )
    if existing is not None:
        if existing.manifest.canonical_bytes() == canonical:
            return existing
        raise ReleaseConflictError(
            "scoped release idempotency key already refers to different content"
        )

    revisions: list[MemoryRevision] = []
    for revision_id in manifest.revision_ids:
        revision = (
            None
            if scope_id is None
            else revision_by_address.get((scope_id, revision_id))
        )
        if revision is None:
            raise RevisionNotFoundError(f"revision {revision_id!r} was not found")
        revisions.append(revision)
    ordered_revisions = tuple(revisions)

    memory_ids: set[str] = set()
    for revision in ordered_revisions:
        if revision.memory_id in memory_ids:
            raise ReleaseConflictError(
                "release contains more than one revision for memory_id "
                f"{revision.memory_id!r}"
            )
        memory_ids.add(revision.memory_id)

    existing = (
        None
        if scope_id is None
        else release_by_address.get((scope_id, release_id))
    )
    if existing is not None and existing.manifest.canonical_bytes() != canonical:
        raise ReleaseConflictError(f"release ID collision for {release_id!r}")
    expected = existing
    if existing is None:
        if scope_id is None:
            scope_id = _ensure_scope_id(cursor, manifest.scope)
        created_at = datetime.now(UTC)
        created_at_text = created_at.isoformat()
        storage_hash = _record_storage_hash(
            record_kind="release",
            scope=manifest.scope,
            record_id=release_id,
            content_hash=content_hash,
            created_at_text=created_at_text,
        )
        expected = MemoryRelease(
            release_id=release_id,
            manifest=manifest,
            content_hash=content_hash,
            created_at=created_at,
        )
        cursor.execute(
            """INSERT INTO memory_releases (
    scope_id, release_id, canonical, content_hash, created_at, storage_hash
) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                scope_id,
                release_id,
                canonical,
                content_hash,
                created_at_text,
                storage_hash,
            ),
        )
        for position, revision in enumerate(ordered_revisions):
            cursor.execute(
                """INSERT INTO memory_release_revisions (
    scope_id, release_id, position, revision_id, memory_id
) VALUES (?, ?, ?, ?, ?)""",
                (
                    scope_id,
                    release_id,
                    position,
                    revision.revision_id,
                    revision.memory_id,
                ),
            )

    assert scope_id is not None
    binding_hash = _release_binding_hash(
        scope=manifest.scope,
        idempotency_key=idempotency_key,
        release_id=release_id,
    )
    cursor.execute(
        """INSERT INTO memory_release_aliases (
    scope_id, idempotency_key, release_id, binding_hash
) VALUES (?, ?, ?, ?)""",
        (scope_id, idempotency_key, release_id, binding_hash),
    )
    assert expected is not None
    (
        _readback_scope_by_id,
        _readback_revision_by_address,
        readback_release_by_address,
        readback_release_by_alias,
        readback_revisions_by_release,
    ) = _load_release_snapshot(cursor)
    address = (scope_id, release_id)
    inserted = readback_release_by_address.get(address)
    if inserted is None:
        raise MemoryPersistenceCorruptionError(
            "inserted release row could not be reloaded"
        )
    if inserted != expected:
        raise MemoryPersistenceCorruptionError(
            "inserted release did not round-trip exactly"
        )
    if readback_revisions_by_release.get(address) != ordered_revisions:
        raise MemoryPersistenceCorruptionError(
            "inserted release members did not round-trip exactly"
        )
    if readback_release_by_alias.get((scope_id, idempotency_key)) != inserted:
        raise MemoryPersistenceCorruptionError(
            "inserted release alias did not round-trip exactly"
        )
    return inserted


def _register_application_root_locked(
    cursor: sqlite3.Cursor,
    *,
    scope: MemoryScope,
    release_id: str,
) -> MemoryApplicationRootV1:
    state = _load_application_state(cursor)
    scope_id = _find_scope_id_in_index(state.scope_by_id, scope)
    release = (
        None
        if scope_id is None
        else state.release_by_address.get((scope_id, release_id))
    )
    if release is None:
        raise ReleaseNotFoundError(f"release {release_id!r} was not found")
    assert scope_id is not None
    existing = state.root_by_scope_id.get(scope_id)
    if existing is not None:
        if existing.release_id == release_id:
            return existing
        raise MemoryApplicationRootConflictError(
            "scope already has a different immutable application root"
        )
    if any(
        address_scope_id == scope_id
        for address_scope_id, _application_id in state.application_by_address
    ):
        raise MemoryApplicationRootConflictError(
            "application root must be registered before the first application"
        )
    root = MemoryApplicationRootV1.create(
        scope=scope,
        release_id=release.release_id,
        release_content_sha256=release.content_hash,
    )
    created_at_text = root.created_at.isoformat()
    cursor.execute(
        """INSERT INTO memory_application_roots (
    scope_id, root_id, canonical, content_hash, created_at, storage_hash,
    release_id, release_content_hash
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scope_id,
            root.root_id,
            root.canonical_bytes(),
            root.content_hash,
            created_at_text,
            _record_storage_hash(
                record_kind="memory_application_root",
                scope=scope,
                record_id=root.root_id,
                content_hash=root.content_hash,
                created_at_text=created_at_text,
            ),
            root.release_id,
            root.release_content_sha256,
        ),
    )
    readback = _load_application_state(cursor).root_by_scope_id.get(scope_id)
    if readback != root:
        raise MemoryPersistenceCorruptionError(
            "inserted application root did not round-trip exactly"
        )
    return readback


def _commit_memory_application_locked(
    cursor: sqlite3.Cursor,
    proposal: MemoryApplicationProposal,
) -> MemoryApplicationV1:
    state = _load_application_state(cursor)
    scope_id = _find_scope_id_in_index(state.scope_by_id, proposal.scope)
    existing = (
        None
        if scope_id is None
        else state.application_by_idempotency.get(
            (scope_id, proposal.idempotency_key)
        )
    )
    if existing is not None:
        if existing.proposal == proposal:
            return existing
        raise MemoryApplicationConflictError(
            "scoped application idempotency key has different content"
        )
    if scope_id is None:
        raise MemoryApplicationConflictError(
            "application scope has no source snapshot or root release"
        )
    snapshot = state.snapshot_by_address.get(
        (scope_id, proposal.source_snapshot_id)
    )
    base_release = state.release_by_address.get(
        (scope_id, proposal.source_base_release_id)
    )
    if snapshot is None:
        raise EvidenceSnapshotNotFoundError(
            f"evidence snapshot {proposal.source_snapshot_id!r} was not found"
        )
    if base_release is None:
        raise ReleaseNotFoundError(
            f"release {proposal.source_base_release_id!r} was not found"
        )
    base_address = (scope_id, base_release.release_id)
    if base_address not in state.root_by_release and base_address not in (
        state.application_by_result_release
    ):
        raise MemoryApplicationConflictError(
            "source base release is neither the root nor a committed result"
        )
    prior_scope_applications = tuple(
        application
        for (application_scope_id, _application_id), application in (
            state.application_by_address.items()
        )
        if application_scope_id == scope_id
    )
    expected_head_release_id = (
        state.root_by_scope_id[scope_id].release_id
        if not prior_scope_applications
        else max(
            prior_scope_applications,
            key=lambda item: item.application_order,
        ).result_release_id
    )
    if base_release.release_id != expected_head_release_id:
        raise MemoryApplicationConflictError(
            "source base release is not the scope's canonical head"
        )
    if any(
        evidence_scope_id == scope_id
        and ingest_order > snapshot.evidence_high_watermark
        for (evidence_scope_id, _evidence_id), ingest_order in (
            state.ingest_order_by_address.items()
        )
    ):
        raise MemoryApplicationStaleSnapshotError(
            "same-scope evidence arrived after the source snapshot was sealed"
        )
    member_by_id = {member.evidence_id: member for member in snapshot.members}
    if any(
        evidence_id not in member_by_id
        for update in proposal.updates
        for evidence_id in update.evidence_ids
    ):
        raise MemoryApplicationConflictError(
            "application update references evidence outside the source snapshot"
        )

    base_revisions = state.revisions_by_release[base_address]
    base_position = {
        revision.revision_id: position
        for position, revision in enumerate(base_revisions)
    }
    result_revision_ids = [revision.revision_id for revision in base_revisions]
    candidates: list[MemoryCandidate] = []
    revisions: list[MemoryRevision] = []
    release_positions: list[int] = []
    add_count = 0
    for ordinal, update in enumerate(proposal.updates):
        if update.operation is RevisionOperation.SUPERSEDE:
            assert update.parent_revision_id is not None
            position = base_position.get(update.parent_revision_id)
            if position is None:
                raise MemoryApplicationConflictError(
                    "SUPERSEDE parent is not an exact source base tip"
                )
        else:
            position = len(base_revisions) + add_count
            add_count += 1
        candidate_proposal = CandidateProposal(
            scope=proposal.scope,
            content=update.content,
            evidence_ids=update.evidence_ids,
            idempotency_key=_application_child_idempotency_key(
                proposal,
                record_kind="candidate",
                ordinal=ordinal,
            ),
        )
        candidate_hash = sha256(candidate_proposal.canonical_bytes()).hexdigest()
        candidate_id = f"cand_{candidate_hash[:24]}"
        if (scope_id, candidate_id) in state.candidate_by_address:
            raise MemoryApplicationConflictError(
                "application candidate already exists without this application"
            )
        candidate = _append_candidate_locked(cursor, candidate_proposal)
        _application_transaction_fault_hook("after_candidate")
        revision_proposal = RevisionProposal(
            scope=proposal.scope,
            candidate_id=candidate.candidate_id,
            operation=update.operation,
            parent_revision_id=update.parent_revision_id,
            idempotency_key=_application_child_idempotency_key(
                proposal,
                record_kind="revision",
                ordinal=ordinal,
            ),
        )
        revision_hash = sha256(revision_proposal.canonical_bytes()).hexdigest()
        revision_id = f"rev_{revision_hash[:24]}"
        if (scope_id, revision_id) in state.revision_by_address:
            raise MemoryApplicationConflictError(
                "application revision already exists without this application"
            )
        revision = _append_revision_locked(cursor, revision_proposal)
        _application_transaction_fault_hook("after_revision")
        if update.operation is RevisionOperation.SUPERSEDE:
            result_revision_ids[position] = revision.revision_id
        else:
            result_revision_ids.append(revision.revision_id)
        candidates.append(candidate)
        revisions.append(revision)
        release_positions.append(position)

    result_manifest = ReleaseManifest(
        scope=proposal.scope,
        revision_ids=tuple(result_revision_ids),
    )
    result_hash = sha256(result_manifest.canonical_bytes()).hexdigest()
    result_release_id = f"rel_{result_hash[:24]}"
    if (scope_id, result_release_id) in state.release_by_address:
        raise MemoryApplicationConflictError(
            "application result release already exists without this application"
        )
    result_release = _append_release_locked(
        cursor,
        result_manifest,
        idempotency_key=_application_child_idempotency_key(
            proposal,
            record_kind="release",
        ),
    )
    _application_transaction_fault_hook("after_release")
    application_order = len(state.application_by_address)
    applied_updates = tuple(
        AppliedMemoryUpdateV1(
            ordinal=ordinal,
            release_position=release_positions[ordinal],
            operation=revision.proposal.operation,
            grounding=tuple(
                member_by_id[evidence_id]
                for evidence_id in candidate.proposal.evidence_ids
            ),
            candidate_id=candidate.candidate_id,
            candidate_content_sha256=candidate.content_hash,
            revision_id=revision.revision_id,
            revision_content_sha256=revision.content_hash,
            memory_id=revision.memory_id,
            generation=revision.generation,
            parent_revision_id=revision.proposal.parent_revision_id,
        )
        for ordinal, (candidate, revision) in enumerate(
            zip(candidates, revisions, strict=True)
        )
    )
    application = MemoryApplicationV1.create(
        proposal=proposal,
        source_snapshot_content_sha256=snapshot.content_hash,
        source_evidence_high_watermark=snapshot.evidence_high_watermark,
        base_release_content_sha256=base_release.content_hash,
        result_release_id=result_release.release_id,
        result_release_content_sha256=result_release.content_hash,
        result_revision_ids=result_release.manifest.revision_ids,
        applied_updates=applied_updates,
        application_order=application_order,
    )
    created_at_text = application.created_at.isoformat()
    cursor.execute(
        """INSERT INTO memory_applications (
    application_order, scope_id, application_id, canonical, content_hash,
    created_at, storage_hash, source_snapshot_id, source_snapshot_content_hash,
    source_evidence_high_watermark, base_release_id, base_release_content_hash,
    result_release_id, result_release_content_hash, projector_id,
    projector_version_hash, policy_id, policy_version_hash, policy_input_hash,
    decision_hash, policy_context, update_count, idempotency_key
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            application.application_order,
            scope_id,
            application.application_id,
            application.canonical_bytes(),
            application.content_hash,
            created_at_text,
            _record_storage_hash(
                record_kind="memory_application",
                scope=proposal.scope,
                record_id=application.application_id,
                content_hash=application.content_hash,
                created_at_text=created_at_text,
            ),
            proposal.source_snapshot_id,
            application.source_snapshot_content_sha256,
            application.source_evidence_high_watermark,
            proposal.source_base_release_id,
            application.base_release_content_sha256,
            application.result_release_id,
            application.result_release_content_sha256,
            proposal.projector_id,
            proposal.projector_version_sha256,
            proposal.policy_id,
            proposal.policy_version_sha256,
            proposal.policy_input_sha256,
            proposal.decision_sha256,
            proposal.policy_context,
            len(application.applied_updates),
            proposal.idempotency_key,
        ),
    )
    _application_transaction_fault_hook("after_application")
    for update in application.applied_updates:
        cursor.execute(
            """INSERT INTO memory_application_revisions (
    scope_id, application_id, ordinal, revision_id,
    revision_content_hash, binding_hash
) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                scope_id,
                application.application_id,
                update.ordinal,
                update.revision_id,
                update.revision_content_sha256,
                _application_revision_binding_hash(
                    scope=proposal.scope,
                    application_id=application.application_id,
                    ordinal=update.ordinal,
                    revision_id=update.revision_id,
                    revision_content_hash=update.revision_content_sha256,
                ),
            ),
        )
        _application_transaction_fault_hook("after_edge")
    readback = _load_application_state(cursor).application_by_address.get(
        (scope_id, application.application_id)
    )
    if readback != application:
        raise MemoryPersistenceCorruptionError(
            "inserted memory application did not round-trip exactly"
        )
    _application_transaction_fault_hook("after_readback")
    return readback


class SQLiteMemoryStore:
    """Local SQLite backend for immutable evidence and memory history."""

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        self._database_path = _snapshot_database_path(database_path)
        _initialize_database(self._database_path)

    def append(self, event: EvidenceEvent) -> EvidenceRecord:
        """Persist evidence or return the existing scoped idempotent record."""

        if type(event) is not EvidenceEvent:
            raise TypeError("event must be an EvidenceEvent")
        canonical = event.canonical_bytes()
        content_hash = sha256(canonical).hexdigest()
        evidence_id = f"evd_{content_hash[:24]}"

        with _write_transaction(self._database_path) as cursor:
            scope_id = _ensure_scope_id(cursor, event.scope)
            scoped_records = _load_scope_evidence(cursor, event.scope, scope_id)
            existing = next(
                (
                    record
                    for record in scoped_records
                    if record.event.idempotency_key == event.idempotency_key
                ),
                None,
            )
            if existing is not None:
                if existing.event.canonical_bytes() == canonical:
                    return existing
                raise EvidenceConflictError(
                    "scoped idempotency key already refers to different evidence"
                )

            existing = next(
                (
                    record
                    for record in scoped_records
                    if record.evidence_id == evidence_id
                ),
                None,
            )
            if existing is not None:
                if existing.event.canonical_bytes() == canonical:
                    return existing
                raise EvidenceConflictError(
                    f"evidence ID collision for {evidence_id!r}"
                )

            created_at = datetime.now(UTC)
            created_at_text = created_at.isoformat()
            storage_hash = _record_storage_hash(
                record_kind="evidence",
                scope=event.scope,
                record_id=evidence_id,
                content_hash=content_hash,
                created_at_text=created_at_text,
            )
            cursor.execute(
                """INSERT INTO memory_evidence (
    scope_id, evidence_id, canonical, content_hash, created_at,
    storage_hash, session_id, run_id, sequence_no, kind, payload,
    observed_at, idempotency_key
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scope_id,
                    evidence_id,
                    canonical,
                    content_hash,
                    created_at_text,
                    storage_hash,
                    event.session_id,
                    event.run_id,
                    event.sequence_no,
                    event.kind.value,
                    event.payload,
                    event.observed_at.isoformat(),
                    event.idempotency_key,
                ),
            )
            row = cursor.execute(
                "SELECT MAX(ingest_order) FROM memory_evidence_ingest_orders"
            ).fetchone()
            if row is None or len(row) != 1 or type(row[0]) not in {int, type(None)}:
                raise MemoryPersistenceCorruptionError(
                    "evidence ingest-order maximum is invalid"
                )
            ingest_order = 0 if row[0] is None else row[0] + 1
            if not 0 <= ingest_order <= 2**63 - 1:
                raise MemoryPersistenceCorruptionError(
                    "evidence ingest-order range is exhausted"
                )
            cursor.execute(
                """INSERT INTO memory_evidence_ingest_orders (
    ingest_order, scope_id, evidence_id, binding_hash
) VALUES (?, ?, ?, ?)""",
                (
                    ingest_order,
                    scope_id,
                    evidence_id,
                    _evidence_ingest_binding_hash(
                        scope=event.scope,
                        evidence_id=evidence_id,
                        ingest_order=ingest_order,
                    ),
                ),
            )
            _validate_ingest_orders_locked(cursor)
            inserted = _load_evidence(
                cursor,
                event.scope,
                scope_id,
                evidence_id,
            )
            if inserted is None:
                raise MemoryPersistenceCorruptionError(
                    "inserted evidence row could not be reloaded"
                )
            return inserted

    def get(self, scope: MemoryScope, evidence_id: str) -> EvidenceRecord:
        """Load evidence only from its exact public scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        evidence_id = _validate_string(
            evidence_id,
            "evidence_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            scope_id = _find_scope_id(cursor, scope)
            scoped_records = _load_scope_evidence(cursor, scope, scope_id)
            record = next(
                (
                    scoped_record
                    for scoped_record in scoped_records
                    if scoped_record.evidence_id == evidence_id
                ),
                None,
            )
            if record is None:
                raise EvidenceNotFoundError(f"evidence {evidence_id!r} was not found")
            return record

    def list(
        self,
        scope: MemoryScope,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        """Load one scoped snapshot, optionally narrowed by session and run."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        if session_id is not None:
            session_id = _validate_string(
                session_id,
                "session_id",
                allow_blank=True,
            )
        if run_id is not None:
            run_id = _validate_string(run_id, "run_id", allow_blank=True)

        with _read_transaction(self._database_path) as cursor:
            scope_id = _find_scope_id(cursor, scope)
            records = (
                record
                for record in _load_scope_evidence(cursor, scope, scope_id)
                if (session_id is None or record.event.session_id == session_id)
                and (run_id is None or record.event.run_id == run_id)
            )
            return tuple(sorted(records, key=_evidence_sort_key))

    def seal_evidence_snapshot(
        self,
        spec: EvidenceSnapshotSpec,
        *,
        idempotency_key: str,
    ) -> EvidenceSnapshot:
        """Atomically seal the complete predicate match at one ingest watermark."""

        if type(spec) is not EvidenceSnapshotSpec:
            raise TypeError("spec must be an EvidenceSnapshotSpec")
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")

        with _write_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                evidence_by_address,
                ingest_order_by_address,
                snapshot_by_address,
                snapshot_by_alias,
            ) = _load_evidence_snapshot_state(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, spec.scope)
            existing = (
                None
                if scope_id is None
                else snapshot_by_alias.get((scope_id, idempotency_key))
            )
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise EvidenceSnapshotConflictError(
                    "scoped snapshot idempotency key already refers to a "
                    "different specification"
                )

            if scope_id is None:
                scope_id = _ensure_scope_id(cursor, spec.scope)
                scope_by_id[scope_id] = spec.scope
            evidence_high_watermark = len(ingest_order_by_address) - 1
            members = _snapshot_members_from_state(
                spec=spec,
                scope_id=scope_id,
                evidence_high_watermark=evidence_high_watermark,
                evidence_by_address=evidence_by_address,
                ingest_order_by_address=ingest_order_by_address,
            )
            provisional = EvidenceSnapshot(
                snapshot_id="pending",
                spec=spec,
                evidence_high_watermark=evidence_high_watermark,
                ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
                members=members,
                content_hash="pending",
                created_at=datetime.now(UTC),
            )
            canonical = provisional.canonical_bytes()
            content_hash = sha256(canonical).hexdigest()
            snapshot_id = f"esnap_{content_hash[:24]}"
            existing = snapshot_by_address.get((scope_id, snapshot_id))
            if existing is not None:
                if existing.canonical_bytes() != canonical:
                    raise EvidenceSnapshotConflictError(
                        f"evidence snapshot ID collision for {snapshot_id!r}"
                    )
                snapshot = existing
            else:
                created_at = provisional.created_at
                created_at_text = created_at.isoformat()
                storage_hash = _record_storage_hash(
                    record_kind="evidence_snapshot",
                    scope=spec.scope,
                    record_id=snapshot_id,
                    content_hash=content_hash,
                    created_at_text=created_at_text,
                )
                cursor.execute(
                    """INSERT INTO memory_evidence_snapshots (
    scope_id, snapshot_id, canonical, content_hash, created_at, storage_hash,
    allowed_kinds_canonical, cutoff_utc, evidence_high_watermark,
    ordering_policy, member_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        scope_id,
                        snapshot_id,
                        canonical,
                        content_hash,
                        created_at_text,
                        storage_hash,
                        _allowed_kinds_canonical_bytes(spec.allowed_kinds),
                        spec.cutoff.isoformat(),
                        evidence_high_watermark,
                        EVIDENCE_SNAPSHOT_ORDERING_POLICY,
                        len(members),
                    ),
                )
                snapshot = EvidenceSnapshot(
                    snapshot_id=snapshot_id,
                    spec=spec,
                    evidence_high_watermark=evidence_high_watermark,
                    ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
                    members=members,
                    content_hash=content_hash,
                    created_at=created_at,
                )
            cursor.execute(
                """INSERT INTO memory_evidence_snapshot_aliases (
    scope_id, idempotency_key, snapshot_id, binding_hash
) VALUES (?, ?, ?, ?)""",
                (
                    scope_id,
                    idempotency_key,
                    snapshot_id,
                    _snapshot_binding_hash(
                        scope=spec.scope,
                        idempotency_key=idempotency_key,
                        snapshot_id=snapshot_id,
                    ),
                ),
            )
            (
                _scope_by_id,
                _evidence_by_address,
                _ingest_order_by_address,
                _snapshot_by_address,
                reloaded_by_alias,
            ) = _load_evidence_snapshot_state(cursor)
            reloaded = reloaded_by_alias.get((scope_id, idempotency_key))
            if reloaded is None or reloaded != snapshot:
                raise MemoryPersistenceCorruptionError(
                    "inserted evidence snapshot could not be reloaded exactly"
                )
            return reloaded

    def get_evidence_snapshot(
        self,
        scope: MemoryScope,
        snapshot_id: str,
    ) -> EvidenceSnapshot:
        """Load an evidence snapshot only from its exact public scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        snapshot_id = _validate_string(
            snapshot_id,
            "snapshot_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _evidence_by_address,
                _ingest_order_by_address,
                snapshot_by_address,
                _snapshot_by_alias,
            ) = _load_evidence_snapshot_state(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            snapshot = (
                None
                if scope_id is None
                else snapshot_by_address.get((scope_id, snapshot_id))
            )
            if snapshot is None:
                raise EvidenceSnapshotNotFoundError(
                    f"evidence snapshot {snapshot_id!r} was not found"
                )
            return snapshot

    def get_evidence_snapshot_evidence(
        self,
        scope: MemoryScope,
        snapshot_id: str,
    ) -> tuple[EvidenceRecord, ...]:
        """Load exactly the records committed by one evidence snapshot."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        snapshot_id = _validate_string(
            snapshot_id,
            "snapshot_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                evidence_by_address,
                _ingest_order_by_address,
                snapshot_by_address,
                _snapshot_by_alias,
            ) = _load_evidence_snapshot_state(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            snapshot = (
                None
                if scope_id is None
                else snapshot_by_address.get((scope_id, snapshot_id))
            )
            if snapshot is None or scope_id is None:
                raise EvidenceSnapshotNotFoundError(
                    f"evidence snapshot {snapshot_id!r} was not found"
                )
            records = tuple(
                evidence_by_address[(scope_id, member.evidence_id)]
                for member in snapshot.members
            )
            if tuple(
                (record.evidence_id, record.content_hash) for record in records
            ) != tuple(
                (member.evidence_id, member.evidence_content_hash)
                for member in snapshot.members
            ):
                raise MemoryPersistenceCorruptionError(
                    "evidence snapshot records disagree with committed members"
                )
            return records

    def append_candidate(self, proposal: CandidateProposal) -> MemoryCandidate:
        """Persist one evidence-grounded candidate or return its exact retry."""

        if type(proposal) is not CandidateProposal:
            raise TypeError("proposal must be a CandidateProposal")
        with _write_transaction(self._database_path) as cursor:
            return _append_candidate_locked(cursor, proposal)

    def get_candidate(
        self,
        scope: MemoryScope,
        candidate_id: str,
    ) -> MemoryCandidate:
        """Load one candidate only from its exact public scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        candidate_id = _validate_string(
            candidate_id,
            "candidate_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _evidence_by_address,
                candidate_by_address,
                _candidate_by_idempotency,
            ) = _load_candidate_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            candidate = (
                None
                if scope_id is None
                else candidate_by_address.get((scope_id, candidate_id))
            )
            if candidate is None:
                raise CandidateNotFoundError(
                    f"candidate {candidate_id!r} was not found"
                )
            return candidate

    def get_candidate_evidence(
        self,
        scope: MemoryScope,
        candidate_id: str,
    ) -> tuple[EvidenceRecord, ...]:
        """Resolve one candidate's evidence in its exact proposal order."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        candidate_id = _validate_string(
            candidate_id,
            "candidate_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                evidence_by_address,
                candidate_by_address,
                _candidate_by_idempotency,
            ) = _load_candidate_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            candidate = (
                None
                if scope_id is None
                else candidate_by_address.get((scope_id, candidate_id))
            )
            if candidate is None:
                raise CandidateNotFoundError(
                    f"candidate {candidate_id!r} was not found"
                )
            return tuple(
                evidence_by_address[(scope_id, evidence_id)]
                for evidence_id in candidate.proposal.evidence_ids
            )

    def list_candidates(self, scope: MemoryScope) -> tuple[MemoryCandidate, ...]:
        """Return a stable candidate snapshot ordered by public identifier."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _evidence_by_address,
                candidate_by_address,
                _candidate_by_idempotency,
            ) = _load_candidate_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            if scope_id is None:
                return ()
            return tuple(
                sorted(
                    (
                        candidate
                        for (stored_scope_id, _candidate_id), candidate in (
                            candidate_by_address.items()
                        )
                        if stored_scope_id == scope_id
                    ),
                    key=lambda candidate: candidate.candidate_id,
                )
            )

    def append_revision(self, proposal: RevisionProposal) -> MemoryRevision:
        """Persist one immutable candidate transition or return its exact retry."""

        if type(proposal) is not RevisionProposal:
            raise TypeError("proposal must be a RevisionProposal")
        with _write_transaction(self._database_path) as cursor:
            return _append_revision_locked(cursor, proposal)

    def get_revision(
        self,
        scope: MemoryScope,
        revision_id: str,
    ) -> MemoryRevision:
        """Load one revision only from its exact public scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        revision_id = _validate_string(
            revision_id,
            "revision_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _candidate_by_address,
                revision_by_address,
                _revision_by_idempotency,
                _revision_by_candidate,
            ) = _load_revision_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            revision = (
                None
                if scope_id is None
                else revision_by_address.get((scope_id, revision_id))
            )
            if revision is None:
                raise RevisionNotFoundError(f"revision {revision_id!r} was not found")
            return revision

    def list_revisions(
        self,
        scope: MemoryScope,
        *,
        memory_id: str | None = None,
    ) -> tuple[MemoryRevision, ...]:
        """Return a trusted parent-before-child revision snapshot."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        if memory_id is not None:
            memory_id = _validate_string(
                memory_id,
                "memory_id",
                allow_blank=True,
            )
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _candidate_by_address,
                revision_by_address,
                _revision_by_idempotency,
                _revision_by_candidate,
            ) = _load_revision_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            if scope_id is None:
                return ()
            revisions = (
                revision
                for (stored_scope_id, _revision_id), revision in (
                    revision_by_address.items()
                )
                if stored_scope_id == scope_id
                and (memory_id is None or revision.memory_id == memory_id)
            )
            return tuple(
                sorted(
                    revisions,
                    key=lambda revision: (
                        revision.memory_id,
                        revision.generation,
                        revision.revision_id,
                    ),
                )
            )

    def append_release(
        self, manifest: ReleaseManifest, *, idempotency_key: str
    ) -> MemoryRelease:
        """Persist one healthy immutable release manifest or exact retry."""

        if type(manifest) is not ReleaseManifest:
            raise TypeError("manifest must be a ReleaseManifest")
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        with _write_transaction(self._database_path) as cursor:
            return _append_release_locked(
                cursor,
                manifest,
                idempotency_key=idempotency_key,
            )

    def get_release(self, scope: MemoryScope, release_id: str) -> MemoryRelease:
        """Load one release only from its exact public scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        release_id = _validate_string(release_id, "release_id", allow_blank=True)
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _revision_by_address,
                release_by_address,
                _release_by_alias,
                _revisions_by_release,
            ) = _load_release_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            release = (
                None
                if scope_id is None
                else release_by_address.get((scope_id, release_id))
            )
            if release is None:
                raise ReleaseNotFoundError(f"release {release_id!r} was not found")
            return release

    def get_release_revisions(
        self, scope: MemoryScope, release_id: str
    ) -> tuple[MemoryRevision, ...]:
        """Resolve one release's revisions in exact manifest order."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        release_id = _validate_string(release_id, "release_id", allow_blank=True)
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _revision_by_address,
                release_by_address,
                _release_by_alias,
                revisions_by_release,
            ) = _load_release_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            address = None if scope_id is None else (scope_id, release_id)
            if address is None or address not in release_by_address:
                raise ReleaseNotFoundError(f"release {release_id!r} was not found")
            return revisions_by_release[address]

    def list_releases(self, scope: MemoryScope) -> tuple[MemoryRelease, ...]:
        """Return a stable release snapshot ordered by public identifier."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with _read_transaction(self._database_path) as cursor:
            (
                scope_by_id,
                _revision_by_address,
                release_by_address,
                _release_by_alias,
                _revisions_by_release,
            ) = _load_release_snapshot(cursor)
            scope_id = _find_scope_id_in_index(scope_by_id, scope)
            if scope_id is None:
                return ()
            return tuple(
                sorted(
                    (
                        release
                        for (stored_scope_id, _release_id), release in (
                            release_by_address.items()
                        )
                        if stored_scope_id == scope_id
                    ),
                    key=lambda release: release.release_id,
                )
            )

    def register_memory_application_root(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> MemoryApplicationRootV1:
        """Commit the one explicit non-application release trusted by a scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        release_id = _validate_string(release_id, "release_id", allow_blank=True)
        with _write_transaction(self._database_path) as cursor:
            return _register_application_root_locked(
                cursor,
                scope=scope,
                release_id=release_id,
            )

    def get_memory_application_root(
        self,
        scope: MemoryScope,
    ) -> MemoryApplicationRootV1:
        """Load the exact immutable application root for one scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        with _read_transaction(self._database_path) as cursor:
            state = _load_application_state(cursor)
            scope_id = _find_scope_id_in_index(state.scope_by_id, scope)
            root = None if scope_id is None else state.root_by_scope_id.get(scope_id)
            if root is None:
                raise MemoryApplicationRootNotFoundError(
                    "memory application root was not found"
                )
            return root

    def commit_memory_application(
        self,
        proposal: MemoryApplicationProposal,
    ) -> MemoryApplicationV1:
        """Atomically publish updates, their release, and provenance ledger."""

        if type(proposal) is not MemoryApplicationProposal:
            raise TypeError("proposal must be a MemoryApplicationProposal")
        with _write_transaction(self._database_path) as cursor:
            return _commit_memory_application_locked(cursor, proposal)

    def get_memory_application(
        self,
        scope: MemoryScope,
        application_id: str,
    ) -> MemoryApplicationV1:
        """Load one committed application by exact scope and content ID."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        application_id = _validate_string(
            application_id,
            "application_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            state = _load_application_state(cursor)
            scope_id = _find_scope_id_in_index(state.scope_by_id, scope)
            application = (
                None
                if scope_id is None
                else state.application_by_address.get((scope_id, application_id))
            )
            if application is None:
                raise MemoryApplicationNotFoundError(
                    f"application {application_id!r} was not found"
                )
            return application

    def get_memory_application_for_revision(
        self,
        scope: MemoryScope,
        revision_id: str,
    ) -> MemoryApplicationV1:
        """Resolve the unique committed application that created a revision."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        revision_id = _validate_string(
            revision_id,
            "revision_id",
            allow_blank=True,
        )
        with _read_transaction(self._database_path) as cursor:
            state = _load_application_state(cursor)
            scope_id = _find_scope_id_in_index(state.scope_by_id, scope)
            application = (
                None
                if scope_id is None
                else state.application_by_revision.get((scope_id, revision_id))
            )
            if application is None:
                raise MemoryApplicationNotFoundError(
                    f"revision {revision_id!r} has no committed application"
                )
            return application

    def get_memory_application_for_release(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> MemoryApplicationV1:
        """Resolve the unique application that atomically published a release."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        release_id = _validate_string(release_id, "release_id", allow_blank=True)
        with _read_transaction(self._database_path) as cursor:
            state = _load_application_state(cursor)
            scope_id = _find_scope_id_in_index(state.scope_by_id, scope)
            application = (
                None
                if scope_id is None
                else state.application_by_result_release.get((scope_id, release_id))
            )
            if application is None:
                raise MemoryApplicationNotFoundError(
                    f"release {release_id!r} has no committed application"
                )
            return application
