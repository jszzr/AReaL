# SPDX-License-Identifier: Apache-2.0

"""File-backed SQLite implementation of immutable Memory Service history."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256

from areal.v2.memory_service._sqlite_backend import (
    _initialize_database,
    _read_transaction,
    _record_storage_hash,
    _snapshot_database_path,
    _write_transaction,
)
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryPersistenceCorruptionError,
)
from areal.v2.memory_service.history_types import CandidateProposal, MemoryCandidate
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

_CANDIDATE_SELECT = """SELECT candidate_id, canonical, content_hash,
       created_at, storage_hash, content, idempotency_key
FROM memory_candidates
WHERE scope_id = ? AND candidate_id = ?"""

_CANDIDATE_ROW_TYPES = (str, bytes, str, str, str, str, str)

_MAX_SCOPE_ID = 2**63 - 1


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


class SQLiteMemoryStore:
    """Local SQLite backend for immutable evidence and candidate history."""

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

    def append_candidate(self, proposal: CandidateProposal) -> MemoryCandidate:
        """Persist one evidence-grounded candidate or return its exact retry."""

        if type(proposal) is not CandidateProposal:
            raise TypeError("proposal must be a CandidateProposal")
        canonical = proposal.canonical_bytes()
        content_hash = sha256(canonical).hexdigest()
        candidate_id = f"cand_{content_hash[:24]}"

        with _write_transaction(self._database_path) as cursor:
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
                if (
                    scope_id is None
                    or (
                        scope_id,
                        evidence_id,
                    )
                    not in evidence_by_address
                ):
                    raise EvidenceNotFoundError(
                        f"evidence {evidence_id!r} was not found"
                    )
            assert scope_id is not None

            existing = candidate_by_address.get((scope_id, candidate_id))
            if existing is not None:
                if existing.proposal.canonical_bytes() == canonical:
                    return existing
                raise CandidateConflictError(
                    f"candidate ID collision for {candidate_id!r}"
                )

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
