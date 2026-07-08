# SPDX-License-Identifier: Apache-2.0

"""Durable SQLite implementation of the Memory Service contracts."""

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
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryPersistenceCorruptionError,
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

_MAX_SCOPE_ID = 2**63 - 1


def _require_scope_id(value: object, message: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_SCOPE_ID:
        raise MemoryPersistenceCorruptionError(message)
    return value


def _find_scope_id(cursor: sqlite3.Cursor, scope: MemoryScope) -> int | None:
    row = cursor.execute(
        "SELECT scope_id FROM memory_scopes "
        "WHERE tenant_id = ? AND namespace = ? AND subject_id = ?",
        (scope.tenant_id, scope.namespace, scope.subject_id),
    ).fetchone()
    if row is None:
        return None
    if len(row) != 1:
        raise MemoryPersistenceCorruptionError(
            "memory scope row does not contain exactly one identifier"
        )
    return _require_scope_id(
        row[0],
        "memory scope identifier is not a positive signed 64-bit integer",
    )


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


def _find_evidence_id_by_idempotency_key(
    cursor: sqlite3.Cursor,
    scope_id: int,
    idempotency_key: str,
) -> str | None:
    row = cursor.execute(
        "SELECT evidence_id FROM memory_evidence "
        "WHERE scope_id = ? AND idempotency_key = ?",
        (scope_id, idempotency_key),
    ).fetchone()
    if row is None:
        return None
    if len(row) != 1 or type(row[0]) is not str:
        raise MemoryPersistenceCorruptionError(
            "stored evidence idempotency index is invalid"
        )
    return row[0]


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


class SQLiteMemoryStore:
    """Local durable backend for immutable Memory Service records."""

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
            existing_id = _find_evidence_id_by_idempotency_key(
                cursor,
                scope_id,
                event.idempotency_key,
            )
            if existing_id is not None:
                existing = _load_evidence(
                    cursor,
                    event.scope,
                    scope_id,
                    existing_id,
                )
                if existing is None:
                    raise MemoryPersistenceCorruptionError(
                        "evidence idempotency index refers to a missing row"
                    )
                if existing.event.canonical_bytes() == canonical:
                    return existing
                raise EvidenceConflictError(
                    "scoped idempotency key already refers to different evidence"
                )

            existing = _load_evidence(
                cursor,
                event.scope,
                scope_id,
                evidence_id,
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
            record = (
                None
                if scope_id is None
                else _load_evidence(cursor, scope, scope_id, evidence_id)
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
            if scope_id is None:
                return ()
            sql = "SELECT evidence_id FROM memory_evidence WHERE scope_id = ?"
            parameters: list[object] = [scope_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                parameters.append(session_id)
            if run_id is not None:
                sql += " AND run_id = ?"
                parameters.append(run_id)
            rows = cursor.execute(sql, parameters).fetchall()
            records: list[EvidenceRecord] = []
            for row in rows:
                if len(row) != 1 or type(row[0]) is not str:
                    raise MemoryPersistenceCorruptionError(
                        "evidence listing contains an invalid identifier"
                    )
                record = _load_evidence(cursor, scope, scope_id, row[0])
                if record is None:
                    raise MemoryPersistenceCorruptionError(
                        "evidence listing refers to a missing row"
                    )
                records.append(record)
            return tuple(sorted(records, key=_evidence_sort_key))
