# SPDX-License-Identifier: Apache-2.0

"""Evidence store contracts and an in-memory implementation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceNotFoundError,
    EvidenceSnapshotConflictError,
    EvidenceSnapshotNotFoundError,
)
from areal.v2.memory_service.snapshot_types import (
    EVIDENCE_SNAPSHOT_ORDERING_POLICY,
    EvidenceSnapshot,
    EvidenceSnapshotMember,
    EvidenceSnapshotSpec,
    _snapshot_canonical_bytes,
)
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceRecord,
    MemoryScope,
    _validate_string,
)


class EvidenceStore(Protocol):
    """Storage contract for immutable evidence records."""

    def append(self, event: EvidenceEvent) -> EvidenceRecord:
        """Persist an event or return its existing idempotent record."""

        ...

    def get(self, scope: MemoryScope, evidence_id: str) -> EvidenceRecord:
        """Return evidence from a scope or raise ``EvidenceNotFoundError``."""

        ...

    def list(
        self,
        scope: MemoryScope,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        """Return deterministically ordered evidence matching the filters."""

        ...


class EvidenceSnapshotStore(EvidenceStore, Protocol):
    """Backward-compatible extension for complete, immutable evidence seals."""

    def seal_evidence_snapshot(
        self,
        spec: EvidenceSnapshotSpec,
        *,
        idempotency_key: str,
    ) -> EvidenceSnapshot:
        """Atomically seal every evidence record matching ``spec``."""

        ...

    def get_evidence_snapshot(
        self,
        scope: MemoryScope,
        snapshot_id: str,
    ) -> EvidenceSnapshot:
        """Return a snapshot only from its exact public scope."""

        ...

    def get_evidence_snapshot_evidence(
        self,
        scope: MemoryScope,
        snapshot_id: str,
    ) -> tuple[EvidenceRecord, ...]:
        """Return sealed records in their immutable snapshot order."""

        ...


class InMemoryEvidenceStore:
    """Lock-protected process-local storage for immutable evidence records."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_evidence_id: dict[tuple[MemoryScope, str], EvidenceRecord] = {}
        self._by_idempotency_key: dict[tuple[MemoryScope, str], EvidenceRecord] = {}
        self._by_scope: dict[MemoryScope, list[EvidenceRecord]] = {}
        self._ingest_order_by_evidence_id: dict[tuple[MemoryScope, str], int] = {}
        self._next_ingest_order = 0
        self._snapshot_by_id: dict[
            tuple[MemoryScope, str], EvidenceSnapshot
        ] = {}
        self._snapshot_by_idempotency_key: dict[
            tuple[MemoryScope, str], EvidenceSnapshot
        ] = {}

    def append(self, event: EvidenceEvent) -> EvidenceRecord:
        """Persist an event, enforcing scoped idempotency and collision safety."""

        if type(event) is not EvidenceEvent:
            raise TypeError("event must be an EvidenceEvent")
        canonical_bytes = event.canonical_bytes()
        content_hash = hashlib.sha256(canonical_bytes).hexdigest()
        evidence_id = f"evd_{content_hash[:24]}"
        evidence_index = (event.scope, evidence_id)
        idempotency_index = (event.scope, event.idempotency_key)

        with self._lock:
            existing_record = self._by_idempotency_key.get(idempotency_index)
            if existing_record is not None:
                if existing_record.event.canonical_bytes() == canonical_bytes:
                    return existing_record
                raise EvidenceConflictError(
                    "scoped idempotency key already refers to different evidence"
                )

            existing_record = self._by_evidence_id.get(evidence_index)
            if existing_record is not None:
                if existing_record.event.canonical_bytes() == canonical_bytes:
                    return existing_record
                raise EvidenceConflictError(
                    f"evidence ID collision for {evidence_id!r}"
                )

            record = EvidenceRecord(
                evidence_id=evidence_id,
                event=event,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
            )
            ingest_order = self._next_ingest_order
            self._by_evidence_id[evidence_index] = record
            self._by_idempotency_key[idempotency_index] = record
            self._by_scope.setdefault(event.scope, []).append(record)
            self._ingest_order_by_evidence_id[evidence_index] = ingest_order
            self._next_ingest_order += 1
            return record

    def get(self, scope: MemoryScope, evidence_id: str) -> EvidenceRecord:
        """Return evidence only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        evidence_id = _validate_string(evidence_id, "evidence_id", allow_blank=True)
        with self._lock:
            record = self._by_evidence_id.get((scope, evidence_id))
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
        """Return a stable snapshot of records belonging to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        if session_id is not None:
            session_id = _validate_string(session_id, "session_id", allow_blank=True)
        if run_id is not None:
            run_id = _validate_string(run_id, "run_id", allow_blank=True)
        with self._lock:
            matches = (
                record
                for record in self._by_scope.get(scope, ())
                if (session_id is None or record.event.session_id == session_id)
                and (run_id is None or record.event.run_id == run_id)
            )
            return tuple(sorted(matches, key=_record_sort_key))

    def seal_evidence_snapshot(
        self,
        spec: EvidenceSnapshotSpec,
        *,
        idempotency_key: str,
    ) -> EvidenceSnapshot:
        """Seal a complete cutoff-bounded snapshot at one ingest high watermark."""

        if type(spec) is not EvidenceSnapshotSpec:
            raise TypeError("spec must be an EvidenceSnapshotSpec")
        idempotency_key = _validate_string(idempotency_key, "idempotency_key")
        alias_index = (spec.scope, idempotency_key)

        with self._lock:
            existing = self._snapshot_by_idempotency_key.get(alias_index)
            if existing is not None:
                if existing.spec == spec:
                    return existing
                raise EvidenceSnapshotConflictError(
                    "scoped snapshot idempotency key already refers to a "
                    "different specification"
                )

            evidence_high_watermark = self._next_ingest_order - 1
            selected = tuple(
                record
                for record in self._by_scope.get(spec.scope, ())
                if self._ingest_order_by_evidence_id[
                    (spec.scope, record.evidence_id)
                ]
                <= evidence_high_watermark
                and record.event.kind in spec.allowed_kinds
                and record.event.observed_at <= spec.cutoff
            )
            ordered = tuple(sorted(selected, key=_snapshot_record_sort_key))
            members = tuple(
                EvidenceSnapshotMember(
                    evidence_id=record.evidence_id,
                    evidence_content_hash=record.content_hash,
                    ingest_order=self._ingest_order_by_evidence_id[
                        (spec.scope, record.evidence_id)
                    ],
                )
                for record in ordered
            )
            canonical = _snapshot_canonical_bytes(
                spec=spec,
                evidence_high_watermark=evidence_high_watermark,
                ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
                members=members,
            )
            content_hash = hashlib.sha256(canonical).hexdigest()
            snapshot_id = f"esnap_{content_hash[:24]}"
            snapshot_index = (spec.scope, snapshot_id)
            snapshot = self._snapshot_by_id.get(snapshot_index)
            if snapshot is not None:
                if snapshot.canonical_bytes() != canonical:
                    raise EvidenceSnapshotConflictError(
                        f"evidence snapshot ID collision for {snapshot_id!r}"
                    )
            else:
                snapshot = EvidenceSnapshot(
                    snapshot_id=snapshot_id,
                    spec=spec,
                    evidence_high_watermark=evidence_high_watermark,
                    ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
                    members=members,
                    content_hash=content_hash,
                    created_at=datetime.now(UTC),
                )
                self._snapshot_by_id[snapshot_index] = snapshot
            self._snapshot_by_idempotency_key[alias_index] = snapshot
            return snapshot

    def get_evidence_snapshot(
        self,
        scope: MemoryScope,
        snapshot_id: str,
    ) -> EvidenceSnapshot:
        """Return a snapshot only when it belongs to the requested scope."""

        if type(scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        snapshot_id = _validate_string(
            snapshot_id,
            "snapshot_id",
            allow_blank=True,
        )
        with self._lock:
            snapshot = self._snapshot_by_id.get((scope, snapshot_id))
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
        """Return the exact records committed by a stored snapshot."""

        snapshot = self.get_evidence_snapshot(scope, snapshot_id)
        with self._lock:
            return tuple(
                self._by_evidence_id[(scope, member.evidence_id)]
                for member in snapshot.members
            )


def _record_sort_key(record: EvidenceRecord) -> tuple[str, str, int, datetime, str]:
    event = record.event
    return (
        event.session_id,
        event.run_id,
        event.sequence_no,
        event.observed_at,
        record.evidence_id,
    )


def _snapshot_record_sort_key(
    record: EvidenceRecord,
) -> tuple[datetime, int, str]:
    event = record.event
    return (
        event.observed_at,
        event.sequence_no,
        record.evidence_id,
    )
