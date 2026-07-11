# SPDX-License-Identifier: Apache-2.0

"""Durability and linearizability tests for SQLite evidence snapshots."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import pytest

import areal.v2.memory_service._sqlite_backend as sqlite_backend
import areal.v2.memory_service.sqlite_store as sqlite_store_module
from areal.v2.memory_service.errors import (
    EvidenceSnapshotConflictError,
    EvidenceSnapshotNotFoundError,
    MemoryPersistenceCorruptionError,
)
from areal.v2.memory_service.snapshot_types import (
    EvidenceSnapshot,
    EvidenceSnapshotSpec,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore
from areal.v2.memory_service.types import EvidenceEvent, EvidenceKind, MemoryScope

INSTANT = datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)


def make_scope(suffix: str = "1") -> MemoryScope:
    return MemoryScope(
        tenant_id=f"tenant-{suffix}",
        namespace="assistant-memory",
        subject_id=f"user-{suffix}",
    )


def make_event(
    *,
    scope: MemoryScope | None = None,
    suffix: str = "1",
    kind: EvidenceKind = EvidenceKind.USER_MESSAGE,
    observed_at: datetime = INSTANT,
    sequence_no: int = 0,
) -> EvidenceEvent:
    scope = make_scope() if scope is None else scope
    return EvidenceEvent(
        scope=scope,
        session_id="session-1",
        run_id="run-1",
        sequence_no=sequence_no,
        kind=kind,
        payload=f"payload-{suffix}",
        observed_at=observed_at,
        idempotency_key=f"evidence-{suffix}",
    )


def make_spec(
    *,
    scope: MemoryScope | None = None,
    cutoff: datetime = INSTANT,
    allowed_kinds: tuple[EvidenceKind, ...] = (
        EvidenceKind.USER_MESSAGE,
        EvidenceKind.FEEDBACK,
    ),
) -> EvidenceSnapshotSpec:
    return EvidenceSnapshotSpec(
        scope=make_scope() if scope is None else scope,
        allowed_kinds=allowed_kinds,
        cutoff=cutoff,
    )


def test_sqlite_snapshot_filters_orders_and_uses_global_high_watermark(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    scope = make_scope("1")
    other_scope = make_scope("2")

    store.append(
        make_event(
            scope=scope,
            suffix="excluded-kind",
            kind=EvidenceKind.AGENT_MESSAGE,
            observed_at=INSTANT - timedelta(seconds=3),
            sequence_no=9,
        )
    )
    late = store.append(
        make_event(
            scope=scope,
            suffix="late",
            kind=EvidenceKind.FEEDBACK,
            observed_at=INSTANT - timedelta(seconds=1),
            sequence_no=5,
        )
    )
    store.append(make_event(scope=other_scope, suffix="other-scope"))
    tie_first = store.append(
        make_event(
            scope=scope,
            suffix="tie-first",
            kind=EvidenceKind.FEEDBACK,
            observed_at=INSTANT - timedelta(seconds=2),
            sequence_no=2,
        )
    )
    tie_second = store.append(
        make_event(
            scope=scope,
            suffix="tie-second",
            observed_at=INSTANT - timedelta(seconds=2),
            sequence_no=2,
        )
    )
    store.append(
        make_event(
            scope=scope,
            suffix="after-cutoff",
            kind=EvidenceKind.FEEDBACK,
            observed_at=INSTANT + timedelta(microseconds=1),
            sequence_no=1,
        )
    )

    snapshot = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="snapshot-1",
    )

    expected = tuple(
        sorted(
            (tie_first, tie_second, late),
            key=lambda record: (
                record.event.observed_at,
                record.event.sequence_no,
                record.evidence_id,
            ),
        )
    )
    assert snapshot.evidence_high_watermark == 5
    assert tuple(member.evidence_id for member in snapshot.members) == tuple(
        record.evidence_id for record in expected
    )
    assert tuple(member.ingest_order for member in snapshot.members) == tuple(
        {
            late.evidence_id: 1,
            tie_first.evidence_id: 3,
            tie_second.evidence_id: 4,
        }[record.evidence_id]
        for record in expected
    )
    assert store.get_evidence_snapshot_evidence(scope, snapshot.snapshot_id) == expected


def test_sqlite_snapshot_survives_store_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    scope = make_scope()
    store = SQLiteMemoryStore(database_path)
    record = store.append(make_event(scope=scope))
    sealed = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="snapshot-1",
    )

    reopened = SQLiteMemoryStore(database_path)

    assert reopened.get_evidence_snapshot(scope, sealed.snapshot_id) == sealed
    assert reopened.get_evidence_snapshot_evidence(scope, sealed.snapshot_id) == (
        record,
    )
    assert (
        reopened.seal_evidence_snapshot(
            make_spec(scope=scope),
            idempotency_key="snapshot-1",
        )
        == sealed
    )


def test_late_backfill_cannot_change_old_alias_but_new_alias_includes_it(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    scope = make_scope()
    spec = make_spec(scope=scope)
    first = store.append(
        make_event(
            scope=scope,
            suffix="first",
            observed_at=INSTANT - timedelta(seconds=1),
        )
    )
    old = store.seal_evidence_snapshot(spec, idempotency_key="snapshot-old")

    backfill = store.append(
        make_event(
            scope=scope,
            suffix="late-backfill",
            observed_at=INSTANT - timedelta(days=30),
        )
    )

    assert store.seal_evidence_snapshot(spec, idempotency_key="snapshot-old") == old
    assert store.get_evidence_snapshot_evidence(scope, old.snapshot_id) == (first,)

    new = store.seal_evidence_snapshot(spec, idempotency_key="snapshot-new")
    assert new.snapshot_id != old.snapshot_id
    assert new.evidence_high_watermark == old.evidence_high_watermark + 1
    assert store.get_evidence_snapshot_evidence(scope, new.snapshot_id) == (
        backfill,
        first,
    )


def test_empty_snapshot_is_scoped_but_uses_global_high_watermark(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    empty_scope = make_scope("empty")
    other_scope = make_scope("other")
    store.append(make_event(scope=other_scope, suffix="other"))

    empty = store.seal_evidence_snapshot(
        make_spec(scope=empty_scope),
        idempotency_key="same-key",
    )
    other = store.seal_evidence_snapshot(
        make_spec(scope=other_scope),
        idempotency_key="same-key",
    )

    assert empty.evidence_high_watermark == 0
    assert empty.members == ()
    assert store.get_evidence_snapshot_evidence(empty_scope, empty.snapshot_id) == ()
    assert other.snapshot_id != empty.snapshot_id
    assert len(other.members) == 1


def test_snapshot_idempotency_conflicts_only_for_different_scoped_spec(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    scope = make_scope()
    first_spec = make_spec(scope=scope)
    first = store.seal_evidence_snapshot(
        first_spec,
        idempotency_key="snapshot-1",
    )

    assert (
        store.seal_evidence_snapshot(first_spec, idempotency_key="snapshot-1")
        == first
    )
    with pytest.raises(EvidenceSnapshotConflictError, match="different specification"):
        store.seal_evidence_snapshot(
            make_spec(scope=scope, cutoff=INSTANT + timedelta(seconds=1)),
            idempotency_key="snapshot-1",
        )


def test_concurrent_identical_seals_converge_on_one_durable_snapshot(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory.sqlite3"
    scope = make_scope()
    initializer = SQLiteMemoryStore(database_path)
    initializer.append(make_event(scope=scope))
    stores = tuple(SQLiteMemoryStore(database_path) for _ in range(8))
    barrier = Barrier(len(stores))

    def seal(store: SQLiteMemoryStore) -> object:
        barrier.wait()
        return store.seal_evidence_snapshot(
            make_spec(scope=scope),
            idempotency_key="shared-snapshot-key",
        )

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        snapshots = tuple(executor.map(seal, stores))

    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_snapshots"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_snapshot_aliases"
        ).fetchone() == (1,)


def test_concurrent_append_and_seal_have_a_valid_serial_order(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    SQLiteMemoryStore(database_path)
    append_store = SQLiteMemoryStore(database_path)
    seal_store = SQLiteMemoryStore(database_path)
    scope = make_scope()
    event = make_event(scope=scope)
    spec = make_spec(scope=scope)
    barrier = Barrier(2)

    def append() -> object:
        barrier.wait()
        return append_store.append(event)

    def seal() -> object:
        barrier.wait()
        return seal_store.seal_evidence_snapshot(
            spec,
            idempotency_key="racing-snapshot",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(append)
        seal_future = executor.submit(seal)
        record = append_future.result()
        snapshot = seal_future.result()

    assert snapshot.evidence_high_watermark in {-1, 0}
    if snapshot.evidence_high_watermark == -1:
        assert snapshot.members == ()
    else:
        assert tuple(member.evidence_id for member in snapshot.members) == (
            record.evidence_id,
        )

    reopened = SQLiteMemoryStore(database_path)
    assert (
        reopened.seal_evidence_snapshot(spec, idempotency_key="racing-snapshot")
        == snapshot
    )
    after_both = reopened.seal_evidence_snapshot(
        spec,
        idempotency_key="after-race",
    )
    assert after_both.evidence_high_watermark == 0
    assert tuple(member.evidence_id for member in after_both.members) == (
        record.evidence_id,
    )


@pytest.mark.parametrize(
    ("statement", "parameters"),
    [
        (
            "UPDATE memory_evidence_snapshots "
            "SET member_count = member_count + 1",
            (),
        ),
        (
            "UPDATE memory_evidence_snapshot_aliases SET binding_hash = ?",
            ("0" * 64,),
        ),
        (
            "UPDATE memory_evidence_ingest_orders SET binding_hash = ?",
            ("0" * 64,),
        ),
    ],
    ids=("snapshot-header", "snapshot-alias", "ingest-mapping"),
)
def test_snapshot_state_tampering_fails_closed(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    database_path = tmp_path / "memory.sqlite3"
    scope = make_scope()
    store = SQLiteMemoryStore(database_path)
    store.append(make_event(scope=scope))
    snapshot = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="snapshot-1",
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(statement, parameters)
        connection.commit()

    with pytest.raises(MemoryPersistenceCorruptionError):
        store.get_evidence_snapshot(scope, snapshot.snapshot_id)


def test_wrong_scope_is_indistinguishable_from_missing_snapshot(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    owner_scope = make_scope("owner")
    wrong_scope = make_scope("wrong")
    store.append(make_event(scope=wrong_scope, suffix="wrong-scope-record"))
    snapshot = store.seal_evidence_snapshot(
        make_spec(scope=owner_scope),
        idempotency_key="snapshot-1",
    )

    with pytest.raises(EvidenceSnapshotNotFoundError) as wrong_scope_error:
        store.get_evidence_snapshot(wrong_scope, snapshot.snapshot_id)
    with pytest.raises(EvidenceSnapshotNotFoundError) as missing_error:
        store.get_evidence_snapshot(owner_scope, "esnap_missing")

    assert type(wrong_scope_error.value) is type(missing_error.value)
    assert "scope" not in str(wrong_scope_error.value).lower()
    with pytest.raises(EvidenceSnapshotNotFoundError):
        store.get_evidence_snapshot_evidence(wrong_scope, snapshot.snapshot_id)


def test_self_consistent_future_watermark_is_rejected(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    scope = make_scope()
    store = SQLiteMemoryStore(database_path)
    store.append(make_event(scope=scope))
    snapshot = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="snapshot-1",
    )
    impossible = EvidenceSnapshot(
        snapshot_id="pending",
        spec=snapshot.spec,
        evidence_high_watermark=1,
        ordering_policy=snapshot.ordering_policy,
        members=snapshot.members,
        content_hash="pending",
        created_at=snapshot.created_at,
    )
    canonical = impossible.canonical_bytes()
    content_hash = sha256(canonical).hexdigest()
    snapshot_id = f"esnap_{content_hash[:24]}"
    storage_hash = sqlite_backend._record_storage_hash(
        record_kind="evidence_snapshot",
        scope=scope,
        record_id=snapshot_id,
        content_hash=content_hash,
        created_at_text=snapshot.created_at.isoformat(),
    )
    alias_hash = sqlite_backend._snapshot_binding_hash(
        scope=scope,
        idempotency_key="snapshot-1",
        snapshot_id=snapshot_id,
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE memory_evidence_snapshots SET snapshot_id = ?, "
            "canonical = ?, content_hash = ?, storage_hash = ?, "
            "evidence_high_watermark = ?",
            (snapshot_id, canonical, content_hash, storage_hash, 1),
        )
        connection.execute(
            "UPDATE memory_evidence_snapshot_aliases SET snapshot_id = ?, "
            "binding_hash = ?",
            (snapshot_id, alias_hash),
        )
        connection.commit()

    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="snapshot row failed integrity validation",
    ):
        store.get_evidence_snapshot(scope, snapshot_id)


def test_snapshot_hash_collision_rolls_back_loser_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    real_sha256 = sha256

    def collide_snapshots_only(payload: bytes) -> object:
        if b'"ordering_policy"' in payload:
            return FixedDigest()
        return real_sha256(payload)

    database_path = tmp_path / "memory.sqlite3"
    scope = make_scope()
    store = SQLiteMemoryStore(database_path)
    store.append(make_event(scope=scope))
    monkeypatch.setattr(sqlite_store_module, "sha256", collide_snapshots_only)
    first = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="winner",
    )
    store.append(make_event(scope=make_scope("other"), suffix="other"))

    with pytest.raises(EvidenceSnapshotConflictError, match="collision"):
        store.seal_evidence_snapshot(
            make_spec(scope=scope),
            idempotency_key="loser",
        )

    assert first.snapshot_id == "esnap_" + "0" * 24
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_snapshots"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT idempotency_key FROM memory_evidence_snapshot_aliases"
        ).fetchall() == [("winner",)]
