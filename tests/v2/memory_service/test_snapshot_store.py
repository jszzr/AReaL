# SPDX-License-Identifier: Apache-2.0

"""Tests for evidence snapshot values and the in-memory sealing protocol."""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256

import pytest

import areal.v2.memory_service as memory_service
from areal.v2.memory_service import store as store_module
from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceSnapshotConflictError,
    EvidenceSnapshotNotFoundError,
    MemoryServiceError,
)
from areal.v2.memory_service.snapshot_types import (
    EVIDENCE_SNAPSHOT_ORDERING_POLICY,
    EvidenceSnapshot,
    EvidenceSnapshotMember,
    EvidenceSnapshotSpec,
)
from areal.v2.memory_service.store import (
    EvidenceSnapshotStore,
    EvidenceStore,
    InMemoryEvidenceStore,
)
from areal.v2.memory_service.types import EvidenceEvent, EvidenceKind, MemoryScope

INSTANT = datetime(2026, 7, 7, 4, 5, 6, 789000, tzinfo=UTC)


def make_scope(suffix: str = "1") -> MemoryScope:
    return MemoryScope(
        tenant_id=f"tenant-{suffix}",
        namespace="assistant-memory",
        subject_id=f"user-{suffix}",
    )


def make_event(**overrides: object) -> EvidenceEvent:
    values: dict[str, object] = {
        "scope": make_scope(),
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence_no": 0,
        "kind": EvidenceKind.USER_MESSAGE,
        "payload": "hello",
        "observed_at": INSTANT,
        "idempotency_key": "evidence-1",
    }
    values.update(overrides)
    return EvidenceEvent(**values)  # type: ignore[arg-type]


def make_spec(**overrides: object) -> EvidenceSnapshotSpec:
    values: dict[str, object] = {
        "scope": make_scope(),
        "allowed_kinds": (
            EvidenceKind.USER_MESSAGE,
            EvidenceKind.FEEDBACK,
        ),
        "cutoff": INSTANT,
    }
    values.update(overrides)
    return EvidenceSnapshotSpec(**values)  # type: ignore[arg-type]


def make_member(**overrides: object) -> EvidenceSnapshotMember:
    values: dict[str, object] = {
        "evidence_id": "evd_0123456789abcdef01234567",
        "evidence_content_hash": "a" * 64,
        "ingest_order": 0,
    }
    values.update(overrides)
    return EvidenceSnapshotMember(**values)  # type: ignore[arg-type]


def test_snapshot_errors_and_public_exports() -> None:
    assert issubclass(EvidenceSnapshotNotFoundError, MemoryServiceError)
    assert issubclass(EvidenceSnapshotConflictError, MemoryServiceError)
    assert memory_service.EvidenceSnapshot is EvidenceSnapshot
    assert memory_service.EvidenceSnapshotSpec is EvidenceSnapshotSpec
    assert memory_service.EvidenceSnapshotMember is EvidenceSnapshotMember
    assert memory_service.EvidenceSnapshotStore is EvidenceSnapshotStore
    assert memory_service.EVIDENCE_SNAPSHOT_ORDERING_POLICY == (
        EVIDENCE_SNAPSHOT_ORDERING_POLICY
    )
    assert {
        "EvidenceSnapshot",
        "EvidenceSnapshotSpec",
        "EvidenceSnapshotMember",
        "EvidenceSnapshotNotFoundError",
        "EvidenceSnapshotConflictError",
    } <= set(memory_service.__all__)


def test_snapshot_spec_normalizes_set_semantics_and_utc() -> None:
    first = make_spec(
        allowed_kinds=(EvidenceKind.USER_MESSAGE, EvidenceKind.FEEDBACK),
        cutoff=INSTANT.astimezone(timezone(timedelta(hours=8))),
    )
    second = make_spec(
        allowed_kinds=(EvidenceKind.FEEDBACK, EvidenceKind.USER_MESSAGE),
        cutoff=INSTANT,
    )

    assert first == second
    assert first.allowed_kinds == (
        EvidenceKind.FEEDBACK,
        EvidenceKind.USER_MESSAGE,
    )
    assert first.cutoff == INSTANT
    assert first.cutoff.tzinfo is UTC
    assert first.canonical_bytes() == second.canonical_bytes()
    assert json.loads(first.canonical_bytes()) == {
        "allowed_kinds": ["feedback", "user_message"],
        "cutoff_utc": "2026-07-07T04:05:06.789000+00:00",
        "schema_version": 1,
        "scope": {
            "namespace": "assistant-memory",
            "subject_id": "user-1",
            "tenant_id": "tenant-1",
        },
    }


@pytest.mark.parametrize(
    "allowed_kinds",
    [[], (), (EvidenceKind.USER_MESSAGE, EvidenceKind.USER_MESSAGE), ("feedback",)],
)
def test_snapshot_spec_rejects_invalid_allowed_kinds(
    allowed_kinds: object,
) -> None:
    error = TypeError if allowed_kinds in ([], ("feedback",)) else ValueError
    with pytest.raises(error, match="allowed_kinds"):
        make_spec(allowed_kinds=allowed_kinds)


@pytest.mark.parametrize("cutoff", [None, "2026-07-07", datetime(2026, 7, 7)])
def test_snapshot_spec_requires_aware_cutoff(cutoff: object) -> None:
    with pytest.raises((TypeError, ValueError), match="cutoff"):
        make_spec(cutoff=cutoff)


@pytest.mark.parametrize("ingest_order", [True, 1.0, -1, 2**63])
def test_snapshot_member_rejects_invalid_ingest_order(ingest_order: object) -> None:
    with pytest.raises((TypeError, ValueError), match="ingest_order"):
        make_member(ingest_order=ingest_order)


def test_snapshot_canonical_binds_full_ordered_members_but_not_metadata() -> None:
    spec = make_spec()
    first_member = make_member()
    second_member = make_member(
        evidence_id="evd_89abcdef0123456789abcdef",
        evidence_content_hash="b" * 64,
        ingest_order=2,
    )
    first = EvidenceSnapshot(
        snapshot_id="esnap_" + "c" * 24,
        spec=spec,
        evidence_high_watermark=2,
        ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
        members=(first_member, second_member),
        content_hash="c" * 64,
        created_at=INSTANT,
    )
    different_metadata = EvidenceSnapshot(
        snapshot_id="esnap_" + "d" * 24,
        spec=spec,
        evidence_high_watermark=2,
        ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
        members=(first_member, second_member),
        content_hash="d" * 64,
        created_at=INSTANT + timedelta(seconds=1),
    )
    reversed_members = EvidenceSnapshot(
        snapshot_id="esnap_" + "e" * 24,
        spec=spec,
        evidence_high_watermark=2,
        ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
        members=(second_member, first_member),
        content_hash="e" * 64,
        created_at=INSTANT,
    )

    assert first.canonical_bytes() == different_metadata.canonical_bytes()
    assert first.canonical_bytes() != reversed_members.canonical_bytes()
    value = json.loads(first.canonical_bytes())
    assert value["evidence_high_watermark"] == 2
    assert value["ordering_policy"] == EVIDENCE_SNAPSHOT_ORDERING_POLICY
    assert value["members"] == [
        {
            "evidence_content_hash": "a" * 64,
            "evidence_id": first_member.evidence_id,
            "ingest_order": 0,
        },
        {
            "evidence_content_hash": "b" * 64,
            "evidence_id": second_member.evidence_id,
            "ingest_order": 2,
        },
    ]
    assert "created_at" not in value
    assert "snapshot_id" not in value
    assert "content_hash" not in value


def test_snapshot_values_are_frozen_slotted_and_validate_members() -> None:
    spec = make_spec()
    member = make_member()
    snapshot = EvidenceSnapshot(
        snapshot_id="esnap_" + "c" * 24,
        spec=spec,
        evidence_high_watermark=0,
        ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
        members=(member,),
        content_hash="c" * 64,
        created_at=INSTANT,
    )

    for value in (spec, member, snapshot):
        assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        snapshot.content_hash = "d" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="duplicate evidence"):
        EvidenceSnapshot(
            snapshot_id=snapshot.snapshot_id,
            spec=spec,
            evidence_high_watermark=0,
            ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
            members=(member, member),
            content_hash=snapshot.content_hash,
            created_at=INSTANT,
        )
    with pytest.raises(ValueError, match="high_watermark"):
        EvidenceSnapshot(
            snapshot_id=snapshot.snapshot_id,
            spec=spec,
            evidence_high_watermark=-1,
            ordering_policy=EVIDENCE_SNAPSHOT_ORDERING_POLICY,
            members=(member,),
            content_hash=snapshot.content_hash,
            created_at=INSTANT,
        )


def test_seal_api_does_not_accept_caller_supplied_members() -> None:
    parameters = inspect.signature(
        EvidenceSnapshotStore.seal_evidence_snapshot
    ).parameters

    assert tuple(parameters) == ("self", "spec", "idempotency_key")
    assert parameters["idempotency_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not hasattr(EvidenceSnapshotStore, "update_evidence_snapshot")
    assert not hasattr(EvidenceSnapshotStore, "delete_evidence_snapshot")
    assert not hasattr(EvidenceStore, "seal_evidence_snapshot")


def test_append_assigns_global_contiguous_ingest_order_without_retry_gaps() -> None:
    store = InMemoryEvidenceStore()
    first = store.append(make_event(idempotency_key="first"))
    foreign = store.append(
        make_event(scope=make_scope("2"), idempotency_key="foreign")
    )
    third = store.append(make_event(idempotency_key="third", sequence_no=2))

    assert store.append(make_event(idempotency_key="first")) is first
    with pytest.raises(EvidenceConflictError):
        store.append(make_event(idempotency_key="first", payload="changed"))
    assert store._ingest_order_by_evidence_id == {
        (first.event.scope, first.evidence_id): 0,
        (foreign.event.scope, foreign.evidence_id): 1,
        (third.event.scope, third.evidence_id): 2,
    }
    assert store._next_ingest_order == 3


def test_seal_freezes_complete_filtered_set_in_policy_order() -> None:
    store = InMemoryEvidenceStore()
    scope = make_scope()
    foreign = store.append(
        make_event(scope=make_scope("2"), idempotency_key="foreign")
    )
    disallowed = store.append(
        make_event(
            scope=scope,
            kind=EvidenceKind.ENVIRONMENT,
            idempotency_key="disallowed",
        )
    )
    after_cutoff = store.append(
        make_event(
            scope=scope,
            observed_at=INSTANT + timedelta(microseconds=1),
            idempotency_key="after",
        )
    )
    at_cutoff = store.append(
        make_event(
            scope=scope,
            session_id="session-a",
            sequence_no=9,
            kind=EvidenceKind.FEEDBACK,
            idempotency_key="at-cutoff",
        )
    )
    earlier = store.append(
        make_event(
            scope=scope,
            session_id="session-z",
            sequence_no=99,
            observed_at=INSTANT - timedelta(seconds=1),
            idempotency_key="earlier",
        )
    )

    snapshot = store.seal_evidence_snapshot(
        make_spec(scope=scope),
        idempotency_key="snapshot-1",
    )

    assert snapshot.evidence_high_watermark == 4
    assert tuple(member.evidence_id for member in snapshot.members) == (
        earlier.evidence_id,
        at_cutoff.evidence_id,
    )
    assert tuple(member.ingest_order for member in snapshot.members) == (4, 3)
    assert tuple(member.evidence_content_hash for member in snapshot.members) == (
        earlier.content_hash,
        at_cutoff.content_hash,
    )
    assert store.get_evidence_snapshot_evidence(scope, snapshot.snapshot_id) == (
        earlier,
        at_cutoff,
    )
    assert {foreign, disallowed, after_cutoff}.isdisjoint(
        store.get_evidence_snapshot_evidence(scope, snapshot.snapshot_id)
    )
    assert snapshot.content_hash == sha256(snapshot.canonical_bytes()).hexdigest()
    assert snapshot.snapshot_id == f"esnap_{snapshot.content_hash[:24]}"


def test_empty_snapshot_uses_global_high_watermark() -> None:
    store = InMemoryEvidenceStore()
    empty = store.seal_evidence_snapshot(
        make_spec(),
        idempotency_key="empty",
    )
    store.append(make_event(scope=make_scope("2"), idempotency_key="foreign"))
    foreign_bounded = store.seal_evidence_snapshot(
        make_spec(),
        idempotency_key="foreign-bounded",
    )

    assert empty.evidence_high_watermark == -1
    assert empty.members == ()
    assert foreign_bounded.evidence_high_watermark == 0
    assert foreign_bounded.members == ()
    assert foreign_bounded.snapshot_id != empty.snapshot_id


def test_snapshot_alias_retry_precedes_late_backfill_and_new_key_sees_it() -> None:
    store = InMemoryEvidenceStore()
    spec = make_spec()
    first_record = store.append(make_event(idempotency_key="first"))
    first = store.seal_evidence_snapshot(spec, idempotency_key="seal")
    backfill = store.append(
        make_event(
            observed_at=INSTANT - timedelta(days=1),
            idempotency_key="backfill",
        )
    )

    retry = store.seal_evidence_snapshot(spec, idempotency_key="seal")
    second = store.seal_evidence_snapshot(spec, idempotency_key="seal-2")

    assert retry is first
    assert tuple(member.evidence_id for member in first.members) == (
        first_record.evidence_id,
    )
    assert tuple(member.evidence_id for member in second.members) == (
        backfill.evidence_id,
        first_record.evidence_id,
    )
    assert second.evidence_high_watermark == 1
    assert store.get_evidence_snapshot_evidence(spec.scope, first.snapshot_id) == (
        first_record,
    )


def test_snapshot_idempotency_aliases_and_conflicts_are_scoped() -> None:
    store = InMemoryEvidenceStore()
    spec = make_spec()
    first = store.seal_evidence_snapshot(spec, idempotency_key="shared")
    alias = store.seal_evidence_snapshot(spec, idempotency_key="alias")

    assert alias is first
    assert len(store._snapshot_by_id) == 1
    assert len(store._snapshot_by_idempotency_key) == 2
    with pytest.raises(EvidenceSnapshotConflictError, match="idempotency"):
        store.seal_evidence_snapshot(
            make_spec(cutoff=INSTANT + timedelta(seconds=1)),
            idempotency_key="shared",
        )
    foreign = store.seal_evidence_snapshot(
        make_spec(scope=make_scope("2")),
        idempotency_key="shared",
    )
    assert foreign.spec.scope == make_scope("2")


def test_snapshot_lookup_hides_cross_scope_existence() -> None:
    store = InMemoryEvidenceStore()
    snapshot = store.seal_evidence_snapshot(make_spec(), idempotency_key="seal")
    foreign_scope = make_scope("2")

    for snapshot_id in (snapshot.snapshot_id, "esnap_missing"):
        with pytest.raises(
            EvidenceSnapshotNotFoundError,
            match="evidence snapshot .* was not found",
        ):
            store.get_evidence_snapshot(foreign_scope, snapshot_id)
    assert store.get_evidence_snapshot(snapshot.spec.scope, snapshot.snapshot_id) is (
        snapshot
    )


def test_concurrent_appends_and_identical_seals_converge_without_gaps() -> None:
    store = InMemoryEvidenceStore()
    scope = make_scope()
    events = tuple(
        make_event(
            scope=scope if index % 2 == 0 else make_scope("2"),
            sequence_no=index,
            idempotency_key=f"event-{index}",
        )
        for index in range(24)
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = tuple(executor.map(store.append, events))

    assert len(records) == 24
    assert sorted(store._ingest_order_by_evidence_id.values()) == list(range(24))
    assert store._next_ingest_order == 24

    spec = make_spec(scope=scope)
    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = tuple(
            executor.map(
                lambda _: store.seal_evidence_snapshot(
                    spec,
                    idempotency_key="concurrent-seal",
                ),
                range(16),
            )
        )
    assert all(snapshot is snapshots[0] for snapshot in snapshots)
    assert snapshots[0].evidence_high_watermark == 23


def test_snapshot_hash_collision_does_not_leave_a_loser_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    store = InMemoryEvidenceStore()
    spec = make_spec()
    store.append(make_event())
    monkeypatch.setattr(store_module.hashlib, "sha256", lambda _: FixedDigest())
    first = store.seal_evidence_snapshot(spec, idempotency_key="first")
    store.append(make_event(scope=make_scope("2"), idempotency_key="foreign"))

    with pytest.raises(EvidenceSnapshotConflictError, match="collision"):
        store.seal_evidence_snapshot(spec, idempotency_key="loser")

    assert first.snapshot_id == "esnap_" + "0" * 24
    assert (spec.scope, "loser") not in store._snapshot_by_idempotency_key
    assert len(store._snapshot_by_id) == 1
