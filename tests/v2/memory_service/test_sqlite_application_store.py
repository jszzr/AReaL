# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import inspect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from areal.v2.memory_service import sqlite_store as sqlite_store_module
from areal.v2.memory_service.application_store import MemoryApplicationStore
from areal.v2.memory_service.application_types import (
    MemoryApplicationProposal,
    MemoryApplicationUpdateProposal,
)
from areal.v2.memory_service.errors import (
    MemoryApplicationConflictError,
    MemoryApplicationNotFoundError,
    MemoryApplicationRootConflictError,
    MemoryApplicationRootNotFoundError,
    MemoryApplicationStaleSnapshotError,
    MemoryPersistenceCorruptionError,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_types import ReleaseManifest
from areal.v2.memory_service.snapshot_types import EvidenceSnapshotSpec
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

_BASE = datetime(2026, 7, 12, tzinfo=UTC)
_VERSION = "a" * 64


def _append(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    label: str,
    payload: str,
    seconds: int,
) -> EvidenceRecord:
    return store.append(
        EvidenceEvent(
            scope=scope,
            session_id="application-session",
            run_id="application-run",
            sequence_no=seconds,
            kind=EvidenceKind.FEEDBACK,
            payload=payload,
            observed_at=_BASE + timedelta(seconds=seconds),
            idempotency_key=f"evidence-{label}",
        )
    )


def _fixture(tmp_path):
    path = tmp_path / "memory-application.sqlite3"
    store = SQLiteMemoryStore(path)
    scope = MemoryScope("tenant-1", "agent-memory", "subject-1")
    genesis_evidence = _append(
        store,
        scope,
        label="genesis",
        payload="project-abc234 = ABCDE",
        seconds=0,
    )
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=genesis_evidence.event.payload,
            evidence_ids=(genesis_evidence.evidence_id,),
            idempotency_key="genesis-candidate",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="genesis-revision",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="genesis-release",
    )
    root = store.register_memory_application_root(scope, release.release_id)
    return path, store, scope, revision, release, root


def _seal(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    label: str,
):
    return store.seal_evidence_snapshot(
        EvidenceSnapshotSpec(
            scope=scope,
            allowed_kinds=(EvidenceKind.FEEDBACK,),
            cutoff=_BASE + timedelta(seconds=1000),
        ),
        idempotency_key=f"snapshot-{label}",
    )


def _proposal(
    *,
    scope: MemoryScope,
    snapshot_id: str,
    base_release_id: str,
    parent_revision_id: str,
    evidence_id: str,
    label: str,
    value: str = "FGHJK",
) -> MemoryApplicationProposal:
    return MemoryApplicationProposal(
        scope=scope,
        source_snapshot_id=snapshot_id,
        source_base_release_id=base_release_id,
        projector_id="provenance-policy-input-v2",
        projector_version_sha256=_VERSION,
        policy_id="verified-chain-consensus-v1",
        policy_version_sha256="b" * 64,
        policy_input_sha256="c" * 64,
        decision_sha256="d" * 64,
        policy_context='{"profile":"v1"}',
        updates=(
            MemoryApplicationUpdateProposal(
                content=f"project-abc234 = {value}",
                evidence_ids=(evidence_id,),
                operation=RevisionOperation.SUPERSEDE,
                parent_revision_id=parent_revision_id,
            ),
        ),
        idempotency_key=f"application-{label}",
    )


def _multi_update_proposal(
    *,
    scope: MemoryScope,
    snapshot_id: str,
    base_release_id: str,
    parent_revision_id: str,
    supersede_evidence_id: str,
    add_evidence_id: str,
    label: str,
) -> MemoryApplicationProposal:
    return MemoryApplicationProposal(
        scope=scope,
        source_snapshot_id=snapshot_id,
        source_base_release_id=base_release_id,
        projector_id="provenance-policy-input-v2",
        projector_version_sha256=_VERSION,
        policy_id="verified-chain-consensus-v1",
        policy_version_sha256="b" * 64,
        policy_input_sha256="c" * 64,
        decision_sha256="d" * 64,
        policy_context='{"profile":"v1"}',
        updates=(
            MemoryApplicationUpdateProposal(
                content="project-abc234 = FGHJK",
                evidence_ids=(supersede_evidence_id,),
                operation=RevisionOperation.SUPERSEDE,
                parent_revision_id=parent_revision_id,
            ),
            MemoryApplicationUpdateProposal(
                content="project-def567 = LMNPQ",
                evidence_ids=(add_evidence_id,),
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
            ),
        ),
        idempotency_key=f"multi-application-{label}",
    )


def _application_count(path) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM memory_applications").fetchone()
    assert row is not None
    return row[0]


def _commit_linear_applications(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    root_revision,
    root_release,
    snapshot,
    evidence: EvidenceRecord,
    depth: int,
):
    applications = []
    parent_revision_id = root_revision.revision_id
    base_release_id = root_release.release_id
    for index in range(depth):
        application = store.commit_memory_application(
            _proposal(
                scope=scope,
                snapshot_id=snapshot.snapshot_id,
                base_release_id=base_release_id,
                parent_revision_id=parent_revision_id,
                evidence_id=evidence.evidence_id,
                label=f"linear-{index}",
                value="FGHJK" if index % 2 == 0 else "LMNPQ",
            )
        )
        applications.append(application)
        parent_revision_id = application.applied_updates[0].revision_id
        base_release_id = application.result_release_id
    return tuple(applications)


def test_atomic_application_round_trips_through_all_exact_indexes(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="learned",
        payload="verified project code",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="first")
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=evidence.evidence_id,
        label="first",
    )

    application = store.commit_memory_application(proposal)

    assert application.application_order == 0
    assert application.proposal == proposal
    assert application.source_snapshot_content_sha256 == snapshot.content_hash
    assert application.base_release_content_sha256 == root.release_content_sha256
    assert application.result_release_id != root_release.release_id
    assert len(application.applied_updates) == 1
    update = application.applied_updates[0]
    assert update.parent_revision_id == root_revision.revision_id
    assert tuple(item.evidence_id for item in update.grounding) == (
        evidence.evidence_id,
    )
    assert store.get_memory_application(scope, application.application_id) == (
        application
    )
    assert (
        store.get_memory_application_for_revision(scope, update.revision_id)
        == application
    )
    assert (
        store.get_memory_application_for_release(scope, application.result_release_id)
        == application
    )
    assert store.get_memory_application_root(scope) == root


def test_application_replay_view_round_trips_root_and_lineage(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, root = _fixture(tmp_path)
    root_view = store.get_memory_application_replay(scope, root_release.release_id)
    assert root_view.root == root
    assert root_view.root_release == root_release
    assert root_view.root_revisions == (root_revision,)
    assert root_view.steps == ()
    assert root_view.target_release == root_release
    assert root_view.target_revisions == (root_revision,)

    evidence = _append(
        store,
        scope,
        label="replay-lineage",
        payload="verified replay evidence",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="replay-lineage")
    applications = _commit_linear_applications(
        store,
        scope,
        root_revision=root_revision,
        root_release=root_release,
        snapshot=snapshot,
        evidence=evidence,
        depth=3,
    )

    view = store.get_memory_application_replay(
        scope,
        applications[-1].result_release_id,
    )

    assert view.root == root
    assert tuple(step.application for step in view.steps) == applications
    assert all(step.source_snapshot == snapshot for step in view.steps)
    assert all(
        tuple(record.evidence_id for record in step.source_evidence)
        == tuple(member.evidence_id for member in snapshot.members)
        for step in view.steps
    )
    assert view.target_release.release_id == applications[-1].result_release_id
    assert view.target_revisions[-1].revision_id == (
        applications[-1].applied_updates[0].revision_id
    )
    with pytest.raises(ValueError, match="replay step commitments do not agree"):
        dataclasses.replace(
            view.steps[0],
            source_evidence=tuple(reversed(view.steps[0].source_evidence)),
        )
    with pytest.raises(
        ValueError,
        match="replay steps do not form one forward lineage",
    ):
        dataclasses.replace(view, steps=(view.steps[0], view.steps[0]))


def test_application_replay_loads_integrity_validated_state_once(
    tmp_path,
    monkeypatch,
) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="bulk-replay",
        payload="verified bulk replay evidence",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="bulk-replay")
    applications = _commit_linear_applications(
        store,
        scope,
        root_revision=root_revision,
        root_release=root_release,
        snapshot=snapshot,
        evidence=evidence,
        depth=24,
    )
    calls = {"application": 0, "evidence": 0, "release": 0}
    original_application = sqlite_store_module._load_application_state
    original_evidence = sqlite_store_module._load_evidence_snapshot_state
    original_release = sqlite_store_module._load_release_snapshot

    def counted_application(cursor):
        calls["application"] += 1
        return original_application(cursor)

    def counted_evidence(cursor):
        calls["evidence"] += 1
        return original_evidence(cursor)

    def counted_release(cursor):
        calls["release"] += 1
        return original_release(cursor)

    monkeypatch.setattr(
        sqlite_store_module,
        "_load_application_state",
        counted_application,
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_load_evidence_snapshot_state",
        counted_evidence,
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_load_release_snapshot",
        counted_release,
    )

    view = store.get_memory_application_replay(
        scope,
        applications[-1].result_release_id,
    )

    assert len(view.steps) == 24
    assert all(
        step.source_evidence is view.steps[0].source_evidence for step in view.steps
    )
    current_release_id = root_release.release_id
    for step, application in zip(view.steps, applications, strict=True):
        assert step.application == application
        assert application.proposal.source_base_release_id == current_release_id
        assert tuple(
            (record.evidence_id, record.content_hash)
            for record in step.source_evidence
        ) == tuple(
            (member.evidence_id, member.evidence_content_hash)
            for member in step.source_snapshot.members
        )
        assert tuple(item.revision_id for item in step.result_revisions) == (
            application.result_revision_ids
        )
        current_release_id = step.result_release.release_id
    assert calls == {"application": 1, "evidence": 1, "release": 1}


def test_application_replay_rejects_absent_root_and_non_lineage_release(
    tmp_path,
) -> None:
    empty_store = SQLiteMemoryStore(tmp_path / "replay-without-root.sqlite3")
    empty_scope = MemoryScope("tenant-empty", "agent-memory", "subject-empty")
    with pytest.raises(MemoryApplicationRootNotFoundError):
        empty_store.get_memory_application_replay(empty_scope, "rel_missing")

    _path, store, scope, _root_revision, _root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="generic-release",
        payload="project-def567 = LMNPQ",
        seconds=2,
    )
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=evidence.event.payload,
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="generic-release-candidate",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="generic-release-revision",
        )
    )
    generic_release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="generic-release",
    )

    for target_release_id in ("rel_missing", generic_release.release_id):
        with pytest.raises(MemoryApplicationNotFoundError):
            store.get_memory_application_replay(scope, target_release_id)


def test_application_replay_is_scope_local_across_global_order_gaps(tmp_path) -> None:
    _path, store, scope_a, revision_a, release_a, _root_a = _fixture(tmp_path)
    scope_b = MemoryScope("tenant-2", "agent-memory", "subject-2")
    genesis_b = _append(
        store,
        scope_b,
        label="scope-b-genesis",
        payload="project-abc234 = ABCDE",
        seconds=0,
    )
    candidate_b = store.append_candidate(
        CandidateProposal(
            scope=scope_b,
            content=genesis_b.event.payload,
            evidence_ids=(genesis_b.evidence_id,),
            idempotency_key="scope-b-genesis-candidate",
        )
    )
    revision_b = store.append_revision(
        RevisionProposal(
            scope=scope_b,
            candidate_id=candidate_b.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="scope-b-genesis-revision",
        )
    )
    release_b = store.append_release(
        ReleaseManifest(scope=scope_b, revision_ids=(revision_b.revision_id,)),
        idempotency_key="scope-b-genesis-release",
    )
    store.register_memory_application_root(scope_b, release_b.release_id)
    evidence_a = _append(
        store,
        scope_a,
        label="scope-a-learning",
        payload="scope A verified evidence",
        seconds=1,
    )
    evidence_b = _append(
        store,
        scope_b,
        label="scope-b-learning",
        payload="scope B verified evidence",
        seconds=1,
    )
    snapshot_a = _seal(store, scope_a, label="scope-a")
    snapshot_b = _seal(store, scope_b, label="scope-b")
    first_a = store.commit_memory_application(
        _proposal(
            scope=scope_a,
            snapshot_id=snapshot_a.snapshot_id,
            base_release_id=release_a.release_id,
            parent_revision_id=revision_a.revision_id,
            evidence_id=evidence_a.evidence_id,
            label="scope-a-first",
        )
    )
    only_b = store.commit_memory_application(
        _proposal(
            scope=scope_b,
            snapshot_id=snapshot_b.snapshot_id,
            base_release_id=release_b.release_id,
            parent_revision_id=revision_b.revision_id,
            evidence_id=evidence_b.evidence_id,
            label="scope-b-only",
        )
    )
    second_a = store.commit_memory_application(
        _proposal(
            scope=scope_a,
            snapshot_id=snapshot_a.snapshot_id,
            base_release_id=first_a.result_release_id,
            parent_revision_id=first_a.applied_updates[0].revision_id,
            evidence_id=evidence_a.evidence_id,
            label="scope-a-second",
            value="LMNPQ",
        )
    )

    view_a = store.get_memory_application_replay(
        scope_a,
        second_a.result_release_id,
    )

    assert tuple(step.application.application_order for step in view_a.steps) == (
        0,
        2,
    )
    assert all(step.application != only_b for step in view_a.steps)
    view_b = store.get_memory_application_replay(scope_b, only_b.result_release_id)
    with pytest.raises(
        ValueError,
        match="replay steps do not form one forward lineage",
    ):
        dataclasses.replace(view_a, steps=(view_b.steps[0],))
    with pytest.raises(MemoryApplicationNotFoundError):
        store.get_memory_application_replay(scope_b, second_a.result_release_id)


def test_application_replay_fails_closed_on_corrupt_lineage_binding(tmp_path) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="corrupt-replay",
        payload="verified corrupt replay evidence",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="corrupt-replay")
    application = store.commit_memory_application(
        _proposal(
            scope=scope,
            snapshot_id=snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=evidence.evidence_id,
            label="corrupt-replay",
        )
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE memory_applications SET base_release_id = result_release_id "
            "WHERE application_id = ?",
            (application.application_id,),
        )

    with pytest.raises(MemoryPersistenceCorruptionError):
        store.get_memory_application_replay(scope, application.result_release_id)


def test_sqlite_application_api_matches_the_optional_store_protocol() -> None:
    for method_name in (
        "register_memory_application_root",
        "get_memory_application_root",
        "commit_memory_application",
        "get_memory_application",
        "get_memory_application_for_revision",
        "get_memory_application_for_release",
        "get_memory_application_replay",
    ):
        assert inspect.signature(getattr(SQLiteMemoryStore, method_name)) == (
            inspect.signature(getattr(MemoryApplicationStore, method_name))
        )


def test_stale_snapshot_rejects_before_any_history_or_application_write(
    tmp_path,
) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="source",
        payload="verified source",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="stale")
    _append(
        store,
        scope,
        label="arrived-after-seal",
        payload="conflicting failure",
        seconds=2,
    )
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=evidence.evidence_id,
        label="stale",
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    with pytest.raises(MemoryApplicationStaleSnapshotError):
        store.commit_memory_application(proposal)

    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before
    assert _application_count(path) == 0


def test_exact_retry_succeeds_after_later_evidence_arrives(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="retry-source",
        payload="verified source",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="retry")
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=evidence.evidence_id,
        label="retry",
    )
    first = store.commit_memory_application(proposal)
    _append(
        store,
        scope,
        label="post-commit",
        payload="later evidence",
        seconds=2,
    )

    assert store.commit_memory_application(proposal) == first


def test_other_scope_evidence_does_not_starve_a_valid_application(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="tenant-a-source",
        payload="verified tenant A source",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="tenant-a-source")
    other_scope = MemoryScope("tenant-2", "agent-memory", "subject-2")
    _append(
        store,
        other_scope,
        label="tenant-b-unrelated",
        payload="unrelated tenant B evidence",
        seconds=2,
    )
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=evidence.evidence_id,
        label="tenant-a-source",
    )

    application = store.commit_memory_application(proposal)

    assert application.proposal == proposal


@pytest.mark.parametrize(
    "stage",
    (
        "after_candidate",
        "after_revision",
        "after_release",
        "after_application",
        "after_edge",
        "after_readback",
    ),
)
def test_every_application_write_phase_rolls_back_as_one_transaction(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label=f"fault-{stage}",
        payload="verified source",
        seconds=1,
    )
    snapshot = _seal(store, scope, label=f"fault-{stage}")
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=evidence.evidence_id,
        label=f"fault-{stage}",
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    def fail(selected: str) -> None:
        if selected == stage:
            raise RuntimeError(f"injected failure at {stage}")

    monkeypatch.setattr(
        sqlite_store_module, "_application_transaction_fault_hook", fail
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        store.commit_memory_application(proposal)

    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before
    assert _application_count(path) == 0


def test_atomic_application_supports_supersede_and_add_in_one_release(
    tmp_path,
) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    supersede_evidence = _append(
        store,
        scope,
        label="multi-supersede",
        payload="verified replacement",
        seconds=1,
    )
    add_evidence = _append(
        store,
        scope,
        label="multi-add",
        payload="verified addition",
        seconds=2,
    )
    snapshot = _seal(store, scope, label="multi-success")
    proposal = _multi_update_proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        supersede_evidence_id=supersede_evidence.evidence_id,
        add_evidence_id=add_evidence.evidence_id,
        label="success",
    )

    application = store.commit_memory_application(proposal)

    assert len(application.applied_updates) == 2
    assert tuple(item.release_position for item in application.applied_updates) == (
        0,
        1,
    )
    assert tuple(item.operation for item in application.applied_updates) == (
        RevisionOperation.SUPERSEDE,
        RevisionOperation.ADD,
    )
    assert tuple(item.generation for item in application.applied_updates) == (1, 0)
    assert application.result_revision_ids == tuple(
        item.revision_id for item in application.applied_updates
    )


@pytest.mark.parametrize("stage", ("after_candidate", "after_revision", "after_edge"))
def test_failure_on_second_multi_update_child_or_edge_rolls_back_everything(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    supersede_evidence = _append(
        store,
        scope,
        label=f"multi-fault-{stage}-supersede",
        payload="verified replacement",
        seconds=1,
    )
    add_evidence = _append(
        store,
        scope,
        label=f"multi-fault-{stage}-add",
        payload="verified addition",
        seconds=2,
    )
    snapshot = _seal(store, scope, label=f"multi-fault-{stage}")
    proposal = _multi_update_proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        supersede_evidence_id=supersede_evidence.evidence_id,
        add_evidence_id=add_evidence.evidence_id,
        label=f"fault-{stage}",
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )
    hits = 0

    def fail_on_second(selected: str) -> None:
        nonlocal hits
        if selected == stage:
            hits += 1
            if hits == 2:
                raise RuntimeError(f"second {stage} failed")

    monkeypatch.setattr(
        sqlite_store_module,
        "_application_transaction_fault_hook",
        fail_on_second,
    )
    with pytest.raises(RuntimeError, match="second"):
        store.commit_memory_application(proposal)

    assert hits == 2
    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before
    assert _application_count(path) == 0


def test_second_application_must_extend_a_committed_result(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    first_evidence = _append(
        store,
        scope,
        label="generation-one",
        payload="verified generation one",
        seconds=1,
    )
    first_snapshot = _seal(store, scope, label="generation-one")
    first = store.commit_memory_application(
        _proposal(
            scope=scope,
            snapshot_id=first_snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=first_evidence.evidence_id,
            label="generation-one",
        )
    )
    first_revision = first.applied_updates[0]
    second_evidence = _append(
        store,
        scope,
        label="generation-two",
        payload="verified generation two",
        seconds=2,
    )
    second_snapshot = _seal(store, scope, label="generation-two")
    second = store.commit_memory_application(
        _proposal(
            scope=scope,
            snapshot_id=second_snapshot.snapshot_id,
            base_release_id=first.result_release_id,
            parent_revision_id=first_revision.revision_id,
            evidence_id=second_evidence.evidence_id,
            label="generation-two",
            value="LMNPQ",
        )
    )

    assert second.application_order == 1
    assert second.proposal.source_base_release_id == first.result_release_id
    assert second.applied_updates[0].generation == 2

    with pytest.raises(MemoryApplicationNotFoundError):
        store.get_memory_application_for_release(scope, root_release.release_id)


def test_stale_base_cannot_fork_the_scope_after_head_advances(tmp_path) -> None:
    _path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    first_evidence = _append(
        store,
        scope,
        label="head-first",
        payload="verified first head",
        seconds=1,
    )
    first_snapshot = _seal(store, scope, label="head-first")
    store.commit_memory_application(
        _proposal(
            scope=scope,
            snapshot_id=first_snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=first_evidence.evidence_id,
            label="head-first",
        )
    )
    stale_evidence = _append(
        store,
        scope,
        label="stale-base-fork",
        payload="attempted stale branch",
        seconds=2,
    )
    stale_snapshot = _seal(store, scope, label="stale-base-fork")
    stale_proposal = _proposal(
        scope=scope,
        snapshot_id=stale_snapshot.snapshot_id,
        base_release_id=root_release.release_id,
        parent_revision_id=root_revision.revision_id,
        evidence_id=stale_evidence.evidence_id,
        label="stale-base-fork",
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    with pytest.raises(MemoryApplicationConflictError, match="canonical head"):
        store.commit_memory_application(stale_proposal)

    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before


def test_two_writers_from_one_head_commit_exactly_one_application(tmp_path) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="concurrent-head",
        payload="shared source evidence",
        seconds=1,
    )
    snapshot = _seal(store, scope, label="concurrent-head")
    stores = (SQLiteMemoryStore(path), SQLiteMemoryStore(path))
    proposals = (
        _proposal(
            scope=scope,
            snapshot_id=snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=evidence.evidence_id,
            label="concurrent-a",
            value="FGHJK",
        ),
        _proposal(
            scope=scope,
            snapshot_id=snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=evidence.evidence_id,
            label="concurrent-b",
            value="LMNPQ",
        ),
    )
    barrier = Barrier(2)

    def commit(index: int):
        barrier.wait()
        return stores[index].commit_memory_application(proposals[index])

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(commit, index) for index in range(2))
        for future in futures:
            try:
                successes.append(future.result(timeout=10))
            except Exception as error:
                failures.append(error)

    assert len(successes) == 1
    assert len(failures) == 1
    assert type(failures[0]) is MemoryApplicationConflictError
    assert _application_count(path) == 1
    assert len(store.list_candidates(scope)) == 2
    assert len(store.list_revisions(scope)) == 2
    assert len(store.list_releases(scope)) == 2


def test_root_is_one_explicit_immutable_release_per_scope(tmp_path) -> None:
    _path, store, scope, _revision, release, root = _fixture(tmp_path)
    assert store.register_memory_application_root(scope, release.release_id) == root
    empty_release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=()),
        idempotency_key="untrusted-alternate-root",
    )
    with pytest.raises(MemoryApplicationRootConflictError):
        store.register_memory_application_root(scope, empty_release.release_id)


def test_uncommitted_generic_release_cannot_be_an_application_base(tmp_path) -> None:
    _path, store, scope, root_revision, _root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label="generic-base",
        payload="generic update",
        seconds=1,
    )
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content="project-abc234 = FGHJK",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="generic-candidate",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=root_revision.revision_id,
            idempotency_key="generic-revision",
        )
    )
    generic_release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="generic-release",
    )
    source = _append(
        store,
        scope,
        label="generic-child",
        payload="verified child",
        seconds=2,
    )
    snapshot = _seal(store, scope, label="generic-child")
    proposal = _proposal(
        scope=scope,
        snapshot_id=snapshot.snapshot_id,
        base_release_id=generic_release.release_id,
        parent_revision_id=revision.revision_id,
        evidence_id=source.evidence_id,
        label="generic-child",
    )

    with pytest.raises(MemoryApplicationConflictError, match="neither"):
        store.commit_memory_application(proposal)


@pytest.mark.parametrize(
    "mutation",
    (
        "root_canonical",
        "root_content_hash",
        "application_canonical",
        "application_source_hash",
        "application_order",
        "edge_revision_hash",
        "edge_binding_hash",
        "edge_ordinal",
        "cross_scope_edge",
        "missing_edge",
    ),
)
def test_application_loader_rejects_durable_root_application_or_edge_drift(
    tmp_path,
    mutation: str,
) -> None:
    path, store, scope, root_revision, root_release, _root = _fixture(tmp_path)
    evidence = _append(
        store,
        scope,
        label=f"tamper-{mutation}",
        payload="verified source",
        seconds=1,
    )
    snapshot = _seal(store, scope, label=f"tamper-{mutation}")
    application = store.commit_memory_application(
        _proposal(
            scope=scope,
            snapshot_id=snapshot.snapshot_id,
            base_release_id=root_release.release_id,
            parent_revision_id=root_revision.revision_id,
            evidence_id=evidence.evidence_id,
            label=f"tamper-{mutation}",
        )
    )
    with sqlite3.connect(path) as connection:
        if mutation == "root_canonical":
            connection.execute(
                "UPDATE memory_application_roots SET canonical = ?",
                (b"{}",),
            )
        elif mutation == "root_content_hash":
            connection.execute(
                "UPDATE memory_application_roots SET content_hash = ?",
                ("f" * 64,),
            )
        elif mutation == "application_canonical":
            connection.execute(
                "UPDATE memory_applications SET canonical = ?",
                (b"{}",),
            )
        elif mutation == "application_source_hash":
            connection.execute(
                "UPDATE memory_applications SET source_snapshot_content_hash = ?",
                ("f" * 64,),
            )
        elif mutation == "application_order":
            connection.execute("UPDATE memory_applications SET application_order = 7")
        elif mutation == "edge_revision_hash":
            connection.execute(
                "UPDATE memory_application_revisions SET revision_content_hash = ?",
                ("f" * 64,),
            )
        elif mutation == "edge_binding_hash":
            connection.execute(
                "UPDATE memory_application_revisions SET binding_hash = ?",
                ("f" * 64,),
            )
        elif mutation == "edge_ordinal":
            connection.execute("UPDATE memory_application_revisions SET ordinal = 1")
        elif mutation == "cross_scope_edge":
            cursor = connection.execute(
                "INSERT INTO memory_scopes "
                "(tenant_id, namespace, subject_id) VALUES (?, ?, ?)",
                ("foreign-tenant", "agent-memory", "foreign-subject"),
            )
            connection.execute(
                "UPDATE memory_application_revisions SET scope_id = ?",
                (cursor.lastrowid,),
            )
        elif mutation == "missing_edge":
            connection.execute("DELETE FROM memory_application_revisions")

    with pytest.raises(MemoryPersistenceCorruptionError):
        store.get_memory_application(scope, application.application_id)
