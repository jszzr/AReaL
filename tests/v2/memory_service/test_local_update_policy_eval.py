# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from examples.memory_service import local_update_policy_eval as policy

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    EvidenceSnapshotSpec,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

_BASE = datetime(2026, 7, 8, tzinfo=UTC)
_TARGET_KEY = "project-abc234"
_NEW_KEY = "project-def567"
_OLD = "ABCDE"
_CURRENT = "FGHJK"
_WRONG = "LMNPQ"
_NEW_VALUE = "RSTUV"


def _append(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    label: str,
    seconds: int,
    sequence_no: int,
    kind: EvidenceKind,
    payload: str,
):
    return store.append(
        EvidenceEvent(
            scope=scope,
            session_id="capture-session",
            run_id="capture-run",
            sequence_no=sequence_no,
            kind=kind,
            payload=payload,
            observed_at=_BASE + timedelta(seconds=seconds),
            idempotency_key=f"evidence-{label}",
        )
    )


def _fixture(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "policy.sqlite3")
    scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="local-update-policy-v1",
        subject_id="opaque-subject-for-tests",
    )
    old = _append(
        store,
        scope,
        label="old",
        seconds=0,
        sequence_no=0,
        kind=EvidenceKind.USER_MESSAGE,
        payload=f"{_TARGET_KEY} = {_OLD}",
    )
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=old.event.payload,
            evidence_ids=(old.evidence_id,),
            idempotency_key="candidate-old",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="revision-old",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="release-old",
    )
    base_memory = policy.BaseMemoryV1(
        key=_TARGET_KEY,
        value=_OLD,
        memory_id=revision.memory_id,
        revision_id=revision.revision_id,
        generation=revision.generation,
    )
    new_key_feedback = _append(
        store,
        scope,
        label="new-key-feedback",
        seconds=30,
        sequence_no=1,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_NEW_KEY} = {_NEW_VALUE}",
    )
    current_feedback = _append(
        store,
        scope,
        label="current-feedback",
        seconds=60,
        sequence_no=2,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    malformed_feedback = _append(
        store,
        scope,
        label="malformed-feedback",
        seconds=90,
        sequence_no=3,
        kind=EvidenceKind.FEEDBACK,
        payload=f"correction: {_TARGET_KEY} should be {_WRONG}",
    )
    later_noise = _append(
        store,
        scope,
        label="later-noise",
        seconds=120,
        sequence_no=4,
        kind=EvidenceKind.USER_MESSAGE,
        payload=f"{_TARGET_KEY} = {_WRONG}",
    )
    records = (
        old,
        new_key_feedback,
        current_feedback,
        malformed_feedback,
        later_noise,
    )
    value = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=release.release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-fixture",
    )
    return store, scope, records, base_memory, value


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_mapping_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_mapping_keys(child)}
    return set()


def _release_from_record(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    record,
    *,
    label: str,
):
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=record.event.payload,
            evidence_ids=(record.evidence_id,),
            idempotency_key=f"candidate-{label}",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key=f"revision-{label}",
        )
    )
    return store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key=f"release-{label}",
    )


def test_policy_input_is_closed_opaque_cutoff_projection(tmp_path) -> None:
    store, scope, records, base, value = _fixture(tmp_path)

    assert tuple(item.evidence_id for item in value.evidence) == tuple(
        record.evidence_id for record in records
    )
    assert value.base_memories == (base,)
    assert value.policy_scope_token.startswith("scope_")
    assert len(value.policy_scope_token) == 70
    assert "opaque-subject-for-tests" not in value.policy_scope_token
    assert value.cutoff_utc == "2026-07-08T00:02:30+00:00"
    snapshot = store.get_evidence_snapshot(scope, value.evidence_snapshot_id)
    assert value.evidence_snapshot_content_hash == snapshot.content_hash
    assert value.evidence_snapshot_members == snapshot.members
    assert tuple(item.evidence_id for item in value.evidence) == tuple(
        item.evidence_id for item in value.evidence_snapshot_members
    )
    release = store.get_release(scope, value.base_release_id)
    assert value.base_release_content_sha256 == release.content_hash
    assert value.base_memories[0].revision_id in release.manifest.revision_ids

    decoded = json.loads(policy.policy_input_wire_v1(value))
    assert set(decoded) == {
        "base_memories",
        "base_release_content_sha256",
        "base_release_id",
        "cutoff_utc",
        "evidence",
        "evidence_snapshot_content_hash",
        "evidence_snapshot_id",
        "evidence_snapshot_members",
        "policy_scope_token",
        "schema_version",
    }
    forbidden = {
        "arm",
        "case",
        "case_index",
        "database_path",
        "expected",
        "future_query",
        "outcome",
        "query",
        "reward",
        "store",
        "truth",
        "utility",
    }
    assert _all_mapping_keys(decoded).isdisjoint(forbidden)
    # The correction value is legitimately present inside FEEDBACK.  The
    # invariant is absence of a separate scorer-truth channel.
    assert any(_CURRENT in item["payload"] for item in decoded["evidence"])
    assert {item["kind"] for item in decoded["evidence"]} <= set(
        policy.POLICY_EVIDENCE_KINDS
    )


def test_three_policies_separate_trusted_feedback_from_later_noise(tmp_path) -> None:
    _store, _scope, records, _base, value = _fixture(tmp_path)
    evidence_by_payload = {
        record.event.payload: record.evidence_id for record in records
    }

    trusted = policy.run_local_update_policy_v1("feedback_latest", value)
    noop = policy.run_local_update_policy_v1("noop", value)
    blind = policy.run_local_update_policy_v1("latest_any", value)

    trusted_by_key = {update.key: update for update in trusted.updates}
    assert trusted_by_key[_TARGET_KEY].value == _CURRENT
    assert trusted_by_key[_TARGET_KEY].evidence_ids == (
        evidence_by_payload[f"{_TARGET_KEY} = {_CURRENT}"],
    )
    assert trusted_by_key[_NEW_KEY].value == _NEW_VALUE
    assert noop.updates == ()
    blind_by_key = {update.key: update for update in blind.updates}
    assert blind_by_key[_TARGET_KEY].value == _WRONG
    assert blind_by_key[_TARGET_KEY].evidence_ids == (
        evidence_by_payload[f"{_TARGET_KEY} = {_WRONG}"],
    )
    with pytest.raises(policy.LocalUpdatePolicyError) as substitution:
        policy.validate_local_update_decision_v1(
            value,
            trusted,
            expected_policy="latest_any",
        )
    assert substitution.value.reason == "policy_mismatch"


def test_feedback_decision_distinguishes_add_and_grounded_supersede(tmp_path) -> None:
    _store, _scope, records, base, value = _fixture(tmp_path)
    evidence_by_payload = {
        record.event.payload: record.evidence_id for record in records
    }

    decision = policy.run_local_update_policy_v1("feedback_latest", value)
    by_key = {update.key: update for update in decision.updates}

    added = by_key[_NEW_KEY]
    assert added.operation == "add"
    assert added.parent_revision_id is None
    assert added.content == f"{_NEW_KEY} = {_NEW_VALUE}"
    assert added.evidence_ids == (evidence_by_payload[f"{_NEW_KEY} = {_NEW_VALUE}"],)

    superseded = by_key[_TARGET_KEY]
    assert superseded.operation == "supersede"
    assert superseded.parent_revision_id == base.revision_id
    assert superseded.content == f"{_TARGET_KEY} = {_CURRENT}"
    assert (
        policy.validate_local_update_decision_v1(
            value,
            decision,
            expected_policy="feedback_latest",
        )
        is decision
    )


def test_projection_and_commitments_are_deterministic(tmp_path) -> None:
    _store, scope, records, base, value = _fixture(tmp_path)
    replayed = policy.make_policy_input_v1(
        store=_store,
        scope=scope,
        base_release_id=value.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-fixture",
    )
    first = policy.run_local_update_policy_v1("feedback_latest", value)
    second = policy.run_local_update_policy_v1("feedback_latest", replayed)
    read_only = policy.project_policy_input_from_snapshot_v1(
        store=_store,
        scope=scope,
        base_release_id=value.base_release_id,
        evidence_snapshot_id=value.evidence_snapshot_id,
    )

    assert replayed == value
    assert read_only == value
    assert policy.policy_input_wire_v1(replayed) == policy.policy_input_wire_v1(value)
    assert policy.policy_input_sha256_v1(replayed) == policy.policy_input_sha256_v1(
        value
    )
    assert first == second
    assert policy.policy_decision_wire_v1(first) == policy.policy_decision_wire_v1(
        second
    )
    assert policy.policy_decision_sha256_v1(first) == policy.policy_decision_sha256_v1(
        second
    )
    assert policy.policy_input_sha256_v1(value) != policy.policy_decision_sha256_v1(
        first
    )
    assert len(policy.policy_input_sha256_v1(value)) == 64
    assert len(policy.policy_decision_sha256_v1(first)) == 64
    assert len(policy.policy_input_wire_v1(value)) == 2275
    assert len(policy.policy_decision_wire_v1(first)) == 509
    assert (
        policy.policy_input_sha256_v1(value)
        == "12fd986d834e3154a156ff74c6a3f56301651d269fdd397676b89d52ee92bd5f"
    )
    assert (
        policy.policy_decision_sha256_v1(first)
        == "876af90c5c746d3ecb1b4cf34183de5be3f8ee94f0360e8f5c6db7b4123b4c33"
    )


def test_make_never_lists_and_existing_snapshot_projection_never_seals(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, value = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsealed evidence enumeration is forbidden")

    monkeypatch.setattr(SQLiteMemoryStore, "list", forbidden)
    assert (
        policy.make_policy_input_v1(
            store=store,
            scope=scope,
            base_release_id=value.base_release_id,
            cutoff=_BASE + timedelta(seconds=150),
            evidence_snapshot_idempotency_key="snapshot-fixture",
        )
        == value
    )

    monkeypatch.setattr(SQLiteMemoryStore, "seal_evidence_snapshot", forbidden)
    assert (
        policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=value.base_release_id,
            evidence_snapshot_id=value.evidence_snapshot_id,
        )
        == value
    )


def test_late_backfill_is_frozen_by_old_key_and_included_by_new_key(tmp_path) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    backfill = _append(
        store,
        scope,
        label="late-backfill",
        seconds=-60,
        sequence_no=9,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )

    old_key = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=original.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-fixture",
    )
    new_key = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=original.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-after-backfill",
    )

    assert old_key == original
    assert backfill.evidence_id not in {
        item.evidence_id for item in old_key.evidence
    }
    assert backfill.evidence_id in {item.evidence_id for item in new_key.evidence}
    assert new_key.evidence_snapshot_id != old_key.evidence_snapshot_id
    assert new_key.evidence_snapshot_content_hash != (
        old_key.evidence_snapshot_content_hash
    )
    assert tuple(item.evidence_id for item in new_key.evidence) == tuple(
        item.evidence_id for item in new_key.evidence_snapshot_members
    )


def test_snapshot_projection_rejects_missing_foreign_or_wrong_predicate(
    tmp_path,
) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    foreign_scope = MemoryScope(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        subject_id="foreign-snapshot-subject",
    )
    foreign = store.seal_evidence_snapshot(
        EvidenceSnapshotSpec(
            scope=foreign_scope,
            allowed_kinds=(EvidenceKind.USER_MESSAGE, EvidenceKind.FEEDBACK),
            cutoff=_BASE + timedelta(seconds=150),
        ),
        idempotency_key="foreign-snapshot",
    )
    wrong_predicate = store.seal_evidence_snapshot(
        EvidenceSnapshotSpec(
            scope=scope,
            allowed_kinds=(EvidenceKind.OUTCOME,),
            cutoff=_BASE + timedelta(seconds=150),
        ),
        idempotency_key="wrong-predicate-snapshot",
    )

    for snapshot_id in (
        "esnap_" + "0" * 24,
        foreign.snapshot_id,
        wrong_predicate.snapshot_id,
    ):
        with pytest.raises(policy.LocalUpdatePolicyError) as error:
            policy.project_policy_input_from_snapshot_v1(
                store=store,
                scope=scope,
                base_release_id=original.base_release_id,
                evidence_snapshot_id=snapshot_id,
            )
        assert error.value.reason == "evidence_snapshot_invalid"


@pytest.mark.parametrize(
    ("label", "seconds", "kind"),
    (
        ("future-outcome-base", 100, EvidenceKind.OUTCOME),
        ("after-cutoff-base", 180, EvidenceKind.FEEDBACK),
    ),
)
def test_base_release_cannot_import_evidence_outside_bound_snapshot(
    tmp_path,
    label,
    seconds,
    kind,
) -> None:
    store, scope, _records, _base, _original = _fixture(tmp_path)
    leaked = _append(
        store,
        scope,
        label=label,
        seconds=seconds,
        sequence_no=11,
        kind=kind,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    leaked_release = _release_from_record(
        store,
        scope,
        leaked,
        label=label,
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.make_policy_input_v1(
            store=store,
            scope=scope,
            base_release_id=leaked_release.release_id,
            cutoff=_BASE + timedelta(seconds=150),
            evidence_snapshot_idempotency_key=f"snapshot-{label}",
        )

    assert error.value.reason == "base_release_invalid"


def test_base_release_cannot_use_backfill_ingested_after_bound_snapshot(
    tmp_path,
) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    backfill = _append(
        store,
        scope,
        label="post-seal-base-backfill",
        seconds=10,
        sequence_no=12,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    backfilled_release = _release_from_record(
        store,
        scope,
        backfill,
        label="post-seal-base-backfill",
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=backfilled_release.release_id,
            evidence_snapshot_id=original.evidence_snapshot_id,
        )

    assert error.value.reason == "base_release_invalid"


def test_base_release_cannot_hide_out_of_snapshot_evidence_in_parent_lineage(
    tmp_path,
) -> None:
    store, scope, records, _base, original = _fixture(tmp_path)
    leaked_parent_evidence = _append(
        store,
        scope,
        label="leaked-lineage-parent",
        seconds=100,
        sequence_no=13,
        kind=EvidenceKind.OUTCOME,
        payload=f"{_TARGET_KEY} = {_WRONG}",
    )
    parent_candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=leaked_parent_evidence.event.payload,
            evidence_ids=(leaked_parent_evidence.evidence_id,),
            idempotency_key="candidate-leaked-lineage-parent",
        )
    )
    parent = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=parent_candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="revision-leaked-lineage-parent",
        )
    )
    safe_child_evidence = records[2]
    child_candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=safe_child_evidence.event.payload,
            evidence_ids=(safe_child_evidence.evidence_id,),
            idempotency_key="candidate-safe-lineage-child",
        )
    )
    child = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=child_candidate.candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=parent.revision_id,
            idempotency_key="revision-safe-lineage-child",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(child.revision_id,)),
        idempotency_key="release-leaked-lineage",
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=release.release_id,
            evidence_snapshot_id=original.evidence_snapshot_id,
        )

    assert error.value.reason == "base_release_invalid"


def test_base_release_validates_leaked_middle_of_three_generation_lineage(
    tmp_path,
) -> None:
    store, scope, records, safe_root, original = _fixture(tmp_path)
    leaked_middle_evidence = _append(
        store,
        scope,
        label="leaked-lineage-middle",
        seconds=100,
        sequence_no=14,
        kind=EvidenceKind.OUTCOME,
        payload=f"{_TARGET_KEY} = {_WRONG}",
    )
    middle_candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=leaked_middle_evidence.event.payload,
            evidence_ids=(leaked_middle_evidence.evidence_id,),
            idempotency_key="candidate-leaked-lineage-middle",
        )
    )
    middle = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=middle_candidate.candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=safe_root.revision_id,
            idempotency_key="revision-leaked-lineage-middle",
        )
    )
    safe_leaf_evidence = records[2]
    leaf_candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=safe_leaf_evidence.event.payload,
            evidence_ids=(safe_leaf_evidence.evidence_id,),
            idempotency_key="candidate-safe-lineage-leaf",
        )
    )
    leaf = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=leaf_candidate.candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=middle.revision_id,
            idempotency_key="revision-safe-lineage-leaf",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(leaf.revision_id,)),
        idempotency_key="release-leaked-lineage-middle",
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=release.release_id,
            evidence_snapshot_id=original.evidence_snapshot_id,
        )

    assert error.value.reason == "base_release_invalid"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: dataclasses.replace(
            value,
            evidence_snapshot_members=value.evidence_snapshot_members[:-1],
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_snapshot_members=(
                *value.evidence_snapshot_members,
                value.evidence_snapshot_members[-1],
            ),
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_snapshot_members=tuple(
                reversed(value.evidence_snapshot_members)
            ),
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_snapshot_members=(
                dataclasses.replace(
                    value.evidence_snapshot_members[0],
                    evidence_content_hash="0" * 64,
                ),
                *value.evidence_snapshot_members[1:],
            ),
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_snapshot_content_hash="0" * 64,
        ),
    ),
)
def test_deleted_added_reordered_or_rehashed_snapshot_members_fail_closed(
    tmp_path,
    mutation,
) -> None:
    _store, _scope, _records, _base, value = _fixture(tmp_path)

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.policy_input_wire_v1(mutation(value))

    assert error.value.reason in {"closed_schema", "input_invariant"}


def test_legacy_policy_input_without_snapshot_binding_fails_closed(tmp_path) -> None:
    _store, _scope, _records, _base, value = _fixture(tmp_path)

    @dataclasses.dataclass(frozen=True, slots=True)
    class LegacyPolicyInputV1:
        schema_version: int
        policy_scope_token: str
        base_release_id: str
        base_release_content_sha256: str
        cutoff_utc: str
        evidence: tuple[policy.PolicyEvidenceV1, ...]
        base_memories: tuple[policy.BaseMemoryV1, ...]

    legacy = LegacyPolicyInputV1(
        schema_version=value.schema_version,
        policy_scope_token=value.policy_scope_token,
        base_release_id=value.base_release_id,
        base_release_content_sha256=value.base_release_content_sha256,
        cutoff_utc=value.cutoff_utc,
        evidence=value.evidence,
        base_memories=value.base_memories,
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.policy_input_wire_v1(legacy)  # type: ignore[arg-type]

    assert error.value.reason == "closed_schema"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda decision: dataclasses.replace(decision, input_sha256="0" * 64),
        lambda decision: dataclasses.replace(
            decision,
            updates=(
                dataclasses.replace(decision.updates[0], value=_WRONG),
                *decision.updates[1:],
            ),
        ),
        lambda decision: dataclasses.replace(
            decision,
            updates=(
                dataclasses.replace(
                    decision.updates[0], evidence_ids=("evd_" + "0" * 24,)
                ),
                *decision.updates[1:],
            ),
        ),
        lambda decision: dataclasses.replace(
            decision,
            updates=(
                dataclasses.replace(
                    decision.updates[0], parent_revision_id="rev_" + "0" * 24
                ),
                *decision.updates[1:],
            ),
        ),
    ),
)
def test_forged_policy_decisions_fail_closed(tmp_path, mutation) -> None:
    _store, _scope, _records, _base, value = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", value)
    forged = mutation(decision)

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.validate_local_update_decision_v1(
            value,
            forged,
            expected_policy="feedback_latest",
        )

    assert error.value.reason in {"decision_invariant", "decision_mismatch"}


def test_projection_filters_outcomes_future_records_and_foreign_scope(tmp_path) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    foreign_scope = MemoryScope(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        subject_id="foreign-subject",
    )
    foreign = _append(
        store,
        foreign_scope,
        label="foreign",
        seconds=20,
        sequence_no=0,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    before_cutoff_outcome = _append(
        store,
        scope,
        label="outcome-sentinel",
        seconds=100,
        sequence_no=5,
        kind=EvidenceKind.OUTCOME,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    after_cutoff = _append(
        store,
        scope,
        label="future-feedback-sentinel",
        seconds=180,
        sequence_no=6,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_CURRENT}",
    )
    foreign_candidate = store.append_candidate(
        CandidateProposal(
            scope=foreign_scope,
            content=foreign.event.payload,
            evidence_ids=(foreign.evidence_id,),
            idempotency_key="candidate-foreign",
        )
    )
    foreign_revision = store.append_revision(
        RevisionProposal(
            scope=foreign_scope,
            candidate_id=foreign_candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="revision-foreign",
        )
    )
    foreign_release = store.append_release(
        ReleaseManifest(
            scope=foreign_scope,
            revision_ids=(foreign_revision.revision_id,),
        ),
        idempotency_key="release-foreign",
    )

    projected = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=original.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-filtered",
    )
    projected_ids = {item.evidence_id for item in projected.evidence}
    assert foreign.evidence_id not in projected_ids
    assert before_cutoff_outcome.evidence_id not in projected_ids
    assert after_cutoff.evidence_id not in projected_ids
    assert {item.kind for item in projected.evidence} <= set(
        policy.POLICY_EVIDENCE_KINDS
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as release_error:
        policy.make_policy_input_v1(
            store=store,
            scope=scope,
            base_release_id=foreign_release.release_id,
            cutoff=_BASE + timedelta(seconds=150),
            evidence_snapshot_idempotency_key="snapshot-foreign-release",
        )
    assert release_error.value.reason == "base_release_invalid"


def test_base_release_projection_rejects_semantically_ungrounded_candidate(
    tmp_path,
) -> None:
    store, scope, records, _base, _original = _fixture(tmp_path)
    old = records[0]
    forged_candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=f"{_NEW_KEY} = {_NEW_VALUE}",
            evidence_ids=(old.evidence_id,),
            idempotency_key="candidate-ungrounded",
        )
    )
    forged_revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=forged_candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="revision-ungrounded",
        )
    )
    forged_release = store.append_release(
        ReleaseManifest(
            scope=scope,
            revision_ids=(forged_revision.revision_id,),
        ),
        idempotency_key="release-ungrounded",
    )

    with pytest.raises(policy.LocalUpdatePolicyError) as error:
        policy.make_policy_input_v1(
            store=store,
            scope=scope,
            base_release_id=forged_release.release_id,
            cutoff=_BASE + timedelta(seconds=150),
            evidence_snapshot_idempotency_key="snapshot-invalid-base",
        )

    assert error.value.reason == "base_release_invalid"


def test_closed_input_validation_rejects_noncanonical_or_ambiguous_values(
    tmp_path,
) -> None:
    _store, _scope, _records, _base, value = _fixture(tmp_path)

    mutants = (
        dataclasses.replace(value, schema_version=True),
        dataclasses.replace(value, cutoff_utc="2026-07-08T08:02:30+08:00"),
        dataclasses.replace(value, evidence=tuple(reversed(value.evidence))),
        dataclasses.replace(
            value,
            evidence=(
                dataclasses.replace(value.evidence[0], sequence_no=2**63),
                *value.evidence[1:],
            ),
        ),
        dataclasses.replace(
            value,
            evidence=(
                dataclasses.replace(value.evidence[0], kind=EvidenceKind.OUTCOME.value),
                *value.evidence[1:],
            ),
        ),
        dataclasses.replace(value, evidence=(*value.evidence, value.evidence[-1])),
        dataclasses.replace(
            value, base_memories=(*value.base_memories, value.base_memories[0])
        ),
    )
    for mutant in mutants:
        with pytest.raises(policy.LocalUpdatePolicyError):
            policy.policy_input_wire_v1(mutant)

    with pytest.raises(policy.LocalUpdatePolicyError) as unsupported:
        policy.run_local_update_policy_v1("answer_aware", value)
    assert unsupported.value.reason == "unsupported_policy"


def test_malformed_feedback_is_ignored_and_unchanged_facts_are_not_rewritten(
    tmp_path,
) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    _append(
        store,
        scope,
        label="unchanged-feedback",
        seconds=130,
        sequence_no=5,
        kind=EvidenceKind.FEEDBACK,
        payload=f"{_TARGET_KEY} = {_OLD}",
    )
    value = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=original.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-unchanged",
    )

    trusted = policy.run_local_update_policy_v1("feedback_latest", value)

    assert _TARGET_KEY not in {update.key for update in trusted.updates}
    assert {update.key for update in trusted.updates} == {_NEW_KEY}
    assert all("correction:" not in update.content for update in trusted.updates)


def test_equal_time_and_sequence_conflicts_use_preregistered_evidence_id_tie_break(
    tmp_path,
) -> None:
    store, scope, _records, _base, original = _fixture(tmp_path)
    tied = (
        _append(
            store,
            scope,
            label="tie-current",
            seconds=140,
            sequence_no=7,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{_TARGET_KEY} = {_CURRENT}",
        ),
        _append(
            store,
            scope,
            label="tie-wrong",
            seconds=140,
            sequence_no=7,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{_TARGET_KEY} = {_WRONG}",
        ),
    )
    value = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=original.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="snapshot-ties",
    )

    decision = policy.run_local_update_policy_v1("feedback_latest", value)
    target = next(update for update in decision.updates if update.key == _TARGET_KEY)
    expected = max(tied, key=lambda item: item.evidence_id)

    assert target.evidence_ids == (expected.evidence_id,)
    assert target.content == expected.event.payload


@pytest.mark.parametrize(
    "value_type",
    (
        policy.PolicyEvidenceV1,
        policy.BaseMemoryV1,
        policy.PolicyInputV1,
        policy.PolicyUpdateV1,
        policy.PolicyDecisionV1,
    ),
)
def test_policy_value_objects_are_frozen_and_slotted(value_type) -> None:
    assert dataclasses.is_dataclass(value_type)
    assert value_type.__dataclass_params__.frozen is True
    assert "__slots__" in value_type.__dict__
    assert "__dict__" not in value_type.__dict__


def test_hashes_bind_exact_policy_and_evidence_kind(tmp_path) -> None:
    _store, _scope, _records, _base, value = _fixture(tmp_path)
    trusted = policy.run_local_update_policy_v1("feedback_latest", value)
    blind = policy.run_local_update_policy_v1("latest_any", value)
    mutated_input = dataclasses.replace(
        value,
        evidence=(
            value.evidence[0],
            dataclasses.replace(
                value.evidence[1], kind=EvidenceKind.USER_MESSAGE.value
            ),
            *value.evidence[2:],
        ),
    )

    assert policy.policy_decision_sha256_v1(trusted) != (
        policy.policy_decision_sha256_v1(blind)
    )
    assert policy.policy_input_sha256_v1(value) != policy.policy_input_sha256_v1(
        mutated_input
    )
    assert (
        policy.policy_input_sha256_v1(value)
        == hashlib.sha256(
            b"areal-memory-local-update-policy-input-v1\0"
            + policy.policy_input_wire_v1(value)
        ).hexdigest()
    )
