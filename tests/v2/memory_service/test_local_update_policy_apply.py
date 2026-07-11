# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from examples.memory_service import local_update_policy_apply as application
from examples.memory_service import local_update_policy_eval as policy
from tests.v2.memory_service.test_local_update_policy_eval import (
    _BASE,
    _CURRENT,
    _NEW_KEY,
    _NEW_VALUE,
    _OLD,
    _TARGET_KEY,
    _WRONG,
    _all_mapping_keys,
    _fixture,
)

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    MemoryServiceError,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore


def _apply(tmp_path, policy_name: str = "feedback_latest"):
    store, scope, records, base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1(policy_name, policy_input)
    result = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy=policy_name,
    )
    return store, scope, records, base, policy_input, decision, result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def test_apply_publishes_exact_grounded_add_and_supersede_release(tmp_path) -> None:
    store, scope, records, base, policy_input, decision, result = _apply(tmp_path)

    assert result.schema_version == 1
    assert result.policy == "feedback_latest"
    assert result.input_sha256 == policy.policy_input_sha256_v1(policy_input)
    assert result.decision_sha256 == policy.policy_decision_sha256_v1(decision)
    assert result.base_release_id == policy_input.base_release_id
    assert result.changed is True
    assert result.update_count == 2
    assert len(result.revision_ids) == 2
    assert result.evidence_root_sha256 == (
        application.recompute_applied_policy_release_root_v1(result)
    )

    release = store.get_release(scope, result.release_id)
    revisions = store.get_release_revisions(scope, result.release_id)
    assert release.content_hash == result.release_content_sha256
    assert release.manifest.revision_ids == result.revision_ids
    assert tuple(item.revision_id for item in revisions) == result.revision_ids

    receipt_by_operation = {item.operation: item for item in result.updates}
    added = receipt_by_operation["add"]
    superseded = receipt_by_operation["supersede"]
    assert added.generation == 0
    assert added.parent_revision_id is None
    assert superseded.generation == base.generation + 1
    assert superseded.memory_id == base.memory_id
    assert superseded.parent_revision_id == base.revision_id

    candidate_contents = {
        store.get_candidate(scope, item.candidate_id).proposal.content
        for item in result.updates
    }
    assert candidate_contents == {
        f"{_NEW_KEY} = {_NEW_VALUE}",
        f"{_TARGET_KEY} = {_CURRENT}",
    }
    evidence_ids = {record.evidence_id for record in records}
    assert all(
        set(store.get_candidate(scope, item.candidate_id).proposal.evidence_ids)
        <= evidence_ids
        for item in result.updates
    )


def test_application_receipt_and_root_contain_no_answer_preimage(tmp_path) -> None:
    _store, scope, records, _base, _input, _decision, result = _apply(tmp_path)
    encoded = _canonical_json_bytes(dataclasses.asdict(result))

    preimages = (
        _OLD,
        _CURRENT,
        _WRONG,
        _NEW_VALUE,
        _TARGET_KEY,
        _NEW_KEY,
        scope.tenant_id,
        scope.namespace,
        scope.subject_id,
        "policy.sqlite3",
        *(record.event.payload for record in records),
    )
    assert all(value.encode("utf-8") not in encoded for value in preimages)
    assert _all_mapping_keys(json.loads(encoded)).isdisjoint(
        {
            "arm",
            "content",
            "database_path",
            "evidence_ids",
            "future_query",
            "key",
            "outcome",
            "payload",
            "query",
            "reward",
            "scope",
            "value",
        }
    )

    payload = dataclasses.asdict(result)
    observed_root = payload.pop("evidence_root_sha256")
    expected_root = hashlib.sha256(
        b"areal-memory-local-update-release-evidence-v1\0"
        + _canonical_json_bytes(payload)
    ).hexdigest()
    assert observed_root == expected_root


def test_application_is_exactly_idempotent_without_duplicate_records(tmp_path) -> None:
    store, scope, _records, _base, policy_input, decision, first = _apply(tmp_path)
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    second = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    after = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    assert second == first
    assert after == before


def test_noop_reuses_base_graph_and_persists_application_alias(tmp_path) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("noop", policy_input)
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    result = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="noop",
    )

    assert result.changed is False
    assert result.update_count == 0
    assert result.updates == ()
    assert result.release_id == policy_input.base_release_id
    assert (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    ) == before
    with sqlite3.connect(tmp_path / "policy.sqlite3") as connection:
        aliases = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases "
            "ORDER BY idempotency_key"
        ).fetchall()
    assert (f"{result.application_id}-release", result.release_id) in aliases
    assert len(aliases) == 2


def test_negative_control_branches_from_same_base_and_persists_later_noise(
    tmp_path,
) -> None:
    store, scope, _records, base, policy_input = _fixture(tmp_path)
    trusted_decision = policy.run_local_update_policy_v1(
        "feedback_latest", policy_input
    )
    trusted = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=trusted_decision,
        expected_policy="feedback_latest",
    )
    blind_decision = policy.run_local_update_policy_v1("latest_any", policy_input)
    blind = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=blind_decision,
        expected_policy="latest_any",
    )

    assert (
        trusted.base_release_id == blind.base_release_id == policy_input.base_release_id
    )
    assert trusted.release_id != blind.release_id
    blind_target = next(
        update
        for update in blind.updates
        if store.get_revision(scope, update.revision_id).memory_id == base.memory_id
    )
    candidate = store.get_candidate(scope, blind_target.candidate_id)
    assert candidate.proposal.content == f"{_TARGET_KEY} = {_WRONG}"


def test_supersede_replaces_in_place_and_add_appends_after_all_base_members(
    tmp_path,
) -> None:
    store, scope, records, base, _original = _fixture(tmp_path)

    def add_base(record, label):
        candidate = store.append_candidate(
            CandidateProposal(
                scope=scope,
                content=record.event.payload,
                evidence_ids=(record.evidence_id,),
                idempotency_key=f"candidate-base-{label}",
            )
        )
        return store.append_revision(
            RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
                idempotency_key=f"revision-base-{label}",
            )
        )

    new_key_base = add_base(records[1], "new-key")
    third_key = "project-jkl678"
    third_value = "WXY23"
    third_record = store.append(
        EvidenceEvent(
            scope=scope,
            session_id="capture-session",
            run_id="capture-run",
            sequence_no=8,
            kind=EvidenceKind.USER_MESSAGE,
            payload=f"{third_key} = {third_value}",
            observed_at=_BASE + timedelta(seconds=10),
            idempotency_key="evidence-third-base",
        )
    )
    third_base = add_base(third_record, "third")
    added_key = "project-mnp789"
    added_value = "Z2345"
    store.append(
        EvidenceEvent(
            scope=scope,
            session_id="capture-session",
            run_id="capture-run",
            sequence_no=9,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{added_key} = {added_value}",
            observed_at=_BASE + timedelta(seconds=70),
            idempotency_key="evidence-new-add",
        )
    )
    base_release = store.append_release(
        ReleaseManifest(
            scope=scope,
            revision_ids=(
                third_base.revision_id,
                base.revision_id,
                new_key_base.revision_id,
            ),
        ),
        idempotency_key="release-three-base",
    )
    policy_input = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=base_release.release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="policy-input-three-base-snapshot",
    )
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)

    result = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )

    receipt_by_operation = {item.operation: item for item in result.updates}
    superseded = receipt_by_operation["supersede"]
    added = receipt_by_operation["add"]
    assert result.base_revision_count == 3
    assert result.result_revision_count == 4
    assert result.revision_ids[0] == third_base.revision_id
    assert result.revision_ids[1] == superseded.revision_id
    assert result.revision_ids[2] == new_key_base.revision_id
    assert result.revision_ids[3] == added.revision_id
    assert superseded.release_position == 1
    assert added.release_position == 3


def test_apply_rejects_policy_substitution_and_forged_decision_before_writes(
    tmp_path,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    forged = dataclasses.replace(decision, input_sha256="0" * 64)
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    for candidate, expected_policy in (
        (decision, "latest_any"),
        (forged, "feedback_latest"),
    ):
        with pytest.raises(application.LocalUpdateApplyError) as error:
            application.apply_local_update_decision_v1(
                store=store,
                scope=scope,
                policy_input=policy_input,
                decision=candidate,
                expected_policy=expected_policy,
            )
        assert error.value.reason == "decision_invalid"

    assert (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    ) == before


def test_constructible_input_payload_forgery_is_replayed_from_store_before_writes(
    tmp_path,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    forged_input = dataclasses.replace(
        policy_input,
        evidence=(
            *policy_input.evidence[:-1],
            dataclasses.replace(
                policy_input.evidence[-1],
                payload=f"{_TARGET_KEY} = {_CURRENT}",
            ),
        ),
    )
    forged_decision = policy.run_local_update_policy_v1(
        "feedback_latest",
        forged_input,
    )
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=forged_input,
            decision=forged_decision,
            expected_policy="feedback_latest",
        )

    assert error.value.reason == "input_drift"
    assert (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    ) == before


def test_constructible_input_cannot_omit_snapshot_member_before_writes(
    tmp_path,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    forged_input = dataclasses.replace(
        policy_input,
        evidence_snapshot_members=policy_input.evidence_snapshot_members[:-1],
        evidence=policy_input.evidence[:-1],
    )
    forged_decision = policy.run_local_update_policy_v1(
        "feedback_latest",
        forged_input,
    )
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=forged_input,
            decision=forged_decision,
            expected_policy="feedback_latest",
        )

    assert error.value.reason == "input_drift"
    assert (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    ) == before


def test_second_snapshot_replay_guards_first_candidate_write(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = policy.project_policy_input_from_snapshot_v1
    calls = 0

    def drift_on_prewrite(**kwargs):
        nonlocal calls
        calls += 1
        projected = original(**kwargs)
        if calls == 2:
            return dataclasses.replace(
                projected,
                evidence_snapshot_content_hash="0" * 64,
            )
        return projected

    monkeypatch.setattr(
        policy,
        "project_policy_input_from_snapshot_v1",
        drift_on_prewrite,
    )
    before = (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    )

    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )

    assert calls == 2
    assert error.value.reason == "input_drift"
    assert (
        len(store.list_candidates(scope)),
        len(store.list_revisions(scope)),
        len(store.list_releases(scope)),
    ) == before


def test_apply_only_reads_bound_snapshot_and_never_reseals(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original_project = policy.project_policy_input_from_snapshot_v1
    calls: list[tuple[str, str]] = []

    def seal_must_not_run(*_args, **_kwargs):
        raise AssertionError("apply attempted to seal a new snapshot")

    def observe_projection(**kwargs):
        calls.append(
            (kwargs["base_release_id"], kwargs["evidence_snapshot_id"])
        )
        return original_project(**kwargs)

    monkeypatch.setattr(
        SQLiteMemoryStore,
        "seal_evidence_snapshot",
        seal_must_not_run,
    )
    monkeypatch.setattr(
        policy,
        "project_policy_input_from_snapshot_v1",
        observe_projection,
    )

    result = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )

    assert len(calls) == 4
    assert {snapshot_id for _release_id, snapshot_id in calls} == {
        policy_input.evidence_snapshot_id
    }
    assert tuple(release_id for release_id, _snapshot_id in calls[:3]) == (
        policy_input.base_release_id,
    ) * 3
    assert calls[-1][0] == result.release_id


def test_later_backdated_evidence_does_not_rewrite_the_sealed_policy_snapshot(
    tmp_path,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    store.append(
        EvidenceEvent(
            scope=scope,
            session_id="late-writer-session",
            run_id="late-writer-run",
            sequence_no=0,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{_TARGET_KEY} = {_WRONG}",
            observed_at=_BASE + timedelta(seconds=140),
            idempotency_key="late-backdated-feedback",
        )
    )
    live_input = policy.make_policy_input_v1(
        store=store,
        scope=scope,
        base_release_id=policy_input.base_release_id,
        cutoff=_BASE + timedelta(seconds=150),
        evidence_snapshot_idempotency_key="policy-input-after-backfill",
    )
    assert live_input != policy_input
    assert policy.run_local_update_policy_v1("feedback_latest", live_input) != decision

    result = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )

    assert result.input_sha256 == policy.policy_input_sha256_v1(policy_input)
    target = next(
        store.get_candidate(scope, item.candidate_id)
        for item in result.updates
        if item.operation == "supersede"
    )
    assert target.proposal.content == f"{_TARGET_KEY} = {_CURRENT}"


def test_revision_failure_leaves_no_published_release_and_retry_recovers(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = SQLiteMemoryStore.append_revision

    def fail_policy_revision(self, proposal):
        if proposal.idempotency_key.startswith("apply_"):
            raise MemoryServiceError("private answer must not escape")
        return original(self, proposal)

    monkeypatch.setattr(SQLiteMemoryStore, "append_revision", fail_policy_revision)
    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )
    assert error.value.reason == "persistence_invalid"
    assert str(error.value) == "persistence_invalid"
    assert error.value.__cause__ is None
    assert tuple(item.release_id for item in store.list_releases(scope)) == (
        policy_input.base_release_id,
    )
    assert len(store.list_candidates(scope)) == 2
    assert len(store.list_revisions(scope)) == 1

    monkeypatch.setattr(SQLiteMemoryStore, "append_revision", original)
    recovered = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    assert recovered.changed is True
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3
    assert len(store.list_releases(scope)) == 2


def test_second_update_failure_leaves_partial_orphans_and_retry_converges(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = SQLiteMemoryStore.append_revision
    application_calls = 0

    def fail_second_policy_revision(self, proposal):
        nonlocal application_calls
        if proposal.idempotency_key.startswith("apply_"):
            application_calls += 1
            if application_calls == 2:
                raise MemoryServiceError("private answer must not escape")
        return original(self, proposal)

    monkeypatch.setattr(
        SQLiteMemoryStore,
        "append_revision",
        fail_second_policy_revision,
    )
    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )
    assert error.value.reason == "persistence_invalid"
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 2
    assert tuple(item.release_id for item in store.list_releases(scope)) == (
        policy_input.base_release_id,
    )

    monkeypatch.setattr(SQLiteMemoryStore, "append_revision", original)
    recovered = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    assert recovered.changed is True
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3
    assert len(store.list_releases(scope)) == 2


def test_release_failure_keeps_new_revisions_unpublished_until_retry(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = SQLiteMemoryStore.append_release

    def fail_policy_release(self, manifest, *, idempotency_key):
        if idempotency_key.startswith("apply_"):
            raise MemoryServiceError("private answer must not escape")
        return original(self, manifest, idempotency_key=idempotency_key)

    monkeypatch.setattr(SQLiteMemoryStore, "append_release", fail_policy_release)
    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )
    assert error.value.reason == "publication_invalid"
    assert error.value.__cause__ is None
    assert tuple(item.release_id for item in store.list_releases(scope)) == (
        policy_input.base_release_id,
    )
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3

    monkeypatch.setattr(SQLiteMemoryStore, "append_release", original)
    recovered = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    assert recovered.changed is True
    assert len(store.list_releases(scope)) == 2


def test_release_commit_with_lost_ack_is_recovered_by_exact_retry(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = SQLiteMemoryStore.append_release
    lost = False

    def commit_then_lose_ack(self, manifest, *, idempotency_key):
        nonlocal lost
        if idempotency_key.startswith("apply_") and not lost:
            self.append(
                EvidenceEvent(
                    scope=scope,
                    session_id="ack-race-session",
                    run_id="ack-race-run",
                    sequence_no=0,
                    kind=EvidenceKind.FEEDBACK,
                    payload=f"{_TARGET_KEY} = {_WRONG}",
                    observed_at=_BASE + timedelta(seconds=145),
                    idempotency_key="ack-race-backdated-feedback",
                )
            )
        release = original(self, manifest, idempotency_key=idempotency_key)
        if idempotency_key.startswith("apply_") and not lost:
            lost = True
            raise MemoryServiceError("private answer must not escape")
        return release

    monkeypatch.setattr(
        SQLiteMemoryStore,
        "append_release",
        commit_then_lose_ack,
    )
    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )
    assert error.value.reason == "publication_invalid"
    assert len(store.list_releases(scope)) == 2

    recovered = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    replayed = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    assert replayed == recovered
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3
    assert len(store.list_releases(scope)) == 2


def test_post_commit_projection_failure_leaves_shadow_release_then_retry_recovers(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)
    original = policy.project_policy_input_from_snapshot_v1

    def fail_post_commit_projection(**kwargs):
        if kwargs["base_release_id"] != policy_input.base_release_id:
            raise policy.LocalUpdatePolicyError("test_post_commit_failure")
        return original(**kwargs)

    monkeypatch.setattr(
        policy,
        "project_policy_input_from_snapshot_v1",
        fail_post_commit_projection,
    )
    with pytest.raises(application.LocalUpdateApplyError) as error:
        application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )
    assert error.value.reason == "publication_invalid"
    assert len(store.list_releases(scope)) == 2

    monkeypatch.setattr(
        policy,
        "project_policy_input_from_snapshot_v1",
        original,
    )
    recovered = application.apply_local_update_decision_v1(
        store=store,
        scope=scope,
        policy_input=policy_input,
        decision=decision,
        expected_policy="feedback_latest",
    )
    assert recovered.changed is True
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3
    assert len(store.list_releases(scope)) == 2


def test_concurrent_identical_applications_converge_to_one_release(tmp_path) -> None:
    store, scope, _records, _base, policy_input = _fixture(tmp_path)
    decision = policy.run_local_update_policy_v1("feedback_latest", policy_input)

    def apply_once():
        return application.apply_local_update_decision_v1(
            store=store,
            scope=scope,
            policy_input=policy_input,
            decision=decision,
            expected_policy="feedback_latest",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: apply_once(), range(2)))

    assert results[0] == results[1]
    assert len(store.list_candidates(scope)) == 3
    assert len(store.list_revisions(scope)) == 3
    assert len(store.list_releases(scope)) == 2


@pytest.mark.parametrize(
    "value_type",
    (application.AppliedUpdateReceiptV1, application.AppliedPolicyReleaseV1),
)
def test_application_receipts_are_frozen_and_slotted(value_type) -> None:
    assert dataclasses.is_dataclass(value_type)
    assert value_type.__dataclass_params__.frozen is True
    assert "__slots__" in value_type.__dict__
    assert "__dict__" not in value_type.__dict__


def test_root_changes_for_every_public_receipt_mutation(tmp_path) -> None:
    _store, _scope, _records, _base, _input, _decision, result = _apply(tmp_path)
    mutations = (
        dataclasses.replace(
            result,
            release_id="rel_" + "0" * 24,
            release_content_sha256="0" * 64,
        ),
        dataclasses.replace(
            result,
            updates=(
                dataclasses.replace(
                    result.updates[0],
                    update_commitment_sha256="0" * 64,
                ),
                *result.updates[1:],
            ),
        ),
        dataclasses.replace(
            result,
            updates=(
                dataclasses.replace(result.updates[0], evidence_count=2),
                *result.updates[1:],
            ),
        ),
    )

    for mutant in mutations:
        assert application.recompute_applied_policy_release_root_v1(mutant) != (
            result.evidence_root_sha256
        )


def test_receipt_recompute_rejects_impossible_id_hash_and_state_relations(
    tmp_path,
) -> None:
    _store, _scope, _records, _base, _input, _decision, result = _apply(tmp_path)
    added_index = next(
        index
        for index, update in enumerate(result.updates)
        if update.operation == "add"
    )
    added = result.updates[added_index]
    impossible_updates = list(result.updates)
    impossible_updates[added_index] = dataclasses.replace(
        added,
        memory_id="mem_" + "0" * 24,
    )
    mutants = (
        dataclasses.replace(result, application_id="apply_" + "0" * 64),
        dataclasses.replace(result, changed=False),
        dataclasses.replace(
            result,
            updates=(
                dataclasses.replace(
                    result.updates[0],
                    candidate_id="cand_" + "0" * 24,
                ),
                *result.updates[1:],
            ),
        ),
        dataclasses.replace(result, updates=tuple(impossible_updates)),
        dataclasses.replace(
            result,
            updates=(
                dataclasses.replace(result.updates[0], generation=2**63),
                *result.updates[1:],
            ),
        ),
    )

    for mutant in mutants:
        with pytest.raises(application.LocalUpdateApplyError) as error:
            application.recompute_applied_policy_release_root_v1(mutant)
        assert error.value.reason == "receipt_invalid"
