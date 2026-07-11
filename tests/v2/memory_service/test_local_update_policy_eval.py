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
    )
    first = policy.run_local_update_policy_v1("feedback_latest", value)
    second = policy.run_local_update_policy_v1("feedback_latest", replayed)

    assert replayed == value
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
    assert len(policy.policy_input_wire_v1(value)) == 1314
    assert len(policy.policy_decision_wire_v1(first)) == 509
    assert (
        policy.policy_input_sha256_v1(value)
        == "a96cb41e6b3487166a09d7cdbf3a86f5671c6ba90996312848c969c247ac976f"
    )
    assert (
        policy.policy_decision_sha256_v1(first)
        == "496a73dad44c27792ce8e3c1430c4b2671250e5c6057655c1d0f6266a2400eff"
    )


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
