# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from examples.memory_service import local_update_provenance as provenance
from examples.memory_service import local_update_provenance_apply as provenance_apply
from examples.memory_service import local_update_provenance_projection as projection

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryApplicationProposal,
    MemoryApplicationUpdateProposal,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

_BASE = datetime(2026, 7, 12, tzinfo=UTC)
_KEY = "project-abc234"
_OTHER_KEY = "project-def567"
_OLD = "ABCDE"
_NEW = "FGHJK"
_OTHER = "LMNPQ"
_AGENT_VERSION = "a" * 64
_TOOL_VERSION = "b" * 64
_EVALUATOR_VERSION = "c" * 64


def _profile(**overrides: object) -> projection.ProvenanceProfileV1:
    values: dict[str, object] = {
        "schema_version": 1,
        "fact_namespace": "project_registry",
        "agent_id": "memory-agent",
        "agent_version_sha256": _AGENT_VERSION,
        "tool_name": "code_registry.lookup",
        "tool_version_sha256": _TOOL_VERSION,
        "evaluator_id": "code_registry.verifier",
        "evaluator_version_sha256": _EVALUATOR_VERSION,
    }
    values.update(overrides)
    return projection.ProvenanceProfileV1(**values)  # type: ignore[arg-type]


def _receipt(label: str) -> provenance.OpaqueReceiptCommitmentV1:
    raw = f"receipt:{label}".encode()
    return provenance.OpaqueReceiptCommitmentV1(
        kind="local_test_receipt",
        schema_version=1,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _append(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    label: str,
    kind: EvidenceKind,
    payload: str,
    seconds: int,
    sequence_no: int,
    session_id: str = "learning-session",
    run_id: str = "learning-run",
) -> EvidenceRecord:
    return store.append(
        EvidenceEvent(
            scope=scope,
            session_id=session_id,
            run_id=run_id,
            sequence_no=sequence_no,
            kind=kind,
            payload=payload,
            observed_at=_BASE + timedelta(seconds=seconds),
            idempotency_key=f"evidence-{label}",
        )
    )


def _fixture(
    tmp_path,
) -> tuple[SQLiteMemoryStore, MemoryScope, str, EvidenceRecord]:
    store = SQLiteMemoryStore(tmp_path / "provenance-policy.sqlite3")
    scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="provenance-policy-v2",
        subject_id="subject-1",
    )
    old = _append(
        store,
        scope,
        label="base-old",
        kind=EvidenceKind.USER_MESSAGE,
        payload=f"{_KEY} = {_OLD}",
        seconds=0,
        sequence_no=0,
        session_id="base-session",
        run_id="base-run",
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
    store.register_memory_application_root(scope, release.release_id)
    return store, scope, release.release_id, old


def _target_identity(
    value: EvidenceRecord | tuple[str, str],
) -> tuple[str, str]:
    if type(value) is EvidenceRecord:
        return value.evidence_id, value.content_hash
    return value


def _link(
    relation: provenance.ProvenanceRelationV1,
    target: EvidenceRecord | tuple[str, str],
) -> provenance.ProvenanceLinkV1:
    evidence_id, content_hash = _target_identity(target)
    return provenance.ProvenanceLinkV1(
        relation=relation,
        target_evidence_id=evidence_id,
        target_evidence_content_sha256=content_hash,
    )


def _append_chain(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    prefix: str = "chain",
    claim_key: str = _KEY,
    claim_value: str = _NEW,
    call_key: str = _KEY,
    result_key: str = _KEY,
    call_id: str = "call-1",
    result_call_id: str = "call-1",
    call_agent_id: str = "memory-agent",
    call_tool_name: str = "code_registry.lookup",
    result_tool_name: str = "code_registry.lookup",
    evaluator_id: str = "code_registry.verifier",
    result_status: provenance.ToolResultStatusV1 = provenance.ToolResultStatusV1.OK,
    result_value: str | None = _NEW,
    error_code: str | None = None,
    verdict: provenance.VerificationVerdictV1 = provenance.VerificationVerdictV1.PASS,
    claim_kind: EvidenceKind = EvidenceKind.FEEDBACK,
    claim_run: str = "learning-run",
    call_run: str = "learning-run",
    result_run: str = "learning-run",
    outcome_run: str = "learning-run",
    claim_session: str = "learning-session",
    call_session: str = "learning-session",
    result_session: str = "learning-session",
    outcome_session: str = "learning-session",
    claim_trajectory: str = "learning-run",
    call_trajectory: str = "learning-run",
    result_trajectory: str = "learning-run",
    outcome_trajectory: str = "learning-run",
    seconds: tuple[int, int, int, int] = (10, 11, 12, 13),
    call_parent_role: str = "claim",
    result_parent_role: str = "call",
    outcome_parent_role: str = "result",
    include_outcome: bool = True,
    existing_claim: EvidenceRecord | None = None,
) -> dict[str, EvidenceRecord]:
    claim_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id=claim_trajectory,
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.USER,
            producer_id="subject-1",
            version_sha256=None,
        ),
        links=(),
        body=provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key=claim_key,
            value=claim_value,
        ),
        receipt=None,
    )
    claim = existing_claim
    if claim is None:
        claim = _append(
            store,
            scope,
            label=f"{prefix}-claim",
            kind=claim_kind,
            payload=provenance.provenance_payload_wire_v1(claim_payload),
            seconds=seconds[0],
            sequence_no=1,
            session_id=claim_session,
            run_id=claim_run,
        )
    missing_hash = "f" * 64
    missing = (f"evd_{missing_hash[:24]}", missing_hash)
    call_parent = claim if call_parent_role == "claim" else missing
    request_hash = provenance.tool_request_content_sha256_v1(
        tool_name=call_tool_name,
        tool_version_sha256=_TOOL_VERSION,
        fact_namespace="project_registry",
        key=call_key,
    )
    call_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id=call_trajectory,
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.AGENT,
            producer_id=call_agent_id,
            version_sha256=_AGENT_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.TRIGGERED_BY, call_parent),),
        body=provenance.ToolCallBodyV1(
            call_id=call_id,
            tool_name=call_tool_name,
            tool_version_sha256=_TOOL_VERSION,
            fact_namespace="project_registry",
            key=call_key,
            request_content_sha256=request_hash,
        ),
        receipt=None,
    )
    call = _append(
        store,
        scope,
        label=f"{prefix}-call",
        kind=EvidenceKind.TOOL_CALL,
        payload=provenance.provenance_payload_wire_v1(call_payload),
        seconds=seconds[1],
        sequence_no=2,
        session_id=call_session,
        run_id=call_run,
    )
    if result_status is provenance.ToolResultStatusV1.ERROR and error_code is None:
        error_code = "lookup_error"
    result_parent: EvidenceRecord | tuple[str, str]
    if result_parent_role == "call":
        result_parent = call
    elif result_parent_role == "claim":
        result_parent = claim
    else:
        result_parent = missing
    result_hash = provenance.tool_result_content_sha256_v1(
        call_id=result_call_id,
        tool_name=result_tool_name,
        tool_version_sha256=_TOOL_VERSION,
        status=result_status,
        fact_namespace="project_registry",
        key=result_key,
        value=result_value,
        error_code=error_code,
    )
    result_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id=result_trajectory,
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.TOOL,
            producer_id=result_tool_name,
            version_sha256=_TOOL_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.RESULT_OF, result_parent),),
        body=provenance.ToolResultBodyV1(
            call_id=result_call_id,
            tool_name=result_tool_name,
            tool_version_sha256=_TOOL_VERSION,
            status=result_status,
            fact_namespace="project_registry",
            key=result_key,
            value=result_value,
            error_code=error_code,
            result_content_sha256=result_hash,
        ),
        receipt=_receipt(f"{prefix}-result"),
    )
    result = _append(
        store,
        scope,
        label=f"{prefix}-result",
        kind=EvidenceKind.TOOL_RESULT,
        payload=provenance.provenance_payload_wire_v1(result_payload),
        seconds=seconds[2],
        sequence_no=3,
        session_id=result_session,
        run_id=result_run,
    )
    records = {"claim": claim, "call": call, "result": result}
    if not include_outcome:
        return records
    outcome_parent: EvidenceRecord | tuple[str, str]
    if outcome_parent_role == "result":
        outcome_parent = result
    elif outcome_parent_role == "call":
        outcome_parent = call
    else:
        outcome_parent = missing
    outcome_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id=outcome_trajectory,
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id=evaluator_id,
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, outcome_parent),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=verdict,
            evaluator_id=evaluator_id,
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt(f"{prefix}-outcome"),
    )
    outcome = _append(
        store,
        scope,
        label=f"{prefix}-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(outcome_payload),
        seconds=seconds[3],
        sequence_no=4,
        session_id=outcome_session,
        run_id=outcome_run,
    )
    records["outcome"] = outcome
    return records


def _append_reverification_outcome(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    result: EvidenceRecord,
    *,
    prefix: str,
    seconds: int,
    evaluator_id: str = "code_registry.verifier",
    verdict: provenance.VerificationVerdictV1 = provenance.VerificationVerdictV1.PASS,
) -> EvidenceRecord:
    payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id=result.event.run_id,
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id=evaluator_id,
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(
            _link(provenance.ProvenanceRelationV1.EVALUATES, result),
        ),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=verdict,
            evaluator_id=evaluator_id,
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt(f"{prefix}-outcome"),
    )
    return _append(
        store,
        scope,
        label=f"{prefix}-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(payload),
        seconds=seconds,
        sequence_no=result.event.sequence_no + 1,
        session_id=result.event.session_id,
        run_id=result.event.run_id,
    )


def _make_input(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    release_id: str,
    *,
    label: str = "main",
    profile: projection.ProvenanceProfileV1 | None = None,
) -> projection.PolicyInputV2:
    return projection.make_policy_input_v2(
        store=store,
        scope=scope,
        base_release_id=release_id,
        cutoff=_BASE + timedelta(seconds=100),
        evidence_snapshot_idempotency_key=f"snapshot-{label}",
        provenance_profile=_profile() if profile is None else profile,
    )


def _resolve(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    value: projection.PolicyInputV2,
) -> projection.ProvenanceGraphV1:
    return projection.resolve_provenance_graph_v1(
        store=store,
        scope=scope,
        value=value,
    )


def _application_update(
    source_input: projection.PolicyInputV2,
    *,
    evidence_ids: tuple[str, ...],
    value: str,
) -> MemoryApplicationUpdateProposal:
    return MemoryApplicationUpdateProposal(
        content=f"{_KEY} = {value}",
        evidence_ids=evidence_ids,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=source_input.base_memories[0].revision_id,
    )


def _commit_application(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    source_input: projection.PolicyInputV2,
    *,
    evidence_ids: tuple[str, ...],
    value: str = _NEW,
    label: str,
):
    return provenance_apply.commit_verified_chain_application_v1(
        store=store,
        scope=scope,
        source_input=source_input,
        updates=(
            _application_update(
                source_input,
                evidence_ids=evidence_ids,
                value=value,
            ),
        ),
        idempotency_key=f"provenance-application-{label}",
    )


def _commit_unchecked_application(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    source_input: projection.PolicyInputV2,
    *,
    evidence_ids: tuple[str, ...],
    value: str = _NEW,
    label: str,
    commitment_mutation: str | None = None,
):
    updates = (
        _application_update(
            source_input,
            evidence_ids=evidence_ids,
            value=value,
        ),
    )
    projector_id = projection.PROVENANCE_PROJECTOR_ID_V1
    projector_version = projection.PROVENANCE_PROJECTOR_VERSION_SHA256_V1
    policy_id = projection.VERIFIED_CHAIN_POLICY_ID_V1
    policy_version = projection.VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1
    policy_input_sha256 = projection.policy_input_sha256_v2(source_input)
    decision_sha256 = projection.verified_chain_decision_sha256_v1(
        source_input,
        updates,
    )
    policy_context = projection.provenance_application_context_v1(
        source_input.provenance_profile
    )
    if commitment_mutation == "projector_id":
        projector_id = "other-projector"
    elif commitment_mutation == "projector_version":
        projector_version = "f" * 64
    elif commitment_mutation == "policy_id":
        policy_id = "other-policy"
    elif commitment_mutation == "policy_version":
        policy_version = "f" * 64
    elif commitment_mutation == "policy_input":
        policy_input_sha256 = "f" * 64
    elif commitment_mutation == "decision":
        decision_sha256 = "f" * 64
    elif commitment_mutation == "policy_context":
        policy_context = projection.provenance_application_context_v1(
            _profile(evaluator_id="code_registry.verifier.v2")
        )
    elif commitment_mutation is not None:
        raise AssertionError(f"unknown commitment mutation: {commitment_mutation}")
    return store.commit_memory_application(
        MemoryApplicationProposal(
            scope=scope,
            source_snapshot_id=source_input.evidence_snapshot_id,
            source_base_release_id=source_input.base_release_id,
            projector_id=projector_id,
            projector_version_sha256=projector_version,
            policy_id=policy_id,
            policy_version_sha256=policy_version,
            policy_input_sha256=policy_input_sha256,
            decision_sha256=decision_sha256,
            policy_context=policy_context,
            updates=updates,
            idempotency_key=f"provenance-application-{label}",
        )
    )


def test_profile_and_policy_input_are_canonical_and_store_authentic(tmp_path) -> None:
    store, scope, release_id, old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    policy_input = _make_input(store, scope, release_id)

    assert policy_input.schema_version == 2
    assert policy_input.learning_allowed_kinds == (
        "feedback",
        "outcome",
        "tool_call",
        "tool_result",
        "user_message",
    )
    assert policy_input.evidence_high_watermark == 4
    assert policy_input.base_memories[0].learning_evidence_after_ingest_order == -1
    assert len(policy_input.evidence) == 5
    assert policy_input.evidence[0].evidence_id == old.evidence_id
    assert policy_input.evidence[0].provenance_payload_sha256 is None
    assert all(
        item.provenance_payload_sha256 is not None for item in policy_input.evidence[1:]
    )
    assert len(policy_input.base_memories) == 1
    base = policy_input.base_memories[0]
    assert (base.key, base.value) == (_KEY, _OLD)
    assert tuple(item.evidence_id for item in base.grounding_evidence) == (
        old.evidence_id,
    )
    assert policy_input.provenance_profile_sha256 == (
        projection.provenance_profile_sha256_v1(_profile())
    )
    assert projection.provenance_profile_wire_v1(_profile()) == (
        b'{"agent_id":"memory-agent","agent_version_sha256":'
        + f'"{_AGENT_VERSION}",'.encode()
        + b'"evaluator_id":"code_registry.verifier",'
        + f'"evaluator_version_sha256":"{_EVALUATOR_VERSION}",'.encode()
        + b'"fact_namespace":"project_registry","schema_version":1,'
        + b'"tool_name":"code_registry.lookup",'
        + f'"tool_version_sha256":"{_TOOL_VERSION}"}}'.encode()
    )
    assert policy_input.provenance_profile_sha256 == (
        "8e433079ba7a3b65bf7ec39169ba2300da503debfba31d6e7ce21deab45fb71a"
    )
    assert projection.policy_input_sha256_v2(policy_input) == (
        "91e8af7246ecc2d4257c4cc64ab0eb1b50b5119c3e6751af9b3f3cf87ae50aca"
    )

    restored = projection.project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=release_id,
        evidence_snapshot_id=policy_input.evidence_snapshot_id,
        provenance_profile=_profile(),
    )
    assert restored == policy_input
    graph = _resolve(store, scope, policy_input)
    assert graph.rejected_outcome_evidence_ids == ()
    assert len(graph.verified_fact_chains) == 1
    chain = graph.verified_fact_chains[0]
    assert (chain.key, chain.value, chain.trajectory_id) == (
        _KEY,
        _NEW,
        "learning-run",
    )
    assert (
        chain.claim.evidence_id,
        chain.tool_call.evidence_id,
        chain.tool_result.evidence_id,
        chain.outcome.evidence_id,
    ) == (
        records["claim"].evidence_id,
        records["call"].evidence_id,
        records["result"].evidence_id,
        records["outcome"].evidence_id,
    )


def test_make_is_idempotent_for_the_same_snapshot_spec(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope)
    first = _make_input(store, scope, release_id)
    second = _make_input(store, scope, release_id)
    assert second == first
    assert projection.policy_input_wire_v2(second) == (
        projection.policy_input_wire_v2(first)
    )


def test_provenance_grounded_release_round_trips_as_the_next_base(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    first = _make_input(store, scope, release_id, label="first-round")
    graph = _resolve(store, scope, first)
    chain = graph.verified_fact_chains[0]
    grounding_ids = tuple(
        member.evidence_id
        for member in (
            chain.claim,
            chain.tool_call,
            chain.tool_result,
            chain.outcome,
        )
    )
    application = _commit_application(
        store,
        scope,
        first,
        evidence_ids=grounding_ids,
        label="round-trip",
    )
    second = _make_input(
        store,
        scope,
        application.result_release_id,
        label="second-round",
    )
    assert len(second.base_memories) == 1
    base = second.base_memories[0]
    assert (base.key, base.value, base.generation) == (_KEY, _NEW, 1)
    assert tuple(member.evidence_id for member in base.grounding_evidence) == (
        grounding_ids
    )
    assert base.origin_application_id == application.application_id
    assert base.origin_application_content_sha256 == application.content_hash
    assert base.learning_evidence_after_ingest_order == (
        application.source_evidence_high_watermark
    )
    assert _resolve(store, scope, second).verified_fact_chains == ()

    # A later conflicting outcome changes the new update graph, but cannot
    # retroactively invalidate the exact evidence subset grounding this release.
    conflict_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(
            _link(
                provenance.ProvenanceRelationV1.EVALUATES,
                records["result"],
            ),
        ),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.FAIL,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("post-release-conflict"),
    )
    _append(
        store,
        scope,
        label="post-release-conflict",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(conflict_payload),
        seconds=14,
        sequence_no=5,
    )
    third = _make_input(
        store,
        scope,
        application.result_release_id,
        label="third-round",
    )
    assert third.base_memories[0].value == _NEW
    assert _resolve(store, scope, third).verified_fact_chains == ()


def test_preexisting_conflict_cannot_be_hidden_by_cherry_picking_pass_members(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope, prefix="preexisting-conflict")
    conflict_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, records["result"]),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.FAIL,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("preexisting-conflict-fail"),
    )
    _append(
        store,
        scope,
        label="preexisting-conflict-fail",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(conflict_payload),
        seconds=14,
        sequence_no=5,
    )
    source = _make_input(store, scope, release_id, label="preexisting-conflict")
    assert _resolve(store, scope, source).verified_fact_chains == ()
    cherry_picked_ids = tuple(
        records[role].evidence_id for role in ("claim", "call", "result", "outcome")
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    with pytest.raises(
        provenance_apply.LocalUpdateProvenanceApplyError,
        match="decision_invalid",
    ):
        provenance_apply.commit_verified_chain_application_v1(
            store=store,
            scope=scope,
            source_input=source,
            updates=(
                _application_update(
                    source,
                    evidence_ids=cherry_picked_ids,
                    value=_NEW,
                ),
            ),
            idempotency_key="reject-cherry-picked-before-write",
        )
    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before

    application = _commit_unchecked_application(
        store,
        scope,
        source,
        evidence_ids=cherry_picked_ids,
        label="cherry-picked-conflict",
    )

    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        _make_input(
            store,
            scope,
            application.result_release_id,
            label="reject-cherry-picked-conflict",
        )


def test_verified_apply_closes_validation_to_commit_evidence_race(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="stale-apply")
    source = _make_input(store, scope, release_id, label="stale-apply")
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    grounding_ids = tuple(
        member.evidence_id
        for member in (
            chain.claim,
            chain.tool_call,
            chain.tool_result,
            chain.outcome,
        )
    )
    _append(
        store,
        scope,
        label="stale-apply-arrival",
        kind=EvidenceKind.FEEDBACK,
        payload="arrived between validation and commit",
        seconds=30,
        sequence_no=30,
    )
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    with pytest.raises(
        provenance_apply.LocalUpdateProvenanceApplyError,
        match="source_snapshot_stale",
    ):
        provenance_apply.commit_verified_chain_application_v1(
            store=store,
            scope=scope,
            source_input=source,
            updates=(
                _application_update(
                    source,
                    evidence_ids=grounding_ids,
                    value=_NEW,
                ),
            ),
            idempotency_key="stale-apply",
        )
    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before


@pytest.mark.parametrize(
    "commitment_mutation",
    (
        "projector_id",
        "projector_version",
        "policy_id",
        "policy_version",
        "policy_input",
        "decision",
        "policy_context",
    ),
)
def test_semantically_valid_update_with_mismatched_commitment_is_not_a_base(
    tmp_path,
    commitment_mutation: str,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    label = f"mismatched-{commitment_mutation}"
    _append_chain(store, scope, prefix=label)
    source = _make_input(store, scope, release_id, label=label)
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    application = _commit_unchecked_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        ),
        label=label,
        commitment_mutation=commitment_mutation,
    )

    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        _make_input(
            store,
            scope,
            application.result_release_id,
            label=f"reject-{label}",
        )


def test_semantically_valid_generic_core_application_replays_as_a_base(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="generic-core-valid")
    source = _make_input(store, scope, release_id, label="generic-core-valid")
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    application = _commit_unchecked_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        ),
        label="generic-core-valid",
    )

    current = _make_input(
        store,
        scope,
        application.result_release_id,
        label="generic-core-valid-result",
    )
    assert current.base_memories[0].value == _NEW
    assert current.base_memories[0].origin_application_id == application.application_id


def test_same_value_supersede_cannot_bypass_apply_through_core_store(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(
        store,
        scope,
        prefix="same-value",
        claim_value=_OLD,
        result_value=_OLD,
    )
    source = _make_input(store, scope, release_id, label="same-value")
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    application = _commit_unchecked_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        ),
        value=_OLD,
        label="same-value",
    )

    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        _make_input(
            store,
            scope,
            application.result_release_id,
            label="reject-same-value",
        )


def test_historical_application_replays_its_profile_after_current_profile_upgrade(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="profile-v1")
    source = _make_input(store, scope, release_id, label="profile-v1")
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    application = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        ),
        label="profile-v1",
    )
    upgraded_profile = _profile(
        evaluator_id="code_registry.verifier.v2",
        evaluator_version_sha256="f" * 64,
    )

    current = _make_input(
        store,
        scope,
        application.result_release_id,
        label="profile-v2",
        profile=upgraded_profile,
    )

    assert current.base_memories[0].value == _NEW
    assert current.base_memories[0].origin_application_id == (
        application.application_id
    )
    assert _resolve(store, scope, current).verified_fact_chains == ()


def test_two_application_generations_replay_each_historical_profile(
    tmp_path,
    monkeypatch,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="generation-v1")
    first_input = _make_input(store, scope, release_id, label="generation-v1")
    first_chain = _resolve(store, scope, first_input).verified_fact_chains[0]
    first = _commit_application(
        store,
        scope,
        first_input,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                first_chain.claim,
                first_chain.tool_call,
                first_chain.tool_result,
                first_chain.outcome,
            )
        ),
        label="generation-v1",
    )
    evaluator_v2 = "code_registry.verifier.v2"
    _append_chain(
        store,
        scope,
        prefix="generation-v2",
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="generation-v2-call",
        result_call_id="generation-v2-call",
        evaluator_id=evaluator_v2,
        seconds=(20, 21, 22, 23),
    )
    profile_v2 = _profile(evaluator_id=evaluator_v2)
    second_input = _make_input(
        store,
        scope,
        first.result_release_id,
        label="generation-v2",
        profile=profile_v2,
    )
    second_chain = _resolve(store, scope, second_input).verified_fact_chains[0]
    second = _commit_application(
        store,
        scope,
        second_input,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                second_chain.claim,
                second_chain.tool_call,
                second_chain.tool_result,
                second_chain.outcome,
            )
        ),
        value=_OTHER,
        label="generation-v2",
    )

    final_input = _make_input(
        store,
        scope,
        second.result_release_id,
        label="generation-final",
        profile=_profile(evaluator_id="code_registry.verifier.v3"),
    )

    assert second.application_order == 1
    assert final_input.base_memories[0].value == _OTHER
    assert final_input.base_memories[0].generation == 2
    assert final_input.base_memories[0].origin_application_id == second.application_id

    calls = {"replay": 0, "snapshot": 0, "snapshot_evidence": 0}
    original_replay = SQLiteMemoryStore.get_memory_application_replay
    original_snapshot = SQLiteMemoryStore.get_evidence_snapshot
    original_snapshot_evidence = SQLiteMemoryStore.get_evidence_snapshot_evidence

    def counted_replay(self, requested_scope, target_release_id):
        calls["replay"] += 1
        return original_replay(self, requested_scope, target_release_id)

    def counted_snapshot(self, requested_scope, snapshot_id):
        calls["snapshot"] += 1
        return original_snapshot(self, requested_scope, snapshot_id)

    def counted_snapshot_evidence(self, requested_scope, snapshot_id):
        calls["snapshot_evidence"] += 1
        return original_snapshot_evidence(self, requested_scope, snapshot_id)

    def forbidden_layer_getter(*_args, **_kwargs):
        raise AssertionError("bulk application replay must not use per-layer getters")

    monkeypatch.setattr(
        SQLiteMemoryStore,
        "get_memory_application_replay",
        counted_replay,
    )
    monkeypatch.setattr(
        SQLiteMemoryStore,
        "get_evidence_snapshot",
        counted_snapshot,
    )
    monkeypatch.setattr(
        SQLiteMemoryStore,
        "get_evidence_snapshot_evidence",
        counted_snapshot_evidence,
    )
    for method_name in (
        "get_memory_application_for_release",
        "get_release",
        "get_release_revisions",
    ):
        monkeypatch.setattr(
            SQLiteMemoryStore,
            method_name,
            forbidden_layer_getter,
        )

    replayed = projection.project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=second.result_release_id,
        evidence_snapshot_id=final_input.evidence_snapshot_id,
        provenance_profile=_profile(evaluator_id="code_registry.verifier.v3"),
    )

    assert replayed == final_input
    assert calls == {"replay": 1, "snapshot": 1, "snapshot_evidence": 1}


def test_same_profile_can_evolve_one_memory_across_two_evidence_epochs(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="stable-profile-generation-1")
    first_input = _make_input(
        store,
        scope,
        release_id,
        label="stable-profile-generation-1",
    )
    first_chain = _resolve(store, scope, first_input).verified_fact_chains[0]
    first = _commit_application(
        store,
        scope,
        first_input,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                first_chain.claim,
                first_chain.tool_call,
                first_chain.tool_result,
                first_chain.outcome,
            )
        ),
        label="stable-profile-generation-1",
    )

    _append_chain(
        store,
        scope,
        prefix="stable-profile-generation-2",
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="stable-profile-generation-2-call",
        result_call_id="stable-profile-generation-2-call",
        seconds=(20, 21, 22, 23),
    )
    second_input = _make_input(
        store,
        scope,
        first.result_release_id,
        label="stable-profile-generation-2",
    )
    second_graph = _resolve(store, scope, second_input)
    assert tuple(chain.value for chain in second_graph.verified_fact_chains) == (
        _OTHER,
    )
    assert second_input.base_memories[
        0
    ].learning_evidence_after_ingest_order == first.source_evidence_high_watermark
    second_chain = second_graph.verified_fact_chains[0]
    second = _commit_application(
        store,
        scope,
        second_input,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                second_chain.claim,
                second_chain.tool_call,
                second_chain.tool_result,
                second_chain.outcome,
            )
        ),
        value=_OTHER,
        label="stable-profile-generation-2",
    )

    final_input = _make_input(
        store,
        scope,
        second.result_release_id,
        label="stable-profile-final",
    )
    assert final_input.base_memories[0].value == _OTHER
    assert final_input.base_memories[0].generation == 2
    assert final_input.base_memories[0].origin_application_id == second.application_id
    assert _resolve(store, scope, final_input).verified_fact_chains == ()


def test_new_outcome_can_reverify_a_result_from_the_closed_epoch(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope, prefix="initial-verification")
    source = _make_input(store, scope, release_id, label="initial-verification")
    chain = _resolve(store, scope, source).verified_fact_chains[0]
    application = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        ),
        label="initial-verification",
    )

    second_outcome = _append_reverification_outcome(
        store,
        scope,
        records["result"],
        prefix="explicit-reverification",
        seconds=20,
    )
    current = _make_input(
        store,
        scope,
        application.result_release_id,
        label="explicit-reverification",
    )
    graph = _resolve(store, scope, current)

    assert len(graph.verified_fact_chains) == 1
    assert graph.verified_fact_chains[0].outcome.evidence_id == (
        second_outcome.evidence_id
    )
    assert graph.rejected_outcome_evidence_ids == ()


def test_new_outcome_can_finish_a_chain_whose_ancestors_precede_the_epoch(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="epoch-one")
    delayed = _append_chain(
        store,
        scope,
        prefix="delayed-verification",
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="delayed-verification-call",
        result_call_id="delayed-verification-call",
        seconds=(20, 21, 22, 23),
        include_outcome=False,
    )
    source = _make_input(store, scope, release_id, label="delayed-source")
    first_chain = _resolve(store, scope, source).verified_fact_chains[0]
    first = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                first_chain.claim,
                first_chain.tool_call,
                first_chain.tool_result,
                first_chain.outcome,
            )
        ),
        label="delayed-source",
    )

    delayed_outcome = _append_reverification_outcome(
        store,
        scope,
        delayed["result"],
        prefix="delayed-verification",
        seconds=30,
    )
    current = _make_input(
        store,
        scope,
        first.result_release_id,
        label="delayed-current",
    )
    graph = _resolve(store, scope, current)

    assert tuple(chain.value for chain in graph.verified_fact_chains) == (_OTHER,)
    assert graph.verified_fact_chains[0].outcome.evidence_id == (
        delayed_outcome.evidence_id
    )
    assert graph.verified_fact_chains[0].tool_result.evidence_id == (
        delayed["result"].evidence_id
    )
    assert graph.verified_fact_chains[0].tool_result.ingest_order <= (
        current.base_memories[0].learning_evidence_after_ingest_order
    )


def test_partial_update_does_not_consume_an_absent_keys_valid_chain(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="partial-a")
    _append_chain(
        store,
        scope,
        prefix="partial-b",
        claim_key=_OTHER_KEY,
        call_key=_OTHER_KEY,
        result_key=_OTHER_KEY,
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="partial-b-call",
        result_call_id="partial-b-call",
        seconds=(20, 21, 22, 23),
    )
    source = _make_input(store, scope, release_id, label="partial-source")
    graph = _resolve(store, scope, source)
    chain_a = next(chain for chain in graph.verified_fact_chains if chain.key == _KEY)
    first = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain_a.claim,
                chain_a.tool_call,
                chain_a.tool_result,
                chain_a.outcome,
            )
        ),
        label="partial-a",
    )

    second_input = _make_input(
        store,
        scope,
        first.result_release_id,
        label="partial-b",
    )
    second_graph = _resolve(store, scope, second_input)
    assert tuple(chain.key for chain in second_graph.verified_fact_chains) == (
        _OTHER_KEY,
    )
    chain_b = second_graph.verified_fact_chains[0]
    second = provenance_apply.commit_verified_chain_application_v1(
        store=store,
        scope=scope,
        source_input=second_input,
        updates=(
            MemoryApplicationUpdateProposal(
                content=f"{_OTHER_KEY} = {_OTHER}",
                evidence_ids=tuple(
                    member.evidence_id
                    for member in (
                        chain_b.claim,
                        chain_b.tool_call,
                        chain_b.tool_result,
                        chain_b.outcome,
                    )
                ),
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
            ),
        ),
        idempotency_key="partial-b",
    )

    final_input = _make_input(
        store,
        scope,
        second.result_release_id,
        label="partial-final",
    )
    assert tuple((base.key, base.value) for base in final_input.base_memories) == (
        (_KEY, _NEW),
        (_OTHER_KEY, _OTHER),
    )
    assert _resolve(store, scope, final_input).verified_fact_chains == ()


def test_unrelated_update_cannot_wash_out_an_absent_keys_conflict(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="washout-a")
    _append_chain(
        store,
        scope,
        prefix="washout-b-one",
        claim_key=_OTHER_KEY,
        call_key=_OTHER_KEY,
        result_key=_OTHER_KEY,
        call_id="washout-b-one-call",
        result_call_id="washout-b-one-call",
        seconds=(20, 21, 22, 23),
    )
    _append_chain(
        store,
        scope,
        prefix="washout-b-two",
        claim_key=_OTHER_KEY,
        call_key=_OTHER_KEY,
        result_key=_OTHER_KEY,
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="washout-b-two-call",
        result_call_id="washout-b-two-call",
        seconds=(30, 31, 32, 33),
    )
    source = _make_input(store, scope, release_id, label="washout-source")
    source_graph = _resolve(store, scope, source)
    chain_a = next(
        chain for chain in source_graph.verified_fact_chains if chain.key == _KEY
    )
    first = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                chain_a.claim,
                chain_a.tool_call,
                chain_a.tool_result,
                chain_a.outcome,
            )
        ),
        label="washout-a",
    )
    current = _make_input(
        store,
        scope,
        first.result_release_id,
        label="washout-current",
    )
    conflicting = tuple(
        chain
        for chain in _resolve(store, scope, current).verified_fact_chains
        if chain.key == _OTHER_KEY
    )
    assert {chain.value for chain in conflicting} == {_NEW, _OTHER}
    before = (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    )

    with pytest.raises(
        provenance_apply.LocalUpdateProvenanceApplyError,
        match="decision_invalid",
    ):
        provenance_apply.commit_verified_chain_application_v1(
            store=store,
            scope=scope,
            source_input=current,
            updates=(
                MemoryApplicationUpdateProposal(
                    content=f"{_OTHER_KEY} = {_NEW}",
                    evidence_ids=tuple(
                        member.evidence_id
                        for member in (
                            conflicting[0].claim,
                            conflicting[0].tool_call,
                            conflicting[0].tool_result,
                            conflicting[0].outcome,
                        )
                    ),
                    operation=RevisionOperation.ADD,
                    parent_revision_id=None,
                ),
            ),
            idempotency_key="reject-washout-b",
        )
    assert (
        store.list_candidates(scope),
        store.list_revisions(scope),
        store.list_releases(scope),
    ) == before


def test_profile_upgrade_does_not_reopen_previously_sealed_backlog(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, prefix="sealed-profile-v1")
    evaluator_v2 = "code_registry.verifier.v2"
    v2_records = _append_chain(
        store,
        scope,
        prefix="sealed-profile-v2",
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="sealed-profile-v2-call",
        result_call_id="sealed-profile-v2-call",
        evaluator_id=evaluator_v2,
        seconds=(20, 21, 22, 23),
    )
    source = _make_input(store, scope, release_id, label="sealed-profile-v1")
    v1_chain = _resolve(store, scope, source).verified_fact_chains[0]
    first = _commit_application(
        store,
        scope,
        source,
        evidence_ids=tuple(
            member.evidence_id
            for member in (
                v1_chain.claim,
                v1_chain.tool_call,
                v1_chain.tool_result,
                v1_chain.outcome,
            )
        ),
        label="sealed-profile-v1",
    )
    profile_v2 = _profile(evaluator_id=evaluator_v2)
    upgraded = _make_input(
        store,
        scope,
        first.result_release_id,
        label="sealed-profile-v2",
        profile=profile_v2,
    )
    assert _resolve(store, scope, upgraded).verified_fact_chains == ()

    new_outcome = _append_reverification_outcome(
        store,
        scope,
        v2_records["result"],
        prefix="sealed-profile-v2-reverification",
        seconds=30,
        evaluator_id=evaluator_v2,
    )
    reverified = _make_input(
        store,
        scope,
        first.result_release_id,
        label="sealed-profile-v2-reverified",
        profile=profile_v2,
    )
    graph = _resolve(store, scope, reverified)
    assert tuple(chain.value for chain in graph.verified_fact_chains) == (_OTHER,)
    assert graph.verified_fact_chains[0].outcome.evidence_id == new_outcome.evidence_id


@pytest.mark.parametrize(
    "mutation",
    ("missing_outcome", "reversed_order", "wrong_value", "extra_legacy"),
)
def test_provenance_base_requires_exact_complete_matching_chain(
    tmp_path,
    mutation: str,
) -> None:
    store, scope, release_id, old = _fixture(tmp_path)
    _append_chain(store, scope)
    first = _make_input(store, scope, release_id, label="source")
    chain = _resolve(store, scope, first).verified_fact_chains[0]
    evidence_ids = tuple(
        member.evidence_id
        for member in (
            chain.claim,
            chain.tool_call,
            chain.tool_result,
            chain.outcome,
        )
    )
    content = f"{_KEY} = {_NEW}"
    if mutation == "missing_outcome":
        evidence_ids = evidence_ids[:-1]
    elif mutation == "reversed_order":
        evidence_ids = tuple(reversed(evidence_ids))
    elif mutation == "wrong_value":
        content = f"{_KEY} = {_OTHER}"
    elif mutation == "extra_legacy":
        evidence_ids = (*evidence_ids, old.evidence_id)
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=content,
            evidence_ids=evidence_ids,
            idempotency_key=f"invalid-provenance-candidate-{mutation}",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=first.base_memories[0].revision_id,
            idempotency_key=f"invalid-provenance-revision-{mutation}",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key=f"invalid-provenance-release-{mutation}",
    )
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        _make_input(
            store,
            scope,
            release.release_id,
            label=f"invalid-{mutation}",
        )


def test_consensus_grounding_can_cover_multiple_complete_equal_chains(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(
        store, scope, prefix="consensus-a", call_id="call-a", result_call_id="call-a"
    )
    _append_chain(
        store,
        scope,
        prefix="consensus-b",
        call_id="call-b",
        result_call_id="call-b",
        seconds=(20, 21, 22, 23),
    )
    first = _make_input(store, scope, release_id, label="consensus-source")
    chains = _resolve(store, scope, first).verified_fact_chains
    assert len(chains) == 2
    evidence_ids = tuple(
        member.evidence_id
        for chain in chains
        for member in (
            chain.claim,
            chain.tool_call,
            chain.tool_result,
            chain.outcome,
        )
    )
    application = _commit_application(
        store,
        scope,
        first,
        evidence_ids=evidence_ids,
        label="consensus",
    )
    second = _make_input(
        store,
        scope,
        application.result_release_id,
        label="consensus-next",
    )
    assert (
        tuple(
            member.evidence_id for member in second.base_memories[0].grounding_evidence
        )
        == evidence_ids
    )


def test_consensus_grounding_deduplicates_a_claim_shared_by_two_chains(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    first_records = _append_chain(
        store,
        scope,
        prefix="shared-claim-a",
        call_id="shared-call-a",
        result_call_id="shared-call-a",
    )
    _append_chain(
        store,
        scope,
        prefix="shared-claim-b",
        call_id="shared-call-b",
        result_call_id="shared-call-b",
        seconds=(10, 20, 21, 22),
        existing_claim=first_records["claim"],
    )
    first = _make_input(store, scope, release_id, label="shared-claim-source")
    chains = _resolve(store, scope, first).verified_fact_chains
    assert len(chains) == 2
    evidence_ids = tuple(
        dict.fromkeys(
            member.evidence_id
            for chain in chains
            for member in (
                chain.claim,
                chain.tool_call,
                chain.tool_result,
                chain.outcome,
            )
        )
    )
    assert len(evidence_ids) == 7
    application = _commit_application(
        store,
        scope,
        first,
        evidence_ids=evidence_ids,
        label="shared-claim",
    )
    second = _make_input(
        store,
        scope,
        application.result_release_id,
        label="shared-claim-next",
    )
    assert (
        tuple(
            member.evidence_id for member in second.base_memories[0].grounding_evidence
        )
        == evidence_ids
    )


def test_constructible_input_is_not_accepted_as_a_store_capability(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope)
    authentic = _make_input(store, scope, release_id)
    forged_hash = "f" * 64
    forged = dataclasses.replace(
        authentic,
        evidence_snapshot_id=f"esnap_{forged_hash[:24]}",
        evidence_snapshot_content_hash=forged_hash,
    )
    # The DTO is structurally self-consistent and can be hashed, but the public
    # resolver must reload its claimed snapshot instead of trusting it.
    assert len(projection.policy_input_wire_v2(forged)) > 0
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="evidence_snapshot_invalid",
    ):
        _resolve(store, scope, forged)
    forged_event = dataclasses.replace(
        authentic,
        evidence=(
            dataclasses.replace(authentic.evidence[0], session_id="forged-session"),
            *authentic.evidence[1:],
        ),
    )
    assert len(projection.policy_input_wire_v2(forged_event)) > 0
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="input_not_store_authentic",
    ):
        _resolve(store, scope, forged_event)
    forged_base = dataclasses.replace(
        authentic,
        base_memories=(dataclasses.replace(authentic.base_memories[0], value=_OTHER),),
    )
    # Pure wire validation cannot authenticate a store-derived fact value.
    assert len(projection.policy_input_wire_v2(forged_base)) > 0
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="input_not_store_authentic",
    ):
        _resolve(store, scope, forged_base)


def test_snapshot_high_watermark_blocks_late_backfill_and_cutoff_filters_future(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    future = _append(
        store,
        scope,
        label="future-evidence",
        kind=EvidenceKind.FEEDBACK,
        payload="future payload",
        seconds=200,
        sequence_no=5,
    )
    first = _make_input(store, scope, release_id, label="first")
    assert first.evidence_high_watermark == 5
    assert future.evidence_id not in {item.evidence_id for item in first.evidence}

    backfill = _append(
        store,
        scope,
        label="late-backfill",
        kind=EvidenceKind.FEEDBACK,
        payload="late backfill",
        seconds=50,
        sequence_no=6,
    )
    restored = projection.project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=release_id,
        evidence_snapshot_id=first.evidence_snapshot_id,
        provenance_profile=_profile(),
    )
    assert restored == first
    assert backfill.evidence_id not in {item.evidence_id for item in restored.evidence}

    second = _make_input(store, scope, release_id, label="second")
    assert second.evidence_high_watermark == 6
    assert backfill.evidence_id in {item.evidence_id for item in second.evidence}
    assert future.evidence_id not in {item.evidence_id for item in second.evidence}
    assert len(_resolve(store, scope, second).verified_fact_chains) == 1
    assert records["outcome"].evidence_id in {
        item.evidence_id for item in second.evidence
    }


def test_invalid_base_is_rejected_before_snapshot_side_effect(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        projection.make_policy_input_v2(
            store=store,
            scope=scope,
            base_release_id=f"rel_{'f' * 24}",
            cutoff=_BASE + timedelta(seconds=100),
            evidence_snapshot_idempotency_key="snapshot-guard",
            provenance_profile=_profile(),
        )
    records = _append_chain(store, scope)
    value = projection.make_policy_input_v2(
        store=store,
        scope=scope,
        base_release_id=release_id,
        cutoff=_BASE + timedelta(seconds=100),
        evidence_snapshot_idempotency_key="snapshot-guard",
        provenance_profile=_profile(),
    )
    assert records["outcome"].evidence_id in {
        item.evidence_id for item in value.evidence
    }


def test_v1_base_grounding_cannot_use_a_wrong_evidence_kind(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "wrong-kind-base.sqlite3")
    scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="provenance-policy-v2",
        subject_id="wrong-kind-subject",
    )
    wrong = _append(
        store,
        scope,
        label="wrong-kind-base",
        kind=EvidenceKind.TOOL_RESULT,
        payload=f"{_KEY} = {_OLD}",
        seconds=0,
        sequence_no=0,
        session_id="base-session",
        run_id="base-run",
    )
    candidate = store.append_candidate(
        CandidateProposal(
            scope=scope,
            content=wrong.event.payload,
            evidence_ids=(wrong.evidence_id,),
            idempotency_key="wrong-kind-candidate",
        )
    )
    revision = store.append_revision(
        RevisionProposal(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=RevisionOperation.ADD,
            parent_revision_id=None,
            idempotency_key="wrong-kind-revision",
        )
    )
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=(revision.revision_id,)),
        idempotency_key="wrong-kind-release",
    )
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="base_release_invalid",
    ):
        projection.make_policy_input_v2(
            store=store,
            scope=scope,
            base_release_id=release.release_id,
            cutoff=_BASE + timedelta(seconds=100),
            evidence_snapshot_idempotency_key="wrong-kind-snapshot",
            provenance_profile=_profile(),
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"result_parent_role": "claim"},
        {"result_parent_role": "missing"},
        {"outcome_parent_role": "call"},
        {"outcome_parent_role": "missing"},
        {"result_call_id": "other-call"},
        {"result_key": _OTHER_KEY},
        {"call_key": _OTHER_KEY},
        {"call_agent_id": "unknown-agent"},
        {"call_tool_name": "other.lookup", "result_tool_name": "other.lookup"},
        {"result_tool_name": "other.lookup"},
        {"evaluator_id": "unknown.verifier"},
        {"result_run": "other-run"},
        {"claim_run": "other-run"},
        {"outcome_session": "other-session"},
        {"result_trajectory": "other-trajectory"},
        {"outcome_trajectory": "other-trajectory"},
        {"call_trajectory": "other-trajectory"},
        {"call_parent_role": "missing"},
        {"verdict": provenance.VerificationVerdictV1.FAIL},
        {"verdict": provenance.VerificationVerdictV1.UNKNOWN},
        {
            "result_status": provenance.ToolResultStatusV1.ERROR,
            "result_value": None,
        },
    ),
)
def test_broken_untrusted_or_non_pass_chains_abstain(
    tmp_path,
    overrides: dict[str, object],
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope, **overrides)  # type: ignore[arg-type]
    policy_input = _make_input(store, scope, release_id)
    graph = _resolve(store, scope, policy_input)
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (records["outcome"].evidence_id,)


def test_missing_outcome_produces_no_chain_without_fabricating_rejection(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope, include_outcome=False)
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == ()


@pytest.mark.parametrize(
    "claim_kind", (EvidenceKind.FEEDBACK, EvidenceKind.USER_MESSAGE)
)
def test_verified_tool_result_not_untrusted_claim_supplies_the_fact_value(
    tmp_path,
    claim_kind: EvidenceKind,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(
        store,
        scope,
        claim_kind=claim_kind,
        claim_value=_OTHER,
        result_value=_NEW,
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert len(graph.verified_fact_chains) == 1
    assert graph.verified_fact_chains[0].value == _NEW


def test_malformed_result_is_retained_but_cannot_enter_a_verified_chain(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope, include_outcome=False)
    malformed = _append(
        store,
        scope,
        label="malformed-result",
        kind=EvidenceKind.TOOL_RESULT,
        payload="not canonical provenance",
        seconds=20,
        sequence_no=5,
    )
    outcome_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, malformed),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.PASS,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("malformed-outcome"),
    )
    outcome = _append(
        store,
        scope,
        label="malformed-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(outcome_payload),
        seconds=21,
        sequence_no=6,
    )
    policy_input = _make_input(store, scope, release_id)
    projected_malformed = next(
        item
        for item in policy_input.evidence
        if item.evidence_id == malformed.evidence_id
    )
    assert projected_malformed.provenance_payload_sha256 is None
    graph = _resolve(store, scope, policy_input)
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (outcome.evidence_id,)
    assert records["result"].evidence_id in {
        item.evidence_id for item in policy_input.evidence
    }


def test_matching_target_id_with_wrong_full_hash_is_rejected(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope, include_outcome=False)
    result = records["result"]
    replacement = "0" if result.content_hash[-1] != "0" else "1"
    wrong_hash = result.content_hash[:-1] + replacement
    assert wrong_hash[:24] == result.content_hash[:24]
    outcome_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(
            provenance.ProvenanceLinkV1(
                relation=provenance.ProvenanceRelationV1.EVALUATES,
                target_evidence_id=result.evidence_id,
                target_evidence_content_sha256=wrong_hash,
            ),
        ),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.PASS,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("wrong-full-hash"),
    )
    outcome = _append(
        store,
        scope,
        label="wrong-full-hash",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(outcome_payload),
        seconds=14,
        sequence_no=5,
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (outcome.evidence_id,)


def test_multiple_profile_matching_outcomes_for_one_result_are_ambiguous(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    result = records["result"]
    second_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, result),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.FAIL,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("second-outcome"),
    )
    second = _append(
        store,
        scope,
        label="second-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(second_payload),
        seconds=14,
        sequence_no=5,
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert set(graph.rejected_outcome_evidence_ids) == {
        records["outcome"].evidence_id,
        second.evidence_id,
    }


def test_multiple_verified_results_for_one_call_are_ambiguous(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    call = records["call"]
    second_hash = provenance.tool_result_content_sha256_v1(
        call_id="call-1",
        tool_name="code_registry.lookup",
        tool_version_sha256=_TOOL_VERSION,
        status=provenance.ToolResultStatusV1.OK,
        fact_namespace="project_registry",
        key=_KEY,
        value=_OTHER,
        error_code=None,
    )
    second_result_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.TOOL,
            producer_id="code_registry.lookup",
            version_sha256=_TOOL_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.RESULT_OF, call),),
        body=provenance.ToolResultBodyV1(
            call_id="call-1",
            tool_name="code_registry.lookup",
            tool_version_sha256=_TOOL_VERSION,
            status=provenance.ToolResultStatusV1.OK,
            fact_namespace="project_registry",
            key=_KEY,
            value=_OTHER,
            error_code=None,
            result_content_sha256=second_hash,
        ),
        receipt=_receipt("second-result"),
    )
    second_result = _append(
        store,
        scope,
        label="second-result",
        kind=EvidenceKind.TOOL_RESULT,
        payload=provenance.provenance_payload_wire_v1(second_result_payload),
        seconds=14,
        sequence_no=5,
    )
    second_outcome_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(
            _link(
                provenance.ProvenanceRelationV1.EVALUATES,
                second_result,
            ),
        ),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.PASS,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("second-result-outcome"),
    )
    second_outcome = _append(
        store,
        scope,
        label="second-result-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(second_outcome_payload),
        seconds=15,
        sequence_no=6,
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert set(graph.rejected_outcome_evidence_ids) == {
        records["outcome"].evidence_id,
        second_outcome.evidence_id,
    }


def test_distinct_call_events_cannot_reuse_one_logical_call_id(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    first = _append_chain(store, scope, prefix="first-logical-call")
    second = _append_chain(
        store,
        scope,
        prefix="second-logical-call",
        claim_key=_OTHER_KEY,
        call_key=_OTHER_KEY,
        result_key=_OTHER_KEY,
        claim_value=_OTHER,
        result_value=_OTHER,
        seconds=(20, 21, 22, 23),
    )
    assert first["call"].evidence_id != second["call"].evidence_id
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert set(graph.rejected_outcome_evidence_ids) == {
        first["outcome"].evidence_id,
        second["outcome"].evidence_id,
    }


def test_unknown_evaluator_cannot_veto_one_trusted_outcome(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    result = records["result"]
    unknown_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="unknown.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, result),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.FAIL,
            evaluator_id="unknown.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("unknown-outcome"),
    )
    unknown = _append(
        store,
        scope,
        label="unknown-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(unknown_payload),
        seconds=14,
        sequence_no=5,
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert len(graph.verified_fact_chains) == 1
    assert graph.verified_fact_chains[0].outcome.evidence_id == (
        records["outcome"].evidence_id
    )
    assert graph.rejected_outcome_evidence_ids == (unknown.evidence_id,)


def test_cross_run_link_integrity_control_rejects_the_chain(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(
        store,
        scope,
        prefix="run-a",
        include_outcome=False,
        claim_run="run-a",
        call_run="run-a",
        result_run="run-a",
        claim_session="session-a",
        call_session="session-a",
        result_session="session-a",
        claim_trajectory="run-a",
        call_trajectory="run-a",
        result_trajectory="run-a",
    )
    run_b = _append_chain(
        store,
        scope,
        prefix="run-b",
        include_outcome=False,
        claim_key=_OTHER_KEY,
        call_key=_OTHER_KEY,
        result_key=_OTHER_KEY,
        claim_value=_OTHER,
        result_value=_OTHER,
        call_id="call-b",
        result_call_id="call-b",
        claim_run="run-b",
        call_run="run-b",
        result_run="run-b",
        claim_session="session-b",
        call_session="session-b",
        result_session="session-b",
        claim_trajectory="run-b",
        call_trajectory="run-b",
        result_trajectory="run-b",
        seconds=(20, 21, 22, 23),
    )
    shuffled_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="run-a",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(
            _link(
                provenance.ProvenanceRelationV1.EVALUATES,
                run_b["result"],
            ),
        ),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.PASS,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("shuffled-outcome"),
    )
    outcome = _append(
        store,
        scope,
        label="shuffled-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(shuffled_payload),
        seconds=30,
        sequence_no=4,
        session_id="session-a",
        run_id="run-a",
    )
    graph = _resolve(store, scope, _make_input(store, scope, release_id))
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (outcome.evidence_id,)


def test_ingest_precedence_not_backdated_observed_time_controls_links(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    # The recorded event time is deliberately reversed.  Ingest order remains
    # claim -> call -> result -> outcome, so the complete chain is admissible.
    _append_chain(store, scope, seconds=(40, 30, 20, 10))
    policy_input = _make_input(store, scope, release_id)
    assert tuple(item.kind for item in policy_input.evidence[1:]) == (
        "outcome",
        "tool_result",
        "tool_call",
        "feedback",
    )
    graph = _resolve(store, scope, policy_input)
    assert len(graph.verified_fact_chains) == 1


def test_real_child_ingested_before_parent_is_rejected(tmp_path) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    claim_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.USER,
            producer_id="subject-1",
            version_sha256=None,
        ),
        links=(),
        body=provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key=_KEY,
            value=_NEW,
        ),
        receipt=None,
    )
    claim = _append(
        store,
        scope,
        label="late-parent-claim",
        kind=EvidenceKind.FEEDBACK,
        payload=provenance.provenance_payload_wire_v1(claim_payload),
        seconds=10,
        sequence_no=1,
    )
    request_hash = provenance.tool_request_content_sha256_v1(
        tool_name="code_registry.lookup",
        tool_version_sha256=_TOOL_VERSION,
        fact_namespace="project_registry",
        key=_KEY,
    )
    call_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.AGENT,
            producer_id="memory-agent",
            version_sha256=_AGENT_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.TRIGGERED_BY, claim),),
        body=provenance.ToolCallBodyV1(
            call_id="late-call",
            tool_name="code_registry.lookup",
            tool_version_sha256=_TOOL_VERSION,
            fact_namespace="project_registry",
            key=_KEY,
            request_content_sha256=request_hash,
        ),
        receipt=None,
    )
    call_event = EvidenceEvent(
        scope=scope,
        session_id="learning-session",
        run_id="learning-run",
        sequence_no=2,
        kind=EvidenceKind.TOOL_CALL,
        payload=provenance.provenance_payload_wire_v1(call_payload),
        observed_at=_BASE + timedelta(seconds=11),
        idempotency_key="evidence-late-parent-call",
    )
    call_hash = hashlib.sha256(call_event.canonical_bytes()).hexdigest()
    call_target = (f"evd_{call_hash[:24]}", call_hash)
    result_hash = provenance.tool_result_content_sha256_v1(
        call_id="late-call",
        tool_name="code_registry.lookup",
        tool_version_sha256=_TOOL_VERSION,
        status=provenance.ToolResultStatusV1.OK,
        fact_namespace="project_registry",
        key=_KEY,
        value=_NEW,
        error_code=None,
    )
    result_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.TOOL,
            producer_id="code_registry.lookup",
            version_sha256=_TOOL_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.RESULT_OF, call_target),),
        body=provenance.ToolResultBodyV1(
            call_id="late-call",
            tool_name="code_registry.lookup",
            tool_version_sha256=_TOOL_VERSION,
            status=provenance.ToolResultStatusV1.OK,
            fact_namespace="project_registry",
            key=_KEY,
            value=_NEW,
            error_code=None,
            result_content_sha256=result_hash,
        ),
        receipt=_receipt("late-parent-result"),
    )
    result = _append(
        store,
        scope,
        label="late-parent-result",
        kind=EvidenceKind.TOOL_RESULT,
        payload=provenance.provenance_payload_wire_v1(result_payload),
        seconds=12,
        sequence_no=3,
    )
    call = store.append(call_event)
    outcome_payload = provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="learning-run",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=_EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, result),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=provenance.VerificationVerdictV1.PASS,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=_EVALUATOR_VERSION,
        ),
        receipt=_receipt("late-parent-outcome"),
    )
    outcome = _append(
        store,
        scope,
        label="late-parent-outcome",
        kind=EvidenceKind.OUTCOME,
        payload=provenance.provenance_payload_wire_v1(outcome_payload),
        seconds=13,
        sequence_no=4,
    )
    policy_input = _make_input(store, scope, release_id)
    by_id = {item.evidence_id: item for item in policy_input.evidence}
    assert by_id[result.evidence_id].ingest_order < by_id[call.evidence_id].ingest_order
    graph = _resolve(store, scope, policy_input)
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (outcome.evidence_id,)


def test_input_cannot_forge_ingest_precedence_without_member_mismatch(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    claim_records = _append_chain(store, scope, include_outcome=False)
    # A direct DTO mutation cannot override the snapshot-member commitment.
    policy_input = _make_input(store, scope, release_id)
    result_item = next(
        item
        for item in policy_input.evidence
        if item.evidence_id == claim_records["result"].evidence_id
    )
    call_item = next(
        item
        for item in policy_input.evidence
        if item.evidence_id == claim_records["call"].evidence_id
    )
    forged_result = dataclasses.replace(
        result_item,
        ingest_order=call_item.ingest_order - 1,
    )
    forged_evidence = tuple(
        forged_result if item.evidence_id == result_item.evidence_id else item
        for item in policy_input.evidence
    )
    forged = dataclasses.replace(policy_input, evidence=forged_evidence)
    with pytest.raises(
        projection.LocalUpdateProvenanceProjectionError,
        match="input_invariant",
    ):
        projection.policy_input_wire_v2(forged)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: dataclasses.replace(
            value,
            provenance_profile_sha256="f" * 64,
        ),
        lambda value: dataclasses.replace(
            value,
            learning_allowed_kinds=("feedback",),
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_high_watermark=-1,
        ),
        lambda value: dataclasses.replace(
            value,
            evidence_ordering_policy="latest-first",
        ),
        lambda value: dataclasses.replace(
            value,
            evidence=(
                dataclasses.replace(
                    value.evidence[0],
                    provenance_payload_sha256="f" * 64,
                ),
                *value.evidence[1:],
            ),
        ),
    ),
)
def test_policy_input_wire_revalidates_all_commitments(
    tmp_path, mutator: object
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    _append_chain(store, scope)
    policy_input = _make_input(store, scope, release_id)
    forged = mutator(policy_input)  # type: ignore[operator]
    with pytest.raises(projection.LocalUpdateProvenanceProjectionError):
        projection.policy_input_wire_v2(forged)


def test_profile_is_part_of_the_input_hash_and_unknown_profile_abstains(
    tmp_path,
) -> None:
    store, scope, release_id, _old = _fixture(tmp_path)
    records = _append_chain(store, scope)
    accepted = _make_input(store, scope, release_id, label="accepted")
    other_profile = _profile(evaluator_id="other.verifier")
    other = _make_input(
        store,
        scope,
        release_id,
        label="other",
        profile=other_profile,
    )
    assert projection.policy_input_sha256_v2(other) != (
        projection.policy_input_sha256_v2(accepted)
    )
    graph = _resolve(store, scope, other)
    assert graph.verified_fact_chains == ()
    assert graph.rejected_outcome_evidence_ids == (records["outcome"].evidence_id,)
