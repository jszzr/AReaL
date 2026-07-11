# SPDX-License-Identifier: Apache-2.0

"""Tests for the final live-sidecar plus full-ledger analysis gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.memory_service import infbridge_final_analysis as final_analysis
from examples.memory_service import infbridge_receipt_join as receipt_join
from examples.memory_service import infbridge_run_analyzer as run_analyzer
from examples.memory_service import infbridge_sidecar_join as sidecar_join
from examples.memory_service import scoped_codebook_eval as helpfulness
from tests.v2.memory_service.test_infbridge_receipt_join import (
    _PRIVATE_ANSWERS,
    _build_join_case,
    _raw_sidecars,
    _ReceiptJoinCase,
    _synthetic_aggregate,
)

_FINAL_ROOT_DOMAIN = b"areal-memory-infbridge-final-analysis-evidence-v1\0"
_POLICY = "live-sidecar-full-ledger-replay-scored-consistency-only-v1"


@pytest.fixture(scope="module")
def final_case(tmp_path_factory: pytest.TempPathFactory) -> _ReceiptJoinCase:
    return _build_join_case(tmp_path_factory.mktemp("infbridge-final-analysis"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _receipt_gate(
    fixture: _ReceiptJoinCase,
) -> tuple[
    sidecar_join.ValidatedModelSidecarAggregateV1,
    receipt_join.ModelRunReceiptJoinV1,
]:
    aggregate = _synthetic_aggregate(fixture)
    joined = receipt_join._join_validated_model_run_receipts_v1(
        fixture.live.manifest,
        fixture.live.envelope,
        aggregate,
        fixture.receipt_snapshot,
    )
    return aggregate, joined


def _recovered(fixture: _ReceiptJoinCase) -> run_analyzer.RecoveredModelDryRunV1:
    return run_analyzer.recover_infbridge_model_dry_run_v1(
        fixture.ledger_database_path,
        fixture.live.manifest,
        fixture.live.tokenizer,
        fixture.live.envelope,
    )


def _attrition_evaluation(
    recovered: run_analyzer.RecoveredModelDryRunV1,
) -> helpfulness.ModelEvaluationResult:
    return helpfulness.ModelEvaluationResult(
        validity="invalid",
        efficacy="not-assessed",
        safety="not-assessed",
        stale_susceptibility="not-assessed",
        invalid_reasons=("attrition",),
        summary=None,
        attrition=tuple(
            helpfulness.ModelRunAttrition(
                case_index=row.case_index,
                arm=row.arm,
                reason="model_call_failure",
                attempted=True,
            )
            for row in recovered.ledger_attrition
        ),
    )


def _evidence(
    recovered: run_analyzer.RecoveredModelDryRunV1,
    *,
    leakage_found_count: int = 0,
) -> final_analysis._ValidatedModelEvidenceV1:
    succeeded_calls = tuple(
        call for call in recovered.dry_run.calls if call.execution is not None
    )
    sentinels = tuple(
        SimpleNamespace(
            reason="release_found" if index < leakage_found_count else "foreign_scope"
        )
        for index in range(helpfulness.MODEL_CASE_COUNT)
    )
    return final_analysis._ValidatedModelEvidenceV1(
        observations=tuple(
            SimpleNamespace(execution_index=index)
            for index, _ in enumerate(succeeded_calls)
        ),
        schedule=tuple(
            SimpleNamespace(execution_index=index)
            for index, _ in enumerate(succeeded_calls)
        ),
        outcome_slots=tuple((call.case_index, call.arm) for call in succeeded_calls),
        leakage_sentinels=sentinels,
        succeeded_count=recovered.succeeded_count,
        attrition_count=recovered.attrition_count,
        leakage_found_count=leakage_found_count,
    )


def _final_root_value(
    result: final_analysis.InfBridgeFinalAnalysisV1,
) -> dict[str, object]:
    return {
        "attrition_count": result.attrition_count,
        "case_count": result.case_count,
        "envelope_sha256": result.envelope_sha256,
        "evaluation_commitment": {
            "byte_count": result.evaluation_commitment.byte_count,
            "sha256": result.evaluation_commitment.sha256,
            "wire_type": result.evaluation_commitment.wire_type,
        },
        "evaluation_invalid_reasons": list(result.evaluation.invalid_reasons),
        "evaluation_validity": result.evaluation.validity,
        "evidence_outcome_count": result.outcome_count,
        "leakage_found_count": result.leakage_found_count,
        "leakage_sentinel_count": result.leakage_sentinel_count,
        "analysis_invalid_reasons": list(result.analysis_invalid_reasons),
        "analysis_validity": result.analysis_validity,
        "ledger_attrition": [
            {
                "arm": row.arm,
                "case_index": row.case_index,
                "reason": row.reason,
                "slot_index": row.slot_index,
            }
            for row in result.ledger_attrition
        ],
        "ledger_policy": result.ledger_policy,
        "ledger_projection_policy": result.ledger_projection_policy,
        "ledger_run_id": result.ledger_run_id,
        "ledger_run_root_sha256": result.ledger_run_root_sha256,
        "ledger_seal_kind": result.ledger_seal_kind,
        "manifest_sha256": result.manifest_sha256,
        "policy": result.policy,
        "receipt_join_policy": result.receipt_join_policy,
        "receipt_join_root_sha256": result.receipt_join_root_sha256,
        "schema_version": result.schema_version,
        "sidecar_aggregate_root_sha256": result.sidecar_aggregate_root_sha256,
        "sidecar_policy": result.sidecar_policy,
        "slot_count": result.slot_count,
        "succeeded_count": result.succeeded_count,
    }


def _recompute_final_root(result: final_analysis.InfBridgeFinalAnalysisV1) -> str:
    return hashlib.sha256(
        _FINAL_ROOT_DOMAIN + _canonical_bytes(_final_root_value(result))
    ).hexdigest()


def _assert_no_preimages(value: object) -> None:
    assert not isinstance(value, (bytes, bytearray, memoryview))
    if is_dataclass(value):
        for field in fields(value):
            _assert_no_preimages(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_preimages(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_preimages(key)
            _assert_no_preimages(item)


def _assert_final_error(
    error: pytest.ExceptionInfo[final_analysis.InfBridgeFinalAnalysisError],
    reason: str,
) -> None:
    assert type(error.value) is final_analysis.InfBridgeFinalAnalysisError
    assert error.value.reason == reason
    assert str(error.value) == reason


def test_merges_response_free_source_facts_with_recovered_decoded_response(
    final_case: _ReceiptJoinCase,
) -> None:
    fixture = final_case
    recovered = _recovered(fixture)
    call = recovered.dry_run.calls[0]
    execution = call.execution
    assert execution is not None
    capture_response = helpfulness.wire_loads(
        fixture.live.sidecar.capture.exchange.response_wire_utf8.decode("utf-8")
    )
    observation_response = helpfulness.wire_loads(
        fixture.live.sidecar.observations[0].exchange.response_wire_utf8.decode("utf-8")
    )
    assert type(capture_response) is helpfulness.ModelCaptureChildResponse
    assert type(observation_response) is helpfulness.ModelObservationChildResponse
    assert not hasattr(observation_response.observation, "response")
    assert not hasattr(observation_response.observation, "model_call_receipt")

    merged = final_analysis._merge_model_execution_observation_v1(
        slot_index=0,
        registration=fixture.live.registration,
        arm_call=fixture.live.registration.arm_calls[0],
        capture_response=capture_response,
        observation_response=observation_response,
        execution=execution,
    )

    source = observation_response.observation
    assert merged.execution_index == 0
    assert (
        merged.source_kind,
        merged.scope,
        merged.release_id,
        merged.entries,
        merged.reader_audit,
    ) == (
        source.source_kind,
        source.scope,
        source.release_id,
        source.entries,
        source.reader_audit,
    )
    assert (
        merged.capture_pid,
        merged.capture_process_instance_id,
        merged.future_pid,
        merged.future_process_instance_id,
    ) == (
        capture_response.pid,
        capture_response.process_instance_id,
        observation_response.pid,
        observation_response.process_instance_id,
    )
    assert merged.consumer_input_receipt == execution.consumer_input_receipt
    assert merged.model_call_receipt == execution.model_call_receipt
    assert merged.rendered_context_token_count == (
        execution.rendered_context_token_count
    )
    assert merged.response == execution.response == _PRIVATE_ANSWERS[0].decode("ascii")

    arm_offsets = {
        arm_call.arm: offset
        for offset, arm_call in enumerate(fixture.live.registration.arm_calls)
    }
    current_offset = arm_offsets["current_release"]
    oracle_offset = arm_offsets["oracle"]
    current_call = fixture.live.registration.arm_calls[current_offset]
    oracle_call = fixture.live.registration.arm_calls[oracle_offset]
    assert current_call.prepared_call.prompt == oracle_call.prepared_call.prompt
    current_observation = helpfulness.wire_loads(
        fixture.live.sidecar.observations[
            current_offset
        ].exchange.response_wire_utf8.decode("utf-8")
    )
    assert type(current_observation) is helpfulness.ModelObservationChildResponse
    prepared = current_call.prepared_call
    current_execution = helpfulness.ModelCallExecution(
        response="PRIVATE-SAME-PROMPT-ANSWER",
        consumer_input_receipt=prepared.consumer_input_receipt,
        model_call_receipt=prepared.expected_receipt,
        rendered_context_token_count=prepared.rendered_context_token_count,
        valid=True,
        invalid_reason=None,
    )
    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as relabel_error:
        final_analysis._merge_model_execution_observation_v1(
            slot_index=oracle_offset,
            registration=fixture.live.registration,
            arm_call=oracle_call,
            capture_response=capture_response,
            observation_response=current_observation,
            execution=current_execution,
        )
    _assert_final_error(relabel_error, "observation_invalid")


def test_release_found_probe_maps_to_an_explicit_non_foreign_scope_reason(
    final_case: _ReceiptJoinCase,
) -> None:
    fixture = final_case
    found = replace(
        fixture.live.probe_response,
        probe=replace(fixture.live.probe_response.probe, outcome="release_found"),
    )
    sentinel = final_analysis._leakage_sentinel_v1(
        registration=fixture.live.registration,
        capture_response=fixture.live.capture_response,
        probe_response=found,
    )

    assert sentinel.reason == "release_found"
    assert sentinel.execution_index == helpfulness.MODEL_CASE_COUNT * len(
        helpfulness.MODEL_ARMS
    )
    assert sentinel.requested_scope == found.probe.requested_scope
    assert sentinel.companion_scope == (
        fixture.live.registration.identity.references.capture.foreign_scope
    )
    assert sentinel.capture_process_instance_id == (
        fixture.live.capture_response.process_instance_id
    )
    assert sentinel.future_process_instance_id == found.probe.future_process_instance_id


def test_public_gate_order_final_root_and_output_are_answer_preimage_free(
    final_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    evidence = _evidence(recovered)
    evaluation = _attrition_evaluation(recovered)
    events: list[str] = []

    def receipt_gate(*args: object) -> receipt_join.ModelRunReceiptJoinV1:
        assert args == (
            fixture.ledger_database_path,
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
        events.append("receipt")
        return joined

    def recover(*args: object) -> run_analyzer.RecoveredModelDryRunV1:
        assert args == (
            fixture.ledger_database_path,
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
        )
        events.append("recover")
        return recovered

    def aggregate_gate(*args: object) -> object:
        assert args == (
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
        events.append("aggregate")
        return aggregate

    def build(**kwargs: object) -> final_analysis._ValidatedModelEvidenceV1:
        assert kwargs == {
            "manifest": fixture.live.manifest,
            "receipt_gate": joined,
            "recovered": recovered,
            "aggregate": aggregate,
        }
        events.append("build")
        return evidence

    def parent_join(*args: object, **kwargs: object) -> tuple[object, ...]:
        assert args == (evidence.observations, evidence.schedule)
        assert kwargs == {"enforce_scripted_outcomes": False}
        events.append("parent_join")
        return tuple(SimpleNamespace(slot=index) for index in range(3))

    def analyze(**kwargs: object) -> helpfulness.ModelEvaluationResult:
        assert kwargs["manifest"] is fixture.live.manifest
        assert kwargs["tokenizer"] is fixture.live.tokenizer
        assert kwargs["dry_run"] is recovered.dry_run
        assert kwargs["leakage_sentinels"] is evidence.leakage_sentinels
        events.append("analyze")
        return evaluation

    monkeypatch.setattr(
        final_analysis.receipt_join,
        "validate_live_model_run_receipt_join_v1",
        receipt_gate,
    )
    monkeypatch.setattr(
        final_analysis.run_analyzer,
        "recover_infbridge_model_dry_run_v1",
        recover,
    )
    monkeypatch.setattr(
        final_analysis.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        aggregate_gate,
    )
    monkeypatch.setattr(
        final_analysis,
        "_build_validated_model_evidence_v1",
        build,
    )
    monkeypatch.setattr(helpfulness, "parent_join_and_score", parent_join)
    monkeypatch.setattr(helpfulness, "analyze_model_run", analyze)

    result = final_analysis.analyze_live_infbridge_model_run_v1(
        fixture.ledger_database_path,
        fixture.live.manifest,
        fixture.live.tokenizer,
        fixture.live.envelope,
        _raw_sidecars(fixture),
    )

    assert events == [
        "receipt",
        "recover",
        "aggregate",
        "build",
        "parent_join",
        "analyze",
    ]
    assert result.policy == _POLICY
    assert result.evaluation is evaluation
    assert result.evaluation.validity == "invalid"
    assert result.evaluation.invalid_reasons == ("attrition",)
    assert result.analysis_validity == "invalid"
    assert result.analysis_invalid_reasons == ("attrition",)
    assert result.sidecar_aggregate_root_sha256 == (
        aggregate.commitment.aggregate_root_sha256
    )
    assert result.receipt_join_root_sha256 == joined.run_receipt_root_sha256
    assert result.evidence_root_sha256 == _recompute_final_root(result)
    evaluation_wire = helpfulness.wire_dumps(evaluation).encode("utf-8")
    assert result.evaluation_commitment.wire_type == "model_evaluation_result"
    assert result.evaluation_commitment.byte_count == len(evaluation_wire)
    assert (
        result.evaluation_commitment.sha256
        == hashlib.sha256(evaluation_wire).hexdigest()
    )
    _assert_no_preimages(result)
    assert all(
        answer.decode("ascii") not in repr(result) for answer in _PRIVATE_ANSWERS
    )


def test_full_recovery_identity_drift_fails_before_evidence_or_scoring(
    final_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    forbidden: list[str] = []

    monkeypatch.setattr(
        final_analysis.receipt_join,
        "validate_live_model_run_receipt_join_v1",
        lambda *_args: joined,
    )
    monkeypatch.setattr(
        final_analysis.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        lambda *_args: aggregate,
    )

    def fail(name: str):
        def inner(*_args: object, **_kwargs: object) -> None:
            forbidden.append(name)
            raise AssertionError(f"identity drift reached {name}")

        return inner

    monkeypatch.setattr(
        final_analysis,
        "_build_validated_model_evidence_v1",
        fail("build"),
    )
    monkeypatch.setattr(helpfulness, "parent_join_and_score", fail("parent_join"))
    monkeypatch.setattr(helpfulness, "analyze_model_run", fail("analyze"))

    digest = hashlib.sha256(b"different-full-recovery-identity").hexdigest()
    mutations = (
        replace(recovered, run_id=digest),
        replace(recovered, run_root_sha256=digest),
        replace(recovered, manifest_sha256=digest),
        replace(recovered, run_envelope_sha256=digest),
        replace(recovered, seal_kind="complete"),
        replace(recovered, succeeded_count=recovered.succeeded_count + 1),
        replace(recovered, attrition_count=recovered.attrition_count - 1),
        replace(
            recovered,
            dry_run=replace(recovered.dry_run, manifest_sha256=digest),
        ),
        replace(
            recovered,
            ledger_attrition=(
                replace(recovered.ledger_attrition[0], reason="other_failure"),
                *recovered.ledger_attrition[1:],
            ),
        ),
    )
    for mutant in mutations:
        monkeypatch.setattr(
            final_analysis.run_analyzer,
            "recover_infbridge_model_dry_run_v1",
            lambda *_args, value=mutant: value,
        )
        with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as error:
            final_analysis.analyze_live_infbridge_model_run_v1(
                fixture.ledger_database_path,
                fixture.live.manifest,
                fixture.live.tokenizer,
                fixture.live.envelope,
                _raw_sidecars(fixture),
            )
        _assert_final_error(error, "evidence_drift")
        assert forbidden == []


def test_build_rejects_decoded_response_commitment_and_case_leaf_drift(
    final_case: _ReceiptJoinCase,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    first_case = joined.cases[0]
    first_slot = first_case.slots[0]
    assert first_slot.ledger_state == "SUCCEEDED"
    assert first_slot.decoded_response_commitment is not None
    wrong_response_slot = replace(
        first_slot,
        decoded_response_commitment=replace(
            first_slot.decoded_response_commitment,
            byte_count=first_slot.decoded_response_commitment.byte_count + 1,
        ),
    )
    wrong_response_gate = replace(
        joined,
        cases=(
            replace(first_case, slots=(wrong_response_slot, *first_case.slots[1:])),
            *joined.cases[1:],
        ),
    )

    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as response_error:
        final_analysis._build_validated_model_evidence_v1(
            manifest=fixture.live.manifest,
            receipt_gate=wrong_response_gate,
            recovered=recovered,
            aggregate=aggregate,
        )
    _assert_final_error(response_error, "evidence_drift")

    wrong_leaf_case = replace(
        aggregate.cases[0],
        commitment=replace(
            aggregate.cases[0].commitment,
            probe_leaf_sha256=hashlib.sha256(b"different-probe-leaf").hexdigest(),
        ),
    )
    wrong_leaf_aggregate = replace(
        aggregate,
        cases=(wrong_leaf_case, *aggregate.cases[1:]),
    )
    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as leaf_error:
        final_analysis._build_validated_model_evidence_v1(
            manifest=fixture.live.manifest,
            receipt_gate=joined,
            recovered=recovered,
            aggregate=wrong_leaf_aggregate,
        )
    _assert_final_error(leaf_error, "evidence_drift")


def test_evaluation_attrition_must_exactly_match_the_sealed_slot_projection(
    final_case: _ReceiptJoinCase,
) -> None:
    fixture = final_case
    _aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    evidence = _evidence(recovered)
    evaluation = _attrition_evaluation(recovered)
    forged = replace(evaluation, attrition=evaluation.attrition[:-1])

    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as error:
        final_analysis._validate_evaluation_attrition_v1(
            receipt_gate=joined,
            evidence=evidence,
            evaluation=forged,
        )
    _assert_final_error(error, "analysis_invalid")


def test_parent_join_and_analyzer_exceptions_are_closed_before_report_creation(
    final_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    evidence = _evidence(recovered)
    analyze_calls: list[str] = []

    monkeypatch.setattr(
        final_analysis.receipt_join,
        "validate_live_model_run_receipt_join_v1",
        lambda *_args: joined,
    )
    monkeypatch.setattr(
        final_analysis.run_analyzer,
        "recover_infbridge_model_dry_run_v1",
        lambda *_args: recovered,
    )
    monkeypatch.setattr(
        final_analysis.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        lambda *_args: aggregate,
    )
    monkeypatch.setattr(
        final_analysis,
        "_build_validated_model_evidence_v1",
        lambda **_kwargs: evidence,
    )

    def parent_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private parent failure")

    def track_analyze(**_kwargs: object) -> None:
        analyze_calls.append("analyze")
        raise AssertionError("analyzer reached after parent failure")

    monkeypatch.setattr(helpfulness, "parent_join_and_score", parent_failure)
    monkeypatch.setattr(helpfulness, "analyze_model_run", track_analyze)
    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as parent_error:
        final_analysis.analyze_live_infbridge_model_run_v1(
            fixture.ledger_database_path,
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
    _assert_final_error(parent_error, "analysis_invalid")
    assert analyze_calls == []

    monkeypatch.setattr(
        helpfulness,
        "parent_join_and_score",
        lambda *_args, **_kwargs: tuple(
            SimpleNamespace(slot=index) for index in range(recovered.succeeded_count)
        ),
    )

    def analysis_failure(**_kwargs: object) -> None:
        analyze_calls.append("analyze")
        raise RuntimeError("private analyzer failure")

    monkeypatch.setattr(helpfulness, "analyze_model_run", analysis_failure)
    with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as analysis_error:
        final_analysis.analyze_live_infbridge_model_run_v1(
            fixture.ledger_database_path,
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
    _assert_final_error(analysis_error, "analysis_invalid")
    assert analyze_calls == ["analyze"]


def test_attrition_does_not_hide_release_found_from_the_outer_analysis_signal(
    final_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    evidence = _evidence(recovered, leakage_found_count=1)
    evaluation = _attrition_evaluation(recovered)

    monkeypatch.setattr(
        final_analysis.receipt_join,
        "validate_live_model_run_receipt_join_v1",
        lambda *_args: joined,
    )
    monkeypatch.setattr(
        final_analysis.run_analyzer,
        "recover_infbridge_model_dry_run_v1",
        lambda *_args: recovered,
    )
    monkeypatch.setattr(
        final_analysis.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        lambda *_args: aggregate,
    )
    monkeypatch.setattr(
        final_analysis,
        "_build_validated_model_evidence_v1",
        lambda **_kwargs: evidence,
    )
    monkeypatch.setattr(
        helpfulness,
        "parent_join_and_score",
        lambda *_args, **_kwargs: tuple(
            SimpleNamespace(slot=index) for index in range(recovered.succeeded_count)
        ),
    )

    def analyze(**kwargs: object) -> helpfulness.ModelEvaluationResult:
        sentinels = kwargs["leakage_sentinels"]
        assert type(sentinels) is tuple
        assert sum(item.reason == "release_found" for item in sentinels) == 1
        return evaluation

    monkeypatch.setattr(helpfulness, "analyze_model_run", analyze)

    result = final_analysis.analyze_live_infbridge_model_run_v1(
        fixture.ledger_database_path,
        fixture.live.manifest,
        fixture.live.tokenizer,
        fixture.live.envelope,
        _raw_sidecars(fixture),
    )

    assert result.leakage_found_count == 1
    assert result.evaluation.validity == "invalid"
    assert result.evaluation.invalid_reasons == ("attrition",)
    assert result.analysis_validity == "invalid"
    assert result.analysis_invalid_reasons == (
        "cross_scope_leakage",
        "attrition",
    )


def test_memory_database_drift_after_full_replay_fails_before_scoring(
    final_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = final_case
    aggregate, joined = _receipt_gate(fixture)
    recovered = _recovered(fixture)
    database_path = Path(fixture.live.capture_request.database_path)
    original_bytes = database_path.read_bytes()
    events: list[str] = []

    monkeypatch.setattr(
        final_analysis.receipt_join,
        "validate_live_model_run_receipt_join_v1",
        lambda *_args: joined,
    )

    def recover_then_mutate(*_args: object) -> run_analyzer.RecoveredModelDryRunV1:
        events.append("recover")
        with database_path.open("ab") as stream:
            stream.write(b"post-full-replay-memory-drift")
        return recovered

    def real_case_sweep(*_args: object) -> object:
        events.append("aggregate")
        sidecar_join.validate_live_model_case_sidecar_v1(
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            fixture.live.sidecar,
        )
        return aggregate

    monkeypatch.setattr(
        final_analysis.run_analyzer,
        "recover_infbridge_model_dry_run_v1",
        recover_then_mutate,
    )
    monkeypatch.setattr(
        final_analysis.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        real_case_sweep,
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        events.append("forbidden")
        raise AssertionError("Memory drift reached scoring")

    monkeypatch.setattr(
        final_analysis,
        "_build_validated_model_evidence_v1",
        forbidden,
    )
    monkeypatch.setattr(helpfulness, "parent_join_and_score", forbidden)
    monkeypatch.setattr(helpfulness, "analyze_model_run", forbidden)

    try:
        with pytest.raises(final_analysis.InfBridgeFinalAnalysisError) as error:
            final_analysis.analyze_live_infbridge_model_run_v1(
                fixture.ledger_database_path,
                fixture.live.manifest,
                fixture.live.tokenizer,
                fixture.live.envelope,
                _raw_sidecars(fixture),
            )
        _assert_final_error(error, "sidecar_invalid")
    finally:
        database_path.write_bytes(original_bytes)

    assert events == ["recover", "aggregate"]
    assert (
        sidecar_join.validate_live_model_case_sidecar_v1(
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            fixture.live.sidecar,
        ).commitment.case_index
        == 0
    )
