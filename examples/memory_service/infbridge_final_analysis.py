# SPDX-License-Identifier: Apache-2.0

"""Full-replay and scoring gate for the audited Memory model run.

This is the first layer allowed to decode model answers or invoke scorer-owned
truth.  It first completes the 64-case response-free sidecar/receipt gate,
fully replays the sealed InfBridge ledger, then validates all raw sidecars once
more before merging source evidence with successful model executions.

The final report contains no answer preimages.  Its evidence root proves local
consistency among validated snapshots and a deterministic evaluation result;
it is not a freshness proof, remote-model attestation, cross-database atomic
snapshot, signature, or defense against a writer who can recompute every root.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from examples.memory_service import infbridge_receipt_join as receipt_join
from examples.memory_service import infbridge_run_analyzer as run_analyzer
from examples.memory_service import infbridge_sidecar_join as sidecars
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    RunEnvelopeV2,
)
from examples.memory_service.infbridge_run_analyzer import (
    LedgerModelAttritionV1,
    RecoveredModelDryRunV1,
)
from examples.memory_service.infbridge_run_ledger import RunLedgerError

__all__ = [
    "CanonicalEvaluationCommitmentV1",
    "InfBridgeFinalAnalysisError",
    "InfBridgeFinalAnalysisV1",
    "analyze_live_infbridge_model_run_v1",
]


class InfBridgeFinalAnalysisError(RuntimeError):
    """Closed reason for refusing full replay or evidence-to-score binding."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("final analysis reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationCommitmentV1:
    wire_type: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InfBridgeFinalAnalysisV1:
    """Constructible answer-preimage-free report, not a trust capability.

    Consumers must use ``analysis_validity`` and ``analysis_invalid_reasons``
    for the whole run.  ``evaluation`` preserves the legacy evaluator result,
    whose attrition precedence can otherwise hide a simultaneous leakage probe.
    """

    schema_version: int
    policy: str
    manifest_sha256: str
    envelope_sha256: str
    sidecar_policy: str
    sidecar_aggregate_root_sha256: str
    receipt_join_policy: str
    receipt_join_root_sha256: str
    ledger_projection_policy: str
    ledger_policy: str
    ledger_run_id: str
    ledger_run_root_sha256: str
    ledger_seal_kind: str
    case_count: int
    slot_count: int
    succeeded_count: int
    attrition_count: int
    outcome_count: int
    leakage_sentinel_count: int
    leakage_found_count: int
    analysis_validity: str
    analysis_invalid_reasons: tuple[str, ...]
    ledger_attrition: tuple[LedgerModelAttritionV1, ...]
    evaluation_commitment: CanonicalEvaluationCommitmentV1
    evaluation: helpfulness.ModelEvaluationResult
    evidence_root_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedModelEvidenceV1:
    observations: tuple[helpfulness.ExecutionObservation, ...]
    schedule: tuple[helpfulness.ParentScheduleItem, ...]
    outcome_slots: tuple[tuple[int, str], ...]
    leakage_sentinels: tuple[helpfulness.LeakageSentinelTrace, ...]
    succeeded_count: int
    attrition_count: int
    leakage_found_count: int


_SCHEMA_VERSION = 1
_POLICY = "live-sidecar-full-ledger-replay-scored-consistency-only-v1"
_EVIDENCE_ROOT_DOMAIN = b"areal-memory-infbridge-final-analysis-evidence-v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _load_canonical_child_response(
    exchange: sidecars.CanonicalChildExchangeV1,
    expected_type: type[object],
) -> object:
    if (
        type(exchange) is not sidecars.CanonicalChildExchangeV1
        or type(exchange.response_wire_utf8) is not bytes
        or not exchange.response_wire_utf8
    ):
        raise InfBridgeFinalAnalysisError("observation_invalid")
    try:
        text = exchange.response_wire_utf8.decode("utf-8", errors="strict")
        response = helpfulness.wire_loads(text)
        if (
            type(response) is not expected_type
            or helpfulness.wire_dumps(response).encode("utf-8", errors="strict")
            != exchange.response_wire_utf8
        ):
            raise InfBridgeFinalAnalysisError("observation_invalid")
        return response
    except InfBridgeFinalAnalysisError:
        raise
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        helpfulness.WireProtocolError,
    ) as error:
        raise InfBridgeFinalAnalysisError("observation_invalid") from error


def _merge_model_execution_observation_v1(
    *,
    slot_index: int,
    registration: helpfulness.ModelCaseRegistration,
    arm_call: helpfulness.ModelArmCallRegistration,
    capture_response: helpfulness.ModelCaptureChildResponse,
    observation_response: helpfulness.ModelObservationChildResponse,
    execution: helpfulness.ModelCallExecution,
) -> helpfulness.ExecutionObservation:
    observation = observation_response.observation
    prepared = arm_call.prepared_call
    arm_offsets = tuple(
        index
        for index, registered in enumerate(registration.arm_calls)
        if helpfulness._exact_typed_tree_equal(registered, arm_call)
    )
    if (
        type(slot_index) is not int
        or len(arm_offsets) != 1
        or slot_index
        != registration.identity.case.case_index * len(helpfulness.MODEL_ARMS)
        + arm_offsets[0]
        or type(capture_response) is not helpfulness.ModelCaptureChildResponse
        or type(observation_response) is not helpfulness.ModelObservationChildResponse
        or type(execution) is not helpfulness.ModelCallExecution
        or type(execution.response) is not str
        or execution.valid is not True
        or execution.invalid_reason is not None
        or not helpfulness._exact_typed_tree_equal(
            execution.consumer_input_receipt,
            prepared.consumer_input_receipt,
        )
        or not helpfulness._exact_typed_tree_equal(
            execution.model_call_receipt,
            prepared.expected_receipt,
        )
        or execution.rendered_context_token_count
        != prepared.rendered_context_token_count
        or observation.rendered_context_sha256 != arm_call.rendered_context_sha256
        or observation.rendered_context_utf8_bytes
        != arm_call.rendered_context_utf8_bytes
    ):
        raise InfBridgeFinalAnalysisError("observation_invalid")
    merged = helpfulness.ExecutionObservation(
        execution_index=slot_index,
        source_kind=observation.source_kind,
        scope=observation.scope,
        capture_session_ids=(capture_response.references.capture.capture_session_ids),
        future_session_id=observation.future_session_id,
        future_run_id=observation.future_run_id,
        capture_pid=capture_response.pid,
        future_pid=observation.future_pid,
        capture_process_instance_id=capture_response.process_instance_id,
        future_process_instance_id=observation.future_process_instance_id,
        release_id=observation.release_id,
        eligible_ids=observation.eligible_ids,
        retrieved_ids=observation.retrieved_ids,
        returned_ids=observation.returned_ids,
        source_evidence_ids=observation.source_evidence_ids,
        entries=observation.entries,
        reader_audit=observation.reader_audit,
        rendered_context_sha256=observation.rendered_context_sha256,
        rendered_context_utf8_bytes=observation.rendered_context_utf8_bytes,
        rendered_context_token_count=execution.rendered_context_token_count,
        consumer_input_receipt=execution.consumer_input_receipt,
        model_call_receipt=execution.model_call_receipt,
        query_sha256=registration.query_sha256,
        history_length=execution.consumer_input_receipt.received_history_length,
        response=execution.response,
    )
    try:
        scheduled = helpfulness.make_parent_schedule_item(
            execution_index=slot_index,
            case=registration.identity.case,
            references=registration.identity.references,
            arm=arm_call.arm,
        )
        if (
            merged.source_kind != scheduled.source_kind
            or merged.scope != scheduled.scope
            or merged.release_id != scheduled.release_id
            or merged.capture_session_ids != scheduled.capture_session_ids
        ):
            raise helpfulness.ObservationValidationError("assignment_mismatch")
        helpfulness._validate_parent_source_contract(merged, scheduled)
    except (
        TypeError,
        ValueError,
        helpfulness.ChildExecutionValidationError,
        helpfulness.ObservationValidationError,
    ) as error:
        raise InfBridgeFinalAnalysisError("observation_invalid") from error
    return merged


def _leakage_sentinel_v1(
    *,
    registration: helpfulness.ModelCaseRegistration,
    capture_response: helpfulness.ModelCaptureChildResponse,
    probe_response: helpfulness.ModelLeakageProbeChildResponse,
) -> helpfulness.LeakageSentinelTrace:
    case = registration.identity.case
    references = registration.identity.references
    probe = probe_response.probe
    if probe.outcome == "release_not_found":
        reason = "foreign_scope"
    elif probe.outcome == "release_found":
        reason = "release_found"
    else:
        raise InfBridgeFinalAnalysisError("observation_invalid")
    return helpfulness.LeakageSentinelTrace(
        schema_version=helpfulness.SCHEMA_VERSION,
        case_id=case.case_id,
        case_manifest_sha256=registration.identity.case_manifest_sha256,
        execution_index=(
            helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS) + case.case_index
        ),
        requested_scope=probe.requested_scope,
        companion_scope=references.capture.foreign_scope,
        foreign_release_id=probe.release_id,
        foreign_evidence_id=references.capture.foreign_evidence_id,
        future_session_id=probe.future_session_id,
        future_run_id=probe.future_run_id,
        capture_pid=capture_response.pid,
        future_pid=probe.future_pid,
        capture_process_instance_id=capture_response.process_instance_id,
        future_process_instance_id=probe.future_process_instance_id,
        reason=reason,
        history_length=0,
    )


def _build_validated_model_evidence_v1(
    *,
    manifest: helpfulness.ModelRunManifest,
    receipt_gate: receipt_join.ModelRunReceiptJoinV1,
    recovered: RecoveredModelDryRunV1,
    aggregate: sidecars.ValidatedModelSidecarAggregateV1,
) -> _ValidatedModelEvidenceV1:
    if (
        type(receipt_gate) is not receipt_join.ModelRunReceiptJoinV1
        or type(recovered) is not RecoveredModelDryRunV1
        or type(aggregate) is not sidecars.ValidatedModelSidecarAggregateV1
        or type(recovered.dry_run) is not helpfulness.ModelDryRunResult
        or type(recovered.dry_run.calls) is not tuple
        or len(recovered.dry_run.calls) != receipt_gate.slot_count
        or len(aggregate.cases) != helpfulness.MODEL_CASE_COUNT
    ):
        raise InfBridgeFinalAnalysisError("evidence_drift")

    observations: list[helpfulness.ExecutionObservation] = []
    schedule: list[helpfulness.ParentScheduleItem] = []
    outcome_slots: list[tuple[int, str]] = []
    leakage_sentinels: list[helpfulness.LeakageSentinelTrace] = []
    succeeded_count = 0
    attrition_count = 0
    call_index = 0
    for case_index, (registration, validated_case, gate_case) in enumerate(
        zip(manifest.cases, aggregate.cases, receipt_gate.cases, strict=True)
    ):
        if (
            registration.identity.case.case_index != case_index
            or validated_case.commitment.case_index != case_index
            or gate_case.case_index != case_index
            or validated_case.commitment.case_root_sha256
            != gate_case.sidecar_case_root_sha256
            or validated_case.commitment.capture_leaf_sha256
            != gate_case.capture_leaf_sha256
            or validated_case.commitment.observation_leaf_sha256s
            != gate_case.observation_leaf_sha256s
            or validated_case.commitment.probe_leaf_sha256
            != gate_case.probe_leaf_sha256
        ):
            raise InfBridgeFinalAnalysisError("evidence_drift")
        capture_value = _load_canonical_child_response(
            validated_case.capture.exchange,
            helpfulness.ModelCaptureChildResponse,
        )
        assert type(capture_value) is helpfulness.ModelCaptureChildResponse
        capture_response = capture_value
        if not helpfulness._exact_typed_tree_equal(
            capture_response.references,
            registration.identity.references,
        ):
            raise InfBridgeFinalAnalysisError("evidence_drift")

        if len(validated_case.observations) != len(registration.arm_calls):
            raise InfBridgeFinalAnalysisError("evidence_drift")
        for arm_offset, (arm_call, sidecar, gate_slot) in enumerate(
            zip(
                registration.arm_calls,
                validated_case.observations,
                gate_case.slots,
                strict=True,
            )
        ):
            slot_index = case_index * len(helpfulness.MODEL_ARMS) + arm_offset
            dry_call = recovered.dry_run.calls[call_index]
            call_index += 1
            if (
                type(dry_call) is not helpfulness.ModelDryRunCall
                or dry_call.case_index != case_index
                or dry_call.arm != arm_call.arm
                or dry_call.attempt_index != 0
                or gate_slot.slot_index != slot_index
                or gate_slot.case_index != case_index
                or gate_slot.arm != arm_call.arm
            ):
                raise InfBridgeFinalAnalysisError("evidence_drift")
            execution = dry_call.execution
            if execution is None:
                if gate_slot.ledger_state != "ATTRITION":
                    raise InfBridgeFinalAnalysisError("evidence_drift")
                attrition_count += 1
                continue
            if gate_slot.ledger_state != "SUCCEEDED":
                raise InfBridgeFinalAnalysisError("evidence_drift")
            try:
                response_bytes = execution.response.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise InfBridgeFinalAnalysisError("evidence_drift") from error
            if (
                gate_slot.decoded_response_commitment is None
                or gate_slot.decoded_response_commitment.byte_count
                != len(response_bytes)
                or gate_slot.decoded_response_commitment.sha256
                != hashlib.sha256(response_bytes).hexdigest()
            ):
                raise InfBridgeFinalAnalysisError("evidence_drift")
            observation_value = _load_canonical_child_response(
                sidecar.exchange,
                helpfulness.ModelObservationChildResponse,
            )
            assert type(observation_value) is helpfulness.ModelObservationChildResponse
            observation = _merge_model_execution_observation_v1(
                slot_index=slot_index,
                registration=registration,
                arm_call=arm_call,
                capture_response=capture_response,
                observation_response=observation_value,
                execution=execution,
            )
            observations.append(observation)
            try:
                schedule.append(
                    helpfulness.make_parent_schedule_item(
                        execution_index=slot_index,
                        case=registration.identity.case,
                        references=registration.identity.references,
                        arm=arm_call.arm,
                    )
                )
            except (
                TypeError,
                ValueError,
                helpfulness.ChildExecutionValidationError,
            ) as error:
                raise InfBridgeFinalAnalysisError("observation_invalid") from error
            outcome_slots.append((case_index, arm_call.arm))
            succeeded_count += 1

        probe_value = _load_canonical_child_response(
            validated_case.probe.exchange,
            helpfulness.ModelLeakageProbeChildResponse,
        )
        assert type(probe_value) is helpfulness.ModelLeakageProbeChildResponse
        leakage_sentinels.append(
            _leakage_sentinel_v1(
                registration=registration,
                capture_response=capture_response,
                probe_response=probe_value,
            )
        )

    sentinels = tuple(leakage_sentinels)
    leakage_found_count = sum(
        sentinel.reason != "foreign_scope" for sentinel in sentinels
    )
    if (
        call_index != receipt_gate.slot_count
        or succeeded_count != receipt_gate.succeeded_count
        or attrition_count != receipt_gate.attrition_count
        or succeeded_count != recovered.succeeded_count
        or attrition_count != recovered.attrition_count
        or len(sentinels) != helpfulness.MODEL_CASE_COUNT
    ):
        raise InfBridgeFinalAnalysisError("evidence_drift")
    return _ValidatedModelEvidenceV1(
        observations=tuple(observations),
        schedule=tuple(schedule),
        outcome_slots=tuple(outcome_slots),
        leakage_sentinels=sentinels,
        succeeded_count=succeeded_count,
        attrition_count=attrition_count,
        leakage_found_count=leakage_found_count,
    )


def _ledger_attrition_value(
    attrition: tuple[LedgerModelAttritionV1, ...],
) -> list[dict[str, object]]:
    return [
        {
            "arm": row.arm,
            "case_index": row.case_index,
            "reason": row.reason,
            "slot_index": row.slot_index,
        }
        for row in attrition
    ]


def _expected_ledger_attrition_v1(
    receipt_gate: receipt_join.ModelRunReceiptJoinV1,
) -> tuple[LedgerModelAttritionV1, ...]:
    rows: list[LedgerModelAttritionV1] = []
    for case in receipt_gate.cases:
        for slot in case.slots:
            if slot.ledger_state != "ATTRITION":
                continue
            if type(slot.terminal_reason) is not str or not slot.terminal_reason:
                raise InfBridgeFinalAnalysisError("evidence_drift")
            rows.append(
                LedgerModelAttritionV1(
                    slot_index=slot.slot_index,
                    case_index=slot.case_index,
                    arm=slot.arm,
                    reason=slot.terminal_reason,
                )
            )
    return tuple(rows)


def _validate_evaluation_attrition_v1(
    *,
    receipt_gate: receipt_join.ModelRunReceiptJoinV1,
    evidence: _ValidatedModelEvidenceV1,
    evaluation: helpfulness.ModelEvaluationResult,
) -> None:
    expected = tuple(
        helpfulness.ModelRunAttrition(
            case_index=slot.case_index,
            arm=slot.arm,
            reason="model_call_failure",
            attempted=True,
        )
        for case in receipt_gate.cases
        for slot in case.slots
        if slot.ledger_state == "ATTRITION"
    )
    if evidence.attrition_count:
        valid = bool(
            evaluation.validity == "invalid"
            and evaluation.invalid_reasons == ("attrition",)
            and helpfulness._exact_typed_tree_equal(evaluation.attrition, expected)
            and len(expected) == evidence.attrition_count
        )
    else:
        valid = bool(
            not expected
            and "attrition" not in evaluation.invalid_reasons
            and evaluation.attrition == ()
        )
    if not valid:
        raise InfBridgeFinalAnalysisError("analysis_invalid")


def _final_report_v1(
    *,
    receipt_gate: receipt_join.ModelRunReceiptJoinV1,
    recovered: RecoveredModelDryRunV1,
    aggregate: sidecars.ValidatedModelSidecarAggregateV1,
    evidence: _ValidatedModelEvidenceV1,
    evaluation: helpfulness.ModelEvaluationResult,
) -> InfBridgeFinalAnalysisV1:
    try:
        evaluation_wire = helpfulness.wire_dumps(evaluation).encode(
            "utf-8", errors="strict"
        )
        decoded = helpfulness.wire_loads(evaluation_wire.decode("utf-8"))
        if not helpfulness._exact_typed_tree_equal(decoded, evaluation):
            raise InfBridgeFinalAnalysisError("observation_invalid")
    except InfBridgeFinalAnalysisError:
        raise
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        helpfulness.WireProtocolError,
    ) as error:
        raise InfBridgeFinalAnalysisError("observation_invalid") from error
    evaluation_commitment = CanonicalEvaluationCommitmentV1(
        wire_type="model_evaluation_result",
        byte_count=len(evaluation_wire),
        sha256=hashlib.sha256(evaluation_wire).hexdigest(),
    )
    analysis_invalid_reasons = tuple(evaluation.invalid_reasons)
    if (
        evidence.leakage_found_count
        and "cross_scope_leakage" not in analysis_invalid_reasons
    ):
        analysis_invalid_reasons = (
            "cross_scope_leakage",
            *analysis_invalid_reasons,
        )
    analysis_validity = "invalid" if analysis_invalid_reasons else evaluation.validity
    case_count = helpfulness.MODEL_CASE_COUNT
    slot_count = case_count * len(helpfulness.MODEL_ARMS)
    root_value = {
        "attrition_count": evidence.attrition_count,
        "case_count": case_count,
        "envelope_sha256": receipt_gate.envelope_sha256,
        "evaluation_commitment": {
            "byte_count": evaluation_commitment.byte_count,
            "sha256": evaluation_commitment.sha256,
            "wire_type": evaluation_commitment.wire_type,
        },
        "evaluation_invalid_reasons": list(evaluation.invalid_reasons),
        "evaluation_validity": evaluation.validity,
        "evidence_outcome_count": len(evidence.outcome_slots),
        "leakage_found_count": evidence.leakage_found_count,
        "leakage_sentinel_count": len(evidence.leakage_sentinels),
        "analysis_invalid_reasons": list(analysis_invalid_reasons),
        "analysis_validity": analysis_validity,
        "ledger_attrition": _ledger_attrition_value(recovered.ledger_attrition),
        "ledger_policy": receipt_gate.ledger_policy,
        "ledger_projection_policy": receipt_gate.ledger_projection_policy,
        "ledger_run_id": recovered.run_id,
        "ledger_run_root_sha256": recovered.run_root_sha256,
        "ledger_seal_kind": recovered.seal_kind,
        "manifest_sha256": receipt_gate.manifest_sha256,
        "policy": _POLICY,
        "receipt_join_policy": receipt_gate.policy,
        "receipt_join_root_sha256": receipt_gate.run_receipt_root_sha256,
        "schema_version": _SCHEMA_VERSION,
        "sidecar_aggregate_root_sha256": (aggregate.commitment.aggregate_root_sha256),
        "sidecar_policy": aggregate.commitment.policy,
        "slot_count": slot_count,
        "succeeded_count": evidence.succeeded_count,
    }
    evidence_root_sha256 = hashlib.sha256(
        _EVIDENCE_ROOT_DOMAIN + _canonical_json_bytes(root_value)
    ).hexdigest()
    return InfBridgeFinalAnalysisV1(
        schema_version=_SCHEMA_VERSION,
        policy=_POLICY,
        manifest_sha256=receipt_gate.manifest_sha256,
        envelope_sha256=receipt_gate.envelope_sha256,
        sidecar_policy=aggregate.commitment.policy,
        sidecar_aggregate_root_sha256=(aggregate.commitment.aggregate_root_sha256),
        receipt_join_policy=receipt_gate.policy,
        receipt_join_root_sha256=receipt_gate.run_receipt_root_sha256,
        ledger_projection_policy=receipt_gate.ledger_projection_policy,
        ledger_policy=receipt_gate.ledger_policy,
        ledger_run_id=recovered.run_id,
        ledger_run_root_sha256=recovered.run_root_sha256,
        ledger_seal_kind=recovered.seal_kind,
        case_count=case_count,
        slot_count=slot_count,
        succeeded_count=evidence.succeeded_count,
        attrition_count=evidence.attrition_count,
        outcome_count=len(evidence.outcome_slots),
        leakage_sentinel_count=len(evidence.leakage_sentinels),
        leakage_found_count=evidence.leakage_found_count,
        analysis_validity=analysis_validity,
        analysis_invalid_reasons=analysis_invalid_reasons,
        ledger_attrition=recovered.ledger_attrition,
        evaluation_commitment=evaluation_commitment,
        evaluation=evaluation,
        evidence_root_sha256=evidence_root_sha256,
    )


def analyze_live_infbridge_model_run_v1(
    ledger_database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    model_sidecars: tuple[sidecars.ModelCaseSidecarV1, ...],
) -> InfBridgeFinalAnalysisV1:
    """Fully replay and score only after the global response-free gate."""

    try:
        receipt_gate = receipt_join.validate_live_model_run_receipt_join_v1(
            ledger_database_path,
            manifest,
            tokenizer,
            envelope,
            model_sidecars,
        )
    except receipt_join.InfBridgeReceiptJoinError as error:
        raise InfBridgeFinalAnalysisError("receipt_join_invalid") from error
    try:
        recovered = run_analyzer.recover_infbridge_model_dry_run_v1(
            ledger_database_path,
            manifest,
            tokenizer,
            envelope,
        )
    except (run_analyzer.InfBridgeRunAnalyzerError, RunLedgerError) as error:
        raise InfBridgeFinalAnalysisError("ledger_replay_invalid") from error
    if (
        type(recovered.schema_version) is not int
        or recovered.schema_version != 1
        or type(recovered.dry_run) is not helpfulness.ModelDryRunResult
        or recovered.dry_run.manifest_sha256 != receipt_gate.manifest_sha256
        or type(recovered.ledger_attrition) is not tuple
        or not helpfulness._exact_typed_tree_equal(
            recovered.ledger_attrition,
            _expected_ledger_attrition_v1(receipt_gate),
        )
        or recovered.run_id != receipt_gate.ledger_run_id
        or recovered.run_root_sha256 != receipt_gate.ledger_run_root_sha256
        or recovered.manifest_sha256 != receipt_gate.manifest_sha256
        or recovered.run_envelope_sha256 != receipt_gate.envelope_sha256
        or recovered.seal_kind != receipt_gate.ledger_seal_kind
        or recovered.succeeded_count != receipt_gate.succeeded_count
        or recovered.attrition_count != receipt_gate.attrition_count
    ):
        raise InfBridgeFinalAnalysisError("evidence_drift")
    try:
        aggregate = sidecars.validate_live_model_sidecar_aggregate_v1(
            manifest,
            tokenizer,
            envelope,
            model_sidecars,
        )
    except sidecars.ModelSidecarJoinError as error:
        raise InfBridgeFinalAnalysisError("sidecar_invalid") from error
    if (
        aggregate.commitment.aggregate_root_sha256
        != receipt_gate.sidecar_aggregate_root_sha256
        or aggregate.commitment.manifest_sha256 != receipt_gate.manifest_sha256
        or aggregate.commitment.envelope_sha256 != receipt_gate.envelope_sha256
        or aggregate.commitment.policy != receipt_gate.sidecar_policy
    ):
        raise InfBridgeFinalAnalysisError("evidence_drift")
    evidence = _build_validated_model_evidence_v1(
        manifest=manifest,
        receipt_gate=receipt_gate,
        recovered=recovered,
        aggregate=aggregate,
    )
    try:
        traces = helpfulness.parent_join_and_score(
            evidence.observations,
            evidence.schedule,
            enforce_scripted_outcomes=False,
        )
    except helpfulness.ObservationValidationError as error:
        raise InfBridgeFinalAnalysisError("observation_invalid") from error
    except Exception as error:
        raise InfBridgeFinalAnalysisError("analysis_invalid") from error
    if len(traces) != len(evidence.outcome_slots):
        raise InfBridgeFinalAnalysisError("observation_invalid")
    outcomes = tuple(
        helpfulness.ModelArmOutcome(
            case_index=case_index,
            arm=arm,
            trace=trace,
        )
        for (case_index, arm), trace in zip(
            evidence.outcome_slots,
            traces,
            strict=True,
        )
    )
    try:
        evaluation = helpfulness.analyze_model_run(
            manifest=manifest,
            tokenizer=tokenizer,
            dry_run=recovered.dry_run,
            outcomes=outcomes,
            leakage_sentinels=evidence.leakage_sentinels,
        )
    except Exception as error:
        raise InfBridgeFinalAnalysisError("analysis_invalid") from error
    if type(evaluation) is not helpfulness.ModelEvaluationResult:
        raise InfBridgeFinalAnalysisError("observation_invalid")
    _validate_evaluation_attrition_v1(
        receipt_gate=receipt_gate,
        evidence=evidence,
        evaluation=evaluation,
    )
    return _final_report_v1(
        receipt_gate=receipt_gate,
        recovered=recovered,
        aggregate=aggregate,
        evidence=evidence,
        evaluation=evaluation,
    )
