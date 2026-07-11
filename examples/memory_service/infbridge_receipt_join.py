# SPDX-License-Identifier: Apache-2.0

"""Join all live Memory sidecar cases to answer-blind ledger receipts.

The join proves a local consistency chain from the actual Memory source bytes
through the frozen prompt/token receipts and InfBridge call plan.  A succeeded
slot continues through its canonical model receipt; an attrited slot instead
binds its sealed terminal reason and ledger leaf.  Both end at the sealed run
root.  The API accepts raw sidecars and a ledger path, then revalidates both;
public constructible validation reports are never a trust boundary.

This stage does not parse generation traces or response evidence, decode a
model answer, normalize a response, or score an outcome.  Its hashes are local
integrity commitments, not remote-model attestation, replay freshness,
confidentiality, or authenticity against a writer who can recompute the whole
ledger.  The full ledger replay remains mandatory before later scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Literal

from examples.memory_service import infbridge_model_adapter as adapter
from examples.memory_service import infbridge_run_ledger as ledger_module
from examples.memory_service import infbridge_sidecar_join as sidecars
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    CallPlanV2,
    RunEnvelopeV2,
    infbridge_run_envelope_v2_sha256,
)
from examples.memory_service.infbridge_run_ledger import (
    LedgerArtifactCommitmentV1,
    RunLedgerError,
    RunLedgerReceiptSlotV1,
    RunLedgerReceiptSnapshotV1,
    load_run_ledger_receipt_snapshot_v1,
)

__all__ = [
    "InfBridgeReceiptJoinError",
    "ModelCaseReceiptJoinV1",
    "ModelCaseReceiptSlotV1",
    "ModelRunReceiptJoinV1",
    "validate_live_model_run_receipt_join_v1",
]


class InfBridgeReceiptJoinError(RuntimeError):
    """Closed reason for refusing a sidecar-to-ledger receipt join."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("receipt join reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ModelCaseReceiptSlotV1:
    """One Memory observation joined without returning an answer preimage."""

    slot_index: int
    case_index: int
    arm: str
    source_kind: str
    observation_leaf_sha256: str
    rendered_context_sha256: str
    rendered_context_utf8_bytes: int
    consumer_query_sha256: str
    consumer_history_length: int
    submitted_prompt_sha256: str
    prompt_context_start: int
    prompt_context_end: int
    evaluator_input_token_ids_sha256: str
    adapter_input_token_ids_sha256: str
    input_token_count: int
    request_id: str
    plan_sha256: str
    ledger_state: Literal["SUCCEEDED", "ATTRITION"]
    attempt_count: int
    terminal_reason: str | None
    receipt_commitment: LedgerArtifactCommitmentV1 | None
    trace_commitment: LedgerArtifactCommitmentV1 | None
    response_evidence_commitment: LedgerArtifactCommitmentV1 | None
    decoded_response_commitment: LedgerArtifactCommitmentV1 | None
    ledger_leaf_sha256: str
    slot_join_sha256: str


@dataclass(frozen=True, slots=True)
class ModelCaseReceiptJoinV1:
    """Constructible validation report; later public gates must revalidate."""

    schema_version: int
    policy: str
    manifest_sha256: str
    envelope_sha256: str
    ledger_run_id: str
    ledger_run_root_sha256: str
    ledger_seal_kind: Literal["complete", "complete_with_attrition"]
    sidecar_policy: str
    ledger_projection_policy: str
    ledger_policy: str
    case_index: int
    sidecar_case_root_sha256: str
    capture_leaf_sha256: str
    observation_leaf_sha256s: tuple[str, ...]
    probe_leaf_sha256: str
    succeeded_count: int
    attrition_count: int
    slots: tuple[ModelCaseReceiptSlotV1, ...]
    case_receipt_root_sha256: str


@dataclass(frozen=True, slots=True)
class ModelRunReceiptJoinV1:
    """Preimage-free 64-case report produced after the global sidecar gate."""

    schema_version: int
    policy: str
    manifest_sha256: str
    envelope_sha256: str
    sidecar_policy: str
    sidecar_aggregate_root_sha256: str
    ledger_projection_policy: str
    ledger_policy: str
    ledger_run_id: str
    ledger_run_root_sha256: str
    ledger_seal_kind: Literal["complete", "complete_with_attrition"]
    case_count: int
    slot_count: int
    succeeded_count: int
    attrition_count: int
    cases: tuple[ModelCaseReceiptJoinV1, ...]
    run_receipt_root_sha256: str


_SCHEMA_VERSION = 1
_POLICY = "live-sidecar-sealed-ledger-answer-content-blind-consistency-only-v1"
_SLOT_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-slot-join-v1\0"
_CASE_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-case-join-v1\0"
_RUN_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-run-join-v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_projection_header_v1(ledger: object) -> RunLedgerReceiptSnapshotV1:
    expected_slot_count = helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
    if (
        type(ledger) is not RunLedgerReceiptSnapshotV1
        or type(ledger.schema_version) is not int
        or ledger.schema_version != 1
        or type(ledger.projection_policy) is not str
        or ledger.projection_policy != "sealed-answer-content-blind-receipts-v1"
        or type(ledger.ledger_policy) is not str
        or ledger.ledger_policy != "single-cursor-prewrite-v1"
        or type(ledger.call_count) is not int
        or ledger.call_count != expected_slot_count
        or type(ledger.seal_kind) is not str
        or ledger.seal_kind not in ("complete", "complete_with_attrition")
        or not _is_sha256(ledger.run_id)
        or not _is_sha256(ledger.run_root_sha256)
        or type(ledger.slots) is not tuple
        or len(ledger.slots) != expected_slot_count
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    return ledger


def _commitment_value(
    commitment: LedgerArtifactCommitmentV1 | None,
) -> dict[str, object] | None:
    if commitment is None:
        return None
    return {
        "byte_count": commitment.byte_count,
        "sha256": commitment.sha256,
    }


def _validate_terminal_artifacts(slot: RunLedgerReceiptSlotV1) -> None:
    commitments = (
        slot.receipt_commitment,
        slot.trace_commitment,
        slot.response_evidence_commitment,
        slot.decoded_response_commitment,
    )
    if slot.state == "SUCCEEDED":
        if (
            slot.attempt_count != 1
            or slot.terminal_reason is not None
            or slot.canonical_receipt is None
            or any(
                type(value) is not LedgerArtifactCommitmentV1 for value in commitments
            )
        ):
            raise InfBridgeReceiptJoinError("receipt_chain")
        return
    if slot.state == "ATTRITION":
        if (
            slot.attempt_count != 1
            or type(slot.terminal_reason) is not str
            or not slot.terminal_reason
            or slot.canonical_receipt is not None
            or any(value is not None for value in commitments)
        ):
            raise InfBridgeReceiptJoinError("receipt_chain")
        return
    raise InfBridgeReceiptJoinError("receipt_chain")


def _make_slot_join(
    *,
    registration: helpfulness.ModelCaseRegistration,
    arm_call: helpfulness.ModelArmCallRegistration,
    observation_leaf: sidecars.ModelSidecarLeafCommitmentV1,
    plan: CallPlanV2,
    ledger_slot: RunLedgerReceiptSlotV1,
    manifest_sha256: str,
    envelope_sha256: str,
) -> ModelCaseReceiptSlotV1:
    case_index = registration.identity.case.case_index
    prepared = arm_call.prepared_call
    consumer = prepared.consumer_input_receipt
    try:
        local_receipt = helpfulness.make_model_call_receipt(
            submitted_prompt=prepared.prompt,
            context_start=prepared.context_start,
            context_end=prepared.context_end,
            input_token_ids=prepared.input_token_ids,
        )
        source = helpfulness._fast_source_spec(
            registration.identity.case,
            registration.identity.references,
            arm_call.arm,
        )
        adapter_input_token_ids_sha256 = adapter._token_ids_sha256(
            prepared.input_token_ids
        )
    except (TypeError, ValueError, helpfulness.ModelProtocolError) as error:
        raise InfBridgeReceiptJoinError("receipt_chain") from error
    if (
        type(ledger_slot) is not RunLedgerReceiptSlotV1
        or not helpfulness._exact_typed_tree_equal(
            local_receipt,
            prepared.expected_receipt,
        )
        or not helpfulness._exact_typed_tree_equal(plan, ledger_slot.plan)
        or ledger_slot.plan_sha256
        != ledger_module._plan_sha256(ledger_module._plan_bytes(plan))
        or plan.slot_index != observation_leaf.logical_index
        or plan.case_index != case_index
        or plan.arm != arm_call.arm
        or observation_leaf.role != "observation"
        or observation_leaf.case_index != case_index
        or observation_leaf.arm != arm_call.arm
        or consumer.received_context_sha256 != arm_call.rendered_context_sha256
        or consumer.received_context_utf8_bytes != arm_call.rendered_context_utf8_bytes
        or consumer.received_query_sha256 != registration.query_sha256
        or consumer.received_history_length != 0
        or local_receipt.submitted_prompt_context_sha256
        != arm_call.rendered_context_sha256
        or local_receipt.submitted_prompt_context_end
        - local_receipt.submitted_prompt_context_start
        != arm_call.rendered_context_utf8_bytes
        or local_receipt.submitted_input_token_count != plan.input_token_count
        or local_receipt.submitted_input_token_count != len(prepared.input_token_ids)
        or adapter_input_token_ids_sha256 != plan.input_token_ids_sha256
        or prepared.context_start != local_receipt.submitted_prompt_context_start
        or prepared.context_end != local_receipt.submitted_prompt_context_end
        or hashlib.sha256(
            prepared.prompt[prepared.context_start : prepared.context_end]
        ).hexdigest()
        != arm_call.rendered_context_sha256
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    _validate_terminal_artifacts(ledger_slot)
    if ledger_slot.state == "SUCCEEDED":
        receipt = ledger_slot.canonical_receipt
        assert receipt is not None
        try:
            receipt_bytes = adapter.audited_model_call_receipt_v2_bytes(receipt)
        except adapter.InfBridgeModelAdapterError as error:
            raise InfBridgeReceiptJoinError("receipt_chain") from error
        if (
            receipt.manifest_sha256 != manifest_sha256
            or receipt.run_envelope_sha256 != envelope_sha256
            or receipt.slot_index != plan.slot_index
            or receipt.case_index != plan.case_index
            or receipt.arm != plan.arm
            or receipt.request_id != plan.request_id
            or ledger_slot.receipt_commitment is None
            or ledger_slot.receipt_commitment.byte_count != len(receipt_bytes)
            or ledger_slot.receipt_commitment.sha256
            != hashlib.sha256(receipt_bytes).hexdigest()
            or ledger_slot.trace_commitment is None
            or receipt.generation_trace_sha256 != ledger_slot.trace_commitment.sha256
            or ledger_slot.response_evidence_commitment is None
            or receipt.generation_response_evidence_sha256
            != ledger_slot.response_evidence_commitment.sha256
            or receipt.generation_response_evidence_byte_count
            != ledger_slot.response_evidence_commitment.byte_count
            or ledger_slot.decoded_response_commitment is None
            or receipt.decoded_response_utf8_sha256
            != ledger_slot.decoded_response_commitment.sha256
            or receipt.decoded_response_utf8_bytes
            != ledger_slot.decoded_response_commitment.byte_count
        ):
            raise InfBridgeReceiptJoinError("receipt_chain")
    value = {
        "arm": arm_call.arm,
        "case_index": case_index,
        "consumer_history_length": consumer.received_history_length,
        "consumer_query_sha256": consumer.received_query_sha256,
        "decoded_response_commitment": _commitment_value(
            ledger_slot.decoded_response_commitment
        ),
        "adapter_input_token_ids_sha256": plan.input_token_ids_sha256,
        "attempt_count": ledger_slot.attempt_count,
        "input_token_count": plan.input_token_count,
        "ledger_leaf_sha256": ledger_slot.ledger_leaf_sha256,
        "ledger_state": ledger_slot.state,
        "evaluator_input_token_ids_sha256": (
            local_receipt.submitted_input_token_ids_sha256
        ),
        "observation_leaf_sha256": observation_leaf.leaf_sha256,
        "plan_sha256": ledger_slot.plan_sha256,
        "policy": _POLICY,
        "prompt_context_end": local_receipt.submitted_prompt_context_end,
        "prompt_context_start": local_receipt.submitted_prompt_context_start,
        "receipt_commitment": _commitment_value(ledger_slot.receipt_commitment),
        "rendered_context_sha256": arm_call.rendered_context_sha256,
        "rendered_context_utf8_bytes": arm_call.rendered_context_utf8_bytes,
        "request_id": plan.request_id,
        "response_evidence_commitment": _commitment_value(
            ledger_slot.response_evidence_commitment
        ),
        "schema_version": _SCHEMA_VERSION,
        "slot_index": plan.slot_index,
        "source_kind": source.source_kind,
        "submitted_prompt_sha256": local_receipt.submitted_prompt_sha256,
        "terminal_reason": ledger_slot.terminal_reason,
        "trace_commitment": _commitment_value(ledger_slot.trace_commitment),
    }
    slot_join_sha256 = hashlib.sha256(
        _SLOT_JOIN_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()
    return ModelCaseReceiptSlotV1(
        slot_index=plan.slot_index,
        case_index=case_index,
        arm=arm_call.arm,
        source_kind=source.source_kind,
        observation_leaf_sha256=observation_leaf.leaf_sha256,
        rendered_context_sha256=arm_call.rendered_context_sha256,
        rendered_context_utf8_bytes=arm_call.rendered_context_utf8_bytes,
        consumer_query_sha256=consumer.received_query_sha256,
        consumer_history_length=consumer.received_history_length,
        submitted_prompt_sha256=local_receipt.submitted_prompt_sha256,
        prompt_context_start=local_receipt.submitted_prompt_context_start,
        prompt_context_end=local_receipt.submitted_prompt_context_end,
        evaluator_input_token_ids_sha256=(
            local_receipt.submitted_input_token_ids_sha256
        ),
        adapter_input_token_ids_sha256=plan.input_token_ids_sha256,
        input_token_count=plan.input_token_count,
        request_id=plan.request_id,
        plan_sha256=ledger_slot.plan_sha256,
        ledger_state=ledger_slot.state,
        attempt_count=ledger_slot.attempt_count,
        terminal_reason=ledger_slot.terminal_reason,
        receipt_commitment=ledger_slot.receipt_commitment,
        trace_commitment=ledger_slot.trace_commitment,
        response_evidence_commitment=ledger_slot.response_evidence_commitment,
        decoded_response_commitment=ledger_slot.decoded_response_commitment,
        ledger_leaf_sha256=ledger_slot.ledger_leaf_sha256,
        slot_join_sha256=slot_join_sha256,
    )


def _join_validated_model_case_receipts_v1(
    manifest: helpfulness.ModelRunManifest,
    envelope: RunEnvelopeV2,
    validated: sidecars.ValidatedModelCaseSidecarV1,
    ledger: RunLedgerReceiptSnapshotV1,
) -> ModelCaseReceiptJoinV1:
    if type(validated) is not sidecars.ValidatedModelCaseSidecarV1:
        raise InfBridgeReceiptJoinError("receipt_chain")
    ledger = _validate_projection_header_v1(ledger)
    commitment = validated.commitment
    case_index = commitment.case_index
    if (
        commitment.manifest_sha256 != ledger.manifest_sha256
        or commitment.envelope_sha256 != ledger.run_envelope_sha256
        or commitment.envelope_sha256 != infbridge_run_envelope_v2_sha256(envelope)
        or type(validated.leaves) is not tuple
        or len(validated.leaves) != len(helpfulness.MODEL_ARMS) + 2
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    registration = manifest.cases[case_index]
    slot_joins: list[ModelCaseReceiptSlotV1] = []
    for arm_offset, arm_call in enumerate(registration.arm_calls):
        slot_index = case_index * len(helpfulness.MODEL_ARMS) + arm_offset
        slot_joins.append(
            _make_slot_join(
                registration=registration,
                arm_call=arm_call,
                observation_leaf=validated.leaves[arm_offset + 1],
                plan=envelope.call_plans[slot_index],
                ledger_slot=ledger.slots[slot_index],
                manifest_sha256=commitment.manifest_sha256,
                envelope_sha256=commitment.envelope_sha256,
            )
        )
    slots = tuple(slot_joins)
    if len(slots) != len(helpfulness.MODEL_ARMS) or tuple(
        slot.arm for slot in slots
    ) != tuple(call.arm for call in registration.arm_calls):
        raise InfBridgeReceiptJoinError("receipt_chain")
    succeeded_count = sum(slot.ledger_state == "SUCCEEDED" for slot in slots)
    attrition_count = sum(slot.ledger_state == "ATTRITION" for slot in slots)
    if succeeded_count + attrition_count != len(slots):
        raise InfBridgeReceiptJoinError("receipt_chain")
    root_value = {
        "attrition_count": attrition_count,
        "case_index": case_index,
        "capture_leaf_sha256": commitment.capture_leaf_sha256,
        "envelope_sha256": commitment.envelope_sha256,
        "ledger_policy": ledger.ledger_policy,
        "ledger_projection_policy": ledger.projection_policy,
        "ledger_run_id": ledger.run_id,
        "ledger_run_root_sha256": ledger.run_root_sha256,
        "ledger_seal_kind": ledger.seal_kind,
        "manifest_sha256": commitment.manifest_sha256,
        "observation_leaf_sha256s": list(commitment.observation_leaf_sha256s),
        "policy": _POLICY,
        "probe_leaf_sha256": commitment.probe_leaf_sha256,
        "schema_version": _SCHEMA_VERSION,
        "sidecar_case_root_sha256": commitment.case_root_sha256,
        "sidecar_policy": commitment.policy,
        "slot_count": len(slots),
        "slot_join_sha256s": [slot.slot_join_sha256 for slot in slots],
        "succeeded_count": succeeded_count,
    }
    case_receipt_root_sha256 = hashlib.sha256(
        _CASE_JOIN_DOMAIN + _canonical_json_bytes(root_value)
    ).hexdigest()
    return ModelCaseReceiptJoinV1(
        schema_version=_SCHEMA_VERSION,
        policy=_POLICY,
        manifest_sha256=commitment.manifest_sha256,
        envelope_sha256=commitment.envelope_sha256,
        ledger_run_id=ledger.run_id,
        ledger_run_root_sha256=ledger.run_root_sha256,
        ledger_seal_kind=ledger.seal_kind,
        sidecar_policy=commitment.policy,
        ledger_projection_policy=ledger.projection_policy,
        ledger_policy=ledger.ledger_policy,
        case_index=case_index,
        sidecar_case_root_sha256=commitment.case_root_sha256,
        capture_leaf_sha256=commitment.capture_leaf_sha256,
        observation_leaf_sha256s=commitment.observation_leaf_sha256s,
        probe_leaf_sha256=commitment.probe_leaf_sha256,
        succeeded_count=succeeded_count,
        attrition_count=attrition_count,
        slots=slots,
        case_receipt_root_sha256=case_receipt_root_sha256,
    )


def _join_validated_model_run_receipts_v1(
    manifest: helpfulness.ModelRunManifest,
    envelope: RunEnvelopeV2,
    aggregate: sidecars.ValidatedModelSidecarAggregateV1,
    ledger: RunLedgerReceiptSnapshotV1,
) -> ModelRunReceiptJoinV1:
    if (
        type(aggregate) is not sidecars.ValidatedModelSidecarAggregateV1
        or type(ledger) is not RunLedgerReceiptSnapshotV1
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    ledger = _validate_projection_header_v1(ledger)
    sidecar_commitment = aggregate.commitment
    if (
        type(sidecar_commitment) is not sidecars.ModelSidecarAggregateCommitmentV1
        or sidecar_commitment.manifest_sha256 != ledger.manifest_sha256
        or sidecar_commitment.envelope_sha256 != ledger.run_envelope_sha256
        or sidecar_commitment.envelope_sha256
        != infbridge_run_envelope_v2_sha256(envelope)
        or type(aggregate.cases) is not tuple
        or len(aggregate.cases) != helpfulness.MODEL_CASE_COUNT
        or tuple(case.commitment.case_index for case in aggregate.cases)
        != tuple(range(helpfulness.MODEL_CASE_COUNT))
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    cases = tuple(
        _join_validated_model_case_receipts_v1(
            manifest,
            envelope,
            case,
            ledger,
        )
        for case in aggregate.cases
    )
    case_count = len(cases)
    slot_count = sum(len(case.slots) for case in cases)
    succeeded_count = sum(case.succeeded_count for case in cases)
    attrition_count = sum(case.attrition_count for case in cases)
    expected_slot_count = helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
    if (
        case_count != helpfulness.MODEL_CASE_COUNT
        or slot_count != expected_slot_count
        or succeeded_count + attrition_count != slot_count
    ):
        raise InfBridgeReceiptJoinError("receipt_chain")
    root_value = {
        "attrition_count": attrition_count,
        "case_count": case_count,
        "case_receipt_root_sha256s": [case.case_receipt_root_sha256 for case in cases],
        "envelope_sha256": sidecar_commitment.envelope_sha256,
        "ledger_policy": ledger.ledger_policy,
        "ledger_projection_policy": ledger.projection_policy,
        "ledger_run_id": ledger.run_id,
        "ledger_run_root_sha256": ledger.run_root_sha256,
        "ledger_seal_kind": ledger.seal_kind,
        "manifest_sha256": sidecar_commitment.manifest_sha256,
        "policy": _POLICY,
        "schema_version": _SCHEMA_VERSION,
        "sidecar_aggregate_root_sha256": (sidecar_commitment.aggregate_root_sha256),
        "sidecar_policy": sidecar_commitment.policy,
        "slot_count": slot_count,
        "succeeded_count": succeeded_count,
    }
    run_receipt_root_sha256 = hashlib.sha256(
        _RUN_JOIN_DOMAIN + _canonical_json_bytes(root_value)
    ).hexdigest()
    return ModelRunReceiptJoinV1(
        schema_version=_SCHEMA_VERSION,
        policy=_POLICY,
        manifest_sha256=sidecar_commitment.manifest_sha256,
        envelope_sha256=sidecar_commitment.envelope_sha256,
        sidecar_policy=sidecar_commitment.policy,
        sidecar_aggregate_root_sha256=(sidecar_commitment.aggregate_root_sha256),
        ledger_projection_policy=ledger.projection_policy,
        ledger_policy=ledger.ledger_policy,
        ledger_run_id=ledger.run_id,
        ledger_run_root_sha256=ledger.run_root_sha256,
        ledger_seal_kind=ledger.seal_kind,
        case_count=case_count,
        slot_count=slot_count,
        succeeded_count=succeeded_count,
        attrition_count=attrition_count,
        cases=cases,
        run_receipt_root_sha256=run_receipt_root_sha256,
    )


def validate_live_model_run_receipt_join_v1(
    ledger_database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    model_sidecars: tuple[sidecars.ModelCaseSidecarV1, ...],
) -> ModelRunReceiptJoinV1:
    """Join one sealed ledger only after all raw sidecars pass globally.

    All 64 response-free sidecar cases are validated before the ledger path is
    touched.  They are validated again after the answer-hash-bearing receipt
    projection is loaded.  Any drift fails the whole run before a report is
    returned.  This narrows ordinary file races but is not an atomic snapshot
    across the Memory and ledger databases.
    """

    try:
        first = sidecars.validate_live_model_sidecar_aggregate_v1(
            manifest,
            tokenizer,
            envelope,
            model_sidecars,
        )
    except sidecars.ModelSidecarJoinError as error:
        raise InfBridgeReceiptJoinError("sidecar_invalid") from error
    try:
        ledger_snapshot = load_run_ledger_receipt_snapshot_v1(
            ledger_database_path,
            manifest,
            tokenizer,
            envelope,
        )
    except RunLedgerError as error:
        raise InfBridgeReceiptJoinError("ledger_invalid") from error
    try:
        second = sidecars.validate_live_model_sidecar_aggregate_v1(
            manifest,
            tokenizer,
            envelope,
            model_sidecars,
        )
    except sidecars.ModelSidecarJoinError as error:
        ledger_snapshot = None
        raise InfBridgeReceiptJoinError("sidecar_invalid") from error
    if not helpfulness._exact_typed_tree_equal(first, second):
        ledger_snapshot = None
        raise InfBridgeReceiptJoinError("sidecar_invalid")
    return _join_validated_model_run_receipts_v1(
        manifest,
        envelope,
        second,
        ledger_snapshot,
    )
