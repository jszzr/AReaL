# SPDX-License-Identifier: Apache-2.0

"""Live, answer-free validation for one model case's Memory sidecars.

This first join stage validates one isolated capture, six source observations,
and one scoped leakage probe against the frozen manifest and InfBridge run
envelope.  It deliberately does not load the model-call ledger, decode a model
answer, normalize a response, or score an outcome.

The resulting root is an execution-local internal-consistency commitment.  It
includes absolute paths, inode-derived receipts, PIDs, and process UUIDs; it is
neither a cross-machine reproducibility hash nor an authenticity proof.  The
current envelope also has no fresh experiment nonce, so this stage makes no
same-run or replay-freshness claim.  Callers must construct exchanges directly
from the exact ``run_isolated_child_raw`` result in the live parent process.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from typing import Literal

from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    InfBridgeModelAdapterError,
    RunEnvelopeV2,
    infbridge_run_envelope_v2_sha256,
    validate_infbridge_run_envelope_v2,
)

__all__ = [
    "CanonicalChildExchangeV1",
    "CanonicalWireCommitmentV1",
    "ModelCaptureSidecarV1",
    "ModelCaseSidecarCommitmentV1",
    "ModelCaseSidecarV1",
    "ModelObservationSidecarV1",
    "ModelProbeSidecarV1",
    "ModelSidecarAggregateCommitmentV1",
    "ModelSidecarJoinError",
    "ModelSidecarLeafCommitmentV1",
    "ValidatedModelCaseSidecarV1",
    "ValidatedModelSidecarAggregateV1",
    "validate_live_model_case_sidecar_v1",
    "validate_live_model_sidecar_aggregate_v1",
]

SidecarRole = Literal["capture", "observation", "leakage_probe"]


class ModelSidecarJoinError(RuntimeError):
    """Closed reason for refusing one live sidecar case."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("sidecar join reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CanonicalChildExchangeV1:
    """Exact child stdin/stdout plus the PID observed from ``Popen``."""

    request_wire_utf8: bytes
    response_wire_utf8: bytes
    launcher_pid: int


@dataclass(frozen=True, slots=True)
class ModelCaptureSidecarV1:
    case_index: int
    exchange: CanonicalChildExchangeV1


@dataclass(frozen=True, slots=True)
class ModelObservationSidecarV1:
    slot_index: int
    case_index: int
    arm: str
    exchange: CanonicalChildExchangeV1


@dataclass(frozen=True, slots=True)
class ModelProbeSidecarV1:
    case_index: int
    logical_execution_index: int
    exchange: CanonicalChildExchangeV1


@dataclass(frozen=True, slots=True)
class ModelCaseSidecarV1:
    capture: ModelCaptureSidecarV1
    observations: tuple[ModelObservationSidecarV1, ...]
    probe: ModelProbeSidecarV1


@dataclass(frozen=True, slots=True)
class CanonicalWireCommitmentV1:
    wire_type: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSidecarLeafCommitmentV1:
    role: SidecarRole
    logical_index: int
    case_index: int
    arm: str | None
    opaque_execution_token_hex: str | None
    launcher_pid: int
    request: CanonicalWireCommitmentV1
    response: CanonicalWireCommitmentV1
    leaf_sha256: str


@dataclass(frozen=True, slots=True)
class ModelCaseSidecarCommitmentV1:
    schema_version: int
    policy: str
    manifest_sha256: str
    envelope_sha256: str
    case_index: int
    capture_leaf_sha256: str
    observation_leaf_sha256s: tuple[str, ...]
    probe_leaf_sha256: str
    case_root_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedModelCaseSidecarV1:
    """Answer-free report; later joins must revalidate its raw sidecar."""

    commitment: ModelCaseSidecarCommitmentV1
    leaves: tuple[ModelSidecarLeafCommitmentV1, ...]
    capture: ModelCaptureSidecarV1
    observations: tuple[ModelObservationSidecarV1, ...]
    probe: ModelProbeSidecarV1


@dataclass(frozen=True, slots=True)
class ModelSidecarAggregateCommitmentV1:
    """Ordered 64-case commitment to all response-free model sidecars."""

    schema_version: int
    policy: str
    manifest_sha256: str
    envelope_sha256: str
    case_count: int
    capture_count: int
    observation_count: int
    probe_count: int
    leaf_count: int
    case_root_sha256s: tuple[str, ...]
    aggregate_root_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedModelSidecarAggregateV1:
    """Constructible report; later public gates must revalidate raw cases."""

    commitment: ModelSidecarAggregateCommitmentV1
    cases: tuple[ValidatedModelCaseSidecarV1, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedCaseDetailsV1:
    result: ValidatedModelCaseSidecarV1
    capture_request: helpfulness.ModelCaptureChildRequest
    capture_response: helpfulness.ModelCaptureChildResponse
    observation_rows: tuple[
        tuple[
            helpfulness.ModelObservationChildRequest,
            helpfulness.ModelObservationChildResponse,
            ModelSidecarLeafCommitmentV1,
        ],
        ...,
    ]
    probe_row: tuple[
        helpfulness.ModelLeakageProbeChildRequest,
        helpfulness.ModelLeakageProbeChildResponse,
        ModelSidecarLeafCommitmentV1,
    ]


_SCHEMA_VERSION = 1
_POLICY = "live-honest-child-consistency-only-v1"
_MAX_CHILD_WIRE_BYTES = 16 * 1024 * 1024
_MAX_AGGREGATE_WIRE_BYTES = 256 * 1024 * 1024
_OBSERVATION_COUNT = len(helpfulness.MODEL_ARMS)
_PROBE_LOGICAL_BASE = helpfulness.MODEL_CASE_COUNT * _OBSERVATION_COUNT

_CAPTURE_LEAF_DOMAIN = b"areal-memory-model-sidecar-capture-leaf-v1\0"
_OBSERVATION_LEAF_DOMAIN = b"areal-memory-model-sidecar-observation-leaf-v1\0"
_PROBE_LEAF_DOMAIN = b"areal-memory-model-sidecar-leakage-probe-leaf-v1\0"
_CASE_ROOT_DOMAIN = b"areal-memory-model-sidecar-case-root-v1\0"
_AGGREGATE_ROOT_DOMAIN = b"areal-memory-model-sidecar-aggregate-root-v1\0"

_LEAF_DOMAINS: dict[SidecarRole, bytes] = {
    "capture": _CAPTURE_LEAF_DOMAIN,
    "observation": _OBSERVATION_LEAF_DOMAIN,
    "leakage_probe": _PROBE_LEAF_DOMAIN,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _wire_commitment(
    wire_type: str,
    value: bytes,
) -> CanonicalWireCommitmentV1:
    return CanonicalWireCommitmentV1(
        wire_type=wire_type,
        byte_count=len(value),
        sha256=hashlib.sha256(value).hexdigest(),
    )


def _wire_commitment_value(
    value: CanonicalWireCommitmentV1,
) -> dict[str, object]:
    return {
        "byte_count": value.byte_count,
        "sha256": value.sha256,
        "wire_type": value.wire_type,
    }


def _decode_exchange(
    exchange: CanonicalChildExchangeV1,
    *,
    request_type: type[object],
    response_type: type[object],
    request_wire_type: str,
    response_wire_type: str,
) -> tuple[
    object,
    object,
    CanonicalWireCommitmentV1,
    CanonicalWireCommitmentV1,
]:
    if (
        type(exchange) is not CanonicalChildExchangeV1
        or type(exchange.request_wire_utf8) is not bytes
        or type(exchange.response_wire_utf8) is not bytes
        or not exchange.request_wire_utf8
        or not exchange.response_wire_utf8
        or len(exchange.request_wire_utf8) > _MAX_CHILD_WIRE_BYTES
        or len(exchange.response_wire_utf8) > _MAX_CHILD_WIRE_BYTES
        or type(exchange.launcher_pid) is not int
        or exchange.launcher_pid <= 0
    ):
        raise ModelSidecarJoinError("closed_schema")
    try:
        request_text = exchange.request_wire_utf8.decode("utf-8", errors="strict")
        response_text = exchange.response_wire_utf8.decode("utf-8", errors="strict")
        request = helpfulness.wire_loads(request_text)
        response = helpfulness.wire_loads(response_text)
        if (
            type(request) is not request_type
            or type(response) is not response_type
            or helpfulness.wire_dumps(request).encode("utf-8", errors="strict")
            != exchange.request_wire_utf8
            or helpfulness.wire_dumps(response).encode("utf-8", errors="strict")
            != exchange.response_wire_utf8
        ):
            raise ModelSidecarJoinError("closed_schema")
    except ModelSidecarJoinError:
        raise
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        helpfulness.WireProtocolError,
    ) as error:
        raise ModelSidecarJoinError("closed_schema") from error
    return (
        request,
        response,
        _wire_commitment(request_wire_type, exchange.request_wire_utf8),
        _wire_commitment(response_wire_type, exchange.response_wire_utf8),
    )


def _leaf_commitment(
    *,
    role: SidecarRole,
    logical_index: int,
    case_index: int,
    arm: str | None,
    opaque_execution_token: int | None,
    launcher_pid: int,
    request: CanonicalWireCommitmentV1,
    response: CanonicalWireCommitmentV1,
) -> ModelSidecarLeafCommitmentV1:
    opaque_hex = (
        None if opaque_execution_token is None else f"{opaque_execution_token:032x}"
    )
    value = {
        "arm": arm,
        "case_index": case_index,
        "launcher_pid": launcher_pid,
        "logical_index": logical_index,
        "opaque_execution_token_hex": opaque_hex,
        "request": _wire_commitment_value(request),
        "response": _wire_commitment_value(response),
        "role": role,
        "schema_version": _SCHEMA_VERSION,
    }
    leaf_sha256 = hashlib.sha256(
        _LEAF_DOMAINS[role] + _canonical_json_bytes(value)
    ).hexdigest()
    return ModelSidecarLeafCommitmentV1(
        role=role,
        logical_index=logical_index,
        case_index=case_index,
        arm=arm,
        opaque_execution_token_hex=opaque_hex,
        launcher_pid=launcher_pid,
        request=request,
        response=response,
        leaf_sha256=leaf_sha256,
    )


def _validate_live_process(response: object, *, launcher_pid: int) -> None:
    try:
        helpfulness._validate_child_process_response(
            response,  # type: ignore[arg-type]
            expected_pid=launcher_pid,
        )
    except helpfulness.ChildExecutionValidationError as error:
        raise ModelSidecarJoinError("process_isolation") from error


def _validate_capture(
    registration: helpfulness.ModelCaseRegistration,
    sidecar: ModelCaptureSidecarV1,
) -> tuple[
    helpfulness.ModelCaptureChildRequest,
    helpfulness.ModelCaptureChildResponse,
    ModelSidecarLeafCommitmentV1,
]:
    case_index = registration.identity.case.case_index
    if (
        type(sidecar) is not ModelCaptureSidecarV1
        or type(sidecar.case_index) is not int
        or sidecar.case_index != case_index
    ):
        raise ModelSidecarJoinError("capture_mismatch")
    (
        request_value,
        response_value,
        request_commitment,
        response_commitment,
    ) = _decode_exchange(
        sidecar.exchange,
        request_type=helpfulness.ModelCaptureChildRequest,
        response_type=helpfulness.ModelCaptureChildResponse,
        request_wire_type="model_capture_child_request",
        response_wire_type="model_capture_child_response",
    )
    request = request_value
    response = response_value
    assert type(request) is helpfulness.ModelCaptureChildRequest
    assert type(response) is helpfulness.ModelCaptureChildResponse
    try:
        case, references = helpfulness._model_capture_assignment(request)
        helpfulness._model_capture_directory_identity(
            os.path.dirname(request.database_path)
        )
        observed_receipt = helpfulness._model_capture_database_receipt(
            request.database_path,
            expected_identity=(
                response.database_receipt.device,
                response.database_receipt.inode,
            ),
            durable=False,
        )
    except (
        helpfulness.ChildExecutionValidationError,
        helpfulness.WireProtocolError,
    ) as error:
        raise ModelSidecarJoinError("capture_mismatch") from error
    identity = registration.identity
    if (
        request.case_index != case_index
        or request.model_attempt != registration.model_attempt
        or request.case_manifest_sha256 != identity.case_manifest_sha256
        or response.case_index != case_index
        or response.model_attempt != registration.model_attempt
        or response.case_manifest_sha256 != identity.case_manifest_sha256
        or not helpfulness._exact_typed_tree_equal(case, identity.case)
        or not helpfulness._exact_typed_tree_equal(references, identity.references)
        or not helpfulness._exact_typed_tree_equal(
            response.references,
            identity.references,
        )
        or response.database_receipt != observed_receipt
    ):
        raise ModelSidecarJoinError("capture_mismatch")
    _validate_live_process(response, launcher_pid=sidecar.exchange.launcher_pid)
    leaf = _leaf_commitment(
        role="capture",
        logical_index=case_index,
        case_index=case_index,
        arm=None,
        opaque_execution_token=None,
        launcher_pid=sidecar.exchange.launcher_pid,
        request=request_commitment,
        response=response_commitment,
    )
    return request, response, leaf


def _validate_observation(
    *,
    registration: helpfulness.ModelCaseRegistration,
    arm_call: helpfulness.ModelArmCallRegistration,
    arm_offset: int,
    capture_request: helpfulness.ModelCaptureChildRequest,
    capture_response: helpfulness.ModelCaptureChildResponse,
    sidecar: ModelObservationSidecarV1,
) -> tuple[
    helpfulness.ModelObservationChildRequest,
    helpfulness.ModelObservationChildResponse,
    ModelSidecarLeafCommitmentV1,
]:
    identity = registration.identity
    case = identity.case
    case_index = case.case_index
    slot_index = case_index * _OBSERVATION_COUNT + arm_offset
    if (
        type(sidecar) is not ModelObservationSidecarV1
        or type(sidecar.slot_index) is not int
        or type(sidecar.case_index) is not int
        or type(sidecar.arm) is not str
        or sidecar.slot_index != slot_index
        or sidecar.case_index != case_index
        or sidecar.arm != arm_call.arm
    ):
        raise ModelSidecarJoinError("inventory_mismatch")
    (
        request_value,
        response_value,
        request_commitment,
        response_commitment,
    ) = _decode_exchange(
        sidecar.exchange,
        request_type=helpfulness.ModelObservationChildRequest,
        response_type=helpfulness.ModelObservationChildResponse,
        request_wire_type="model_observation_child_request",
        response_wire_type="model_observation_child_response",
    )
    request = request_value
    response = response_value
    assert type(request) is helpfulness.ModelObservationChildRequest
    assert type(response) is helpfulness.ModelObservationChildResponse
    expected_source = helpfulness._fast_source_spec(
        case,
        identity.references,
        arm_call.arm,
    )
    if (
        request.database_path != capture_request.database_path
        or request.database_receipt != capture_response.database_receipt
        or request.scope != identity.references.capture.local_scope
        or not helpfulness._exact_typed_tree_equal(request.source, expected_source)
        or request.renderer_version != "memory-codebook/v1"
    ):
        raise ModelSidecarJoinError("assignment_mismatch")
    try:
        helpfulness._validate_model_observation_child_assignment(response, request)
    except (
        helpfulness.ChildExecutionValidationError,
        helpfulness.WireProtocolError,
    ) as error:
        raise ModelSidecarJoinError("assignment_mismatch") from error
    _validate_live_process(response, launcher_pid=sidecar.exchange.launcher_pid)
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=request.execution_index,
        case=case,
        references=identity.references,
        arm=arm_call.arm,
    )
    observation = response.observation
    try:
        helpfulness._validate_parent_source_contract(observation, schedule)
        resolved_entries = helpfulness._model_observation_resolved_entries(observation)
        rendered = helpfulness.render_context(resolved_entries)
    except (
        helpfulness.ChildExecutionValidationError,
        helpfulness.ObservationValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise ModelSidecarJoinError("source_mismatch") from error
    prepared = arm_call.prepared_call
    prompt_context = prepared.prompt[prepared.context_start : prepared.context_end]
    consumer = prepared.consumer_input_receipt
    expected_receipt = prepared.expected_receipt
    if (
        rendered.bytes != prompt_context
        or rendered.entry_receipts != observation.entries
        or hashlib.sha256(rendered.bytes).hexdigest()
        != observation.rendered_context_sha256
        or len(rendered.bytes) != observation.rendered_context_utf8_bytes
        or arm_call.rendered_context_sha256 != observation.rendered_context_sha256
        or arm_call.rendered_context_utf8_bytes
        != observation.rendered_context_utf8_bytes
        or consumer.received_context_sha256 != observation.rendered_context_sha256
        or consumer.received_context_utf8_bytes
        != observation.rendered_context_utf8_bytes
        or expected_receipt.submitted_prompt_context_sha256
        != observation.rendered_context_sha256
        or expected_receipt.submitted_prompt_context_end
        - expected_receipt.submitted_prompt_context_start
        != observation.rendered_context_utf8_bytes
    ):
        raise ModelSidecarJoinError("render_mismatch")
    leaf = _leaf_commitment(
        role="observation",
        logical_index=slot_index,
        case_index=case_index,
        arm=arm_call.arm,
        opaque_execution_token=request.execution_index,
        launcher_pid=sidecar.exchange.launcher_pid,
        request=request_commitment,
        response=response_commitment,
    )
    return request, response, leaf


def _validate_probe(
    *,
    registration: helpfulness.ModelCaseRegistration,
    capture_request: helpfulness.ModelCaptureChildRequest,
    capture_response: helpfulness.ModelCaptureChildResponse,
    sidecar: ModelProbeSidecarV1,
) -> tuple[
    helpfulness.ModelLeakageProbeChildRequest,
    helpfulness.ModelLeakageProbeChildResponse,
    ModelSidecarLeafCommitmentV1,
]:
    identity = registration.identity
    case_index = identity.case.case_index
    logical_index = _PROBE_LOGICAL_BASE + case_index
    if (
        type(sidecar) is not ModelProbeSidecarV1
        or type(sidecar.case_index) is not int
        or type(sidecar.logical_execution_index) is not int
        or sidecar.case_index != case_index
        or sidecar.logical_execution_index != logical_index
    ):
        raise ModelSidecarJoinError("inventory_mismatch")
    (
        request_value,
        response_value,
        request_commitment,
        response_commitment,
    ) = _decode_exchange(
        sidecar.exchange,
        request_type=helpfulness.ModelLeakageProbeChildRequest,
        response_type=helpfulness.ModelLeakageProbeChildResponse,
        request_wire_type="model_leakage_probe_child_request",
        response_wire_type="model_leakage_probe_child_response",
    )
    request = request_value
    response = response_value
    assert type(request) is helpfulness.ModelLeakageProbeChildRequest
    assert type(response) is helpfulness.ModelLeakageProbeChildResponse
    if (
        request.database_path != capture_request.database_path
        or request.database_receipt != capture_response.database_receipt
        or request.scope != identity.references.capture.local_scope
        or request.release_id
        != identity.references.releases.foreign_sentinel_release_id
    ):
        raise ModelSidecarJoinError("probe_mismatch")
    try:
        helpfulness._validate_model_leakage_probe_child_assignment(
            response,
            request,
        )
    except (
        helpfulness.ChildExecutionValidationError,
        helpfulness.WireProtocolError,
    ) as error:
        raise ModelSidecarJoinError("probe_mismatch") from error
    _validate_live_process(response, launcher_pid=sidecar.exchange.launcher_pid)
    leaf = _leaf_commitment(
        role="leakage_probe",
        logical_index=logical_index,
        case_index=case_index,
        arm=None,
        opaque_execution_token=request.execution_index,
        launcher_pid=sidecar.exchange.launcher_pid,
        request=request_commitment,
        response=response_commitment,
    )
    return request, response, leaf


def _component_instance_ids(response: object) -> tuple[str, ...]:
    state = response.state_receipt  # type: ignore[union-attr]
    return tuple(
        getattr(state, field.name)
        for field in fields(state)
        if field.name.endswith("_instance_id") and field.name != "process_instance_id"
    )


def _validate_case_identity_separation(
    *,
    registration: helpfulness.ModelCaseRegistration,
    capture_response: helpfulness.ModelCaptureChildResponse,
    observation_rows: tuple[
        tuple[
            helpfulness.ModelObservationChildRequest,
            helpfulness.ModelObservationChildResponse,
            ModelSidecarLeafCommitmentV1,
        ],
        ...,
    ],
    probe_row: tuple[
        helpfulness.ModelLeakageProbeChildRequest,
        helpfulness.ModelLeakageProbeChildResponse,
        ModelSidecarLeafCommitmentV1,
    ],
) -> None:
    observation_requests = tuple(row[0] for row in observation_rows)
    observation_responses = tuple(row[1] for row in observation_rows)
    probe_request, probe_response, _probe_leaf = probe_row
    execution_tokens = tuple(
        request.execution_index for request in observation_requests
    ) + (probe_request.execution_index,)
    process_ids = (
        (capture_response.process_instance_id,)
        + tuple(response.process_instance_id for response in observation_responses)
        + (probe_response.process_instance_id,)
    )
    session_ids = tuple(
        request.future_session_id for request in observation_requests
    ) + (probe_response.probe.future_session_id,)
    run_ids = tuple(request.future_run_id for request in observation_requests) + (
        probe_response.probe.future_run_id,
    )
    component_ids = tuple(
        identity
        for response in (*observation_responses, probe_response)
        for identity in _component_instance_ids(response)
    )
    capture_sessions = registration.identity.references.capture.capture_session_ids
    if (
        len(set(execution_tokens)) != len(execution_tokens)
        or len(set(process_ids)) != len(process_ids)
        or len(set(session_ids)) != len(session_ids)
        or len(set(run_ids)) != len(run_ids)
        or len(set(component_ids)) != len(component_ids)
        or set(capture_sessions).intersection(session_ids)
    ):
        raise ModelSidecarJoinError("identity_reuse")


def _case_commitment(
    *,
    manifest_sha256: str,
    envelope_sha256: str,
    case_index: int,
    capture_leaf: ModelSidecarLeafCommitmentV1,
    observation_leaves: tuple[ModelSidecarLeafCommitmentV1, ...],
    probe_leaf: ModelSidecarLeafCommitmentV1,
) -> ModelCaseSidecarCommitmentV1:
    observation_hashes = tuple(leaf.leaf_sha256 for leaf in observation_leaves)
    value = {
        "capture_leaf_sha256": capture_leaf.leaf_sha256,
        "case_index": case_index,
        "envelope_sha256": envelope_sha256,
        "leaf_count": 1 + len(observation_hashes) + 1,
        "manifest_sha256": manifest_sha256,
        "observation_leaf_sha256s": list(observation_hashes),
        "policy": _POLICY,
        "probe_leaf_sha256": probe_leaf.leaf_sha256,
        "schema_version": _SCHEMA_VERSION,
    }
    root = hashlib.sha256(_CASE_ROOT_DOMAIN + _canonical_json_bytes(value)).hexdigest()
    return ModelCaseSidecarCommitmentV1(
        schema_version=_SCHEMA_VERSION,
        policy=_POLICY,
        manifest_sha256=manifest_sha256,
        envelope_sha256=envelope_sha256,
        case_index=case_index,
        capture_leaf_sha256=capture_leaf.leaf_sha256,
        observation_leaf_sha256s=observation_hashes,
        probe_leaf_sha256=probe_leaf.leaf_sha256,
        case_root_sha256=root,
    )


def _validate_case_sidecar_schema_v1(sidecar: object) -> ModelCaseSidecarV1:
    if (
        type(sidecar) is not ModelCaseSidecarV1
        or type(sidecar.capture) is not ModelCaptureSidecarV1
        or type(sidecar.observations) is not tuple
        or type(sidecar.probe) is not ModelProbeSidecarV1
        or type(sidecar.capture.case_index) is not int
        or sidecar.capture.case_index not in range(helpfulness.MODEL_CASE_COUNT)
    ):
        raise ModelSidecarJoinError("closed_schema")
    return sidecar


def _validate_sidecar_run_identity_v1(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> tuple[str, str]:
    if (
        type(manifest) is not helpfulness.ModelRunManifest
        or type(envelope) is not RunEnvelopeV2
    ):
        raise ModelSidecarJoinError("closed_schema")
    try:
        validate_infbridge_run_envelope_v2(manifest, tokenizer, envelope)
        manifest_sha256 = helpfulness.model_run_manifest_sha256(
            manifest,
            tokenizer,
        )
        envelope_sha256 = infbridge_run_envelope_v2_sha256(envelope)
    except (InfBridgeModelAdapterError, helpfulness.ModelProtocolError) as error:
        raise ModelSidecarJoinError("run_identity") from error
    return manifest_sha256, envelope_sha256


def _validate_live_model_case_sidecar_against_run_v1(
    manifest: helpfulness.ModelRunManifest,
    sidecar_value: object,
    *,
    manifest_sha256: str,
    envelope_sha256: str,
) -> _ValidatedCaseDetailsV1:
    sidecar = _validate_case_sidecar_schema_v1(sidecar_value)
    case_index = sidecar.capture.case_index
    registration = manifest.cases[case_index]
    expected_arms = tuple(call.arm for call in registration.arm_calls)
    if (
        len(sidecar.observations) != _OBSERVATION_COUNT
        or tuple(type(row) for row in sidecar.observations)
        != (ModelObservationSidecarV1,) * _OBSERVATION_COUNT
        or tuple(row.arm for row in sidecar.observations) != expected_arms
    ):
        raise ModelSidecarJoinError("inventory_mismatch")

    capture_request, capture_response, capture_leaf = _validate_capture(
        registration,
        sidecar.capture,
    )
    observation_rows = tuple(
        _validate_observation(
            registration=registration,
            arm_call=arm_call,
            arm_offset=arm_offset,
            capture_request=capture_request,
            capture_response=capture_response,
            sidecar=row,
        )
        for arm_offset, (arm_call, row) in enumerate(
            zip(registration.arm_calls, sidecar.observations, strict=True)
        )
    )
    probe_row = _validate_probe(
        registration=registration,
        capture_request=capture_request,
        capture_response=capture_response,
        sidecar=sidecar.probe,
    )
    _validate_case_identity_separation(
        registration=registration,
        capture_response=capture_response,
        observation_rows=observation_rows,
        probe_row=probe_row,
    )
    try:
        final_receipt = helpfulness._model_capture_database_receipt(
            capture_request.database_path,
            expected_identity=(
                capture_response.database_receipt.device,
                capture_response.database_receipt.inode,
            ),
            durable=False,
        )
    except helpfulness.ChildExecutionValidationError as error:
        raise ModelSidecarJoinError("receipt_mismatch") from error
    if final_receipt != capture_response.database_receipt:
        raise ModelSidecarJoinError("receipt_mismatch")

    observation_leaves = tuple(row[2] for row in observation_rows)
    probe_leaf = probe_row[2]
    commitment = _case_commitment(
        manifest_sha256=manifest_sha256,
        envelope_sha256=envelope_sha256,
        case_index=case_index,
        capture_leaf=capture_leaf,
        observation_leaves=observation_leaves,
        probe_leaf=probe_leaf,
    )
    leaves = (capture_leaf, *observation_leaves, probe_leaf)
    result = ValidatedModelCaseSidecarV1(
        commitment=commitment,
        leaves=leaves,
        capture=sidecar.capture,
        observations=sidecar.observations,
        probe=sidecar.probe,
    )
    return _ValidatedCaseDetailsV1(
        result=result,
        capture_request=capture_request,
        capture_response=capture_response,
        observation_rows=observation_rows,
        probe_row=probe_row,
    )


def validate_live_model_case_sidecar_v1(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    sidecar: ModelCaseSidecarV1,
) -> ValidatedModelCaseSidecarV1:
    """Validate one live 1+6+1 sidecar without loading model responses."""

    _validate_case_sidecar_schema_v1(sidecar)
    manifest_sha256, envelope_sha256 = _validate_sidecar_run_identity_v1(
        manifest,
        tokenizer,
        envelope,
    )
    return _validate_live_model_case_sidecar_against_run_v1(
        manifest,
        sidecar,
        manifest_sha256=manifest_sha256,
        envelope_sha256=envelope_sha256,
    ).result


def _aggregate_wire_byte_count(sidecar: ModelCaseSidecarV1) -> int:
    records = (sidecar.capture, *sidecar.observations, sidecar.probe)
    total = 0
    for record in records:
        exchange = record.exchange
        if (
            type(exchange) is not CanonicalChildExchangeV1
            or type(exchange.request_wire_utf8) is not bytes
            or type(exchange.response_wire_utf8) is not bytes
            or not exchange.request_wire_utf8
            or not exchange.response_wire_utf8
            or len(exchange.request_wire_utf8) > _MAX_CHILD_WIRE_BYTES
            or len(exchange.response_wire_utf8) > _MAX_CHILD_WIRE_BYTES
        ):
            raise ModelSidecarJoinError("closed_schema")
        total += len(exchange.request_wire_utf8) + len(exchange.response_wire_utf8)
    return total


def _validate_aggregate_identity_separation_v1(
    manifest: helpfulness.ModelRunManifest,
    details: tuple[_ValidatedCaseDetailsV1, ...],
) -> None:
    execution_tokens: list[int] = []
    process_ids: list[str] = []
    future_sessions: list[str] = []
    future_runs: list[str] = []
    component_ids: list[str] = []
    database_paths: list[str] = []
    database_identities: list[tuple[int, int]] = []
    for detail in details:
        observation_requests = tuple(row[0] for row in detail.observation_rows)
        observation_responses = tuple(row[1] for row in detail.observation_rows)
        probe_request, probe_response, _probe_leaf = detail.probe_row
        execution_tokens.extend(
            request.execution_index for request in observation_requests
        )
        execution_tokens.append(probe_request.execution_index)
        process_ids.append(detail.capture_response.process_instance_id)
        process_ids.extend(
            response.process_instance_id for response in observation_responses
        )
        process_ids.append(probe_response.process_instance_id)
        future_sessions.extend(
            request.future_session_id for request in observation_requests
        )
        future_sessions.append(probe_response.probe.future_session_id)
        future_runs.extend(request.future_run_id for request in observation_requests)
        future_runs.append(probe_response.probe.future_run_id)
        component_ids.extend(
            component_id
            for response in (*observation_responses, probe_response)
            for component_id in _component_instance_ids(response)
        )
        database_paths.append(detail.capture_request.database_path)
        database_identities.append(
            (
                detail.capture_response.database_receipt.device,
                detail.capture_response.database_receipt.inode,
            )
        )

    capture_sessions = {
        session_id
        for registration in manifest.cases
        for session_id in (
            *registration.identity.references.capture.capture_session_ids,
            f"{registration.identity.case.case_id}-capture-foreign",
        )
    }
    capture_runs = {
        f"{registration.identity.case.case_id}-run-{role}"
        for registration in manifest.cases
        for role in ("old", "new", "control", "foreign")
    }
    all_logical_ids = (
        *capture_sessions,
        *capture_runs,
        *future_sessions,
        *future_runs,
    )
    expected_future_count = helpfulness.MODEL_CASE_COUNT * (_OBSERVATION_COUNT + 1)
    expected_process_count = helpfulness.MODEL_CASE_COUNT * (_OBSERVATION_COUNT + 2)
    expected_component_count = helpfulness.MODEL_CASE_COUNT * (
        _OBSERVATION_COUNT * 6 + 2
    )
    if (
        len(execution_tokens) != expected_future_count
        or len(set(execution_tokens)) != expected_future_count
        or len(process_ids) != expected_process_count
        or len(set(process_ids)) != expected_process_count
        or len(future_sessions) != expected_future_count
        or len(set(future_sessions)) != expected_future_count
        or len(future_runs) != expected_future_count
        or len(set(future_runs)) != expected_future_count
        or len(component_ids) != expected_component_count
        or len(set(component_ids)) != expected_component_count
        or len(set((*process_ids, *component_ids)))
        != expected_process_count + expected_component_count
        or len(set(all_logical_ids)) != len(all_logical_ids)
        or len(set(database_paths)) != helpfulness.MODEL_CASE_COUNT
        or len(set(database_identities)) != helpfulness.MODEL_CASE_COUNT
    ):
        raise ModelSidecarJoinError("identity_reuse")


def _final_aggregate_database_sweep_v1(
    details: tuple[_ValidatedCaseDetailsV1, ...],
) -> None:
    for detail in details:
        receipt = detail.capture_response.database_receipt
        try:
            observed = helpfulness._model_capture_database_receipt(
                detail.capture_request.database_path,
                expected_identity=(receipt.device, receipt.inode),
                durable=False,
            )
        except helpfulness.ChildExecutionValidationError as error:
            raise ModelSidecarJoinError("receipt_mismatch") from error
        if observed != receipt:
            raise ModelSidecarJoinError("receipt_mismatch")


def _aggregate_commitment_v1(
    *,
    manifest_sha256: str,
    envelope_sha256: str,
    cases: tuple[ValidatedModelCaseSidecarV1, ...],
) -> ModelSidecarAggregateCommitmentV1:
    case_roots = tuple(case.commitment.case_root_sha256 for case in cases)
    case_count = helpfulness.MODEL_CASE_COUNT
    observation_count = case_count * _OBSERVATION_COUNT
    leaf_count = case_count * (_OBSERVATION_COUNT + 2)
    value = {
        "capture_count": case_count,
        "case_count": case_count,
        "case_root_sha256s": list(case_roots),
        "envelope_sha256": envelope_sha256,
        "leaf_count": leaf_count,
        "manifest_sha256": manifest_sha256,
        "observation_count": observation_count,
        "policy": _POLICY,
        "probe_count": case_count,
        "schema_version": _SCHEMA_VERSION,
    }
    root = hashlib.sha256(
        _AGGREGATE_ROOT_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()
    return ModelSidecarAggregateCommitmentV1(
        schema_version=_SCHEMA_VERSION,
        policy=_POLICY,
        manifest_sha256=manifest_sha256,
        envelope_sha256=envelope_sha256,
        case_count=case_count,
        capture_count=case_count,
        observation_count=observation_count,
        probe_count=case_count,
        leaf_count=leaf_count,
        case_root_sha256s=case_roots,
        aggregate_root_sha256=root,
    )


def validate_live_model_sidecar_aggregate_v1(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    sidecars: tuple[ModelCaseSidecarV1, ...],
) -> ValidatedModelSidecarAggregateV1:
    """Validate all 64 response-free sidecar cases before ledger access.

    The final database sweep narrows ordinary sequential-validation drift but
    does not make 64 independent Memory databases one atomic snapshot.
    """

    if type(sidecars) is not tuple or len(sidecars) != helpfulness.MODEL_CASE_COUNT:
        raise ModelSidecarJoinError("inventory_mismatch")
    typed_sidecars = tuple(_validate_case_sidecar_schema_v1(row) for row in sidecars)
    if tuple(row.capture.case_index for row in typed_sidecars) != tuple(
        range(helpfulness.MODEL_CASE_COUNT)
    ):
        raise ModelSidecarJoinError("inventory_mismatch")
    if any(
        len(row.observations) != _OBSERVATION_COUNT
        or any(type(item) is not ModelObservationSidecarV1 for item in row.observations)
        for row in typed_sidecars
    ):
        raise ModelSidecarJoinError("inventory_mismatch")
    total_wire_bytes = sum(_aggregate_wire_byte_count(row) for row in typed_sidecars)
    if total_wire_bytes > _MAX_AGGREGATE_WIRE_BYTES:
        raise ModelSidecarJoinError("closed_schema")

    manifest_sha256, envelope_sha256 = _validate_sidecar_run_identity_v1(
        manifest,
        tokenizer,
        envelope,
    )
    details = tuple(
        _validate_live_model_case_sidecar_against_run_v1(
            manifest,
            row,
            manifest_sha256=manifest_sha256,
            envelope_sha256=envelope_sha256,
        )
        for row in typed_sidecars
    )
    _validate_aggregate_identity_separation_v1(manifest, details)
    _final_aggregate_database_sweep_v1(details)
    cases = tuple(detail.result for detail in details)
    commitment = _aggregate_commitment_v1(
        manifest_sha256=manifest_sha256,
        envelope_sha256=envelope_sha256,
        cases=cases,
    )
    return ValidatedModelSidecarAggregateV1(
        commitment=commitment,
        cases=cases,
    )
