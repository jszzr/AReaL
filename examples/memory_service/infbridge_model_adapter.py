# SPDX-License-Identifier: Apache-2.0

"""Strict client-local InfBridge evidence for the Memory helpfulness run.

This module is intentionally parallel to the v1 model-boundary receipt types in
``scoped_codebook_eval``.  It never upgrades those self-reported receipts into
remote attestation.  The evidence below binds preregistered calls to what the
local InfBridge prepared, retried, parsed, and decoded; it cannot authenticate
the remote server, model identity, or model weights.  It freezes the supplied
decoder's concrete type but does not prove that the object came from the
tokenizer artifact named by the manifest; formal analysis must load that
artifact through a trusted verifier.  This module also provides only per-call
evidence.  Run ordering, at-most-once scheduling, attrition, and completeness
belong to the durable run ledger layered above it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from types import CodeType, FunctionType, MethodType
from typing import Protocol

from examples.memory_service import scoped_codebook_eval as helpfulness

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.v2.inference_service.client_trace import (
    GenerationPhysicalTrace,
    GenerationResponseEvidence,
    ParsedResponseJSONEvidence,
    generation_physical_trace_bytes,
    generation_physical_trace_from_bytes,
    generation_physical_trace_sha256,
    generation_response_evidence_bytes,
    generation_response_evidence_from_bytes,
    generation_response_evidence_sha256,
    generation_response_evidence_values,
    prepared_request_json_sha256,
    validate_generation_physical_trace_response,
)
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import (
    GenerationTraceValidationError,
    InfBridge,
)
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

__all__ = [
    "AuditedModelCallExecutionV2",
    "AuditedModelCallReceiptV2",
    "AuditedDecoderTokenizer",
    "CallPlanV2",
    "DecoderAuditMaterialV2",
    "EvidencePolicyV2",
    "InfBridgeModelAdapter",
    "InfBridgeModelAdapterError",
    "ModelAdapterDecodingV2",
    "RunEnvelopeV2",
    "RuntimeConfigV2",
    "audited_model_call_execution_v2_from_artifacts",
    "audited_model_call_receipt_v2_bytes",
    "audited_model_call_receipt_v2_from_bytes",
    "audited_model_call_receipt_v2_sha256",
    "infbridge_run_envelope_v2_bytes",
    "infbridge_run_envelope_v2_from_bytes",
    "infbridge_run_envelope_v2_sha256",
    "prepare_infbridge_run_envelope_v2",
    "validate_infbridge_model_call_v2",
    "validate_infbridge_run_envelope_v2",
]

_ADAPTER_ALGORITHM = "areal-memory-infbridge-adapter-v2"
_EXECUTION_MODE = "single-call-evidence-only"
_DECODE_POLICY = "tokenizer-decode-skip-special-no-cleanup-v2"
_TOKEN_IDS_DOMAIN = b"areal-memory-token-ids-v2\0"
_REQUEST_ID_DOMAIN = b"areal-memory-infbridge-request-id-v2\0"
_RUNTIME_CONFIG_DOMAIN = b"areal-memory-infbridge-runtime-config-v2\0"
_RECEIPT_KIND = "areal-memory-audited-model-call-receipt-v2"
_ENVELOPE_KIND = "areal-memory-infbridge-run-envelope-v2"
_PAUSE_STATE_KIND = f"{PauseState.__module__}.{PauseState.__qualname__}"
_DECODER_CALLABLE_SOURCE_DOMAIN = b"areal-memory-decoder-callable-source-v2\0"
_DECODER_CALLABLE_RUNTIME_DOMAIN = b"areal-memory-decoder-callable-runtime-v2\0"
_DECODER_STATE_DOMAIN = b"areal-memory-decoder-state-v2\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class AuditedDecoderTokenizer(Protocol):
    """Tokenizer surface implemented only by a trusted, state-complete wrapper."""

    def encode(
        self,
        value: bytes,
        *,
        add_special_tokens: bool,
    ) -> tuple[int, ...]: ...

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...

    def memory_audit_material(self) -> DecoderAuditMaterialV2: ...


@dataclass(frozen=True, slots=True)
class DecoderAuditMaterialV2:
    """Trusted wrapper snapshot used to bind tokenizer artifact and state.

    ``decoder_state_bytes`` must cover every mutable input that can affect
    decoding: tokenizer config and special tokens, added vocabulary, referenced
    globals/helpers, native-library versions, and wrapper options.  The adapter
    can verify this contract but cannot make an incomplete caller snapshot
    complete; formal runs therefore require a reviewed concrete wrapper.
    """

    tokenizer_id: str
    tokenizer_artifact_bytes: bytes = field(repr=False)
    decoder_state_bytes: bytes = field(repr=False)


class InfBridgeModelAdapterError(RuntimeError):
    """A closed, user-inspectable reason for rejecting adapter evidence."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("adapter error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ModelAdapterDecodingV2:
    mode: str
    n_samples: int
    temperature: str
    top_p: str
    top_k: int
    max_new_tokens: int
    stop_token_ids: tuple[int, ...]
    ignore_eos: bool
    skip_special_tokens: bool
    stop_sequences: tuple[str, ...]
    frequency_penalty: str
    use_beam_search: bool
    with_lora: bool
    decode_policy: str
    decoder_kind: str
    tokenizer_id: str
    tokenizer_artifact_sha256: str
    decoder_state_sha256: str
    audit_callable_source_sha256: str
    audit_callable_runtime_sha256: str
    decoder_callable_source_sha256: str
    decoder_callable_runtime_sha256: str
    python_cache_tag: str
    python_version: str


@dataclass(frozen=True, slots=True)
class EvidencePolicyV2:
    """Post-parse persistence acceptance limits for response sidecars.

    These limits do not cap HTTP body allocation; transport-level streaming
    limits require a separate InfBridge/server change.
    """

    schema_version: int
    require_response_preimages: bool
    max_response_json_bytes_per_attempt: int
    max_response_evidence_bytes_per_call: int


@dataclass(frozen=True, slots=True)
class RuntimeConfigV2:
    backend_kind: str
    backend_addr_sha256: str
    configured_attempt_limit: int
    request_timeout_hex: str
    resubmit_wait_hex: str
    pause_state_kind: str
    expected_client_version: int
    expected_client_version_epoch: int


@dataclass(frozen=True, slots=True)
class CallPlanV2:
    slot_index: int
    case_index: int
    arm: str
    request_id: str
    input_token_ids_sha256: str
    input_token_count: int
    expected_endpoint: str
    expected_method: str
    first_prepared_request_json_sha256: str


@dataclass(frozen=True, slots=True)
class RunEnvelopeV2:
    schema_version: int
    adapter_algorithm: str
    execution_mode: str
    manifest_sha256: str
    runtime_config_sha256: str
    decoding: ModelAdapterDecodingV2
    evidence_policy: EvidencePolicyV2
    runtime: RuntimeConfigV2
    call_plans: tuple[CallPlanV2, ...]


@dataclass(frozen=True, slots=True)
class AuditedModelCallReceiptV2:
    schema_version: int
    manifest_sha256: str
    run_envelope_sha256: str
    slot_index: int
    case_index: int
    arm: str
    request_id: str
    generation_trace_sha256: str
    generation_response_evidence_sha256: str
    generation_response_evidence_byte_count: int
    decoded_response_utf8_sha256: str
    decoded_response_utf8_bytes: int


@dataclass(frozen=True, slots=True)
class AuditedModelCallExecutionV2:
    response: str
    trace: GenerationPhysicalTrace
    response_evidence: GenerationResponseEvidence
    receipt: AuditedModelCallReceiptV2
    legacy_execution: helpfulness.ModelCallExecution


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON numbers are forbidden")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON keys are forbidden")
        value[key] = item
    return value


def _parse_canonical_json_object(
    value: object,
    *,
    max_bytes: int,
    reason: str,
) -> dict[str, object]:
    if type(value) is not bytes or not value or len(value) > max_bytes:
        raise InfBridgeModelAdapterError(reason)
    try:
        text = value.decode("ascii")
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        if type(decoded) is not dict or _canonical_json_bytes(decoded) != value:
            raise ValueError("artifact is not one canonical JSON object")
    except (
        UnicodeDecodeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ) as error:
        raise InfBridgeModelAdapterError(reason) from error
    return decoded


def _require_json_keys(
    value: object,
    expected: frozenset[str],
    *,
    reason: str,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected:
        raise InfBridgeModelAdapterError(reason)
    return value


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _type_kind(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_source_sha256(value: object) -> str:
    function = getattr(value, "__func__", value)
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if type(module) is not str or type(qualname) is not str:
        raise InfBridgeModelAdapterError("decoder_mismatch")
    try:
        source = textwrap.dedent(inspect.getsource(function))
        normalized_source = (
            "\n".join(line.rstrip() for line in source.splitlines()).rstrip() + "\n"
        )
    except (OSError, TypeError) as error:
        raise InfBridgeModelAdapterError("decoder_mismatch") from error
    value_tree = {
        "module": module,
        "qualname": qualname,
        "normalized_source_text": normalized_source,
    }
    return hashlib.sha256(
        _DECODER_CALLABLE_SOURCE_DOMAIN + _canonical_json_bytes(value_tree)
    ).hexdigest()


def _runtime_code_constant_value(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) is tuple:
        return {"tuple": [_runtime_code_constant_value(item) for item in value]}
    if type(value) is frozenset:
        items = [_runtime_code_constant_value(item) for item in value]
        return {
            "frozenset": sorted(
                items,
                key=_canonical_json_bytes,
            )
        }
    if type(value) is CodeType:
        return {"code": _runtime_code_value(value)}
    if value is Ellipsis:
        return {"ellipsis": True}
    raise InfBridgeModelAdapterError("decoder_mismatch")


def _runtime_code_value(code: CodeType) -> dict[str, object]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_hex": code.co_code.hex(),
        "exceptiontable_hex": code.co_exceptiontable.hex(),
        "consts": [_runtime_code_constant_value(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_runtime_sha256(value: object) -> str:
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    if type(code) is not CodeType:
        raise InfBridgeModelAdapterError("decoder_mismatch")
    value_tree = {
        "python_cache_tag": sys.implementation.cache_tag,
        "runtime_code": _runtime_code_value(code),
    }
    return hashlib.sha256(
        _DECODER_CALLABLE_RUNTIME_DOMAIN + _canonical_json_bytes(value_tree)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class _DecoderObservationV2:
    decoder_kind: str
    tokenizer_id: str
    tokenizer_artifact_sha256: str
    decoder_state_sha256: str
    audit_callable_source_sha256: str
    audit_callable_runtime_sha256: str
    decoder_callable_source_sha256: str
    decoder_callable_runtime_sha256: str
    python_cache_tag: str
    python_version: str


def _bound_instance_method(value: object, name: str) -> MethodType:
    try:
        bound_method = getattr(value, name)
        declared_method = inspect.getattr_static(type(value), name)
    except Exception as error:
        raise InfBridgeModelAdapterError("decoder_mismatch") from error
    if (
        type(bound_method) is not MethodType
        or bound_method.__self__ is not value
        or type(bound_method.__func__) is not FunctionType
        or declared_method is not bound_method.__func__
    ):
        raise InfBridgeModelAdapterError("decoder_mismatch")
    return bound_method


def _decoder_snapshot(
    tokenizer: object,
) -> tuple[_DecoderObservationV2, MethodType]:
    audit_callable = _bound_instance_method(tokenizer, "memory_audit_material")
    try:
        material = audit_callable()
    except Exception as error:
        raise InfBridgeModelAdapterError("decoder_mismatch") from error
    decode_callable = _bound_instance_method(tokenizer, "decode")
    if (
        type(material) is not DecoderAuditMaterialV2
        or type(material.tokenizer_id) is not str
        or not material.tokenizer_id
        or type(material.tokenizer_artifact_bytes) is not bytes
        or not material.tokenizer_artifact_bytes
        or len(material.tokenizer_artifact_bytes) > 67_108_864
        or type(material.decoder_state_bytes) is not bytes
        or not material.decoder_state_bytes
        or len(material.decoder_state_bytes) > 8_388_608
    ):
        raise InfBridgeModelAdapterError("decoder_mismatch")
    python_cache_tag = sys.implementation.cache_tag
    if type(python_cache_tag) is not str or not python_cache_tag:
        raise InfBridgeModelAdapterError("decoder_mismatch")
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    return (
        _DecoderObservationV2(
            decoder_kind=_type_kind(tokenizer),
            tokenizer_id=material.tokenizer_id,
            tokenizer_artifact_sha256=hashlib.sha256(
                material.tokenizer_artifact_bytes
            ).hexdigest(),
            decoder_state_sha256=hashlib.sha256(
                _DECODER_STATE_DOMAIN + material.decoder_state_bytes
            ).hexdigest(),
            audit_callable_source_sha256=_callable_source_sha256(audit_callable),
            audit_callable_runtime_sha256=_callable_runtime_sha256(audit_callable),
            decoder_callable_source_sha256=_callable_source_sha256(decode_callable),
            decoder_callable_runtime_sha256=_callable_runtime_sha256(decode_callable),
            python_cache_tag=python_cache_tag,
            python_version=python_version,
        ),
        decode_callable,
    )


def _decoder_observation(tokenizer: object) -> _DecoderObservationV2:
    observation, _ = _decoder_snapshot(tokenizer)
    return observation


def _assert_decoder_matches(
    tokenizer: object,
    decoding: ModelAdapterDecodingV2,
) -> MethodType:
    observed, decode_callable = _decoder_snapshot(tokenizer)
    expected = _DecoderObservationV2(
        decoder_kind=decoding.decoder_kind,
        tokenizer_id=decoding.tokenizer_id,
        tokenizer_artifact_sha256=decoding.tokenizer_artifact_sha256,
        decoder_state_sha256=decoding.decoder_state_sha256,
        audit_callable_source_sha256=decoding.audit_callable_source_sha256,
        audit_callable_runtime_sha256=decoding.audit_callable_runtime_sha256,
        decoder_callable_source_sha256=decoding.decoder_callable_source_sha256,
        decoder_callable_runtime_sha256=decoding.decoder_callable_runtime_sha256,
        python_cache_tag=decoding.python_cache_tag,
        python_version=decoding.python_version,
    )
    if not _exact_tree_equal(observed, expected):
        raise InfBridgeModelAdapterError("decoder_mismatch")
    return decode_callable


def _float_hex(value: object, *, positive: bool) -> str:
    if type(value) not in (int, float):
        raise InfBridgeModelAdapterError("runtime_mismatch")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise InfBridgeModelAdapterError("runtime_mismatch") from error
    if not math.isfinite(normalized) or (
        normalized <= 0 if positive else normalized < 0
    ):
        raise InfBridgeModelAdapterError("runtime_mismatch")
    return normalized.hex()


def _is_canonical_float_hex(value: object, *, positive: bool) -> bool:
    if type(value) is not str:
        return False
    try:
        decoded = float.fromhex(value)
    except (OverflowError, ValueError):
        return False
    return bool(
        math.isfinite(decoded)
        and (decoded > 0 if positive else decoded >= 0)
        and decoded.hex() == value
    )


def _exact_tree_equal(actual: object, expected: object) -> bool:
    """Compare frozen evidence without Python bool/int or subclass coercion."""

    if type(actual) is not type(expected):
        return False
    if is_dataclass(expected) and not isinstance(expected, type):
        return all(
            _exact_tree_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
            for field in fields(expected)
        )
    if type(expected) is tuple:
        actual_tuple = actual
        expected_tuple = expected
        return len(actual_tuple) == len(expected_tuple) and all(  # type: ignore[arg-type]
            _exact_tree_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(  # type: ignore[arg-type]
                actual_tuple,
                expected_tuple,
                strict=True,
            )
        )
    return bool(actual == expected)


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    if type(token_ids) is not tuple or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    encoded = _canonical_json_bytes(list(token_ids))
    return hashlib.sha256(_TOKEN_IDS_DOMAIN + encoded).hexdigest()


def _decoding_value(decoding: ModelAdapterDecodingV2) -> dict[str, object]:
    return {
        "mode": decoding.mode,
        "n_samples": decoding.n_samples,
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "top_k": decoding.top_k,
        "max_new_tokens": decoding.max_new_tokens,
        "stop_token_ids": list(decoding.stop_token_ids),
        "ignore_eos": decoding.ignore_eos,
        "skip_special_tokens": decoding.skip_special_tokens,
        "stop_sequences": list(decoding.stop_sequences),
        "frequency_penalty": decoding.frequency_penalty,
        "use_beam_search": decoding.use_beam_search,
        "with_lora": decoding.with_lora,
        "decode_policy": decoding.decode_policy,
        "decoder_kind": decoding.decoder_kind,
        "tokenizer_id": decoding.tokenizer_id,
        "tokenizer_artifact_sha256": decoding.tokenizer_artifact_sha256,
        "decoder_state_sha256": decoding.decoder_state_sha256,
        "audit_callable_source_sha256": decoding.audit_callable_source_sha256,
        "audit_callable_runtime_sha256": decoding.audit_callable_runtime_sha256,
        "decoder_callable_source_sha256": decoding.decoder_callable_source_sha256,
        "decoder_callable_runtime_sha256": decoding.decoder_callable_runtime_sha256,
        "python_cache_tag": decoding.python_cache_tag,
        "python_version": decoding.python_version,
    }


def _evidence_policy_value(policy: EvidencePolicyV2) -> dict[str, object]:
    return {
        "schema_version": policy.schema_version,
        "require_response_preimages": policy.require_response_preimages,
        "max_response_json_bytes_per_attempt": (
            policy.max_response_json_bytes_per_attempt
        ),
        "max_response_evidence_bytes_per_call": (
            policy.max_response_evidence_bytes_per_call
        ),
    }


def _runtime_value(runtime: RuntimeConfigV2) -> dict[str, object]:
    return {
        "backend_kind": runtime.backend_kind,
        "backend_addr_sha256": runtime.backend_addr_sha256,
        "configured_attempt_limit": runtime.configured_attempt_limit,
        "request_timeout_hex": runtime.request_timeout_hex,
        "resubmit_wait_hex": runtime.resubmit_wait_hex,
        "pause_state_kind": runtime.pause_state_kind,
        "expected_client_version": runtime.expected_client_version,
        "expected_client_version_epoch": runtime.expected_client_version_epoch,
    }


def _runtime_config_sha256(
    decoding: ModelAdapterDecodingV2,
    evidence_policy: EvidencePolicyV2,
    runtime: RuntimeConfigV2,
) -> str:
    value = {
        "adapter_algorithm": _ADAPTER_ALGORITHM,
        "execution_mode": _EXECUTION_MODE,
        "decoding": _decoding_value(decoding),
        "evidence_policy": _evidence_policy_value(evidence_policy),
        "runtime": _runtime_value(runtime),
    }
    return hashlib.sha256(
        _RUNTIME_CONFIG_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _call_plan_value(plan: CallPlanV2) -> dict[str, object]:
    return {
        "slot_index": plan.slot_index,
        "case_index": plan.case_index,
        "arm": plan.arm,
        "request_id": plan.request_id,
        "input_token_ids_sha256": plan.input_token_ids_sha256,
        "input_token_count": plan.input_token_count,
        "expected_endpoint": plan.expected_endpoint,
        "expected_method": plan.expected_method,
        "first_prepared_request_json_sha256": (plan.first_prepared_request_json_sha256),
    }


def _validate_decoding(decoding: object) -> ModelAdapterDecodingV2:
    if type(decoding) is not ModelAdapterDecodingV2:
        raise InfBridgeModelAdapterError("run_envelope")
    if (
        type(decoding.mode) is not str
        or decoding.mode != "greedy"
        or type(decoding.n_samples) is not int
        or decoding.n_samples != 1
        or type(decoding.temperature) is not str
        or decoding.temperature != "0"
        or type(decoding.top_p) is not str
        or decoding.top_p != "1"
        or type(decoding.top_k) is not int
        or decoding.top_k != 100_000_000
        or type(decoding.max_new_tokens) is not int
        or decoding.max_new_tokens <= 0
        or type(decoding.stop_token_ids) is not tuple
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in decoding.stop_token_ids
        )
        or len(set(decoding.stop_token_ids)) != len(decoding.stop_token_ids)
        or type(decoding.ignore_eos) is not bool
        or decoding.ignore_eos is not False
        or type(decoding.skip_special_tokens) is not bool
        or decoding.skip_special_tokens is not True
        or type(decoding.stop_sequences) is not tuple
        or decoding.stop_sequences != ()
        or type(decoding.frequency_penalty) is not str
        or decoding.frequency_penalty != "0"
        or type(decoding.use_beam_search) is not bool
        or decoding.use_beam_search is not False
        or type(decoding.with_lora) is not bool
        or decoding.with_lora is not False
        or type(decoding.decode_policy) is not str
        or decoding.decode_policy != _DECODE_POLICY
        or type(decoding.decoder_kind) is not str
        or not decoding.decoder_kind
        or type(decoding.tokenizer_id) is not str
        or not decoding.tokenizer_id
        or not _is_sha256(decoding.tokenizer_artifact_sha256)
        or not _is_sha256(decoding.decoder_state_sha256)
        or not _is_sha256(decoding.audit_callable_source_sha256)
        or not _is_sha256(decoding.audit_callable_runtime_sha256)
        or not _is_sha256(decoding.decoder_callable_source_sha256)
        or not _is_sha256(decoding.decoder_callable_runtime_sha256)
        or type(decoding.python_cache_tag) is not str
        or not decoding.python_cache_tag
        or type(decoding.python_version) is not str
        or decoding.python_version
        != ".".join(str(item) for item in sys.version_info[:3])
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    return decoding


def _validate_evidence_policy(policy: object) -> EvidencePolicyV2:
    if type(policy) is not EvidencePolicyV2:
        raise InfBridgeModelAdapterError("run_envelope")
    if (
        type(policy.schema_version) is not int
        or policy.schema_version != 1
        or type(policy.require_response_preimages) is not bool
        or policy.require_response_preimages is not True
        or type(policy.max_response_json_bytes_per_attempt) is not int
        or policy.max_response_json_bytes_per_attempt <= 0
        or type(policy.max_response_evidence_bytes_per_call) is not int
        or policy.max_response_evidence_bytes_per_call
        < policy.max_response_json_bytes_per_attempt
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    return policy


def _validate_runtime(runtime: object) -> RuntimeConfigV2:
    if type(runtime) is not RuntimeConfigV2:
        raise InfBridgeModelAdapterError("run_envelope")
    if (
        type(runtime.backend_kind) is not str
        or runtime.backend_kind != SGLangBridgeBackend.__qualname__
        or not _is_sha256(runtime.backend_addr_sha256)
        or type(runtime.configured_attempt_limit) is not int
        or runtime.configured_attempt_limit <= 0
        or not _is_canonical_float_hex(runtime.request_timeout_hex, positive=True)
        or not _is_canonical_float_hex(runtime.resubmit_wait_hex, positive=False)
        or type(runtime.pause_state_kind) is not str
        or runtime.pause_state_kind != _PAUSE_STATE_KIND
        or type(runtime.expected_client_version) is not int
        or type(runtime.expected_client_version_epoch) is not int
        or runtime.expected_client_version_epoch < 0
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    return runtime


def _request_id(
    *,
    manifest_sha256: str,
    runtime_config_sha256: str,
    case_index: int,
    arm: str,
) -> str:
    if not _is_sha256(manifest_sha256) or not _is_sha256(runtime_config_sha256):
        raise InfBridgeModelAdapterError("run_envelope")
    if type(case_index) is not int or type(arm) is not str:
        raise InfBridgeModelAdapterError("run_envelope")
    slot = _canonical_json_bytes({"case_index": case_index, "arm": arm})
    digest = hashlib.sha256(
        _REQUEST_ID_DOMAIN
        + bytes.fromhex(manifest_sha256)
        + bytes.fromhex(runtime_config_sha256)
        + slot
    ).hexdigest()
    return f"areal-memory-v2-{digest}"


def _make_model_request(
    prepared_call: helpfulness.PreparedModelCall,
    plan: CallPlanV2,
    decoding: ModelAdapterDecodingV2,
) -> ModelRequest:
    if type(prepared_call) is not helpfulness.PreparedModelCall:
        raise InfBridgeModelAdapterError("run_envelope")
    gconfig = GenerationHyperparameters(
        n_samples=1,
        max_new_tokens=decoding.max_new_tokens,
        min_new_tokens=0,
        max_tokens=len(prepared_call.input_token_ids) + decoding.max_new_tokens,
        greedy=True,
        top_p=1.0,
        top_k=100_000_000,
        temperature=0.0,
        stop_token_ids=list(decoding.stop_token_ids),
        ignore_eos=False,
        skip_special_tokens=True,
        stop=None,
        frequency_penalty=0.0,
        use_beam_search=False,
    )
    return ModelRequest(
        rid=plan.request_id,
        input_ids=list(prepared_call.input_token_ids),
        gconfig=gconfig,
        metadata={},
        image_data=[],
        vision_msg_vllm=None,
    )


def _first_request_facts(
    prepared_call: helpfulness.PreparedModelCall,
    plan_stub: CallPlanV2,
    decoding: ModelAdapterDecodingV2,
    runtime: RuntimeConfigV2,
) -> tuple[str, str, str, tuple[int, ...]]:
    backend = SGLangBridgeBackend()
    request = _make_model_request(prepared_call, plan_stub, decoding)
    http_request = backend.build_generation_request(
        request,
        with_lora=False,
        version=runtime.expected_client_version,
    )
    effective_budget = backend.get_generation_max_new_tokens(http_request)
    if effective_budget != decoding.max_new_tokens:
        raise InfBridgeModelAdapterError("run_envelope")
    backend.patch_generation_request(
        http_request,
        request,
        [],
        effective_budget,
    )
    submitted = backend.snapshot_generation_input_ids(http_request)
    return (
        http_request.endpoint,
        "GET" if http_request.method == "GET" else "POST",
        prepared_request_json_sha256(http_request.payload),
        submitted,
    )


def _make_envelope(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    *,
    decoding: ModelAdapterDecodingV2,
    evidence_policy: EvidencePolicyV2,
    runtime: RuntimeConfigV2,
) -> RunEnvelopeV2:
    _validate_decoding(decoding)
    _validate_evidence_policy(evidence_policy)
    _validate_runtime(runtime)
    if (
        decoding.tokenizer_id != manifest.tokenizer_id
        or decoding.tokenizer_artifact_sha256 != manifest.tokenizer_sha256
    ):
        raise InfBridgeModelAdapterError("decoder_mismatch")
    _assert_decoder_matches(tokenizer, decoding)
    try:
        manifest_sha256 = helpfulness.model_run_manifest_sha256(manifest, tokenizer)
    except Exception as error:
        raise InfBridgeModelAdapterError("run_envelope") from error
    runtime_hash = _runtime_config_sha256(decoding, evidence_policy, runtime)
    plans: list[CallPlanV2] = []
    for case_index, registration in enumerate(manifest.cases):
        for arm_call in registration.arm_calls:
            slot_index = len(plans)
            prepared = arm_call.prepared_call
            request_id = _request_id(
                manifest_sha256=manifest_sha256,
                runtime_config_sha256=runtime_hash,
                case_index=case_index,
                arm=arm_call.arm,
            )
            stub = CallPlanV2(
                slot_index=slot_index,
                case_index=case_index,
                arm=arm_call.arm,
                request_id=request_id,
                input_token_ids_sha256=_token_ids_sha256(prepared.input_token_ids),
                input_token_count=len(prepared.input_token_ids),
                expected_endpoint="/generate",
                expected_method="POST",
                first_prepared_request_json_sha256="0" * 64,
            )
            endpoint, method, request_hash, submitted = _first_request_facts(
                prepared,
                stub,
                decoding,
                runtime,
            )
            if submitted != prepared.input_token_ids:
                raise InfBridgeModelAdapterError("run_envelope")
            plans.append(
                CallPlanV2(
                    slot_index=slot_index,
                    case_index=case_index,
                    arm=arm_call.arm,
                    request_id=request_id,
                    input_token_ids_sha256=stub.input_token_ids_sha256,
                    input_token_count=stub.input_token_count,
                    expected_endpoint=endpoint,
                    expected_method=method,
                    first_prepared_request_json_sha256=request_hash,
                )
            )
    envelope = RunEnvelopeV2(
        schema_version=2,
        adapter_algorithm=_ADAPTER_ALGORITHM,
        execution_mode=_EXECUTION_MODE,
        manifest_sha256=manifest_sha256,
        runtime_config_sha256=runtime_hash,
        decoding=decoding,
        evidence_policy=evidence_policy,
        runtime=runtime,
        call_plans=tuple(plans),
    )
    infbridge_run_envelope_v2_bytes(envelope)
    return envelope


def _validate_envelope_shape(envelope: object) -> RunEnvelopeV2:
    if type(envelope) is not RunEnvelopeV2:
        raise InfBridgeModelAdapterError("run_envelope")
    decoding = _validate_decoding(envelope.decoding)
    evidence_policy = _validate_evidence_policy(envelope.evidence_policy)
    runtime = _validate_runtime(envelope.runtime)
    expected_call_count = helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
    if (
        type(envelope.schema_version) is not int
        or envelope.schema_version != 2
        or type(envelope.adapter_algorithm) is not str
        or envelope.adapter_algorithm != _ADAPTER_ALGORITHM
        or type(envelope.execution_mode) is not str
        or envelope.execution_mode != _EXECUTION_MODE
        or not _is_sha256(envelope.manifest_sha256)
        or not _is_sha256(envelope.runtime_config_sha256)
        or envelope.runtime_config_sha256
        != _runtime_config_sha256(decoding, evidence_policy, runtime)
        or type(envelope.call_plans) is not tuple
        or len(envelope.call_plans) != expected_call_count
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    for slot_index, plan in enumerate(envelope.call_plans):
        expected_case_index, arm_offset = divmod(
            slot_index,
            len(helpfulness.MODEL_ARMS),
        )
        expected_arm = helpfulness.model_arm_order(expected_case_index)[arm_offset]
        if (
            type(plan) is not CallPlanV2
            or type(plan.slot_index) is not int
            or plan.slot_index != slot_index
            or type(plan.case_index) is not int
            or plan.case_index != expected_case_index
            or type(plan.arm) is not str
            or plan.arm != expected_arm
            or type(plan.request_id) is not str
            or plan.request_id
            != _request_id(
                manifest_sha256=envelope.manifest_sha256,
                runtime_config_sha256=envelope.runtime_config_sha256,
                case_index=plan.case_index,
                arm=plan.arm,
            )
            or not _is_sha256(plan.input_token_ids_sha256)
            or type(plan.input_token_count) is not int
            or plan.input_token_count <= 0
            or type(plan.expected_endpoint) is not str
            or plan.expected_endpoint != "/generate"
            or type(plan.expected_method) is not str
            or plan.expected_method != "POST"
            or not _is_sha256(plan.first_prepared_request_json_sha256)
        ):
            raise InfBridgeModelAdapterError("run_envelope")
    return envelope


def infbridge_run_envelope_v2_bytes(envelope: RunEnvelopeV2) -> bytes:
    envelope = _validate_envelope_shape(envelope)
    value = {
        "kind": _ENVELOPE_KIND,
        "schema_version": envelope.schema_version,
        "adapter_algorithm": envelope.adapter_algorithm,
        "execution_mode": envelope.execution_mode,
        "manifest_sha256": envelope.manifest_sha256,
        "runtime_config_sha256": envelope.runtime_config_sha256,
        "decoding": _decoding_value(envelope.decoding),
        "evidence_policy": _evidence_policy_value(envelope.evidence_policy),
        "runtime": _runtime_value(envelope.runtime),
        "call_plans": [_call_plan_value(plan) for plan in envelope.call_plans],
    }
    return _canonical_json_bytes(value)


def infbridge_run_envelope_v2_from_bytes(value: bytes) -> RunEnvelopeV2:
    """Strictly load one canonical envelope for the current Python runtime."""

    reason = "run_envelope"
    decoded = _parse_canonical_json_object(
        value,
        max_bytes=8_388_608,
        reason=reason,
    )
    _require_json_keys(
        decoded,
        frozenset(
            {
                "kind",
                "schema_version",
                "adapter_algorithm",
                "execution_mode",
                "manifest_sha256",
                "runtime_config_sha256",
                "decoding",
                "evidence_policy",
                "runtime",
                "call_plans",
            }
        ),
        reason=reason,
    )
    if decoded["kind"] != _ENVELOPE_KIND:
        raise InfBridgeModelAdapterError(reason)
    decoding_value = _require_json_keys(
        decoded["decoding"],
        frozenset(
            {
                "mode",
                "n_samples",
                "temperature",
                "top_p",
                "top_k",
                "max_new_tokens",
                "stop_token_ids",
                "ignore_eos",
                "skip_special_tokens",
                "stop_sequences",
                "frequency_penalty",
                "use_beam_search",
                "with_lora",
                "decode_policy",
                "decoder_kind",
                "tokenizer_id",
                "tokenizer_artifact_sha256",
                "decoder_state_sha256",
                "audit_callable_source_sha256",
                "audit_callable_runtime_sha256",
                "decoder_callable_source_sha256",
                "decoder_callable_runtime_sha256",
                "python_cache_tag",
                "python_version",
            }
        ),
        reason=reason,
    )
    evidence_value = _require_json_keys(
        decoded["evidence_policy"],
        frozenset(
            {
                "schema_version",
                "require_response_preimages",
                "max_response_json_bytes_per_attempt",
                "max_response_evidence_bytes_per_call",
            }
        ),
        reason=reason,
    )
    runtime_value = _require_json_keys(
        decoded["runtime"],
        frozenset(
            {
                "backend_kind",
                "backend_addr_sha256",
                "configured_attempt_limit",
                "request_timeout_hex",
                "resubmit_wait_hex",
                "pause_state_kind",
                "expected_client_version",
                "expected_client_version_epoch",
            }
        ),
        reason=reason,
    )
    stop_token_ids = decoding_value["stop_token_ids"]
    stop_sequences = decoding_value["stop_sequences"]
    call_plan_values = decoded["call_plans"]
    if (
        type(stop_token_ids) is not list
        or type(stop_sequences) is not list
        or type(call_plan_values) is not list
    ):
        raise InfBridgeModelAdapterError(reason)
    decoding_arguments = dict(decoding_value)
    decoding_arguments["stop_token_ids"] = tuple(stop_token_ids)
    decoding_arguments["stop_sequences"] = tuple(stop_sequences)
    call_plan_keys = frozenset(
        {
            "slot_index",
            "case_index",
            "arm",
            "request_id",
            "input_token_ids_sha256",
            "input_token_count",
            "expected_endpoint",
            "expected_method",
            "first_prepared_request_json_sha256",
        }
    )
    call_plans = tuple(
        CallPlanV2(
            **_require_json_keys(item, call_plan_keys, reason=reason)  # type: ignore[arg-type]
        )
        for item in call_plan_values
    )
    envelope = RunEnvelopeV2(
        schema_version=decoded["schema_version"],  # type: ignore[arg-type]
        adapter_algorithm=decoded["adapter_algorithm"],  # type: ignore[arg-type]
        execution_mode=decoded["execution_mode"],  # type: ignore[arg-type]
        manifest_sha256=decoded["manifest_sha256"],  # type: ignore[arg-type]
        runtime_config_sha256=decoded["runtime_config_sha256"],  # type: ignore[arg-type]
        decoding=ModelAdapterDecodingV2(
            **decoding_arguments,  # type: ignore[arg-type]
        ),
        evidence_policy=EvidencePolicyV2(
            **evidence_value,  # type: ignore[arg-type]
        ),
        runtime=RuntimeConfigV2(
            **runtime_value,  # type: ignore[arg-type]
        ),
        call_plans=call_plans,
    )
    _validate_envelope_shape(envelope)
    if infbridge_run_envelope_v2_bytes(envelope) != value:
        raise InfBridgeModelAdapterError(reason)
    return envelope


def infbridge_run_envelope_v2_sha256(envelope: RunEnvelopeV2) -> str:
    return hashlib.sha256(infbridge_run_envelope_v2_bytes(envelope)).hexdigest()


def _bridge_runtime(bridge: InfBridge) -> RuntimeConfigV2:
    if type(bridge) is not InfBridge or type(bridge.backend) is not SGLangBridgeBackend:
        raise InfBridgeModelAdapterError("unsupported_backend")
    if (
        type(bridge.backend_addr) is not str
        or not bridge.backend_addr
        or type(bridge.max_resubmit_retries) is not int
        or bridge.max_resubmit_retries <= 0
        or type(bridge.pause_state) is not PauseState
        or type(bridge.get_version()) is not int
        or type(bridge.get_version_epoch()) is not int
        or bridge.get_version_epoch() < 0
    ):
        raise InfBridgeModelAdapterError("runtime_mismatch")
    try:
        backend_addr_sha256 = hashlib.sha256(
            bridge.backend_addr.encode("utf-8", errors="strict")
        ).hexdigest()
    except UnicodeEncodeError as error:
        raise InfBridgeModelAdapterError("runtime_mismatch") from error
    return RuntimeConfigV2(
        backend_kind=type(bridge.backend).__qualname__,
        backend_addr_sha256=backend_addr_sha256,
        configured_attempt_limit=bridge.max_resubmit_retries,
        request_timeout_hex=_float_hex(bridge.request_timeout, positive=True),
        resubmit_wait_hex=_float_hex(bridge.resubmit_wait, positive=False),
        pause_state_kind=_type_kind(bridge.pause_state),
        expected_client_version=bridge.get_version(),
        expected_client_version_epoch=bridge.get_version_epoch(),
    )


def prepare_infbridge_run_envelope_v2(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    bridge: InfBridge,
    *,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...] = (),
    max_response_json_bytes_per_attempt: int = 1_048_576,
    max_response_evidence_bytes_per_call: int = 8_388_608,
) -> RunEnvelopeV2:
    """Freeze all 384 SGLang text calls without invoking the transport."""

    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise InfBridgeModelAdapterError("run_envelope")
    if type(stop_token_ids) is not tuple or any(
        type(token_id) is not int or token_id < 0 for token_id in stop_token_ids
    ):
        raise InfBridgeModelAdapterError("run_envelope")
    evidence_policy = EvidencePolicyV2(
        schema_version=1,
        require_response_preimages=True,
        max_response_json_bytes_per_attempt=max_response_json_bytes_per_attempt,
        max_response_evidence_bytes_per_call=(max_response_evidence_bytes_per_call),
    )
    _validate_evidence_policy(evidence_policy)
    decoder = _decoder_observation(tokenizer)
    if (
        decoder.tokenizer_id != manifest.tokenizer_id
        or decoder.tokenizer_artifact_sha256 != manifest.tokenizer_sha256
    ):
        raise InfBridgeModelAdapterError("decoder_mismatch")
    decoding = ModelAdapterDecodingV2(
        mode="greedy",
        n_samples=1,
        temperature="0",
        top_p="1",
        top_k=100_000_000,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_sequences=(),
        frequency_penalty="0",
        use_beam_search=False,
        with_lora=False,
        decode_policy=_DECODE_POLICY,
        decoder_kind=decoder.decoder_kind,
        tokenizer_id=decoder.tokenizer_id,
        tokenizer_artifact_sha256=decoder.tokenizer_artifact_sha256,
        decoder_state_sha256=decoder.decoder_state_sha256,
        audit_callable_source_sha256=decoder.audit_callable_source_sha256,
        audit_callable_runtime_sha256=decoder.audit_callable_runtime_sha256,
        decoder_callable_source_sha256=decoder.decoder_callable_source_sha256,
        decoder_callable_runtime_sha256=decoder.decoder_callable_runtime_sha256,
        python_cache_tag=decoder.python_cache_tag,
        python_version=decoder.python_version,
    )
    return _make_envelope(
        manifest,
        tokenizer,
        decoding=decoding,
        evidence_policy=evidence_policy,
        runtime=_bridge_runtime(bridge),
    )


def _manifest_call(
    manifest: helpfulness.ModelRunManifest,
    *,
    case_index: int,
    arm: str,
) -> tuple[helpfulness.PreparedModelCall, int]:
    if (
        type(case_index) is not int
        or case_index not in range(helpfulness.MODEL_CASE_COUNT)
        or type(arm) is not str
        or arm not in helpfulness.MODEL_ARMS
    ):
        raise InfBridgeModelAdapterError("call_slot")
    registration = manifest.cases[case_index]
    for arm_offset, arm_call in enumerate(registration.arm_calls):
        if arm_call.arm == arm:
            return (
                arm_call.prepared_call,
                case_index * len(helpfulness.MODEL_ARMS) + arm_offset,
            )
    raise InfBridgeModelAdapterError("call_slot")


def _validate_envelope_against_manifest(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> None:
    _validate_envelope_shape(envelope)
    expected = _make_envelope(
        manifest,
        tokenizer,
        decoding=envelope.decoding,
        evidence_policy=envelope.evidence_policy,
        runtime=envelope.runtime,
    )
    if not _exact_tree_equal(envelope, expected):
        raise InfBridgeModelAdapterError("run_envelope")


def validate_infbridge_run_envelope_v2(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> None:
    """Rebuild all frozen calls and verify one envelope semantically."""

    _validate_envelope_against_manifest(manifest, tokenizer, envelope)


def _assert_bridge_matches_runtime(
    bridge: InfBridge,
    runtime: RuntimeConfigV2,
    *,
    pause_state: PauseState | None = None,
) -> None:
    if pause_state is not None and bridge.pause_state is not pause_state:
        raise InfBridgeModelAdapterError("runtime_mismatch")
    observed = _bridge_runtime(bridge)
    if not _exact_tree_equal(observed, runtime):
        raise InfBridgeModelAdapterError("runtime_mismatch")


def _replay_trace_requests(
    prepared_call: helpfulness.PreparedModelCall,
    plan: CallPlanV2,
    envelope: RunEnvelopeV2,
    trace: GenerationPhysicalTrace,
) -> None:
    try:
        generation_physical_trace_bytes(trace)
    except Exception as error:
        raise InfBridgeModelAdapterError("trace_mismatch") from error
    runtime = envelope.runtime
    decoding = envelope.decoding
    if (
        trace.request_id != plan.request_id
        or trace.backend_kind != runtime.backend_kind
        or trace.backend_addr_sha256 != runtime.backend_addr_sha256
        or trace.request_input_token_ids != prepared_call.input_token_ids
        or trace.effective_max_new_tokens != decoding.max_new_tokens
        or trace.configured_attempt_limit != runtime.configured_attempt_limit
        or trace.initial_client_version != runtime.expected_client_version
        or trace.final_client_version != runtime.expected_client_version
        or trace.initial_client_version_epoch != runtime.expected_client_version_epoch
        or trace.final_client_version_epoch != runtime.expected_client_version_epoch
        or not trace.attempts
        or _token_ids_sha256(prepared_call.input_token_ids)
        != plan.input_token_ids_sha256
        or len(prepared_call.input_token_ids) != plan.input_token_count
    ):
        raise InfBridgeModelAdapterError("trace_mismatch")

    backend = SGLangBridgeBackend()
    request = _make_model_request(prepared_call, plan, decoding)
    http_request = backend.build_generation_request(
        request,
        with_lora=False,
        version=runtime.expected_client_version,
    )
    accumulated: list[int] = []
    for attempt_index, attempt in enumerate(trace.attempts):
        remaining = decoding.max_new_tokens - len(accumulated)
        if remaining <= 0:
            raise InfBridgeModelAdapterError("trace_mismatch")
        backend.patch_generation_request(
            http_request,
            request,
            accumulated,
            remaining,
        )
        submitted = backend.snapshot_generation_input_ids(http_request)
        expected_hash = prepared_request_json_sha256(http_request.payload)
        method = "GET" if http_request.method == "GET" else "POST"
        if (
            attempt.attempt_index != attempt_index
            or attempt.remaining_new_tokens != remaining
            or attempt.endpoint != plan.expected_endpoint
            or attempt.endpoint != http_request.endpoint
            or attempt.method != plan.expected_method
            or attempt.method != method
            or attempt.submitted_input_token_ids is None
            or attempt.submitted_input_token_ids != submitted
            or attempt.submitted_input_token_ids
            != prepared_call.input_token_ids + tuple(accumulated)
            or attempt.prepared_request_json_sha256 != expected_hash
            or (
                attempt_index == 0
                and expected_hash != plan.first_prepared_request_json_sha256
            )
            or attempt.client_version_before_send != runtime.expected_client_version
            or attempt.client_version_after_receive != runtime.expected_client_version
            or attempt.output_version_label != runtime.expected_client_version
            or attempt.client_version_epoch_before_send
            != runtime.expected_client_version_epoch
            or attempt.client_version_epoch_after_receive
            != runtime.expected_client_version_epoch
            or attempt.output_version_epoch != runtime.expected_client_version_epoch
        ):
            raise InfBridgeModelAdapterError("trace_mismatch")
        accumulated.extend(attempt.output_token_ids)
    if trace.terminal_reason == "attempt_limit":
        raise InfBridgeModelAdapterError("attempt_limit")


def _replay_response_evidence(
    trace: GenerationPhysicalTrace,
    response_evidence: GenerationResponseEvidence,
    policy: EvidencePolicyV2,
) -> bytes:
    if type(response_evidence) is not GenerationResponseEvidence:
        raise InfBridgeModelAdapterError("response_evidence")
    if type(response_evidence.attempts) is not tuple or any(
        type(attempt) is not ParsedResponseJSONEvidence
        or type(attempt.canonical_json_bytes) is not bytes
        or len(attempt.canonical_json_bytes)
        > policy.max_response_json_bytes_per_attempt
        for attempt in response_evidence.attempts
    ):
        raise InfBridgeModelAdapterError("response_evidence")
    try:
        response_values = generation_response_evidence_values(
            trace,
            response_evidence,
        )
        encoded_evidence = generation_response_evidence_bytes(response_evidence)
    except Exception as error:
        raise InfBridgeModelAdapterError("response_evidence") from error
    if len(encoded_evidence) > policy.max_response_evidence_bytes_per_call:
        raise InfBridgeModelAdapterError("response_evidence")

    backend = SGLangBridgeBackend()
    for attempt, response_value in zip(
        trace.attempts,
        response_values,
        strict=True,
    ):
        try:
            replayed = backend.parse_generation_response(response_value)
        except Exception as error:
            raise InfBridgeModelAdapterError("response_evidence") from error
        if (
            type(replayed.output_tokens) is not list
            or any(
                type(token_id) is not int or token_id < 0
                for token_id in replayed.output_tokens
            )
            or tuple(replayed.output_tokens) != attempt.output_token_ids
            or type(replayed.output_logprobs) is not list
            or len(replayed.output_logprobs) != attempt.output_logprob_count
            or type(replayed.stop_reason) is not str
            or replayed.stop_reason != attempt.raw_stop_reason
        ):
            raise InfBridgeModelAdapterError("response_evidence")
    return encoded_evidence


def _decode_response(
    tokenizer: AuditedDecoderTokenizer,
    trace: GenerationPhysicalTrace,
    decoding: ModelAdapterDecodingV2,
) -> tuple[str, bytes]:
    decode_callable = _assert_decoder_matches(tokenizer, decoding)
    try:
        response = decode_callable(
            trace.final_output_token_ids,
            skip_special_tokens=decoding.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:
        raise InfBridgeModelAdapterError("decode_failure") from error
    if type(response) is not str:
        raise InfBridgeModelAdapterError("decode_failure")
    _assert_decoder_matches(tokenizer, decoding)
    try:
        encoded = response.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise InfBridgeModelAdapterError("decode_failure") from error
    return response, encoded


def _expected_legacy_execution(
    prepared_call: helpfulness.PreparedModelCall,
    response: str,
) -> helpfulness.ModelCallExecution:
    receipt = helpfulness.make_model_call_receipt(
        submitted_prompt=prepared_call.prompt,
        context_start=prepared_call.context_start,
        context_end=prepared_call.context_end,
        input_token_ids=prepared_call.input_token_ids,
    )
    if not _exact_tree_equal(receipt, prepared_call.expected_receipt):
        raise InfBridgeModelAdapterError("run_envelope")
    return helpfulness.ModelCallExecution(
        response=response,
        consumer_input_receipt=prepared_call.consumer_input_receipt,
        model_call_receipt=receipt,
        rendered_context_token_count=prepared_call.rendered_context_token_count,
        valid=True,
        invalid_reason=None,
    )


def _receipt_value(receipt: AuditedModelCallReceiptV2) -> dict[str, object]:
    return {
        "kind": _RECEIPT_KIND,
        "schema_version": receipt.schema_version,
        "manifest_sha256": receipt.manifest_sha256,
        "run_envelope_sha256": receipt.run_envelope_sha256,
        "slot_index": receipt.slot_index,
        "case_index": receipt.case_index,
        "arm": receipt.arm,
        "request_id": receipt.request_id,
        "generation_trace_sha256": receipt.generation_trace_sha256,
        "generation_response_evidence_sha256": (
            receipt.generation_response_evidence_sha256
        ),
        "generation_response_evidence_byte_count": (
            receipt.generation_response_evidence_byte_count
        ),
        "decoded_response_utf8_sha256": receipt.decoded_response_utf8_sha256,
        "decoded_response_utf8_bytes": receipt.decoded_response_utf8_bytes,
    }


def _validate_receipt_shape(receipt: object) -> AuditedModelCallReceiptV2:
    if type(receipt) is not AuditedModelCallReceiptV2:
        raise InfBridgeModelAdapterError("receipt_mismatch")
    if (
        type(receipt.schema_version) is not int
        or receipt.schema_version != 2
        or not _is_sha256(receipt.manifest_sha256)
        or not _is_sha256(receipt.run_envelope_sha256)
        or type(receipt.slot_index) is not int
        or receipt.slot_index < 0
        or receipt.slot_index
        >= helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
        or type(receipt.case_index) is not int
        or receipt.case_index < 0
        or receipt.case_index >= helpfulness.MODEL_CASE_COUNT
        or type(receipt.arm) is not str
        or receipt.arm not in helpfulness.MODEL_ARMS
        or type(receipt.request_id) is not str
        or not receipt.request_id
        or not _is_sha256(receipt.generation_trace_sha256)
        or not _is_sha256(receipt.generation_response_evidence_sha256)
        or type(receipt.generation_response_evidence_byte_count) is not int
        or receipt.generation_response_evidence_byte_count <= 0
        or not _is_sha256(receipt.decoded_response_utf8_sha256)
        or type(receipt.decoded_response_utf8_bytes) is not int
        or receipt.decoded_response_utf8_bytes < 0
    ):
        raise InfBridgeModelAdapterError("receipt_mismatch")
    return receipt


def audited_model_call_receipt_v2_bytes(
    receipt: AuditedModelCallReceiptV2,
) -> bytes:
    return _canonical_json_bytes(_receipt_value(_validate_receipt_shape(receipt)))


def audited_model_call_receipt_v2_from_bytes(
    value: bytes,
) -> AuditedModelCallReceiptV2:
    """Strictly load one canonical audited call receipt."""

    reason = "receipt_mismatch"
    decoded = _parse_canonical_json_object(
        value,
        max_bytes=16_384,
        reason=reason,
    )
    _require_json_keys(
        decoded,
        frozenset(
            {
                "kind",
                "schema_version",
                "manifest_sha256",
                "run_envelope_sha256",
                "slot_index",
                "case_index",
                "arm",
                "request_id",
                "generation_trace_sha256",
                "generation_response_evidence_sha256",
                "generation_response_evidence_byte_count",
                "decoded_response_utf8_sha256",
                "decoded_response_utf8_bytes",
            }
        ),
        reason=reason,
    )
    if decoded["kind"] != _RECEIPT_KIND:
        raise InfBridgeModelAdapterError(reason)
    arguments = dict(decoded)
    del arguments["kind"]
    receipt = AuditedModelCallReceiptV2(
        **arguments,  # type: ignore[arg-type]
    )
    _validate_receipt_shape(receipt)
    if audited_model_call_receipt_v2_bytes(receipt) != value:
        raise InfBridgeModelAdapterError(reason)
    return receipt


def audited_model_call_receipt_v2_sha256(
    receipt: AuditedModelCallReceiptV2,
) -> str:
    return hashlib.sha256(audited_model_call_receipt_v2_bytes(receipt)).hexdigest()


def audited_model_call_execution_v2_from_artifacts(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    *,
    receipt_bytes: bytes,
    trace_bytes: bytes,
    response_evidence_bytes: bytes,
    decoded_response_utf8: bytes,
    backend: SGLangBridgeBackend,
) -> AuditedModelCallExecutionV2:
    """Strictly load and replay-audit one persisted successful call."""

    try:
        receipt = audited_model_call_receipt_v2_from_bytes(receipt_bytes)
        trace = generation_physical_trace_from_bytes(trace_bytes)
        response_evidence = generation_response_evidence_from_bytes(
            response_evidence_bytes
        )
        if type(decoded_response_utf8) is not bytes:
            raise TypeError("decoded response must be exact bytes")
        response = decoded_response_utf8.decode("utf-8", errors="strict")
    except InfBridgeModelAdapterError:
        raise
    except Exception as error:
        raise InfBridgeModelAdapterError("receipt_mismatch") from error
    prepared_call, slot_index = _manifest_call(
        manifest,
        case_index=receipt.case_index,
        arm=receipt.arm,
    )
    if slot_index != receipt.slot_index:
        raise InfBridgeModelAdapterError("receipt_mismatch")
    execution = AuditedModelCallExecutionV2(
        response=response,
        trace=trace,
        response_evidence=response_evidence,
        receipt=receipt,
        legacy_execution=_expected_legacy_execution(prepared_call, response),
    )
    validate_infbridge_model_call_v2(
        manifest,
        tokenizer,
        envelope,
        execution,
        backend=backend,
    )
    return execution


def _validate_execution_for_plan(
    *,
    prepared_call: helpfulness.PreparedModelCall,
    plan: CallPlanV2,
    envelope: RunEnvelopeV2,
    execution: AuditedModelCallExecutionV2,
    tokenizer: AuditedDecoderTokenizer,
    envelope_sha256: str,
) -> None:
    if type(execution) is not AuditedModelCallExecutionV2:
        raise InfBridgeModelAdapterError("receipt_mismatch")
    if (
        type(execution.response) is not str
        or type(execution.trace) is not GenerationPhysicalTrace
        or type(execution.response_evidence) is not GenerationResponseEvidence
        or not _is_sha256(envelope_sha256)
    ):
        raise InfBridgeModelAdapterError("receipt_mismatch")
    _replay_trace_requests(prepared_call, plan, envelope, execution.trace)
    response_evidence_bytes = _replay_response_evidence(
        execution.trace,
        execution.response_evidence,
        envelope.evidence_policy,
    )
    response, response_bytes = _decode_response(
        tokenizer,
        execution.trace,
        envelope.decoding,
    )
    expected_legacy = _expected_legacy_execution(prepared_call, response)
    expected_receipt = AuditedModelCallReceiptV2(
        schema_version=2,
        manifest_sha256=envelope.manifest_sha256,
        run_envelope_sha256=envelope_sha256,
        slot_index=plan.slot_index,
        case_index=plan.case_index,
        arm=plan.arm,
        request_id=plan.request_id,
        generation_trace_sha256=generation_physical_trace_sha256(execution.trace),
        generation_response_evidence_sha256=(
            generation_response_evidence_sha256(execution.response_evidence)
        ),
        generation_response_evidence_byte_count=len(response_evidence_bytes),
        decoded_response_utf8_sha256=hashlib.sha256(response_bytes).hexdigest(),
        decoded_response_utf8_bytes=len(response_bytes),
    )
    if (
        execution.response != response
        or not _exact_tree_equal(execution.receipt, expected_receipt)
        or not _exact_tree_equal(execution.legacy_execution, expected_legacy)
    ):
        raise InfBridgeModelAdapterError("receipt_mismatch")
    audited_model_call_receipt_v2_bytes(execution.receipt)


def validate_infbridge_model_call_v2(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    execution: AuditedModelCallExecutionV2,
    *,
    backend: SGLangBridgeBackend,
) -> None:
    """Replay one persisted client-local call against its frozen run envelope."""

    if type(backend) is not SGLangBridgeBackend:
        raise InfBridgeModelAdapterError("unsupported_backend")
    _validate_envelope_against_manifest(manifest, tokenizer, envelope)
    if type(execution) is not AuditedModelCallExecutionV2:
        raise InfBridgeModelAdapterError("receipt_mismatch")
    receipt = _validate_receipt_shape(execution.receipt)
    prepared_call, slot_index = _manifest_call(
        manifest,
        case_index=receipt.case_index,
        arm=receipt.arm,
    )
    if slot_index != receipt.slot_index:
        raise InfBridgeModelAdapterError("receipt_mismatch")
    plan = envelope.call_plans[slot_index]
    _validate_execution_for_plan(
        prepared_call=prepared_call,
        plan=plan,
        envelope=envelope,
        execution=execution,
        tokenizer=tokenizer,
        envelope_sha256=infbridge_run_envelope_v2_sha256(envelope),
    )


class InfBridgeModelAdapter:
    """Audit one preregistered SGLang slot at a time.

    This low-level adapter does not establish run ordering, uniqueness, or
    completeness.  Formal experiments must use the durable run ledger rather
    than exposing :meth:`submit` to an experiment caller.
    """

    def __init__(
        self,
        manifest: helpfulness.ModelRunManifest,
        tokenizer: AuditedDecoderTokenizer,
        envelope: RunEnvelopeV2,
        bridge: InfBridge,
    ) -> None:
        _validate_envelope_against_manifest(manifest, tokenizer, envelope)
        _assert_bridge_matches_runtime(bridge, envelope.runtime)
        self._manifest = manifest
        self._tokenizer = tokenizer
        self._envelope = envelope
        self._bridge = bridge
        self._pause_state = bridge.pause_state
        self._envelope_sha256 = infbridge_run_envelope_v2_sha256(envelope)

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        """Execute and immediately replay-audit one frozen call slot."""

        _assert_bridge_matches_runtime(
            self._bridge,
            self._envelope.runtime,
            pause_state=self._pause_state,
        )
        _assert_decoder_matches(self._tokenizer, self._envelope.decoding)
        prepared_call, slot_index = _manifest_call(
            self._manifest,
            case_index=case_index,
            arm=arm,
        )
        plan = self._envelope.call_plans[slot_index]
        request = _make_model_request(
            prepared_call,
            plan,
            self._envelope.decoding,
        )
        try:
            output = await self._bridge.agenerate_with_trace_and_response_evidence(
                request
            )
        except GenerationTraceValidationError as error:
            raise InfBridgeModelAdapterError("trace_mismatch") from error
        except Exception as error:
            raise InfBridgeModelAdapterError("generation_failure") from error
        if type(output) is not tuple or len(output) != 3:
            raise InfBridgeModelAdapterError("trace_mismatch")
        response, trace, response_evidence = output
        if (
            type(response) is not ModelResponse
            or type(trace) is not GenerationPhysicalTrace
            or type(response_evidence) is not GenerationResponseEvidence
        ):
            raise InfBridgeModelAdapterError("trace_mismatch")
        try:
            validate_generation_physical_trace_response(response, trace)
        except Exception as error:
            raise InfBridgeModelAdapterError("trace_mismatch") from error
        _assert_bridge_matches_runtime(
            self._bridge,
            self._envelope.runtime,
            pause_state=self._pause_state,
        )
        _replay_trace_requests(prepared_call, plan, self._envelope, trace)
        encoded_evidence = _replay_response_evidence(
            trace,
            response_evidence,
            self._envelope.evidence_policy,
        )
        decoded, decoded_bytes = _decode_response(
            self._tokenizer,
            trace,
            self._envelope.decoding,
        )
        receipt = AuditedModelCallReceiptV2(
            schema_version=2,
            manifest_sha256=self._envelope.manifest_sha256,
            run_envelope_sha256=self._envelope_sha256,
            slot_index=plan.slot_index,
            case_index=plan.case_index,
            arm=plan.arm,
            request_id=plan.request_id,
            generation_trace_sha256=generation_physical_trace_sha256(trace),
            generation_response_evidence_sha256=(
                generation_response_evidence_sha256(response_evidence)
            ),
            generation_response_evidence_byte_count=len(encoded_evidence),
            decoded_response_utf8_sha256=hashlib.sha256(decoded_bytes).hexdigest(),
            decoded_response_utf8_bytes=len(decoded_bytes),
        )
        execution = AuditedModelCallExecutionV2(
            response=decoded,
            trace=trace,
            response_evidence=response_evidence,
            receipt=receipt,
            legacy_execution=_expected_legacy_execution(prepared_call, decoded),
        )
        _validate_execution_for_plan(
            prepared_call=prepared_call,
            plan=plan,
            envelope=self._envelope,
            execution=execution,
            tokenizer=self._tokenizer,
            envelope_sha256=self._envelope_sha256,
        )
        return execution
