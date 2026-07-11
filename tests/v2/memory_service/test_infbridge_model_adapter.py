"""Contract tests for the strict Memory helpfulness InfBridge adapter.

These tests deliberately use the real ``InfBridge`` generation loop and the
exact built-in SGLang bridge.  Only the HTTP transport is replaced, so the
evidence under test is produced at the same client boundary as a real run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedModelCallExecutionV2,
    AuditedModelCallReceiptV2,
    CallPlanV2,
    DecoderAuditMaterialV2,
    EvidencePolicyV2,
    InfBridgeModelAdapter,
    InfBridgeModelAdapterError,
    ModelAdapterDecodingV2,
    RunEnvelopeV2,
    RuntimeConfigV2,
    audited_model_call_execution_v2_from_artifacts,
    audited_model_call_receipt_v2_bytes,
    audited_model_call_receipt_v2_from_bytes,
    audited_model_call_receipt_v2_sha256,
    infbridge_run_envelope_v2_bytes,
    infbridge_run_envelope_v2_from_bytes,
    infbridge_run_envelope_v2_sha256,
    prepare_infbridge_run_envelope_v2,
    validate_infbridge_model_call_v2,
)

from areal.v2.inference_service.client_trace import (
    GenerationPhysicalTrace,
    GenerationResponseEvidence,
    generation_physical_trace_bytes,
    generation_physical_trace_sha256,
    generation_response_evidence_bytes,
    generation_response_evidence_sha256,
    generation_response_evidence_values,
    prepared_request_json_sha256,
)
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend

_CANONICAL_RUNTIME = sys.implementation.cache_tag == "cpython-312" and sys.version_info[
    :3
] == (3, 12, 13)


class _ByteTokenizer:
    """Deterministic tokenizer with an exact reversible output-token surface."""

    _ARTIFACT_BYTES = b'bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'bytes-tokenizer-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="byte-tokenizer-v2",
            tokenizer_artifact_bytes=self._ARTIFACT_BYTES,
            decoder_state_bytes=self._DECODER_STATE_BYTES,
        )

    def encode(
        self,
        value: bytes,
        *,
        add_special_tokens: bool,
    ) -> tuple[int, ...]:
        if not value and not add_special_tokens:
            return ()
        marker = 257 if add_special_tokens else 256
        return (marker, *value)

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if type(token_ids) is not tuple or any(
            type(token_id) is not int or token_id not in range(256)
            for token_id in token_ids
        ):
            raise TypeError("decode expects a tuple of byte token IDs")
        if skip_special_tokens is not True:
            raise ValueError("the preregistered adapter must skip special tokens")
        if clean_up_tokenization_spaces is not False:
            raise ValueError("the preregistered adapter must disable cleanup")
        return bytes(token_ids).decode("utf-8")


class _SpoofedDecodeCallable:
    """Callable that lies about the function whose code it executes."""

    __func__ = _ByteTokenizer.decode

    def __call__(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        return "DRIFT"


def _sglang_response(
    output: bytes,
    stop_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "meta_info": {
            "finish_reason": {"type": stop_reason},
            "output_token_logprobs": [
                (-0.01 * (index + 1), token_id) for index, token_id in enumerate(output)
            ],
        }
    }


def _json_semantic(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def _make_bridge(
    *,
    backend_addr: str = "http://memory-model.test/",
    max_resubmit_retries: int = 3,
    version: int = 17,
) -> InfBridge:
    return InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=backend_addr,
        pause_state=PauseState(),
        request_timeout=9.0,
        max_resubmit_retries=max_resubmit_retries,
        resubmit_wait=0.0,
        version=version,
    )


@pytest.fixture(scope="module")
def tokenizer() -> _ByteTokenizer:
    return _ByteTokenizer()


@pytest.fixture(scope="module")
def manifest(tokenizer: _ByteTokenizer) -> helpfulness.ModelRunManifest:
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="a" * 40,
        evaluator_commit_sha="b" * 40,
        model_id="audited-local-model",
        model_weights_sha256="c" * 64,
        tokenizer_id="byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(_ByteTokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    return prepared.manifest


def _first_arm(manifest: helpfulness.ModelRunManifest) -> str:
    return manifest.cases[0].arm_calls[0].arm


def _assert_reason(
    error: pytest.ExceptionInfo[InfBridgeModelAdapterError], reason: str
):
    assert type(error.value) is InfBridgeModelAdapterError
    assert error.value.reason == reason
    assert str(error.value) == reason


@pytest.mark.asyncio
async def test_prepare_freezes_all_384_calls_before_any_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    send = AsyncMock(side_effect=AssertionError("preparation must not send"))
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=32,
        )

        assert type(envelope) is RunEnvelopeV2
        assert envelope.schema_version == 2
        assert envelope.adapter_algorithm == "areal-memory-infbridge-adapter-v2"
        assert envelope.execution_mode == "single-call-evidence-only"
        assert envelope.manifest_sha256 == (
            "c00b7a528bdfdd0e3cbff5cb1ae8b539ebd4c8411f31a414a5309865c7042e65"
        )
        assert type(envelope.decoding) is ModelAdapterDecodingV2
        assert envelope.decoding.mode == "greedy"
        assert envelope.decoding.n_samples == 1
        assert envelope.decoding.temperature == "0"
        assert envelope.decoding.top_p == "1"
        assert envelope.decoding.top_k == 100_000_000
        assert envelope.decoding.max_new_tokens == 32
        assert envelope.decoding.stop_token_ids == ()
        assert envelope.decoding.ignore_eos is False
        assert envelope.decoding.skip_special_tokens is True
        assert envelope.decoding.stop_sequences == ()
        assert envelope.decoding.frequency_penalty == "0"
        assert envelope.decoding.use_beam_search is False
        assert envelope.decoding.with_lora is False
        assert envelope.decoding.decode_policy == (
            "tokenizer-decode-skip-special-no-cleanup-v2"
        )
        assert envelope.decoding.decoder_kind == (
            "tests.v2.memory_service.test_infbridge_model_adapter._ByteTokenizer"
        )
        assert envelope.decoding.tokenizer_id == "byte-tokenizer-v2"
        assert envelope.decoding.tokenizer_artifact_sha256 == (
            "de09141b9c53f64b44289828372aa31701d5a3fad071c0558f1705b7f4b6dd06"
        )
        assert envelope.decoding.decoder_state_sha256 == (
            "db0f1583c3c81e469bd3d988582a9e455a569ca141fd1cc738e1ef3b87a83814"
        )
        assert envelope.decoding.audit_callable_source_sha256 == (
            "7720ac9939941e9c22a48b119bbeb63668fc02f67c4d1a2a4315d83deb75ea1e"
        )
        assert len(envelope.decoding.audit_callable_runtime_sha256) == 64
        assert envelope.decoding.decoder_callable_source_sha256 == (
            "92fe3a35abd58457cd6cb8ffa5f03b5d0cc7e4f3766387d9b04e265de81a7bf9"
        )
        assert len(envelope.decoding.decoder_callable_runtime_sha256) == 64
        if _CANONICAL_RUNTIME:
            assert envelope.decoding.audit_callable_runtime_sha256 == (
                "db8b1a9c65a51bfcd0470ad15335c9d42d9a7b16c8a176d9d92c890d75d1e4e9"
            )
            assert envelope.decoding.decoder_callable_runtime_sha256 == (
                "e09e46dbfbd4d733a893f411ce8a241038df4fc2e9ad0e5fb2b9446c7e3978a1"
            )
        assert envelope.decoding.python_cache_tag == sys.implementation.cache_tag
        assert envelope.decoding.python_version == ".".join(
            str(item) for item in sys.version_info[:3]
        )
        assert type(envelope.evidence_policy) is EvidencePolicyV2
        assert envelope.evidence_policy.schema_version == 1
        assert envelope.evidence_policy.require_response_preimages is True
        assert envelope.evidence_policy.max_response_json_bytes_per_attempt == 1_048_576
        assert (
            envelope.evidence_policy.max_response_evidence_bytes_per_call == 8_388_608
        )
        assert type(envelope.runtime) is RuntimeConfigV2
        assert envelope.runtime.backend_kind == "SGLangBridgeBackend"
        assert (
            envelope.runtime.backend_addr_sha256
            == hashlib.sha256(b"http://memory-model.test").hexdigest()
        )
        assert envelope.runtime.configured_attempt_limit == 3
        assert envelope.runtime.request_timeout_hex == "0x1.2000000000000p+3"
        assert envelope.runtime.resubmit_wait_hex == "0x0.0p+0"
        assert envelope.runtime.pause_state_kind == (
            "areal.v2.inference_service.data_proxy.pause.PauseState"
        )
        assert envelope.runtime.expected_client_version == 17
        assert envelope.runtime.expected_client_version_epoch == 0
        assert len(envelope.runtime_config_sha256) == 64
        if _CANONICAL_RUNTIME:
            assert envelope.runtime_config_sha256 == (
                "0da3f1ad0b8bfb5af936111782ea6565f0becdc847d94e0b0fcd9194c1388929"
            )
        assert type(envelope.call_plans) is tuple
        assert len(envelope.call_plans) == manifest.call_count == 384
        assert all(type(plan) is CallPlanV2 for plan in envelope.call_plans)
        assert tuple(
            (plan.slot_index, plan.case_index, plan.arm) for plan in envelope.call_plans
        ) == tuple(
            (
                case_index * len(helpfulness.MODEL_ARMS) + arm_offset,
                case_index,
                arm_call.arm,
            )
            for case_index, registration in enumerate(manifest.cases)
            for arm_offset, arm_call in enumerate(registration.arm_calls)
        )
        assert len({plan.request_id for plan in envelope.call_plans}) == 384
        for plan, (_, _, prepared_call) in zip(
            envelope.call_plans,
            (
                (case_index, arm_call.arm, arm_call.prepared_call)
                for case_index, registration in enumerate(manifest.cases)
                for arm_call in registration.arm_calls
            ),
            strict=True,
        ):
            canonical_token_ids = json.dumps(
                list(prepared_call.input_token_ids),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            assert (
                plan.input_token_ids_sha256
                == hashlib.sha256(
                    b"areal-memory-token-ids-v2\0" + canonical_token_ids
                ).hexdigest()
            )
            assert plan.input_token_count == len(prepared_call.input_token_ids)
            assert plan.expected_endpoint == "/generate"
            assert plan.expected_method == "POST"
            assert len(plan.first_prepared_request_json_sha256) == 64
        assert send.await_count == 0
        assert infbridge_run_envelope_v2_bytes(envelope)
        assert len(infbridge_run_envelope_v2_sha256(envelope)) == 64
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_envelope_is_deterministic_canonical_ascii(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    first_bridge = _make_bridge(backend_addr="http://memory-model.test/")
    second_bridge = _make_bridge(backend_addr="http://memory-model.test")
    try:
        first = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            first_bridge,
            max_new_tokens=32,
        )
        second = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            second_bridge,
            max_new_tokens=32,
        )

        assert first == second
        assert tuple(plan.request_id for plan in first.call_plans) == tuple(
            plan.request_id for plan in second.call_plans
        )
        encoded = infbridge_run_envelope_v2_bytes(first)
        value = json.loads(encoded)
        assert encoded == json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        assert infbridge_run_envelope_v2_from_bytes(encoded) == first
        assert set(value) == {
            "adapter_algorithm",
            "call_plans",
            "decoding",
            "evidence_policy",
            "execution_mode",
            "kind",
            "manifest_sha256",
            "runtime",
            "runtime_config_sha256",
            "schema_version",
        }
        assert value["kind"] == "areal-memory-infbridge-run-envelope-v2"
        assert set(value["decoding"]) == {
            "decoder_kind",
            "audit_callable_source_sha256",
            "audit_callable_runtime_sha256",
            "decoder_callable_source_sha256",
            "decoder_callable_runtime_sha256",
            "decoder_state_sha256",
            "frequency_penalty",
            "ignore_eos",
            "max_new_tokens",
            "mode",
            "n_samples",
            "python_cache_tag",
            "python_version",
            "decode_policy",
            "skip_special_tokens",
            "stop_sequences",
            "stop_token_ids",
            "temperature",
            "tokenizer_artifact_sha256",
            "tokenizer_id",
            "top_k",
            "top_p",
            "use_beam_search",
            "with_lora",
        }
        assert set(value["evidence_policy"]) == {
            "max_response_evidence_bytes_per_call",
            "max_response_json_bytes_per_attempt",
            "require_response_preimages",
            "schema_version",
        }
        assert set(value["runtime"]) == {
            "backend_addr_sha256",
            "backend_kind",
            "configured_attempt_limit",
            "expected_client_version",
            "expected_client_version_epoch",
            "pause_state_kind",
            "request_timeout_hex",
            "resubmit_wait_hex",
        }
        assert set(value["call_plans"][0]) == {
            "arm",
            "case_index",
            "expected_endpoint",
            "expected_method",
            "first_prepared_request_json_sha256",
            "input_token_count",
            "input_token_ids_sha256",
            "request_id",
            "slot_index",
        }
        if _CANONICAL_RUNTIME:
            assert len(encoded) == 166_306
        else:
            assert len(encoded) > 160_000
        assert first.call_plans[0].input_token_ids_sha256 == (
            "2f037d7e4cbb1146053aeba4cf198a0103e1bc0b60654eac4b15bc005ebfb606"
        )
        assert first.call_plans[0].first_prepared_request_json_sha256 == (
            "2e8d4d8bb07b0bc3c63bf7c7396845d75a086f3580b105c3097c66543f6837bd"
        )
        if _CANONICAL_RUNTIME:
            assert first.call_plans[0].request_id == (
                "areal-memory-v2-"
                "d3164ab16575bff613c1dcb3b705669810d98959b8ce3e3e66428266156f1301"
            )
            assert infbridge_run_envelope_v2_sha256(first) == (
                "615e6b1d29c8ba8255efe851aff736a438e185ee95de22d2496be8ca481eba45"
            )
        else:
            assert first.call_plans[0].request_id.startswith("areal-memory-v2-")
            assert len(first.call_plans[0].request_id) == 80
            assert len(infbridge_run_envelope_v2_sha256(first)) == 64
        assert (
            infbridge_run_envelope_v2_sha256(first)
            == hashlib.sha256(encoded).hexdigest()
        )
    finally:
        await first_bridge.aclose()
        await second_bridge.aclose()


@pytest.mark.asyncio
async def test_envelope_loader_rejects_noncanonical_and_closed_schema_values(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=32,
        )
        encoded = infbridge_run_envelope_v2_bytes(envelope)
        unknown_field = json.loads(encoded)
        unknown_field["unknown"] = 1
        wrong_nested_type = json.loads(encoded)
        wrong_nested_type["decoding"]["stop_token_ids"] = {}
        variants = (
            b" " + encoded,
            b'{"adapter_algorithm":"duplicate",' + encoded[1:],
            json.dumps(
                unknown_field,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
            json.dumps(
                wrong_nested_type,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
        )
        for variant in variants:
            with pytest.raises(InfBridgeModelAdapterError) as error:
                infbridge_run_envelope_v2_from_bytes(variant)
            _assert_reason(error, "run_envelope")
    finally:
        await bridge.aclose()


def test_receipt_loader_rejects_noncanonical_and_closed_schema_values() -> None:
    receipt = AuditedModelCallReceiptV2(
        schema_version=2,
        manifest_sha256="a" * 64,
        run_envelope_sha256="b" * 64,
        slot_index=0,
        case_index=0,
        arm=helpfulness.MODEL_ARMS[0],
        request_id="request-id",
        generation_trace_sha256="c" * 64,
        generation_response_evidence_sha256="d" * 64,
        generation_response_evidence_byte_count=1,
        decoded_response_utf8_sha256="e" * 64,
        decoded_response_utf8_bytes=0,
    )
    encoded = audited_model_call_receipt_v2_bytes(receipt)
    assert audited_model_call_receipt_v2_from_bytes(encoded) == receipt
    unknown_field = json.loads(encoded)
    unknown_field["unknown"] = 1
    variants = (
        encoded + b"\n",
        b'{"arm":"duplicate",' + encoded[1:],
        json.dumps(
            unknown_field,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii"),
    )
    for variant in variants:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            audited_model_call_receipt_v2_from_bytes(variant)
        _assert_reason(error, "receipt_mismatch")


@pytest.mark.asyncio
async def test_normal_stop_returns_a_valid_audited_execution_and_v1_projection(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    sent_payloads: list[dict[str, Any]] = []
    raw_response = _sglang_response(b"ANSWER", "stop")

    async def send(http_request, **kwargs):
        sent_payloads.append(deepcopy(http_request.payload))
        return deepcopy(raw_response)

    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=32,
        )
        adapter = InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
        arm = _first_arm(manifest)

        execution = await adapter.submit(0, arm)

        assert type(execution) is AuditedModelCallExecutionV2
        assert execution.response == "ANSWER"
        assert type(execution.receipt) is AuditedModelCallReceiptV2
        assert execution.receipt.schema_version == 2
        assert execution.receipt.manifest_sha256 == envelope.manifest_sha256
        assert execution.receipt.run_envelope_sha256 == (
            infbridge_run_envelope_v2_sha256(envelope)
        )
        assert execution.receipt.slot_index == 0
        assert execution.receipt.case_index == 0
        assert execution.receipt.arm == arm
        assert execution.receipt.request_id == envelope.call_plans[0].request_id
        assert execution.receipt.generation_trace_sha256 == (
            generation_physical_trace_sha256(execution.trace)
        )
        assert type(execution.response_evidence) is GenerationResponseEvidence
        response_evidence_bytes = generation_response_evidence_bytes(
            execution.response_evidence
        )
        assert execution.receipt.generation_response_evidence_sha256 == (
            generation_response_evidence_sha256(execution.response_evidence)
        )
        assert execution.receipt.generation_response_evidence_byte_count == len(
            response_evidence_bytes
        )
        assert generation_response_evidence_values(
            execution.trace,
            execution.response_evidence,
        ) == (_json_semantic(raw_response),)
        assert (
            execution.receipt.decoded_response_utf8_sha256
            == hashlib.sha256(b"ANSWER").hexdigest()
        )
        receipt_bytes = audited_model_call_receipt_v2_bytes(execution.receipt)
        assert (
            audited_model_call_receipt_v2_from_bytes(receipt_bytes) == execution.receipt
        )
        assert set(json.loads(receipt_bytes)) == {
            "arm",
            "case_index",
            "decoded_response_utf8_bytes",
            "decoded_response_utf8_sha256",
            "generation_response_evidence_byte_count",
            "generation_response_evidence_sha256",
            "generation_trace_sha256",
            "kind",
            "manifest_sha256",
            "request_id",
            "run_envelope_sha256",
            "schema_version",
            "slot_index",
        }
        assert len(receipt_bytes) == 768
        if _CANONICAL_RUNTIME:
            assert audited_model_call_receipt_v2_sha256(execution.receipt) == (
                "835b3bf58c1b4937c3cb42bd33719dee9b11d5e0d01488d549bf54e448fe0c58"
            )
        else:
            assert len(audited_model_call_receipt_v2_sha256(execution.receipt)) == 64
        assert execution.trace.request_id == envelope.call_plans[0].request_id
        assert execution.trace.request_input_token_ids == (
            manifest.cases[0].arm_calls[0].prepared_call.input_token_ids
        )
        assert execution.trace.final_output_token_ids == tuple(b"ANSWER")
        assert execution.trace.final_stop_reason == "stop"
        assert execution.trace.terminal_reason == "backend_stop"
        assert len(sent_payloads) == 1
        assert sent_payloads[0] == {
            "image_data": [],
            "input_ids": list(
                manifest.cases[0].arm_calls[0].prepared_call.input_token_ids
            ),
            "return_logprob": True,
            "sampling_params": {
                "frequency_penalty": 0.0,
                "ignore_eos": False,
                "max_new_tokens": 32,
                "skip_special_tokens": True,
                "stop_token_ids": [],
                "temperature": 0.0,
                "top_k": 100_000_000,
                "top_p": 1.0,
            },
            "stream": False,
        }
        reloaded = audited_model_call_execution_v2_from_artifacts(
            manifest,
            tokenizer,
            envelope,
            receipt_bytes=receipt_bytes,
            trace_bytes=generation_physical_trace_bytes(execution.trace),
            response_evidence_bytes=response_evidence_bytes,
            decoded_response_utf8=b"ANSWER",
            backend=SGLangBridgeBackend(),
        )
        assert reloaded == execution
        with pytest.raises(InfBridgeModelAdapterError) as error:
            audited_model_call_execution_v2_from_artifacts(
                manifest,
                tokenizer,
                envelope,
                receipt_bytes=receipt_bytes,
                trace_bytes=generation_physical_trace_bytes(execution.trace),
                response_evidence_bytes=response_evidence_bytes,
                decoded_response_utf8=b"DRIFT",
                backend=SGLangBridgeBackend(),
            )
        _assert_reason(error, "receipt_mismatch")
        assert envelope.call_plans[0].first_prepared_request_json_sha256 == (
            prepared_request_json_sha256(sent_payloads[0])
        )

        legacy = execution.legacy_execution
        prepared = manifest.cases[0].arm_calls[0].prepared_call
        assert type(legacy) is helpfulness.ModelCallExecution
        assert legacy.response == "ANSWER"
        assert legacy.consumer_input_receipt == prepared.consumer_input_receipt
        assert legacy.model_call_receipt == prepared.expected_receipt
        assert (
            legacy.rendered_context_token_count == prepared.rendered_context_token_count
        )
        assert legacy.valid is True
        assert legacy.invalid_reason is None
        assert (
            validate_infbridge_model_call_v2(
                manifest,
                tokenizer,
                envelope,
                execution,
                backend=bridge.backend,
            )
            is None
        )
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_abort_resubmit_binds_every_prefix_budget_and_request_hash(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    sent_payloads: list[dict[str, Any]] = []
    raw_responses = (
        _sglang_response(b"AB", "abort"),
        _sglang_response(b"C", "stop"),
    )

    async def send(http_request, **kwargs):
        sent_payloads.append(deepcopy(http_request.payload))
        return deepcopy(raw_responses[len(sent_payloads) - 1])

    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=8,
        )
        execution = await InfBridgeModelAdapter(
            manifest,
            tokenizer,
            envelope,
            bridge,
        ).submit(0, _first_arm(manifest))
        original_ids = manifest.cases[0].arm_calls[0].prepared_call.input_token_ids

        assert execution.response == "ABC"
        assert len(execution.trace.attempts) == 2
        assert tuple(
            attempt.submitted_input_token_ids for attempt in execution.trace.attempts
        ) == (original_ids, (*original_ids, *b"AB"))
        assert tuple(
            attempt.remaining_new_tokens for attempt in execution.trace.attempts
        ) == (8, 6)
        assert tuple(
            attempt.raw_stop_reason for attempt in execution.trace.attempts
        ) == ("abort", "stop")
        assert tuple(
            attempt.prepared_request_json_sha256 for attempt in execution.trace.attempts
        ) == tuple(prepared_request_json_sha256(value) for value in sent_payloads)
        assert envelope.call_plans[0].first_prepared_request_json_sha256 == (
            execution.trace.attempts[0].prepared_request_json_sha256
        )
        assert generation_response_evidence_values(
            execution.trace,
            execution.response_evidence,
        ) == tuple(_json_semantic(value) for value in raw_responses)
        assert execution.receipt.generation_response_evidence_sha256 == (
            generation_response_evidence_sha256(execution.response_evidence)
        )
        assert execution.receipt.generation_response_evidence_byte_count == len(
            generation_response_evidence_bytes(execution.response_evidence)
        )
        validate_infbridge_model_call_v2(
            manifest,
            tokenizer,
            envelope,
            execution,
            backend=bridge.backend,
        )
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("backend", "unsupported_backend"),
        ("address", "runtime_mismatch"),
        ("attempt_limit", "runtime_mismatch"),
        ("version", "runtime_mismatch"),
    ),
)
async def test_runtime_changes_are_rejected_before_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    mutation: str,
    reason: str,
) -> None:
    bridge = _make_bridge()
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    if mutation == "backend":
        bridge.backend = VLLMBridgeBackend()
    elif mutation == "address":
        bridge.backend_addr = "http://different-model.test"
    elif mutation == "attempt_limit":
        bridge.max_resubmit_retries += 1
    else:
        bridge.set_version(bridge.get_version() + 1)
    send = AsyncMock(side_effect=AssertionError("mismatch must fail before transport"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, reason)
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_version_change_during_transport_is_rejected(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge(version=17)
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )

    async def send(http_request, **kwargs):
        bridge.set_version(18)
        return _sglang_response(b"ANSWER", "stop")

    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "runtime_mismatch")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_aba_version_change_during_transport_is_rejected_by_epoch(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge(version=17)
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )

    async def send(http_request, **kwargs):
        bridge.set_version(18)
        bridge.set_version(17)
        return _sglang_response(b"ANSWER", "stop")

    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "runtime_mismatch")
        assert bridge.get_version() == 17
        assert bridge.get_version_epoch() == 2
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribute", "changed_value"),
    (
        ("request_timeout", 10.0),
        ("resubmit_wait", 0.25),
    ),
)
async def test_transport_runtime_drift_is_rejected_before_send(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    attribute: str,
    changed_value: float,
) -> None:
    bridge = _make_bridge()
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    adapter = InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    setattr(bridge, attribute, changed_value)
    send = AsyncMock(return_value=_sglang_response(b"ANSWER", "stop"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await adapter.submit(0, _first_arm(manifest))

        _assert_reason(error, "runtime_mismatch")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_pause_state_replacement_is_rejected_before_send(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    adapter = InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    bridge.pause_state = PauseState()
    send = AsyncMock(return_value=_sglang_response(b"ANSWER", "stop"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await adapter.submit(0, _first_arm(manifest))

        _assert_reason(error, "runtime_mismatch")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("per_attempt_limit", "per_call_limit"),
    ((32, 8_388_608), (256, 256)),
    ids=("attempt-preimage", "call-bundle"),
)
async def test_response_evidence_limits_are_frozen_and_fail_closed(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    per_attempt_limit: int,
    per_call_limit: int,
) -> None:
    bridge = _make_bridge()
    send = AsyncMock(return_value=_sglang_response(b"ANSWER", "stop"))
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=16,
            max_response_json_bytes_per_attempt=per_attempt_limit,
            max_response_evidence_bytes_per_call=per_call_limit,
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "response_evidence")
        assert send.await_count == 1
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        "decode_callable",
        "spoofed_func",
        "runtime_code",
        "audit_callable",
        "audit_runtime_code",
        "decoder_state",
        "tokenizer_artifact",
    ),
)
async def test_decoder_mutation_is_rejected_before_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    mutation: str,
) -> None:
    bridge = _make_bridge()
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    adapter = InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    send = AsyncMock(return_value=_sglang_response(b"ANSWER", "stop"))
    bridge._send_request = send
    if mutation == "decode_callable":

        def drift_decode(
            token_ids: tuple[int, ...],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            return "DRIFT"

        tokenizer.decode = drift_decode  # type: ignore[method-assign]
    elif mutation == "spoofed_func":
        tokenizer.decode = _SpoofedDecodeCallable()  # type: ignore[method-assign]
    elif mutation == "runtime_code":
        original_code = type(tokenizer).decode.__code__

        def drift_runtime_code(
            self,
            token_ids: tuple[int, ...],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            return "DRIFT"

        type(tokenizer).decode.__code__ = drift_runtime_code.__code__.replace(
            co_filename=original_code.co_filename,
            co_firstlineno=original_code.co_firstlineno,
            co_name=original_code.co_name,
            co_qualname=original_code.co_qualname,
        )
    elif mutation == "audit_callable":
        tokenizer.memory_audit_material = lambda: DecoderAuditMaterialV2(  # type: ignore[method-assign]
            tokenizer_id="byte-tokenizer-v2",
            tokenizer_artifact_bytes=tokenizer._ARTIFACT_BYTES,
            decoder_state_bytes=tokenizer._DECODER_STATE_BYTES,
        )
    elif mutation == "audit_runtime_code":
        original_audit_code = type(tokenizer).memory_audit_material.__code__

        def drift_audit_material(self) -> DecoderAuditMaterialV2:
            material = DecoderAuditMaterialV2(
                tokenizer_id="byte-tokenizer-v2",
                tokenizer_artifact_bytes=self._ARTIFACT_BYTES,
                decoder_state_bytes=self._DECODER_STATE_BYTES,
            )
            return material

        type(
            tokenizer
        ).memory_audit_material.__code__ = drift_audit_material.__code__.replace(
            co_filename=original_audit_code.co_filename,
            co_firstlineno=original_audit_code.co_firstlineno,
            co_name=original_audit_code.co_name,
            co_qualname=original_audit_code.co_qualname,
        )
    elif mutation == "decoder_state":
        tokenizer._DECODER_STATE_BYTES = b"mutated-decoder-state"
    else:
        tokenizer._ARTIFACT_BYTES = b"mutated-tokenizer-artifact"
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await adapter.submit(0, _first_arm(manifest))

        _assert_reason(error, "decoder_mismatch")
        assert send.await_count == 0
    finally:
        if mutation in ("decode_callable", "spoofed_func"):
            del tokenizer.decode
        elif mutation == "runtime_code":
            type(tokenizer).decode.__code__ = original_code
        elif mutation == "audit_callable":
            del tokenizer.memory_audit_material
        elif mutation == "audit_runtime_code":
            type(tokenizer).memory_audit_material.__code__ = original_audit_code
        elif mutation == "decoder_state":
            del tokenizer._DECODER_STATE_BYTES
        else:
            del tokenizer._ARTIFACT_BYTES
        await bridge.aclose()


@pytest.mark.asyncio
async def test_attempt_limit_is_not_accepted_as_a_model_execution(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge(max_resubmit_retries=2)
    send = AsyncMock(return_value=_sglang_response(b"A", "abort"))
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=16,
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "attempt_limit")
        assert send.await_count == 2
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_server_output_larger_than_remaining_budget_is_rejected(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    send = AsyncMock(return_value=_sglang_response(b"ABCDE", "stop"))
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=4,
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "trace_mismatch")
        assert send.await_count == 1
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("transport", "malformed_response"))
async def test_generation_failures_use_a_closed_adapter_reason(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    failure_kind: str,
) -> None:
    bridge = _make_bridge()
    send = (
        AsyncMock(side_effect=RuntimeError("transport failed"))
        if failure_kind == "transport"
        else AsyncMock(return_value={})
    )
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=4,
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "generation_failure")
        assert send.await_count == 1
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_generation_cancellation_is_not_swallowed(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    cancellation = asyncio.CancelledError("cancelled")
    bridge = _make_bridge()
    send = AsyncMock(side_effect=cancellation)
    bridge._send_request = send
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=4,
        )
        with pytest.raises(asyncio.CancelledError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        assert error.value is cancellation
        assert send.await_count == 1
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_changed_envelope_configuration_is_rejected_before_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    tampered = replace(
        envelope,
        runtime=replace(
            envelope.runtime,
            expected_client_version=envelope.runtime.expected_client_version + 1,
        ),
    )
    send = AsyncMock(side_effect=AssertionError("invalid envelope must not send"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                tampered,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "run_envelope")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_non_sglang_backend_is_rejected_during_preparation_without_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    bridge.backend = VLLMBridgeBackend()
    send = AsyncMock(side_effect=AssertionError("unsupported backend must not send"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            prepare_infbridge_run_envelope_v2(
                manifest,
                tokenizer,
                bridge,
                max_new_tokens=16,
            )

        _assert_reason(error, "unsupported_backend")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("huge_timeout", "surrogate_address"))
async def test_runtime_encoding_failures_use_closed_reason(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    mutation: str,
) -> None:
    bridge = _make_bridge()
    if mutation == "huge_timeout":
        bridge.request_timeout = 10**10_000
    else:
        bridge.backend_addr = "http://invalid-\ud800"
    send = AsyncMock(side_effect=AssertionError("invalid runtime must not send"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            prepare_infbridge_run_envelope_v2(
                manifest,
                tokenizer,
                bridge,
                max_new_tokens=16,
            )

        _assert_reason(error, "runtime_mismatch")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_nonfinite_or_overflowing_runtime_hex_is_rejected_canonically(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=16,
        )
        tampered = replace(
            envelope,
            runtime=replace(
                envelope.runtime,
                request_timeout_hex="0x1p+999999999",
            ),
        )

        with pytest.raises(InfBridgeModelAdapterError) as error:
            infbridge_run_envelope_v2_bytes(tampered)
        _assert_reason(error, "run_envelope")
    finally:
        await bridge.aclose()


async def _valid_execution(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    bridge: InfBridge,
) -> tuple[RunEnvelopeV2, AuditedModelCallExecutionV2, dict[str, Any]]:
    sent_payloads: list[dict[str, Any]] = []

    async def send(http_request, **kwargs):
        sent_payloads.append(deepcopy(http_request.payload))
        return _sglang_response(b"ANSWER", "stop")

    bridge._send_request = send
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=16,
    )
    execution = await InfBridgeModelAdapter(
        manifest,
        tokenizer,
        envelope,
        bridge,
    ).submit(0, _first_arm(manifest))
    assert len(sent_payloads) == 1
    return envelope, execution, sent_payloads[0]


def _rebind_execution_to_trace(
    execution: AuditedModelCallExecutionV2,
    trace: GenerationPhysicalTrace,
) -> AuditedModelCallExecutionV2:
    """Recompute every unkeyed digest while retaining response JSON preimages."""

    trace_sha256 = generation_physical_trace_sha256(trace)
    evidence = replace(
        execution.response_evidence,
        generation_trace_sha256=trace_sha256,
    )
    evidence_bytes = generation_response_evidence_bytes(evidence)
    return replace(
        execution,
        trace=trace,
        response_evidence=evidence,
        receipt=replace(
            execution.receipt,
            generation_trace_sha256=trace_sha256,
            generation_response_evidence_sha256=(
                generation_response_evidence_sha256(evidence)
            ),
            generation_response_evidence_byte_count=len(evidence_bytes),
        ),
    )


@pytest.mark.asyncio
async def test_jointly_forged_submitted_ids_and_trace_hash_are_still_rejected(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope, execution, sent_payload = await _valid_execution(
            manifest,
            tokenizer,
            bridge,
        )
        original_attempt = execution.trace.attempts[0]
        assert original_attempt.submitted_input_token_ids is not None
        forged_ids = (
            original_attempt.submitted_input_token_ids[0] + 1,
            *original_attempt.submitted_input_token_ids[1:],
        )
        forged_payload = deepcopy(sent_payload)
        forged_payload["input_ids"] = list(forged_ids)
        forged_attempt = replace(
            original_attempt,
            submitted_input_token_ids=forged_ids,
            prepared_request_json_sha256=prepared_request_json_sha256(forged_payload),
        )
        forged_trace = replace(execution.trace, attempts=(forged_attempt,))
        forged_execution = _rebind_execution_to_trace(execution, forged_trace)

        # The generic trace is internally self-consistent after the attacker
        # updates both values.  Only replaying the preregistered call plan can
        # expose that this was not the frozen model input.
        assert forged_execution.receipt.generation_trace_sha256 == (
            generation_physical_trace_sha256(forged_trace)
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            validate_infbridge_model_call_v2(
                manifest,
                tokenizer,
                envelope,
                forged_execution,
                backend=bridge.backend,
            )

        _assert_reason(error, "trace_mismatch")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_missing_submitted_token_evidence_is_rejected_even_with_new_trace_hash(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope, execution, _ = await _valid_execution(
            manifest,
            tokenizer,
            bridge,
        )
        missing_attempt = replace(
            execution.trace.attempts[0],
            submitted_input_token_ids=None,
        )
        missing_trace = replace(execution.trace, attempts=(missing_attempt,))
        missing_execution = _rebind_execution_to_trace(execution, missing_trace)

        with pytest.raises(InfBridgeModelAdapterError) as error:
            validate_infbridge_model_call_v2(
                manifest,
                tokenizer,
                envelope,
                missing_execution,
                backend=bridge.backend,
            )

        _assert_reason(error, "trace_mismatch")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_jointly_forged_decoded_response_and_receipt_are_rejected_by_redecode(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope, execution, _ = await _valid_execution(
            manifest,
            tokenizer,
            bridge,
        )
        forged_response = "FORGED"
        forged_execution = replace(
            execution,
            response=forged_response,
            receipt=replace(
                execution.receipt,
                decoded_response_utf8_sha256=hashlib.sha256(
                    forged_response.encode("utf-8")
                ).hexdigest(),
            ),
            legacy_execution=replace(
                execution.legacy_execution,
                response=forged_response,
            ),
        )

        with pytest.raises(InfBridgeModelAdapterError) as error:
            validate_infbridge_model_call_v2(
                manifest,
                tokenizer,
                envelope,
                forged_execution,
                backend=bridge.backend,
            )

        _assert_reason(error, "receipt_mismatch")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_offline_parser_replay_rejects_joint_output_and_receipt_forgery(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope, execution, _ = await _valid_execution(
            manifest,
            tokenizer,
            bridge,
        )
        forged_bytes = b"FORGED"
        forged_response = forged_bytes.decode("ascii")
        forged_attempt = replace(
            execution.trace.attempts[0],
            output_token_ids=tuple(forged_bytes),
            output_logprob_count=len(forged_bytes),
        )
        forged_trace = replace(
            execution.trace,
            attempts=(forged_attempt,),
            final_output_token_ids=tuple(forged_bytes),
        )
        rebound = _rebind_execution_to_trace(execution, forged_trace)
        forged_execution = replace(
            rebound,
            response=forged_response,
            receipt=replace(
                rebound.receipt,
                decoded_response_utf8_sha256=hashlib.sha256(forged_bytes).hexdigest(),
                decoded_response_utf8_bytes=len(forged_bytes),
            ),
            legacy_execution=replace(
                rebound.legacy_execution,
                response=forged_response,
            ),
        )

        # All editable hashes and projections have been recomputed, while the
        # canonical response JSON preimage still contains ANSWER.  Generic
        # sidecar/trace binding therefore passes; only replaying the fixed
        # SGLang parser can expose the changed output tokens.
        assert tuple(
            attempt.canonical_json_bytes
            for attempt in forged_execution.response_evidence.attempts
        ) == tuple(
            attempt.canonical_json_bytes
            for attempt in execution.response_evidence.attempts
        )
        replayed_json = generation_response_evidence_values(
            forged_execution.trace,
            forged_execution.response_evidence,
        )
        assert [
            pair[1] for pair in replayed_json[0]["meta_info"]["output_token_logprobs"]
        ] == list(b"ANSWER")
        assert forged_execution.receipt.generation_trace_sha256 == (
            generation_physical_trace_sha256(forged_trace)
        )
        assert forged_execution.receipt.generation_response_evidence_sha256 == (
            generation_response_evidence_sha256(forged_execution.response_evidence)
        )

        with pytest.raises(InfBridgeModelAdapterError) as error:
            validate_infbridge_model_call_v2(
                manifest,
                tokenizer,
                envelope,
                forged_execution,
                backend=bridge.backend,
            )

        _assert_reason(error, "response_evidence")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_new_tokens", "stop_token_ids"),
    (
        (True, ()),
        (16, [7]),
        (16, (True,)),
        (16, (-1,)),
    ),
)
async def test_preparation_rejects_noncanonical_decoding_types_before_transport(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
    max_new_tokens: object,
    stop_token_ids: object,
) -> None:
    bridge = _make_bridge()
    send = AsyncMock(side_effect=AssertionError("invalid config must not send"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            prepare_infbridge_run_envelope_v2(
                manifest,
                tokenizer,
                bridge,
                max_new_tokens=max_new_tokens,  # type: ignore[arg-type]
                stop_token_ids=stop_token_ids,  # type: ignore[arg-type]
            )

        _assert_reason(error, "run_envelope")
        assert send.await_count == 0
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_canonical_encoder_revalidates_exact_nested_types(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    bridge = _make_bridge()
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=16,
        )
        tampered = deepcopy(envelope)
        bad_plan = deepcopy(tampered.call_plans[0])
        object.__setattr__(bad_plan, "input_token_count", True)
        object.__setattr__(
            tampered,
            "call_plans",
            (bad_plan, *tampered.call_plans[1:]),
        )

        for encoder in (
            infbridge_run_envelope_v2_bytes,
            infbridge_run_envelope_v2_sha256,
        ):
            with pytest.raises(InfBridgeModelAdapterError) as error:
                encoder(tampered)
            _assert_reason(error, "run_envelope")

        with pytest.raises(InfBridgeModelAdapterError) as error:
            infbridge_run_envelope_v2_bytes(object())  # type: ignore[arg-type]
        _assert_reason(error, "run_envelope")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_decode_exception_is_closed_as_decode_failure(
    manifest: helpfulness.ModelRunManifest,
) -> None:
    class BrokenDecodeTokenizer(_ByteTokenizer):
        def decode(
            self,
            token_ids: tuple[int, ...],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            raise RuntimeError("decoder unavailable")

    tokenizer = BrokenDecodeTokenizer()
    bridge = _make_bridge()
    bridge._send_request = AsyncMock(return_value=_sglang_response(b"ANSWER", "stop"))
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=16,
        )
        with pytest.raises(InfBridgeModelAdapterError) as error:
            await InfBridgeModelAdapter(
                manifest,
                tokenizer,
                envelope,
                bridge,
            ).submit(0, _first_arm(manifest))

        _assert_reason(error, "decode_failure")
    finally:
        await bridge.aclose()


@pytest.mark.asyncio
async def test_sglang_subclass_is_not_accepted_as_the_built_in_backend(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ByteTokenizer,
) -> None:
    class SGLangSubclass(SGLangBridgeBackend):
        pass

    bridge = _make_bridge()
    bridge.backend = SGLangSubclass()
    send = AsyncMock(side_effect=AssertionError("subclass must not send"))
    bridge._send_request = send
    try:
        with pytest.raises(InfBridgeModelAdapterError) as error:
            prepare_infbridge_run_envelope_v2(
                manifest,
                tokenizer,
                bridge,
                max_new_tokens=16,
            )

        _assert_reason(error, "unsupported_backend")
        assert send.await_count == 0
    finally:
        await bridge.aclose()
