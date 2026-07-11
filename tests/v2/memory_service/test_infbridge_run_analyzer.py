# SPDX-License-Identifier: Apache-2.0

"""Tests for recovering evaluator calls without fabricating Memory evidence."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.memory_service import infbridge_run_analyzer as analyzer_module
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedModelCallExecutionV2,
    DecoderAuditMaterialV2,
    InfBridgeModelAdapter,
    InfBridgeModelAdapterError,
    RunEnvelopeV2,
    prepare_infbridge_run_envelope_v2,
)
from examples.memory_service.infbridge_run_analyzer import (
    InfBridgeRunAnalyzerError,
    LedgerModelAttritionV1,
    recover_infbridge_model_dry_run_v1,
)
from examples.memory_service.infbridge_run_ledger import (
    RunLedgerSessionV1,
    load_run_ledger,
)
from examples.memory_service.infbridge_run_runner import (
    prepare_infbridge_ledger,
    run_infbridge_ledger,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


class _AnalyzerByteTokenizer:
    _ARTIFACT_BYTES = b'analyzer-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'analyzer-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="analyzer-byte-tokenizer-v2",
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
        if skip_special_tokens is not True or clean_up_tokenization_spaces is not False:
            raise ValueError("analyzer decode policy drift")
        return bytes(token_ids).decode("utf-8")


def _build_inputs() -> tuple[
    helpfulness.ModelRunManifest,
    _AnalyzerByteTokenizer,
    RunEnvelopeV2,
    InfBridge,
]:
    tokenizer = _AnalyzerByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="7" * 40,
        evaluator_commit_sha="8" * 40,
        model_id="analyzer-test-model",
        model_weights_sha256="9" * 64,
        tokenizer_id="analyzer-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://analyzer-model.test/",
        pause_state=PauseState(),
        request_timeout=7.0,
        max_resubmit_retries=2,
        resubmit_wait=0.0,
        version=31,
    )
    envelope = prepare_infbridge_run_envelope_v2(
        prepared.manifest,
        tokenizer,
        bridge,
        max_new_tokens=48,
    )
    return prepared.manifest, tokenizer, envelope, bridge


def _sglang_response(output: bytes) -> dict[str, object]:
    return {
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                (-0.01 * (index + 1), token_id) for index, token_id in enumerate(output)
            ],
        }
    }


class _SixSuccessThenAttritionAdapter:
    def __init__(self, adapter: InfBridgeModelAdapter, bridge: InfBridge) -> None:
        self.adapter = adapter
        self.bridge = bridge
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        call_index = self.calls
        self.calls += 1
        if call_index >= 6:
            raise InfBridgeModelAdapterError("generation_failure")
        self.bridge._send_request = AsyncMock(
            return_value=_sglang_response(arm.encode("ascii"))
        )
        return await self.adapter.submit(case_index, arm)


@pytest.fixture(scope="module")
def sealed_mixed_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    Path,
    helpfulness.ModelRunManifest,
    _AnalyzerByteTokenizer,
    RunEnvelopeV2,
]:
    root = tmp_path_factory.mktemp("infbridge-analyzer")
    database_path = root / "mixed.sqlite3"
    coordination_directory = root / "coordination"
    manifest, tokenizer, envelope, bridge = _build_inputs()
    adapter = _SixSuccessThenAttritionAdapter(
        InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge),
        bridge,
    )
    try:
        sealed = asyncio.run(
            run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
                coordination_directory=coordination_directory,
            )
        )
    finally:
        asyncio.run(bridge.aclose())
    assert sealed.status == "SEALED"
    assert sealed.seal_kind == "complete_with_attrition"
    assert adapter.calls == 384
    return database_path, manifest, tokenizer, envelope


def _assert_analyzer_reason(
    error: pytest.ExceptionInfo[InfBridgeRunAnalyzerError],
    reason: str,
) -> None:
    assert type(error.value) is InfBridgeRunAnalyzerError
    assert error.value.reason == reason


def test_recovers_successes_and_precise_attrition_in_plan_order(
    sealed_mixed_run: tuple[
        Path,
        helpfulness.ModelRunManifest,
        _AnalyzerByteTokenizer,
        RunEnvelopeV2,
    ],
) -> None:
    database_path, manifest, tokenizer, envelope = sealed_mixed_run
    recovered = recover_infbridge_model_dry_run_v1(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)

    assert recovered.schema_version == 1
    assert recovered.run_id == snapshot.run_id
    assert recovered.run_root_sha256 == snapshot.stored_run_root_sha256
    assert recovered.manifest_sha256 == snapshot.manifest_sha256
    assert recovered.run_envelope_sha256 == snapshot.run_envelope_sha256
    assert recovered.seal_kind == "complete_with_attrition"
    assert recovered.succeeded_count == 6
    assert recovered.attrition_count == 378
    assert recovered.dry_run.validity == "invalid"
    assert recovered.dry_run.manifest_sha256 == recovered.manifest_sha256
    assert recovered.dry_run.manifest_sha256 == (
        helpfulness.model_run_manifest_sha256(manifest, tokenizer)
    )
    assert len(recovered.dry_run.calls) == 384
    assert len(recovered.dry_run.invalid_calls) == 378
    assert len(recovered.ledger_attrition) == 378

    observed_slots = tuple(
        (call.case_index, call.arm) for call in recovered.dry_run.calls
    )
    expected_slots = tuple((plan.case_index, plan.arm) for plan in envelope.call_plans)
    assert observed_slots == expected_slots
    assert tuple(call.arm for call in recovered.dry_run.calls[:6]) == tuple(
        plan.arm for plan in envelope.call_plans[:6]
    )
    assert tuple(call.arm for call in recovered.dry_run.calls[:6]) != (
        helpfulness.MODEL_ARMS
    )
    assert tuple(
        call.execution.response
        for call in recovered.dry_run.calls[:6]
        if call.execution is not None
    ) == tuple(plan.arm for plan in envelope.call_plans[:6])
    assert all(call.execution is None for call in recovered.dry_run.calls[6:])
    assert all(call.attempt_index == 0 for call in recovered.dry_run.calls)
    assert recovered.dry_run.invalid_calls == tuple(
        helpfulness.ModelDryRunInvalidCall(
            case_index=plan.case_index,
            arm=plan.arm,
            attempted=True,
            reason="model_call_failure",
        )
        for plan in envelope.call_plans[6:]
    )
    assert recovered.ledger_attrition == tuple(
        LedgerModelAttritionV1(
            slot_index=plan.slot_index,
            case_index=plan.case_index,
            arm=plan.arm,
            reason="generation_failure",
        )
        for plan in envelope.call_plans[6:]
    )
    without_source_evidence = helpfulness.analyze_model_run(
        manifest=manifest,
        tokenizer=tokenizer,
        dry_run=recovered.dry_run,
        outcomes=(),
        leakage_sentinels=(),
    )
    assert without_source_evidence.validity == "invalid"
    assert without_source_evidence.invalid_reasons == ("execution_completeness",)
    assert without_source_evidence.summary is None


def test_recovery_replays_each_success_artifact_again(
    sealed_mixed_run: tuple[
        Path,
        helpfulness.ModelRunManifest,
        _AnalyzerByteTokenizer,
        RunEnvelopeV2,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, manifest, tokenizer, envelope = sealed_mixed_run
    original = analyzer_module.audited_model_call_execution_v2_from_artifacts
    replayed_slots: list[int] = []

    def record_replay(*args, **kwargs):
        execution = original(*args, **kwargs)
        replayed_slots.append(execution.receipt.slot_index)
        return execution

    monkeypatch.setattr(
        analyzer_module,
        "audited_model_call_execution_v2_from_artifacts",
        record_replay,
    )
    first = recover_infbridge_model_dry_run_v1(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    second = recover_infbridge_model_dry_run_v1(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )

    assert first == second
    assert replayed_slots == [*range(6), *range(6)]


def test_recovery_does_not_downgrade_replay_failure_to_attrition(
    sealed_mixed_run: tuple[
        Path,
        helpfulness.ModelRunManifest,
        _AnalyzerByteTokenizer,
        RunEnvelopeV2,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, manifest, tokenizer, envelope = sealed_mixed_run

    def fail_replay(*_args, **_kwargs):
        raise InfBridgeModelAdapterError("trace_mismatch")

    monkeypatch.setattr(
        analyzer_module,
        "audited_model_call_execution_v2_from_artifacts",
        fail_replay,
    )
    with pytest.raises(InfBridgeRunAnalyzerError) as error:
        recover_infbridge_model_dry_run_v1(
            database_path,
            manifest,
            tokenizer,
            envelope,
        )

    _assert_analyzer_reason(error, "artifact_replay")


def test_complete_projection_produces_valid_dry_run(
    sealed_mixed_run: tuple[
        Path,
        helpfulness.ModelRunManifest,
        _AnalyzerByteTokenizer,
        RunEnvelopeV2,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, manifest, tokenizer, envelope = sealed_mixed_run
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    success_slots = tuple(
        replace(
            slot,
            state="SUCCEEDED",
            attempt_count=1,
            receipt_bytes=b"validated-by-loader-contract",
            trace_bytes=b"validated-by-loader-contract",
            response_evidence_bytes=b"validated-by-loader-contract",
            decoded_response_utf8=b"UNKNOWN",
            terminal_reason=None,
        )
        for slot in snapshot.slots
    )
    complete_snapshot = replace(
        snapshot,
        seal_kind="complete",
        slots=success_slots,
    )
    replay_index = 0

    def replay_validated_artifacts(*_args, **_kwargs):
        nonlocal replay_index
        plan = envelope.call_plans[replay_index]
        replay_index += 1
        registration = manifest.cases[plan.case_index]
        arm_call = next(call for call in registration.arm_calls if call.arm == plan.arm)
        prepared = arm_call.prepared_call
        legacy_execution = helpfulness.ModelCallExecution(
            response="UNKNOWN",
            consumer_input_receipt=prepared.consumer_input_receipt,
            model_call_receipt=prepared.expected_receipt,
            rendered_context_token_count=prepared.rendered_context_token_count,
            valid=True,
            invalid_reason=None,
        )
        return SimpleNamespace(legacy_execution=legacy_execution)

    monkeypatch.setattr(
        analyzer_module, "load_run_ledger", lambda *_args: complete_snapshot
    )
    monkeypatch.setattr(
        analyzer_module,
        "audited_model_call_execution_v2_from_artifacts",
        replay_validated_artifacts,
    )
    recovered = recover_infbridge_model_dry_run_v1(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )

    assert replay_index == 384
    assert recovered.seal_kind == "complete"
    assert recovered.succeeded_count == 384
    assert recovered.attrition_count == 0
    assert recovered.ledger_attrition == ()
    assert recovered.dry_run.validity == "valid"
    assert recovered.dry_run.invalid_calls == ()
    assert all(call.execution is not None for call in recovered.dry_run.calls)


@pytest.mark.parametrize(
    ("sealed", "reason"),
    ((False, "run_not_sealed"), (True, "run_indeterminate")),
)
def test_refuses_open_or_indeterminate_run(
    tmp_path: Path,
    sealed_mixed_run: tuple[
        Path,
        helpfulness.ModelRunManifest,
        _AnalyzerByteTokenizer,
        RunEnvelopeV2,
    ],
    sealed: bool,
    reason: str,
) -> None:
    _mixed_path, manifest, tokenizer, envelope = sealed_mixed_run
    database_path = tmp_path / f"{reason}.sqlite3"
    asyncio.run(
        prepare_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            coordination_directory=tmp_path / "coordination",
        )
    )
    if sealed:
        session = RunLedgerSessionV1.resume(
            database_path,
            manifest,
            tokenizer,
            envelope,
        )
        session.mark_started(0)
        session.seal_indeterminate(0, "unexpected_failure")

    with pytest.raises(InfBridgeRunAnalyzerError) as error:
        recover_infbridge_model_dry_run_v1(
            database_path,
            manifest,
            tokenizer,
            envelope,
        )

    _assert_analyzer_reason(error, reason)
