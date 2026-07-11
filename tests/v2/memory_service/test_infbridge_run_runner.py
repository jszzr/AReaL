# SPDX-License-Identifier: Apache-2.0

"""Crash and ordering tests for the durable Memory model-run cursor."""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import AsyncMock

import pytest

from examples.memory_service import infbridge_run_ledger as ledger_module
from examples.memory_service import infbridge_run_runner as runner_module
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedModelCallExecutionV2,
    DecoderAuditMaterialV2,
    InfBridgeModelAdapter,
    InfBridgeModelAdapterError,
    RunEnvelopeV2,
    prepare_infbridge_run_envelope_v2,
)
from examples.memory_service.infbridge_run_ledger import (
    RunLedgerError,
    RunLedgerSessionV1,
    RunLedgerSnapshotV1,
    _slot_leaf_sha256,
    initialize_run_ledger,
    load_run_ledger,
)
from examples.memory_service.infbridge_run_runner import (
    _exclusive_run_lock,
    _open_coordination_directory,
    prepare_infbridge_ledger,
)
from examples.memory_service.infbridge_run_runner import (
    run_infbridge_ledger as _run_infbridge_ledger,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


async def run_infbridge_ledger(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _RunnerByteTokenizer,
    envelope: RunEnvelopeV2,
    adapter: Any,
    *,
    mode: str,
    coordination_directory: str | os.PathLike[str] | None = None,
) -> RunLedgerSnapshotV1:
    directory = (
        Path(database_path).parent / "coordination"
        if coordination_directory is None
        else coordination_directory
    )
    return await _run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        adapter,
        mode=mode,  # type: ignore[arg-type]
        coordination_directory=directory,
    )


class _RunnerByteTokenizer:
    _ARTIFACT_BYTES = b'runner-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'runner-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="runner-byte-tokenizer-v2",
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
            raise ValueError("runner decode policy drift")
        return bytes(token_ids).decode("utf-8")


def _build_inputs() -> tuple[
    helpfulness.ModelRunManifest,
    _RunnerByteTokenizer,
    RunEnvelopeV2,
    InfBridge,
]:
    tokenizer = _RunnerByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="4" * 40,
        evaluator_commit_sha="5" * 40,
        model_id="runner-test-model",
        model_weights_sha256="6" * 64,
        tokenizer_id="runner-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://runner-model.test/",
        pause_state=PauseState(),
        request_timeout=9.0,
        max_resubmit_retries=3,
        resubmit_wait=0.0,
        version=29,
    )
    envelope = prepare_infbridge_run_envelope_v2(
        prepared.manifest,
        tokenizer,
        bridge,
        max_new_tokens=32,
    )
    return prepared.manifest, tokenizer, envelope, bridge


@pytest.fixture(scope="module")
def runner_inputs() -> tuple[
    helpfulness.ModelRunManifest,
    _RunnerByteTokenizer,
    RunEnvelopeV2,
    InfBridge,
]:
    values = _build_inputs()
    yield values
    asyncio.run(values[3].aclose())


def _assert_reason(error: pytest.ExceptionInfo[RunLedgerError], reason: str) -> None:
    assert type(error.value) is RunLedgerError
    assert error.value.reason == reason


def _sglang_response(output: bytes) -> dict[str, object]:
    return {
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                (-0.01 * (index + 1), token_id) for index, token_id in enumerate(output)
            ],
        }
    }


class _NoCallAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        raise AssertionError(f"unexpected adapter call: {case_index}/{arm}")


class _AllAttritionAdapter:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.calls: list[tuple[int, str]] = []

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        slot_index = len(self.calls)
        connection = sqlite3.connect(self.database_path)
        try:
            assert connection.execute(
                "SELECT state, attempt_count FROM run_ledger_slots "
                "WHERE slot_index = ?",
                (slot_index,),
            ).fetchone() == ("STARTED", 1)
        finally:
            connection.close()
        self.calls.append((case_index, arm))
        raise InfBridgeModelAdapterError("generation_failure")


class _UnexpectedAdapter:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        raise self.error


class _BlockingAdapter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        self.entered.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _SuppressCancellationAdapter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            raise InfBridgeModelAdapterError("generation_failure") from None
        raise AssertionError("unreachable")


class _InvalidExecutionAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def submit(  # type: ignore[override]
        self,
        case_index: int,
        arm: str,
    ) -> object:
        self.calls += 1
        return object()


class _CorruptSuccessAdapter:
    def __init__(self, real_adapter: InfBridgeModelAdapter) -> None:
        self.real_adapter = real_adapter
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        execution = await self.real_adapter.submit(case_index, arm)
        return replace(execution, response=f"{execution.response}-tampered")


class _OneSuccessThenCrashAdapter:
    def __init__(self, real_adapter: InfBridgeModelAdapter) -> None:
        self.real_adapter = real_adapter
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        call_index = self.calls
        self.calls += 1
        if call_index == 0:
            return await self.real_adapter.submit(case_index, arm)
        raise RuntimeError("crash after one durable success")


class _RecordingCrashAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls.append((case_index, arm))
        raise RuntimeError("recording crash")


class _AttritionThenCrashAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls.append((case_index, arm))
        if len(self.calls) == 1:
            raise InfBridgeModelAdapterError("generation_failure")
        raise RuntimeError("crash after recovered terminal ACK")


class _RollbackStartedAdapter:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.calls = 0

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls += 1
        connection = sqlite3.connect(self.database_path)
        try:
            plan_sha256 = connection.execute(
                "SELECT plan_sha256 FROM run_ledger_slots WHERE slot_index = 0"
            ).fetchone()[0]
            planned_leaf = _slot_leaf_sha256(
                slot_index=0,
                plan_sha256=plan_sha256,
                state="PLANNED",
                attempt_count=0,
                receipt_bytes=None,
                trace_bytes=None,
                response_evidence_bytes=None,
                decoded_response_utf8=None,
                terminal_reason=None,
            )
            connection.execute(
                "UPDATE run_ledger_slots SET state = 'PLANNED', "
                "attempt_count = 0, leaf_sha256 = ? WHERE slot_index = 0",
                (planned_leaf,),
            )
            connection.commit()
        finally:
            connection.close()
        raise InfBridgeModelAdapterError("generation_failure")


class _BusyAttritionThenCrashAdapter:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.calls: list[tuple[int, str]] = []
        self.reader: sqlite3.Connection | None = None
        self.reader_released = False

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        self.calls.append((case_index, arm))
        if len(self.calls) > 1:
            raise RuntimeError("crash after BUSY recovery")
        reader = sqlite3.connect(self.database_path)
        reader.execute("BEGIN")
        assert reader.execute(
            "SELECT state FROM run_ledger_slots WHERE slot_index = 0"
        ).fetchone() == ("STARTED",)
        self.reader = reader
        raise InfBridgeModelAdapterError("generation_failure")

    def release_reader(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None
            self.reader_released = True


class _ExitAfterSideEffectAdapter:
    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> NoReturn:
        descriptor = os.open(
            self.marker_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, f"{case_index}:{arm}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os._exit(73)


class _AttritionAfterSideEffectAdapter:
    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2:
        descriptor = os.open(
            self.marker_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, f"{case_index}:{arm}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise InfBridgeModelAdapterError("generation_failure")


def _crash_worker(
    database_path: str,
    coordination_directory: str,
    marker_path: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = _build_inputs()
    asyncio.run(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            _ExitAfterSideEffectAdapter(marker_path),
            mode="resume",
            coordination_directory=coordination_directory,
        )
    )


def _terminal_crash_worker(
    database_path: str,
    coordination_directory: str,
    marker_path: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = _build_inputs()
    original = RunLedgerSessionV1.record_attrition

    def commit_then_exit(
        session: RunLedgerSessionV1,
        slot_index: int,
        reason: str,
    ) -> NoReturn:
        original(session, slot_index, reason)
        os._exit(74)

    RunLedgerSessionV1.record_attrition = commit_then_exit  # type: ignore[method-assign]
    asyncio.run(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            _AttritionAfterSideEffectAdapter(marker_path),
            mode="resume",
            coordination_directory=coordination_directory,
        )
    )


def _hold_lock_worker(
    coordination_directory: str,
    ready_pipe,
    release_pipe,
) -> None:
    async def hold() -> None:
        _path, descriptor = _open_coordination_directory(coordination_directory)
        try:
            async with _exclusive_run_lock(descriptor, "test.lock"):
                ready_pipe.send("locked")
                release_pipe.recv()
        finally:
            os.close(descriptor)

    asyncio.run(hold())


def _probe_lock_worker(
    coordination_directory: str,
    ready_pipe,
    acquired_pipe,
) -> None:
    async def probe() -> None:
        _path, descriptor = _open_coordination_directory(coordination_directory)
        try:
            ready_pipe.send("ready")
            async with _exclusive_run_lock(descriptor, "test.lock"):
                acquired_pipe.send("acquired")
        finally:
            os.close(descriptor)

    asyncio.run(probe())


@pytest.mark.asyncio
async def test_runner_precommits_and_visits_each_fixed_slot_once(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "all-attrition.sqlite3"
    adapter = _AllAttritionAdapter(database_path)
    original_seal = RunLedgerSessionV1.seal_complete
    seal_ack_lost = False

    def seal_then_raise(session: RunLedgerSessionV1) -> RunLedgerSnapshotV1:
        nonlocal seal_ack_lost
        snapshot = original_seal(session)
        if not seal_ack_lost:
            seal_ack_lost = True
            raise RunLedgerError("ledger_persistence")
        return snapshot

    monkeypatch.setattr(RunLedgerSessionV1, "seal_complete", seal_then_raise)

    sealed = await run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        adapter,
        mode="new",
    )

    assert sealed.status == "SEALED"
    assert sealed.seal_kind == "complete_with_attrition"
    assert adapter.calls == [
        (plan.case_index, plan.arm) for plan in envelope.call_plans
    ]
    assert len(adapter.calls) == 384
    assert seal_ack_lost
    assert all(slot.state == "ATTRITION" for slot in sealed.slots)
    no_call = _NoCallAdapter()
    resumed = await run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        no_call,
        mode="resume",
    )
    assert resumed == sealed
    assert no_call.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        (RuntimeError("unexpected"), "unexpected_failure"),
        (asyncio.CancelledError("cancelled"), "runner_cancelled"),
    ),
)
async def test_runner_seals_unexpected_or_cancelled_call_without_continuing(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    failure: BaseException,
    reason: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / f"{reason}.sqlite3"
    adapter = _UnexpectedAdapter(failure)

    with pytest.raises(type(failure)) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    assert error.value is failure
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.seal_kind == "indeterminate"
    assert snapshot.slots[0].state == "INDETERMINATE"
    assert snapshot.slots[0].terminal_reason == reason
    assert all(slot.state == "PLANNED" for slot in snapshot.slots[1:])


@pytest.mark.asyncio
async def test_external_task_cancellation_is_sealed_and_never_reissued(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "task-cancel.sqlite3"
    adapter = _BlockingAdapter()
    task = asyncio.create_task(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )
    )
    await asyncio.wait_for(adapter.entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].state == "INDETERMINATE"
    assert snapshot.slots[0].terminal_reason == "runner_cancelled"
    no_call = _NoCallAdapter()
    resumed = await run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        no_call,
        mode="resume",
    )
    assert resumed == snapshot
    assert no_call.calls == 0


@pytest.mark.asyncio
async def test_adapter_cannot_suppress_pending_task_cancellation(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "suppressed-cancel.sqlite3"
    adapter = _SuppressCancellationAdapter()
    task = asyncio.create_task(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )
    )
    await asyncio.wait_for(adapter.entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "runner_cancelled"


@pytest.mark.asyncio
async def test_unknown_adapter_reason_is_protocol_failure(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    failure = InfBridgeModelAdapterError("invented_reason")
    adapter = _UnexpectedAdapter(failure)
    database_path = tmp_path / "unknown-adapter-reason.sqlite3"

    with pytest.raises(InfBridgeModelAdapterError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    assert error.value is failure
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "unexpected_failure"


@pytest.mark.asyncio
async def test_invalid_success_artifact_fails_closed(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "invalid-success.sqlite3"
    adapter = _InvalidExecutionAdapter()

    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,  # type: ignore[arg-type]
            mode="new",
        )

    _assert_reason(error, "ledger_artifact")
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "artifact_validation_failure"


@pytest.mark.asyncio
async def test_exact_execution_type_with_corrupt_artifact_fails_closed(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, bridge = runner_inputs
    database_path = tmp_path / "corrupt-success.sqlite3"
    original_send = bridge._send_request
    bridge._send_request = AsyncMock(return_value=_sglang_response(b"ANSWER"))
    adapter = _CorruptSuccessAdapter(
        InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    )
    try:
        with pytest.raises(RunLedgerError) as error:
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
            )
    finally:
        bridge._send_request = original_send

    _assert_reason(error, "ledger_artifact")
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "artifact_validation_failure"


@pytest.mark.asyncio
async def test_resume_seals_orphan_started_without_adapter_call(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "orphan.sqlite3"
    await prepare_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        coordination_directory=tmp_path / "coordination",
    )
    RunLedgerSessionV1.resume(
        database_path, manifest, tokenizer, envelope
    ).mark_started(0)
    adapter = _NoCallAdapter()

    sealed = await run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        adapter,
        mode="resume",
    )

    assert adapter.calls == 0
    assert sealed.status == "SEALED"
    assert sealed.slots[0].state == "INDETERMINATE"
    assert sealed.slots[0].terminal_reason == "orphan_started"


@pytest.mark.asyncio
async def test_runner_persists_success_then_seals_later_unexpected_call(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, bridge = runner_inputs
    database_path = tmp_path / "success-then-crash.sqlite3"
    original_send = bridge._send_request
    bridge._send_request = AsyncMock(return_value=_sglang_response(b"ANSWER"))
    adapter = _OneSuccessThenCrashAdapter(
        InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    )
    try:
        with pytest.raises(RuntimeError, match="one durable success"):
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
            )
    finally:
        bridge._send_request = original_send

    assert adapter.calls == 2
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].state == "SUCCEEDED"
    assert snapshot.slots[0].decoded_response_utf8 == b"ANSWER"
    assert snapshot.slots[1].state == "INDETERMINATE"
    assert all(slot.state == "PLANNED" for slot in snapshot.slots[2:])


@pytest.mark.asyncio
async def test_success_commit_with_lost_ack_is_not_reissued(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, bridge = runner_inputs
    database_path = tmp_path / "success-lost-ack.sqlite3"
    original_send = bridge._send_request
    bridge._send_request = AsyncMock(return_value=_sglang_response(b"ANSWER"))
    adapter = _OneSuccessThenCrashAdapter(
        InfBridgeModelAdapter(manifest, tokenizer, envelope, bridge)
    )
    original_record = RunLedgerSessionV1.record_success
    injected = False

    def commit_then_raise(
        session: RunLedgerSessionV1,
        slot_index: int,
        execution: AuditedModelCallExecutionV2,
    ) -> RunLedgerSnapshotV1:
        nonlocal injected
        snapshot = original_record(session, slot_index, execution)
        if not injected:
            injected = True
            raise RunLedgerError("ledger_persistence")
        return snapshot

    try:
        monkeypatch.setattr(
            RunLedgerSessionV1,
            "record_success",
            commit_then_raise,
        )
        with pytest.raises(RuntimeError, match="one durable success"):
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
            )
    finally:
        bridge._send_request = original_send

    assert injected
    assert adapter.calls == 2
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.slots[0].state == "SUCCEEDED"
    assert snapshot.slots[0].decoded_response_utf8 == b"ANSWER"
    assert snapshot.slots[1].state == "INDETERMINATE"


def test_process_exit_after_side_effect_is_never_reissued(
    tmp_path: Path,
) -> None:
    manifest, tokenizer, envelope, bridge = _build_inputs()
    database_path = tmp_path / "hard-crash.sqlite3"
    coordination_directory = tmp_path / "coordination"
    marker_path = tmp_path / "remote-side-effect.log"
    asyncio.run(
        prepare_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            coordination_directory=coordination_directory,
        )
    )
    asyncio.run(bridge.aclose())
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_worker,
        args=(
            str(database_path),
            str(coordination_directory),
            str(marker_path),
        ),
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=20)
    assert not process.is_alive()
    assert process.exitcode == 73
    assert marker_path.read_text(encoding="ascii").count("\n") == 1
    crashed = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert crashed.status == "OPEN"
    assert crashed.slots[0].state == "STARTED"

    manifest, tokenizer, envelope, bridge = _build_inputs()
    adapter = _NoCallAdapter()
    sealed = asyncio.run(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="resume",
            coordination_directory=coordination_directory,
        )
    )
    asyncio.run(bridge.aclose())
    assert adapter.calls == 0
    assert sealed.status == "SEALED"
    assert sealed.slots[0].state == "INDETERMINATE"
    assert sealed.slots[0].terminal_reason == "orphan_started"
    assert marker_path.read_text(encoding="ascii").count("\n") == 1


def test_process_exit_after_terminal_commit_resumes_at_next_slot(
    tmp_path: Path,
) -> None:
    manifest, tokenizer, envelope, bridge = _build_inputs()
    database_path = tmp_path / "terminal-hard-crash.sqlite3"
    coordination_directory = tmp_path / "coordination"
    marker_path = tmp_path / "terminal-side-effect.log"
    asyncio.run(
        prepare_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            coordination_directory=coordination_directory,
        )
    )
    asyncio.run(bridge.aclose())
    process = multiprocessing.get_context("spawn").Process(
        target=_terminal_crash_worker,
        args=(
            str(database_path),
            str(coordination_directory),
            str(marker_path),
        ),
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=20)
    assert not process.is_alive()
    assert process.exitcode == 74
    assert marker_path.read_text(encoding="ascii").count("\n") == 1
    committed = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert committed.status == "OPEN"
    assert committed.slots[0].state == "ATTRITION"
    assert committed.slots[1].state == "PLANNED"

    manifest, tokenizer, envelope, bridge = _build_inputs()
    adapter = _RecordingCrashAdapter()
    with pytest.raises(RuntimeError, match="recording crash"):
        asyncio.run(
            run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="resume",
                coordination_directory=coordination_directory,
            )
        )
    asyncio.run(bridge.aclose())
    assert adapter.calls == [
        (envelope.call_plans[1].case_index, envelope.call_plans[1].arm)
    ]
    sealed = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert sealed.slots[0].state == "ATTRITION"
    assert sealed.slots[1].state == "INDETERMINATE"
    assert marker_path.read_text(encoding="ascii").count("\n") == 1


def test_live_runner_lock_is_never_stolen_by_timeout(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    coordination_directory = str(tmp_path / "coordination")
    holder_parent, holder_child = context.Pipe()
    release_parent, release_child = context.Pipe()
    probe_ready_parent, probe_ready_child = context.Pipe()
    probe_parent, probe_child = context.Pipe()
    holder = context.Process(
        target=_hold_lock_worker,
        args=(coordination_directory, holder_child, release_child),
    )
    probe = context.Process(
        target=_probe_lock_worker,
        args=(coordination_directory, probe_ready_child, probe_child),
    )
    try:
        holder.start()
        assert holder_parent.poll(20)
        assert holder_parent.recv() == "locked"
        probe.start()
        assert probe_ready_parent.poll(20)
        assert probe_ready_parent.recv() == "ready"
        assert not probe_parent.poll(1.0)
        assert probe.is_alive()
        release_parent.send("release")
        assert probe_parent.poll(20)
        assert probe_parent.recv() == "acquired"
    finally:
        if holder.is_alive():
            release_parent.send("release")
        holder.join(timeout=20)
        probe.join(timeout=20)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=20)
        if probe.is_alive():
            probe.terminate()
            probe.join(timeout=20)
    assert holder.exitcode == 0
    assert probe.exitcode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ("before_commit", "after_commit"))
async def test_terminal_persistence_retry_never_reissues_model_call(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "lost-terminal-ack.sqlite3"
    adapter = _AttritionThenCrashAdapter()
    original = RunLedgerSessionV1.record_attrition
    injected = False

    def commit_then_raise(
        session: RunLedgerSessionV1,
        slot_index: int,
        reason: str,
    ):
        nonlocal injected
        if not injected:
            injected = True
            if failure_phase == "before_commit":
                raise RunLedgerError("ledger_persistence")
            original(session, slot_index, reason)
            raise RunLedgerError("ledger_persistence")
        return original(session, slot_index, reason)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            RunLedgerSessionV1,
            "record_attrition",
            commit_then_raise,
        )
        with pytest.raises(RuntimeError, match="recovered terminal ACK"):
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
            )
    assert adapter.calls == [
        (envelope.call_plans[0].case_index, envelope.call_plans[0].arm),
        (envelope.call_plans[1].case_index, envelope.call_plans[1].arm),
    ]
    after_ack_loss = load_run_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    assert after_ack_loss.slots[0].state == "ATTRITION"
    assert after_ack_loss.slots[1].state == "INDETERMINATE"


@pytest.mark.asyncio
async def test_keyboard_interrupt_after_terminal_commit_never_calls_next_slot(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "terminal-keyboard-interrupt.sqlite3"
    adapter = _AttritionThenCrashAdapter()
    original = RunLedgerSessionV1.record_attrition

    def commit_then_interrupt(
        session: RunLedgerSessionV1,
        slot_index: int,
        reason: str,
    ) -> NoReturn:
        original(session, slot_index, reason)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        RunLedgerSessionV1,
        "record_attrition",
        commit_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    assert adapter.calls == [
        (envelope.call_plans[0].case_index, envelope.call_plans[0].arm)
    ]
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.slots[0].state == "ATTRITION"
    assert snapshot.slots[1].state == "PLANNED"


@pytest.mark.asyncio
async def test_persistent_terminal_failure_is_sealed_and_not_reissued(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "persistent-terminal-failure.sqlite3"
    adapter = _UnexpectedAdapter(InfBridgeModelAdapterError("generation_failure"))
    persistence_attempts = 0

    def always_fail(
        _session: RunLedgerSessionV1,
        _slot_index: int,
        _reason: str,
    ) -> RunLedgerSnapshotV1:
        nonlocal persistence_attempts
        persistence_attempts += 1
        raise RunLedgerError("ledger_persistence")

    monkeypatch.setattr(RunLedgerSessionV1, "record_attrition", always_fail)
    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    _assert_reason(error, "ledger_persistence")
    assert persistence_attempts == runner_module._TERMINAL_RETRY_COUNT
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "terminal_persistence_failure"


@pytest.mark.asyncio
async def test_cancellation_during_terminal_retry_is_sealed(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "cancel-terminal-retry.sqlite3"
    adapter = _UnexpectedAdapter(InfBridgeModelAdapterError("generation_failure"))
    retry_entered = asyncio.Event()
    original_retry = runner_module._retry_after_remote_call

    def fail_terminal(
        _session: RunLedgerSessionV1,
        _slot_index: int,
        _reason: str,
    ) -> RunLedgerSnapshotV1:
        raise RunLedgerError("ledger_persistence")

    async def slow_retry(
        session: RunLedgerSessionV1,
        slot_index: int,
    ) -> None:
        retry_entered.set()
        await original_retry(session, slot_index, delay=60.0)

    monkeypatch.setattr(RunLedgerSessionV1, "record_attrition", fail_terminal)
    monkeypatch.setattr(runner_module, "_retry_after_remote_call", slow_retry)
    task = asyncio.create_task(
        run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )
    )
    await asyncio.wait_for(retry_entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "runner_cancelled"


@pytest.mark.asyncio
async def test_post_call_started_rollback_is_sealed_without_reissue(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "lost-started.sqlite3"
    adapter = _RollbackStartedAdapter(database_path)

    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    _assert_reason(error, "ledger_state")
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].state == "INDETERMINATE"
    assert snapshot.slots[0].terminal_reason == "started_record_lost"
    no_call = _NoCallAdapter()
    resumed = await run_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        no_call,
        mode="resume",
    )
    assert resumed == snapshot
    assert no_call.calls == 0


@pytest.mark.asyncio
async def test_sqlite_busy_retries_terminal_write_not_model_call(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "busy-terminal.sqlite3"
    adapter = _BusyAttritionThenCrashAdapter(database_path)
    original = RunLedgerSessionV1.record_attrition
    persistence_attempts = 0

    def count_attempts(
        session: RunLedgerSessionV1,
        slot_index: int,
        reason: str,
    ) -> RunLedgerSnapshotV1:
        nonlocal persistence_attempts
        persistence_attempts += 1
        try:
            return original(session, slot_index, reason)
        except RunLedgerError as error:
            if persistence_attempts == 1:
                assert error.reason == "ledger_persistence"
                adapter.release_reader()
            raise

    monkeypatch.setattr(ledger_module, "_BUSY_TIMEOUT_MS", 25)
    monkeypatch.setattr(
        RunLedgerSessionV1,
        "record_attrition",
        count_attempts,
    )
    try:
        with pytest.raises(RuntimeError, match="BUSY recovery"):
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
            )
    finally:
        adapter.release_reader()

    assert adapter.reader_released
    assert persistence_attempts >= 2
    assert adapter.calls == [
        (envelope.call_plans[0].case_index, envelope.call_plans[0].arm),
        (envelope.call_plans[1].case_index, envelope.call_plans[1].arm),
    ]
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.slots[0].state == "ATTRITION"
    assert snapshot.slots[1].state == "INDETERMINATE"


@pytest.mark.asyncio
async def test_started_commit_failure_never_reaches_adapter(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "started-commit-failure.sqlite3"
    adapter = _NoCallAdapter()

    def fail_started(_session: RunLedgerSessionV1, _slot_index: int):
        raise RunLedgerError("ledger_persistence")

    monkeypatch.setattr(RunLedgerSessionV1, "mark_started", fail_started)
    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    _assert_reason(error, "ledger_persistence")
    assert adapter.calls == 0
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert all(slot.state == "PLANNED" for slot in snapshot.slots)


@pytest.mark.asyncio
async def test_started_commit_with_lost_ack_calls_adapter_exactly_once(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "started-lost-ack.sqlite3"
    failure = RuntimeError("stop after STARTED recovery")
    adapter = _UnexpectedAdapter(failure)
    original = RunLedgerSessionV1.mark_started
    injected = False

    def commit_then_raise(
        session: RunLedgerSessionV1,
        slot_index: int,
    ) -> RunLedgerSnapshotV1:
        nonlocal injected
        snapshot = original(session, slot_index)
        if not injected:
            injected = True
            raise RunLedgerError("ledger_persistence")
        return snapshot

    monkeypatch.setattr(RunLedgerSessionV1, "mark_started", commit_then_raise)
    with pytest.raises(RuntimeError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    assert error.value is failure
    assert adapter.calls == 1
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "SEALED"
    assert snapshot.slots[0].terminal_reason == "unexpected_failure"


@pytest.mark.asyncio
async def test_keyboard_interrupt_after_started_commit_never_calls_adapter(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    database_path = tmp_path / "started-keyboard-interrupt.sqlite3"
    adapter = _NoCallAdapter()
    original = RunLedgerSessionV1.mark_started

    def commit_then_interrupt(
        session: RunLedgerSessionV1,
        slot_index: int,
    ) -> NoReturn:
        original(session, slot_index)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        RunLedgerSessionV1,
        "mark_started",
        commit_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="new",
        )

    assert adapter.calls == 0
    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert snapshot.status == "OPEN"
    assert snapshot.slots[0].state == "STARTED"


@pytest.mark.asyncio
async def test_same_event_loop_competitor_waits_without_blocking_or_calling(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    first_adapter = _BlockingAdapter()
    first_task = asyncio.create_task(
        run_infbridge_ledger(
            tmp_path / "first.sqlite3",
            manifest,
            tokenizer,
            envelope,
            first_adapter,
            mode="new",
            coordination_directory=coordination_directory,
        )
    )
    await asyncio.wait_for(first_adapter.entered.wait(), timeout=10)
    second_adapter = _NoCallAdapter()
    second_database = tmp_path / "second.sqlite3"
    second_task = asyncio.create_task(
        run_infbridge_ledger(
            second_database,
            manifest,
            tokenizer,
            envelope,
            second_adapter,
            mode="new",
            coordination_directory=coordination_directory,
        )
    )
    await asyncio.sleep(0.15)
    assert not second_task.done()
    assert second_adapter.calls == 0
    assert not second_database.exists()

    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    assert first_adapter.calls == 1
    third_adapter = _NoCallAdapter()
    third = await asyncio.wait_for(
        run_infbridge_ledger(
            tmp_path / "first.sqlite3",
            manifest,
            tokenizer,
            envelope,
            third_adapter,
            mode="resume",
            coordination_directory=coordination_directory,
        ),
        timeout=10,
    )
    assert third.status == "SEALED"
    assert third_adapter.calls == 0


@pytest.mark.asyncio
async def test_same_run_id_cannot_be_rebound_to_another_database(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    first_database = tmp_path / "first.sqlite3"
    await prepare_infbridge_ledger(
        first_database,
        manifest,
        tokenizer,
        envelope,
        coordination_directory=coordination_directory,
    )
    adapter = _NoCallAdapter()

    for mode in ("new", "resume"):
        with pytest.raises(RunLedgerError) as error:
            await run_infbridge_ledger(
                tmp_path / f"other-{mode}.sqlite3",
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode=mode,
                coordination_directory=coordination_directory,
            )
        _assert_reason(error, "ledger_coordination")
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_ready_binding_rejects_same_run_database_inode_replacement(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    database_path = tmp_path / "bound.sqlite3"
    replacement_path = tmp_path / "replacement.sqlite3"
    await prepare_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        coordination_directory=coordination_directory,
    )
    initialize_run_ledger(replacement_path, manifest, tokenizer, envelope)
    os.replace(replacement_path, database_path)
    adapter = _NoCallAdapter()

    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="resume",
            coordination_directory=coordination_directory,
        )

    _assert_reason(error, "ledger_coordination")
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_resume_rechecks_binding_after_session_open(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    database_path = tmp_path / "bound.sqlite3"
    replacement_path = tmp_path / "replacement.sqlite3"
    await prepare_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        coordination_directory=coordination_directory,
    )
    initialize_run_ledger(replacement_path, manifest, tokenizer, envelope)
    original = runner_module._require_coordination_binding
    replaced = False

    def replace_after_first_check(
        coordination_descriptor: int,
        run_id: str,
        checked_database_path: str,
    ) -> tuple[int, int]:
        nonlocal replaced
        identity = original(
            coordination_descriptor,
            run_id,
            checked_database_path,
        )
        if not replaced:
            replaced = True
            os.replace(replacement_path, database_path)
        return identity

    monkeypatch.setattr(
        runner_module,
        "_require_coordination_binding",
        replace_after_first_check,
    )
    adapter = _NoCallAdapter()
    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="resume",
            coordination_directory=coordination_directory,
        )

    _assert_reason(error, "ledger_coordination")
    assert replaced
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_finalize_crash_leaves_fail_closed_reservation(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    database_path = tmp_path / "reserved.sqlite3"
    adapter = _NoCallAdapter()
    original_replace = runner_module.os.replace

    def fail_ready_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if source.endswith(".ready") and destination.endswith(".json"):
            raise OSError("injected crash before READY rename")
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(runner_module.os, "replace", fail_ready_replace)
        with pytest.raises(RunLedgerError) as error:
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode="new",
                coordination_directory=coordination_directory,
            )
        _assert_reason(error, "ledger_coordination")
    assert database_path.exists()
    assert adapter.calls == 0
    assert tuple(coordination_directory.glob("*.ready"))

    for mode in ("new", "resume"):
        with pytest.raises(RunLedgerError) as retry_error:
            await run_infbridge_ledger(
                database_path,
                manifest,
                tokenizer,
                envelope,
                adapter,
                mode=mode,
                coordination_directory=coordination_directory,
            )
        _assert_reason(retry_error, "ledger_coordination")
    assert adapter.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    ("content", "permissions", "hardlink", "symlink"),
)
async def test_resume_rejects_tampered_coordination_binding(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    tamper: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "coordination"
    database_path = tmp_path / "bound.sqlite3"
    prepared = await prepare_infbridge_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
        coordination_directory=coordination_directory,
    )
    binding = coordination_directory / f"{prepared.run_id}.json"
    if tamper == "content":
        binding.write_bytes(b"{}")
    elif tamper == "permissions":
        binding.chmod(0o644)
    elif tamper == "hardlink":
        os.link(binding, coordination_directory / "binding-alias")
    else:
        target = tmp_path / "binding-target"
        target.write_bytes(binding.read_bytes())
        target.chmod(0o600)
        binding.unlink()
        binding.symlink_to(target)

    adapter = _NoCallAdapter()
    with pytest.raises(RunLedgerError) as error:
        await run_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            adapter,
            mode="resume",
            coordination_directory=coordination_directory,
        )
    _assert_reason(error, "ledger_coordination")
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_coordination_directory_must_be_private(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    coordination_directory = tmp_path / "wide-coordination"
    coordination_directory.mkdir(mode=0o700)
    coordination_directory.chmod(0o755)
    database_path = tmp_path / "never-created.sqlite3"

    with pytest.raises(RunLedgerError) as error:
        await prepare_infbridge_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
            coordination_directory=coordination_directory,
        )

    _assert_reason(error, "ledger_coordination")
    assert not database_path.exists()


@pytest.mark.asyncio
async def test_runner_requires_explicit_new_or_resume_mode(
    tmp_path: Path,
    runner_inputs: tuple[
        helpfulness.ModelRunManifest,
        _RunnerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = runner_inputs
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(RunLedgerError) as missing_error:
        await run_infbridge_ledger(
            missing,
            manifest,
            tokenizer,
            envelope,
            _NoCallAdapter(),
            mode="resume",
        )
    _assert_reason(missing_error, "ledger_coordination")
    assert not missing.exists()

    existing = tmp_path / "existing.sqlite3"
    RunLedgerSessionV1.create(existing, manifest, tokenizer, envelope)
    with pytest.raises(RunLedgerError) as exists_error:
        await run_infbridge_ledger(
            existing,
            manifest,
            tokenizer,
            envelope,
            _NoCallAdapter(),
            mode="new",
        )
    _assert_reason(exists_error, "ledger_exists")
    no_adoption = _NoCallAdapter()
    with pytest.raises(RunLedgerError) as adoption_error:
        await run_infbridge_ledger(
            existing,
            manifest,
            tokenizer,
            envelope,
            no_adoption,
            mode="resume",
        )
    _assert_reason(adoption_error, "ledger_coordination")
    assert no_adoption.calls == 0

    with pytest.raises(RunLedgerError) as mode_error:
        await run_infbridge_ledger(
            tmp_path / "invalid.sqlite3",
            manifest,
            tokenizer,
            envelope,
            _NoCallAdapter(),
            mode="invalid",  # type: ignore[arg-type]
        )
    _assert_reason(mode_error, "ledger_mode")
