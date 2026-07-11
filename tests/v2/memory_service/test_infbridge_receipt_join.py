# SPDX-License-Identifier: Apache-2.0

"""Answer-blind joins from raw Memory sidecars to sealed model receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from examples.memory_service import infbridge_model_adapter as adapter_module
from examples.memory_service import infbridge_receipt_join as receipt_join
from examples.memory_service import infbridge_run_analyzer as run_analyzer
from examples.memory_service import infbridge_run_ledger as ledger_module
from examples.memory_service import infbridge_sidecar_join as sidecar_join
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    InfBridgeModelAdapter,
)
from examples.memory_service.infbridge_run_ledger import (
    LedgerArtifactCommitmentV1,
    RunLedgerReceiptSnapshotV1,
    RunLedgerSessionV1,
    RunLedgerSnapshotV1,
    load_run_ledger,
    load_run_ledger_receipt_snapshot_v1,
)
from tests.v2.memory_service.test_infbridge_sidecar_join import (
    _build_live_case,
    _exchange_with,
    _LiveModelCase,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_PRIVATE_ANSWERS = tuple(
    f"PRIVATE-RECEIPT-JOIN-ANSWER-{index}".encode("ascii") for index in range(6)
)
_SLOT_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-slot-join-v1\0"
_CASE_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-case-join-v1\0"
_RUN_JOIN_DOMAIN = b"areal-memory-model-sidecar-ledger-run-join-v1\0"
_POLICY = "live-sidecar-sealed-ledger-answer-content-blind-consistency-only-v1"


@dataclass(frozen=True, slots=True)
class _ReceiptJoinCase:
    live: _LiveModelCase
    ledger_database_path: Path
    full_snapshot: RunLedgerSnapshotV1
    receipt_snapshot: RunLedgerReceiptSnapshotV1
    validated_case: sidecar_join.ValidatedModelCaseSidecarV1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sglang_response(output: bytes) -> dict[str, Any]:
    return {
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                (-0.01 * (index + 1), token_id) for index, token_id in enumerate(output)
            ],
        }
    }


async def _success_executions(live: _LiveModelCase) -> tuple[object, ...]:
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://sidecar-model.test/",
        pause_state=PauseState(),
        request_timeout=7.0,
        max_resubmit_retries=2,
        resubmit_wait=0.0,
        version=41,
    )
    executions: list[object] = []
    try:
        adapter = InfBridgeModelAdapter(
            live.manifest,
            live.tokenizer,
            live.envelope,
            bridge,
        )
        for slot_index, arm_call in enumerate(live.registration.arm_calls):
            bridge._send_request = AsyncMock(
                return_value=_sglang_response(_PRIVATE_ANSWERS[slot_index])
            )
            executions.append(await adapter.submit(0, arm_call.arm))
    finally:
        await bridge.aclose()
    return tuple(executions)


def _seal_planned_suffix_as_attrition(
    database_path: Path,
    snapshot: RunLedgerSnapshotV1,
) -> None:
    slots = list(snapshot.slots)
    updates: list[tuple[object, ...]] = []
    for slot_index in range(len(helpfulness.MODEL_ARMS), len(slots)):
        slot = slots[slot_index]
        assert slot.state == "PLANNED"
        leaf_sha256 = ledger_module._slot_leaf_sha256(
            slot_index=slot.plan.slot_index,
            plan_sha256=slot.plan_sha256,
            state="ATTRITION",
            attempt_count=1,
            receipt_bytes=None,
            trace_bytes=None,
            response_evidence_bytes=None,
            decoded_response_utf8=None,
            terminal_reason="generation_failure",
        )
        slots[slot_index] = replace(
            slot,
            state="ATTRITION",
            attempt_count=1,
            terminal_reason="generation_failure",
            leaf_sha256=leaf_sha256,
        )
        updates.append(
            (
                "ATTRITION",
                1,
                "generation_failure",
                leaf_sha256,
                slot_index,
            )
        )
    sealed_slots = tuple(slots)
    run_root_sha256 = ledger_module._run_root_sha256(
        run_id=snapshot.run_id,
        manifest_sha256=snapshot.manifest_sha256,
        run_envelope_sha256=snapshot.run_envelope_sha256,
        status="SEALED",
        seal_kind="complete_with_attrition",
        slots=sealed_slots,
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "UPDATE run_ledger_slots SET state = ?, attempt_count = ?, "
            "terminal_reason = ?, leaf_sha256 = ? WHERE slot_index = ?",
            updates,
        )
        connection.execute(
            "UPDATE run_ledger_header SET status = 'SEALED', "
            "seal_kind = 'complete_with_attrition', run_root_sha256 = ? "
            "WHERE singleton = 1",
            (run_root_sha256,),
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _build_join_case(root: Path) -> _ReceiptJoinCase:
    live = _build_live_case(root / "sidecar")
    database_path = root / "model-run-ledger.sqlite3"
    session = RunLedgerSessionV1.create(
        database_path,
        live.manifest,
        live.tokenizer,
        live.envelope,
    )
    executions = asyncio.run(_success_executions(live))
    for slot_index, execution in enumerate(executions):
        session.mark_started(slot_index)
        if slot_index % 2 == 0:
            session.record_success(slot_index, execution)  # type: ignore[arg-type]
        else:
            session.record_attrition(slot_index, "generation_failure")
    _seal_planned_suffix_as_attrition(database_path, session.snapshot)
    full_snapshot = load_run_ledger(
        database_path,
        live.manifest,
        live.tokenizer,
        live.envelope,
    )
    receipt_snapshot = load_run_ledger_receipt_snapshot_v1(
        database_path,
        live.manifest,
        live.tokenizer,
        live.envelope,
    )
    assert full_snapshot.status == "SEALED"
    assert full_snapshot.seal_kind == "complete_with_attrition"
    assert tuple(slot.state for slot in full_snapshot.slots[:6]) == (
        "SUCCEEDED",
        "ATTRITION",
        "SUCCEEDED",
        "ATTRITION",
        "SUCCEEDED",
        "ATTRITION",
    )
    return _ReceiptJoinCase(
        live=live,
        ledger_database_path=database_path,
        full_snapshot=full_snapshot,
        receipt_snapshot=receipt_snapshot,
        validated_case=sidecar_join.validate_live_model_case_sidecar_v1(
            live.manifest,
            live.tokenizer,
            live.envelope,
            live.sidecar,
        ),
    )


@pytest.fixture(scope="module")
def receipt_join_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> _ReceiptJoinCase:
    return _build_join_case(tmp_path_factory.mktemp("infbridge-receipt-join"))


def _commitment_value(
    commitment: LedgerArtifactCommitmentV1 | None,
) -> dict[str, object] | None:
    if commitment is None:
        return None
    return {
        "byte_count": commitment.byte_count,
        "sha256": commitment.sha256,
    }


def _slot_value(slot: receipt_join.ModelCaseReceiptSlotV1) -> dict[str, object]:
    return {
        "adapter_input_token_ids_sha256": slot.adapter_input_token_ids_sha256,
        "arm": slot.arm,
        "attempt_count": slot.attempt_count,
        "case_index": slot.case_index,
        "consumer_history_length": slot.consumer_history_length,
        "consumer_query_sha256": slot.consumer_query_sha256,
        "decoded_response_commitment": _commitment_value(
            slot.decoded_response_commitment
        ),
        "evaluator_input_token_ids_sha256": (slot.evaluator_input_token_ids_sha256),
        "input_token_count": slot.input_token_count,
        "ledger_leaf_sha256": slot.ledger_leaf_sha256,
        "ledger_state": slot.ledger_state,
        "observation_leaf_sha256": slot.observation_leaf_sha256,
        "plan_sha256": slot.plan_sha256,
        "policy": _POLICY,
        "prompt_context_end": slot.prompt_context_end,
        "prompt_context_start": slot.prompt_context_start,
        "receipt_commitment": _commitment_value(slot.receipt_commitment),
        "rendered_context_sha256": slot.rendered_context_sha256,
        "rendered_context_utf8_bytes": slot.rendered_context_utf8_bytes,
        "request_id": slot.request_id,
        "response_evidence_commitment": _commitment_value(
            slot.response_evidence_commitment
        ),
        "schema_version": 1,
        "slot_index": slot.slot_index,
        "source_kind": slot.source_kind,
        "submitted_prompt_sha256": slot.submitted_prompt_sha256,
        "terminal_reason": slot.terminal_reason,
        "trace_commitment": _commitment_value(slot.trace_commitment),
    }


def _recompute_slot_join_sha256(
    slot: receipt_join.ModelCaseReceiptSlotV1,
) -> str:
    return hashlib.sha256(
        _SLOT_JOIN_DOMAIN + _canonical_bytes(_slot_value(slot))
    ).hexdigest()


def _recompute_case_join_sha256(
    joined: receipt_join.ModelCaseReceiptJoinV1,
) -> str:
    value = {
        "attrition_count": joined.attrition_count,
        "case_index": joined.case_index,
        "capture_leaf_sha256": joined.capture_leaf_sha256,
        "envelope_sha256": joined.envelope_sha256,
        "ledger_policy": joined.ledger_policy,
        "ledger_projection_policy": joined.ledger_projection_policy,
        "ledger_run_id": joined.ledger_run_id,
        "ledger_run_root_sha256": joined.ledger_run_root_sha256,
        "ledger_seal_kind": joined.ledger_seal_kind,
        "manifest_sha256": joined.manifest_sha256,
        "observation_leaf_sha256s": list(joined.observation_leaf_sha256s),
        "policy": joined.policy,
        "probe_leaf_sha256": joined.probe_leaf_sha256,
        "schema_version": joined.schema_version,
        "sidecar_case_root_sha256": joined.sidecar_case_root_sha256,
        "sidecar_policy": joined.sidecar_policy,
        "slot_count": len(joined.slots),
        "slot_join_sha256s": [slot.slot_join_sha256 for slot in joined.slots],
        "succeeded_count": joined.succeeded_count,
    }
    return hashlib.sha256(_CASE_JOIN_DOMAIN + _canonical_bytes(value)).hexdigest()


def _recompute_run_join_sha256(
    joined: receipt_join.ModelRunReceiptJoinV1,
) -> str:
    value = {
        "attrition_count": joined.attrition_count,
        "case_count": joined.case_count,
        "case_receipt_root_sha256s": [
            case.case_receipt_root_sha256 for case in joined.cases
        ],
        "envelope_sha256": joined.envelope_sha256,
        "ledger_policy": joined.ledger_policy,
        "ledger_projection_policy": joined.ledger_projection_policy,
        "ledger_run_id": joined.ledger_run_id,
        "ledger_run_root_sha256": joined.ledger_run_root_sha256,
        "ledger_seal_kind": joined.ledger_seal_kind,
        "manifest_sha256": joined.manifest_sha256,
        "policy": joined.policy,
        "schema_version": joined.schema_version,
        "sidecar_aggregate_root_sha256": (joined.sidecar_aggregate_root_sha256),
        "sidecar_policy": joined.sidecar_policy,
        "slot_count": joined.slot_count,
        "succeeded_count": joined.succeeded_count,
    }
    return hashlib.sha256(_RUN_JOIN_DOMAIN + _canonical_bytes(value)).hexdigest()


def _assert_no_byte_preimages(value: object) -> None:
    assert not isinstance(value, (bytes, bytearray, memoryview))
    if is_dataclass(value):
        for field in fields(value):
            _assert_no_byte_preimages(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_byte_preimages(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_byte_preimages(key)
            _assert_no_byte_preimages(item)


def _assert_join_error(
    error: pytest.ExceptionInfo[receipt_join.InfBridgeReceiptJoinError],
    reason: str,
) -> None:
    assert type(error.value) is receipt_join.InfBridgeReceiptJoinError
    assert error.value.reason == reason
    assert str(error.value) == reason


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _join_real_case(
    fixture: _ReceiptJoinCase,
    *,
    validated: sidecar_join.ValidatedModelCaseSidecarV1 | None = None,
) -> receipt_join.ModelCaseReceiptJoinV1:
    return receipt_join._join_validated_model_case_receipts_v1(
        fixture.live.manifest,
        fixture.live.envelope,
        fixture.validated_case if validated is None else validated,
        fixture.receipt_snapshot,
    )


def _synthetic_aggregate(
    fixture: _ReceiptJoinCase,
    *,
    case_zero: sidecar_join.ValidatedModelCaseSidecarV1 | None = None,
) -> sidecar_join.ValidatedModelSidecarAggregateV1:
    """Build typed orchestration data; single-case validation stays real above."""

    base = fixture.validated_case if case_zero is None else case_zero
    cases: list[sidecar_join.ValidatedModelCaseSidecarV1] = []
    for case_index, registration in enumerate(fixture.live.manifest.cases):
        template = base if case_index == 0 else fixture.validated_case
        capture = replace(
            template.leaves[0],
            logical_index=case_index,
            case_index=case_index,
            arm=None,
            opaque_execution_token_hex=None,
            leaf_sha256=(
                template.leaves[0].leaf_sha256
                if case_index == 0
                else _digest(f"synthetic-capture-{case_index}")
            ),
        )
        observations = tuple(
            replace(
                template.leaves[offset + 1],
                logical_index=case_index * len(helpfulness.MODEL_ARMS) + offset,
                case_index=case_index,
                arm=arm_call.arm,
                opaque_execution_token_hex=(
                    f"{(1 << 127) + case_index * 16 + offset:032x}"
                ),
                leaf_sha256=(
                    template.leaves[offset + 1].leaf_sha256
                    if case_index == 0
                    else _digest(f"synthetic-observation-{case_index}-{offset}")
                ),
            )
            for offset, arm_call in enumerate(registration.arm_calls)
        )
        probe = replace(
            template.leaves[-1],
            logical_index=(
                helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS) + case_index
            ),
            case_index=case_index,
            arm=None,
            opaque_execution_token_hex=f"{(1 << 128) + case_index:033x}",
            leaf_sha256=(
                template.leaves[-1].leaf_sha256
                if case_index == 0
                else _digest(f"synthetic-probe-{case_index}")
            ),
        )
        case_root = (
            template.commitment.case_root_sha256
            if case_index == 0
            else _digest(f"synthetic-case-root-{case_index}")
        )
        commitment = replace(
            template.commitment,
            case_index=case_index,
            capture_leaf_sha256=capture.leaf_sha256,
            observation_leaf_sha256s=tuple(leaf.leaf_sha256 for leaf in observations),
            probe_leaf_sha256=probe.leaf_sha256,
            case_root_sha256=case_root,
        )
        cases.append(
            replace(
                template,
                commitment=commitment,
                leaves=(capture, *observations, probe),
            )
        )
    typed_cases = tuple(cases)
    case_roots = tuple(case.commitment.case_root_sha256 for case in typed_cases)
    aggregate_value = {
        "capture_count": helpfulness.MODEL_CASE_COUNT,
        "case_count": helpfulness.MODEL_CASE_COUNT,
        "case_root_sha256s": list(case_roots),
        "envelope_sha256": base.commitment.envelope_sha256,
        "leaf_count": helpfulness.MODEL_CASE_COUNT * (len(helpfulness.MODEL_ARMS) + 2),
        "manifest_sha256": base.commitment.manifest_sha256,
        "observation_count": helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS),
        "policy": base.commitment.policy,
        "probe_count": helpfulness.MODEL_CASE_COUNT,
        "schema_version": 1,
    }
    aggregate_root = hashlib.sha256(
        b"areal-memory-model-sidecar-aggregate-root-v1\0"
        + _canonical_bytes(aggregate_value)
    ).hexdigest()
    return sidecar_join.ValidatedModelSidecarAggregateV1(
        commitment=sidecar_join.ModelSidecarAggregateCommitmentV1(
            schema_version=1,
            policy=base.commitment.policy,
            manifest_sha256=base.commitment.manifest_sha256,
            envelope_sha256=base.commitment.envelope_sha256,
            case_count=helpfulness.MODEL_CASE_COUNT,
            capture_count=helpfulness.MODEL_CASE_COUNT,
            observation_count=(
                helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS)
            ),
            probe_count=helpfulness.MODEL_CASE_COUNT,
            leaf_count=(
                helpfulness.MODEL_CASE_COUNT * (len(helpfulness.MODEL_ARMS) + 2)
            ),
            case_root_sha256s=case_roots,
            aggregate_root_sha256=aggregate_root,
        ),
        cases=typed_cases,
    )


def _raw_sidecars(
    fixture: _ReceiptJoinCase,
) -> tuple[sidecar_join.ModelCaseSidecarV1, ...]:
    # The aggregate validator is replaced only in orchestration tests.  The
    # public surface still receives raw sidecar objects, never reports.
    return (fixture.live.sidecar,) * helpfulness.MODEL_CASE_COUNT


def _install_public_inputs(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _ReceiptJoinCase,
    aggregates: tuple[sidecar_join.ValidatedModelSidecarAggregateV1, ...],
    events: list[str],
    *,
    projection: RunLedgerReceiptSnapshotV1 | None = None,
) -> None:
    aggregate_calls = 0

    def validate_aggregate(*args: object) -> object:
        nonlocal aggregate_calls
        assert args == (
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
        events.append(f"aggregate-{aggregate_calls + 1}")
        value = aggregates[aggregate_calls]
        aggregate_calls += 1
        return value

    def load_projection(*args: object) -> RunLedgerReceiptSnapshotV1:
        assert args == (
            fixture.ledger_database_path,
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
        )
        events.append("loader")
        return fixture.receipt_snapshot if projection is None else projection

    monkeypatch.setattr(
        receipt_join.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        validate_aggregate,
    )
    monkeypatch.setattr(
        receipt_join,
        "load_run_ledger_receipt_snapshot_v1",
        load_projection,
    )


def _validate_public(
    fixture: _ReceiptJoinCase,
) -> receipt_join.ModelRunReceiptJoinV1:
    return receipt_join.validate_live_model_run_receipt_join_v1(
        fixture.ledger_database_path,
        fixture.live.manifest,
        fixture.live.tokenizer,
        fixture.live.envelope,
        _raw_sidecars(fixture),
    )


def test_pure_case_join_binds_real_six_slot_chain_and_two_token_hash_schemes(
    receipt_join_case: _ReceiptJoinCase,
) -> None:
    fixture = receipt_join_case
    live = fixture.live
    joined = _join_real_case(fixture)

    assert type(joined) is receipt_join.ModelCaseReceiptJoinV1
    assert joined.policy == _POLICY
    assert joined.sidecar_policy == fixture.validated_case.commitment.policy
    assert joined.ledger_projection_policy == (
        fixture.receipt_snapshot.projection_policy
    )
    assert joined.ledger_policy == fixture.receipt_snapshot.ledger_policy
    assert (joined.succeeded_count, joined.attrition_count) == (3, 3)
    assert len(joined.slots) == len(helpfulness.MODEL_ARMS) == 6

    for offset, slot in enumerate(joined.slots):
        arm_call = live.registration.arm_calls[offset]
        prepared = arm_call.prepared_call
        ledger_slot = fixture.receipt_snapshot.slots[offset]
        evaluator_bytes = json.dumps(
            list(prepared.input_token_ids), separators=(",", ":")
        ).encode("ascii")
        adapter_bytes = _canonical_bytes(list(prepared.input_token_ids))
        evaluator_hash = hashlib.sha256(evaluator_bytes).hexdigest()
        adapter_hash = hashlib.sha256(
            b"areal-memory-token-ids-v2\0" + adapter_bytes
        ).hexdigest()
        assert (slot.slot_index, slot.case_index, slot.arm) == (
            offset,
            0,
            arm_call.arm,
        )
        assert slot.evaluator_input_token_ids_sha256 == evaluator_hash
        assert slot.adapter_input_token_ids_sha256 == adapter_hash
        assert evaluator_hash != adapter_hash
        assert slot.observation_leaf_sha256 == (
            fixture.validated_case.leaves[offset + 1].leaf_sha256
        )
        assert slot.plan_sha256 == ledger_slot.plan_sha256
        assert slot.ledger_state == ledger_slot.state
        assert slot.attempt_count == 1
        assert slot.slot_join_sha256 == _recompute_slot_join_sha256(slot)

    by_arm = {slot.arm: slot for slot in joined.slots}
    current = by_arm["current_release"]
    oracle = by_arm["oracle"]
    current_call = next(
        call for call in live.registration.arm_calls if call.arm == "current_release"
    )
    oracle_call = next(
        call for call in live.registration.arm_calls if call.arm == "oracle"
    )
    assert current_call.prepared_call.prompt == oracle_call.prepared_call.prompt
    assert current.submitted_prompt_sha256 == oracle.submitted_prompt_sha256
    assert current.evaluator_input_token_ids_sha256 == (
        oracle.evaluator_input_token_ids_sha256
    )
    assert current.adapter_input_token_ids_sha256 == (
        oracle.adapter_input_token_ids_sha256
    )
    assert (current.source_kind, oracle.source_kind) == ("release", "oracle")
    assert current.slot_index != oracle.slot_index
    assert current.observation_leaf_sha256 != oracle.observation_leaf_sha256
    assert current.slot_join_sha256 != oracle.slot_join_sha256
    assert joined.case_receipt_root_sha256 == _recompute_case_join_sha256(joined)
    _assert_no_byte_preimages(joined)


def test_public_gate_validates_all_sidecars_then_loads_once_then_revalidates(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    aggregate = _synthetic_aggregate(fixture)
    events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (aggregate, aggregate),
        events,
    )
    original_join = receipt_join._join_validated_model_case_receipts_v1

    def tracked_join(*args: object, **kwargs: object) -> object:
        validated = args[2]
        events.append(f"join-{validated.commitment.case_index}")  # type: ignore[union-attr]
        return original_join(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        receipt_join,
        "_join_validated_model_case_receipts_v1",
        tracked_join,
    )

    joined = _validate_public(fixture)

    assert events == [
        "aggregate-1",
        "loader",
        "aggregate-2",
        *(f"join-{case_index}" for case_index in range(64)),
    ]
    assert type(joined) is receipt_join.ModelRunReceiptJoinV1
    assert joined.policy == _POLICY
    assert (joined.case_count, joined.slot_count) == (64, 384)
    assert (joined.succeeded_count, joined.attrition_count) == (3, 381)
    assert tuple(case.case_index for case in joined.cases) == tuple(range(64))
    assert tuple(slot.slot_index for case in joined.cases for slot in case.slots) == (
        tuple(range(384))
    )
    for case in joined.cases:
        for slot in case.slots:
            assert slot.slot_join_sha256 == _recompute_slot_join_sha256(slot)
        assert case.case_receipt_root_sha256 == _recompute_case_join_sha256(case)
    assert joined.run_receipt_root_sha256 == _recompute_run_join_sha256(joined)
    _assert_no_byte_preimages(joined)
    assert all(
        answer.decode("ascii") not in repr(joined) for answer in _PRIVATE_ANSWERS
    )


def test_invalid_64th_sidecar_never_touches_path_or_loader(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    path_calls: list[str] = []
    loader_calls: list[str] = []

    class ExplodingPath:
        def __fspath__(self) -> str:
            path_calls.append("path")
            raise AssertionError("ledger path touched before global sidecar gate")

    def forbidden_loader(*_args: object) -> None:
        loader_calls.append("loader")
        raise AssertionError("loader called for invalid global sidecars")

    monkeypatch.setattr(
        receipt_join,
        "load_run_ledger_receipt_snapshot_v1",
        forbidden_loader,
    )
    invalid = (fixture.live.sidecar,) * 63 + (object(),)
    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as error:
        receipt_join.validate_live_model_run_receipt_join_v1(
            ExplodingPath(),
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            invalid,  # type: ignore[arg-type]
        )
    _assert_join_error(error, "sidecar_invalid")
    assert path_calls == []
    assert loader_calls == []


def test_second_aggregate_drift_rejects_the_whole_run_after_one_load(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    first = _synthetic_aggregate(fixture)
    drifted = replace(
        first,
        commitment=replace(
            first.commitment,
            aggregate_root_sha256=_digest("post-load-sidecar-drift"),
        ),
    )
    events: list[str] = []
    _install_public_inputs(monkeypatch, fixture, (first, drifted), events)

    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as error:
        _validate_public(fixture)
    _assert_join_error(error, "sidecar_invalid")
    assert events == ["aggregate-1", "loader", "aggregate-2"]


def test_public_api_rejects_constructible_sidecar_and_ledger_reports(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    aggregate = _synthetic_aggregate(fixture)
    path_calls: list[str] = []

    class ExplodingPath:
        def __fspath__(self) -> str:
            path_calls.append("path")
            raise AssertionError("path touched for a prevalidated sidecar report")

    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as sidecar_error:
        receipt_join.validate_live_model_run_receipt_join_v1(
            ExplodingPath(),
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            aggregate,  # type: ignore[arg-type]
        )
    _assert_join_error(sidecar_error, "sidecar_invalid")
    assert path_calls == []

    aggregate_calls = 0

    def validate_aggregate(*_args: object) -> object:
        nonlocal aggregate_calls
        aggregate_calls += 1
        return aggregate

    monkeypatch.setattr(
        receipt_join.sidecars,
        "validate_live_model_sidecar_aggregate_v1",
        validate_aggregate,
    )
    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as ledger_error:
        receipt_join.validate_live_model_run_receipt_join_v1(
            fixture.receipt_snapshot,  # type: ignore[arg-type]
            fixture.live.manifest,
            fixture.live.tokenizer,
            fixture.live.envelope,
            _raw_sidecars(fixture),
        )
    _assert_join_error(ledger_error, "ledger_invalid")
    assert aggregate_calls == 1


def test_cross_chain_mismatch_from_loader_fails_the_global_run(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    aggregate = _synthetic_aggregate(fixture)
    first = fixture.receipt_snapshot.slots[0]
    second = fixture.receipt_snapshot.slots[1]
    forged_first = replace(first, plan=second.plan)
    projection = replace(
        fixture.receipt_snapshot,
        slots=(forged_first, *fixture.receipt_snapshot.slots[1:]),
    )
    events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (aggregate, aggregate),
        events,
        projection=projection,
    )

    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as error:
        _validate_public(fixture)
    _assert_join_error(error, "receipt_chain")
    assert events == ["aggregate-1", "loader", "aggregate-2"]


def test_private_join_rechecks_projection_header_contract(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    aggregate = _synthetic_aggregate(fixture)
    projection = replace(fixture.receipt_snapshot, call_count=383)
    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as private_error:
        receipt_join._join_validated_model_case_receipts_v1(
            fixture.live.manifest,
            fixture.live.envelope,
            fixture.validated_case,
            projection,
        )
    _assert_join_error(private_error, "receipt_chain")

    events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (aggregate, aggregate),
        events,
        projection=projection,
    )

    with pytest.raises(receipt_join.InfBridgeReceiptJoinError) as error:
        _validate_public(fixture)

    _assert_join_error(error, "receipt_chain")
    assert events == ["aggregate-1", "loader", "aggregate-2"]


def test_release_found_changes_case_aggregate_and_run_roots_not_slot_joins(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    live = fixture.live
    found_response = replace(
        live.probe_response,
        probe=replace(live.probe_response.probe, outcome="release_found"),
    )
    found_sidecar = replace(
        live.sidecar,
        probe=replace(
            live.sidecar.probe,
            exchange=_exchange_with(
                live.sidecar.probe.exchange,
                response=found_response,
            ),
        ),
    )
    found_case = sidecar_join.validate_live_model_case_sidecar_v1(
        live.manifest,
        live.tokenizer,
        live.envelope,
        found_sidecar,
    )
    baseline_aggregate = _synthetic_aggregate(fixture)
    found_aggregate = _synthetic_aggregate(fixture, case_zero=found_case)

    baseline_events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (baseline_aggregate, baseline_aggregate),
        baseline_events,
    )
    baseline = _validate_public(fixture)
    found_events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (found_aggregate, found_aggregate),
        found_events,
    )
    found = _validate_public(fixture)

    assert baseline_events == ["aggregate-1", "loader", "aggregate-2"]
    assert found_events == ["aggregate-1", "loader", "aggregate-2"]
    assert baseline.sidecar_aggregate_root_sha256 != (
        found.sidecar_aggregate_root_sha256
    )
    assert baseline.cases[0].probe_leaf_sha256 != found.cases[0].probe_leaf_sha256
    assert baseline.cases[0].sidecar_case_root_sha256 != (
        found.cases[0].sidecar_case_root_sha256
    )
    assert tuple(slot.slot_join_sha256 for slot in baseline.cases[0].slots) == (
        tuple(slot.slot_join_sha256 for slot in found.cases[0].slots)
    )
    assert baseline.cases[0].case_receipt_root_sha256 != (
        found.cases[0].case_receipt_root_sha256
    )
    assert baseline.run_receipt_root_sha256 != found.run_receipt_root_sha256


def test_public_join_never_decodes_replays_normalizes_or_scores_answers(
    receipt_join_case: _ReceiptJoinCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = receipt_join_case
    aggregate = _synthetic_aggregate(fixture)
    events: list[str] = []
    _install_public_inputs(
        monkeypatch,
        fixture,
        (aggregate, aggregate),
        events,
    )
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            forbidden_calls.append(name)
            raise AssertionError(f"receipt join invoked forbidden operation: {name}")

        return fail

    monkeypatch.setattr(type(fixture.live.tokenizer), "decode", forbidden("decode"))
    monkeypatch.setattr(
        ledger_module,
        "audited_model_call_execution_v2_from_artifacts",
        forbidden("artifact_replay"),
    )
    monkeypatch.setattr(
        run_analyzer,
        "recover_infbridge_model_dry_run_v1",
        forbidden("dry_run_recovery"),
    )
    monkeypatch.setattr(helpfulness, "normalize_response", forbidden("normalize"))
    monkeypatch.setattr(helpfulness, "utility", forbidden("utility"))
    monkeypatch.setattr(
        helpfulness,
        "parent_join_and_score",
        forbidden("score"),
    )
    monkeypatch.setattr(
        helpfulness,
        "analyze_model_run",
        forbidden("analyze"),
    )
    monkeypatch.setattr(
        adapter_module,
        "audited_model_call_execution_v2_from_artifacts",
        forbidden("adapter_replay"),
        raising=False,
    )

    joined = _validate_public(fixture)

    assert events == ["aggregate-1", "loader", "aggregate-2"]
    assert forbidden_calls == []
    _assert_no_byte_preimages(joined)
