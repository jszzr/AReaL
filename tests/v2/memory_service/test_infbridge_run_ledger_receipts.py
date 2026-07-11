# SPDX-License-Identifier: Apache-2.0

"""Answer-content-blind receipt projection tests for the model-run ledger."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from examples.memory_service import infbridge_run_ledger as ledger_module
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    DecoderAuditMaterialV2,
    InfBridgeModelAdapter,
    RunEnvelopeV2,
    prepare_infbridge_run_envelope_v2,
)
from examples.memory_service.infbridge_run_ledger import (
    LedgerArtifactCommitmentV1,
    RunLedgerError,
    RunLedgerReceiptSlotV1,
    RunLedgerReceiptSnapshotV1,
    RunLedgerSessionV1,
    RunLedgerSnapshotV1,
    load_run_ledger,
    load_run_ledger_receipt_snapshot_v1,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_PRIVATE_ANSWER = b"ANSWER-CONTENT-MUST-STAY-PRIVATE"
_SUCCESS_ARTIFACT_COLUMNS = (
    "receipt_bytes",
    "trace_bytes",
    "response_evidence_bytes",
    "decoded_response_utf8",
)
_LEDGER_LEAF_DOMAIN = b"areal-memory-run-ledger-leaf-v1\0"
_LEDGER_ROOT_HEADER_DOMAIN = b"areal-memory-run-ledger-root-header-v1\0"
_LEDGER_ROOT_FOLD_DOMAIN = b"areal-memory-run-ledger-root-fold-v1\0"


class _ReceiptByteTokenizer:
    _ARTIFACT_BYTES = b'receipt-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'receipt-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def __init__(self) -> None:
        self.fail_decode = False

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="receipt-byte-tokenizer-v2",
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
        if self.fail_decode:
            raise AssertionError("receipt projection decoded model-answer tokens")
        if skip_special_tokens is not True or clean_up_tokenization_spaces is not False:
            raise ValueError("receipt decode policy drift")
        return bytes(token_ids).decode("utf-8")


class _SealedLedgerFixture:
    def __init__(
        self,
        database_path: Path,
        manifest: helpfulness.ModelRunManifest,
        tokenizer: _ReceiptByteTokenizer,
        envelope: RunEnvelopeV2,
        full_snapshot: RunLedgerSnapshotV1,
    ) -> None:
        self.database_path = database_path
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.envelope = envelope
        self.full_snapshot = full_snapshot


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


async def _one_success_execution(
    manifest: helpfulness.ModelRunManifest,
    tokenizer: _ReceiptByteTokenizer,
    envelope: RunEnvelopeV2,
    bridge: InfBridge,
):
    original_send = bridge._send_request
    bridge._send_request = AsyncMock(return_value=_sglang_response(_PRIVATE_ANSWER))
    try:
        return await InfBridgeModelAdapter(
            manifest,
            tokenizer,
            envelope,
            bridge,
        ).submit(0, envelope.call_plans[0].arm)
    finally:
        bridge._send_request = original_send


def _seal_remaining_slots_as_attrition(
    database_path: Path,
    snapshot: RunLedgerSnapshotV1,
) -> None:
    slots = list(snapshot.slots)
    updates: list[tuple[object, ...]] = []
    for slot_index in range(1, len(slots)):
        slot = slots[slot_index]
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


@pytest.fixture(scope="module")
def sealed_ledger(
    tmp_path_factory: pytest.TempPathFactory,
) -> _SealedLedgerFixture:
    tokenizer = _ReceiptByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="7" * 40,
        evaluator_commit_sha="8" * 40,
        model_id="receipt-projection-test-model",
        model_weights_sha256="9" * 64,
        tokenizer_id="receipt-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    manifest = prepared.manifest
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://receipt-model.test/",
        pause_state=PauseState(),
        request_timeout=9.0,
        max_resubmit_retries=3,
        resubmit_wait=0.0,
        version=31,
    )
    envelope = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=64,
    )
    database_path = tmp_path_factory.mktemp("receipt-ledger") / "sealed.sqlite3"
    session = RunLedgerSessionV1.create(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    session.mark_started(0)
    execution = asyncio.run(
        _one_success_execution(manifest, tokenizer, envelope, bridge)
    )
    success = session.record_success(0, execution)
    _seal_remaining_slots_as_attrition(database_path, success)
    full_snapshot = load_run_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    assert full_snapshot.status == "SEALED"
    assert full_snapshot.seal_kind == "complete_with_attrition"
    assert full_snapshot.slots[0].decoded_response_utf8 == _PRIVATE_ANSWER
    fixture = _SealedLedgerFixture(
        database_path,
        manifest,
        tokenizer,
        envelope,
        full_snapshot,
    )
    yield fixture
    asyncio.run(bridge.aclose())


def _copy_database(tmp_path: Path, fixture: _SealedLedgerFixture, name: str) -> Path:
    target = tmp_path / name
    shutil.copyfile(fixture.database_path, target)
    os.chmod(target, 0o600)
    return target


def _success_artifacts(database_path: Path) -> dict[str, bytes]:
    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT receipt_bytes, trace_bytes, response_evidence_bytes, "
            "decoded_response_utf8 FROM run_ledger_slots WHERE slot_index = 0"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert all(type(value) is bytes for value in row)
    return dict(zip(_SUCCESS_ARTIFACT_COLUMNS, row, strict=True))


def _rehash_success_slot_and_run_root(
    database_path: Path,
    fixture: _SealedLedgerFixture,
) -> None:
    artifacts = _success_artifacts(database_path)
    original = fixture.full_snapshot.slots[0]
    leaf_sha256 = ledger_module._slot_leaf_sha256(
        slot_index=original.plan.slot_index,
        plan_sha256=original.plan_sha256,
        state="SUCCEEDED",
        attempt_count=1,
        receipt_bytes=artifacts["receipt_bytes"],
        trace_bytes=artifacts["trace_bytes"],
        response_evidence_bytes=artifacts["response_evidence_bytes"],
        decoded_response_utf8=artifacts["decoded_response_utf8"],
        terminal_reason=None,
    )
    changed = replace(
        original,
        receipt_bytes=artifacts["receipt_bytes"],
        trace_bytes=artifacts["trace_bytes"],
        response_evidence_bytes=artifacts["response_evidence_bytes"],
        decoded_response_utf8=artifacts["decoded_response_utf8"],
        leaf_sha256=leaf_sha256,
    )
    run_root_sha256 = ledger_module._run_root_sha256(
        run_id=fixture.full_snapshot.run_id,
        manifest_sha256=fixture.full_snapshot.manifest_sha256,
        run_envelope_sha256=fixture.full_snapshot.run_envelope_sha256,
        status="SEALED",
        seal_kind="complete_with_attrition",
        slots=(changed, *fixture.full_snapshot.slots[1:]),
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE run_ledger_slots SET leaf_sha256 = ? WHERE slot_index = 0",
            (leaf_sha256,),
        )
        connection.execute(
            "UPDATE run_ledger_header SET run_root_sha256 = ? WHERE singleton = 1",
            (run_root_sha256,),
        )
    finally:
        connection.close()


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


def _assert_ledger_corruption(error: pytest.ExceptionInfo[RunLedgerError]) -> None:
    assert type(error.value) is RunLedgerError
    assert error.value.reason == "ledger_corruption"


def _commitment_value(
    commitment: LedgerArtifactCommitmentV1 | None,
) -> dict[str, object] | None:
    if commitment is None:
        return None
    return {
        "byte_count": commitment.byte_count,
        "sha256": commitment.sha256,
    }


def _recompute_projection_run_root(
    projection: RunLedgerReceiptSnapshotV1,
) -> str:
    leaf_hashes: list[str] = []
    for slot in projection.slots:
        leaf_value = {
            "attempt_count": slot.attempt_count,
            "decoded_response_utf8": _commitment_value(
                slot.decoded_response_commitment
            ),
            "plan_sha256": slot.plan_sha256,
            "receipt": _commitment_value(slot.receipt_commitment),
            "response_evidence": _commitment_value(slot.response_evidence_commitment),
            "slot_index": slot.plan.slot_index,
            "state": slot.state,
            "terminal_reason": slot.terminal_reason,
            "trace": _commitment_value(slot.trace_commitment),
        }
        leaf_sha256 = hashlib.sha256(
            _LEDGER_LEAF_DOMAIN + _canonical_bytes(leaf_value)
        ).hexdigest()
        assert leaf_sha256 == slot.ledger_leaf_sha256
        leaf_hashes.append(leaf_sha256)
    header = {
        "call_count": projection.call_count,
        "ledger_policy": projection.ledger_policy,
        "manifest_sha256": projection.manifest_sha256,
        "run_envelope_sha256": projection.run_envelope_sha256,
        "run_id": projection.run_id,
        "schema_version": 1,
        "seal_kind": projection.seal_kind,
        "status": "SEALED",
    }
    root = hashlib.sha256(
        _LEDGER_ROOT_HEADER_DOMAIN + _canonical_bytes(header)
    ).digest()
    for slot_index, leaf_sha256 in enumerate(leaf_hashes):
        root = hashlib.sha256(
            _LEDGER_ROOT_FOLD_DOMAIN
            + root
            + slot_index.to_bytes(8, "big", signed=False)
            + bytes.fromhex(leaf_sha256)
        ).digest()
    return root.hex()


def test_sealed_projection_preserves_identity_root_and_384_receipt_slots(
    sealed_ledger: _SealedLedgerFixture,
) -> None:
    projection = load_run_ledger_receipt_snapshot_v1(
        sealed_ledger.database_path,
        sealed_ledger.manifest,
        sealed_ledger.tokenizer,
        sealed_ledger.envelope,
    )

    assert type(projection) is RunLedgerReceiptSnapshotV1
    assert projection.schema_version == 1
    assert projection.projection_policy == ("sealed-answer-content-blind-receipts-v1")
    assert projection.ledger_policy == sealed_ledger.full_snapshot.ledger_policy
    assert projection.run_id == sealed_ledger.full_snapshot.run_id
    assert projection.manifest_sha256 == sealed_ledger.full_snapshot.manifest_sha256
    assert (
        projection.run_envelope_sha256
        == sealed_ledger.full_snapshot.run_envelope_sha256
    )
    assert projection.call_count == 384
    assert projection.seal_kind == "complete_with_attrition"
    assert projection.run_root_sha256 == (
        sealed_ledger.full_snapshot.computed_run_root_sha256
    )
    assert len(projection.slots) == 384
    assert all(type(slot) is RunLedgerReceiptSlotV1 for slot in projection.slots)
    assert tuple(slot.plan for slot in projection.slots) == (
        sealed_ledger.envelope.call_plans
    )
    assert tuple(slot.ledger_leaf_sha256 for slot in projection.slots) == tuple(
        slot.leaf_sha256 for slot in sealed_ledger.full_snapshot.slots
    )
    assert tuple(slot.plan.slot_index for slot in projection.slots) == tuple(range(384))

    artifacts = _success_artifacts(sealed_ledger.database_path)
    success = projection.slots[0]
    assert success.state == "SUCCEEDED"
    assert success.attempt_count == 1
    assert success.terminal_reason is None
    assert success.canonical_receipt is not None
    assert success.canonical_receipt.slot_index == 0
    commitments = (
        success.receipt_commitment,
        success.trace_commitment,
        success.response_evidence_commitment,
        success.decoded_response_commitment,
    )
    assert all(type(value) is LedgerArtifactCommitmentV1 for value in commitments)
    for commitment, artifact in zip(
        commitments,
        (artifacts[column] for column in _SUCCESS_ARTIFACT_COLUMNS),
        strict=True,
    ):
        assert commitment is not None
        assert commitment.byte_count == len(artifact)
        assert commitment.sha256 == hashlib.sha256(artifact).hexdigest()

    attrition = projection.slots[1]
    assert attrition.state == "ATTRITION"
    assert attrition.attempt_count == 1
    assert attrition.canonical_receipt is None
    assert attrition.receipt_commitment is None
    assert attrition.trace_commitment is None
    assert attrition.response_evidence_commitment is None
    assert attrition.decoded_response_commitment is None
    assert attrition.terminal_reason == "generation_failure"
    _assert_no_byte_preimages(projection)
    assert _PRIVATE_ANSWER.decode("ascii") not in repr(projection)
    assert _recompute_projection_run_root(projection) == projection.run_root_sha256


def test_receipt_projection_never_replays_or_decodes_answer_content(
    sealed_ledger: _SealedLedgerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_calls = 0

    def forbid_replay(*_args: object, **_kwargs: object) -> None:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("receipt projection replayed model artifacts")

    monkeypatch.setattr(
        ledger_module,
        "audited_model_call_execution_v2_from_artifacts",
        forbid_replay,
    )
    monkeypatch.setattr(sealed_ledger.tokenizer, "fail_decode", True)

    projection = load_run_ledger_receipt_snapshot_v1(
        sealed_ledger.database_path,
        sealed_ledger.manifest,
        sealed_ledger.tokenizer,
        sealed_ledger.envelope,
    )

    assert replay_calls == 0
    assert projection.slots[0].state == "SUCCEEDED"


@pytest.mark.parametrize("ledger_state", ("open", "indeterminate"))
def test_receipt_projection_requires_a_semantically_terminal_seal(
    tmp_path: Path,
    sealed_ledger: _SealedLedgerFixture,
    ledger_state: str,
) -> None:
    database_path = tmp_path / f"{ledger_state}.sqlite3"
    session = RunLedgerSessionV1.create(
        database_path,
        sealed_ledger.manifest,
        sealed_ledger.tokenizer,
        sealed_ledger.envelope,
    )
    if ledger_state == "indeterminate":
        session.mark_started(0)
        session.seal_indeterminate(0, "unexpected_failure")

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger_receipt_snapshot_v1(
            database_path,
            sealed_ledger.manifest,
            sealed_ledger.tokenizer,
            sealed_ledger.envelope,
        )

    assert error.value.reason == "ledger_state"


@pytest.mark.parametrize("artifact_column", _SUCCESS_ARTIFACT_COLUMNS)
@pytest.mark.parametrize("rehash_ledger", (False, True))
def test_projection_rejects_each_success_blob_tamper_even_if_ledger_is_rehashed(
    tmp_path: Path,
    sealed_ledger: _SealedLedgerFixture,
    artifact_column: str,
    rehash_ledger: bool,
) -> None:
    database_path = _copy_database(
        tmp_path,
        sealed_ledger,
        f"tampered-{artifact_column}-{rehash_ledger}.sqlite3",
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        original = connection.execute(
            f"SELECT {artifact_column} FROM run_ledger_slots WHERE slot_index = 0"
        ).fetchone()
        assert original is not None and type(original[0]) is bytes
        connection.execute(
            f"UPDATE run_ledger_slots SET {artifact_column} = ? WHERE slot_index = 0",
            (sqlite3.Binary(original[0] + b"x"),),
        )
    finally:
        connection.close()
    if rehash_ledger:
        _rehash_success_slot_and_run_root(database_path, sealed_ledger)

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger_receipt_snapshot_v1(
            database_path,
            sealed_ledger.manifest,
            sealed_ledger.tokenizer,
            sealed_ledger.envelope,
        )

    _assert_ledger_corruption(error)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("manifest_sha256", "0" * 64),
        ("run_envelope_sha256", "1" * 64),
        ("slot_index", False),
        ("case_index", False),
        ("slot_index", 1),
        ("case_index", 1),
        ("arm", "raw_history"),
        ("request_id", "forged-request-id"),
    ),
)
def test_projection_rejects_rehashed_canonical_receipt_identity_forgery(
    tmp_path: Path,
    sealed_ledger: _SealedLedgerFixture,
    field: str,
    forged_value: object,
) -> None:
    database_path = _copy_database(
        tmp_path,
        sealed_ledger,
        f"forged-receipt-{field}.sqlite3",
    )
    artifacts = _success_artifacts(database_path)
    receipt_value = json.loads(artifacts["receipt_bytes"].decode("ascii"))
    assert type(receipt_value) is dict
    receipt_value[field] = forged_value
    forged_receipt = json.dumps(
        receipt_value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE run_ledger_slots SET receipt_bytes = ? WHERE slot_index = 0",
            (sqlite3.Binary(forged_receipt),),
        )
    finally:
        connection.close()
    _rehash_success_slot_and_run_root(database_path, sealed_ledger)

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger_receipt_snapshot_v1(
            database_path,
            sealed_ledger.manifest,
            sealed_ledger.tokenizer,
            sealed_ledger.envelope,
        )

    _assert_ledger_corruption(error)


def test_existing_full_loader_still_replays_success_artifacts(
    sealed_ledger: _SealedLedgerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_calls = 0

    def fail_replay(*_args: object, **_kwargs: object) -> None:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("full loader replay sentinel")

    monkeypatch.setattr(
        ledger_module,
        "audited_model_call_execution_v2_from_artifacts",
        fail_replay,
    )

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(
            sealed_ledger.database_path,
            sealed_ledger.manifest,
            sealed_ledger.tokenizer,
            sealed_ledger.envelope,
        )

    _assert_ledger_corruption(error)
    assert replay_calls == 1
