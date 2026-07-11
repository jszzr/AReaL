# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the independent Memory helpfulness run ledger."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import sys
from dataclasses import replace
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
    audited_model_call_receipt_v2_bytes,
    prepare_infbridge_run_envelope_v2,
)
from examples.memory_service.infbridge_run_ledger import (
    RunLedgerError,
    RunLedgerSlotSnapshotV1,
    RunLedgerSnapshotV1,
    initialize_run_ledger,
    load_run_ledger,
)

from areal.v2.inference_service.client_trace import (
    generation_physical_trace_bytes,
    generation_response_evidence_bytes,
)
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_CANONICAL_RUNTIME = sys.implementation.cache_tag == "cpython-312" and sys.version_info[
    :3
] == (3, 12, 13)


class _LedgerByteTokenizer:
    _ARTIFACT_BYTES = b'ledger-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'ledger-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="ledger-byte-tokenizer-v2",
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
            raise ValueError("ledger decode policy drift")
        return bytes(token_ids).decode("utf-8")


@pytest.fixture(scope="module")
def ledger_inputs() -> tuple[
    helpfulness.ModelRunManifest,
    _LedgerByteTokenizer,
    RunEnvelopeV2,
    InfBridge,
]:
    tokenizer = _LedgerByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="1" * 40,
        evaluator_commit_sha="2" * 40,
        model_id="ledger-test-model",
        model_weights_sha256="3" * 64,
        tokenizer_id="ledger-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://ledger-model.test/",
        pause_state=PauseState(),
        request_timeout=9.0,
        max_resubmit_retries=3,
        resubmit_wait=0.0,
        version=23,
    )
    envelope = prepare_infbridge_run_envelope_v2(
        prepared.manifest,
        tokenizer,
        bridge,
        max_new_tokens=32,
    )
    yield prepared.manifest, tokenizer, envelope, bridge
    asyncio.run(bridge.aclose())


def _assert_reason(error: pytest.ExceptionInfo[RunLedgerError], reason: str) -> None:
    assert type(error.value) is RunLedgerError
    assert error.value.reason == reason
    assert str(error.value) == reason


def _connect_raw(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def _rewrite_empty_artifact_slot(
    connection: sqlite3.Connection,
    slot: RunLedgerSlotSnapshotV1,
    *,
    state: str,
    terminal_reason: str | None,
) -> RunLedgerSlotSnapshotV1:
    attempt_count = 0 if state == "PLANNED" else 1
    leaf_sha256 = ledger_module._slot_leaf_sha256(
        slot_index=slot.plan.slot_index,
        plan_sha256=slot.plan_sha256,
        state=state,
        attempt_count=attempt_count,
        receipt_bytes=None,
        trace_bytes=None,
        response_evidence_bytes=None,
        decoded_response_utf8=None,
        terminal_reason=terminal_reason,
    )
    connection.execute(
        "UPDATE run_ledger_slots SET state = ?, attempt_count = ?, "
        "terminal_reason = ?, leaf_sha256 = ? WHERE slot_index = ?",
        (
            state,
            attempt_count,
            terminal_reason,
            leaf_sha256,
            slot.plan.slot_index,
        ),
    )
    return replace(
        slot,
        state=state,
        attempt_count=attempt_count,
        terminal_reason=terminal_reason,
        leaf_sha256=leaf_sha256,
    )


def _seal_raw(
    connection: sqlite3.Connection,
    snapshot: RunLedgerSnapshotV1,
    slots: tuple[RunLedgerSlotSnapshotV1, ...],
    *,
    seal_kind: str,
) -> str:
    root = ledger_module._run_root_sha256(
        run_id=snapshot.run_id,
        manifest_sha256=snapshot.manifest_sha256,
        run_envelope_sha256=snapshot.run_envelope_sha256,
        status="SEALED",
        seal_kind=seal_kind,
        slots=slots,
    )
    connection.execute(
        "UPDATE run_ledger_header SET status = 'SEALED', seal_kind = ?, "
        "run_root_sha256 = ? WHERE singleton = 1",
        (seal_kind, root),
    )
    return root


def _sglang_response(output: bytes) -> dict[str, Any]:
    return {
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [
                (-0.01 * (index + 1), token_id) for index, token_id in enumerate(output)
            ],
        }
    }


def test_schema_and_catalog_golden_vectors_are_versioned_explicitly() -> None:
    assert ledger_module._SCHEMA_VERSION == 1
    assert ledger_module._schema_spec_sha256() == (
        "7c747c9c213695739a8d28f65a3df19afba4a680433e4a553baaae4ac741cb55"
    )
    assert ledger_module._SCHEMA_SPEC_SHA256 == (
        "7c747c9c213695739a8d28f65a3df19afba4a680433e4a553baaae4ac741cb55"
    )
    assert (
        ledger_module._sha256(
            ledger_module._canonical_json_bytes(ledger_module._EXPECTED_CATALOG_ROWS)
        )
        == "c7cf712c96fb609412c676b932e82d38c564f7de335d2f1c5cc675619ee23cdc"
    )
    assert ledger_module._EXPECTED_CATALOG_SHA256 == (
        "c7cf712c96fb609412c676b932e82d38c564f7de335d2f1c5cc675619ee23cdc"
    )


def test_initialize_precommits_exactly_384_planned_slots_and_reopens(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "run-ledger.sqlite3"

    snapshot = initialize_run_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )

    assert type(snapshot) is RunLedgerSnapshotV1
    assert snapshot.schema_version == 1
    assert snapshot.ledger_policy == "single-cursor-prewrite-v1"
    assert len(snapshot.run_id) == 64
    assert snapshot.manifest_sha256 == envelope.manifest_sha256
    assert len(snapshot.run_envelope_sha256) == 64
    assert snapshot.call_count == 384
    assert snapshot.status == "OPEN"
    assert snapshot.seal_kind is None
    assert snapshot.stored_run_root_sha256 is None
    assert len(snapshot.computed_run_root_sha256) == 64
    assert len(snapshot.slots) == 384
    assert all(type(slot) is RunLedgerSlotSnapshotV1 for slot in snapshot.slots)
    assert all(slot.state == "PLANNED" for slot in snapshot.slots)
    assert all(slot.attempt_count == 0 for slot in snapshot.slots)
    assert all(slot.receipt_bytes is None for slot in snapshot.slots)
    assert tuple(slot.plan for slot in snapshot.slots) == envelope.call_plans
    assert len({slot.plan.request_id for slot in snapshot.slots}) == 384
    assert len({slot.leaf_sha256 for slot in snapshot.slots}) == 384
    if _CANONICAL_RUNTIME:
        assert snapshot.run_id == (
            "d4c9027ef445c91efb7d52df60d780b0ec376f8f2bb65f2eec40a817ff059c6c"
        )
        assert snapshot.computed_run_root_sha256 == (
            "d3a7e589af37efe7ad4de834edb8b4822a3e56abf049eec4002fbe60e047cbd6"
        )
        assert snapshot.slots[0].leaf_sha256 == (
            "6e7a8cf9fd7da21525f0535d4072c5dab679bf2f55ff5ad91946b9b19fecd81b"
        )
        assert snapshot.slots[-1].leaf_sha256 == (
            "9e0f09ef93b8926654a003637d0cf57c817f63ac0267ca81205087d0a0717127"
        )

    reopened = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert reopened == snapshot
    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    _assert_reason(error, "ledger_exists")
    assert database_path.stat().st_mode & 0o777 == 0o600

    connection = _connect_raw(database_path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone() == (0x41524C31,)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
            "run_ledger_schema_metadata",
            "run_ledger_header",
            "run_ledger_slots",
        }
        assert connection.execute(
            "SELECT count(*), min(slot_index), max(slot_index), "
            "sum(attempt_count) FROM run_ledger_slots"
        ).fetchone() == (384, 0, 383, 0)
    finally:
        connection.close()
    configured = ledger_module._connect(str(database_path))
    try:
        assert configured.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert configured.execute("PRAGMA synchronous").fetchone() == (3,)
        assert configured.execute("PRAGMA fullfsync").fetchone() == (1,)
    finally:
        configured.close()


def test_same_preregistration_has_same_run_identity_and_root_across_databases(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    first = initialize_run_ledger(
        tmp_path / "first.sqlite3",
        manifest,
        tokenizer,
        envelope,
    )
    second = initialize_run_ledger(
        tmp_path / "second.sqlite3",
        manifest,
        tokenizer,
        envelope,
    )

    assert first == second
    assert first.run_id == second.run_id
    assert first.computed_run_root_sha256 == second.computed_run_root_sha256


def test_existing_database_rejects_a_different_run_identity(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, bridge = ledger_inputs
    database_path = tmp_path / "identity.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    changed = prepare_infbridge_run_envelope_v2(
        manifest,
        tokenizer,
        bridge,
        max_new_tokens=33,
    )

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, changed)

    _assert_reason(error, "ledger_identity")
    assert (
        load_run_ledger(database_path, manifest, tokenizer, envelope).status == "OPEN"
    )


def test_creation_rebuilds_and_rejects_shape_valid_but_forged_call_plan(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    forged_plan = replace(
        envelope.call_plans[0],
        input_token_ids_sha256="f" * 64,
    )
    forged = replace(
        envelope,
        call_plans=(forged_plan, *envelope.call_plans[1:]),
    )
    database_path = tmp_path / "forged-plan.sqlite3"

    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(database_path, manifest, tokenizer, forged)

    _assert_reason(error, "ledger_identity")
    assert not database_path.exists()


@pytest.mark.parametrize(
    "tamper_sql",
    (
        "UPDATE run_ledger_slots SET plan_bytes = x'00' WHERE slot_index = 0",
        "UPDATE run_ledger_slots SET leaf_sha256 = printf('%064d', 0) "
        "WHERE slot_index = 0",
        "UPDATE run_ledger_slots SET state = 'STARTED', attempt_count = 1 "
        "WHERE slot_index = 0",
        "UPDATE run_ledger_slots SET plan_bytes = zeroblob(16385) WHERE slot_index = 0",
        "DELETE FROM run_ledger_slots WHERE slot_index = 383",
    ),
)
def test_snapshot_rejects_slot_content_state_leaf_or_completeness_tampering(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    tamper_sql: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = (
        tmp_path / f"tamper-{hashlib.sha256(tamper_sql.encode()).hexdigest()}.db"
    )
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        connection.execute(tamper_sql)
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_corruption")


@pytest.mark.asyncio
async def test_success_artifacts_cannot_be_reassigned_to_another_slot(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, bridge = ledger_inputs
    database_path = tmp_path / "cross-slot.sqlite3"
    snapshot = initialize_run_ledger(
        database_path,
        manifest,
        tokenizer,
        envelope,
    )
    original_send = bridge._send_request
    bridge._send_request = AsyncMock(return_value=_sglang_response(b"ANSWER"))
    try:
        source_execution = await InfBridgeModelAdapter(
            manifest,
            tokenizer,
            envelope,
            bridge,
        ).submit(0, envelope.call_plans[0].arm)
    finally:
        bridge._send_request = original_send
    receipt_bytes = audited_model_call_receipt_v2_bytes(source_execution.receipt)
    trace_bytes = generation_physical_trace_bytes(source_execution.trace)
    response_evidence_bytes = generation_response_evidence_bytes(
        source_execution.response_evidence
    )
    decoded_response_utf8 = source_execution.response.encode("utf-8")
    target = snapshot.slots[1]
    leaf_sha256 = ledger_module._slot_leaf_sha256(
        slot_index=target.plan.slot_index,
        plan_sha256=target.plan_sha256,
        state="SUCCEEDED",
        attempt_count=1,
        receipt_bytes=receipt_bytes,
        trace_bytes=trace_bytes,
        response_evidence_bytes=response_evidence_bytes,
        decoded_response_utf8=decoded_response_utf8,
        terminal_reason=None,
    )
    connection = _connect_raw(database_path)
    try:
        connection.execute(
            "UPDATE run_ledger_slots SET state = 'SUCCEEDED', attempt_count = 1, "
            "receipt_bytes = ?, trace_bytes = ?, response_evidence_bytes = ?, "
            "decoded_response_utf8 = ?, leaf_sha256 = ? WHERE slot_index = 1",
            (
                sqlite3.Binary(receipt_bytes),
                sqlite3.Binary(trace_bytes),
                sqlite3.Binary(response_evidence_bytes),
                sqlite3.Binary(decoded_response_utf8),
                leaf_sha256,
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_corruption")


@pytest.mark.parametrize(
    ("state", "terminal_reason"),
    (
        ("STARTED", None),
        ("ATTRITION", "attempt_limit"),
    ),
)
def test_loader_accepts_legal_open_single_cursor_prefixes(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    state: str,
    terminal_reason: str | None,
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / f"legal-{state}.sqlite3"
    initial = initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        _rewrite_empty_artifact_slot(
            connection,
            initial.slots[0],
            state=state,
            terminal_reason=terminal_reason,
        )
    finally:
        connection.close()

    loaded = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert loaded.status == "OPEN"
    assert loaded.slots[0].state == state
    assert all(slot.state == "PLANNED" for slot in loaded.slots[1:])


def test_loader_accepts_legal_indeterminate_seal_with_planned_suffix(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "legal-indeterminate.sqlite3"
    initial = initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        changed = _rewrite_empty_artifact_slot(
            connection,
            initial.slots[0],
            state="INDETERMINATE",
            terminal_reason="orphan_started",
        )
        slots = (changed, *initial.slots[1:])
        expected_root = _seal_raw(
            connection,
            initial,
            slots,
            seal_kind="indeterminate",
        )
    finally:
        connection.close()

    loaded = load_run_ledger(database_path, manifest, tokenizer, envelope)
    assert loaded.status == "SEALED"
    assert loaded.seal_kind == "indeterminate"
    assert loaded.stored_run_root_sha256 == expected_root
    assert loaded.computed_run_root_sha256 == expected_root
    assert loaded.slots[0].state == "INDETERMINATE"


@pytest.mark.parametrize(
    "scenario",
    ("gap_started", "two_started", "open_indeterminate", "bad_complete_seal"),
)
def test_loader_rejects_rehashed_illegal_run_state_sequences(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
    scenario: str,
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / f"illegal-{scenario}.sqlite3"
    initial = initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    slots = list(initial.slots)
    connection = _connect_raw(database_path)
    try:
        if scenario == "gap_started":
            slots[1] = _rewrite_empty_artifact_slot(
                connection,
                slots[1],
                state="STARTED",
                terminal_reason=None,
            )
        elif scenario == "two_started":
            for index in (0, 1):
                slots[index] = _rewrite_empty_artifact_slot(
                    connection,
                    slots[index],
                    state="STARTED",
                    terminal_reason=None,
                )
        elif scenario == "open_indeterminate":
            slots[0] = _rewrite_empty_artifact_slot(
                connection,
                slots[0],
                state="INDETERMINATE",
                terminal_reason="orphan_started",
            )
        else:
            _seal_raw(
                connection,
                initial,
                tuple(slots),
                seal_kind="complete",
            )
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_corruption")


def test_snapshot_rejects_unknown_schema_objects(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "schema-tamper.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        connection.execute("CREATE TABLE injected (value INTEGER)")
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_schema")


def test_snapshot_rejects_rebuilt_weak_schema_even_with_forged_catalog_hash(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "weak-schema.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        cursor = connection.cursor()
        cursor.execute("BEGIN")
        cursor.execute("CREATE TABLE weak_slots AS SELECT * FROM run_ledger_slots")
        cursor.execute("DROP TABLE run_ledger_slots")
        cursor.execute("ALTER TABLE weak_slots RENAME TO run_ledger_slots")
        forged_catalog_sha256 = ledger_module._catalog_sha256(cursor)
        cursor.execute(
            "UPDATE run_ledger_schema_metadata "
            "SET schema_catalog_sha256 = ? WHERE singleton = 1",
            (forged_catalog_sha256,),
        )
        cursor.execute("COMMIT")
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_schema")


def test_snapshot_rejects_header_identity_tampering(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "header-tamper.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        connection.execute(
            "UPDATE run_ledger_header SET manifest_bytes = x'00' WHERE singleton = 1"
        )
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_identity")


def test_initialize_never_recreates_an_erased_existing_run(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "erased-run.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        connection.execute("BEGIN")
        connection.execute("DELETE FROM run_ledger_slots")
        connection.execute("DELETE FROM run_ledger_header")
        connection.execute("COMMIT")
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_exists")
    connection = _connect_raw(database_path)
    try:
        assert connection.execute(
            "SELECT count(*) FROM run_ledger_header"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM run_ledger_slots"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_load_missing_database_does_not_create_it(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "missing.sqlite3"

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_persistence")
    assert not database_path.exists()


def test_creation_never_overwrites_a_preexisting_empty_file(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "preexisting-empty.sqlite3"
    database_path.write_bytes(b"")
    database_path.chmod(0o600)

    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_exists")
    assert database_path.read_bytes() == b""


def test_symlink_and_hardlink_database_aliases_are_rejected(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    target.chmod(0o600)
    symlink = tmp_path / "symlink.sqlite3"
    symlink.symlink_to(target)
    with pytest.raises(RunLedgerError) as symlink_error:
        initialize_run_ledger(symlink, manifest, tokenizer, envelope)
    _assert_reason(symlink_error, "ledger_path")
    assert target.read_bytes() == b""

    database_path = tmp_path / "original.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    hardlink = tmp_path / "hardlink.sqlite3"
    hardlink.hardlink_to(database_path)
    with pytest.raises(RunLedgerError) as hardlink_error:
        load_run_ledger(hardlink, manifest, tokenizer, envelope)
    _assert_reason(hardlink_error, "ledger_path")
    hardlink.unlink()
    assert (
        load_run_ledger(
            database_path,
            manifest,
            tokenizer,
            envelope,
        ).status
        == "OPEN"
    )


def test_unicode_uri_punctuation_path_round_trips_without_aliasing(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "中文 空格#?%.sqlite3"

    created = initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    loaded = load_run_ledger(database_path, manifest, tokenizer, envelope)

    assert loaded == created


def test_wal_mode_is_rejected_without_automatic_conversion(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "wal.sqlite3"
    initialize_run_ledger(database_path, manifest, tokenizer, envelope)
    connection = _connect_raw(database_path)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    finally:
        connection.close()

    with pytest.raises(RunLedgerError) as error:
        load_run_ledger(database_path, manifest, tokenizer, envelope)

    _assert_reason(error, "ledger_schema")
    connection = _connect_raw(database_path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "database_path",
    (
        ":memory:",
        b"bytes.db",
        "\x00bad",
        "",
        "   ",
        "\ud800",
        "//tmp/areal-ledger-double-slash.sqlite3",
    ),
)
def test_ledger_requires_one_durable_utf8_string_path(
    database_path: object,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs

    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(
            database_path,  # type: ignore[arg-type]
            manifest,
            tokenizer,
            envelope,
        )

    _assert_reason(error, "ledger_path")


def test_invalid_caller_envelope_is_rejected_before_database_creation(
    tmp_path: Path,
    ledger_inputs: tuple[
        helpfulness.ModelRunManifest,
        _LedgerByteTokenizer,
        RunEnvelopeV2,
        InfBridge,
    ],
) -> None:
    manifest, tokenizer, envelope, _bridge = ledger_inputs
    database_path = tmp_path / "invalid-envelope.sqlite3"
    invalid = replace(envelope, schema_version=3)

    with pytest.raises(RunLedgerError) as error:
        initialize_run_ledger(database_path, manifest, tokenizer, invalid)

    _assert_reason(error, "ledger_identity")
    assert not database_path.exists()
