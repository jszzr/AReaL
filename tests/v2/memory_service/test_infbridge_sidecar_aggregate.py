# SPDX-License-Identifier: Apache-2.0

"""Tests for answer-free aggregation of all 64 live Memory sidecars."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from examples.memory_service import infbridge_sidecar_join as sidecar_join
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    DecoderAuditMaterialV2,
    RunEnvelopeV2,
    prepare_infbridge_run_envelope_v2,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_AGGREGATE_ROOT_DOMAIN = b"areal-memory-model-sidecar-aggregate-root-v1\0"
_OPAQUE_TOKEN_BASE = (1 << 127) | 0x6A18_0000_0000_0000
_FAKE_PID_BASE = os.getpid() + 100_000


class _AggregateByteTokenizer:
    _ARTIFACT_BYTES = b'aggregate-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'aggregate-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="aggregate-byte-tokenizer-v2",
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
            raise ValueError("aggregate decode policy drift")
        return bytes(token_ids).decode("utf-8")


@dataclass(frozen=True, slots=True)
class _AggregateFixture:
    manifest: helpfulness.ModelRunManifest
    tokenizer: _AggregateByteTokenizer
    envelope: RunEnvelopeV2
    sidecars: tuple[sidecar_join.ModelCaseSidecarV1, ...]
    capture_requests: tuple[helpfulness.ModelCaptureChildRequest, ...]
    capture_responses: tuple[helpfulness.ModelCaptureChildResponse, ...]
    observation_requests: tuple[
        tuple[helpfulness.ModelObservationChildRequest, ...], ...
    ]
    observation_responses: tuple[
        tuple[helpfulness.ModelObservationChildResponse, ...], ...
    ]
    probe_requests: tuple[helpfulness.ModelLeakageProbeChildRequest, ...]
    probe_responses: tuple[helpfulness.ModelLeakageProbeChildResponse, ...]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _uuid4(index: int) -> str:
    return str(uuid.UUID(int=index + 1, version=4))


def _component_id(logical_index: int, component: str) -> str:
    return hashlib.sha256(
        f"aggregate-sidecar|{logical_index}|{component}".encode("ascii")
    ).hexdigest()


def _exchange(
    request: object,
    response: object,
    *,
    launcher_pid: int,
) -> sidecar_join.CanonicalChildExchangeV1:
    return sidecar_join.CanonicalChildExchangeV1(
        request_wire_utf8=helpfulness.wire_dumps(request).encode("utf-8"),
        response_wire_utf8=helpfulness.wire_dumps(response).encode("utf-8"),
        launcher_pid=launcher_pid,
    )


def _synthetic_observation_response(
    response: helpfulness.ModelObservationChildResponse,
    *,
    logical_index: int,
) -> helpfulness.ModelObservationChildResponse:
    pid = _FAKE_PID_BASE + 1 + logical_index
    process_id = _uuid4(1 + logical_index)
    state = response.state_receipt
    state_updates = {
        field.name: _component_id(logical_index, field.name)
        for field in fields(state)
        if field.name.endswith("_instance_id") and field.name != "process_instance_id"
    }
    return replace(
        response,
        observation=replace(
            response.observation,
            future_pid=pid,
            future_process_instance_id=process_id,
        ),
        state_receipt=replace(
            state,
            **state_updates,
            process_instance_id=process_id,
        ),
        pid=pid,
        process_instance_id=process_id,
        isolated_mode=True,
        visible_forbidden_environment=(),
        environment_clean=True,
    )


def _synthetic_probe_response(
    response: helpfulness.ModelLeakageProbeChildResponse,
    *,
    case_index: int,
) -> helpfulness.ModelLeakageProbeChildResponse:
    logical_index = helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS) + (
        case_index
    )
    pid = _FAKE_PID_BASE + 1 + logical_index
    process_id = _uuid4(1 + logical_index)
    state = response.state_receipt
    state_updates = {
        field.name: _component_id(logical_index, field.name)
        for field in fields(state)
        if field.name.endswith("_instance_id") and field.name != "process_instance_id"
    }
    return replace(
        response,
        probe=replace(
            response.probe,
            future_pid=pid,
            future_process_instance_id=process_id,
        ),
        state_receipt=replace(
            state,
            **state_updates,
            process_instance_id=process_id,
        ),
        pid=pid,
        process_instance_id=process_id,
        isolated_mode=True,
        visible_forbidden_environment=(),
        environment_clean=True,
    )


def _build_aggregate_fixture(root: Path) -> _AggregateFixture:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    tokenizer = _AggregateByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="a" * 40,
        evaluator_commit_sha="b" * 40,
        model_id="aggregate-sidecar-test-model",
        model_weights_sha256="c" * 64,
        tokenizer_id="aggregate-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    manifest = prepared.manifest
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://aggregate-sidecar-model.test/",
        pause_state=PauseState(),
        request_timeout=7.0,
        max_resubmit_retries=2,
        resubmit_wait=0.0,
        version=43,
    )
    try:
        envelope = prepare_infbridge_run_envelope_v2(
            manifest,
            tokenizer,
            bridge,
            max_new_tokens=48,
        )
    finally:
        asyncio.run(bridge.aclose())

    sidecars: list[sidecar_join.ModelCaseSidecarV1] = []
    capture_requests: list[helpfulness.ModelCaptureChildRequest] = []
    capture_responses: list[helpfulness.ModelCaptureChildResponse] = []
    observation_requests: list[
        tuple[helpfulness.ModelObservationChildRequest, ...]
    ] = []
    observation_responses: list[
        tuple[helpfulness.ModelObservationChildResponse, ...]
    ] = []
    probe_requests: list[helpfulness.ModelLeakageProbeChildRequest] = []
    probe_responses: list[helpfulness.ModelLeakageProbeChildResponse] = []

    for case_index, registration in enumerate(manifest.cases):
        case_root = root / f"case-{case_index:03d}"
        case_root.mkdir(mode=0o700)
        database_path = case_root / "memory.sqlite3"
        capture_request = helpfulness.ModelCaptureChildRequest(
            case_index=case_index,
            model_attempt=registration.model_attempt,
            case_manifest_sha256=registration.identity.case_manifest_sha256,
            database_path=str(database_path),
        )
        raw_capture = helpfulness.execute_model_capture_child_request(capture_request)
        capture_pid = _FAKE_PID_BASE + 10_000 + case_index
        capture_response = replace(
            raw_capture,
            pid=capture_pid,
            process_instance_id=_uuid4(10_000 + case_index),
            isolated_mode=True,
            visible_forbidden_environment=(),
            environment_clean=True,
        )

        case_observation_sidecars: list[sidecar_join.ModelObservationSidecarV1] = []
        case_observation_requests: list[helpfulness.ModelObservationChildRequest] = []
        case_observation_responses: list[helpfulness.ModelObservationChildResponse] = []
        for arm_offset, arm_call in enumerate(registration.arm_calls):
            slot_index = case_index * len(helpfulness.MODEL_ARMS) + arm_offset
            execution_token = _OPAQUE_TOKEN_BASE + slot_index
            future_session_id, future_run_id = helpfulness._opaque_future_identity(
                execution_token
            )
            request = helpfulness.ModelObservationChildRequest(
                execution_index=execution_token,
                database_path=str(database_path),
                database_receipt=capture_response.database_receipt,
                scope=capture_response.references.capture.local_scope,
                source=helpfulness._fast_source_spec(
                    registration.identity.case,
                    capture_response.references,
                    arm_call.arm,
                ),
                future_session_id=future_session_id,
                future_run_id=future_run_id,
                renderer_version="memory-codebook/v1",
            )
            response = _synthetic_observation_response(
                helpfulness.execute_model_observation_child_request(request),
                logical_index=slot_index,
            )
            case_observation_requests.append(request)
            case_observation_responses.append(response)
            case_observation_sidecars.append(
                sidecar_join.ModelObservationSidecarV1(
                    slot_index=slot_index,
                    case_index=case_index,
                    arm=arm_call.arm,
                    exchange=_exchange(request, response, launcher_pid=response.pid),
                )
            )

        probe_logical_index = (
            helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS) + case_index
        )
        probe_request = helpfulness.ModelLeakageProbeChildRequest(
            execution_index=_OPAQUE_TOKEN_BASE + probe_logical_index,
            database_path=str(database_path),
            database_receipt=capture_response.database_receipt,
            scope=capture_response.references.capture.local_scope,
            release_id=(
                capture_response.references.releases.foreign_sentinel_release_id
            ),
        )
        probe_response = _synthetic_probe_response(
            helpfulness.execute_model_leakage_probe_child_request(probe_request),
            case_index=case_index,
        )
        sidecars.append(
            sidecar_join.ModelCaseSidecarV1(
                capture=sidecar_join.ModelCaptureSidecarV1(
                    case_index=case_index,
                    exchange=_exchange(
                        capture_request,
                        capture_response,
                        launcher_pid=capture_response.pid,
                    ),
                ),
                observations=tuple(case_observation_sidecars),
                probe=sidecar_join.ModelProbeSidecarV1(
                    case_index=case_index,
                    logical_execution_index=probe_logical_index,
                    exchange=_exchange(
                        probe_request,
                        probe_response,
                        launcher_pid=probe_response.pid,
                    ),
                ),
            )
        )
        capture_requests.append(capture_request)
        capture_responses.append(capture_response)
        observation_requests.append(tuple(case_observation_requests))
        observation_responses.append(tuple(case_observation_responses))
        probe_requests.append(probe_request)
        probe_responses.append(probe_response)

    return _AggregateFixture(
        manifest=manifest,
        tokenizer=tokenizer,
        envelope=envelope,
        sidecars=tuple(sidecars),
        capture_requests=tuple(capture_requests),
        capture_responses=tuple(capture_responses),
        observation_requests=tuple(observation_requests),
        observation_responses=tuple(observation_responses),
        probe_requests=tuple(probe_requests),
        probe_responses=tuple(probe_responses),
    )


@pytest.fixture(scope="module")
def aggregate_fixture(tmp_path_factory: pytest.TempPathFactory) -> _AggregateFixture:
    return _build_aggregate_fixture(tmp_path_factory.mktemp("sidecar-aggregate"))


@pytest.fixture(autouse=True)
def _allow_synthetic_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all process bindings except the unavailable live-Popen observation."""

    def validate(response: object, *, launcher_pid: int) -> None:
        assert response.pid == launcher_pid  # type: ignore[union-attr]
        assert response.pid != os.getpid()  # type: ignore[union-attr]
        assert response.isolated_mode is True  # type: ignore[union-attr]
        assert response.environment_clean is True  # type: ignore[union-attr]
        assert response.visible_forbidden_environment == ()  # type: ignore[union-attr]
        assert helpfulness._is_canonical_uuid4(  # type: ignore[attr-defined]
            response.process_instance_id  # type: ignore[union-attr]
        )
        if type(response) is helpfulness.ModelObservationChildResponse:
            assert response.observation.future_pid == response.pid
            assert (
                response.observation.future_process_instance_id
                == response.process_instance_id
            )
        if type(response) is helpfulness.ModelLeakageProbeChildResponse:
            assert response.probe.future_pid == response.pid
            assert (
                response.probe.future_process_instance_id
                == response.process_instance_id
            )

    monkeypatch.setattr(sidecar_join, "_validate_live_process", validate)


def _validate(
    fixture: _AggregateFixture,
    sidecars: tuple[sidecar_join.ModelCaseSidecarV1, ...] | None = None,
) -> sidecar_join.ValidatedModelSidecarAggregateV1:
    return sidecar_join.validate_live_model_sidecar_aggregate_v1(
        fixture.manifest,
        fixture.tokenizer,
        fixture.envelope,
        fixture.sidecars if sidecars is None else sidecars,
    )


def _response_exchange(
    exchange: sidecar_join.CanonicalChildExchangeV1,
    response: object,
) -> sidecar_join.CanonicalChildExchangeV1:
    return replace(
        exchange,
        response_wire_utf8=helpfulness.wire_dumps(response).encode("utf-8"),
    )


def _replace_case(
    sidecars: tuple[sidecar_join.ModelCaseSidecarV1, ...],
    case_index: int,
    sidecar: sidecar_join.ModelCaseSidecarV1,
) -> tuple[sidecar_join.ModelCaseSidecarV1, ...]:
    values = list(sidecars)
    values[case_index] = sidecar
    return tuple(values)


def _assert_rejected(
    fixture: _AggregateFixture,
    sidecars: tuple[sidecar_join.ModelCaseSidecarV1, ...],
) -> None:
    with pytest.raises(sidecar_join.ModelSidecarJoinError):
        _validate(fixture, sidecars)


def test_full_inventory_and_aggregate_root_are_exact_and_answer_free(
    aggregate_fixture: _AggregateFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("answer semantics are forbidden in the sidecar aggregate")

    for name in (
        "normalize_response",
        "abstained",
        "utility",
        "parent_join_and_score",
        "analyze_model_run",
    ):
        monkeypatch.setattr(helpfulness, name, forbidden)

    validated = _validate(aggregate_fixture)
    commitment = validated.commitment
    assert commitment.schema_version == 1
    assert commitment.policy == "live-honest-child-consistency-only-v1"
    assert commitment.case_count == helpfulness.MODEL_CASE_COUNT == 64
    assert commitment.capture_count == 64
    assert commitment.observation_count == 64 * len(helpfulness.MODEL_ARMS) == 384
    assert commitment.probe_count == 64
    assert commitment.leaf_count == 512
    assert len(validated.cases) == 64
    assert commitment.case_root_sha256s == tuple(
        case.commitment.case_root_sha256 for case in validated.cases
    )

    payload = {
        "capture_count": commitment.capture_count,
        "case_count": commitment.case_count,
        "case_root_sha256s": list(commitment.case_root_sha256s),
        "envelope_sha256": commitment.envelope_sha256,
        "leaf_count": commitment.leaf_count,
        "manifest_sha256": commitment.manifest_sha256,
        "observation_count": commitment.observation_count,
        "policy": commitment.policy,
        "probe_count": commitment.probe_count,
        "schema_version": commitment.schema_version,
    }
    assert (
        commitment.aggregate_root_sha256
        == hashlib.sha256(
            _AGGREGATE_ROOT_DOMAIN + _canonical_bytes(payload)
        ).hexdigest()
    )

    module_tree = ast.parse(Path(sidecar_join.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
                imported_modules.update(
                    f"{node.module}.{alias.name}" for alias in node.names
                )
    assert not any(
        forbidden_module in imported
        for imported in imported_modules
        for forbidden_module in (
            "infbridge_run_ledger",
            "infbridge_run_analyzer",
            "infbridge_receipt_join",
        )
    )


def test_inventory_requires_exact_tuple_order_count_and_scalar_types(
    aggregate_fixture: _AggregateFixture,
) -> None:
    sidecars = aggregate_fixture.sidecars
    reordered = (sidecars[1], sidecars[0], *sidecars[2:])
    dropped = sidecars[:-1]
    duplicated = (*sidecars[:-1], sidecars[0])
    bool_case = _replace_case(
        sidecars,
        0,
        replace(
            sidecars[0],
            capture=replace(sidecars[0].capture, case_index=False),
        ),
    )
    for mutant in (reordered, dropped, duplicated, bool_case):
        _assert_rejected(aggregate_fixture, mutant)
    with pytest.raises(sidecar_join.ModelSidecarJoinError):
        sidecar_join.validate_live_model_sidecar_aggregate_v1(
            aggregate_fixture.manifest,
            aggregate_fixture.tokenizer,
            aggregate_fixture.envelope,
            list(sidecars),  # type: ignore[arg-type]
        )


def test_aggregate_wire_budget_is_global_and_fail_closed(
    aggregate_fixture: _AggregateFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = sum(
        sidecar_join._aggregate_wire_byte_count(sidecar)
        for sidecar in aggregate_fixture.sidecars
    )
    monkeypatch.setattr(sidecar_join, "_MAX_AGGREGATE_WIRE_BYTES", total - 1)

    with pytest.raises(sidecar_join.ModelSidecarJoinError) as error:
        _validate(aggregate_fixture)

    assert error.value.reason == "closed_schema"


def test_cross_case_execution_session_run_process_and_component_reuse_is_closed(
    aggregate_fixture: _AggregateFixture,
) -> None:
    sidecars = aggregate_fixture.sidecars

    case_one = sidecars[1]
    observation = case_one.observations[0]
    request = aggregate_fixture.observation_requests[1][0]
    response = aggregate_fixture.observation_responses[1][0]
    reused_token = aggregate_fixture.observation_requests[0][0].execution_index
    reused_session, reused_run = helpfulness._opaque_future_identity(reused_token)
    token_request = replace(
        request,
        execution_index=reused_token,
        future_session_id=reused_session,
        future_run_id=reused_run,
    )
    token_response = replace(
        response,
        observation=replace(
            response.observation,
            execution_index=reused_token,
            future_session_id=reused_session,
            future_run_id=reused_run,
        ),
        state_receipt=replace(
            response.state_receipt,
            execution_index=reused_token,
            logical_session_id=reused_session,
            logical_run_id=reused_run,
        ),
    )
    token_case = replace(
        case_one,
        observations=(
            replace(
                observation,
                exchange=replace(
                    observation.exchange,
                    request_wire_utf8=helpfulness.wire_dumps(token_request).encode(
                        "utf-8"
                    ),
                    response_wire_utf8=helpfulness.wire_dumps(token_response).encode(
                        "utf-8"
                    ),
                ),
            ),
            *case_one.observations[1:],
        ),
    )

    capture_response = aggregate_fixture.capture_responses[1]
    reused_process = aggregate_fixture.capture_responses[0].process_instance_id
    process_case = replace(
        case_one,
        capture=replace(
            case_one.capture,
            exchange=_response_exchange(
                case_one.capture.exchange,
                replace(capture_response, process_instance_id=reused_process),
            ),
        ),
    )

    reused_component = aggregate_fixture.observation_responses[0][
        0
    ].state_receipt.store_instance_id
    component_response = replace(
        response,
        state_receipt=replace(
            response.state_receipt,
            store_instance_id=reused_component,
        ),
    )
    component_case = replace(
        case_one,
        observations=(
            replace(
                observation,
                exchange=_response_exchange(observation.exchange, component_response),
            ),
            *case_one.observations[1:],
        ),
    )

    for mutant in (token_case, process_case, component_case):
        _assert_rejected(aggregate_fixture, _replace_case(sidecars, 1, mutant))

    manifest_sha256, envelope_sha256 = sidecar_join._validate_sidecar_run_identity_v1(
        aggregate_fixture.manifest,
        aggregate_fixture.tokenizer,
        aggregate_fixture.envelope,
    )
    details = tuple(
        sidecar_join._validate_live_model_case_sidecar_against_run_v1(
            aggregate_fixture.manifest,
            sidecar,
            manifest_sha256=manifest_sha256,
            envelope_sha256=envelope_sha256,
        )
        for sidecar in sidecars
    )
    first_request = details[0].observation_rows[0][0]
    second_row = details[1].observation_rows[0]
    second_request = second_row[0]
    for field_name in ("execution_index", "future_session_id", "future_run_id"):
        mutated_request = replace(
            second_request,
            **{field_name: getattr(first_request, field_name)},
        )
        mutated_detail = replace(
            details[1],
            observation_rows=(
                (mutated_request, second_row[1], second_row[2]),
                *details[1].observation_rows[1:],
            ),
        )
        with pytest.raises(sidecar_join.ModelSidecarJoinError):
            sidecar_join._validate_aggregate_identity_separation_v1(
                aggregate_fixture.manifest,
                (details[0], mutated_detail, *details[2:]),
            )


def test_final_database_sweep_detects_post_validation_toctou(
    aggregate_fixture: _AggregateFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_validate = sidecar_join._validate_live_model_case_sidecar_against_run_v1
    first_path = Path(aggregate_fixture.capture_requests[0].database_path)
    original_bytes = first_path.read_bytes()
    mutation_done = False

    def validate_then_mutate(
        manifest: helpfulness.ModelRunManifest,
        sidecar: object,
        *,
        manifest_sha256: str,
        envelope_sha256: str,
    ) -> object:
        nonlocal mutation_done
        validated = real_validate(
            manifest,
            sidecar,
            manifest_sha256=manifest_sha256,
            envelope_sha256=envelope_sha256,
        )
        assert type(sidecar) is sidecar_join.ModelCaseSidecarV1
        if (
            sidecar.capture.case_index == helpfulness.MODEL_CASE_COUNT - 1
            and not mutation_done
        ):
            with first_path.open("ab") as stream:
                stream.write(b"post-validation-mutation")
            mutation_done = True
        return validated

    monkeypatch.setattr(
        sidecar_join,
        "_validate_live_model_case_sidecar_against_run_v1",
        validate_then_mutate,
    )
    try:
        with pytest.raises(sidecar_join.ModelSidecarJoinError):
            _validate(aggregate_fixture)
    finally:
        first_path.write_bytes(original_bytes)

    assert _validate(aggregate_fixture).commitment.case_count == 64
