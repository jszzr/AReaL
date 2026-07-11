# SPDX-License-Identifier: Apache-2.0

"""Tests for strict live-child evidence joined to one registered model case."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from examples.memory_service import infbridge_sidecar_join as sidecar_join
from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    DecoderAuditMaterialV2,
    RunEnvelopeV2,
    infbridge_run_envelope_v2_sha256,
    prepare_infbridge_run_envelope_v2,
)

from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_OPAQUE_TOKEN_BASE = (1 << 127) | 0x5A17_0000_0000_0000
_ORDINARY_MISSING_RELEASE_ID = "rel_000000000000000000000000"
_CASE_ROOT_DOMAIN = b"areal-memory-model-sidecar-case-root-v1\0"
_LEAF_DOMAINS = {
    "capture": b"areal-memory-model-sidecar-capture-leaf-v1\0",
    "observation": b"areal-memory-model-sidecar-observation-leaf-v1\0",
    "leakage_probe": b"areal-memory-model-sidecar-leakage-probe-leaf-v1\0",
}


class _SidecarByteTokenizer:
    _ARTIFACT_BYTES = b'sidecar-bytes-tokenizer-v2:{"vocabulary":"00-ff"}'
    _DECODER_STATE_BYTES = (
        b'sidecar-bytes-decoder-v2:{"skip_special_tokens":true,'
        b'"clean_up_tokenization_spaces":false}'
    )

    def memory_audit_material(self) -> DecoderAuditMaterialV2:
        return DecoderAuditMaterialV2(
            tokenizer_id="sidecar-byte-tokenizer-v2",
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
            raise ValueError("sidecar decode policy drift")
        return bytes(token_ids).decode("utf-8")


@dataclass(frozen=True, slots=True)
class _LiveModelCase:
    manifest: helpfulness.ModelRunManifest
    tokenizer: _SidecarByteTokenizer
    envelope: RunEnvelopeV2
    registration: helpfulness.ModelCaseRegistration
    sidecar: sidecar_join.ModelCaseSidecarV1
    capture_request: helpfulness.ModelCaptureChildRequest
    capture_response: helpfulness.ModelCaptureChildResponse
    observation_requests: tuple[helpfulness.ModelObservationChildRequest, ...]
    observation_responses: tuple[helpfulness.ModelObservationChildResponse, ...]
    probe_request: helpfulness.ModelLeakageProbeChildRequest
    probe_response: helpfulness.ModelLeakageProbeChildResponse


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _raw_exchange(
    request: helpfulness.ModelCaptureChildRequest
    | helpfulness.ModelObservationChildRequest
    | helpfulness.ModelLeakageProbeChildRequest,
    *,
    role: str,
) -> tuple[sidecar_join.CanonicalChildExchangeV1, object]:
    completed = helpfulness.run_isolated_child_raw(
        request,
        role=role,
        timeout_seconds=30,
    )
    assert completed.returncode == 0, completed.stderr
    response = helpfulness.wire_loads(completed.stdout)
    assert helpfulness.wire_dumps(response) == completed.stdout
    return (
        sidecar_join.CanonicalChildExchangeV1(
            request_wire_utf8=helpfulness.wire_dumps(request).encode("utf-8"),
            response_wire_utf8=completed.stdout.encode("utf-8"),
            launcher_pid=completed.pid,
        ),
        response,
    )


def _build_live_case(root: Path) -> _LiveModelCase:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    tokenizer = _SidecarByteTokenizer()
    prepared = helpfulness.prepare_model_run_manifest(
        tokenizer,
        generator_commit_sha="a" * 40,
        evaluator_commit_sha="b" * 40,
        model_id="sidecar-test-model",
        model_weights_sha256="c" * 64,
        tokenizer_id="sidecar-byte-tokenizer-v2",
        tokenizer_sha256=hashlib.sha256(tokenizer._ARTIFACT_BYTES).hexdigest(),
    )
    assert prepared.failure is None
    assert prepared.manifest is not None
    manifest = prepared.manifest
    bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://sidecar-model.test/",
        pause_state=PauseState(),
        request_timeout=7.0,
        max_resubmit_retries=2,
        resubmit_wait=0.0,
        version=41,
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

    registration = manifest.cases[0]
    database_path = root / "case-000.sqlite3"
    capture_request = helpfulness.ModelCaptureChildRequest(
        case_index=0,
        model_attempt=registration.model_attempt,
        case_manifest_sha256=registration.identity.case_manifest_sha256,
        database_path=str(database_path),
    )
    capture_exchange, raw_capture_response = _raw_exchange(
        capture_request,
        role="capture-child",
    )
    assert type(raw_capture_response) is helpfulness.ModelCaptureChildResponse
    capture_response = raw_capture_response

    observation_sidecars: list[sidecar_join.ModelObservationSidecarV1] = []
    observation_requests: list[helpfulness.ModelObservationChildRequest] = []
    observation_responses: list[helpfulness.ModelObservationChildResponse] = []
    for slot_index, arm_call in enumerate(registration.arm_calls):
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
        exchange, raw_response = _raw_exchange(
            request,
            role="model-observation-child",
        )
        assert type(raw_response) is helpfulness.ModelObservationChildResponse
        response = raw_response
        observation_requests.append(request)
        observation_responses.append(response)
        observation_sidecars.append(
            sidecar_join.ModelObservationSidecarV1(
                slot_index=slot_index,
                case_index=0,
                arm=arm_call.arm,
                exchange=exchange,
            )
        )

    probe_token = _OPAQUE_TOKEN_BASE + len(registration.arm_calls)
    probe_request = helpfulness.ModelLeakageProbeChildRequest(
        execution_index=probe_token,
        database_path=str(database_path),
        database_receipt=capture_response.database_receipt,
        scope=capture_response.references.capture.local_scope,
        release_id=(capture_response.references.releases.foreign_sentinel_release_id),
    )
    probe_exchange, raw_probe_response = _raw_exchange(
        probe_request,
        role="model-leakage-probe-child",
    )
    assert type(raw_probe_response) is helpfulness.ModelLeakageProbeChildResponse
    probe_response = raw_probe_response
    sidecar = sidecar_join.ModelCaseSidecarV1(
        capture=sidecar_join.ModelCaptureSidecarV1(
            case_index=0,
            exchange=capture_exchange,
        ),
        observations=tuple(observation_sidecars),
        probe=sidecar_join.ModelProbeSidecarV1(
            case_index=0,
            logical_execution_index=helpfulness.MODEL_CASE_COUNT
            * len(helpfulness.MODEL_ARMS),
            exchange=probe_exchange,
        ),
    )
    return _LiveModelCase(
        manifest=manifest,
        tokenizer=tokenizer,
        envelope=envelope,
        registration=registration,
        sidecar=sidecar,
        capture_request=capture_request,
        capture_response=capture_response,
        observation_requests=tuple(observation_requests),
        observation_responses=tuple(observation_responses),
        probe_request=probe_request,
        probe_response=probe_response,
    )


@pytest.fixture(scope="module")
def live_model_case(tmp_path_factory: pytest.TempPathFactory) -> _LiveModelCase:
    return _build_live_case(tmp_path_factory.mktemp("infbridge-sidecar"))


def _validate(
    live: _LiveModelCase,
    sidecar: sidecar_join.ModelCaseSidecarV1 | None = None,
) -> sidecar_join.ValidatedModelCaseSidecarV1:
    return sidecar_join.validate_live_model_case_sidecar_v1(
        live.manifest,
        live.tokenizer,
        live.envelope,
        live.sidecar if sidecar is None else sidecar,
    )


def _replace_observation(
    sidecar: sidecar_join.ModelCaseSidecarV1,
    index: int,
    value: sidecar_join.ModelObservationSidecarV1,
) -> sidecar_join.ModelCaseSidecarV1:
    observations = list(sidecar.observations)
    observations[index] = value
    return replace(sidecar, observations=tuple(observations))


def _exchange_with(
    exchange: sidecar_join.CanonicalChildExchangeV1,
    *,
    request: object | None = None,
    response: object | None = None,
    launcher_pid: int | None = None,
) -> sidecar_join.CanonicalChildExchangeV1:
    return replace(
        exchange,
        request_wire_utf8=(
            exchange.request_wire_utf8
            if request is None
            else helpfulness.wire_dumps(request).encode("utf-8")
        ),
        response_wire_utf8=(
            exchange.response_wire_utf8
            if response is None
            else helpfulness.wire_dumps(response).encode("utf-8")
        ),
        launcher_pid=(exchange.launcher_pid if launcher_pid is None else launcher_pid),
    )


def _assert_rejected(
    live: _LiveModelCase,
    sidecar: sidecar_join.ModelCaseSidecarV1,
) -> None:
    with pytest.raises(sidecar_join.ModelSidecarJoinError):
        _validate(live, sidecar)


def test_live_case_validates_deterministically_and_binds_rendered_contexts(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    first = _validate(live)
    second = _validate(live)

    assert first == second
    assert first.commitment == second.commitment
    assert first.commitment.schema_version == 1
    assert first.commitment.policy == "live-honest-child-consistency-only-v1"
    assert first.commitment.case_index == 0
    assert first.commitment.manifest_sha256 == (
        helpfulness.model_run_manifest_sha256(live.manifest, live.tokenizer)
    )
    assert first.commitment.envelope_sha256 == (
        infbridge_run_envelope_v2_sha256(live.envelope)
    )
    assert len(first.leaves) == 8
    assert len(first.commitment.observation_leaf_sha256s) == len(helpfulness.MODEL_ARMS)
    assert first.commitment.capture_leaf_sha256 == first.leaves[0].leaf_sha256
    assert first.commitment.observation_leaf_sha256s == tuple(
        leaf.leaf_sha256 for leaf in first.leaves[1:-1]
    )
    assert first.commitment.probe_leaf_sha256 == first.leaves[-1].leaf_sha256

    process_ids = {live.capture_response.process_instance_id}
    for arm_call, record, response in zip(
        live.registration.arm_calls,
        live.sidecar.observations,
        live.observation_responses,
        strict=True,
    ):
        assert record.arm == arm_call.arm
        assert response.observation.rendered_context_sha256 == (
            arm_call.rendered_context_sha256
        )
        assert response.observation.rendered_context_utf8_bytes == (
            arm_call.rendered_context_utf8_bytes
        )
        process_ids.add(response.process_instance_id)
    process_ids.add(live.probe_response.process_instance_id)
    assert len(process_ids) == 8
    assert all(response.pid != os.getpid() for response in live.observation_responses)


def test_observation_inventory_is_ordered_and_duplicate_closed(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    observations = live.sidecar.observations
    reordered = replace(
        live.sidecar,
        observations=(observations[1], observations[0], *observations[2:]),
    )
    dropped = replace(live.sidecar, observations=observations[:-1])
    duplicated = replace(
        live.sidecar,
        observations=(*observations[:-1], observations[0]),
    )

    by_arm = {record.arm: index for index, record in enumerate(observations)}
    current_index = by_arm["current_release"]
    oracle_index = by_arm["oracle"]
    current = observations[current_index]
    oracle = observations[oracle_index]
    assert live.registration.arm_calls[current_index].rendered_context_sha256 == (
        live.registration.arm_calls[oracle_index].rendered_context_sha256
    )
    equal_bytes_splice = _replace_observation(
        live.sidecar,
        current_index,
        replace(current, exchange=oracle.exchange),
    )

    for mutant in (reordered, dropped, duplicated, equal_bytes_splice):
        _assert_rejected(live, mutant)


def test_sidecar_metadata_requires_exact_scalar_types(
    live_model_case: _LiveModelCase,
) -> None:
    class _IntAlias(int):
        pass

    class _StringAlias(str):
        pass

    live = live_model_case
    first = live.sidecar.observations[0]
    second = live.sidecar.observations[1]
    mutants = (
        replace(live.sidecar, capture=replace(live.sidecar.capture, case_index=False)),
        _replace_observation(
            live.sidecar,
            0,
            replace(first, slot_index=False),
        ),
        _replace_observation(
            live.sidecar,
            0,
            replace(first, case_index=False),
        ),
        _replace_observation(
            live.sidecar,
            1,
            replace(second, slot_index=True),
        ),
        _replace_observation(
            live.sidecar,
            0,
            replace(first, arm=_StringAlias(first.arm)),
        ),
        replace(
            live.sidecar,
            probe=replace(live.sidecar.probe, case_index=False),
        ),
        replace(
            live.sidecar,
            probe=replace(
                live.sidecar.probe,
                logical_execution_index=_IntAlias(
                    live.sidecar.probe.logical_execution_index
                ),
            ),
        ),
    )

    for mutant in mutants:
        _assert_rejected(live, mutant)


def test_capture_identity_receipt_and_database_path_are_cross_bound(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    capture = live.sidecar.capture
    request = live.capture_request
    response = live.capture_response
    wrong_path = str(Path(request.database_path).with_name("other.sqlite3"))
    mutants = (
        replace(
            live.sidecar,
            capture=replace(
                capture,
                exchange=_exchange_with(
                    capture.exchange,
                    request=replace(request, database_path=wrong_path),
                ),
            ),
        ),
        replace(
            live.sidecar,
            capture=replace(
                capture,
                exchange=_exchange_with(
                    capture.exchange,
                    response=replace(
                        response,
                        database_receipt=replace(
                            response.database_receipt,
                            sha256="0" * 64,
                        ),
                    ),
                ),
            ),
        ),
        replace(
            live.sidecar,
            capture=replace(
                capture,
                exchange=_exchange_with(
                    capture.exchange,
                    response=replace(response, case_manifest_sha256="0" * 64),
                ),
            ),
        ),
    )

    for mutant in mutants:
        _assert_rejected(live, mutant)


def test_observation_source_arm_token_process_and_state_splices_are_rejected(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    first = live.sidecar.observations[0]
    second = live.sidecar.observations[1]
    first_request = live.observation_requests[0]
    first_response = live.observation_responses[0]
    second_request = live.observation_requests[1]
    second_response = live.observation_responses[1]

    mutations = (
        replace(first, arm=second.arm),
        replace(
            first,
            exchange=_exchange_with(
                first.exchange,
                request=replace(first_request, source=second_request.source),
            ),
        ),
        replace(
            first,
            exchange=_exchange_with(
                first.exchange,
                request=replace(
                    first_request,
                    execution_index=first_request.execution_index + 31,
                ),
            ),
        ),
        replace(
            first,
            exchange=_exchange_with(
                first.exchange,
                response=replace(
                    first_response,
                    process_instance_id=second_response.process_instance_id,
                ),
            ),
        ),
        replace(
            first,
            exchange=_exchange_with(
                first.exchange,
                launcher_pid=second.exchange.launcher_pid,
            ),
        ),
        replace(
            first,
            exchange=_exchange_with(
                first.exchange,
                response=replace(
                    first_response,
                    state_receipt=second_response.state_receipt,
                ),
            ),
        ),
        replace(first, exchange=second.exchange),
    )

    for mutation in mutations:
        _assert_rejected(live, _replace_observation(live.sidecar, 0, mutation))


def test_probe_assignment_is_closed_but_a_found_outcome_is_valid_evidence(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    probe = live.sidecar.probe
    request = live.probe_request
    response = live.probe_response
    assert response.probe.outcome == "release_not_found"
    assert _ORDINARY_MISSING_RELEASE_ID != request.release_id

    ordinary_missing = replace(
        live.sidecar,
        probe=replace(
            probe,
            exchange=_exchange_with(
                probe.exchange,
                request=replace(request, release_id=_ORDINARY_MISSING_RELEASE_ID),
                response=replace(
                    response,
                    probe=replace(
                        response.probe,
                        release_id=_ORDINARY_MISSING_RELEASE_ID,
                    ),
                ),
            ),
        ),
    )
    wrong_scope = replace(
        live.sidecar,
        probe=replace(
            probe,
            exchange=_exchange_with(
                probe.exchange,
                request=replace(
                    request,
                    scope=live.capture_response.references.capture.foreign_scope,
                ),
                response=replace(
                    response,
                    probe=replace(
                        response.probe,
                        requested_scope=(
                            live.capture_response.references.capture.foreign_scope
                        ),
                    ),
                ),
            ),
        ),
    )
    wrong_release = replace(
        live.sidecar,
        probe=replace(
            probe,
            exchange=_exchange_with(
                probe.exchange,
                request=replace(
                    request,
                    release_id=(
                        live.capture_response.references.releases.current_release_id
                    ),
                ),
                response=replace(
                    response,
                    probe=replace(
                        response.probe,
                        release_id=(
                            live.capture_response.references.releases.current_release_id
                        ),
                    ),
                ),
            ),
        ),
    )
    for mutant in (ordinary_missing, wrong_scope, wrong_release):
        _assert_rejected(live, mutant)

    found_sidecar = replace(
        live.sidecar,
        probe=replace(
            probe,
            exchange=_exchange_with(
                probe.exchange,
                response=replace(
                    response,
                    probe=replace(response.probe, outcome="release_found"),
                ),
            ),
        ),
    )
    found = _validate(live, found_sidecar)
    baseline = _validate(live)
    found_response = helpfulness.wire_loads(
        found.probe.exchange.response_wire_utf8.decode("utf-8")
    )
    assert type(found_response) is helpfulness.ModelLeakageProbeChildResponse
    assert found_response.probe.outcome == "release_found"
    assert found.commitment.probe_leaf_sha256 != (baseline.commitment.probe_leaf_sha256)
    assert found.commitment.case_root_sha256 != baseline.commitment.case_root_sha256


def test_leaf_wire_commitments_and_case_root_are_exact(
    live_model_case: _LiveModelCase,
) -> None:
    live = live_model_case
    validated = _validate(live)
    exchanges = (
        live.sidecar.capture.exchange,
        *(record.exchange for record in live.sidecar.observations),
        live.sidecar.probe.exchange,
    )
    request_types = (
        "model_capture_child_request",
        *("model_observation_child_request" for _ in live.observation_requests),
        "model_leakage_probe_child_request",
    )
    response_types = (
        "model_capture_child_response",
        *("model_observation_child_response" for _ in live.observation_responses),
        "model_leakage_probe_child_response",
    )

    for leaf, exchange, request_type, response_type in zip(
        validated.leaves,
        exchanges,
        request_types,
        response_types,
        strict=True,
    ):
        assert leaf.launcher_pid == exchange.launcher_pid
        assert leaf.request.wire_type == request_type
        assert leaf.request.byte_count == len(exchange.request_wire_utf8)
        assert (
            leaf.request.sha256
            == hashlib.sha256(exchange.request_wire_utf8).hexdigest()
        )
        assert leaf.response.wire_type == response_type
        assert leaf.response.byte_count == len(exchange.response_wire_utf8)
        assert (
            leaf.response.sha256
            == hashlib.sha256(exchange.response_wire_utf8).hexdigest()
        )
        leaf_payload = {
            "arm": leaf.arm,
            "case_index": leaf.case_index,
            "launcher_pid": leaf.launcher_pid,
            "logical_index": leaf.logical_index,
            "opaque_execution_token_hex": leaf.opaque_execution_token_hex,
            "request": {
                "byte_count": leaf.request.byte_count,
                "sha256": leaf.request.sha256,
                "wire_type": leaf.request.wire_type,
            },
            "response": {
                "byte_count": leaf.response.byte_count,
                "sha256": leaf.response.sha256,
                "wire_type": leaf.response.wire_type,
            },
            "role": leaf.role,
            "schema_version": 1,
        }
        assert (
            leaf.leaf_sha256
            == hashlib.sha256(
                _LEAF_DOMAINS[leaf.role] + _canonical_bytes(leaf_payload)
            ).hexdigest()
        )

    assert tuple(leaf.role for leaf in validated.leaves) == (
        "capture",
        *("observation" for _ in helpfulness.MODEL_ARMS),
        "leakage_probe",
    )
    assert tuple(leaf.logical_index for leaf in validated.leaves) == (
        0,
        *range(len(helpfulness.MODEL_ARMS)),
        helpfulness.MODEL_CASE_COUNT * len(helpfulness.MODEL_ARMS),
    )
    assert tuple(leaf.arm for leaf in validated.leaves) == (
        None,
        *(call.arm for call in live.registration.arm_calls),
        None,
    )
    assert tuple(leaf.opaque_execution_token_hex for leaf in validated.leaves) == (
        None,
        *(f"{request.execution_index:032x}" for request in live.observation_requests),
        f"{live.probe_request.execution_index:032x}",
    )

    commitment = validated.commitment
    root_payload = {
        "case_index": commitment.case_index,
        "capture_leaf_sha256": commitment.capture_leaf_sha256,
        "envelope_sha256": commitment.envelope_sha256,
        "leaf_count": len(validated.leaves),
        "manifest_sha256": commitment.manifest_sha256,
        "observation_leaf_sha256s": list(commitment.observation_leaf_sha256s),
        "policy": commitment.policy,
        "probe_leaf_sha256": commitment.probe_leaf_sha256,
        "schema_version": commitment.schema_version,
    }
    assert (
        commitment.case_root_sha256
        == hashlib.sha256(
            _CASE_ROOT_DOMAIN + _canonical_bytes(root_payload)
        ).hexdigest()
    )

    first = live.sidecar.observations[0]
    tampered_wire = bytearray(first.exchange.request_wire_utf8)
    tampered_wire[-2] ^= 1
    bit_flip = _replace_observation(
        live.sidecar,
        0,
        replace(
            first,
            exchange=replace(
                first.exchange,
                request_wire_utf8=bytes(tampered_wire),
            ),
        ),
    )
    _assert_rejected(live, bit_flip)
