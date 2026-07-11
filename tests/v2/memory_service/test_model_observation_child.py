# SPDX-License-Identifier: Apache-2.0

"""Tests for consumer- and model-output-free Memory observation children."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness

_ATTEMPT_ONE_SHA256 = "1d840c53233cc9fbfe2454b64798357e06f0fd9e14e74160f4b0f15fdeaebe26"
_EXECUTION_TOKEN = (1 << 127) | 0x123456789ABCDEF


def _canonical_wire(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _all_mapping_keys(value: object) -> set[str]:
    if type(value) is dict:
        result = set(value)
        for part in value.values():
            result.update(_all_mapping_keys(part))
        return result
    if type(value) is list:
        result: set[str] = set()
        for part in value:
            result.update(_all_mapping_keys(part))
        return result
    return set()


def _build_setup(
    root: Path,
    *,
    execution_token: int = _EXECUTION_TOKEN,
) -> tuple[
    helpfulness.CodebookCase,
    helpfulness.ModelCaptureChildResponse,
    helpfulness.ModelObservationChildRequest,
    helpfulness.ParentScheduleItem,
]:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    case = helpfulness._generate_model_candidate(0, model_attempt=1)
    assert case is not None
    assert helpfulness.case_manifest_sha256(case) == _ATTEMPT_ONE_SHA256
    database_path = root / "case.sqlite3"
    capture = helpfulness.execute_model_capture_child_request(
        helpfulness.ModelCaptureChildRequest(
            case_index=0,
            model_attempt=1,
            case_manifest_sha256=_ATTEMPT_ONE_SHA256,
            database_path=str(database_path),
        )
    )
    source = helpfulness._fast_source_spec(
        case,
        capture.references,
        "current_release",
    )
    future_session_id, future_run_id = helpfulness._opaque_future_identity(
        execution_token
    )
    request = helpfulness.ModelObservationChildRequest(
        execution_index=execution_token,
        database_path=str(database_path),
        database_receipt=capture.database_receipt,
        scope=capture.references.capture.local_scope,
        source=source,
        future_session_id=future_session_id,
        future_run_id=future_run_id,
        renderer_version="memory-codebook/v1",
    )
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=execution_token,
        case=case,
        references=capture.references,
        arm="current_release",
    )
    return case, capture, request, schedule


@pytest.fixture(scope="module")
def observation_setup(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    helpfulness.CodebookCase,
    helpfulness.ModelCaptureChildResponse,
    helpfulness.ModelObservationChildRequest,
    helpfulness.ParentScheduleItem,
]:
    return _build_setup(tmp_path_factory.mktemp("model-observation"))


def test_model_observation_wire_has_no_scorer_or_model_output(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
) -> None:
    _case, _capture, request, _schedule = observation_setup
    response = helpfulness.execute_model_observation_child_request(request)
    forbidden_keys = {
        "arm",
        "capture_pid",
        "capture_process_instance_id",
        "case_index",
        "consumer_input_receipt",
        "consumer_version",
        "expected_response",
        "history",
        "history_length",
        "model_call_receipt",
        "normalized_response",
        "query",
        "query_sha256",
        "rendered_context_token_count",
        "response",
        "utility",
    }

    for value in (request, response):
        encoded = helpfulness.wire_dumps(value)
        decoded = helpfulness.wire_loads(encoded)
        payload = json.loads(encoded)["payload"]

        assert type(decoded) is type(value)
        assert decoded == value
        assert helpfulness.wire_dumps(decoded) == encoded
        assert encoded.count("\n") == 1
        assert not forbidden_keys.intersection(_all_mapping_keys(payload))

    encoded_request = json.loads(helpfulness.wire_dumps(request))["payload"]
    encoded_response = json.loads(helpfulness.wire_dumps(response))["payload"]
    assert set(encoded_request) == {
        "database_path",
        "database_receipt",
        "execution_index",
        "future_run_id",
        "future_session_id",
        "renderer_version",
        "scope",
        "source",
    }
    assert set(encoded_response) == {
        "areal_module_path",
        "database_receipt",
        "environment_clean",
        "isolated_mode",
        "observation",
        "pid",
        "process_instance_id",
        "state_receipt",
        "visible_forbidden_environment",
    }
    assert set(encoded_response["observation"]) == {
        "eligible_ids",
        "entries",
        "execution_index",
        "future_pid",
        "future_process_instance_id",
        "future_run_id",
        "future_session_id",
        "reader_audit",
        "release_id",
        "rendered_context_sha256",
        "rendered_context_utf8_bytes",
        "retrieved_ids",
        "returned_ids",
        "scope",
        "source_evidence_ids",
        "source_kind",
    }
    assert set(encoded_response["state_receipt"]) == {
        "audit_instance_id",
        "execution_index",
        "generation_index",
        "logical_run_id",
        "logical_session_id",
        "logical_session_instance_id",
        "reader_instance_id",
        "renderer_instance_id",
        "resolver_instance_id",
        "store_instance_id",
    }

    for value in (request, response, response.observation, response.state_receipt):
        assert forbidden_keys.isdisjoint(field.name for field in fields(value))

    request_wire = json.loads(helpfulness.wire_dumps(request))
    response_wire = json.loads(helpfulness.wire_dumps(response))
    mutants: list[dict[str, object]] = []
    unknown = json.loads(json.dumps(request_wire))
    unknown["payload"]["response"] = "forbidden"
    mutants.append(unknown)
    missing = json.loads(json.dumps(request_wire))
    del missing["payload"]["database_receipt"]
    mutants.append(missing)
    for execution_index in (True, 1.0, 0):
        mutant = json.loads(json.dumps(request_wire))
        mutant["payload"]["execution_index"] = execution_index
        mutants.append(mutant)
    bad_digest = json.loads(json.dumps(request_wire))
    bad_digest["payload"]["database_receipt"]["sha256"] = "A" * 64
    mutants.append(bad_digest)
    injected_answer = json.loads(json.dumps(response_wire))
    injected_answer["payload"]["observation"]["response"] = "forbidden"
    mutants.append(injected_answer)
    bad_generation = json.loads(json.dumps(response_wire))
    bad_generation["payload"]["state_receipt"]["generation_index"] = 1
    mutants.append(bad_generation)

    for mutant in mutants:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_loads(_canonical_wire(mutant))
        assert error.value.reason == "closed_schema"

    first_entry = response.observation.entries[0]
    first_audit = response.observation.reader_audit[0]
    in_memory_mutants = (
        replace(response, visible_forbidden_environment=[]),  # type: ignore[arg-type]
        replace(
            response,
            observation=replace(
                response.observation,
                entries=(
                    replace(first_entry, evidence_ids=[]),  # type: ignore[arg-type]
                    *response.observation.entries[1:],
                ),
            ),
        ),
        replace(
            response,
            observation=replace(
                response.observation,
                reader_audit=(
                    replace(first_audit, requested_ids=[]),  # type: ignore[arg-type]
                    *response.observation.reader_audit[1:],
                ),
            ),
        ),
    )
    for mutant in in_memory_mutants:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_dumps(mutant)
        assert error.value.reason == "closed_schema"


def test_model_observation_executes_reader_and_renderer_without_answer_path(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, capture, request, schedule = observation_setup
    forbidden_calls = Counter()

    def forbidden(*_args, **_kwargs):
        forbidden_calls["answer_path"] += 1
        raise AssertionError("observation-only child reached an answer path")

    for name in (
        "_ItemConsumer",
        "analyze_model_run",
        "compose_model_prompt",
        "consume_scripted",
        "normalize_response",
        "parent_join_and_score",
        "prepare_model_call",
        "submit_model_call",
    ):
        monkeypatch.setattr(helpfulness, name, forbidden)

    response = helpfulness.execute_model_observation_child_request(request)
    observation = response.observation
    expected = schedule.expected_source

    assert forbidden_calls == Counter()
    assert response.database_receipt == capture.database_receipt
    assert observation.eligible_ids == expected.eligible_ids
    assert observation.retrieved_ids == expected.retrieved_ids
    assert observation.returned_ids == expected.returned_ids
    assert observation.source_evidence_ids == expected.source_evidence_ids
    assert observation.entries == expected.entries
    assert observation.reader_audit == expected.reader_audit
    assert observation.rendered_context_sha256 == expected.rendered_context_sha256
    assert (
        observation.rendered_context_utf8_bytes == expected.rendered_context_utf8_bytes
    )
    assert helpfulness._model_observation_resolved_entries(observation)
    state_fields = {field.name for field in fields(response.state_receipt)}
    assert state_fields.isdisjoint(
        {"consumer_instance_id", "history_instance_id", "history_length"}
    )


@pytest.mark.parametrize("arm", helpfulness.MODEL_ARMS)
def test_model_observation_reproduces_every_model_arm_source(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    arm: str,
) -> None:
    case, capture, base_request, _schedule = observation_setup
    arm_offset = helpfulness.MODEL_ARMS.index(arm)
    token = _EXECUTION_TOKEN + 100 + arm_offset
    future_session_id, future_run_id = helpfulness._opaque_future_identity(token)
    request = replace(
        base_request,
        execution_index=token,
        source=helpfulness._fast_source_spec(case, capture.references, arm),
        future_session_id=future_session_id,
        future_run_id=future_run_id,
    )
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=token,
        case=case,
        references=capture.references,
        arm=arm,
    )

    response = helpfulness.execute_model_observation_child_request(request)
    helpfulness._validate_model_observation_child_assignment(response, request)

    assert response.observation.reader_audit == schedule.expected_source.reader_audit
    assert response.observation.entries == schedule.expected_source.entries
    assert (
        response.observation.rendered_context_sha256
        == schedule.expected_source.rendered_context_sha256
    )
    assert (
        response.observation.rendered_context_utf8_bytes
        == schedule.expected_source.rendered_context_utf8_bytes
    )


def test_model_observation_rejects_unrenderable_oracle_before_database_access(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, capture, base_request, _schedule = observation_setup
    oracle = helpfulness._fast_source_spec(case, capture.references, "oracle")
    malformed = replace(
        base_request,
        source=replace(
            oracle,
            oracle_entries=(
                replace(oracle.oracle_entries[0], key="INVALID KEY"),
                *oracle.oracle_entries[1:],
            ),
        ),
    )
    calls = Counter()

    def forbidden_receipt(*_args, **_kwargs):
        calls["database"] += 1
        raise AssertionError("invalid oracle reached database access")

    monkeypatch.setattr(
        helpfulness,
        "_model_capture_database_receipt",
        forbidden_receipt,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_observation_child_request(malformed)

    assert error.value.reason == "assignment_mismatch"
    assert calls == Counter()


def test_model_observation_normalizes_storage_and_renderer_failures(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup

    def fail_storage(*_args, **_kwargs):
        raise helpfulness.MemoryPersistenceError("test storage drift")

    monkeypatch.setattr(
        helpfulness,
        "_new_model_observation_state",
        fail_storage,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as storage_error:
        helpfulness.execute_model_observation_child_request(request)
    assert storage_error.value.reason == "state_reuse"

    monkeypatch.undo()

    def fail_renderer(*_args, **_kwargs):
        raise ValueError("test renderer drift")

    monkeypatch.setattr(helpfulness._ItemRenderer, "run", fail_renderer)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as render_error:
        helpfulness.execute_model_observation_child_request(request)
    assert render_error.value.reason == "assignment_mismatch"


def test_model_observation_uses_fresh_real_isolated_processes(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, capture, first_request, _schedule = observation_setup
    second_token = _EXECUTION_TOKEN + 1
    second_session, second_run = helpfulness._opaque_future_identity(second_token)
    second_request = replace(
        first_request,
        execution_index=second_token,
        future_session_id=second_session,
        future_run_id=second_run,
    )
    launches: list[tuple[list[str], int]] = []
    original_popen = helpfulness.subprocess.Popen
    parent_store_calls = Counter()

    def forbidden_parent_store(*_args, **_kwargs):
        parent_store_calls["store"] += 1
        raise AssertionError("parent opened the captured case SQLite database")

    def record_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launches.append((list(args[0]), process.pid))
        return process

    monkeypatch.setattr(helpfulness.subprocess, "Popen", record_popen)
    monkeypatch.setattr(helpfulness, "SQLiteMemoryStore", forbidden_parent_store)
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-pythonpath")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-exec")
    monkeypatch.setenv("AREAL_TASK6_UNKNOWN_CANARY", "must-not-cross-exec")

    first = helpfulness.run_isolated_child(
        first_request,
        role="model-observation-child",
        timeout_seconds=20,
    )
    second = helpfulness.run_isolated_child(
        second_request,
        role="model-observation-child",
        timeout_seconds=20,
    )

    assert type(first) is helpfulness.ModelObservationChildResponse
    assert type(second) is helpfulness.ModelObservationChildResponse
    assert (
        first.database_receipt == second.database_receipt == (capture.database_receipt)
    )
    assert parent_store_calls == Counter()
    assert launches == [
        (
            [
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                "model-observation-child",
            ],
            first.pid,
        ),
        (
            [
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                "model-observation-child",
            ],
            second.pid,
        ),
    ]
    assert first.pid != os.getpid()
    assert second.pid != os.getpid()
    assert first.observation.future_pid == first.pid
    assert second.observation.future_pid == second.pid
    assert first.observation.future_process_instance_id == first.process_instance_id
    assert second.observation.future_process_instance_id == second.process_instance_id
    assert str(uuid.UUID(first.process_instance_id, version=4)) == (
        first.process_instance_id
    )
    assert str(uuid.UUID(second.process_instance_id, version=4)) == (
        second.process_instance_id
    )
    assert (
        len(
            {
                capture.process_instance_id,
                first.process_instance_id,
                second.process_instance_id,
            }
        )
        == 3
    )
    first_state = {
        value
        for field in fields(first.state_receipt)
        if field.name.endswith("_instance_id")
        for value in (getattr(first.state_receipt, field.name),)
    }
    second_state = {
        value
        for field in fields(second.state_receipt)
        if field.name.endswith("_instance_id")
        for value in (getattr(second.state_receipt, field.name),)
    }
    assert len(first_state) == len(second_state) == 6
    assert first_state.isdisjoint(second_state)
    for response in (first, second):
        assert response.isolated_mode is True
        assert response.visible_forbidden_environment == ()
        assert response.environment_clean is True


def test_model_observation_rejects_aliased_in_process_state(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup
    original_new_state = helpfulness._new_model_observation_state

    def aliased_state(observation_request):
        state = original_new_state(observation_request)
        state.renderer = state.resolver
        state.receipt = replace(
            state.receipt,
            renderer_instance_id=helpfulness._state_identity(
                generation_index=state.generation_index,
                component="renderer",
                value=state.renderer,
            ),
        )
        return state

    monkeypatch.setattr(
        helpfulness,
        "_new_model_observation_state",
        aliased_state,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_observation_child_request(request)

    assert error.value.reason == "state_reuse"


def test_model_observation_parent_rejects_forged_process_source_and_state(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup
    observed = helpfulness.execute_model_observation_child_request(request)
    process_id = "00000000-0000-4000-8000-000000000099"
    base = replace(
        observed,
        pid=12_345,
        process_instance_id=process_id,
        isolated_mode=True,
        areal_module_path=str(
            Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
        ),
        visible_forbidden_environment=(),
        environment_clean=True,
        observation=replace(
            observed.observation,
            future_pid=12_345,
            future_process_instance_id=process_id,
        ),
    )
    alternate_scope = replace(
        request.scope,
        subject_id=f"{request.scope.subject_id}-forged",
    )
    alternate_session, alternate_run = helpfulness._opaque_future_identity(
        request.execution_index + 2
    )
    forged_revision_hash = "1" * 64
    forged_revision_id = f"rev_{forged_revision_hash[:24]}"
    forged_revision_ids = (
        forged_revision_id,
        *base.observation.eligible_ids[1:],
    )
    forged_release_hash = hashlib.sha256(
        helpfulness.ReleaseManifest(
            scope=base.observation.scope,
            revision_ids=forged_revision_ids,
        ).canonical_bytes()
    ).hexdigest()
    self_consistent_wrong_release_anchor = replace(
        base,
        observation=replace(
            base.observation,
            eligible_ids=forged_revision_ids,
            retrieved_ids=forged_revision_ids,
            returned_ids=forged_revision_ids,
            entries=(
                replace(
                    base.observation.entries[0],
                    revision_id=forged_revision_id,
                ),
                *base.observation.entries[1:],
            ),
            reader_audit=(
                replace(
                    base.observation.reader_audit[0],
                    returned_content_hashes=(forged_release_hash,),
                ),
                replace(
                    base.observation.reader_audit[1],
                    requested_ids=(forged_revision_id,),
                    returned_record_ids=(forged_revision_id,),
                    returned_content_hashes=(forged_revision_hash,),
                ),
                *base.observation.reader_audit[2:],
            ),
        ),
    )
    first_revision_event = base.observation.reader_audit[1]
    duplicate_revision_ids = (
        base.observation.eligible_ids[0],
        base.observation.eligible_ids[0],
        *base.observation.eligible_ids[2:],
    )
    duplicate_revision_response = replace(
        base,
        observation=replace(
            base.observation,
            eligible_ids=duplicate_revision_ids,
            retrieved_ids=duplicate_revision_ids,
            returned_ids=duplicate_revision_ids,
            entries=(
                base.observation.entries[0],
                replace(
                    base.observation.entries[1],
                    revision_id=base.observation.entries[0].revision_id,
                ),
                *base.observation.entries[2:],
            ),
            reader_audit=(
                *base.observation.reader_audit[:3],
                replace(
                    base.observation.reader_audit[3],
                    requested_ids=first_revision_event.requested_ids,
                    returned_record_ids=first_revision_event.returned_record_ids,
                    returned_content_hashes=first_revision_event.returned_content_hashes,
                ),
                *base.observation.reader_audit[4:],
            ),
        ),
    )
    variants = (
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    rendered_context_sha256="0" * 64,
                ),
            ),
            "assignment_mismatch",
        ),
        (self_consistent_wrong_release_anchor, "assignment_mismatch"),
        (duplicate_revision_response, "assignment_mismatch"),
        (
            replace(
                base,
                observation=replace(base.observation, reader_audit=()),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    eligible_ids=(*base.observation.eligible_ids, "rev_forged"),
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    entries=(
                        replace(base.observation.entries[0], source_kind="oracle"),
                        *base.observation.entries[1:],
                    ),
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    reader_audit=(
                        replace(
                            base.observation.reader_audit[0],
                            requested_ids=(),
                        ),
                        *base.observation.reader_audit[1:],
                    ),
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    reader_audit=(
                        replace(
                            base.observation.reader_audit[0],
                            returned_content_hashes=("0" * 64,),
                        ),
                        *base.observation.reader_audit[1:],
                    ),
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                state_receipt=replace(
                    base.state_receipt,
                    reader_instance_id=base.state_receipt.store_instance_id,
                ),
            ),
            "state_reuse",
        ),
        (
            replace(
                base,
                state_receipt=replace(
                    base.state_receipt,
                    execution_index=request.execution_index + 1,
                ),
            ),
            "state_reuse",
        ),
        (
            replace(
                base,
                state_receipt=replace(
                    base.state_receipt,
                    logical_session_id=alternate_session,
                ),
            ),
            "state_reuse",
        ),
        (
            replace(
                base,
                state_receipt=replace(
                    base.state_receipt,
                    logical_run_id=alternate_run,
                ),
            ),
            "state_reuse",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    future_pid=base.pid + 1,
                ),
            ),
            "process_isolation",
        ),
        (
            replace(
                base,
                observation=replace(
                    base.observation,
                    future_process_instance_id=("00000000-0000-4000-8000-000000000098"),
                ),
            ),
            "process_isolation",
        ),
        (
            replace(
                base,
                database_receipt=replace(
                    base.database_receipt,
                    sha256="0" * 64,
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                observation=replace(base.observation, scope=alternate_scope),
            ),
            "assignment_mismatch",
        ),
    )

    for forged, reason in variants:
        completed = helpfulness.ChildProcessResult(
            args=[
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                "model-observation-child",
            ],
            pid=forged.pid,
            returncode=0,
            stdout=helpfulness.wire_dumps(forged),
            stderr="",
        )
        monkeypatch.setattr(
            helpfulness,
            "run_isolated_child_raw",
            lambda *_args, _completed=completed, **_kwargs: _completed,
        )
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                request,
                role="model-observation-child",
                timeout_seconds=20,
            )
        assert error.value.reason == reason

    replayed = helpfulness.ChildProcessResult(
        args=[
            sys.executable,
            "-I",
            str(Path(helpfulness.__file__).resolve()),
            "model-observation-child",
        ],
        pid=base.pid + 1,
        returncode=0,
        stdout=helpfulness.wire_dumps(base),
        stderr="",
    )
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: replayed,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as replay_error:
        helpfulness.run_isolated_child(
            request,
            role="model-observation-child",
            timeout_seconds=20,
        )
    assert replay_error.value.reason == "process_isolation"


def test_model_observation_parent_rejects_raw_and_oracle_audit_mutations(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
) -> None:
    case, capture, base_request, _schedule = observation_setup
    raw_request = replace(
        base_request,
        source=helpfulness._fast_source_spec(case, capture.references, "raw_history"),
    )
    raw_response = helpfulness.execute_model_observation_child_request(raw_request)
    raw_event = raw_response.observation.reader_audit[0]
    wrong_raw_hash = (
        "0" * 64
        if not raw_event.returned_content_hashes
        or raw_event.returned_content_hashes[0] != "0" * 64
        else "1" * 64
    )
    forged_raw = replace(
        raw_response,
        observation=replace(
            raw_response.observation,
            reader_audit=(
                replace(
                    raw_event,
                    returned_content_hashes=(
                        wrong_raw_hash,
                        *raw_event.returned_content_hashes[1:],
                    ),
                ),
            ),
        ),
    )
    forged_raw_ids = replace(
        raw_response,
        observation=replace(
            raw_response.observation,
            reader_audit=(
                replace(
                    raw_event,
                    returned_record_ids=(
                        "evd_000000000000000000000000",
                        *raw_event.returned_record_ids[1:],
                    ),
                ),
            ),
        ),
    )

    oracle_request = replace(
        base_request,
        source=helpfulness._fast_source_spec(case, capture.references, "oracle"),
    )
    oracle_response = helpfulness.execute_model_observation_child_request(
        oracle_request
    )
    oracle_event = oracle_response.observation.reader_audit[0]
    forged_oracle_ids = replace(
        oracle_response,
        observation=replace(
            oracle_response.observation,
            reader_audit=(replace(oracle_event, returned_record_ids=("forged",)),),
        ),
    )
    forged_oracle_hash = replace(
        oracle_response,
        observation=replace(
            oracle_response.observation,
            reader_audit=(
                replace(
                    oracle_event,
                    returned_content_hashes=(
                        "0" * 64,
                        *oracle_event.returned_content_hashes[1:],
                    ),
                ),
            ),
        ),
    )

    for response, request in (
        (forged_raw, raw_request),
        (forged_raw_ids, raw_request),
        (forged_oracle_ids, oracle_request),
        (forged_oracle_hash, oracle_request),
    ):
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness._validate_model_observation_child_assignment(
                response,
                request,
            )
        assert error.value.reason == "assignment_mismatch"


def test_model_observation_parent_rejects_self_consistent_forged_oracle(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, capture, base_request, _schedule = observation_setup
    oracle_source = helpfulness._fast_source_spec(case, capture.references, "oracle")
    request = replace(base_request, source=oracle_source)
    observed = helpfulness.execute_model_observation_child_request(request)
    original_entries = oracle_source.oracle_entries
    original_value = original_entries[0].value
    forged_value = (
        "A" if not original_value.startswith("A") else "B"
    ) + original_value[1:]
    forged_entries = (
        replace(original_entries[0], value=forged_value),
        *original_entries[1:],
    )
    rendered = helpfulness.render_context(forged_entries)
    forged_audit = (
        replace(
            observed.observation.reader_audit[0],
            returned_content_hashes=tuple(
                helpfulness._semantic_entry_hash(entry) for entry in forged_entries
            ),
        ),
    )
    child_pid = 12_346
    child_process_id = "00000000-0000-4000-8000-000000000097"
    forged = replace(
        observed,
        pid=child_pid,
        process_instance_id=child_process_id,
        isolated_mode=True,
        areal_module_path=str(
            Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
        ),
        visible_forbidden_environment=(),
        environment_clean=True,
        observation=replace(
            observed.observation,
            future_pid=child_pid,
            future_process_instance_id=child_process_id,
            entries=rendered.entry_receipts,
            reader_audit=forged_audit,
            rendered_context_sha256=hashlib.sha256(rendered.bytes).hexdigest(),
            rendered_context_utf8_bytes=len(rendered.bytes),
        ),
    )
    completed = helpfulness.ChildProcessResult(
        args=[
            sys.executable,
            "-I",
            str(Path(helpfulness.__file__).resolve()),
            "model-observation-child",
        ],
        pid=child_pid,
        returncode=0,
        stdout=helpfulness.wire_dumps(forged),
        stderr="",
    )
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            request,
            role="model-observation-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "assignment_mismatch"


def test_model_observation_rejects_replaced_capture_before_state_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = _build_setup(tmp_path / "pre-read")
    database_path = Path(request.database_path)
    replacement = database_path.with_name("replacement.sqlite3")
    replacement.write_bytes(database_path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, database_path)
    calls = Counter()

    def forbidden(*_args, **_kwargs):
        calls["state_or_spawn"] += 1
        raise AssertionError("receipt mismatch reached state creation or Popen")

    monkeypatch.setattr(helpfulness, "_new_model_observation_state", forbidden)
    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            request,
            role="model-observation-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "state_reuse"
    assert calls == Counter()


def test_model_observation_rejects_wrong_receipt_values_before_spawn(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup
    calls = Counter()

    def forbidden_spawn(*_args, **_kwargs):
        calls["spawn"] += 1
        raise AssertionError("receipt mismatch reached Popen")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_spawn)
    receipt = request.database_receipt
    forged_receipts = (
        replace(receipt, inode=receipt.inode + 1),
        replace(receipt, size_bytes=receipt.size_bytes + 1),
        replace(
            receipt,
            sha256=("0" * 64 if receipt.sha256 != "0" * 64 else "1" * 64),
        ),
    )

    for forged_receipt in forged_receipts:
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                replace(request, database_receipt=forged_receipt),
                role="model-observation-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "state_reuse"

    assert calls == Counter()


def test_model_observation_rejects_database_replacement_after_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = _build_setup(tmp_path / "post-render")
    database_path = Path(request.database_path)
    original_execute = helpfulness._execute_model_source_observation

    def replace_after_render(*args, **kwargs):
        observation = original_execute(*args, **kwargs)
        replacement = database_path.with_name("replacement.sqlite3")
        replacement.write_bytes(database_path.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, database_path)
        return observation

    monkeypatch.setattr(
        helpfulness,
        "_execute_model_source_observation",
        replace_after_render,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_observation_child_request(request)

    assert error.value.reason == "state_reuse"


def test_model_observation_wrong_role_and_timeout_precede_database_access(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup
    calls = Counter()

    def forbidden_receipt(*_args, **_kwargs):
        calls["database"] += 1
        raise AssertionError("invalid launcher arguments reached the database")

    monkeypatch.setattr(
        helpfulness,
        "_model_capture_database_receipt",
        forbidden_receipt,
    )
    with pytest.raises(helpfulness.WireProtocolError) as role_error:
        helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=20,
        )
    with pytest.raises(helpfulness.WireProtocolError) as timeout_error:
        helpfulness.run_isolated_child(
            request,
            role="model-observation-child",
            timeout_seconds=0,
        )

    assert role_error.value.reason == "closed_schema"
    assert timeout_error.value.reason == "closed_schema"
    assert calls == Counter()


def test_model_observation_wrong_renderer_and_future_identity_precede_database(
    observation_setup: tuple[
        helpfulness.CodebookCase,
        helpfulness.ModelCaptureChildResponse,
        helpfulness.ModelObservationChildRequest,
        helpfulness.ParentScheduleItem,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _capture, request, _schedule = observation_setup
    calls = Counter()

    def forbidden_receipt(*_args, **_kwargs):
        calls["database"] += 1
        raise AssertionError("invalid assignment reached the database")

    monkeypatch.setattr(
        helpfulness,
        "_model_capture_database_receipt",
        forbidden_receipt,
    )
    alternate_session, alternate_run = helpfulness._opaque_future_identity(
        request.execution_index + 1
    )
    invalid_requests = (
        replace(request, renderer_version="memory-codebook/v2"),
        replace(request, future_session_id=alternate_session),
        replace(request, future_run_id=alternate_run),
    )

    for invalid_request in invalid_requests:
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                invalid_request,
                role="model-observation-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "assignment_mismatch"

    assert calls == Counter()
