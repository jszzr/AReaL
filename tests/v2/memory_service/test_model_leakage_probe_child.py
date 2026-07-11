# SPDX-License-Identifier: Apache-2.0

"""Tests for isolated, consumer- and model-output-free leakage probes."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness

from areal.v2.memory_service.errors import (
    MemoryPersistenceBusyError,
    MemoryPersistenceCorruptionError,
    MemoryPersistenceError,
    MemoryPersistenceSchemaError,
    MemoryServiceError,
    ReleaseNotFoundError,
)

_ATTEMPT_ONE_SHA256 = "1d840c53233cc9fbfe2454b64798357e06f0fd9e14e74160f4b0f15fdeaebe26"
_EXECUTION_TOKEN = (1 << 127) | 0x223456789ABCDEF
_MISSING_RELEASE_ID = "rel_000000000000000000000000"
_RETURN_NONE = object()


class _SubclassedReleaseNotFoundError(ReleaseNotFoundError):
    pass


@dataclass(frozen=True, slots=True)
class _ProbeSetup:
    case: helpfulness.CodebookCase
    capture: helpfulness.ModelCaptureChildResponse
    request: helpfulness.ModelLeakageProbeChildRequest


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


def _build_probe_setup(
    root: Path,
    *,
    execution_token: int = _EXECUTION_TOKEN,
) -> _ProbeSetup:
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
    request = helpfulness.ModelLeakageProbeChildRequest(
        execution_index=execution_token,
        database_path=str(database_path),
        database_receipt=capture.database_receipt,
        scope=capture.references.capture.local_scope,
        release_id=capture.references.releases.foreign_sentinel_release_id,
    )
    return _ProbeSetup(case=case, capture=capture, request=request)


@pytest.fixture(scope="module")
def probe_setup(tmp_path_factory: pytest.TempPathFactory) -> _ProbeSetup:
    return _build_probe_setup(tmp_path_factory.mktemp("model-leakage-probe"))


def _isolated_response(
    response: helpfulness.ModelLeakageProbeChildResponse,
    *,
    pid: int = 12_345,
    process_instance_id: str = "00000000-0000-4000-8000-000000000099",
) -> helpfulness.ModelLeakageProbeChildResponse:
    return replace(
        response,
        pid=pid,
        process_instance_id=process_instance_id,
        isolated_mode=True,
        areal_module_path=str(
            Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
        ),
        visible_forbidden_environment=(),
        environment_clean=True,
        state_receipt=replace(
            response.state_receipt,
            process_instance_id=process_instance_id,
        ),
        probe=replace(
            response.probe,
            future_pid=pid,
            future_process_instance_id=process_instance_id,
        ),
    )


def _completed_probe(
    response: helpfulness.ModelLeakageProbeChildResponse,
    *,
    pid: int | None = None,
) -> helpfulness.ChildProcessResult:
    return helpfulness.ChildProcessResult(
        args=[
            sys.executable,
            "-I",
            str(Path(helpfulness.__file__).resolve()),
            "model-leakage-probe-child",
        ],
        pid=response.pid if pid is None else pid,
        returncode=0,
        stdout=helpfulness.wire_dumps(response),
        stderr="",
    )


def test_model_leakage_probe_wire_is_minimal_closed_and_exact_typed(
    probe_setup: _ProbeSetup,
) -> None:
    request = probe_setup.request
    response = helpfulness.execute_model_leakage_probe_child_request(request)
    forbidden_keys = {
        "arm",
        "capture_pid",
        "capture_process_instance_id",
        "capture_session_ids",
        "case_id",
        "case_index",
        "case_manifest_sha256",
        "companion_scope",
        "consumer_input_receipt",
        "consumer_version",
        "entries",
        "expected_response",
        "foreign_evidence_id",
        "foreign_scope",
        "history",
        "history_length",
        "model_call_receipt",
        "query",
        "query_sha256",
        "reader_audit",
        "rendered_context_sha256",
        "renderer_version",
        "response",
        "source",
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

    request_payload = json.loads(helpfulness.wire_dumps(request))["payload"]
    response_payload = json.loads(helpfulness.wire_dumps(response))["payload"]
    assert set(request_payload) == {
        "database_path",
        "database_receipt",
        "execution_index",
        "release_id",
        "scope",
    }
    assert set(response_payload) == {
        "areal_module_path",
        "database_receipt",
        "environment_clean",
        "isolated_mode",
        "pid",
        "probe",
        "process_instance_id",
        "state_receipt",
        "visible_forbidden_environment",
    }
    assert set(response_payload["probe"]) == {
        "execution_index",
        "future_pid",
        "future_process_instance_id",
        "future_run_id",
        "future_session_id",
        "outcome",
        "requested_scope",
        "release_id",
    }
    assert set(response_payload["state_receipt"]) == {
        "execution_index",
        "generation_index",
        "logical_run_id",
        "logical_session_id",
        "logical_session_instance_id",
        "process_instance_id",
        "store_instance_id",
    }

    for value in (request, response, response.probe, response.state_receipt):
        assert forbidden_keys.isdisjoint(field.name for field in fields(value))

    request_wire = json.loads(helpfulness.wire_dumps(request))
    response_wire = json.loads(helpfulness.wire_dumps(response))
    mutants: list[dict[str, object]] = []
    unknown = json.loads(json.dumps(request_wire))
    unknown["payload"]["companion_scope"] = request_wire["payload"]["scope"]
    mutants.append(unknown)
    missing = json.loads(json.dumps(request_wire))
    del missing["payload"]["database_receipt"]
    mutants.append(missing)
    for execution_index in (True, 1.0, 0):
        mutant = json.loads(json.dumps(request_wire))
        mutant["payload"]["execution_index"] = execution_index
        mutants.append(mutant)
    for release_id in ("", "not-a-release", "rel_ABCDEF000000000000000000"):
        mutant = json.loads(json.dumps(request_wire))
        mutant["payload"]["release_id"] = release_id
        mutants.append(mutant)
    bad_digest = json.loads(json.dumps(request_wire))
    bad_digest["payload"]["database_receipt"]["sha256"] = "A" * 64
    mutants.append(bad_digest)
    bad_outcome = json.loads(json.dumps(response_wire))
    bad_outcome["payload"]["probe"]["outcome"] = "foreign_scope"
    mutants.append(bad_outcome)
    smuggled_history = json.loads(json.dumps(response_wire))
    smuggled_history["payload"]["probe"]["history_length"] = 0
    mutants.append(smuggled_history)
    bad_generation = json.loads(json.dumps(response_wire))
    bad_generation["payload"]["state_receipt"]["generation_index"] = 1
    mutants.append(bad_generation)
    injected_answer = json.loads(json.dumps(response_wire))
    injected_answer["payload"]["response"] = "SECRET"
    mutants.append(injected_answer)

    for mutant in mutants:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_loads(_canonical_wire(mutant))
        assert error.value.reason == "closed_schema"

    in_memory_mutants = (
        replace(response, visible_forbidden_environment=[]),  # type: ignore[arg-type]
        replace(
            response,
            probe=replace(response.probe, requested_scope={"forged": True}),  # type: ignore[arg-type]
        ),
    )
    for mutant in in_memory_mutants:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_dumps(mutant)
        assert error.value.reason == "closed_schema"


def test_model_leakage_probe_is_a_real_foreign_negative_control_and_only_reads_local(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = probe_setup.capture
    request = probe_setup.request
    store = helpfulness.SQLiteMemoryStore(request.database_path)
    foreign_release = store.get_release(
        capture.references.capture.foreign_scope,
        request.release_id,
    )
    assert foreign_release.release_id == request.release_id
    assert foreign_release.manifest.scope == capture.references.capture.foreign_scope
    with pytest.raises(ReleaseNotFoundError):
        store.get_release(request.scope, request.release_id)

    original_get_release = helpfulness.SQLiteMemoryStore.get_release
    calls: list[tuple[object, str]] = []

    def spy_get_release(self, scope, release_id):
        calls.append((scope, release_id))
        return original_get_release(self, scope, release_id)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("leakage probe reached a non-point-read path")

    monkeypatch.setattr(helpfulness.SQLiteMemoryStore, "get_release", spy_get_release)
    for name in ("get_revision", "get_candidate", "list"):
        monkeypatch.setattr(helpfulness.SQLiteMemoryStore, name, forbidden)
    for name in (
        "ReadAuditSink",
        "ReleaseReadCapability",
        "_ItemResolver",
        "_ItemRenderer",
        "_ItemConsumer",
        "resolve_treatment",
        "render_context",
        "consume_scripted",
    ):
        monkeypatch.setattr(helpfulness, name, forbidden)

    response = helpfulness.execute_model_leakage_probe_child_request(request)
    expected_session_id, expected_run_id = helpfulness._opaque_future_identity(
        request.execution_index
    )

    assert calls == [(request.scope, request.release_id)]
    assert response.database_receipt == request.database_receipt
    assert response.probe.execution_index == request.execution_index
    assert response.probe.requested_scope == request.scope
    assert response.probe.release_id == request.release_id
    assert response.probe.outcome == "release_not_found"
    assert response.probe.future_session_id == expected_session_id
    assert response.probe.future_run_id == expected_run_id
    assert response.state_receipt.logical_session_id == expected_session_id
    assert response.state_receipt.logical_run_id == expected_run_id
    assert response.state_receipt.generation_index == 0


def test_scope_ignoring_store_becomes_a_content_free_found_counterexample(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get_release = helpfulness.SQLiteMemoryStore.get_release
    foreign_scope = probe_setup.capture.references.capture.foreign_scope

    def ignore_requested_scope(store, _requested_scope, release_id):
        return original_get_release(store, foreign_scope, release_id)

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        ignore_requested_scope,
    )

    response = helpfulness.execute_model_leakage_probe_child_request(
        probe_setup.request
    )
    encoded = helpfulness.wire_dumps(response)

    assert response.probe.requested_scope == probe_setup.request.scope
    assert response.probe.release_id == probe_setup.request.release_id
    assert response.probe.outcome == "release_found"
    assert probe_setup.case.target_key not in encoded
    assert probe_setup.case.current_value not in encoded


def test_raw_absence_does_not_claim_that_the_release_exists_in_a_foreign_scope(
    probe_setup: _ProbeSetup,
) -> None:
    """The later manifest join, not this raw child, proves foreign existence."""

    release_assignments = probe_setup.capture.references.releases
    known_release_ids = {
        getattr(release_assignments, field.name)
        for field in fields(release_assignments)
    }
    assert _MISSING_RELEASE_ID not in known_release_ids
    request = replace(probe_setup.request, release_id=_MISSING_RELEASE_ID)

    response = helpfulness.execute_model_leakage_probe_child_request(request)

    assert response.probe.release_id == _MISSING_RELEASE_ID
    assert response.probe.requested_scope == request.scope
    assert response.probe.outcome == "release_not_found"


@pytest.mark.parametrize(
    ("outcome", "expected_reason"),
    (
        (MemoryPersistenceBusyError("busy"), "state_reuse"),
        (MemoryPersistenceSchemaError("schema"), "state_reuse"),
        (MemoryPersistenceCorruptionError("corrupt"), "state_reuse"),
        (MemoryPersistenceError("storage"), "state_reuse"),
        (OSError("io"), "state_reuse"),
        (MemoryServiceError("service"), "assignment_mismatch"),
        (_RETURN_NONE, "assignment_mismatch"),
        (RuntimeError("unexpected"), None),
    ),
    ids=(
        "busy",
        "schema",
        "corruption",
        "persistence",
        "os-error",
        "generic-service",
        "none-return",
        "unexpected-runtime",
    ),
)
def test_only_release_not_found_family_can_become_an_absence(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    expected_reason: str | None,
) -> None:
    def substitute_get_release(*_args, **_kwargs):
        if outcome is _RETURN_NONE:
            return None
        assert isinstance(outcome, BaseException)
        raise outcome

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        substitute_get_release,
    )

    try:
        response = helpfulness.execute_model_leakage_probe_child_request(
            probe_setup.request
        )
    except helpfulness.ChildExecutionValidationError as error:
        if expected_reason is not None:
            assert error.reason == expected_reason
        else:
            assert error.reason != "release_not_found"
    except (AttributeError, RuntimeError, TypeError):
        assert expected_reason is None
    else:
        pytest.fail(f"non-absence outcome was misclassified as {response!r}")


def test_release_not_found_subclass_remains_a_raw_absence(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def specialized_absence(*_args, **_kwargs):
        raise _SubclassedReleaseNotFoundError("specialized absence")

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        specialized_absence,
    )

    response = helpfulness.execute_model_leakage_probe_child_request(
        probe_setup.request
    )

    assert response.probe.outcome == "release_not_found"


def test_visible_local_release_reports_presence_without_serializing_content(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(
        probe_setup.request,
        release_id=probe_setup.capture.references.releases.current_release_id,
    )
    completed = helpfulness.run_isolated_child_raw(
        request,
        role="model-leakage-probe-child",
        timeout_seconds=20,
    )
    decoded = helpfulness.wire_loads(completed.stdout)

    assert completed.returncode == 0
    assert type(decoded) is helpfulness.ModelLeakageProbeChildResponse
    assert decoded.probe.outcome == "release_found"
    assert completed.stdout == helpfulness.wire_dumps(decoded)
    combined = completed.stdout + completed.stderr
    assert probe_setup.case.target_key not in combined
    assert probe_setup.case.current_value not in combined
    assert probe_setup.case.old_value not in combined

    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: completed,
    )
    public = helpfulness.run_isolated_child(
        request,
        role="model-leakage-probe-child",
        timeout_seconds=20,
    )
    assert public == decoded


def test_model_leakage_probe_uses_fresh_real_isolated_processes(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_request = probe_setup.request
    second_request = replace(first_request, execution_index=_EXECUTION_TOKEN + 1)
    launches: list[tuple[list[str], int]] = []
    parent_store_calls = Counter()
    original_popen = helpfulness.subprocess.Popen

    def record_popen(*args, **kwargs):
        child_environment = kwargs["env"]
        assert kwargs["close_fds"] is True
        assert "PYTHONPATH" not in child_environment
        assert "OPENAI_API_KEY" not in child_environment
        assert "AREAL_MODEL_PROBE_CANARY" not in child_environment
        process = original_popen(*args, **kwargs)
        launches.append((list(args[0]), process.pid))
        return process

    def forbidden_parent_store(*_args, **_kwargs):
        parent_store_calls["store"] += 1
        raise AssertionError("parent opened the captured case SQLite database")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", record_popen)
    monkeypatch.setattr(helpfulness, "SQLiteMemoryStore", forbidden_parent_store)
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-pythonpath")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-exec")
    monkeypatch.setenv("AREAL_MODEL_PROBE_CANARY", "must-not-cross-exec")

    first = helpfulness.run_isolated_child(
        first_request,
        role="model-leakage-probe-child",
        timeout_seconds=20,
    )
    second = helpfulness.run_isolated_child(
        second_request,
        role="model-leakage-probe-child",
        timeout_seconds=20,
    )

    assert type(first) is helpfulness.ModelLeakageProbeChildResponse
    assert type(second) is helpfulness.ModelLeakageProbeChildResponse
    assert parent_store_calls == Counter()
    expected_args = [
        sys.executable,
        "-I",
        str(Path(helpfulness.__file__).resolve()),
        "model-leakage-probe-child",
    ]
    assert launches == [
        (expected_args, first.pid),
        (expected_args, second.pid),
    ]
    assert first.pid != os.getpid()
    assert second.pid != os.getpid()
    assert first.probe.future_pid == first.pid
    assert second.probe.future_pid == second.pid
    assert first.probe.future_process_instance_id == first.process_instance_id
    assert second.probe.future_process_instance_id == second.process_instance_id
    assert (
        str(uuid.UUID(first.process_instance_id, version=4))
        == first.process_instance_id
    )
    assert str(uuid.UUID(second.process_instance_id, version=4)) == (
        second.process_instance_id
    )
    assert (
        len(
            {
                probe_setup.capture.process_instance_id,
                first.process_instance_id,
                second.process_instance_id,
            }
        )
        == 3
    )

    for request, response in (
        (first_request, first),
        (second_request, second),
    ):
        session_id, run_id = helpfulness._opaque_future_identity(
            request.execution_index
        )
        assert response.probe.future_session_id == session_id
        assert response.probe.future_run_id == run_id
        assert response.state_receipt.logical_session_id == session_id
        assert response.state_receipt.logical_run_id == run_id
        assert (
            response.state_receipt.process_instance_id == response.process_instance_id
        )
        assert response.isolated_mode is True
        assert response.visible_forbidden_environment == ()
        assert response.environment_clean is True

    first_state = {
        getattr(first.state_receipt, field.name)
        for field in fields(first.state_receipt)
        if field.name.endswith("_instance_id") and field.name != "process_instance_id"
    }
    second_state = {
        getattr(second.state_receipt, field.name)
        for field in fields(second.state_receipt)
        if field.name.endswith("_instance_id") and field.name != "process_instance_id"
    }
    assert len(first_state) == len(second_state) == 2
    assert first_state.isdisjoint(second_state)

    spliced = replace(second, state_receipt=first.state_receipt)
    completed = _completed_probe(spliced, pid=second.pid)
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: completed,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as splice_error:
        helpfulness.run_isolated_child(
            second_request,
            role="model-leakage-probe-child",
            timeout_seconds=20,
        )
    assert splice_error.value.reason == "state_reuse"


def test_model_leakage_probe_rejects_aliased_in_process_state(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_new_state = helpfulness._new_model_leakage_probe_state

    def aliased_state(request):
        state = original_new_state(request)
        state.store = state.logical_session
        state.receipt = replace(
            state.receipt,
            store_instance_id=helpfulness._state_identity(
                generation_index=state.generation_index,
                component="store",
                value=state.store,
            ),
        )
        return state

    monkeypatch.setattr(
        helpfulness,
        "_new_model_leakage_probe_state",
        aliased_state,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_leakage_probe_child_request(probe_setup.request)
    assert error.value.reason == "state_reuse"


def test_model_leakage_probe_rejects_store_bound_to_another_database(
    probe_setup: _ProbeSetup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(
        probe_setup.request,
        release_id=probe_setup.capture.references.releases.current_release_id,
    )
    assert (
        helpfulness.execute_model_leakage_probe_child_request(request).probe.outcome
        == "release_found"
    )
    other_root = tmp_path / "other-capture"
    other_root.mkdir(mode=0o700)
    other_case = helpfulness._generate_model_candidate(1, model_attempt=1)
    assert other_case is not None
    other_database_path = other_root / "case.sqlite3"
    helpfulness.execute_model_capture_child_request(
        helpfulness.ModelCaptureChildRequest(
            case_index=1,
            model_attempt=1,
            case_manifest_sha256=helpfulness.case_manifest_sha256(other_case),
            database_path=str(other_database_path),
        )
    )
    original_new_state = helpfulness._new_model_leakage_probe_state

    def wrong_database_state(probe_request):
        state = original_new_state(probe_request)
        state.store = helpfulness.SQLiteMemoryStore(str(other_database_path))
        state.receipt = replace(
            state.receipt,
            store_instance_id=helpfulness._state_identity(
                generation_index=state.generation_index,
                component="store",
                value=state.store,
            ),
        )
        return state

    monkeypatch.setattr(
        helpfulness,
        "_new_model_leakage_probe_state",
        wrong_database_state,
    )

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_leakage_probe_child_request(request)
    assert error.value.reason == "state_reuse"

    monkeypatch.setattr(
        helpfulness,
        "_new_model_leakage_probe_state",
        original_new_state,
    )
    original_get_release = helpfulness.SQLiteMemoryStore.get_release

    def redirect_after_claim(store, scope, release_id):
        store._database_path = str(other_database_path)
        return original_get_release(store, scope, release_id)

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        redirect_after_claim,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as redirect_error:
        helpfulness.execute_model_leakage_probe_child_request(request)
    assert redirect_error.value.reason == "state_reuse"


def test_model_leakage_probe_parent_rejects_inconsistent_assignment_and_replay(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = probe_setup.request
    observed = helpfulness.execute_model_leakage_probe_child_request(request)
    base = _isolated_response(observed)
    alternate_scope = replace(
        request.scope,
        subject_id=f"{request.scope.subject_id}-forged",
    )
    alternate_session, alternate_run = helpfulness._opaque_future_identity(
        request.execution_index + 1
    )
    alternate_release = "rel_111111111111111111111111"
    variants = (
        (
            replace(
                base,
                probe=replace(base.probe, execution_index=request.execution_index + 1),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                probe=replace(base.probe, requested_scope=alternate_scope),
            ),
            "assignment_mismatch",
        ),
        (
            replace(base, probe=replace(base.probe, release_id=alternate_release)),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                probe=replace(base.probe, future_session_id=alternate_session),
            ),
            "assignment_mismatch",
        ),
        (
            replace(base, probe=replace(base.probe, future_run_id=alternate_run)),
            "assignment_mismatch",
        ),
        (
            replace(base, probe=replace(base.probe, future_pid=base.pid + 1)),
            "process_isolation",
        ),
        (
            replace(
                base,
                probe=replace(
                    base.probe,
                    future_process_instance_id=("00000000-0000-4000-8000-000000000098"),
                ),
            ),
            "process_isolation",
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
                    logical_session_instance_id=(base.state_receipt.store_instance_id),
                ),
            ),
            "state_reuse",
        ),
        (
            replace(
                base,
                database_receipt=replace(
                    base.database_receipt,
                    sha256=(
                        "0" * 64
                        if base.database_receipt.sha256 != "0" * 64
                        else "1" * 64
                    ),
                ),
            ),
            "assignment_mismatch",
        ),
        (
            replace(
                base,
                process_instance_id="not-a-uuid",
                probe=replace(
                    base.probe,
                    future_process_instance_id="not-a-uuid",
                ),
            ),
            "process_isolation",
        ),
    )

    for forged, reason in variants:
        completed = _completed_probe(forged)
        monkeypatch.setattr(
            helpfulness,
            "run_isolated_child_raw",
            lambda *_args, _completed=completed, **_kwargs: _completed,
        )
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                request,
                role="model-leakage-probe-child",
                timeout_seconds=20,
            )
        assert error.value.reason == reason

    replayed = _completed_probe(base, pid=base.pid + 1)
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: replayed,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            request,
            role="model-leakage-probe-child",
            timeout_seconds=20,
        )
    assert error.value.reason == "process_isolation"


def test_model_leakage_probe_rejects_wrong_receipt_and_path_before_spawn(
    probe_setup: _ProbeSetup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Counter()

    def forbidden_spawn(*_args, **_kwargs):
        calls["spawn"] += 1
        raise AssertionError("receipt mismatch reached Popen")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_spawn)
    request = probe_setup.request
    receipt = request.database_receipt
    forged_receipts = (
        replace(receipt, device=receipt.device + 1),
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
                role="model-leakage-probe-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "state_reuse"

    replaced_setup = _build_probe_setup(tmp_path / "replaced-before-spawn")
    database_path = Path(replaced_setup.request.database_path)
    replacement = database_path.with_name("replacement.sqlite3")
    replacement.write_bytes(database_path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, database_path)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            replaced_setup.request,
            role="model-leakage-probe-child",
            timeout_seconds=20,
        )
    assert error.value.reason == "state_reuse"
    assert calls == Counter()


def test_model_leakage_probe_rechecks_database_receipt_after_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _build_probe_setup(tmp_path / "replaced-after-lookup")
    database_path = Path(setup.request.database_path)
    original_get_release = helpfulness.SQLiteMemoryStore.get_release

    def replace_after_absence(store, scope, release_id):
        try:
            return original_get_release(store, scope, release_id)
        except ReleaseNotFoundError:
            replacement = database_path.with_name("replacement.sqlite3")
            replacement.write_bytes(database_path.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, database_path)
            raise

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        replace_after_absence,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_leakage_probe_child_request(setup.request)
    assert error.value.reason == "state_reuse"


def test_model_leakage_probe_rejects_role_timeout_and_malformed_request_pre_db(
    probe_setup: _ProbeSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = probe_setup.request
    calls = Counter()

    def forbidden(*_args, **_kwargs):
        calls["database_or_spawn"] += 1
        raise AssertionError("invalid request reached database access or Popen")

    monkeypatch.setattr(
        helpfulness,
        "_model_capture_database_receipt",
        forbidden,
    )
    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden)

    with pytest.raises(helpfulness.WireProtocolError) as role_error:
        helpfulness.run_isolated_child(
            request,
            role="model-observation-child",
            timeout_seconds=20,
        )
    assert role_error.value.reason == "closed_schema"

    with pytest.raises(helpfulness.WireProtocolError) as timeout_error:
        helpfulness.run_isolated_child(
            request,
            role="model-leakage-probe-child",
            timeout_seconds=0,
        )
    assert timeout_error.value.reason == "closed_schema"

    malformed_requests = (
        replace(request, execution_index=1),
        replace(request, release_id=""),
        replace(request, release_id="not-a-release"),
    )
    for malformed in malformed_requests:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.run_isolated_child(
                malformed,
                role="model-leakage-probe-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "closed_schema"

    assert calls == Counter()
