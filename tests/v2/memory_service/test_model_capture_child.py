# SPDX-License-Identifier: Apache-2.0

"""Tests for isolated capture of the exact preregistered model candidate."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness

_ATTEMPT_ONE_SHA256 = "1d840c53233cc9fbfe2454b64798357e06f0fd9e14e74160f4b0f15fdeaebe26"


def _attempt_one_request(database_path: Path) -> helpfulness.ModelCaptureChildRequest:
    database_path.parent.chmod(0o700)
    case = helpfulness._generate_model_candidate(0, model_attempt=1)
    assert case is not None
    assert helpfulness.case_manifest_sha256(case) == _ATTEMPT_ONE_SHA256
    return helpfulness.ModelCaptureChildRequest(
        case_index=0,
        model_attempt=1,
        case_manifest_sha256=_ATTEMPT_ONE_SHA256,
        database_path=str(database_path),
    )


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


def _wire_response(
    request: helpfulness.ModelCaptureChildRequest,
) -> helpfulness.ModelCaptureChildResponse:
    case = helpfulness._generate_model_candidate(
        request.case_index,
        model_attempt=request.model_attempt,
    )
    assert case is not None
    return helpfulness.ModelCaptureChildResponse(
        case_index=request.case_index,
        model_attempt=request.model_attempt,
        case_manifest_sha256=request.case_manifest_sha256,
        references=helpfulness.derive_case_database_references(case),
        database_receipt=helpfulness.ModelCaptureDatabaseReceipt(
            device=1,
            inode=2,
            size_bytes=3,
            sha256="0" * 64,
        ),
        pid=12_345,
        process_instance_id="00000000-0000-4000-8000-000000000001",
        isolated_mode=True,
        areal_module_path=str(
            Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
        ),
        visible_forbidden_environment=(),
        environment_clean=True,
    )


def _assert_wire_reason(
    error: pytest.ExceptionInfo[helpfulness.WireProtocolError],
) -> None:
    assert type(error.value) is helpfulness.WireProtocolError
    assert error.value.reason == "closed_schema"


def test_model_capture_wire_round_trip_is_canonical_and_exact_typed(
    tmp_path: Path,
) -> None:
    request = _attempt_one_request(tmp_path / "记忆.sqlite3")
    response = _wire_response(request)

    for value in (request, response):
        encoded = helpfulness.wire_dumps(value)
        decoded = helpfulness.wire_loads(encoded)

        assert type(decoded) is type(value)
        assert decoded == value
        assert type(decoded.case_index) is int
        assert type(decoded.model_attempt) is int
        assert encoded.endswith("\n")
        assert encoded.count("\n") == 1
        assert helpfulness.wire_dumps(decoded) == encoded


def test_model_capture_wire_rejects_unknown_missing_and_type_confused_fields(
    tmp_path: Path,
) -> None:
    request = _attempt_one_request(tmp_path / "capture.sqlite3")
    request_wire = json.loads(helpfulness.wire_dumps(request))
    response_wire = json.loads(helpfulness.wire_dumps(_wire_response(request)))
    mutants: list[dict[str, object]] = []

    unknown = json.loads(json.dumps(request_wire))
    unknown["payload"]["expected_response"] = "SECRET"
    mutants.append(unknown)
    missing = json.loads(json.dumps(request_wire))
    del missing["payload"]["case_manifest_sha256"]
    mutants.append(missing)
    for field, value in (
        ("case_index", True),
        ("case_index", -1),
        ("case_index", helpfulness.MODEL_CASE_COUNT),
        ("model_attempt", True),
        ("model_attempt", 1.0),
        ("model_attempt", -1),
        ("model_attempt", helpfulness.MODEL_ATTEMPT_LIMIT),
        ("case_manifest_sha256", _ATTEMPT_ONE_SHA256.upper()),
        ("case_manifest_sha256", _ATTEMPT_ONE_SHA256[:-1]),
        ("database_path", "relative.sqlite3"),
        ("database_path", ":memory:"),
        ("database_path", "   "),
    ):
        mutant = json.loads(json.dumps(request_wire))
        mutant["payload"][field] = value
        mutants.append(mutant)

    response_unknown = json.loads(json.dumps(response_wire))
    response_unknown["payload"]["database_path"] = request.database_path
    mutants.append(response_unknown)
    for field, value in (("model_attempt", True), ("pid", True)):
        mutant = json.loads(json.dumps(response_wire))
        mutant["payload"][field] = value
        mutants.append(mutant)
    for field, value in (
        ("device", True),
        ("inode", 0),
        ("size_bytes", 0),
        ("sha256", "A" * 64),
    ):
        mutant = json.loads(json.dumps(response_wire))
        mutant["payload"]["database_receipt"][field] = value
        mutants.append(mutant)

    for mutant in mutants:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_loads(_canonical_wire(mutant))
        _assert_wire_reason(error)

    with pytest.raises(helpfulness.WireProtocolError) as encode_error:
        helpfulness.wire_dumps(replace(request, model_attempt=True))
    _assert_wire_reason(encode_error)


def test_model_capture_hash_mismatch_fails_before_any_database_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    request = replace(
        _attempt_one_request(database_path),
        case_manifest_sha256="0" * 64,
    )
    calls = Counter()

    def forbidden_build(*_args, **_kwargs):
        calls["build"] += 1
        raise AssertionError("hash mismatch reached the database builder")

    monkeypatch.setattr(helpfulness, "build_case_database", forbidden_build)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_capture_child_request(request)

    assert error.value.reason == "assignment_mismatch"
    assert calls == Counter()
    assert not database_path.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_model_capture_requires_a_canonical_private_parent_directory(
    tmp_path: Path,
) -> None:
    permissive_path = tmp_path / "permissive.sqlite3"
    permissive_request = _attempt_one_request(permissive_path)
    tmp_path.chmod(0o755)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as mode_error:
        helpfulness.execute_model_capture_child_request(permissive_request)

    assert mode_error.value.reason == "state_reuse"
    assert not permissive_path.exists()

    tmp_path.chmod(0o700)
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_path = linked_parent / "linked.sqlite3"
    linked_request = _attempt_one_request(linked_path)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as link_error:
        helpfulness.execute_model_capture_child_request(linked_request)

    assert link_error.value.reason == "state_reuse"
    assert not (real_parent / "linked.sqlite3").exists()


@pytest.mark.parametrize("leaf_kind", ("file", "symlink", "broken_symlink"))
def test_model_capture_never_reuses_or_follows_an_existing_leaf(
    tmp_path: Path,
    leaf_kind: str,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    if leaf_kind == "file":
        database_path.write_bytes(b"owner-data")
        protected_path = database_path
    elif leaf_kind == "symlink":
        target_path.write_bytes(b"owner-data")
        database_path.symlink_to(target_path)
        protected_path = target_path
    else:
        database_path.symlink_to(target_path)
        protected_path = target_path
    request = _attempt_one_request(database_path)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_capture_child_request(request)

    assert error.value.reason == "state_reuse"
    if leaf_kind == "broken_symlink":
        assert database_path.is_symlink()
        assert not protected_path.exists()
    else:
        assert protected_path.read_bytes() == b"owner-data"


def test_model_capture_rejects_post_build_permission_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    request = _attempt_one_request(database_path)
    original = helpfulness.build_case_database

    def drift_permissions(case, path):
        references = original(case, path)
        os.chmod(path, 0o644)
        return references

    monkeypatch.setattr(helpfulness, "build_case_database", drift_permissions)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_capture_child_request(request)

    assert error.value.reason == "state_reuse"
    assert not database_path.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_model_capture_racing_destination_symlink_cannot_modify_its_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    victim_path = tmp_path / "victim.sqlite3"
    victim_path.write_bytes(b"owner-data")
    request = _attempt_one_request(database_path)
    original = helpfulness.build_case_database

    def race_destination(case, staging_path):
        database_path.symlink_to(victim_path)
        return original(case, staging_path)

    monkeypatch.setattr(helpfulness, "build_case_database", race_destination)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_model_capture_child_request(request)

    assert error.value.reason == "state_reuse"
    assert database_path.is_symlink()
    assert victim_path.read_bytes() == b"owner-data"
    assert not tuple(tmp_path.glob(".areal-model-capture-*"))


def test_model_capture_fsyncs_publication_directory_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    request = _attempt_one_request(database_path)
    original_fsync = helpfulness.os.fsync
    fsynced_modes: list[int] = []
    fsynced_directories: list[tuple[int, int]] = []
    parent_stat = tmp_path.stat()
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)

    def record_fsync(file_descriptor: int) -> None:
        file_stat = os.fstat(file_descriptor)
        fsynced_modes.append(file_stat.st_mode)
        if stat.S_ISDIR(file_stat.st_mode):
            fsynced_directories.append((file_stat.st_dev, file_stat.st_ino))
        original_fsync(file_descriptor)

    monkeypatch.setattr(helpfulness.os, "fsync", record_fsync)
    response = helpfulness.execute_model_capture_child_request(request)

    assert type(response) is helpfulness.ModelCaptureChildResponse
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
    assert fsynced_directories == [parent_identity]
    assert database_path.is_file()
    assert not tuple(tmp_path.glob(".areal-model-capture-*"))


def test_model_capture_receipt_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    moved_path = tmp_path / "moved.sqlite3"
    database_path.write_bytes(b"original-database-bytes")
    database_path.chmod(0o600)
    file_stat = database_path.stat()
    identity = (file_stat.st_dev, file_stat.st_ino)
    original_read = helpfulness.os.read
    raced = False

    def replace_path_after_read(file_descriptor: int, count: int) -> bytes:
        nonlocal raced
        part = original_read(file_descriptor, count)
        if not raced:
            raced = True
            os.replace(database_path, moved_path)
            database_path.write_bytes(b"replacement-database-bytes")
            database_path.chmod(0o600)
        return part

    monkeypatch.setattr(helpfulness.os, "read", replace_path_after_read)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._model_capture_database_receipt(
            str(database_path),
            expected_identity=identity,
            durable=False,
        )

    assert error.value.reason == "state_reuse"
    assert moved_path.read_bytes() == b"original-database-bytes"
    assert database_path.read_bytes() == b"replacement-database-bytes"


def test_model_capture_receipt_rejects_oversized_file_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "oversized.sqlite3"
    with database_path.open("wb") as database_file:
        database_file.truncate(helpfulness._MAX_MODEL_CAPTURE_DATABASE_BYTES + 1)
    database_path.chmod(0o600)
    file_stat = database_path.stat()

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("oversized receipt reached os.read")

    monkeypatch.setattr(helpfulness.os, "read", forbidden_read)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._model_capture_database_receipt(
            str(database_path),
            expected_identity=(file_stat.st_dev, file_stat.st_ino),
            durable=False,
        )

    assert error.value.reason == "state_reuse"


def test_model_capture_receipt_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo_path = tmp_path / "capture.fifo"
    os.mkfifo(fifo_path, 0o600)
    fifo_stat = fifo_path.stat()

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._model_capture_database_receipt(
            str(fifo_path),
            expected_identity=(fifo_stat.st_dev, fifo_stat.st_ino),
            durable=False,
        )

    assert error.value.reason == "state_reuse"


def test_model_capture_staging_unlink_error_is_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_directory = tmp_path / "staging"
    staging_directory.mkdir(mode=0o700)
    staging_path = staging_directory / "database.sqlite3"
    staging_path.write_bytes(b"staging")
    staging_path.chmod(0o600)
    file_stat = staging_path.stat()

    def fail_unlink(*_args, **_kwargs):
        raise PermissionError("test unlink denial")

    monkeypatch.setattr(helpfulness.os, "unlink", fail_unlink)
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._cleanup_model_capture_staging(
            str(staging_directory),
            str(staging_path),
            (file_stat.st_dev, file_stat.st_ino),
        )

    assert error.value.reason == "state_reuse"


def test_model_capture_parent_rejects_forged_assignment_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _attempt_one_request(tmp_path / "capture.sqlite3")
    response = _wire_response(request)
    case_zero = helpfulness._generate_model_candidate(0, model_attempt=0)
    assert case_zero is not None
    mutants = (
        replace(response, case_index=1),
        replace(response, model_attempt=2),
        replace(response, case_manifest_sha256="0" * 64),
        replace(
            response,
            references=helpfulness.derive_case_database_references(case_zero),
        ),
    )
    monkeypatch.setattr(
        helpfulness,
        "_model_capture_directory_identity",
        lambda _path: (1, 1),
    )
    monkeypatch.setattr(
        helpfulness,
        "_model_capture_database_receipt",
        lambda *_args, **_kwargs: response.database_receipt,
    )

    for mutant in mutants:
        completed = helpfulness.ChildProcessResult(
            args=[sys.executable, "-I", str(Path(helpfulness.__file__).resolve())],
            pid=response.pid,
            returncode=0,
            stdout=helpfulness.wire_dumps(mutant),
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
                role="capture-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "assignment_mismatch"


def test_model_capture_parent_rejects_correct_fields_without_a_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _attempt_one_request(tmp_path / "missing.sqlite3")
    response = _wire_response(request)
    completed = helpfulness.ChildProcessResult(
        args=[sys.executable, "-I", str(Path(helpfulness.__file__).resolve())],
        pid=response.pid,
        returncode=0,
        stdout=helpfulness.wire_dumps(response),
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
            role="capture-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "state_reuse"
    assert not Path(request.database_path).exists()


def test_model_capture_wrong_role_is_rejected_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _attempt_one_request(tmp_path / "capture.sqlite3")
    calls = Counter()

    def forbidden_popen(*_args, **_kwargs):
        calls["popen"] += 1
        raise AssertionError("wrong role reached Popen")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_popen)
    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child_raw(
            request,
            role="future-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "closed_schema"
    assert calls == Counter()


def test_model_capture_attempt_one_runs_in_a_fresh_isolated_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "capture.sqlite3"
    request = _attempt_one_request(database_path)
    case = helpfulness._generate_model_candidate(0, model_attempt=1)
    assert case is not None
    parent_store_calls = Counter()
    launches: list[tuple[list[str], int]] = []
    original_popen = helpfulness.subprocess.Popen

    def forbidden_parent_store(*_args, **_kwargs):
        parent_store_calls["store"] += 1
        raise AssertionError("parent opened the capture database")

    def record_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launches.append((list(args[0]), process.pid))
        return process

    monkeypatch.setattr(helpfulness, "SQLiteMemoryStore", forbidden_parent_store)
    monkeypatch.setattr(helpfulness.subprocess, "Popen", record_popen)
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-pythonpath")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-exec")
    monkeypatch.setenv("AREAL_TASK6_UNKNOWN_CANARY", "must-not-cross-exec")

    response = helpfulness.run_isolated_child(
        request,
        role="capture-child",
        timeout_seconds=20,
    )

    assert type(response) is helpfulness.ModelCaptureChildResponse
    assert response.case_index == request.case_index
    assert response.model_attempt == request.model_attempt == 1
    assert response.case_manifest_sha256 == request.case_manifest_sha256
    assert response.references == helpfulness.derive_case_database_references(case)
    assert parent_store_calls == Counter()
    assert launches == [
        (
            [
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                "capture-child",
            ],
            response.pid,
        )
    ]
    assert response.pid != os.getpid()
    assert response.process_instance_id != helpfulness.PROCESS_INSTANCE_ID
    assert str(uuid.UUID(response.process_instance_id, version=4)) == (
        response.process_instance_id
    )
    assert response.isolated_mode is True
    assert response.areal_module_path == str(
        Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
    )
    assert response.visible_forbidden_environment == ()
    assert response.environment_clean is True
    file_stat = database_path.stat()
    assert stat.S_ISREG(file_stat.st_mode)
    assert stat.S_IMODE(file_stat.st_mode) == 0o600
    assert file_stat.st_nlink == 1
    assert response.database_receipt == helpfulness.ModelCaptureDatabaseReceipt(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size_bytes=file_stat.st_size,
        sha256=hashlib.sha256(database_path.read_bytes()).hexdigest(),
    )
