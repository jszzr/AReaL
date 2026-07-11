# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier

import pytest

from examples.memory_service import local_update_run_seal as run_seal


def _payload(index: int = 0) -> dict[str, object]:
    return {
        "case_count": 64,
        "manifest_sha256": f"{index + 1:064x}",
        "result": {"accepted": True, "index": index},
    }


def _counts(path: Path) -> tuple[int, int]:
    with sqlite3.connect(path) as connection:
        return (
            connection.execute(
                "SELECT COUNT(*) FROM local_update_run_seals"
            ).fetchone()[0],
            connection.execute(
                "SELECT COUNT(*) FROM local_update_run_seal_aliases"
            ).fetchone()[0],
        )


def test_seal_is_canonical_immutable_reopenable_and_exactly_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "run-seals.sqlite3"
    first = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="run-1",
        payload=_payload(),
    )
    retry = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="run-1",
        payload={
            "result": {"index": 0, "accepted": True},
            "manifest_sha256": f"{1:064x}",
            "case_count": 64,
        },
    )
    reopened = run_seal.get_local_update_run_seal(
        database_path,
        idempotency_key="run-1",
    )

    assert first == retry == reopened
    assert first.seal_id == f"lurs_{first.content_hash[:24]}"
    assert (
        json.dumps(
            json.loads(first.canonical),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        == first.canonical
    )
    assert b"created_at" not in first.canonical
    assert b"run-1" not in first.canonical
    assert _counts(database_path) == (1, 1)
    with pytest.raises(FrozenInstanceError):
        first.seal_id = "changed"  # type: ignore[misc]


def test_distinct_aliases_reuse_core_and_created_at_but_scopes_are_independent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "scoped.sqlite3"
    first = run_seal.seal_local_update_run(
        database_path,
        scope="scope-a",
        idempotency_key="first",
        payload=_payload(),
    )
    alias = run_seal.seal_local_update_run(
        database_path,
        scope="scope-a",
        idempotency_key="second",
        payload=_payload(),
    )
    foreign = run_seal.seal_local_update_run(
        database_path,
        scope="scope-b",
        idempotency_key="first",
        payload=_payload(),
    )

    assert (alias.seal_id, alias.content_hash, alias.canonical, alias.created_at) == (
        first.seal_id,
        first.content_hash,
        first.canonical,
        first.created_at,
    )
    assert alias.idempotency_key == "second"
    assert foreign.seal_id != first.seal_id
    assert _counts(database_path) == (2, 3)


def test_changed_scoped_idempotency_key_conflicts_without_partial_write(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "conflict.sqlite3"
    winner = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="shared",
        payload=_payload(0),
    )

    with pytest.raises(
        run_seal.LocalUpdateRunSealConflictError,
        match="idempotency key",
    ):
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="shared",
            payload=_payload(1),
        )

    assert _counts(database_path) == (1, 1)
    assert (
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="shared",
        )
        == winner
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"not_finite": float("nan")},
        {"tuple_is_not_json": (1, 2)},
        {1: "non-string key"},
        {"surrogate": "\ud800"},
    ],
    ids=("nan", "tuple", "non-string-key", "surrogate"),
)
def test_payload_must_be_a_strict_json_object(
    tmp_path: Path,
    payload: dict[object, object],
) -> None:
    database_path = tmp_path / "invalid.sqlite3"
    with pytest.raises(run_seal.LocalUpdateRunSealValidationError):
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="invalid",
            payload=payload,  # type: ignore[arg-type]
        )
    assert not database_path.exists()


def test_cyclic_payload_is_rejected_before_io(tmp_path: Path) -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload
    database_path = tmp_path / "cyclic.sqlite3"

    with pytest.raises(run_seal.LocalUpdateRunSealValidationError, match="cyclic"):
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="invalid",
            payload=payload,
        )
    assert not database_path.exists()


@pytest.mark.parametrize(
    "stage",
    ("after_core_insert", "after_alias_insert", "after_readback", "before_commit"),
)
def test_precommit_failure_rolls_back_every_stage_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    database_path = tmp_path / f"rollback-{stage}.sqlite3"
    injected = sqlite3.OperationalError(f"injected {stage}")

    def fail(selected: str) -> None:
        if selected == stage:
            raise injected

    monkeypatch.setattr(run_seal, "_transaction_fault_hook", fail)
    with pytest.raises(run_seal.LocalUpdateRunSealPersistenceError) as raised:
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="retry-key",
            payload=_payload(),
        )
    assert raised.value.__cause__ is injected
    assert _counts(database_path) == (0, 0)

    monkeypatch.setattr(run_seal, "_transaction_fault_hook", lambda _stage: None)
    recovered = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="retry-key",
        payload=_payload(),
    )
    assert _counts(database_path) == (1, 1)
    assert (
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="retry-key",
        )
        == recovered
    )


def test_real_commit_followed_by_ack_loss_recovers_with_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "lost-ack.sqlite3"
    injected = sqlite3.OperationalError("lost COMMIT acknowledgement")
    failed = False

    def lose_ack(stage: str) -> None:
        nonlocal failed
        if stage == "after_commit" and not failed:
            failed = True
            raise injected

    monkeypatch.setattr(run_seal, "_transaction_fault_hook", lose_ack)
    with pytest.raises(run_seal.LocalUpdateRunSealCommitUnknownError) as raised:
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="lost-ack",
            payload=_payload(),
        )
    assert raised.value.__cause__ is injected
    assert _counts(database_path) == (1, 1)

    monkeypatch.setattr(run_seal, "_transaction_fault_hook", lambda _stage: None)
    persisted = run_seal.get_local_update_run_seal(
        database_path,
        idempotency_key="lost-ack",
    )
    retried = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="lost-ack",
        payload=_payload(),
    )
    assert retried == persisted
    assert _counts(database_path) == (1, 1)


def test_write_transaction_is_immediate_and_readback_precedes_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    stages: list[str] = []
    original_connect = run_seal._connect

    def recording_connect(path: str, identity: tuple[int, int]):
        connection = original_connect(path, identity)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(run_seal, "_connect", recording_connect)
    monkeypatch.setattr(run_seal, "_transaction_fault_hook", stages.append)
    run_seal.seal_local_update_run(
        tmp_path / "trace.sqlite3",
        idempotency_key="trace",
        payload=_payload(),
    )

    normalized = tuple(statement.upper() for statement in statements)
    begin_index = normalized.index("BEGIN IMMEDIATE")
    core_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("INSERT INTO LOCAL_UPDATE_RUN_SEALS ")
    )
    commit_index = len(normalized) - 1 - normalized[::-1].index("COMMIT")
    assert begin_index < core_index < commit_index
    assert stages == [
        "after_begin",
        "after_core_insert",
        "after_alias_insert",
        "after_readback",
        "before_commit",
        "before_post_commit_identity_check",
        "after_commit",
    ]


def test_postcommit_path_replacement_never_returns_a_false_durable_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "replaced.sqlite3"
    replacement_path = tmp_path / "replacement.sqlite3"
    replacement = run_seal.seal_local_update_run(
        replacement_path,
        idempotency_key="replacement-owner",
        payload=_payload(7),
    )
    replaced = False

    def replace_after_commit(stage: str) -> None:
        nonlocal replaced
        if stage == "before_post_commit_identity_check" and not replaced:
            replaced = True
            os.replace(replacement_path, database_path)

    monkeypatch.setattr(
        run_seal,
        "_transaction_fault_hook",
        replace_after_commit,
    )
    with pytest.raises(run_seal.LocalUpdateRunSealCommitUnknownError) as raised:
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="unlinked-commit",
            payload=_payload(),
        )

    assert replaced
    assert type(raised.value.__cause__) is run_seal.LocalUpdateRunSealCorruptionError
    monkeypatch.setattr(run_seal, "_transaction_fault_hook", lambda _stage: None)
    with pytest.raises(run_seal.LocalUpdateRunSealNotFoundError):
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="unlinked-commit",
        )
    assert (
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="replacement-owner",
        )
        == replacement
    )


def test_concurrent_identical_requests_converge_on_one_durable_alias(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "identical-race.sqlite3"
    worker_count = 8
    barrier = Barrier(worker_count)

    def run(_index: int):
        barrier.wait()
        return run_seal.seal_local_update_run(
            database_path,
            idempotency_key="shared",
            payload=_payload(),
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = tuple(executor.map(run, range(worker_count)))

    assert all(result == results[0] for result in results)
    assert _counts(database_path) == (1, 1)


def test_concurrent_different_requests_have_one_winner_and_no_loser_orphan(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "different-race.sqlite3"
    worker_count = 8
    barrier = Barrier(worker_count)

    def run(index: int):
        barrier.wait()
        try:
            return run_seal.seal_local_update_run(
                database_path,
                idempotency_key="shared",
                payload=_payload(index),
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = tuple(executor.map(run, range(worker_count)))

    winners = tuple(
        outcome for outcome in outcomes if type(outcome) is run_seal.LocalUpdateRunSeal
    )
    conflicts = tuple(
        outcome
        for outcome in outcomes
        if type(outcome) is run_seal.LocalUpdateRunSealConflictError
    )
    assert len(winners) == 1
    assert len(conflicts) == worker_count - 1
    assert _counts(database_path) == (1, 1)
    assert (
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="shared",
        )
        == winners[0]
    )


@pytest.mark.parametrize(
    ("statement", "parameters"),
    [
        (
            "DELETE FROM local_update_run_seal_aliases",
            (),
        ),
        (
            "UPDATE local_update_run_seal_aliases SET binding_hash = ?",
            ("0" * 64,),
        ),
        (
            "UPDATE local_update_run_seals SET storage_hash = ?",
            ("0" * 64,),
        ),
    ],
    ids=("orphan-core", "alias-binding", "core-storage"),
)
def test_orphan_alias_and_core_tampering_fail_closed_globally(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    database_path = tmp_path / "tamper.sqlite3"
    run_seal.seal_local_update_run(
        database_path,
        idempotency_key="owner",
        payload=_payload(),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(statement, parameters)
        connection.commit()

    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError):
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="owner",
        )
    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError):
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="unrelated",
            payload=_payload(2),
        )
    assert _counts(database_path) in {(1, 0), (1, 1)}


def test_alias_pointing_to_missing_core_fails_foreign_key_validation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing-core.sqlite3"
    run_seal.seal_local_update_run(
        database_path,
        idempotency_key="owner",
        payload=_payload(),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE local_update_run_seal_aliases SET seal_id = ?",
            ("lurs_" + "0" * 24,),
        )
        connection.commit()

    with pytest.raises(
        run_seal.LocalUpdateRunSealCorruptionError,
        match="foreign-key",
    ):
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="owner",
        )


@pytest.mark.parametrize(
    "tampered",
    [
        b'{"a":1,"a":1}',
        b'{"value":NaN}',
        b'{ "value":1}',
        b"\xff",
    ],
    ids=("duplicate-key", "nan", "noncanonical-space", "non-ascii"),
)
def test_stored_json_must_remain_strict_and_canonical(
    tmp_path: Path,
    tampered: bytes,
) -> None:
    database_path = tmp_path / "canonical-tamper.sqlite3"
    run_seal.seal_local_update_run(
        database_path,
        idempotency_key="owner",
        payload=_payload(),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE local_update_run_seals SET canonical = ?",
            (tampered,),
        )
        connection.commit()

    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError):
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="owner",
        )


def test_content_id_collision_rolls_back_loser_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "collision.sqlite3"
    monkeypatch.setattr(run_seal, "_content_hash", lambda _canonical: "0" * 64)
    winner = run_seal.seal_local_update_run(
        database_path,
        idempotency_key="winner",
        payload=_payload(0),
    )

    with pytest.raises(run_seal.LocalUpdateRunSealConflictError, match="collision"):
        run_seal.seal_local_update_run(
            database_path,
            idempotency_key="loser",
            payload=_payload(1),
        )

    assert winner.seal_id == "lurs_" + "0" * 24
    assert _counts(database_path) == (1, 1)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT idempotency_key FROM local_update_run_seal_aliases"
        ).fetchall() == [("winner",)]


def test_missing_database_read_does_not_create_and_unsafe_paths_are_rejected(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(run_seal.LocalUpdateRunSealNotFoundError):
        run_seal.get_local_update_run_seal(missing, idempotency_key="missing")
    assert not missing.exists()

    database_path = tmp_path / "safe.sqlite3"
    run_seal.seal_local_update_run(
        database_path,
        idempotency_key="owner",
        payload=_payload(),
    )
    hardlink = tmp_path / "hardlink.sqlite3"
    os.link(database_path, hardlink)
    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError):
        run_seal.get_local_update_run_seal(
            database_path,
            idempotency_key="owner",
        )
    hardlink.unlink()

    symlink = tmp_path / "symlink.sqlite3"
    symlink.symlink_to(database_path)
    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError):
        run_seal.get_local_update_run_seal(
            symlink,
            idempotency_key="owner",
        )


def test_creation_fsyncs_private_file_and_parent_and_connections_are_strong(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_modes: list[int] = []
    original_fsync = run_seal.os.fsync

    def recording_fsync(descriptor: int) -> None:
        observed_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(run_seal.os, "fsync", recording_fsync)
    database_path = tmp_path / "sync.sqlite3"
    run_seal.seal_local_update_run(
        database_path,
        idempotency_key="owner",
        payload=_payload(),
    )

    assert any(stat.S_ISREG(mode) for mode in observed_modes)
    assert any(stat.S_ISDIR(mode) for mode in observed_modes)
    assert database_path.stat().st_mode & 0o777 == 0o600
    identity = run_seal._require_private_database(str(database_path))
    connection = run_seal._connect(str(database_path), identity)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (3,)
        assert connection.execute("PRAGMA fullfsync").fetchone() == (1,)
    finally:
        connection.close()


class _CloseFailureConnection:
    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def close(self) -> None:
        self._inner.close()
        raise RuntimeError("injected close failure")


def test_close_failure_is_mapped_without_overwriting_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "close-read.sqlite3"
    run_seal.seal_local_update_run(
        path,
        idempotency_key="run",
        payload=_payload(),
    )
    original_connect = run_seal._connect

    def connect(database_path, identity):
        return _CloseFailureConnection(original_connect(database_path, identity))

    monkeypatch.setattr(run_seal, "_connect", connect)
    with pytest.raises(run_seal.LocalUpdateRunSealPersistenceError) as close_only:
        run_seal.get_local_update_run_seal(path, idempotency_key="run")
    assert isinstance(close_only.value.__cause__, RuntimeError)

    def corrupt(_cursor):
        raise run_seal.LocalUpdateRunSealCorruptionError("injected corruption")

    monkeypatch.setattr(run_seal, "_validate_schema_locked", corrupt)
    with pytest.raises(run_seal.LocalUpdateRunSealCorruptionError) as primary:
        run_seal.get_local_update_run_seal(path, idempotency_key="run")
    assert primary.value.args == ("injected corruption",)
    assert any(
        "close failed after read error" in note for note in primary.value.__notes__
    )


def test_initialize_close_failure_is_mapped_to_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_connect = run_seal._connect

    def connect(database_path, identity):
        return _CloseFailureConnection(original_connect(database_path, identity))

    monkeypatch.setattr(run_seal, "_connect", connect)
    with pytest.raises(run_seal.LocalUpdateRunSealPersistenceError) as caught:
        run_seal.seal_local_update_run(
            tmp_path / "close-init.sqlite3",
            idempotency_key="run",
            payload=_payload(),
        )
    assert isinstance(caught.value.__cause__, RuntimeError)
