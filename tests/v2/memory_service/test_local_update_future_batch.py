# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from examples.memory_service import local_update_future_batch as future
from examples.memory_service import scoped_codebook_eval as harness

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

_BASE = datetime(2026, 7, 12, tzinfo=UTC)


def _database(tmp_path):
    path = tmp_path / "future.sqlite3"
    store = SQLiteMemoryStore(path)
    scope = MemoryScope(
        tenant_id="memory-eval",
        namespace="local-update-future-v1",
        subject_id="opaque-future-test",
    )
    rows = (
        ("project-abc234", "ABCDE"),
        ("project-def567", "FGHJK"),
    )
    revision_ids: list[str] = []
    for index, (key, value) in enumerate(rows):
        payload = f"{key} = {value}"
        evidence = store.append(
            EvidenceEvent(
                scope=scope,
                session_id="capture-session",
                run_id="capture-run",
                sequence_no=index,
                kind=EvidenceKind.USER_MESSAGE,
                payload=payload,
                observed_at=_BASE + timedelta(seconds=index),
                idempotency_key=f"evidence-{index}",
            )
        )
        candidate = store.append_candidate(
            CandidateProposal(
                scope=scope,
                content=payload,
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=f"candidate-{index}",
            )
        )
        revision = store.append_revision(
            RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
                idempotency_key=f"revision-{index}",
            )
        )
        revision_ids.append(revision.revision_id)
    release = store.append_release(
        ReleaseManifest(scope=scope, revision_ids=tuple(revision_ids)),
        idempotency_key="release",
    )
    return path, scope, release.release_id


def _items(tmp_path):
    path, scope, release_id = _database(tmp_path)
    return (
        future.FutureReleaseQueryV1(
            logical_index=40,
            database_path=str(path),
            scope=scope,
            release_id=release_id,
            query=(
                "What is the current code for project-abc234? "
                "Reply with exactly the code or UNKNOWN."
            ),
        ),
        future.FutureReleaseQueryV1(
            logical_index=7,
            database_path=str(path),
            scope=scope,
            release_id=release_id,
            query=(
                "What is the current code for project-def567? "
                "Reply with exactly the code or UNKNOWN."
            ),
        ),
    )


def test_verified_batch_mixes_wire_order_and_restores_logical_order(
    tmp_path,
    monkeypatch,
) -> None:
    observed_wire_order: list[int] = []
    original_spawn = future._spawn_future_batch

    def spawn(request, timeout_seconds):
        observed_wire_order.extend(item.execution_index for item in request.items)
        return original_spawn(request, timeout_seconds)

    monkeypatch.setattr(future, "_spawn_future_batch", spawn)
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x01" * 32)

    result = future.execute_verified_release_batch_v1(
        _items(tmp_path),
        timeout_seconds=30,
    )

    assert observed_wire_order == sorted(future._derive_opaque_tokens(2, b"\x01" * 32))
    assert tuple(item.logical_index for item in result) == (40, 7)
    assert tuple(item.response for item in result) == ("ABCDE", "FGHJK")
    assert all(
        item.response_sha256 == hashlib.sha256(item.response.encode()).hexdigest()
        for item in result
    )
    assert len({item.execution_token_sha256 for item in result}) == 2


@pytest.mark.parametrize(
    ("tamper", "reason"),
    (
        ("audit", "assignment_mismatch"),
        ("response", "consumer_receipt"),
        ("missing", "assignment_mismatch"),
        ("state_reuse", "state_reuse"),
    ),
)
def test_verified_batch_rejects_tampered_child_before_exposure(
    tmp_path,
    monkeypatch,
    tamper: str,
    reason: str,
) -> None:
    original_spawn = future._spawn_future_batch

    def spawn(request, timeout_seconds):
        spawned = original_spawn(request, timeout_seconds)
        response = spawned.response
        if tamper == "audit":
            first = replace(response.observations[0], reader_audit=())
            changed = replace(
                response,
                observations=(first, *response.observations[1:]),
            )
            return replace(spawned, response=changed)
        if tamper == "response":
            first = replace(response.observations[0], response="WRONG")
            changed = replace(
                response,
                observations=(first, *response.observations[1:]),
            )
            return replace(spawned, response=changed)
        if tamper == "missing":
            changed = replace(response, observations=response.observations[:-1])
            return replace(spawned, response=changed)
        second = replace(
            response.state_receipts[1],
            store_instance_id=response.state_receipts[0].store_instance_id,
        )
        changed = replace(
            response,
            state_receipts=(response.state_receipts[0], second),
        )
        return replace(spawned, response=changed)

    monkeypatch.setattr(future, "_spawn_future_batch", spawn)
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x02" * 32)

    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(
            _items(tmp_path),
            timeout_seconds=30,
        )
    assert caught.value.reason == reason


def test_verified_batch_rejects_foreign_probe_and_database_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    items = _items(tmp_path)
    original_spawn = future._spawn_future_batch

    def foreign_spawn(request, timeout_seconds):
        spawned = original_spawn(request, timeout_seconds)
        response = spawned.response
        item = request.items[0]
        probe = harness.ForeignProbeObservation(
            execution_index=item.execution_index,
            scope=item.scope,
            release_id=item.source.release_id,
            future_session_id=item.future_session_id,
            future_run_id=item.future_run_id,
            future_pid=response.pid,
            future_process_instance_id=response.process_instance_id,
            reason="release_not_found",
            history_length=0,
        )
        return replace(
            spawned,
            response=replace(response, foreign_probes=(probe,)),
        )

    monkeypatch.setattr(future, "_spawn_future_batch", foreign_spawn)
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x03" * 32)

    with pytest.raises(
        future.FutureReleaseValidationError, match="foreign_probe"
    ) as caught:
        future.execute_verified_release_batch_v1(
            items,
            timeout_seconds=30,
        )
    assert caught.value.reason == "foreign_probe"

    def replacement_spawn(request, timeout_seconds):
        spawned = original_spawn(request, timeout_seconds)
        with sqlite3.connect(items[0].database_path) as connection:
            connection.execute("PRAGMA user_version = 77")
        return spawned

    monkeypatch.setattr(future, "_spawn_future_batch", replacement_spawn)
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x04" * 32)

    with pytest.raises(
        future.FutureReleaseValidationError, match="database_changed"
    ) as caught:
        future.execute_verified_release_batch_v1(
            items,
            timeout_seconds=30,
        )
    assert caught.value.reason == "database_changed"


def test_invalid_seed_fails_before_child_execution(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    def spawn(_request, _timeout_seconds):
        nonlocal calls
        calls += 1
        raise AssertionError("must not execute")

    monkeypatch.setattr(future, "_spawn_future_batch", spawn)
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"short")

    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(
            _items(tmp_path),
        )
    assert caught.value.reason == "opaque_seed"
    assert calls == 0


def test_invalid_assignment_consumes_no_entropy_or_child(tmp_path, monkeypatch) -> None:
    seed_calls = 0
    child_calls = 0

    def seed():
        nonlocal seed_calls
        seed_calls += 1
        return b"\x05" * 32

    def spawn(_request, _timeout_seconds):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("must not execute")

    items = _items(tmp_path)
    invalid = (replace(items[0], query="invalid query"), items[1])
    monkeypatch.setattr(future, "_sample_execution_seed", seed)
    monkeypatch.setattr(future, "_spawn_future_batch", spawn)

    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(invalid)
    assert caught.value.reason == "assignment_invalid"
    assert seed_calls == child_calls == 0


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_timeout_is_rejected_before_entropy(
    tmp_path,
    monkeypatch,
    timeout: float,
) -> None:
    calls = 0

    def seed():
        nonlocal calls
        calls += 1
        return b"\x06" * 32

    monkeypatch.setattr(future, "_sample_execution_seed", seed)
    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(
            _items(tmp_path),
            timeout_seconds=timeout,
        )
    assert caught.value.reason == "closed_schema"
    assert calls == 0
    assert not math.isfinite(timeout)


def test_huge_integer_timeout_is_stably_rejected_before_entropy(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    def seed():
        nonlocal calls
        calls += 1
        return b"\x06" * 32

    monkeypatch.setattr(future, "_sample_execution_seed", seed)
    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(
            _items(tmp_path),
            timeout_seconds=10**10000,
        )
    assert caught.value.reason == "closed_schema"
    assert calls == 0


def test_falsey_non_tuple_foreign_probe_collection_is_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    original_spawn = future._spawn_future_batch

    def spawn(request, timeout_seconds):
        spawned = original_spawn(request, timeout_seconds)
        return replace(
            spawned,
            response=replace(spawned.response, foreign_probes=[]),
        )

    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x07" * 32)
    monkeypatch.setattr(future, "_spawn_future_batch", spawn)
    with pytest.raises(future.FutureReleaseValidationError) as caught:
        future.execute_verified_release_batch_v1(
            _items(tmp_path),
            timeout_seconds=30,
        )
    assert caught.value.reason == "child_response"
