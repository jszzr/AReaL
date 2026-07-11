# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable SQLite Memory Service backend."""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
import shutil
import sqlite3
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from time import monotonic
from typing import Any

import pytest

import areal.v2.memory_service._sqlite_backend as sqlite_backend
import areal.v2.memory_service.sqlite_store as sqlite_store_module
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryPersistenceBusyError,
    MemoryPersistenceCorruptionError,
    MemoryPersistenceError,
    MemoryPersistenceSchemaError,
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_store import MemoryReleaseStore
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).upper()


@dataclass(slots=True)
class _SQLiteFailurePlan:
    after_statement: str | None = None
    after_occurrence: int = 1
    after_statement_error: BaseException | None = None
    after_statement_exit_code: int | None = None
    after_statement_check: Callable[[str], None] | None = None
    before_commit_error: BaseException | None = None
    after_commit_error: BaseException | None = None
    rollback_error: BaseException | None = None
    close_error: BaseException | None = None
    events: list[str] = field(default_factory=list)
    _after_matches: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        actions = (
            int(self.after_statement_error is not None)
            + int(self.after_statement_exit_code is not None)
            + int(self.after_statement_check is not None)
        )
        if (self.after_statement is None and actions != 0) or (
            self.after_statement is not None and actions != 1
        ):
            raise ValueError(
                "after_statement requires exactly one error, exit, or check action"
            )
        if type(self.after_occurrence) is not int or self.after_occurrence < 1:
            raise ValueError("after_occurrence must be a positive integer")
        if self.after_statement_exit_code is not None and (
            type(self.after_statement_exit_code) is not int
            or not 1 <= self.after_statement_exit_code <= 255
        ):
            raise ValueError("after_statement_exit_code must be between 1 and 255")
        if self.before_commit_error is not None and self.after_commit_error is not None:
            raise ValueError(
                "before_commit_error and after_commit_error are mutually exclusive"
            )
        if self.after_statement is not None:
            normalized = _normalize_sql(self.after_statement)
            if not normalized:
                raise ValueError("after_statement must not be blank")
            self.after_statement = normalized

    def before_execute(self, normalized_sql: str) -> None:
        self.events.append(f"attempt:{normalized_sql}")
        if normalized_sql == "COMMIT" and self.before_commit_error is not None:
            error = self.before_commit_error
            self.before_commit_error = None
            self.events.append("fail-before:COMMIT")
            raise error
        if normalized_sql == "ROLLBACK" and self.rollback_error is not None:
            error = self.rollback_error
            self.rollback_error = None
            self.events.append("fail-before:ROLLBACK")
            raise error

    def after_execute(self, normalized_sql: str) -> None:
        self.events.append(f"executed:{normalized_sql}")
        if normalized_sql == "COMMIT" and self.after_commit_error is not None:
            error = self.after_commit_error
            self.after_commit_error = None
            self.events.append("fail-after:COMMIT")
            raise error
        if self.after_statement is None or self.after_statement not in normalized_sql:
            return
        self._after_matches += 1
        if self._after_matches < self.after_occurrence:
            return
        if self.after_statement_check is not None:
            # Checks are recurring probes from the threshold onward; error and exit
            # actions remain one-shot at the exact selected occurrence.
            self.after_statement_check(normalized_sql)
            return
        if self._after_matches != self.after_occurrence:
            return
        if self.after_statement_exit_code is not None:
            os._exit(self.after_statement_exit_code)
        assert self.after_statement_error is not None
        error = self.after_statement_error
        self.after_statement_error = None
        self.events.append(f"fail-after:{normalized_sql}")
        raise error


class _FaultInjectingCursor:
    def __init__(
        self,
        real_cursor: sqlite3.Cursor,
        plan: _SQLiteFailurePlan,
    ) -> None:
        self._real_cursor = real_cursor
        self._plan = plan

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _FaultInjectingCursor:
        normalized = _normalize_sql(sql)
        self._plan.before_execute(normalized)
        self._real_cursor.execute(sql, parameters)
        self._plan.after_execute(normalized)
        return self

    def executemany(
        self,
        sql: str,
        parameters: object,
    ) -> _FaultInjectingCursor:
        normalized = _normalize_sql(sql)
        self._plan.before_execute(normalized)
        self._real_cursor.executemany(sql, parameters)
        self._plan.after_execute(normalized)
        return self

    def __iter__(self) -> Any:
        return iter(self._real_cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cursor, name)


class _FaultInjectingConnection:
    def __init__(
        self,
        real_connection: sqlite3.Connection,
        plan: _SQLiteFailurePlan,
    ) -> None:
        self._real_connection = real_connection
        self._plan = plan

    def cursor(
        self,
        *args: object,
        **kwargs: object,
    ) -> _FaultInjectingCursor:
        return _FaultInjectingCursor(
            self._real_connection.cursor(*args, **kwargs),
            self._plan,
        )

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _FaultInjectingCursor:
        return self.cursor().execute(sql, parameters)

    def executemany(
        self,
        sql: str,
        parameters: object,
    ) -> _FaultInjectingCursor:
        return self.cursor().executemany(sql, parameters)

    def close(self) -> None:
        self._plan.events.append("attempt:CLOSE")
        self._real_connection.close()
        self._plan.events.append("executed:CLOSE")
        if self._plan.close_error is not None:
            error = self._plan.close_error
            self._plan.close_error = None
            self._plan.events.append("fail-after:CLOSE")
            raise error

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_connection, name)


class _ReadbackOverrideCursor:
    def __init__(
        self,
        real_cursor: sqlite3.Cursor,
        overrides: dict[str, tuple[object, ...]],
    ) -> None:
        self._real_cursor = real_cursor
        self._overrides = overrides
        self._last_sql = ""

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _ReadbackOverrideCursor:
        self._last_sql = _normalize_sql(sql)
        self._real_cursor.execute(sql, parameters)
        return self

    def fetchone(self) -> tuple[object, ...] | None:
        if self._last_sql in self._overrides:
            return self._overrides[self._last_sql]
        return self._real_cursor.fetchone()

    def __iter__(self) -> Any:
        return iter(self._real_cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cursor, name)


class _ReadbackOverrideConnection:
    def __init__(
        self,
        real_connection: sqlite3.Connection,
        overrides: dict[str, tuple[object, ...]],
    ) -> None:
        self._real_connection = real_connection
        self._overrides = overrides

    def cursor(
        self,
        *args: object,
        **kwargs: object,
    ) -> _ReadbackOverrideCursor:
        return _ReadbackOverrideCursor(
            self._real_connection.cursor(*args, **kwargs),
            self._overrides,
        )

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _ReadbackOverrideCursor:
        return self.cursor().execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_connection, name)


@dataclass(slots=True)
class _SQLiteReadOverridePlan:
    fetchall_transforms: dict[
        str,
        Callable[
            [tuple[tuple[object, ...], ...], object],
            tuple[tuple[object, ...], ...],
        ],
    ] = field(default_factory=dict)
    fetchone_transforms: dict[
        str,
        Callable[
            [tuple[object, ...] | None, object],
            tuple[object, ...] | None,
        ],
    ] = field(default_factory=dict)
    executions: list[tuple[str, object]] = field(default_factory=list)


class _SQLiteReadOverrideCursor:
    def __init__(
        self,
        real_cursor: sqlite3.Cursor,
        plan: _SQLiteReadOverridePlan,
    ) -> None:
        self._real_cursor = real_cursor
        self._plan = plan
        self._last_sql = ""
        self._parameters: object = ()

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _SQLiteReadOverrideCursor:
        self._last_sql = _normalize_sql(sql)
        self._parameters = parameters
        self._plan.executions.append((self._last_sql, parameters))
        self._real_cursor.execute(sql, parameters)
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        rows = tuple(tuple(row) for row in self._real_cursor.fetchall())
        transform = self._plan.fetchall_transforms.get(self._last_sql)
        if transform is not None:
            rows = transform(rows, self._parameters)
        return list(rows)

    def fetchone(self) -> tuple[object, ...] | None:
        row = self._real_cursor.fetchone()
        normalized_row = None if row is None else tuple(row)
        transform = self._plan.fetchone_transforms.get(self._last_sql)
        if transform is not None:
            normalized_row = transform(normalized_row, self._parameters)
        return normalized_row

    def __iter__(self) -> Any:
        return iter(self._real_cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cursor, name)


class _SQLiteReadOverrideConnection:
    def __init__(
        self,
        real_connection: sqlite3.Connection,
        plan: _SQLiteReadOverridePlan,
    ) -> None:
        self._real_connection = real_connection
        self._plan = plan

    def cursor(
        self,
        *args: object,
        **kwargs: object,
    ) -> _SQLiteReadOverrideCursor:
        return _SQLiteReadOverrideCursor(
            self._real_connection.cursor(*args, **kwargs),
            self._plan,
        )

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _SQLiteReadOverrideCursor:
        return self.cursor().execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_connection, name)


def _install_sqlite_read_override_proxy(
    monkeypatch: pytest.MonkeyPatch,
    plan: _SQLiteReadOverridePlan,
) -> None:
    real_connect = sqlite_backend._connect

    def connect(path: str) -> _SQLiteReadOverrideConnection:
        return _SQLiteReadOverrideConnection(real_connect(path), plan)

    monkeypatch.setattr(sqlite_backend, "_connect", connect)


@dataclass(slots=True)
class _SQLiteReleaseTamperPlan:
    trigger_key: str
    tamper: Callable[[sqlite3.Cursor], None]
    observed_sql: dict[str, str]
    events: list[str] = field(default_factory=list)
    trigger_hits: int = 0
    tamper_hits: int = 0


class _SQLiteReleaseTamperCursor:
    def __init__(
        self,
        real_cursor: sqlite3.Cursor,
        plan: _SQLiteReleaseTamperPlan,
    ) -> None:
        self._real_cursor = real_cursor
        self._plan = plan

    def execute(
        self,
        sql: str,
        parameters: object = (),
    ) -> _SQLiteReleaseTamperCursor:
        normalized = _normalize_sql(sql)
        self._real_cursor.execute(sql, parameters)
        if normalized.startswith("INSERT INTO MEMORY_RELEASE_ALIASES"):
            assert isinstance(parameters, tuple)
            assert len(parameters) == 4
            if parameters[1] == self._plan.trigger_key:
                self._plan.trigger_hits += 1
                self._plan.events.append("alias-insert")
                assert self._plan.trigger_hits == 1
                self._plan.tamper(self._real_cursor)
                self._plan.tamper_hits += 1
                self._plan.events.append("tamper")
        elif self._plan.tamper_hits:
            observed = self._plan.observed_sql.get(normalized)
            if observed is not None:
                self._plan.events.append(observed)
            elif normalized == "ROLLBACK":
                self._plan.events.append("rollback")
            elif normalized == "COMMIT":
                self._plan.events.append("commit")
        return self

    def __iter__(self) -> Any:
        return iter(self._real_cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cursor, name)


class _SQLiteReleaseTamperConnection:
    def __init__(
        self,
        real_connection: sqlite3.Connection,
        plan: _SQLiteReleaseTamperPlan,
    ) -> None:
        self._real_connection = real_connection
        self._plan = plan

    def cursor(
        self,
        *args: object,
        **kwargs: object,
    ) -> _SQLiteReleaseTamperCursor:
        return _SQLiteReleaseTamperCursor(
            self._real_connection.cursor(*args, **kwargs),
            self._plan,
        )

    def close(self) -> None:
        self._real_connection.close()
        if self._plan.tamper_hits:
            self._plan.events.append("close")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_connection, name)


def _install_sqlite_release_tamper_proxy(
    monkeypatch: pytest.MonkeyPatch,
    plan: _SQLiteReleaseTamperPlan,
) -> None:
    real_connect = sqlite_backend._connect

    def connect(path: str) -> _SQLiteReleaseTamperConnection:
        return _SQLiteReleaseTamperConnection(real_connect(path), plan)

    monkeypatch.setattr(sqlite_backend, "_connect", connect)


def _install_sqlite_failure_proxy(
    monkeypatch: pytest.MonkeyPatch,
    plan: _SQLiteFailurePlan,
) -> None:
    real_connect = sqlite_backend._connect

    def connect(path: str) -> _FaultInjectingConnection:
        return _FaultInjectingConnection(real_connect(path), plan)

    monkeypatch.setattr(sqlite_backend, "_connect", connect)


def _spawn_initialize_database(
    database_path: str,
    start_event: object,
    result_queue: object,
) -> None:
    if not start_event.wait(timeout=10):  # type: ignore[attr-defined]
        result_queue.put(("timeout", "start event"))  # type: ignore[attr-defined]
        return
    try:
        sqlite_backend._initialize_database(database_path)
    except BaseException as error:
        result_queue.put((type(error).__name__, str(error)))  # type: ignore[attr-defined]
    else:
        result_queue.put(("ok", ""))  # type: ignore[attr-defined]


def _spawn_crash_during_initialize(
    database_path: str,
    sql_marker: str,
) -> None:
    import areal.v2.memory_service._sqlite_backend as backend

    real_connect = backend._connect
    plan = _SQLiteFailurePlan(
        after_statement=sql_marker,
        after_statement_exit_code=24,
    )

    def crashing_connect(path: str) -> _FaultInjectingConnection:
        return _FaultInjectingConnection(real_connect(path), plan)

    backend._connect = crashing_connect
    backend._initialize_database(database_path)


_SCOPE_INSERT_SQL = """
INSERT INTO memory_scopes (
    scope_id,
    tenant_id,
    namespace,
    subject_id
) VALUES (?, ?, ?, ?)
"""


def _insert_test_scope(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        _SCOPE_INSERT_SQL,
        (1, "tenant-1", "assistant-memory", "user-1"),
    )


def _read_test_scopes(database_path: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT scope_id, tenant_id, namespace, subject_id "
                "FROM memory_scopes ORDER BY scope_id"
            ).fetchall()
        ]
    finally:
        connection.close()


_EVIDENCE_INSTANT = datetime(2026, 7, 8, 1, 2, 3, 456000, tzinfo=UTC)


def _make_sqlite_evidence(**overrides: object) -> EvidenceEvent:
    values: dict[str, object] = {
        "scope": MemoryScope("tenant-1", "assistant-memory", "user-1"),
        "session_id": "session-1",
        "run_id": "run-1",
        "sequence_no": 0,
        "kind": EvidenceKind.USER_MESSAGE,
        "payload": "hello, 世界",
        "observed_at": _EVIDENCE_INSTANT,
        "idempotency_key": "evidence-request-1",
    }
    values.update(overrides)
    return EvidenceEvent(**values)  # type: ignore[arg-type]


def _make_sqlite_candidate(**overrides: object) -> CandidateProposal:
    values: dict[str, object] = {
        "scope": MemoryScope("tenant-1", "assistant-memory", "user-1"),
        "content": "remember the durable answer",
        "evidence_ids": ("evd_missing",),
        "idempotency_key": "candidate-request-1",
    }
    values.update(overrides)
    return CandidateProposal(**values)  # type: ignore[arg-type]


def _make_sqlite_revision(**overrides: object) -> RevisionProposal:
    values: dict[str, object] = {
        "scope": MemoryScope("tenant-1", "assistant-memory", "user-1"),
        "candidate_id": "cand_missing",
        "operation": RevisionOperation.ADD,
        "parent_revision_id": None,
        "idempotency_key": "revision-request-1",
    }
    values.update(overrides)
    return RevisionProposal(**values)  # type: ignore[arg-type]


def _append_sqlite_revision_candidate(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    index: int,
    key: str,
) -> tuple[MemoryCandidate, EvidenceRecord]:
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            sequence_no=index,
            payload=f"revision evidence {key}",
            idempotency_key=f"revision-evidence-{key}",
        )
    )
    candidate = store.append_candidate(
        _make_sqlite_candidate(
            scope=scope,
            content=f"revision candidate {key}",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key=f"revision-candidate-{key}",
        )
    )
    return candidate, evidence


def _append_sqlite_release_root(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    *,
    index: int,
    key: str,
) -> tuple[MemoryRevision, MemoryCandidate, EvidenceRecord]:
    candidate, evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=index,
        key=key,
    )
    revision = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidate.candidate_id,
            idempotency_key=f"release-revision-{key}",
        )
    )
    return revision, candidate, evidence


_SQLITE_REVISION_RACE_SIZE = 6
_SQLITE_REVISION_RACE_TIMEOUT_SECONDS = 30.0


def _run_sqlite_revision_race(
    database_path: str | Path,
    proposals: tuple[RevisionProposal, ...],
) -> tuple[MemoryRevision | MemoryServiceError, ...]:
    stores = tuple(SQLiteMemoryStore(database_path) for _proposal in proposals)
    barrier = Barrier(len(proposals), timeout=10.0)

    def worker(index: int) -> MemoryRevision | MemoryServiceError:
        barrier.wait()
        try:
            return stores[index].append_revision(proposals[index])
        except MemoryServiceError as error:
            return error

    with ThreadPoolExecutor(max_workers=len(proposals)) as executor:
        futures = tuple(
            executor.submit(worker, index) for index in range(len(proposals))
        )
        deadline = monotonic() + _SQLITE_REVISION_RACE_TIMEOUT_SECONDS
        return tuple(
            future.result(timeout=max(0.0, deadline - monotonic()))
            for future in futures
        )


_SQLITE_RELEASE_RACE_SIZE = 6
_SQLITE_RELEASE_RACE_TIMEOUT_SECONDS = 30.0


def _run_sqlite_release_race(
    database_path: str | Path,
    requests: tuple[tuple[ReleaseManifest, str], ...],
) -> tuple[MemoryRelease | MemoryServiceError, ...]:
    assert len(requests) == _SQLITE_RELEASE_RACE_SIZE
    stores = tuple(SQLiteMemoryStore(database_path) for _request in requests)
    deadline = monotonic() + _SQLITE_RELEASE_RACE_TIMEOUT_SECONDS
    barrier = Barrier(len(requests))

    def worker(index: int) -> MemoryRelease | MemoryServiceError:
        barrier.wait(timeout=max(0.0, deadline - monotonic()))
        manifest, idempotency_key = requests[index]
        try:
            return stores[index].append_release(
                manifest,
                idempotency_key=idempotency_key,
            )
        except MemoryServiceError as error:
            return error

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = tuple(
            executor.submit(worker, index) for index in range(len(requests))
        )
        return tuple(
            future.result(timeout=max(0.0, deadline - monotonic()))
            for future in futures
        )


def _memory_graph_state(
    database_path: str | Path,
) -> tuple[tuple[int, int, int, int], list[tuple[object, ...]]]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "memory_scopes",
                "memory_evidence",
                "memory_candidates",
                "memory_candidate_evidence",
            )
        )
        foreign_key_violations = [
            tuple(row)
            for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        ]
    finally:
        connection.close()
    assert len(counts) == 4
    assert all(type(count) is int for count in counts)
    return (counts[0], counts[1], counts[2], counts[3]), foreign_key_violations


def _revision_graph_state(
    database_path: str | Path,
) -> tuple[tuple[int, int, int, int, int], list[tuple[object, ...]]]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "memory_scopes",
                "memory_evidence",
                "memory_candidates",
                "memory_candidate_evidence",
                "memory_revisions",
            )
        )
        foreign_key_violations = [
            tuple(row)
            for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        ]
    finally:
        connection.close()
    assert len(counts) == 5
    assert all(type(count) is int for count in counts)
    return (
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
    ), foreign_key_violations


def _revision_graph_rows(
    database_path: str | Path,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.text_factory = bytes
    try:
        return tuple(
            tuple(
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY rowid"
                ).fetchall()
            )
            for table in (
                "memory_scopes",
                "memory_evidence",
                "memory_candidates",
                "memory_candidate_evidence",
                "memory_revisions",
            )
        )
    finally:
        connection.close()


def _release_graph_state(
    database_path: str | Path,
) -> tuple[
    tuple[int, int, int, int, int, int, int, int],
    list[tuple[object, ...]],
]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "memory_scopes",
                "memory_evidence",
                "memory_candidates",
                "memory_candidate_evidence",
                "memory_revisions",
                "memory_releases",
                "memory_release_aliases",
                "memory_release_revisions",
            )
        )
        foreign_key_violations = [
            tuple(row)
            for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        ]
    finally:
        connection.close()
    assert len(counts) == 8
    assert all(type(count) is int for count in counts)
    return (
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        counts[5],
        counts[6],
        counts[7],
    ), foreign_key_violations


def _release_graph_rows(
    database_path: str | Path,
) -> tuple[
    tuple[tuple[tuple[object, ...], ...], ...],
    tuple[tuple[object, ...], ...],
]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.text_factory = bytes
    try:
        rows = tuple(
            tuple(
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY rowid"
                ).fetchall()
            )
            for table in (
                "memory_scopes",
                "memory_evidence",
                "memory_candidates",
                "memory_candidate_evidence",
                "memory_revisions",
                "memory_releases",
                "memory_release_aliases",
                "memory_release_revisions",
            )
        )
        foreign_key_violations = tuple(
            tuple(row)
            for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        )
    finally:
        connection.close()
    return rows, foreign_key_violations


def _assert_sqlite_release_append_failure_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    store: SQLiteMemoryStore,
    database_path: str | Path,
    manifest: ReleaseManifest,
    *,
    idempotency_key: str,
    error_type: type[MemoryServiceError],
    message: str,
) -> MemoryServiceError:
    before = _release_graph_rows(database_path)
    assert before[1] == ()
    plan = _SQLiteFailurePlan()
    with monkeypatch.context() as guarded:
        _install_sqlite_failure_proxy(guarded, plan)
        with pytest.raises(error_type) as raised:
            store.append_release(manifest, idempotency_key=idempotency_key)
    assert type(raised.value) is error_type
    assert str(raised.value) == message
    assert _release_graph_rows(database_path) == before
    assert not any(
        event.startswith("attempt:INSERT INTO MEMORY_") for event in plan.events
    )
    return raised.value


class _StableDigest:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def hexdigest(self) -> str:
        return self._digest


def _stable_digest_oracle(
    digest_by_canonical: dict[bytes, str],
) -> Callable[[bytes], Any]:
    real_sha256 = hashlib.sha256

    def stable_sha256(canonical: bytes) -> Any:
        digest = digest_by_canonical.get(canonical)
        if digest is None:
            return real_sha256(canonical)
        return _StableDigest(digest)

    return stable_sha256


class _SQLiteMemoryScopeSubclass(MemoryScope):
    pass


class _SQLiteEvidenceEventSubclass(EvidenceEvent):
    pass


class _SQLiteCandidateProposalSubclass(CandidateProposal):
    pass


class _SQLiteRevisionProposalSubclass(RevisionProposal):
    pass


class _SQLiteReleaseManifestSubclass(ReleaseManifest):
    pass


class _SnapshotProbeStr(str):
    override_calls: int

    def __new__(cls, value: str) -> _SnapshotProbeStr:
        instance = str.__new__(cls, value)
        instance.override_calls = 0
        return instance

    def __str__(self) -> str:
        self.override_calls += 1
        return "overridden"

    def strip(self, chars: str | None = None) -> str:
        self.override_calls += 1
        return ""

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        self.override_calls += 1
        return b"overridden"

    def __eq__(self, other: object) -> bool:
        self.override_calls += 1
        return False

    def __hash__(self) -> int:
        self.override_calls += 1
        return str.__hash__(self)


def test_persistence_errors_have_one_narrow_hierarchy() -> None:
    assert MemoryPersistenceError.__bases__ == (MemoryServiceError,)
    assert MemoryPersistenceBusyError.__bases__ == (MemoryPersistenceError,)
    assert MemoryPersistenceSchemaError.__bases__ == (MemoryPersistenceError,)
    assert MemoryPersistenceCorruptionError.__bases__ == (MemoryPersistenceError,)
    assert issubclass(MemoryPersistenceError, MemoryServiceError)
    assert issubclass(MemoryPersistenceBusyError, MemoryPersistenceError)
    assert issubclass(MemoryPersistenceSchemaError, MemoryPersistenceError)
    assert issubclass(MemoryPersistenceCorruptionError, MemoryPersistenceError)
    assert not issubclass(MemoryPersistenceSchemaError, MemoryPersistenceBusyError)
    assert not issubclass(
        MemoryPersistenceCorruptionError,
        MemoryPersistenceSchemaError,
    )


def test_database_path_snapshots_one_string_valued_path_like(
    tmp_path: Path,
) -> None:
    class OneShotPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("path-like object was evaluated twice")
            return str(tmp_path / "memory.sqlite3")

    source = OneShotPath()
    result = sqlite_backend._snapshot_database_path(source)
    assert source.calls == 1
    assert type(result) is str
    assert result == os.path.abspath(tmp_path / "memory.sqlite3")


def test_sqlite_store_constructor_snapshots_path_once_and_survives_chdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()

    class OneShotPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("database path was evaluated more than once")
            return "memory.sqlite3"

    source = OneShotPath()
    monkeypatch.chdir(first_directory)
    store = SQLiteMemoryStore(source)
    monkeypatch.chdir(second_directory)
    reopened = SQLiteMemoryStore(first_directory / "memory.sqlite3")

    assert source.calls == 1
    assert store._database_path == str(first_directory / "memory.sqlite3")
    assert reopened._database_path == store._database_path
    assert (first_directory / "memory.sqlite3").is_file()
    assert not (second_directory / "memory.sqlite3").exists()


def test_sqlite_evidence_round_trip_retry_and_reopen_preserve_record(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "memory.sqlite3"
    event = _make_sqlite_evidence()
    first_store = SQLiteMemoryStore(database_path)

    original = first_store.append(event)
    del first_store
    reopened = SQLiteMemoryStore(database_path)
    loaded = reopened.get(event.scope, original.evidence_id)
    retry = reopened.append(_make_sqlite_evidence())

    expected_hash = hashlib.sha256(event.canonical_bytes()).hexdigest()
    assert original.event == event
    assert original.event is not event
    assert loaded == original
    assert retry == original
    assert loaded is not original
    assert retry is not original
    assert loaded.event.canonical_bytes() == event.canonical_bytes()
    assert original.content_hash == expected_hash
    assert original.evidence_id == f"evd_{expected_hash[:24]}"
    assert loaded.created_at == original.created_at == retry.created_at
    assert original.created_at.tzinfo is UTC


def test_sqlite_evidence_exact_input_boundaries_snapshot_before_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    event = _make_sqlite_evidence()
    subclass_event = _SQLiteEvidenceEventSubclass(
        scope=event.scope,
        session_id=event.session_id,
        run_id=event.run_id,
        sequence_no=event.sequence_no,
        kind=event.kind,
        payload=event.payload,
        observed_at=event.observed_at,
        idempotency_key=event.idempotency_key,
    )

    record = store.append(event)
    query_id = _SnapshotProbeStr(record.evidence_id)
    assert store.get(event.scope, query_id) == record
    assert query_id.override_calls == 0
    for missing_id in ("", " \t", "\x00"):
        with pytest.raises(EvidenceNotFoundError) as raised:
            store.get(event.scope, missing_id)
        assert str(raised.value) == f"evidence {missing_id!r} was not found"

    subclass_scope = _SQLiteMemoryScopeSubclass(
        event.scope.tenant_id,
        event.scope.namespace,
        event.scope.subject_id,
    )

    def transaction_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation reached SQLite I/O")

    monkeypatch.setattr(
        sqlite_store_module,
        "_write_transaction",
        transaction_must_not_start,
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_read_transaction",
        transaction_must_not_start,
    )
    with pytest.raises(TypeError, match="event must be an EvidenceEvent"):
        store.append(subclass_event)
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get(subclass_scope, "\ud800")
    with pytest.raises(ValueError, match="evidence_id must be valid UTF-8"):
        store.get(event.scope, "\ud800")


def test_sqlite_evidence_queries_hide_foreign_scope_and_snapshot_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    first_scope = MemoryScope("tenant-1", "assistant-memory", "user-1")
    second_scope = MemoryScope("tenant-1", "assistant-memory", "user-2")
    missing_scope = MemoryScope("tenant-1", "assistant-memory", "user-3")
    first = store.append(_make_sqlite_evidence(scope=first_scope, payload="first"))
    second = store.append(_make_sqlite_evidence(scope=second_scope, payload="second"))

    assert first.evidence_id != second.evidence_id
    assert store.list(first_scope) == (first,)
    assert store.list(second_scope) == (second,)
    assert store.list(missing_scope) == ()
    with pytest.raises(EvidenceNotFoundError) as missing_error:
        store.get(first_scope, "evd_missing")
    with pytest.raises(EvidenceNotFoundError) as foreign_error:
        store.get(second_scope, first.evidence_id)
    assert str(missing_error.value) == "evidence 'evd_missing' was not found"
    assert str(foreign_error.value) == (f"evidence {first.evidence_id!r} was not found")

    session_probe = _SnapshotProbeStr("missing-session")
    run_probe = _SnapshotProbeStr("missing-run")
    assert store.list(first_scope, session_id=session_probe) == ()
    assert store.list(first_scope, run_id=run_probe) == ()
    assert session_probe.override_calls == 0
    assert run_probe.override_calls == 0
    for missing_filter in ("", " \t", "\x00"):
        assert store.list(first_scope, session_id=missing_filter) == ()
        assert store.list(first_scope, run_id=missing_filter) == ()

    subclass_scope = _SQLiteMemoryScopeSubclass(
        first_scope.tenant_id,
        first_scope.namespace,
        first_scope.subject_id,
    )

    def transaction_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation reached SQLite I/O")

    monkeypatch.setattr(
        sqlite_store_module,
        "_read_transaction",
        transaction_must_not_start,
    )
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.list(
            subclass_scope,
            session_id="\ud800",
            run_id=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="session_id must be valid UTF-8"):
        store.list(
            first_scope,
            session_id="\ud800",
            run_id=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="run_id must be valid UTF-8"):
        store.list(first_scope, session_id="", run_id="\ud800")


def test_sqlite_evidence_missing_loader_result_is_address_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    event = _make_sqlite_evidence()
    record = store.append(event)

    def missing_loader(
        _cursor: sqlite3.Cursor,
        _scope: MemoryScope,
        _scope_id: int,
        _evidence_id: str,
    ) -> None:
        return None

    monkeypatch.setattr(sqlite_store_module, "_load_evidence", missing_loader)
    for operation in ("get", "retry", "list"):
        with pytest.raises(
            MemoryPersistenceCorruptionError,
            match="evidence address refers to a missing row",
        ):
            if operation == "get":
                store.get(event.scope, record.evidence_id)
            elif operation == "retry":
                store.append(event)
            else:
                store.list(event.scope)


def test_sqlite_evidence_list_filters_and_uses_python_contract_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    scope = MemoryScope("tenant-1", "assistant-memory", "ordered-user")
    records = (
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-b",
                run_id="run-a",
                sequence_no=0,
                idempotency_key="later-session",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-b",
                sequence_no=0,
                idempotency_key="later-run",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-a",
                sequence_no=3,
                observed_at=datetime(2020, 1, 1, tzinfo=UTC),
                idempotency_key="later-sequence",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-a",
                sequence_no=2,
                payload="tie-b",
                idempotency_key="tie-b",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-a",
                sequence_no=1,
                observed_at=datetime(2026, 7, 8, 4, 30, tzinfo=UTC),
                idempotency_key="later-instant",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-a",
                sequence_no=1,
                observed_at=datetime(
                    2026,
                    7,
                    8,
                    12,
                    0,
                    tzinfo=timezone(timedelta(hours=8)),
                ),
                idempotency_key="earlier-instant",
            )
        ),
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                session_id="session-a",
                run_id="run-a",
                sequence_no=2,
                payload="tie-a",
                idempotency_key="tie-a",
            )
        ),
    )

    real_connect = sqlite_backend._connect

    class ReverseEvidenceRowsCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> ReverseEvidenceRowsCursor:
            self._last_sql = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            if self._last_sql.startswith(
                "SELECT SCOPE_ID, EVIDENCE_ID FROM MEMORY_EVIDENCE"
            ):
                rows.reverse()
            return rows

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class ReverseEvidenceRowsConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> ReverseEvidenceRowsCursor:
            return ReverseEvidenceRowsCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> ReverseEvidenceRowsConnection:
        return ReverseEvidenceRowsConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    expected = tuple(
        sorted(
            records,
            key=lambda record: (
                record.event.session_id,
                record.event.run_id,
                record.event.sequence_no,
                record.event.observed_at,
                record.evidence_id,
            ),
        )
    )

    assert store.list(scope) == expected
    assert store.list(scope, session_id="session-a") == expected[:-1]
    assert store.list(scope, run_id="run-a") == tuple(
        record for record in expected if record.event.run_id == "run-a"
    )
    assert store.list(
        scope,
        session_id="session-a",
        run_id="run-b",
    ) == (records[1],)


def test_sqlite_evidence_idempotency_precedes_true_id_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    scope = MemoryScope("tenant-1", "assistant-memory", "precedence-user")
    idempotency_owner = _make_sqlite_evidence(
        scope=scope,
        payload="idempotency owner",
        idempotency_key="shared-key",
    )
    id_owner = _make_sqlite_evidence(
        scope=scope,
        payload="ID owner",
        idempotency_key="id-owner-key",
    )
    challenger = _make_sqlite_evidence(
        scope=scope,
        payload="conflicts with two different rows",
        idempotency_key="shared-key",
    )
    digest_by_canonical = {
        idempotency_owner.canonical_bytes(): "a" * 64,
        id_owner.canonical_bytes(): "b" * 64,
        challenger.canonical_bytes(): "b" * 64,
    }

    class StableDigest:
        def __init__(self, digest: str) -> None:
            self._digest = digest

        def hexdigest(self) -> str:
            return self._digest

    def collision_sha256(canonical: bytes) -> StableDigest:
        return StableDigest(digest_by_canonical[canonical])

    monkeypatch.setattr(sqlite_store_module, "sha256", collision_sha256)
    first = store.append(idempotency_owner)
    second = store.append(id_owner)
    assert first.evidence_id != second.evidence_id

    with pytest.raises(EvidenceConflictError) as raised:
        store.append(challenger)

    assert str(raised.value) == (
        "scoped idempotency key already refers to different evidence"
    )
    assert store.list(scope) == tuple(
        sorted((first, second), key=sqlite_store_module._evidence_sort_key)
    )


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
def test_sqlite_evidence_collision_is_scoped_atomic_and_loser_key_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    database_path = tmp_path / f"{collision_kind}.sqlite3"
    store = SQLiteMemoryStore(database_path)
    first_scope = MemoryScope("tenant-1", "assistant-memory", "collision-user-1")
    second_scope = MemoryScope("tenant-1", "assistant-memory", "collision-user-2")
    first_event = _make_sqlite_evidence(
        scope=first_scope,
        payload="first",
        idempotency_key="first-key",
    )
    loser_event = _make_sqlite_evidence(
        scope=first_scope,
        payload="loser",
        idempotency_key="loser-key",
    )
    replacement_event = _make_sqlite_evidence(
        scope=first_scope,
        payload="replacement",
        idempotency_key="loser-key",
    )
    assert replacement_event.canonical_bytes() != loser_event.canonical_bytes()
    cross_scope_event = _make_sqlite_evidence(
        scope=second_scope,
        payload="cross scope",
        idempotency_key="first-key",
    )
    shared_prefix = "a" * 24
    first_digest = (
        "a" * 64 if collision_kind == "full-hash" else shared_prefix + "b" * 40
    )
    loser_digest = (
        first_digest if collision_kind == "full-hash" else shared_prefix + "c" * 40
    )
    digest_by_canonical = {
        first_event.canonical_bytes(): first_digest,
        loser_event.canonical_bytes(): loser_digest,
        replacement_event.canonical_bytes(): "d" * 64,
        cross_scope_event.canonical_bytes(): first_digest,
    }

    class StableDigest:
        def __init__(self, digest: str) -> None:
            self._digest = digest

        def hexdigest(self) -> str:
            return self._digest

    def collision_sha256(canonical: bytes) -> StableDigest:
        return StableDigest(digest_by_canonical[canonical])

    monkeypatch.setattr(sqlite_store_module, "sha256", collision_sha256)
    original = store.append(first_event)

    with pytest.raises(EvidenceConflictError) as raised:
        store.append(loser_event)

    assert str(raised.value) == f"evidence ID collision for {original.evidence_id!r}"
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT idempotency_key FROM memory_evidence"
        ).fetchall() == [("first-key",)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()

    cross_scope = store.append(cross_scope_event)
    assert cross_scope.evidence_id == original.evidence_id
    recovered_loser = store.append(replacement_event)

    assert recovered_loser.evidence_id == f"evd_{'d' * 24}"
    assert recovered_loser.event == replacement_event
    with pytest.raises(EvidenceConflictError) as retry_error:
        store.append(loser_event)
    assert str(retry_error.value) == (
        "scoped idempotency key already refers to different evidence"
    )
    assert store.get(first_scope, original.evidence_id) == original
    assert store.get(second_scope, cross_scope.evidence_id) == cross_scope
    assert {record.event.idempotency_key for record in store.list(first_scope)} == {
        "first-key",
        "loser-key",
    }
    assert store.list(second_scope) == (cross_scope,)
    assert digest_by_canonical[loser_event.canonical_bytes()] == loser_digest


def test_sqlite_evidence_failed_scope_insert_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "rollback-user")
    event = _make_sqlite_evidence(scope=scope, idempotency_key="rollback-key")
    injected = sqlite3.IntegrityError("injected after real scope insert")
    plan = _SQLiteFailurePlan(
        after_statement="INSERT INTO memory_scopes",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        store.append(event)

    assert type(raised.value) is MemoryPersistenceError
    assert not isinstance(raised.value, EvidenceConflictError)
    assert raised.value.__cause__ is injected
    assert any(
        entry.startswith("executed:INSERT INTO MEMORY_SCOPES") for entry in plan.events
    )
    assert "executed:ROLLBACK" in plan.events
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_scopes WHERE subject_id = ?",
            (scope.subject_id,),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()

    monkeypatch.undo()
    recovered = store.append(event)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_scopes WHERE subject_id = ?",
            (scope.subject_id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone() == (1,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
    assert store.get(scope, recovered.evidence_id) == recovered


def test_sqlite_get_validates_scope_snapshot_before_evidence_id_absence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "moved-evidence-id.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    moved_evidence_id = f"evd_{'0' * 24}"
    assert moved_evidence_id != record.evidence_id

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        ingest_order = connection.execute(
            "SELECT ingest_order FROM memory_evidence_ingest_orders "
            "WHERE evidence_id = ?",
            (record.evidence_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE memory_evidence SET evidence_id = ? WHERE evidence_id = ?",
            (moved_evidence_id, record.evidence_id),
        )
        connection.execute(
            "UPDATE memory_evidence_ingest_orders SET evidence_id = ?, "
            "binding_hash = ? WHERE evidence_id = ?",
            (
                moved_evidence_id,
                sqlite_backend._evidence_ingest_binding_hash(
                    scope=event.scope,
                    evidence_id=moved_evidence_id,
                    ingest_order=ingest_order,
                ),
                record.evidence_id,
            ),
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.get(event.scope, record.evidence_id)

    assert str(raised.value) == "stored evidence row failed integrity validation"
    assert str(raised.value.__cause__) == (
        "evidence ID disagrees with its content hash"
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT evidence_id FROM memory_evidence"
        ).fetchall() == [(moved_evidence_id,)]
    finally:
        connection.close()


def test_sqlite_append_validates_scope_snapshot_before_idempotency_absence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "moved-idempotency-key.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    moved_idempotency_key = "moved-evidence-request"

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE memory_evidence SET idempotency_key = ? WHERE evidence_id = ?",
            (moved_idempotency_key, record.evidence_id),
        )
    finally:
        connection.close()

    conflicting_event = _make_sqlite_evidence(payload="different payload")
    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.append(conflicting_event)

    assert str(raised.value) == "stored evidence row failed integrity validation"
    assert str(raised.value.__cause__) == (
        "canonical evidence bytes disagree with projections"
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT evidence_id, idempotency_key FROM memory_evidence"
        ).fetchall() == [(record.evidence_id, moved_idempotency_key)]
    finally:
        connection.close()


def test_sqlite_operations_validate_global_evidence_before_scope_absence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "moved-evidence-scope.sqlite3")
    store = SQLiteMemoryStore(database_path)
    source_scope = MemoryScope("tenant-1", "assistant-memory", "source-user")
    destination_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "destination-user",
    )
    source_event = _make_sqlite_evidence(
        scope=source_scope,
        idempotency_key="source-request",
    )
    destination_event = _make_sqlite_evidence(
        scope=destination_scope,
        payload="destination evidence",
        idempotency_key="destination-request",
    )
    source_record = store.append(source_event)
    store.append(destination_event)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        destination_scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (
                destination_scope.tenant_id,
                destination_scope.namespace,
                destination_scope.subject_id,
            ),
        ).fetchone()[0]
        connection.execute(
            "UPDATE memory_evidence SET scope_id = ? WHERE evidence_id = ?",
            (destination_scope_id, source_record.evidence_id),
        )
        ingest_order = connection.execute(
            "SELECT ingest_order FROM memory_evidence_ingest_orders "
            "WHERE evidence_id = ?",
            (source_record.evidence_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE memory_evidence_ingest_orders SET scope_id = ?, "
            "binding_hash = ? WHERE evidence_id = ?",
            (
                destination_scope_id,
                sqlite_backend._evidence_ingest_binding_hash(
                    scope=destination_scope,
                    evidence_id=source_record.evidence_id,
                    ingest_order=ingest_order,
                ),
                source_record.evidence_id,
            ),
        )
    finally:
        connection.close()

    operation_errors: dict[str, Exception | None] = {}
    for operation in ("get", "list", "retry"):
        try:
            if operation == "get":
                store.get(source_scope, source_record.evidence_id)
            elif operation == "list":
                store.list(source_scope)
            else:
                store.append(source_event)
        except Exception as error:
            operation_errors[operation] = error
        else:
            operation_errors[operation] = None

    assert {
        operation: None if error is None else type(error).__name__
        for operation, error in operation_errors.items()
    } == {
        "get": "MemoryPersistenceCorruptionError",
        "list": "MemoryPersistenceCorruptionError",
        "retry": "MemoryPersistenceCorruptionError",
    }
    for error in operation_errors.values():
        assert type(error) is MemoryPersistenceCorruptionError
        assert str(error.__cause__) == (
            "canonical evidence bytes disagree with projections"
        )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_sqlite_operations_validate_evidence_before_rewritten_scope_absence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "rewritten-scope-address.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    rewritten_subject_id = "rewritten-user"

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id, ingest_order = connection.execute(
            "SELECT scope_id, ingest_order FROM memory_evidence_ingest_orders "
            "WHERE evidence_id = ?",
            (record.evidence_id,),
        ).fetchone()
        connection.execute(
            "UPDATE memory_scopes SET subject_id = ? WHERE subject_id = ?",
            (rewritten_subject_id, event.scope.subject_id),
        )
        rewritten_scope = MemoryScope(
            event.scope.tenant_id,
            event.scope.namespace,
            rewritten_subject_id,
        )
        connection.execute(
            "UPDATE memory_evidence_ingest_orders SET binding_hash = ? "
            "WHERE scope_id = ? AND evidence_id = ?",
            (
                sqlite_backend._evidence_ingest_binding_hash(
                    scope=rewritten_scope,
                    evidence_id=record.evidence_id,
                    ingest_order=ingest_order,
                ),
                scope_id,
                record.evidence_id,
            ),
        )
    finally:
        connection.close()

    operation_errors: dict[str, Exception | None] = {}
    for operation in ("get", "list", "retry"):
        try:
            if operation == "get":
                store.get(event.scope, record.evidence_id)
            elif operation == "list":
                store.list(event.scope)
            else:
                store.append(event)
        except Exception as error:
            operation_errors[operation] = error
        else:
            operation_errors[operation] = None

    assert {
        operation: None if error is None else type(error).__name__
        for operation, error in operation_errors.items()
    } == {
        "get": "MemoryPersistenceCorruptionError",
        "list": "MemoryPersistenceCorruptionError",
        "retry": "MemoryPersistenceCorruptionError",
    }
    for error in operation_errors.values():
        assert type(error) is MemoryPersistenceCorruptionError
        assert str(error.__cause__) == (
            "canonical evidence bytes disagree with projections"
        )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT subject_id FROM memory_scopes"
        ).fetchall() == [(rewritten_subject_id,)]
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone() == (1,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_sqlite_list_rejects_duplicate_physical_evidence_addresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "duplicate-address.sqlite3")
    event = _make_sqlite_evidence()
    record = store.append(event)

    class DuplicateAddressCursor:
        def __init__(self) -> None:
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            _parameters: object = (),
        ) -> DuplicateAddressCursor:
            self._last_sql = _normalize_sql(sql)
            return self

        def fetchone(self) -> tuple[object, ...] | None:
            if self._last_sql.startswith("SELECT SCOPE_ID FROM MEMORY_SCOPES"):
                return (1,)
            return None

        def fetchall(self) -> list[tuple[object, ...]]:
            if self._last_sql.startswith(
                "SELECT SCOPE_ID, TENANT_ID, NAMESPACE, SUBJECT_ID FROM MEMORY_SCOPES"
            ):
                return [
                    (
                        1,
                        event.scope.tenant_id,
                        event.scope.namespace,
                        event.scope.subject_id,
                    )
                ]
            if self._last_sql.startswith(
                "SELECT SCOPE_ID, EVIDENCE_ID FROM MEMORY_EVIDENCE"
            ):
                address = (1, record.evidence_id)
                return [address, address]
            if self._last_sql.startswith(
                "SELECT EVIDENCE_ID FROM MEMORY_EVIDENCE WHERE SCOPE_ID = ?"
            ):
                address = (record.evidence_id,)
                return [address, address]
            raise AssertionError(f"unexpected fetchall SQL: {self._last_sql}")

    class DuplicateAddressTransaction:
        def __enter__(self) -> DuplicateAddressCursor:
            return DuplicateAddressCursor()

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        sqlite_store_module,
        "_read_transaction",
        lambda _path: DuplicateAddressTransaction(),
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_load_evidence",
        lambda *_args: record,
    )

    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="evidence address appears multiple times",
    ):
        store.list(event.scope)


@pytest.mark.parametrize(
    "column",
    [
        "evidence_id",
        "canonical",
        "content_hash",
        "created_at",
        "storage_hash",
        "session_id",
        "run_id",
        "sequence_no",
        "kind",
        "payload",
        "observed_at",
        "idempotency_key",
    ],
)
def test_sqlite_evidence_loader_rejects_each_semantic_column_drift(
    tmp_path: Path,
    column: str,
) -> None:
    database_path = str(tmp_path / f"semantic-{column}.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    changed_id = f"evd_{'0' * 24}"
    canonical_variant = b" \n" + event.canonical_bytes()
    assert canonical_variant != event.canonical_bytes()
    assert json.loads(canonical_variant) == json.loads(event.canonical_bytes())
    mutations: dict[str, tuple[str, object]] = {
        "evidence_id": (
            "UPDATE memory_evidence SET evidence_id = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            changed_id,
        ),
        "canonical": (
            "UPDATE memory_evidence SET canonical = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            sqlite3.Binary(canonical_variant),
        ),
        "content_hash": (
            "UPDATE memory_evidence SET content_hash = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "0" * 64,
        ),
        "created_at": (
            "UPDATE memory_evidence SET created_at = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            record.created_at.isoformat().replace("+00:00", "Z"),
        ),
        "storage_hash": (
            "UPDATE memory_evidence SET storage_hash = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "0" * 64,
        ),
        "session_id": (
            "UPDATE memory_evidence SET session_id = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "changed-session",
        ),
        "run_id": (
            "UPDATE memory_evidence SET run_id = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "changed-run",
        ),
        "sequence_no": (
            "UPDATE memory_evidence SET sequence_no = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            event.sequence_no + 1,
        ),
        "kind": (
            "UPDATE memory_evidence SET kind = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            EvidenceKind.FEEDBACK.value,
        ),
        "payload": (
            "UPDATE memory_evidence SET payload = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "changed payload",
        ),
        "observed_at": (
            "UPDATE memory_evidence SET observed_at = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            event.observed_at.astimezone(timezone(timedelta(hours=8))).isoformat(),
        ),
        "idempotency_key": (
            "UPDATE memory_evidence SET idempotency_key = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            "changed-key",
        ),
    }
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (event.scope.tenant_id, event.scope.namespace, event.scope.subject_id),
        ).fetchone()[0]
        sql, changed_value = mutations[column]
        connection.execute(
            sql,
            (changed_value, scope_id, event.idempotency_key),
        )
        if column == "evidence_id":
            ingest_order = connection.execute(
                "SELECT ingest_order FROM memory_evidence_ingest_orders "
                "WHERE evidence_id = ?",
                (record.evidence_id,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE memory_evidence_ingest_orders SET evidence_id = ?, "
                "binding_hash = ? WHERE evidence_id = ?",
                (
                    changed_id,
                    sqlite_backend._evidence_ingest_binding_hash(
                        scope=event.scope,
                        evidence_id=changed_id,
                        ingest_order=ingest_order,
                    ),
                    record.evidence_id,
                ),
            )
    finally:
        connection.close()

    filters: dict[str, str] = {}
    if column == "session_id":
        filters["session_id"] = event.session_id
    elif column == "run_id":
        filters["run_id"] = event.run_id
    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.list(event.scope, **filters)

    assert str(raised.value) == "stored evidence row failed integrity validation"
    if column == "created_at":
        assert "created_at is not exact UTC isoformat text" in str(
            raised.value.__cause__
        )
    if column == "observed_at":
        assert "observed_at is not exact UTC isoformat text" in str(
            raised.value.__cause__
        )


@pytest.mark.parametrize(
    ("column_index", "wrong_value"),
    [
        (0, 7),
        (1, "not-a-blob"),
        (2, b"not-text"),
        (3, b"not-text"),
        (4, b"not-text"),
        (5, b"not-text"),
        (6, b"not-text"),
        (7, "not-an-integer"),
        (8, b"not-text"),
        (9, b"not-text"),
        (10, b"not-text"),
        (11, b"not-text"),
    ],
)
def test_sqlite_evidence_loader_rejects_wrong_storage_class_for_each_column(
    tmp_path: Path,
    column_index: int,
    wrong_value: object,
) -> None:
    database_path = str(tmp_path / f"storage-{column_index}.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (event.scope.tenant_id, event.scope.namespace, event.scope.subject_id),
        ).fetchone()[0]
        real_row = connection.execute(
            sqlite_store_module._EVIDENCE_SELECT,
            (scope_id, record.evidence_id),
        ).fetchone()
    finally:
        connection.close()
    assert real_row is not None
    changed_row = list(real_row)
    changed_row[column_index] = wrong_value

    class StaticRowCursor:
        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticRowCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return tuple(changed_row)

    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="stored evidence row failed integrity validation",
    ):
        sqlite_store_module._load_evidence(
            StaticRowCursor(),  # type: ignore[arg-type]
            event.scope,
            scope_id,
            record.evidence_id,
        )


@pytest.mark.parametrize("operation", ["get", "list", "retry"])
def test_sqlite_invalid_utf8_text_is_corruption_for_all_evidence_reads(
    tmp_path: Path,
    operation: str,
) -> None:
    database_path = str(tmp_path / "invalid-utf8.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE memory_evidence SET payload = CAST(X'80' AS TEXT) "
            "WHERE evidence_id = ?",
            (record.evidence_id,),
        )
        storage_class = connection.execute(
            "SELECT typeof(payload) FROM memory_evidence WHERE evidence_id = ?",
            (record.evidence_id,),
        ).fetchone()
    finally:
        connection.close()
    assert storage_class == ("text",)

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        if operation == "get":
            store.get(event.scope, record.evidence_id)
        elif operation == "list":
            store.list(event.scope)
        else:
            store.append(event)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


def test_sqlite_evidence_loader_binds_coherent_row_to_requested_id(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "requested-id.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (event.scope.tenant_id, event.scope.namespace, event.scope.subject_id),
        ).fetchone()[0]
        coherent_row = connection.execute(
            sqlite_store_module._EVIDENCE_SELECT,
            (scope_id, record.evidence_id),
        ).fetchone()
    finally:
        connection.close()
    assert coherent_row is not None
    different_requested_id = f"evd_{'f' * 24}"
    assert different_requested_id != record.evidence_id

    class CoherentRowCursor:
        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> CoherentRowCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return coherent_row

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        sqlite_store_module._load_evidence(
            CoherentRowCursor(),  # type: ignore[arg-type]
            event.scope,
            scope_id,
            different_requested_id,
        )

    assert str(raised.value) == "stored evidence row failed integrity validation"
    assert str(raised.value.__cause__) == (
        "loaded evidence ID differs from requested ID"
    )


@pytest.mark.parametrize("scope_column", ["tenant_id", "namespace", "subject_id"])
@pytest.mark.parametrize("operation", ["get", "list", "retry"])
def test_sqlite_scope_lookup_rejects_invalid_utf8_identity_before_absence(
    tmp_path: Path,
    scope_column: str,
    operation: str,
) -> None:
    database_path = str(tmp_path / "invalid-scope-utf8.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    mutation_by_column = {
        "tenant_id": ("UPDATE memory_scopes SET tenant_id = CAST(X'80' AS TEXT)"),
        "namespace": ("UPDATE memory_scopes SET namespace = CAST(X'80' AS TEXT)"),
        "subject_id": ("UPDATE memory_scopes SET subject_id = CAST(X'80' AS TEXT)"),
    }
    inspection_by_column = {
        "tenant_id": "SELECT typeof(tenant_id), hex(tenant_id) FROM memory_scopes",
        "namespace": "SELECT typeof(namespace), hex(namespace) FROM memory_scopes",
        "subject_id": "SELECT typeof(subject_id), hex(subject_id) FROM memory_scopes",
    }
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(mutation_by_column[scope_column])
        identity_state = connection.execute(
            inspection_by_column[scope_column]
        ).fetchone()
    finally:
        connection.close()
    assert identity_state == ("text", "80")

    operation_error: Exception | None = None
    try:
        if operation == "get":
            store.get(event.scope, record.evidence_id)
        elif operation == "list":
            store.list(event.scope)
        else:
            store.append(event)
    except Exception as error:
        operation_error = error

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_count = connection.execute(
            "SELECT COUNT(*) FROM memory_scopes"
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone()
        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    finally:
        connection.close()

    assert type(operation_error) is MemoryPersistenceCorruptionError, (
        f"operation error was {type(operation_error).__name__}; "
        f"scope rows={scope_count}; evidence rows={evidence_count}"
    )
    assert isinstance(operation_error.__cause__, UnicodeDecodeError)
    assert scope_count == (1,)
    assert evidence_count == (1,)
    assert foreign_key_violations == []


@pytest.mark.parametrize("scope_column", ["tenant_id", "namespace", "subject_id"])
@pytest.mark.parametrize("invalid_value", ["", " \t"])
def test_sqlite_scope_lookup_rejects_stored_identity_invalid_to_public_contract(
    tmp_path: Path,
    scope_column: str,
    invalid_value: str,
) -> None:
    database_path = str(tmp_path / "invalid-public-scope.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    record = store.append(event)
    mutation_by_column = {
        "tenant_id": "UPDATE memory_scopes SET tenant_id = ?",
        "namespace": "UPDATE memory_scopes SET namespace = ?",
        "subject_id": "UPDATE memory_scopes SET subject_id = ?",
    }
    inspection_by_column = {
        "tenant_id": "SELECT typeof(tenant_id), tenant_id FROM memory_scopes",
        "namespace": "SELECT typeof(namespace), namespace FROM memory_scopes",
        "subject_id": "SELECT typeof(subject_id), subject_id FROM memory_scopes",
    }
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(mutation_by_column[scope_column], (invalid_value,))
        identity_state = connection.execute(
            inspection_by_column[scope_column]
        ).fetchone()
    finally:
        connection.close()
    assert identity_state == ("text", invalid_value)

    operation_errors: dict[str, Exception | None] = {}
    for operation in ("get", "list", "retry"):
        try:
            if operation == "get":
                store.get(event.scope, record.evidence_id)
            elif operation == "list":
                store.list(event.scope)
            else:
                store.append(event)
        except Exception as error:
            operation_errors[operation] = error
        else:
            operation_errors[operation] = None

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_count = connection.execute(
            "SELECT COUNT(*) FROM memory_scopes"
        ).fetchone()
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone()
        foreign_key_violations = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    finally:
        connection.close()

    error_types = {
        operation: None if error is None else type(error).__name__
        for operation, error in operation_errors.items()
    }
    assert error_types == {
        "get": "MemoryPersistenceCorruptionError",
        "list": "MemoryPersistenceCorruptionError",
        "retry": "MemoryPersistenceCorruptionError",
    }, (
        f"operation errors={error_types}; scope rows={scope_count}; "
        f"evidence rows={evidence_count}"
    )
    for error in operation_errors.values():
        assert type(error) is MemoryPersistenceCorruptionError
        assert isinstance(error.__cause__, ValueError)
    assert scope_count == (1,)
    assert evidence_count == (1,)
    assert foreign_key_violations == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (((1, "tenant-1", "assistant-memory"),), "exactly four values"),
        (((1, b"tenant-1", "assistant-memory", "scope-user"),), "text"),
        (((1, "tenant-1", b"assistant-memory", "scope-user"),), "text"),
        (((1, "tenant-1", "assistant-memory", b"scope-user"),), "text"),
        (
            (
                (1, "tenant-1", "assistant-memory", "scope-user"),
                (2, "tenant-1", "assistant-memory", "scope-user"),
            ),
            "multiple rows",
        ),
        (
            (
                (1, "tenant-1", "assistant-memory", "scope-user"),
                (2, "other", "assistant-memory", b"damaged"),
            ),
            "text",
        ),
    ],
)
def test_sqlite_scope_lookup_validates_every_identity_row(
    rows: tuple[tuple[object, ...], ...],
    message: str,
) -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "scope-user")

    class ScopeCursor:
        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> ScopeCursor:
            return self

        def fetchone(self) -> tuple[object, ...] | None:
            return None if not rows else (rows[0][0],)

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(rows)

    with pytest.raises(MemoryPersistenceCorruptionError, match=message):
        sqlite_store_module._find_scope_id(
            ScopeCursor(),  # type: ignore[arg-type]
            scope,
        )


def test_sqlite_scope_lookup_matches_exact_identity_or_returns_none() -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "scope-user")

    class ScopeCursor:
        def __init__(self, rows: tuple[tuple[object, ...], ...]) -> None:
            self._rows = rows

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> ScopeCursor:
            return self

        def fetchone(self) -> tuple[object, ...] | None:
            requested_identity = (
                scope.tenant_id,
                scope.namespace,
                scope.subject_id,
            )
            for row in self._rows:
                if len(row) == 4 and row[1:] == requested_identity:
                    return (row[0],)
            return None

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._rows)

    target_rows = (
        (1, "other", "assistant-memory", "scope-user"),
        (2, "tenant-1", "assistant-memory", "scope-user"),
    )
    assert (
        sqlite_store_module._find_scope_id(
            ScopeCursor(target_rows),  # type: ignore[arg-type]
            scope,
        )
        == 2
    )
    assert (
        sqlite_store_module._find_scope_id(
            ScopeCursor(target_rows[:1]),  # type: ignore[arg-type]
            scope,
        )
        is None
    )


def test_sqlite_evidence_scope_lookup_requires_positive_signed_64_bit_id() -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "scope-user")

    class ScopeCursor:
        def __init__(self, stored_scope_id: object) -> None:
            self._row = (
                stored_scope_id,
                scope.tenant_id,
                scope.namespace,
                scope.subject_id,
            )

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> ScopeCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return (self._row[0],)

        def fetchall(self) -> list[tuple[object, ...]]:
            return [self._row]

    for invalid_scope_id in ("7", True, 0, -1, 2**63):
        with pytest.raises(
            MemoryPersistenceCorruptionError,
            match="positive signed 64-bit",
        ):
            sqlite_store_module._find_scope_id(
                ScopeCursor(invalid_scope_id),  # type: ignore[arg-type]
                scope,
            )
    for valid_scope_id in (1, 2**63 - 1):
        assert (
            sqlite_store_module._find_scope_id(
                ScopeCursor(valid_scope_id),  # type: ignore[arg-type]
                scope,
            )
            == valid_scope_id
        )


@pytest.mark.parametrize(
    ("lastrowid", "persisted_scope_id", "message"),
    [
        ("7", None, "did not return"),
        (True, None, "did not return"),
        (0, 0, "did not return"),
        (-1, -1, "did not return"),
        (2**63, 2**63, "did not return"),
        (7, 8, "did not round-trip"),
    ],
)
def test_sqlite_evidence_scope_insert_validates_lastrowid_and_requery(
    lastrowid: object,
    persisted_scope_id: int | None,
    message: str,
) -> None:
    identity = ("tenant-1", "assistant-memory", "scope-user")
    rows: list[list[tuple[object, ...]]] = [[]]
    if persisted_scope_id is not None:
        rows.append([(persisted_scope_id, *identity)])

    class ScopeCursor:
        def __init__(self) -> None:
            self.lastrowid = lastrowid

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> ScopeCursor:
            return self

        def fetchone(self) -> tuple[object, ...] | None:
            batch = rows.pop(0)
            return None if not batch else (batch[0][0],)

        def fetchall(self) -> list[tuple[object, ...]]:
            return rows.pop(0)

    with pytest.raises(MemoryPersistenceCorruptionError, match=message):
        sqlite_store_module._ensure_scope_id(
            ScopeCursor(),  # type: ignore[arg-type]
            MemoryScope("tenant-1", "assistant-memory", "scope-user"),
        )


def test_sqlite_evidence_append_validates_inserted_row_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "append-readback.sqlite3")
    store = SQLiteMemoryStore(database_path)
    event = _make_sqlite_evidence()
    real_connect = sqlite_backend._connect

    class TamperingCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> TamperingCursor:
            self._real_cursor.execute(sql, parameters)
            if _normalize_sql(sql).startswith("INSERT INTO MEMORY_EVIDENCE"):
                assert isinstance(parameters, tuple)
                self._real_cursor.execute(
                    "UPDATE memory_evidence SET payload = ? "
                    "WHERE scope_id = ? AND idempotency_key = ?",
                    (
                        "tampered after insert",
                        parameters[0],
                        event.idempotency_key,
                    ),
                )
            return self

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class TamperingConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> TamperingCursor:
            return TamperingCursor(self._real_connection.cursor(*args, **kwargs))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> TamperingConnection:
        return TamperingConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="stored evidence row failed integrity validation",
    ):
        store.append(event)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("SELECT COUNT(*) FROM memory_scopes").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence"
        ).fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (b"memory.sqlite3", TypeError, "string-valued"),
        ("", ValueError, "must not be blank"),
        (" \t\n", ValueError, "must not be blank"),
        ("bad\x00path", ValueError, "must not contain NUL"),
        ("\ud800", ValueError, "must be valid UTF-8"),
        (":memory:", ValueError, "durable database file"),
    ],
)
def test_database_path_rejects_non_durable_values(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        sqlite_backend._snapshot_database_path(value)  # type: ignore[arg-type]


def test_relative_path_is_bound_to_constructor_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    path = sqlite_backend._snapshot_database_path("memory.sqlite3")
    monkeypatch.chdir(second)
    sqlite_backend._initialize_database(path)
    assert (first / "memory.sqlite3").is_file()
    assert not (second / "memory.sqlite3").exists()


def test_runtime_floor_is_checked_before_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 7, 16))
    with pytest.raises(MemoryPersistenceSchemaError, match="3[.]7[.]17"):
        sqlite_backend._initialize_database(str(tmp_path / "memory.sqlite3"))
    assert not (tmp_path / "memory.sqlite3").exists()


def test_new_database_has_exact_v2_header_catalog_and_metadata(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(path)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA application_id").fetchone() == (
            sqlite_backend._APPLICATION_ID,
        )
        assert connection.execute("PRAGMA user_version").fetchone() == (
            sqlite_backend._SCHEMA_VERSION,
        )
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            )
        }
        assert len(sqlite_backend._SCHEMA_V1_DDL) == 11
        assert len(sqlite_backend._SCHEMA_V2_ADDITIONS) == 4
        assert len(sqlite_backend._SCHEMA_DDL) == 15
        assert names == (
            sqlite_backend._REQUIRED_TABLES | sqlite_backend._REQUIRED_INDEXES
        )
        assert connection.execute(
            "SELECT schema_spec_hash, schema_catalog_hash "
            "FROM memory_schema_metadata WHERE singleton = 1"
        ).fetchone() == (
            sqlite_backend._SCHEMA_SPEC_HASH,
            sqlite_backend._catalog_hash(connection.cursor()),
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_ingest_orders"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_snapshots"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_evidence_snapshot_aliases"
        ).fetchone() == (0,)
    finally:
        connection.close()


def test_schema_spec_hash_covers_ordered_exact_ddl() -> None:
    encoded = json.dumps(
        sqlite_backend._SCHEMA_DDL,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == sqlite_backend._SCHEMA_SPEC_HASH

    reordered = list(sqlite_backend._SCHEMA_DDL)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    reordered_hash = hashlib.sha256(
        json.dumps(
            reordered,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert reordered_hash != sqlite_backend._SCHEMA_SPEC_HASH


def test_catalog_hash_excludes_autoindexes_and_detects_extra_objects(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(path)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        cursor = connection.cursor()
        before = sqlite_backend._catalog_hash(cursor)
        rows = sqlite_backend._catalog_rows(cursor)
        assert not any(row[1].startswith("sqlite_") for row in rows)
        connection.execute(
            "CREATE VIEW unexpected_memory_view AS SELECT scope_id FROM memory_scopes"
        )
        assert sqlite_backend._catalog_hash(cursor) != before
    finally:
        connection.close()


def test_transaction_post_statement_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected post-statement failure")
    plan = _SQLiteFailurePlan(
        after_statement="INSERT INTO memory_scopes",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        with sqlite_backend._write_transaction(database_path) as cursor:
            _insert_test_scope(cursor)

    assert raised.value.__cause__ is injected
    assert _read_test_scopes(database_path) == []
    assert any(
        event.startswith("executed:INSERT INTO MEMORY_SCOPES") for event in plan.events
    )
    assert "executed:ROLLBACK" in plan.events
    assert "executed:CLOSE" in plan.events


def test_write_begin_ack_failure_rolls_back_closes_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected BEGIN acknowledgement failure")
    plan = _SQLiteFailurePlan(
        after_statement="BEGIN IMMEDIATE",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        with sqlite_backend._write_transaction(database_path):
            raise AssertionError("transaction body must not run")

    assert raised.value.__cause__ is injected
    assert "executed:BEGIN IMMEDIATE" in plan.events
    assert "executed:ROLLBACK" in plan.events
    assert "executed:CLOSE" in plan.events

    with sqlite_backend._write_transaction(database_path) as cursor:
        _insert_test_scope(cursor)
    assert _read_test_scopes(database_path) == [
        (1, "tenant-1", "assistant-memory", "user-1")
    ]


def test_initializer_begin_ack_failure_rolls_back_closes_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    injected = sqlite3.OperationalError("injected BEGIN acknowledgement failure")
    plan = _SQLiteFailurePlan(
        after_statement="BEGIN EXCLUSIVE",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        sqlite_backend._initialize_database(database_path)

    assert raised.value.__cause__ is injected
    assert "executed:BEGIN EXCLUSIVE" in plan.events
    assert "executed:ROLLBACK" in plan.events
    assert "executed:CLOSE" in plan.events

    sqlite_backend._initialize_database(database_path)


def test_initializer_real_commit_followed_by_ack_failure_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    injected = sqlite3.OperationalError("injected acknowledgement failure")
    plan = _SQLiteFailurePlan(after_commit_error=injected)
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        sqlite_backend._initialize_database(database_path)

    assert raised.value.__cause__ is injected
    assert "executed:COMMIT" in plan.events
    assert "fail-after:COMMIT" in plan.events
    assert "attempt:ROLLBACK" in plan.events
    assert "executed:ROLLBACK" not in plan.events
    assert "executed:CLOSE" in plan.events

    sqlite_backend._initialize_database(database_path)


def test_transaction_pre_commit_failure_rolls_back_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected failure before real commit")
    plan = _SQLiteFailurePlan(before_commit_error=injected)
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        with sqlite_backend._write_transaction(database_path) as cursor:
            _insert_test_scope(cursor)

    assert raised.value.__cause__ is injected
    assert "fail-before:COMMIT" in plan.events
    assert "executed:COMMIT" not in plan.events
    assert "executed:ROLLBACK" in plan.events
    assert _read_test_scopes(database_path) == []

    with sqlite_backend._write_transaction(database_path) as cursor:
        _insert_test_scope(cursor)

    assert _read_test_scopes(database_path) == [
        (1, "tenant-1", "assistant-memory", "user-1")
    ]


def test_transaction_real_commit_followed_by_ack_failure_preserves_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected acknowledgement failure")
    plan = _SQLiteFailurePlan(after_commit_error=injected)
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        with sqlite_backend._write_transaction(database_path) as cursor:
            _insert_test_scope(cursor)

    assert raised.value.__cause__ is injected
    assert "executed:COMMIT" in plan.events
    assert "fail-after:COMMIT" in plan.events
    assert "attempt:ROLLBACK" in plan.events
    assert "executed:ROLLBACK" not in plan.events
    assert "executed:CLOSE" in plan.events
    assert _read_test_scopes(database_path) == [
        (1, "tenant-1", "assistant-memory", "user-1")
    ]

    with sqlite_backend._write_transaction(database_path) as cursor:
        existing = cursor.execute(
            "SELECT scope_id FROM memory_scopes "
            "WHERE tenant_id = ? AND namespace = ? AND subject_id = ?",
            ("tenant-1", "assistant-memory", "user-1"),
        ).fetchone()
        if existing is None:
            _insert_test_scope(cursor)
        else:
            assert existing == (1,)

    assert _read_test_scopes(database_path) == [
        (1, "tenant-1", "assistant-memory", "user-1")
    ]


def test_transaction_cleanup_failures_do_not_mask_body_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    original = RuntimeError("body failed")
    plan = _SQLiteFailurePlan(
        rollback_error=sqlite3.OperationalError(
            "cannot rollback - no transaction is active"
        ),
        close_error=sqlite3.OperationalError("injected close failure"),
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(RuntimeError) as raised:
        with sqlite_backend._write_transaction(database_path) as cursor:
            _insert_test_scope(cursor)
            raise original

    assert raised.value is original
    assert "fail-before:ROLLBACK" in plan.events
    assert "executed:CLOSE" in plan.events
    assert "fail-after:CLOSE" in plan.events
    assert _read_test_scopes(database_path) == []
    assert any("ROLLBACK cleanup failed" in note for note in original.__notes__)
    assert any("close cleanup failed" in note for note in original.__notes__)


def test_transaction_close_failure_after_success_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected close failure")
    plan = _SQLiteFailurePlan(close_error=injected)
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        with sqlite_backend._write_transaction(database_path) as cursor:
            _insert_test_scope(cursor)

    assert raised.value.__cause__ is injected
    assert "executed:COMMIT" in plan.events
    assert "executed:CLOSE" in plan.events
    assert _read_test_scopes(database_path) == [
        (1, "tenant-1", "assistant-memory", "user-1")
    ]


def test_write_transaction_uses_begin_immediate(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    statements: list[str] = []
    real_connect = sqlite_backend._connect

    def connect(path: str) -> sqlite3.Connection:
        connection = real_connect(path)
        connection.set_trace_callback(statements.append)
        return connection

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sqlite_backend, "_connect", connect)
        with sqlite_backend._write_transaction(database_path):
            pass

    normalized = [_normalize_sql(statement) for statement in statements]
    assert normalized[0] == "BEGIN IMMEDIATE"
    assert normalized[-1] == "COMMIT"


def test_read_transaction_locks_catalog_before_second_journal_check(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    statements: list[str] = []
    real_connect = sqlite_backend._connect

    def connect(path: str) -> sqlite3.Connection:
        connection = real_connect(path)
        connection.set_trace_callback(statements.append)
        return connection

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sqlite_backend, "_connect", connect)
        with sqlite_backend._read_transaction(database_path):
            pass

    normalized = [_normalize_sql(statement) for statement in statements]
    begin = normalized.index("BEGIN")
    catalog_lock = normalized.index("SELECT NAME FROM MAIN.SQLITE_MASTER LIMIT 1")
    journal = normalized.index("PRAGMA JOURNAL_MODE")
    schema_validation = normalized.index("PRAGMA APPLICATION_ID")
    assert begin < catalog_lock < journal < schema_validation
    assert normalized[-1] == "COMMIT"


def test_initializer_uses_begin_exclusive_without_executescript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    statements: list[str] = []
    real_sqlite_connect = sqlite3.connect

    class NoScriptConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def executescript(self, _script: str) -> None:
            raise AssertionError("executescript must not be used")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(*args: object, **kwargs: object) -> NoScriptConnection:
        connection = real_sqlite_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return NoScriptConnection(connection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    sqlite_backend._initialize_database(database_path)

    normalized = [_normalize_sql(statement) for statement in statements]
    assert "BEGIN EXCLUSIVE" in normalized
    assert normalized.index("PRAGMA APPLICATION_ID = 1095912787") < normalized.index(
        "PRAGMA USER_VERSION = 2"
    )
    assert normalized[-1] == "COMMIT"


def test_interrupted_initialization_rolls_back_every_boundary(
    tmp_path: Path,
) -> None:
    markers = [
        _normalize_sql(statement).split(" (")[0]
        for statement in sqlite_backend._SCHEMA_DDL
    ] + [
        "INSERT INTO memory_schema_metadata",
        "PRAGMA application_id = 1095912787",
        "PRAGMA user_version = 2",
    ]

    for index, marker in enumerate(markers):
        database_path = str(tmp_path / f"interrupted-{index}.sqlite3")
        injected = sqlite3.OperationalError(f"interrupted after {marker}")
        plan = _SQLiteFailurePlan(
            after_statement=marker,
            after_statement_error=injected,
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            _install_sqlite_failure_proxy(monkeypatch, plan)
            with pytest.raises(MemoryPersistenceError) as raised:
                sqlite_backend._initialize_database(database_path)
        assert raised.value.__cause__ is injected

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            assert connection.execute("PRAGMA application_id").fetchone() == (0,)
            assert connection.execute("PRAGMA user_version").fetchone() == (0,)
            assert (
                connection.execute(
                    "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
                ).fetchall()
                == []
            )
        finally:
            connection.close()

        sqlite_backend._initialize_database(database_path)


def test_two_initializers_converge(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_spawn_initialize_database,
            args=(database_path, start_event, result_queue),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert all(process.is_alive() for process in processes)
        start_event.set()
        results = [result_queue.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert results == [("ok", ""), ("ok", "")]
        assert all(process.exitcode == 0 for process in processes)
    finally:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            process.close()
        result_queue.close()
        result_queue.join_thread()

    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA application_id").fetchone() == (
            sqlite_backend._APPLICATION_ID,
        )
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute(
            "SELECT schema_catalog_hash FROM memory_schema_metadata"
        ).fetchone() == (sqlite_backend._catalog_hash(connection.cursor()),)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "sql_marker",
    [
        "CREATE TABLE memory_revisions",
        "PRAGMA application_id = 1095912787",
        "PRAGMA user_version = 2",
    ],
    ids=["ddl", "application-id", "user-version"],
)
def test_initializer_process_death_leaves_reinitializable_state(
    tmp_path: Path,
    sql_marker: str,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_spawn_crash_during_initialize,
        args=(database_path, sql_marker),
    )
    try:
        process.start()
        process.join(timeout=10)
        assert process.exitcode == 24
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        process.close()

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA application_id").fetchone() == (0,)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert (
            connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            == []
        )
    finally:
        connection.close()

    sqlite_backend._initialize_database(database_path)


def test_configuration_queries_delete_mode_without_assigning_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    statements: list[str] = []
    real_connect = sqlite3.connect

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    sqlite_backend._initialize_database(database_path)

    normalized = [_normalize_sql(statement) for statement in statements]
    assert normalized.count("PRAGMA JOURNAL_MODE") >= 2
    first_journal_query = normalized.index("PRAGMA JOURNAL_MODE")
    first_configuration_write = min(
        normalized.index("PRAGMA FOREIGN_KEYS = ON"),
        normalized.index("PRAGMA BUSY_TIMEOUT = 5000"),
        normalized.index("PRAGMA SYNCHRONOUS = FULL"),
    )
    assert first_journal_query < first_configuration_write
    assert not any(
        statement.startswith("PRAGMA JOURNAL_MODE") and "=" in statement
        for statement in normalized
    )


@pytest.mark.parametrize(
    ("pragma", "bad_value", "message"),
    [
        ("PRAGMA JOURNAL_MODE", ("wal",), "DELETE journal"),
        ("PRAGMA FOREIGN_KEYS", (0,), "foreign_keys"),
        ("PRAGMA BUSY_TIMEOUT", (1,), "busy_timeout"),
        ("PRAGMA SYNCHRONOUS", (1,), "synchronous"),
    ],
)
def test_bad_pragma_readback_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pragma: str,
    bad_value: tuple[object, ...],
    message: str,
) -> None:
    database_path = str(tmp_path / f"{pragma.split()[-1].lower()}.sqlite3")
    real_connect = sqlite3.connect

    def connect(*args: object, **kwargs: object) -> _ReadbackOverrideConnection:
        return _ReadbackOverrideConnection(
            real_connect(*args, **kwargs),
            {pragma: bad_value},
        )

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(MemoryPersistenceSchemaError, match=message):
        sqlite_backend._initialize_database(database_path)


def test_vacuumed_empty_v0_database_is_not_adopted(tmp_path: Path) -> None:
    database_path = str(tmp_path / "vacuumed.sqlite3")
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("VACUUM")
        assert connection.execute("PRAGMA page_count").fetchone() == (1,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0,)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute("PRAGMA schema_version").fetchone() != (0,)
        assert (
            connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            == []
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="empty v0"):
        sqlite_backend._initialize_database(database_path)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA page_count").fetchone() == (1,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0,)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert (
            connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            == []
        )
    finally:
        connection.close()


def test_locked_schema_version_rejects_vacuum_between_probe_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "raced-vacuum.sqlite3")
    real_connect = sqlite_backend._connect
    vacuum_calls = 0

    class VacuumAfterFetchCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> VacuumAfterFetchCursor:
            self._last_sql = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchone(self) -> tuple[object, ...] | None:
            nonlocal vacuum_calls
            row = self._real_cursor.fetchone()
            if self._last_sql == "PRAGMA PAGE_COUNT":
                assert self._real_cursor.fetchone() is None
                connection = sqlite3.connect(database_path, isolation_level=None)
                try:
                    connection.execute("VACUUM")
                finally:
                    connection.close()
                vacuum_calls += 1
            return row

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class VacuumAfterFetchConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(self) -> VacuumAfterFetchCursor:
            return VacuumAfterFetchCursor(self._real_connection.cursor())

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> VacuumAfterFetchConnection:
        return VacuumAfterFetchConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)

    with pytest.raises(MemoryPersistenceSchemaError, match="empty v0"):
        sqlite_backend._initialize_database(database_path)

    assert vacuum_calls == 1
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA page_count").fetchone() == (1,)
        assert connection.execute("PRAGMA schema_version").fetchone() != (0,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0,)
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert (
            connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            == []
        )
    finally:
        connection.close()


def test_nonempty_v0_database_is_not_adopted(tmp_path: Path) -> None:
    database_path = str(tmp_path / "foreign.sqlite3")
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="empty v0"):
        sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table'"
        ).fetchall() == [("unrelated",)]
    finally:
        connection.close()


def test_wrong_application_id_is_rejected(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA application_id = 17")
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="application_id"):
        sqlite_backend._initialize_database(database_path)


@pytest.mark.parametrize("version", [3, 2**31 - 1])
def test_unknown_schema_version_is_rejected(tmp_path: Path, version: int) -> None:
    database_path = str(tmp_path / f"memory-{version}.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(f"PRAGMA user_version = {version}")
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="user_version"):
        sqlite_backend._initialize_database(database_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "DROP TABLE memory_release_aliases",
        "DROP TABLE memory_evidence_snapshots",
        "DROP INDEX idx_memory_revisions_sort",
        "DROP INDEX idx_memory_evidence_ingest_scope",
        "CREATE VIEW unexpected_memory_view AS SELECT scope_id FROM memory_scopes",
    ],
)
def test_catalog_drift_is_rejected(tmp_path: Path, mutation: str) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(mutation)
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="catalog"):
        sqlite_backend._initialize_database(database_path)


def test_changed_metadata_hash_is_rejected(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE memory_schema_metadata SET schema_spec_hash = ?",
            ("0" * 64,),
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="specification"):
        sqlite_backend._initialize_database(database_path)


def test_changed_metadata_table_shape_is_reported_as_schema_error(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("DROP TABLE memory_schema_metadata")
        connection.execute(
            "CREATE TABLE memory_schema_metadata (singleton INTEGER PRIMARY KEY)"
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="metadata"):
        sqlite_backend._initialize_database(database_path)


@pytest.mark.parametrize(
    ("error_code", "expected_type"),
    [
        (sqlite3.SQLITE_BUSY, MemoryPersistenceBusyError),
        (sqlite3.SQLITE_IOERR, MemoryPersistenceError),
    ],
)
def test_metadata_read_sqlite_failures_keep_operational_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
    expected_type: type[MemoryPersistenceError],
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    injected = sqlite3.OperationalError("injected metadata read failure")
    injected.sqlite_errorcode = error_code
    plan = _SQLiteFailurePlan(
        after_statement="SELECT * FROM memory_schema_metadata",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        sqlite_backend._initialize_database(database_path)

    assert type(raised.value) is expected_type
    assert raised.value.__cause__ is injected


def test_foreign_key_damage_is_reported_as_corruption(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
        connection.execute(
            "INSERT INTO memory_candidate_evidence "
            "(scope_id, candidate_id, position, evidence_id) VALUES (?, ?, ?, ?)",
            (1, "missing-candidate", 0, "missing-evidence"),
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceCorruptionError, match="foreign key"):
        sqlite_backend._initialize_database(database_path)


def test_empty_wal_database_is_rejected_without_conversion(tmp_path: Path) -> None:
    database_path = str(tmp_path / "empty-wal.sqlite3")
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE temporary_table(value TEXT)")
        connection.execute("DROP TABLE temporary_table")
        assert (
            connection.execute(
                "SELECT name FROM main.sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            == []
        )
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="DELETE journal"):
        sqlite_backend._initialize_database(database_path)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        connection.close()


def test_v2_switched_to_wal_is_rejected_without_conversion(tmp_path: Path) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    finally:
        connection.close()

    with pytest.raises(MemoryPersistenceSchemaError, match="DELETE journal"):
        sqlite_backend._initialize_database(database_path)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        connection.close()


@pytest.mark.parametrize("error_code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED])
def test_sqlite_busy_and_locked_primary_codes_map_to_busy(
    error_code: int,
) -> None:
    error = sqlite3.OperationalError("database unavailable")
    error.sqlite_errorcode = error_code | (7 << 8)
    mapped = sqlite_backend._map_sqlite_error(error)
    assert type(mapped) is MemoryPersistenceBusyError


def test_other_sqlite_errors_map_to_general_persistence_error() -> None:
    error = sqlite3.OperationalError("disk I/O error")
    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
    mapped = sqlite_backend._map_sqlite_error(error)
    assert type(mapped) is MemoryPersistenceError
    assert not isinstance(mapped, MemoryPersistenceBusyError)


def test_record_and_alias_hashes_match_golden_wire_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bytes] = []
    real_sha256 = hashlib.sha256

    def capture(payload: bytes) -> Any:
        seen.append(payload)
        return real_sha256(payload)

    monkeypatch.setattr(sqlite_backend, "_RECORD_SHA256", capture)
    scope = MemoryScope("tenant-1", "assistant-memory", "user-1")

    assert (
        sqlite_backend._record_storage_hash(
            record_kind="evidence",
            scope=scope,
            record_id="evd_a",
            content_hash="a" * 64,
            created_at_text="2026-07-08T00:00:00+00:00",
        )
        == "2b8b502e8b9ac8367f01ed63102cd9efd9d976de1494e26a1628a4dd361d08ae"
    )
    assert (
        sqlite_backend._record_storage_hash(
            record_kind="revision",
            scope=scope,
            record_id="rev_a",
            content_hash="b" * 64,
            created_at_text="2026-07-08T00:00:00+00:00",
            memory_id="mem_a",
            generation=7,
        )
        == "540a7bddb3093f79dd1f34782c7679942b285bece6ba32702b1c404a29a2cbac"
    )
    assert (
        sqlite_backend._record_storage_hash(
            record_kind="evidence_snapshot",
            scope=scope,
            record_id="esnap_a",
            content_hash="c" * 64,
            created_at_text="2026-07-08T00:00:00+00:00",
        )
        == "2d82659f84c45b88e71110c1486dd6e3743263df9df0e5d0a837697de4553ff3"
    )
    assert (
        sqlite_backend._release_binding_hash(
            scope=scope,
            idempotency_key="alias-a",
            release_id="rel_a",
        )
        == "f61c2a3ef34cfe876c675f843c18508e7134c64b3fed251e48ad9885459e71aa"
    )
    assert (
        sqlite_backend._evidence_ingest_binding_hash(
            scope=scope,
            evidence_id="evd_a",
            ingest_order=7,
        )
        == "4392800f39feb7aada54d810b2b67af79ee7293916e3eb7586ba0b757bdbad84"
    )
    assert (
        sqlite_backend._snapshot_binding_hash(
            scope=scope,
            idempotency_key="snapshot-alias-a",
            snapshot_id="esnap_a",
        )
        == "798788f2f670cad7af183b4b47d466e0c6e37d72ae5d94abbda42a1a16dc1252"
    )

    assert seen == [
        b'{"content_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","created_at":"2026-07-08T00:00:00+00:00","record_id":"evd_a","record_kind":"evidence","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"content_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","created_at":"2026-07-08T00:00:00+00:00","generation":7,"memory_id":"mem_a","record_id":"rev_a","record_kind":"revision","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"content_hash":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","created_at":"2026-07-08T00:00:00+00:00","record_id":"esnap_a","record_kind":"evidence_snapshot","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"idempotency_key":"alias-a","record_kind":"release_alias","release_id":"rel_a","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"evidence_id":"evd_a","ingest_order":7,"record_kind":"evidence_ingest_order","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"idempotency_key":"snapshot-alias-a","record_kind":"evidence_snapshot_alias","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"},"snapshot_id":"esnap_a"}',
    ]


@pytest.mark.parametrize(
    "record_kind",
    ["evidence", "candidate", "release", "evidence_snapshot"],
)
def test_nonrevision_storage_hash_rejects_revision_binding_fields(
    record_kind: str,
) -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "user-1")
    with pytest.raises(ValueError, match="revision"):
        sqlite_backend._record_storage_hash(
            record_kind=record_kind,
            scope=scope,
            record_id="record-a",
            content_hash="a" * 64,
            created_at_text="2026-07-08T00:00:00+00:00",
            memory_id="mem-a",
            generation=1,
        )


@pytest.mark.parametrize(
    ("memory_id", "generation"),
    [(None, None), ("mem-a", None), (None, 1)],
)
def test_revision_storage_hash_requires_both_binding_fields(
    memory_id: str | None,
    generation: int | None,
) -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "user-1")
    with pytest.raises(ValueError, match="revision"):
        sqlite_backend._record_storage_hash(
            record_kind="revision",
            scope=scope,
            record_id="rev-a",
            content_hash="a" * 64,
            created_at_text="2026-07-08T00:00:00+00:00",
            memory_id=memory_id,
            generation=generation,
        )


def test_schema_and_record_hash_injection_seams_are_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "memory.sqlite3")
    sqlite_backend._initialize_database(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "user-1")

    def schema_hash_must_not_run(_payload: bytes) -> Any:
        raise AssertionError("record hashing used the schema hash seam")

    monkeypatch.setattr(sqlite_backend, "_SCHEMA_SHA256", schema_hash_must_not_run)
    sqlite_backend._record_storage_hash(
        record_kind="evidence",
        scope=scope,
        record_id="evd-a",
        content_hash="a" * 64,
        created_at_text="2026-07-08T00:00:00+00:00",
    )

    monkeypatch.undo()

    def record_hash_must_not_run(_payload: bytes) -> Any:
        raise AssertionError("schema hashing used the record hash seam")

    monkeypatch.setattr(sqlite_backend, "_RECORD_SHA256", record_hash_must_not_run)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        sqlite_backend._catalog_hash(connection.cursor())
    finally:
        connection.close()


def test_sqlite_candidate_reopen_retry_preserves_ordered_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "candidate.sqlite3")
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")
    first_evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            sequence_no=0,
            payload="first evidence",
            idempotency_key="candidate-evidence-1",
        )
    )
    second_evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            sequence_no=1,
            payload="second evidence",
            idempotency_key="candidate-evidence-2",
        )
    )
    proposal = _make_sqlite_candidate(
        scope=scope,
        evidence_ids=(second_evidence.evidence_id, first_evidence.evidence_id),
    )

    original = store.append_candidate(proposal)
    del store
    reopened = SQLiteMemoryStore(database_path)
    real_connect = sqlite_backend._connect
    reversed_edge_queries: list[int] = []

    class ReverseCandidateEdgesCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> ReverseCandidateEdgesCursor:
            self._last_sql = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            if self._last_sql.startswith(
                "SELECT SCOPE_ID, CANDIDATE_ID, POSITION, EVIDENCE_ID "
                "FROM MEMORY_CANDIDATE_EVIDENCE"
            ):
                rows.reverse()
                reversed_edge_queries.append(len(rows))
            return rows

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class ReverseCandidateEdgesConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> ReverseCandidateEdgesCursor:
            return ReverseCandidateEdgesCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    monkeypatch.setattr(
        sqlite_backend,
        "_connect",
        lambda path: ReverseCandidateEdgesConnection(real_connect(path)),
    )
    loaded = reopened.get_candidate(scope, original.candidate_id)
    provenance = reopened.get_candidate_evidence(scope, original.candidate_id)
    retry = reopened.append_candidate(
        CandidateProposal(
            scope=proposal.scope,
            content=proposal.content,
            evidence_ids=proposal.evidence_ids,
            idempotency_key=proposal.idempotency_key,
        )
    )

    expected_hash = hashlib.sha256(proposal.canonical_bytes()).hexdigest()
    assert original.proposal == proposal
    assert original.proposal is not proposal
    assert loaded == retry == original
    assert loaded is not original
    assert retry is not original
    assert reopened.list_candidates(scope) == (original,)
    assert provenance == (second_evidence, first_evidence)
    assert reversed_edge_queries == [2, 2, 2, 2]
    assert original.content_hash == expected_hash
    assert original.candidate_id == f"cand_{expected_hash[:24]}"
    assert original.created_at.tzinfo is UTC
    assert loaded.created_at == retry.created_at == original.created_at

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_evidence"
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT position, evidence_id FROM memory_candidate_evidence "
            "ORDER BY position"
        ).fetchall() == [
            (0, second_evidence.evidence_id),
            (1, first_evidence.evidence_id),
        ]
    finally:
        connection.close()


def test_sqlite_candidate_queries_are_scoped_sorted_and_snapshot_inputs(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "candidate-queries.sqlite3")
    first_scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user-1")
    second_scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user-2")
    first_evidence = store.append(
        _make_sqlite_evidence(
            scope=first_scope,
            idempotency_key="query-evidence-1",
        )
    )
    second_evidence = store.append(
        _make_sqlite_evidence(
            scope=second_scope,
            payload="foreign evidence",
            idempotency_key="query-evidence-2",
        )
    )
    right = store.append_candidate(
        _make_sqlite_candidate(
            scope=first_scope,
            content="right",
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key="right-attempt",
        )
    )
    left = store.append_candidate(
        _make_sqlite_candidate(
            scope=first_scope,
            content="left",
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key="left-attempt",
        )
    )
    same_content_new_attempt = store.append_candidate(
        _make_sqlite_candidate(
            scope=first_scope,
            content=left.proposal.content,
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key="new-attempt",
        )
    )
    foreign = store.append_candidate(
        _make_sqlite_candidate(
            scope=second_scope,
            content="foreign",
            evidence_ids=(second_evidence.evidence_id,),
            idempotency_key="foreign-attempt",
        )
    )
    expected = tuple(
        sorted(
            (right, left, same_content_new_attempt), key=lambda item: item.candidate_id
        )
    )

    assert left.candidate_id != same_content_new_attempt.candidate_id
    assert store.list_candidates(first_scope) == expected
    assert store.list_candidates(second_scope) == (foreign,)
    assert store.get_candidate(first_scope, left.candidate_id) == left
    snapshot = store.list_candidates(first_scope)
    store.append_candidate(
        _make_sqlite_candidate(
            scope=first_scope,
            content="later",
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key="later-attempt",
        )
    )
    assert snapshot == expected

    with pytest.raises(CandidateNotFoundError) as foreign_error:
        store.get_candidate(second_scope, left.candidate_id)
    with pytest.raises(CandidateNotFoundError) as missing_error:
        store.get_candidate(
            MemoryScope("tenant-1", "assistant-memory", "candidate-user-3"),
            left.candidate_id,
        )
    assert type(foreign_error.value) is CandidateNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)


def test_sqlite_candidate_public_boundaries_fail_before_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "candidate-boundaries.sqlite3")
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            idempotency_key="boundary-evidence",
        )
    )
    candidate = store.append_candidate(
        _make_sqlite_candidate(
            scope=scope,
            evidence_ids=(evidence.evidence_id,),
        )
    )
    query_id = _SnapshotProbeStr(candidate.candidate_id)

    assert store.get_candidate(scope, query_id) == candidate
    assert store.get_candidate_evidence(scope, query_id) == (evidence,)
    assert query_id.override_calls == 0

    for missing_id in ("", " \t", "\x00"):
        with pytest.raises(CandidateNotFoundError) as get_error:
            store.get_candidate(scope, missing_id)
        with pytest.raises(CandidateNotFoundError) as evidence_error:
            store.get_candidate_evidence(scope, missing_id)
        assert (
            str(get_error.value)
            == f"candidate {str.__str__(missing_id)!r} was not found"
        )
        assert str(evidence_error.value) == str(get_error.value)
    assert store.list_candidates(scope) == (candidate,)

    def transaction_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation reached SQLite I/O")

    monkeypatch.setattr(
        sqlite_store_module, "_read_transaction", transaction_must_not_start
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_write_transaction",
        transaction_must_not_start,
    )
    proposal = _make_sqlite_candidate(scope=scope)
    proposal_subclass = _SQLiteCandidateProposalSubclass(
        proposal.scope,
        proposal.content,
        proposal.evidence_ids,
        proposal.idempotency_key,
    )
    scope_subclass = _SQLiteMemoryScopeSubclass(
        scope.tenant_id,
        scope.namespace,
        scope.subject_id,
    )

    with pytest.raises(TypeError, match="proposal must be a CandidateProposal"):
        store.append_candidate(proposal_subclass)
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get_candidate(scope_subclass, "cand_missing")
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get_candidate_evidence(scope_subclass, "cand_missing")
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.list_candidates(scope_subclass)
    for invalid_id in (None, 7):
        with pytest.raises(TypeError, match="candidate_id must be a string"):
            store.get_candidate(scope, invalid_id)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="candidate_id must be a string"):
            store.get_candidate_evidence(scope, invalid_id)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="candidate_id must be valid UTF-8"):
        store.get_candidate(scope, "\ud800")
    with pytest.raises(ValueError, match="candidate_id must be valid UTF-8"):
        store.get_candidate_evidence(scope, "\ud800")


def test_sqlite_candidate_list_order_is_independent_of_sqlite_row_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "candidate-order.sqlite3")
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            idempotency_key="order-evidence",
        )
    )
    candidates = tuple(
        store.append_candidate(
            _make_sqlite_candidate(
                scope=scope,
                content=f"candidate-{index}",
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=f"candidate-order-{index}",
            )
        )
        for index in range(3)
    )
    real_connect = sqlite_backend._connect
    reversed_queries: list[int] = []

    class ReverseCandidateRowsCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> ReverseCandidateRowsCursor:
            self._last_sql = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            if self._last_sql.startswith(
                "SELECT SCOPE_ID, CANDIDATE_ID FROM MEMORY_CANDIDATES"
            ):
                rows.reverse()
                reversed_queries.append(len(rows))
            return rows

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class ReverseCandidateRowsConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> ReverseCandidateRowsCursor:
            return ReverseCandidateRowsCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    monkeypatch.setattr(
        sqlite_backend,
        "_connect",
        lambda path: ReverseCandidateRowsConnection(real_connect(path)),
    )

    assert store.list_candidates(scope) == tuple(
        sorted(candidates, key=lambda item: item.candidate_id)
    )
    assert reversed_queries == [3]


@pytest.mark.parametrize(
    "invalid_evidence_kind",
    ["first-missing", "late-missing", "late-foreign"],
)
def test_sqlite_candidate_missing_or_foreign_evidence_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_evidence_kind: str,
) -> None:
    database_path = tmp_path / f"candidate-{invalid_evidence_kind}.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")

    if invalid_evidence_kind == "first-missing":
        missing_evidence_id = "evd_first_missing"
        evidence_ids = (missing_evidence_id,)
        expected_state = (0, 0, 0, 0)

        def scope_creation_must_not_run(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("candidate validation attempted to create a scope")

        monkeypatch.setattr(
            sqlite_store_module,
            "_ensure_scope_id",
            scope_creation_must_not_run,
        )
    else:
        first = store.append(
            _make_sqlite_evidence(
                scope=scope,
                idempotency_key=f"{invalid_evidence_kind}-first-evidence",
            )
        )
        if invalid_evidence_kind == "late-missing":
            missing_evidence_id = "evd_late_missing"
            expected_state = (1, 1, 0, 0)
        else:
            foreign = store.append(
                _make_sqlite_evidence(
                    scope=MemoryScope(
                        "tenant-1",
                        "assistant-memory",
                        "foreign-candidate-user",
                    ),
                    payload="foreign candidate evidence",
                    idempotency_key="late-foreign-evidence",
                )
            )
            missing_evidence_id = foreign.evidence_id
            expected_state = (2, 2, 0, 0)
        evidence_ids = (first.evidence_id, missing_evidence_id)

    with pytest.raises(EvidenceNotFoundError) as raised:
        store.append_candidate(
            _make_sqlite_candidate(
                scope=scope,
                evidence_ids=evidence_ids,
                idempotency_key=f"{invalid_evidence_kind}-candidate",
            )
        )

    assert type(raised.value) is EvidenceNotFoundError
    assert str(raised.value) == f"evidence {missing_evidence_id!r} was not found"
    assert store.list_candidates(scope) == ()
    assert _memory_graph_state(database_path) == (expected_state, [])


def test_sqlite_candidate_idempotency_conflict_precedes_missing_evidence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "candidate-idempotency-precedence.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "foreign-candidate-user",
    )
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            idempotency_key="idempotency-owner-evidence",
        )
    )
    foreign = store.append(
        _make_sqlite_evidence(
            scope=foreign_scope,
            payload="foreign candidate evidence",
            idempotency_key="idempotency-foreign-evidence",
        )
    )
    owner = store.append_candidate(
        _make_sqlite_candidate(
            scope=scope,
            content="idempotency owner",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key="shared-candidate-key",
        )
    )

    for invalid_evidence_id in ("evd_missing", foreign.evidence_id):
        with pytest.raises(CandidateConflictError) as raised:
            store.append_candidate(
                _make_sqlite_candidate(
                    scope=scope,
                    content=f"changed for {invalid_evidence_id}",
                    evidence_ids=(invalid_evidence_id,),
                    idempotency_key="shared-candidate-key",
                )
            )

        assert type(raised.value) is CandidateConflictError
        assert str(raised.value) == (
            "scoped candidate idempotency key already refers to different content"
        )
        assert store.list_candidates(scope) == (owner,)
        assert _memory_graph_state(database_path) == ((2, 2, 1, 1), [])


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
@pytest.mark.parametrize("invalid_evidence_kind", ["missing", "foreign"])
def test_sqlite_candidate_missing_evidence_precedes_id_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
    invalid_evidence_kind: str,
) -> None:
    database_path = tmp_path / (
        f"candidate-{invalid_evidence_kind}-{collision_kind}-precedence.sqlite3"
    )
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-user")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "foreign-candidate-user",
    )
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            idempotency_key="collision-owner-evidence",
        )
    )
    foreign = store.append(
        _make_sqlite_evidence(
            scope=foreign_scope,
            payload="foreign candidate evidence",
            idempotency_key="collision-foreign-evidence",
        )
    )
    owner_proposal = _make_sqlite_candidate(
        scope=scope,
        content="candidate ID owner",
        evidence_ids=(evidence.evidence_id,),
        idempotency_key="candidate-id-owner",
    )
    owner = store.append_candidate(owner_proposal)
    invalid_evidence_id = (
        "evd_missing" if invalid_evidence_kind == "missing" else foreign.evidence_id
    )
    later_invalid_evidence_id = (
        "evd_later_missing"
        if invalid_evidence_kind == "missing"
        else "evd_missing_after_foreign"
    )
    challenger = _make_sqlite_candidate(
        scope=scope,
        content=f"{invalid_evidence_kind} collision challenger",
        evidence_ids=(
            evidence.evidence_id,
            invalid_evidence_id,
            later_invalid_evidence_id,
        ),
        idempotency_key=f"{invalid_evidence_kind}-collision-challenger",
    )
    if collision_kind == "full-hash":
        challenger_digest = owner.content_hash
    else:
        replacement_suffix = (
            "0" * 40 if owner.content_hash[24:] != "0" * 40 else "1" * 40
        )
        challenger_digest = owner.content_hash[:24] + replacement_suffix
    digest_by_canonical = {
        owner_proposal.canonical_bytes(): owner.content_hash,
        challenger.canonical_bytes(): challenger_digest,
    }
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )

    assert challenger_digest[:24] == owner.content_hash[:24]
    assert f"cand_{challenger_digest[:24]}" == owner.candidate_id
    with pytest.raises(EvidenceNotFoundError) as raised:
        store.append_candidate(challenger)

    assert type(raised.value) is EvidenceNotFoundError
    assert str(raised.value) == f"evidence {invalid_evidence_id!r} was not found"
    assert store.list_candidates(scope) == (owner,)
    assert _memory_graph_state(database_path) == ((2, 2, 1, 1), [])


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
def test_sqlite_candidate_collision_is_scoped_atomic_and_loser_key_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    database_path = tmp_path / f"candidate-{collision_kind}.sqlite3"
    store = SQLiteMemoryStore(database_path)
    first_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "candidate-collision-user-1",
    )
    second_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "candidate-collision-user-2",
    )
    first_evidence = store.append(
        _make_sqlite_evidence(
            scope=first_scope,
            idempotency_key="first-collision-evidence",
        )
    )
    second_evidence = store.append(
        _make_sqlite_evidence(
            scope=second_scope,
            payload="cross-scope candidate evidence",
            idempotency_key="second-collision-evidence",
        )
    )
    winner_proposal = _make_sqlite_candidate(
        scope=first_scope,
        content="candidate collision winner",
        evidence_ids=(first_evidence.evidence_id,),
        idempotency_key="winner-key",
    )
    loser_proposal = _make_sqlite_candidate(
        scope=first_scope,
        content="candidate collision loser",
        evidence_ids=(first_evidence.evidence_id,),
        idempotency_key="loser-key",
    )
    replacement_proposal = _make_sqlite_candidate(
        scope=first_scope,
        content="candidate collision replacement",
        evidence_ids=(first_evidence.evidence_id,),
        idempotency_key="loser-key",
    )
    cross_scope_proposal = _make_sqlite_candidate(
        scope=second_scope,
        content="cross-scope candidate",
        evidence_ids=(second_evidence.evidence_id,),
        idempotency_key="loser-key",
    )
    shared_prefix = "a" * 24
    winner_digest = (
        "a" * 64 if collision_kind == "full-hash" else shared_prefix + "b" * 40
    )
    loser_digest = (
        winner_digest if collision_kind == "full-hash" else shared_prefix + "c" * 40
    )
    replacement_digest = "d" * 64
    digest_by_canonical = {
        winner_proposal.canonical_bytes(): winner_digest,
        loser_proposal.canonical_bytes(): loser_digest,
        replacement_proposal.canonical_bytes(): replacement_digest,
        cross_scope_proposal.canonical_bytes(): winner_digest,
    }
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )

    assert _memory_graph_state(database_path) == ((2, 2, 0, 0), [])
    winner = store.append_candidate(winner_proposal)
    assert winner.content_hash == winner_digest
    assert _memory_graph_state(database_path) == ((2, 2, 1, 1), [])

    with pytest.raises(CandidateConflictError) as raised:
        store.append_candidate(loser_proposal)

    assert type(raised.value) is CandidateConflictError
    assert str(raised.value) == (f"candidate ID collision for {winner.candidate_id!r}")
    assert _memory_graph_state(database_path) == ((2, 2, 1, 1), [])

    cross_scope = store.append_candidate(cross_scope_proposal)
    assert cross_scope.candidate_id == winner.candidate_id
    assert store.get_candidate(first_scope, winner.candidate_id) == winner
    assert store.get_candidate(second_scope, winner.candidate_id) == cross_scope
    assert _memory_graph_state(database_path) == ((2, 2, 2, 2), [])

    recovered_loser_key = store.append_candidate(replacement_proposal)
    assert recovered_loser_key.candidate_id == f"cand_{replacement_digest[:24]}"
    assert recovered_loser_key.proposal == replacement_proposal
    assert _memory_graph_state(database_path) == ((2, 2, 3, 3), [])

    with pytest.raises(CandidateConflictError) as retry_error:
        store.append_candidate(loser_proposal)
    assert type(retry_error.value) is CandidateConflictError
    assert str(retry_error.value) == (
        "scoped candidate idempotency key already refers to different content"
    )
    assert store.list_candidates(first_scope) == tuple(
        sorted(
            (winner, recovered_loser_key),
            key=lambda candidate: candidate.candidate_id,
        )
    )
    assert store.list_candidates(second_scope) == (cross_scope,)
    assert _memory_graph_state(database_path) == ((2, 2, 3, 3), [])


def test_sqlite_candidate_coherent_address_drift_is_corruption_before_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest_by_canonical: dict[bytes, str] = {}
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )

    for mutation in ("scope", "candidate-id", "idempotency"):
        database_path = tmp_path / f"candidate-coherent-{mutation}-drift.sqlite3"
        store = SQLiteMemoryStore(database_path)
        source_scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-{mutation}-source",
        )
        foreign_scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-{mutation}-foreign",
        )
        source_event = _make_sqlite_evidence(
            scope=source_scope,
            payload=f"source evidence for {mutation}",
            idempotency_key=f"source-evidence-{mutation}",
        )
        foreign_event = _make_sqlite_evidence(
            scope=foreign_scope,
            payload=f"foreign evidence for {mutation}",
            idempotency_key=f"foreign-evidence-{mutation}",
        )
        shared_prefix = hashlib.sha256(mutation.encode()).hexdigest()[:24]
        digest_by_canonical.update(
            {
                source_event.canonical_bytes(): shared_prefix + "a" * 40,
                foreign_event.canonical_bytes(): shared_prefix + "b" * 40,
            }
        )
        source_evidence = store.append(source_event)
        foreign_evidence = store.append(foreign_event)
        assert source_evidence.evidence_id == foreign_evidence.evidence_id
        proposal = _make_sqlite_candidate(
            scope=source_scope,
            content=f"candidate before coherent {mutation} drift",
            evidence_ids=(source_evidence.evidence_id,),
            idempotency_key=f"candidate-{mutation}-request",
        )
        candidate = store.append_candidate(proposal)

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
            source_scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (
                    source_scope.tenant_id,
                    source_scope.namespace,
                    source_scope.subject_id,
                ),
            ).fetchone()[0]
            foreign_scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (
                    foreign_scope.tenant_id,
                    foreign_scope.namespace,
                    foreign_scope.subject_id,
                ),
            ).fetchone()[0]
            if mutation == "scope":
                connection.execute(
                    "UPDATE memory_candidate_evidence SET scope_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ?",
                    (foreign_scope_id, source_scope_id, candidate.candidate_id),
                )
                connection.execute(
                    "UPDATE memory_candidates SET scope_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ?",
                    (foreign_scope_id, source_scope_id, candidate.candidate_id),
                )
                expected_cause = "canonical candidate bytes disagree with projections"
            elif mutation == "candidate-id":
                moved_candidate_id = f"cand_{'0' * 24}"
                assert moved_candidate_id != candidate.candidate_id
                connection.execute(
                    "UPDATE memory_candidate_evidence SET candidate_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ?",
                    (moved_candidate_id, source_scope_id, candidate.candidate_id),
                )
                connection.execute(
                    "UPDATE memory_candidates SET candidate_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ?",
                    (moved_candidate_id, source_scope_id, candidate.candidate_id),
                )
                expected_cause = "candidate ID disagrees with its content hash"
            else:
                connection.execute(
                    "UPDATE memory_candidates SET idempotency_key = ? "
                    "WHERE scope_id = ? AND candidate_id = ?",
                    (
                        f"moved-{proposal.idempotency_key}",
                        source_scope_id,
                        candidate.candidate_id,
                    ),
                )
                expected_cause = "canonical candidate bytes disagree with projections"
        finally:
            connection.close()

        corrupted_state = _memory_graph_state(database_path)
        assert corrupted_state == ((2, 2, 1, 1), [])
        for operation in ("get-missing", "list", "get-evidence", "retry"):
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                if operation == "get-missing":
                    store.get_candidate(source_scope, "cand_missing")
                elif operation == "list":
                    store.list_candidates(source_scope)
                elif operation == "get-evidence":
                    store.get_candidate_evidence(
                        source_scope,
                        candidate.candidate_id,
                    )
                else:
                    store.append_candidate(proposal)

            assert type(raised.value) is MemoryPersistenceCorruptionError
            assert str(raised.value) == (
                "stored candidate row failed integrity validation"
            )
            assert str(raised.value.__cause__) == expected_cause
            assert _memory_graph_state(database_path) == corrupted_state


def test_sqlite_candidate_relation_gap_delete_and_substitution_are_corruption(
    tmp_path: Path,
) -> None:
    mutations = (
        "position-gap",
        "deleted-edge",
        "same-scope-substitution",
        "edge-scope",
        "edge-candidate",
        "edge-evidence",
    )

    for mutation in mutations:
        database_path = tmp_path / f"candidate-relation-{mutation}.sqlite3"
        store = SQLiteMemoryStore(database_path)
        scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-relation-{mutation}",
        )
        foreign_scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-relation-{mutation}-foreign",
        )
        evidence = tuple(
            store.append(
                _make_sqlite_evidence(
                    scope=scope,
                    sequence_no=index,
                    payload=f"candidate relation evidence {index}",
                    idempotency_key=f"relation-{mutation}-evidence-{index}",
                )
            )
            for index in range(3)
        )
        store.append(
            _make_sqlite_evidence(
                scope=foreign_scope,
                payload="foreign relation evidence",
                idempotency_key=f"relation-{mutation}-foreign-evidence",
            )
        )
        proposal = _make_sqlite_candidate(
            scope=scope,
            content=f"candidate relation owner for {mutation}",
            evidence_ids=(evidence[0].evidence_id, evidence[1].evidence_id),
            idempotency_key=f"relation-{mutation}-candidate",
        )
        candidate = store.append_candidate(proposal)

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
            scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (scope.tenant_id, scope.namespace, scope.subject_id),
            ).fetchone()[0]
            foreign_scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (
                    foreign_scope.tenant_id,
                    foreign_scope.namespace,
                    foreign_scope.subject_id,
                ),
            ).fetchone()[0]
            parameters = (scope_id, candidate.candidate_id)
            if mutation == "position-gap":
                connection.execute(
                    "UPDATE memory_candidate_evidence SET position = 2 "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 1",
                    parameters,
                )
                expected_message = (
                    "candidate evidence positions are not contiguous from zero"
                )
                expected_cause = None
            elif mutation == "deleted-edge":
                connection.execute(
                    "DELETE FROM memory_candidate_evidence "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 1",
                    parameters,
                )
                expected_message = "stored candidate row failed integrity validation"
                expected_cause = "canonical candidate bytes disagree with projections"
            elif mutation == "same-scope-substitution":
                connection.execute(
                    "UPDATE memory_candidate_evidence SET evidence_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 1",
                    (evidence[2].evidence_id, *parameters),
                )
                expected_message = "stored candidate row failed integrity validation"
                expected_cause = "canonical candidate bytes disagree with projections"
            elif mutation == "edge-scope":
                connection.execute(
                    "UPDATE memory_candidate_evidence SET scope_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 0",
                    (foreign_scope_id, *parameters),
                )
                expected_message = (
                    "Memory Service SQLite data failed foreign key validation"
                )
                expected_cause = None
            elif mutation == "edge-candidate":
                connection.execute(
                    "UPDATE memory_candidate_evidence SET candidate_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 0",
                    ("cand_edge_projection_drift", *parameters),
                )
                expected_message = (
                    "Memory Service SQLite data failed foreign key validation"
                )
                expected_cause = None
            else:
                connection.execute(
                    "UPDATE memory_candidate_evidence SET evidence_id = ? "
                    "WHERE scope_id = ? AND candidate_id = ? AND position = 0",
                    ("evd_edge_projection_drift", *parameters),
                )
                expected_message = (
                    "Memory Service SQLite data failed foreign key validation"
                )
                expected_cause = None
        finally:
            connection.close()

        corrupted_state = _memory_graph_state(database_path)
        expected_edge_count = 1 if mutation == "deleted-edge" else 2
        assert corrupted_state[0] == (2, 4, 1, expected_edge_count)
        if mutation in {
            "position-gap",
            "deleted-edge",
            "same-scope-substitution",
        }:
            assert corrupted_state[1] == []
        else:
            assert corrupted_state[1]

        for operation in ("get-missing", "list", "get-evidence", "retry"):
            counts_before, foreign_keys_before = _memory_graph_state(database_path)
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                if operation == "get-missing":
                    store.get_candidate(scope, "cand_missing")
                elif operation == "list":
                    store.list_candidates(scope)
                elif operation == "get-evidence":
                    store.get_candidate_evidence(scope, candidate.candidate_id)
                else:
                    store.append_candidate(proposal)

            assert type(raised.value) is MemoryPersistenceCorruptionError
            assert str(raised.value) == expected_message
            if expected_cause is None:
                assert raised.value.__cause__ is None
            else:
                assert str(raised.value.__cause__) == expected_cause
            counts_after, foreign_keys_after = _memory_graph_state(database_path)
            assert counts_after == counts_before
            assert foreign_keys_after == foreign_keys_before


def test_sqlite_candidate_loader_rejects_each_scalar_integrity_drift(
    tmp_path: Path,
) -> None:
    mutations = (
        "canonical",
        "content",
        "content-hash-suffix",
        "created-at-z",
        "created-at-offset",
        "storage-hash",
    )

    for mutation in mutations:
        database_path = tmp_path / f"candidate-scalar-{mutation}.sqlite3"
        store = SQLiteMemoryStore(database_path)
        scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-scalar-{mutation}",
        )
        evidence = store.append(
            _make_sqlite_evidence(
                scope=scope,
                idempotency_key=f"candidate-scalar-{mutation}-evidence",
            )
        )
        proposal = _make_sqlite_candidate(
            scope=scope,
            content=f"candidate scalar owner for {mutation}",
            evidence_ids=(evidence.evidence_id,),
            idempotency_key=f"candidate-scalar-{mutation}-request",
        )
        candidate = store.append_candidate(proposal)
        canonical_variant = b" \n" + proposal.canonical_bytes()
        assert canonical_variant != proposal.canonical_bytes()
        assert json.loads(canonical_variant) == json.loads(proposal.canonical_bytes())
        replacement_suffix = (
            "0" * 40 if candidate.content_hash[24:] != "0" * 40 else "1" * 40
        )
        changed_content_hash = candidate.content_hash[:24] + replacement_suffix
        assert changed_content_hash != candidate.content_hash
        mutations_by_name: dict[str, tuple[str, object, str]] = {
            "canonical": (
                "UPDATE memory_candidates SET canonical = ? WHERE candidate_id = ?",
                sqlite3.Binary(canonical_variant),
                "canonical candidate bytes disagree with projections",
            ),
            "content": (
                "UPDATE memory_candidates SET content = ? WHERE candidate_id = ?",
                "changed candidate content",
                "canonical candidate bytes disagree with projections",
            ),
            "content-hash-suffix": (
                "UPDATE memory_candidates SET content_hash = ? WHERE candidate_id = ?",
                changed_content_hash,
                "candidate content hash disagrees with canonical bytes",
            ),
            "created-at-z": (
                "UPDATE memory_candidates SET created_at = ? WHERE candidate_id = ?",
                candidate.created_at.isoformat().replace("+00:00", "Z"),
                "candidate created_at is not exact UTC isoformat text",
            ),
            "created-at-offset": (
                "UPDATE memory_candidates SET created_at = ? WHERE candidate_id = ?",
                candidate.created_at.astimezone(
                    timezone(timedelta(hours=8))
                ).isoformat(),
                "candidate created_at is not exact UTC isoformat text",
            ),
            "storage-hash": (
                "UPDATE memory_candidates SET storage_hash = ? WHERE candidate_id = ?",
                "0" * 64,
                "candidate storage hash disagrees with stored metadata",
            ),
        }
        sql, changed_value, expected_cause = mutations_by_name[mutation]

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            stored_storage_hash = connection.execute(
                "SELECT storage_hash FROM memory_candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()[0]
            if mutation == "storage-hash":
                assert changed_value != stored_storage_hash
            connection.execute(sql, (changed_value, candidate.candidate_id))
        finally:
            connection.close()

        corrupted_state = _memory_graph_state(database_path)
        assert corrupted_state == ((1, 1, 1, 1), [])
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            store.get_candidate(scope, candidate.candidate_id)

        assert type(raised.value) is MemoryPersistenceCorruptionError
        assert str(raised.value) == ("stored candidate row failed integrity validation")
        assert type(raised.value.__cause__) is ValueError
        assert str(raised.value.__cause__) == expected_cause
        assert _memory_graph_state(database_path) == corrupted_state


def test_sqlite_candidate_invalid_utf8_is_corruption_for_get_list_retry(
    tmp_path: Path,
) -> None:
    for operation in ("get", "list", "retry"):
        database_path = tmp_path / f"candidate-invalid-utf8-{operation}.sqlite3"
        store = SQLiteMemoryStore(database_path)
        scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-invalid-utf8-{operation}",
        )
        evidence = store.append(
            _make_sqlite_evidence(
                scope=scope,
                idempotency_key=f"candidate-invalid-utf8-{operation}-evidence",
            )
        )
        proposal = _make_sqlite_candidate(
            scope=scope,
            evidence_ids=(evidence.evidence_id,),
            idempotency_key=f"candidate-invalid-utf8-{operation}-request",
        )
        candidate = store.append_candidate(proposal)

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute(
                "UPDATE memory_candidates SET content = CAST(X'80' AS TEXT) "
                "WHERE candidate_id = ?",
                (candidate.candidate_id,),
            )
            stored_content = connection.execute(
                "SELECT typeof(content), hex(content) FROM memory_candidates "
                "WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
        finally:
            connection.close()
        assert stored_content == ("text", "80")

        corrupted_state = _memory_graph_state(database_path)
        assert corrupted_state == ((1, 1, 1, 1), [])
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            if operation == "get":
                store.get_candidate(scope, candidate.candidate_id)
            elif operation == "list":
                store.list_candidates(scope)
            else:
                store.append_candidate(proposal)

        assert type(raised.value) is MemoryPersistenceCorruptionError
        assert str(raised.value) == "SQLite TEXT contains invalid UTF-8"
        assert type(raised.value.__cause__) is UnicodeDecodeError
        assert raised.value.__cause__.object == b"\x80"
        assert raised.value.__cause__.start == 0
        assert raised.value.__cause__.end == 1
        assert _memory_graph_state(database_path) == corrupted_state


def test_sqlite_candidate_loader_rejects_malformed_scalar_and_relation_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "candidate-malformed-loader-rows.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-malformed")
    evidence = store.append(
        _make_sqlite_evidence(
            scope=scope,
            idempotency_key="candidate-malformed-evidence",
        )
    )
    proposal = _make_sqlite_candidate(
        scope=scope,
        evidence_ids=(evidence.evidence_id,),
        idempotency_key="candidate-malformed-request",
    )
    candidate = store.append_candidate(proposal)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
        candidate_row = connection.execute(
            sqlite_store_module._CANDIDATE_SELECT,
            (scope_id, candidate.candidate_id),
        ).fetchone()
    finally:
        connection.close()
    assert candidate_row is not None

    class StaticScalarCursor:
        def __init__(self, row: tuple[object, ...]) -> None:
            self._row = row

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticScalarCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return self._row

    scalar_cases: list[
        tuple[str, tuple[object, ...], str, type[ValueError] | type[TypeError]]
    ] = [
        (
            "wrong-arity",
            tuple(candidate_row[:-1]),
            "candidate row has the wrong field count",
            ValueError,
        )
    ]
    wrong_storage_values = (
        b"candidate-id-is-not-text",
        "canonical-is-not-a-blob",
        b"content-hash-is-not-text",
        b"created-at-is-not-text",
        b"storage-hash-is-not-text",
        b"content-is-not-text",
        b"idempotency-key-is-not-text",
    )
    for index, wrong_value in enumerate(wrong_storage_values):
        changed_row = list(candidate_row)
        changed_row[index] = wrong_value
        scalar_cases.append(
            (
                f"wrong-storage-{index}",
                tuple(changed_row),
                f"candidate row field {index} has the wrong storage class",
                TypeError,
            )
        )

    for case, row, expected_cause, cause_type in scalar_cases:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_candidate(
                StaticScalarCursor(row),  # type: ignore[arg-type]
                scope,
                scope_id,
                candidate.candidate_id,
                proposal.evidence_ids,
            )

        assert str(raised.value) == (
            "stored candidate row failed integrity validation"
        ), case
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert type(raised.value.__cause__) is cause_type, case
        assert str(raised.value.__cause__) == expected_cause, case

    with pytest.raises(MemoryPersistenceCorruptionError) as requested_id_error:
        sqlite_store_module._load_candidate(
            StaticScalarCursor(tuple(candidate_row)),  # type: ignore[arg-type]
            scope,
            scope_id,
            "cand_requested_id_drift",
            proposal.evidence_ids,
        )
    assert str(requested_id_error.value) == (
        "stored candidate row failed integrity validation"
    )
    assert type(requested_id_error.value) is MemoryPersistenceCorruptionError
    assert type(requested_id_error.value.__cause__) is ValueError
    assert str(requested_id_error.value.__cause__) == (
        "loaded candidate ID differs from requested ID"
    )

    class StaticRowsCursor:
        def __init__(self, rows: tuple[tuple[object, ...], ...]) -> None:
            self._rows = rows

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticRowsCursor:
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._rows)

    address_cases = (
        (
            "wrong-arity",
            ((scope_id,),),
            "candidate address row does not contain exactly two values",
        ),
        (
            "boolean-scope",
            ((True, candidate.candidate_id),),
            "candidate address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "overflow-scope",
            ((2**63, candidate.candidate_id),),
            "candidate address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "nontext-candidate",
            ((scope_id, b"candidate-id"),),
            "candidate address contains a non-text identifier",
        ),
    )
    for case, rows, expected_message in address_cases:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_candidate_addresses(
                StaticRowsCursor(rows),  # type: ignore[arg-type]
                {scope_id: scope},
            )
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_message, case
        assert raised.value.__cause__ is None, case

    edge_cases = (
        (
            "wrong-arity",
            ((scope_id, candidate.candidate_id, 0),),
            "candidate evidence row does not contain exactly four values",
        ),
        (
            "boolean-scope",
            ((True, candidate.candidate_id, 0, evidence.evidence_id),),
            "candidate evidence contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "overflow-scope",
            ((2**63, candidate.candidate_id, 0, evidence.evidence_id),),
            "candidate evidence contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "boolean-position",
            ((scope_id, candidate.candidate_id, True, evidence.evidence_id),),
            "candidate evidence position is not a non-negative signed 64-bit integer",
        ),
        (
            "overflow-position",
            ((scope_id, candidate.candidate_id, 2**63, evidence.evidence_id),),
            "candidate evidence position is not a non-negative signed 64-bit integer",
        ),
        (
            "nontext-candidate",
            ((scope_id, b"candidate-id", 0, evidence.evidence_id),),
            "candidate evidence identifiers must be text",
        ),
        (
            "nontext-evidence",
            ((scope_id, candidate.candidate_id, 0, b"evidence-id"),),
            "candidate evidence identifiers must be text",
        ),
    )
    for case, rows, expected_message in edge_cases:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_candidate_edges(
                StaticRowsCursor(rows),  # type: ignore[arg-type]
                ((scope_id, candidate.candidate_id),),
                {(scope_id, evidence.evidence_id): evidence},
            )
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_message, case
        assert raised.value.__cause__ is None, case
    assert _memory_graph_state(database_path) == ((1, 1, 1, 1), [])


def test_sqlite_candidate_snapshot_validates_unrelated_evidence_before_any_candidate_outcome(
    tmp_path: Path,
) -> None:
    for operation in ("missing-get", "list", "get-evidence", "exact-retry"):
        database_path = tmp_path / f"candidate-unrelated-evidence-{operation}.sqlite3"
        store = SQLiteMemoryStore(database_path)
        candidate_scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-unrelated-{operation}",
        )
        unrelated_scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-unrelated-{operation}-foreign",
        )
        candidate_evidence = store.append(
            _make_sqlite_evidence(
                scope=candidate_scope,
                idempotency_key=f"candidate-unrelated-{operation}-evidence",
            )
        )
        unrelated_evidence = store.append(
            _make_sqlite_evidence(
                scope=unrelated_scope,
                payload="unrelated evidence-only scope",
                idempotency_key=f"candidate-unrelated-{operation}-foreign-evidence",
            )
        )
        proposal = _make_sqlite_candidate(
            scope=candidate_scope,
            evidence_ids=(candidate_evidence.evidence_id,),
            idempotency_key=f"candidate-unrelated-{operation}-request",
        )
        candidate = store.append_candidate(proposal)
        moved_evidence_id = (
            f"evd_{'0' * 24}"
            if unrelated_evidence.evidence_id != f"evd_{'0' * 24}"
            else f"evd_{'1' * 24}"
        )

        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
            unrelated_scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (
                    unrelated_scope.tenant_id,
                    unrelated_scope.namespace,
                    unrelated_scope.subject_id,
                ),
            ).fetchone()[0]
            connection.execute(
                "UPDATE memory_evidence SET evidence_id = ? "
                "WHERE scope_id = ? AND evidence_id = ?",
                (
                    moved_evidence_id,
                    unrelated_scope_id,
                    unrelated_evidence.evidence_id,
                ),
            )
            ingest_order = connection.execute(
                "SELECT ingest_order FROM memory_evidence_ingest_orders "
                "WHERE scope_id = ? AND evidence_id = ?",
                (unrelated_scope_id, unrelated_evidence.evidence_id),
            ).fetchone()[0]
            connection.execute(
                "UPDATE memory_evidence_ingest_orders SET evidence_id = ?, "
                "binding_hash = ? WHERE scope_id = ? AND evidence_id = ?",
                (
                    moved_evidence_id,
                    sqlite_backend._evidence_ingest_binding_hash(
                        scope=unrelated_scope,
                        evidence_id=moved_evidence_id,
                        ingest_order=ingest_order,
                    ),
                    unrelated_scope_id,
                    unrelated_evidence.evidence_id,
                ),
            )
            changed_row = connection.execute(
                "SELECT evidence_id, canonical, content_hash "
                "FROM memory_evidence WHERE scope_id = ?",
                (unrelated_scope_id,),
            ).fetchone()
        finally:
            connection.close()
        assert changed_row == (
            moved_evidence_id,
            unrelated_evidence.event.canonical_bytes(),
            unrelated_evidence.content_hash,
        )

        corrupted_state = _memory_graph_state(database_path)
        assert corrupted_state == ((2, 2, 1, 1), [])
        exact_retry = CandidateProposal(
            scope=proposal.scope,
            content=proposal.content,
            evidence_ids=proposal.evidence_ids,
            idempotency_key=proposal.idempotency_key,
        )
        assert exact_retry == proposal
        assert exact_retry is not proposal

        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            if operation == "missing-get":
                store.get_candidate(candidate_scope, "cand_missing")
            elif operation == "list":
                store.list_candidates(candidate_scope)
            elif operation == "get-evidence":
                store.get_candidate_evidence(
                    candidate_scope,
                    candidate.candidate_id,
                )
            else:
                store.append_candidate(exact_retry)

        assert type(raised.value) is MemoryPersistenceCorruptionError
        assert str(raised.value) == ("stored evidence row failed integrity validation")
        assert type(raised.value.__cause__) is ValueError
        assert str(raised.value.__cause__) == (
            "evidence ID disagrees with its content hash"
        )
        assert _memory_graph_state(database_path) == corrupted_state


def test_sqlite_candidate_snapshot_rejects_duplicate_addresses_edges_and_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_address_sql = _normalize_sql(
        "SELECT scope_id, candidate_id FROM memory_candidates"
    )
    candidate_edge_sql = _normalize_sql(
        "SELECT scope_id, candidate_id, position, evidence_id "
        "FROM memory_candidate_evidence"
    )
    candidate_scalar_sql = _normalize_sql(sqlite_store_module._CANDIDATE_SELECT)

    @dataclass(slots=True)
    class CandidateSnapshotInjection:
        case: str
        scope_id: int
        candidate_id: str
        first_evidence_id: str
        second_evidence_id: str
        virtual_candidate_id: str | None = None
        virtual_candidate_row: tuple[object, ...] | None = None
        hits: dict[str, int] = field(
            default_factory=lambda: {
                "candidate-address": 0,
                "candidate-edge": 0,
                "candidate-scalar": 0,
            }
        )

        def project_rows(
            self,
            normalized_sql: str,
            rows: list[tuple[object, ...]],
        ) -> list[tuple[object, ...]]:
            if normalized_sql == candidate_address_sql:
                if self.case == "candidate-address":
                    self.hits["candidate-address"] += 1
                    return [*rows, (self.scope_id, self.candidate_id)]
                if self.case == "scoped-idempotency":
                    assert self.virtual_candidate_id is not None
                    self.hits["candidate-address"] += 1
                    return [
                        *rows,
                        (self.scope_id, self.virtual_candidate_id),
                    ]
            if normalized_sql == candidate_edge_sql:
                if self.case == "edge-position":
                    self.hits["candidate-edge"] += 1
                    return [
                        *rows,
                        (
                            self.scope_id,
                            self.candidate_id,
                            0,
                            self.second_evidence_id,
                        ),
                    ]
                if self.case == "edge-evidence":
                    self.hits["candidate-edge"] += 1
                    return [
                        *rows,
                        (
                            self.scope_id,
                            self.candidate_id,
                            1,
                            self.first_evidence_id,
                        ),
                    ]
                if self.case == "scoped-idempotency":
                    assert self.virtual_candidate_id is not None
                    self.hits["candidate-edge"] += 1
                    return [
                        *rows,
                        (
                            self.scope_id,
                            self.virtual_candidate_id,
                            0,
                            self.second_evidence_id,
                        ),
                    ]
            return rows

        def project_row(
            self,
            normalized_sql: str,
            parameters: object,
            row: tuple[object, ...] | None,
        ) -> tuple[object, ...] | None:
            if (
                self.case == "scoped-idempotency"
                and normalized_sql == candidate_scalar_sql
                and parameters == (self.scope_id, self.virtual_candidate_id)
            ):
                assert row is None
                assert self.virtual_candidate_row is not None
                self.hits["candidate-scalar"] += 1
                return self.virtual_candidate_row
            return row

    class CandidateSnapshotCursor:
        def __init__(
            self,
            real_cursor: sqlite3.Cursor,
            injection: CandidateSnapshotInjection,
        ) -> None:
            self._real_cursor = real_cursor
            self._injection = injection
            self._normalized_sql = ""
            self._parameters: object = ()

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> CandidateSnapshotCursor:
            self._normalized_sql = _normalize_sql(sql)
            self._parameters = parameters
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            return self._injection.project_rows(self._normalized_sql, rows)

        def fetchone(self) -> tuple[object, ...] | None:
            row = self._real_cursor.fetchone()
            return self._injection.project_row(
                self._normalized_sql,
                self._parameters,
                None if row is None else tuple(row),
            )

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class CandidateSnapshotConnection:
        def __init__(
            self,
            real_connection: sqlite3.Connection,
            injection: CandidateSnapshotInjection,
        ) -> None:
            self._real_connection = real_connection
            self._injection = injection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> CandidateSnapshotCursor:
            return CandidateSnapshotCursor(
                self._real_connection.cursor(*args, **kwargs),
                self._injection,
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    real_connect = sqlite_backend._connect
    for case in (
        "candidate-address",
        "edge-position",
        "edge-evidence",
        "scoped-idempotency",
    ):
        database_path = tmp_path / f"candidate-duplicate-{case}.sqlite3"
        store = SQLiteMemoryStore(database_path)
        scope = MemoryScope(
            "tenant-1",
            "assistant-memory",
            f"candidate-duplicate-{case}",
        )
        first_evidence = store.append(
            _make_sqlite_evidence(
                scope=scope,
                sequence_no=0,
                payload=f"first evidence for {case}",
                idempotency_key=f"candidate-duplicate-{case}-evidence-1",
            )
        )
        second_evidence = store.append(
            _make_sqlite_evidence(
                scope=scope,
                sequence_no=1,
                payload=f"second evidence for {case}",
                idempotency_key=f"candidate-duplicate-{case}-evidence-2",
            )
        )
        proposal = _make_sqlite_candidate(
            scope=scope,
            content=f"candidate duplicate owner for {case}",
            evidence_ids=(first_evidence.evidence_id,),
            idempotency_key=f"candidate-duplicate-{case}-request",
        )
        candidate = store.append_candidate(proposal)
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            scope_id = connection.execute(
                "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
                "AND namespace = ? AND subject_id = ?",
                (scope.tenant_id, scope.namespace, scope.subject_id),
            ).fetchone()[0]
        finally:
            connection.close()

        injection = CandidateSnapshotInjection(
            case=case,
            scope_id=scope_id,
            candidate_id=candidate.candidate_id,
            first_evidence_id=first_evidence.evidence_id,
            second_evidence_id=second_evidence.evidence_id,
        )
        if case == "scoped-idempotency":
            virtual_proposal = _make_sqlite_candidate(
                scope=scope,
                content="fully coherent virtual duplicate idempotency candidate",
                evidence_ids=(second_evidence.evidence_id,),
                idempotency_key=proposal.idempotency_key,
            )
            virtual_canonical = virtual_proposal.canonical_bytes()
            virtual_content_hash = hashlib.sha256(virtual_canonical).hexdigest()
            virtual_candidate_id = f"cand_{virtual_content_hash[:24]}"
            assert virtual_candidate_id != candidate.candidate_id
            virtual_created_at = datetime(
                2026,
                7,
                8,
                3,
                4,
                5,
                678000,
                tzinfo=UTC,
            ).isoformat()
            virtual_storage_hash = sqlite_store_module._record_storage_hash(
                record_kind="candidate",
                scope=scope,
                record_id=virtual_candidate_id,
                content_hash=virtual_content_hash,
                created_at_text=virtual_created_at,
            )
            injection.virtual_candidate_id = virtual_candidate_id
            injection.virtual_candidate_row = (
                virtual_candidate_id,
                virtual_canonical,
                virtual_content_hash,
                virtual_created_at,
                virtual_storage_hash,
                virtual_proposal.content,
                virtual_proposal.idempotency_key,
            )

        baseline_state = _memory_graph_state(database_path)
        assert baseline_state == ((1, 2, 1, 1), [])

        def connect(
            path: str,
            selected_injection: CandidateSnapshotInjection = injection,
        ) -> CandidateSnapshotConnection:
            return CandidateSnapshotConnection(
                real_connect(path),
                selected_injection,
            )

        monkeypatch.setattr(sqlite_backend, "_connect", connect)
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            if case == "candidate-address":
                store.get_candidate(scope, "cand_missing")
            elif case in {"edge-position", "edge-evidence"}:
                store.list_candidates(scope)
            else:
                store.append_candidate(
                    CandidateProposal(
                        scope=proposal.scope,
                        content=proposal.content,
                        evidence_ids=proposal.evidence_ids,
                        idempotency_key=proposal.idempotency_key,
                    )
                )
        monkeypatch.undo()

        expected_messages = {
            "candidate-address": "candidate address appears multiple times",
            "edge-position": "candidate evidence position appears multiple times",
            "edge-evidence": "candidate contains the same evidence multiple times",
            "scoped-idempotency": (
                "candidate idempotency key appears multiple times in one scope"
            ),
        }
        expected_hits = {
            "candidate-address": {
                "candidate-address": 1,
                "candidate-edge": 0,
                "candidate-scalar": 0,
            },
            "edge-position": {
                "candidate-address": 0,
                "candidate-edge": 1,
                "candidate-scalar": 0,
            },
            "edge-evidence": {
                "candidate-address": 0,
                "candidate-edge": 1,
                "candidate-scalar": 0,
            },
            "scoped-idempotency": {
                "candidate-address": 1,
                "candidate-edge": 1,
                "candidate-scalar": 1,
            },
        }
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_messages[case], case
        assert raised.value.__cause__ is None, case
        assert injection.hits == expected_hits[case], case
        assert _memory_graph_state(database_path) == baseline_state, case
        assert store.list_candidates(scope) == (candidate,), case


def test_sqlite_candidate_edge_failure_rolls_back_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "candidate-edge-failure.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-edge-failure")
    evidence = tuple(
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                sequence_no=index,
                payload=f"candidate edge failure evidence {index}",
                idempotency_key=f"candidate-edge-failure-evidence-{index}",
            )
        )
        for index in range(3)
    )
    proposal = _make_sqlite_candidate(
        scope=scope,
        content="candidate whose second edge write fails",
        evidence_ids=(
            evidence[2].evidence_id,
            evidence[0].evidence_id,
            evidence[1].evidence_id,
        ),
        idempotency_key="candidate-edge-failure-request",
    )
    baseline_state = _memory_graph_state(database_path)
    assert baseline_state == ((1, 3, 0, 0), [])
    injected = sqlite3.OperationalError("injected second candidate edge failure")
    injected.sqlite_errorcode = sqlite3.SQLITE_IOERR
    plan = _SQLiteFailurePlan(
        after_statement="INSERT INTO memory_candidate_evidence",
        after_occurrence=2,
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        store.append_candidate(proposal)

    assert type(raised.value) is MemoryPersistenceError
    assert raised.value.__cause__ is injected
    edge_insert_events = [
        index
        for index, event in enumerate(plan.events)
        if event.startswith("executed:INSERT INTO MEMORY_CANDIDATE_EVIDENCE")
    ]
    assert len(edge_insert_events) == 2
    failure_index = next(
        index
        for index, event in enumerate(plan.events)
        if event.startswith("fail-after:INSERT INTO MEMORY_CANDIDATE_EVIDENCE")
    )
    rollback_index = plan.events.index("executed:ROLLBACK")
    close_index = plan.events.index("executed:CLOSE")
    assert edge_insert_events[-1] < failure_index < rollback_index < close_index
    assert "attempt:COMMIT" not in plan.events
    assert "executed:COMMIT" not in plan.events
    assert _memory_graph_state(database_path) == baseline_state

    monkeypatch.undo()
    recovered = store.append_candidate(proposal)

    assert recovered.proposal == proposal
    assert store.get_candidate_evidence(scope, recovered.candidate_id) == (
        evidence[2],
        evidence[0],
        evidence[1],
    )
    assert _memory_graph_state(database_path) == ((1, 3, 1, 3), [])


def test_sqlite_candidate_post_insert_graph_readback_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "candidate-post-insert-readback.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "candidate-readback")
    evidence = tuple(
        store.append(
            _make_sqlite_evidence(
                scope=scope,
                sequence_no=index,
                payload=f"candidate readback evidence {index}",
                idempotency_key=f"candidate-readback-evidence-{index}",
            )
        )
        for index in range(4)
    )
    proposal = _make_sqlite_candidate(
        scope=scope,
        content="candidate whose final edge is tampered before readback",
        evidence_ids=(
            evidence[2].evidence_id,
            evidence[0].evidence_id,
            evidence[1].evidence_id,
        ),
        idempotency_key="candidate-readback-request",
    )
    expected_candidate_id = (
        f"cand_{hashlib.sha256(proposal.canonical_bytes()).hexdigest()[:24]}"
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
    finally:
        connection.close()
    baseline_state = _memory_graph_state(database_path)
    assert baseline_state == ((1, 4, 0, 0), [])
    edge_scan_sql = _normalize_sql(
        "SELECT scope_id, candidate_id, position, evidence_id "
        "FROM memory_candidate_evidence"
    )
    candidate_scalar_sql = _normalize_sql(sqlite_store_module._CANDIDATE_SELECT)
    probe_hits = {"tamper": 0, "edge-scan": 0, "candidate-scalar": 0}
    executed_sql: list[str] = []
    real_connect = sqlite_backend._connect

    class TamperFinalCandidateEdgeCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> TamperFinalCandidateEdgeCursor:
            normalized = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            executed_sql.append(normalized)
            if normalized.startswith("INSERT INTO MEMORY_CANDIDATE_EVIDENCE"):
                assert isinstance(parameters, tuple)
                assert len(parameters) == 4
                inserted_scope_id, candidate_id, position, evidence_id = parameters
                if position == len(proposal.evidence_ids) - 1:
                    assert inserted_scope_id == scope_id
                    assert candidate_id == expected_candidate_id
                    assert evidence_id == proposal.evidence_ids[-1]
                    self._real_cursor.execute(
                        "UPDATE memory_candidate_evidence SET evidence_id = ? "
                        "WHERE scope_id = ? AND candidate_id = ? AND position = ?",
                        (
                            evidence[3].evidence_id,
                            inserted_scope_id,
                            candidate_id,
                            position,
                        ),
                    )
                    assert self._real_cursor.rowcount == 1
                    probe_hits["tamper"] += 1
            elif probe_hits["tamper"]:
                if normalized == edge_scan_sql:
                    probe_hits["edge-scan"] += 1
                elif normalized == candidate_scalar_sql and parameters == (
                    scope_id,
                    expected_candidate_id,
                ):
                    probe_hits["candidate-scalar"] += 1
            return self

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class TamperFinalCandidateEdgeConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> TamperFinalCandidateEdgeCursor:
            return TamperFinalCandidateEdgeCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> TamperFinalCandidateEdgeConnection:
        return TamperFinalCandidateEdgeConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.append_candidate(proposal)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert str(raised.value) == "stored candidate row failed integrity validation"
    assert type(raised.value.__cause__) is ValueError
    assert str(raised.value.__cause__) == (
        "canonical candidate bytes disagree with projections"
    )
    assert probe_hits == {"tamper": 1, "edge-scan": 1, "candidate-scalar": 1}
    assert "ROLLBACK" in executed_sql
    assert "COMMIT" not in executed_sql
    assert _memory_graph_state(database_path) == baseline_state

    monkeypatch.undo()
    recovered = store.append_candidate(proposal)

    assert recovered.candidate_id == expected_candidate_id
    assert store.get_candidate_evidence(scope, recovered.candidate_id) == (
        evidence[2],
        evidence[0],
        evidence[1],
    )
    assert evidence[3] not in store.get_candidate_evidence(
        scope,
        recovered.candidate_id,
    )
    assert _memory_graph_state(database_path) == ((1, 4, 1, 3), [])


def test_sqlite_revision_insert_failure_rolls_back_and_candidate_is_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-insert-failure.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-insert-failure")
    candidate, evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=0,
        key="revision-insert-failure",
    )
    proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidate.candidate_id,
        idempotency_key="revision-insert-failure-request",
    )
    expected_revision_id = (
        f"rev_{hashlib.sha256(proposal.canonical_bytes()).hexdigest()[:24]}"
    )
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((1, 1, 1, 1, 0), [])
    injected = sqlite3.OperationalError("injected after real revision insert")
    injected.sqlite_errorcode = sqlite3.SQLITE_IOERR
    plan = _SQLiteFailurePlan(
        after_statement="INSERT INTO memory_revisions",
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        store.append_revision(proposal)

    assert type(raised.value) is MemoryPersistenceError
    assert raised.value.__cause__ is injected
    insert_events = [
        index
        for index, event in enumerate(plan.events)
        if event.startswith("executed:INSERT INTO MEMORY_REVISIONS")
    ]
    assert len(insert_events) == 1
    failure_index = next(
        index
        for index, event in enumerate(plan.events)
        if event.startswith("fail-after:INSERT INTO MEMORY_REVISIONS")
    )
    rollback_index = plan.events.index("executed:ROLLBACK")
    close_index = plan.events.index("executed:CLOSE")
    assert insert_events[0] < failure_index < rollback_index < close_index
    assert "attempt:COMMIT" not in plan.events
    assert "executed:COMMIT" not in plan.events
    assert _revision_graph_state(database_path) == baseline_state
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        candidate_row = connection.execute(
            "SELECT candidate_id FROM memory_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()
        evidence_row = connection.execute(
            "SELECT evidence_id FROM memory_evidence WHERE evidence_id = ?",
            (evidence.evidence_id,),
        ).fetchone()
        revision_rows = connection.execute(
            "SELECT revision_id FROM memory_revisions "
            "WHERE revision_id = ? OR candidate_id = ? OR idempotency_key = ?",
            (
                expected_revision_id,
                candidate.candidate_id,
                proposal.idempotency_key,
            ),
        ).fetchall()
    finally:
        connection.close()
    assert candidate_row == (candidate.candidate_id,)
    assert evidence_row == (evidence.evidence_id,)
    assert revision_rows == []

    monkeypatch.undo()
    recovered = store.append_revision(proposal)

    assert recovered.revision_id == expected_revision_id
    assert recovered.proposal == proposal
    assert store.get_candidate(scope, candidate.candidate_id) == candidate
    assert store.get_candidate_evidence(scope, candidate.candidate_id) == (evidence,)
    assert (
        SQLiteMemoryStore(database_path).get_revision(
            scope,
            expected_revision_id,
        )
        == recovered
    )
    assert _revision_graph_state(database_path) == ((1, 1, 1, 1, 1), [])


def test_sqlite_revision_post_insert_global_readback_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-post-insert-readback.sqlite3"
    store = SQLiteMemoryStore(database_path)
    unrelated_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-readback-unrelated",
    )
    target_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-readback-target",
    )
    unrelated_candidate, _unrelated_evidence = _append_sqlite_revision_candidate(
        store,
        unrelated_scope,
        index=0,
        key="revision-readback-unrelated",
    )
    unrelated_revision = store.append_revision(
        _make_sqlite_revision(
            scope=unrelated_scope,
            candidate_id=unrelated_candidate.candidate_id,
            idempotency_key="revision-readback-unrelated-root",
        )
    )
    target_candidate, _target_evidence = _append_sqlite_revision_candidate(
        store,
        target_scope,
        index=0,
        key="revision-readback-target",
    )
    target_proposal = _make_sqlite_revision(
        scope=target_scope,
        candidate_id=target_candidate.candidate_id,
        idempotency_key="revision-readback-target-root",
    )
    target_revision_id = (
        f"rev_{hashlib.sha256(target_proposal.canonical_bytes()).hexdigest()[:24]}"
    )
    assert unrelated_revision.generation == 0
    tampered_generation = 1
    tampered_storage_hash = sqlite_store_module._record_storage_hash(
        record_kind="revision",
        scope=unrelated_scope,
        record_id=unrelated_revision.revision_id,
        content_hash=unrelated_revision.content_hash,
        created_at_text=unrelated_revision.created_at.isoformat(),
        memory_id=unrelated_revision.memory_id,
        generation=tampered_generation,
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_rows = connection.execute(
            "SELECT scope_id, subject_id FROM memory_scopes"
        ).fetchall()
        original_unrelated_row = connection.execute(
            "SELECT generation, storage_hash FROM memory_revisions "
            "WHERE revision_id = ?",
            (unrelated_revision.revision_id,),
        ).fetchone()
    finally:
        connection.close()
    scope_id_by_subject = {subject_id: scope_id for scope_id, subject_id in scope_rows}
    unrelated_scope_id = scope_id_by_subject[unrelated_scope.subject_id]
    target_scope_id = scope_id_by_subject[target_scope.subject_id]
    assert original_unrelated_row == (
        unrelated_revision.generation,
        sqlite_store_module._record_storage_hash(
            record_kind="revision",
            scope=unrelated_scope,
            record_id=unrelated_revision.revision_id,
            content_hash=unrelated_revision.content_hash,
            created_at_text=unrelated_revision.created_at.isoformat(),
            memory_id=unrelated_revision.memory_id,
            generation=unrelated_revision.generation,
        ),
    )
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((2, 2, 2, 2, 1), [])
    revision_address_sql = _normalize_sql(
        "SELECT scope_id, revision_id FROM memory_revisions"
    )
    revision_scalar_sql = _normalize_sql(sqlite_store_module._REVISION_SELECT)
    probe_hits = {
        "insert": 0,
        "tamper": 0,
        "address-scan": 0,
        "unrelated-scalar": 0,
        "topology": 0,
    }
    events: list[str] = []
    real_connect = sqlite_backend._connect
    real_validate_revision_topology = sqlite_store_module._validate_revision_topology

    class TamperUnrelatedRevisionCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> TamperUnrelatedRevisionCursor:
            normalized = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            if normalized.startswith("INSERT INTO MEMORY_REVISIONS"):
                assert isinstance(parameters, tuple)
                assert len(parameters) == 12
                assert parameters[0] == target_scope_id
                assert parameters[1] == target_revision_id
                assert parameters[6] == target_candidate.candidate_id
                probe_hits["insert"] += 1
                events.append("insert")
                assert probe_hits["insert"] == 1
                self._real_cursor.execute(
                    "UPDATE memory_revisions "
                    "SET generation = ?, storage_hash = ? "
                    "WHERE scope_id = ? AND revision_id = ?",
                    (
                        tampered_generation,
                        tampered_storage_hash,
                        unrelated_scope_id,
                        unrelated_revision.revision_id,
                    ),
                )
                assert self._real_cursor.rowcount == 1
                self._real_cursor.execute("PRAGMA foreign_key_check")
                assert self._real_cursor.fetchall() == []
                probe_hits["tamper"] += 1
                events.append("tamper")
            elif probe_hits["tamper"]:
                if normalized == revision_address_sql:
                    probe_hits["address-scan"] += 1
                    events.append("address-scan")
                elif normalized == revision_scalar_sql and parameters == (
                    unrelated_scope_id,
                    unrelated_revision.revision_id,
                ):
                    probe_hits["unrelated-scalar"] += 1
                    events.append("unrelated-scalar")
                elif normalized == "ROLLBACK":
                    events.append("rollback")
                elif normalized == "COMMIT":
                    events.append("commit")
            return self

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class TamperUnrelatedRevisionConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> TamperUnrelatedRevisionCursor:
            return TamperUnrelatedRevisionCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def close(self) -> None:
            self._real_connection.close()
            events.append("close")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> TamperUnrelatedRevisionConnection:
        return TamperUnrelatedRevisionConnection(real_connect(path))

    def observe_revision_topology(
        revision_by_address: dict[tuple[int, str], MemoryRevision],
        parent_by_address: dict[tuple[int, str], tuple[int, str] | None],
    ) -> None:
        if probe_hits["tamper"]:
            probe_hits["topology"] += 1
            events.append("topology")
        real_validate_revision_topology(revision_by_address, parent_by_address)

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    monkeypatch.setattr(
        sqlite_store_module,
        "_validate_revision_topology",
        observe_revision_topology,
    )

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.append_revision(target_proposal)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert str(raised.value) == "ADD revision generation is not zero"
    assert raised.value.__cause__ is None
    assert probe_hits == {
        "insert": 1,
        "tamper": 1,
        "address-scan": 1,
        "unrelated-scalar": 1,
        "topology": 1,
    }
    assert events == [
        "insert",
        "tamper",
        "address-scan",
        "unrelated-scalar",
        "topology",
        "rollback",
        "close",
    ]
    assert "commit" not in events
    assert _revision_graph_state(database_path) == baseline_state
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        restored_unrelated_row = connection.execute(
            "SELECT generation, storage_hash FROM memory_revisions "
            "WHERE scope_id = ? AND revision_id = ?",
            (unrelated_scope_id, unrelated_revision.revision_id),
        ).fetchone()
        target_rows = connection.execute(
            "SELECT revision_id FROM memory_revisions "
            "WHERE scope_id = ? AND (revision_id = ? OR candidate_id = ? "
            "OR idempotency_key = ?)",
            (
                target_scope_id,
                target_revision_id,
                target_candidate.candidate_id,
                target_proposal.idempotency_key,
            ),
        ).fetchall()
    finally:
        connection.close()
    assert restored_unrelated_row == original_unrelated_row
    assert target_rows == []

    monkeypatch.undo()
    recovered = store.append_revision(target_proposal)

    assert recovered.revision_id == target_revision_id
    assert (
        store.get_revision(
            unrelated_scope,
            unrelated_revision.revision_id,
        )
        == unrelated_revision
    )
    assert (
        SQLiteMemoryStore(database_path).get_revision(
            target_scope,
            target_revision_id,
        )
        == recovered
    )
    assert _revision_graph_state(database_path) == ((2, 2, 2, 2, 2), [])


def test_sqlite_revision_reopen_retry_preserves_root_child_and_sibling_topology(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-topology.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-topology")
    candidate_evidence = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"topology-{index}",
        )
        for index in range(4)
    )
    candidates = tuple(item[0] for item in candidate_evidence)
    evidence = tuple(item[1] for item in candidate_evidence)
    root_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[0].candidate_id,
        idempotency_key="topology-root",
    )
    root = store.append_revision(root_proposal)
    refine_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[1].candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=root.revision_id,
        idempotency_key="topology-refine",
    )
    contradict_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[2].candidate_id,
        operation=RevisionOperation.CONTRADICT,
        parent_revision_id=root.revision_id,
        idempotency_key="topology-contradict",
    )
    refine = store.append_revision(refine_proposal)
    contradict = store.append_revision(contradict_proposal)
    supersede_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[3].candidate_id,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=refine.revision_id,
        idempotency_key="topology-supersede",
    )
    supersede = store.append_revision(supersede_proposal)
    originals = (root, refine, contradict, supersede)
    proposals = (
        root_proposal,
        refine_proposal,
        contradict_proposal,
        supersede_proposal,
    )

    for revision, proposal in zip(originals, proposals, strict=True):
        expected_hash = hashlib.sha256(proposal.canonical_bytes()).hexdigest()
        assert revision.revision_id == f"rev_{expected_hash[:24]}"
        assert revision.content_hash == expected_hash
        assert revision.proposal == proposal
        assert revision.proposal is not proposal
        assert revision.created_at.tzinfo is UTC
    assert root.memory_id == f"mem_{root.content_hash[:24]}"
    assert root.generation == 0
    assert root.proposal.operation is RevisionOperation.ADD
    assert root.proposal.parent_revision_id is None
    assert (
        refine.memory_id
        == contradict.memory_id
        == supersede.memory_id
        == (root.memory_id)
    )
    assert refine.generation == contradict.generation == 1
    assert supersede.generation == 2
    assert refine.proposal.parent_revision_id == root.revision_id
    assert contradict.proposal.parent_revision_id == root.revision_id
    assert supersede.proposal.parent_revision_id == refine.revision_id
    assert _revision_graph_state(database_path) == ((1, 4, 4, 4, 4), [])

    del store
    reopened = SQLiteMemoryStore(database_path)
    loaded = tuple(
        reopened.get_revision(scope, revision.revision_id) for revision in originals
    )
    retries = tuple(
        reopened.append_revision(
            RevisionProposal(
                scope=proposal.scope,
                candidate_id=proposal.candidate_id,
                operation=proposal.operation,
                parent_revision_id=proposal.parent_revision_id,
                idempotency_key=proposal.idempotency_key,
            )
        )
        for proposal in proposals
    )

    assert loaded == retries == originals
    for original, persisted, retry in zip(originals, loaded, retries, strict=True):
        assert persisted is not original
        assert retry is not original
        assert persisted.revision_id == retry.revision_id == original.revision_id
        assert persisted.memory_id == retry.memory_id == original.memory_id
        assert persisted.generation == retry.generation == original.generation
        assert persisted.proposal == retry.proposal == original.proposal
        assert persisted.content_hash == retry.content_hash == original.content_hash
        assert persisted.created_at == retry.created_at == original.created_at
    assert reopened.list_revisions(scope) == tuple(
        sorted(
            originals,
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    assert reopened.list_revisions(scope, memory_id=root.memory_id) == tuple(
        sorted(
            originals,
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    for index, revision in enumerate(originals):
        candidate = reopened.get_candidate(scope, revision.proposal.candidate_id)
        assert candidate == candidates[index]
        assert reopened.get_candidate_evidence(scope, candidate.candidate_id) == (
            evidence[index],
        )

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        rows = connection.execute(
            "SELECT revision_id, candidate_id, memory_id, generation, operation, "
            "parent_revision_id, idempotency_key FROM memory_revisions"
        ).fetchall()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    assert len(rows) == 4
    assert len({row[0] for row in rows}) == 4
    assert len({row[1] for row in rows}) == 4
    assert foreign_keys == []
    assert _revision_graph_state(database_path) == ((1, 4, 4, 4, 4), [])


def test_sqlite_revision_queries_sort_filter_and_snapshot_public_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-queries.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-query-user")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-query-foreign-user",
    )
    candidate_evidence = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"query-{index}",
        )
        for index in range(6)
    )
    candidates = tuple(item[0] for item in candidate_evidence)
    first_root_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[0].candidate_id,
        idempotency_key="query-root-1",
    )
    second_root_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[1].candidate_id,
        idempotency_key="query-root-2",
    )

    def proposal_hash(proposal: RevisionProposal) -> str:
        return hashlib.sha256(proposal.canonical_bytes()).hexdigest()

    root_proposals = tuple(
        sorted(
            (first_root_proposal, second_root_proposal),
            key=proposal_hash,
        )
    )
    lower_root_proposal, higher_root_proposal = root_proposals
    higher_root = store.append_revision(higher_root_proposal)
    lower_root = store.append_revision(lower_root_proposal)
    assert lower_root.memory_id < higher_root.memory_id
    refine_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[2].candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=lower_root.revision_id,
        idempotency_key="query-refine",
    )
    contradict_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[3].candidate_id,
        operation=RevisionOperation.CONTRADICT,
        parent_revision_id=lower_root.revision_id,
        idempotency_key="query-contradict",
    )
    sibling_proposals = tuple(
        sorted(
            (refine_proposal, contradict_proposal),
            key=proposal_hash,
            reverse=True,
        )
    )
    sibling_by_operation = {
        proposal.operation: store.append_revision(proposal)
        for proposal in sibling_proposals
    }
    refine = sibling_by_operation[RevisionOperation.REFINE]
    contradict = sibling_by_operation[RevisionOperation.CONTRADICT]
    grandchild_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[4].candidate_id,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=refine.revision_id,
        idempotency_key="query-grandchild",
    )
    grandchild = store.append_revision(grandchild_proposal)
    initial_scope_revisions = (
        higher_root,
        lower_root,
        refine,
        contradict,
        grandchild,
    )
    expected = tuple(
        sorted(
            initial_scope_revisions,
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    expected_revision_ids = tuple(item.revision_id for item in expected)
    assert tuple(item.revision_id for item in initial_scope_revisions) != tuple(
        expected_revision_ids
    )

    foreign_candidate, _foreign_evidence = _append_sqlite_revision_candidate(
        store,
        foreign_scope,
        index=0,
        key="query-foreign",
    )
    foreign = store.append_revision(
        _make_sqlite_revision(
            scope=foreign_scope,
            candidate_id=foreign_candidate.candidate_id,
            idempotency_key="query-foreign-revision",
        )
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
    finally:
        connection.close()
    revision_address_sql = _normalize_sql(
        "SELECT scope_id, revision_id FROM memory_revisions"
    )
    presented_revision_addresses: list[tuple[tuple[object, ...], ...]] = []
    real_connect = sqlite_backend._connect

    class ReverseRevisionAddressCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> ReverseRevisionAddressCursor:
            self._last_sql = _normalize_sql(sql)
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            if self._last_sql == revision_address_sql:
                rows_by_revision_id = {
                    row[1]: row for row in rows if row[0] == scope_id
                }
                foreign_rows = [row for row in rows if row[0] != scope_id]
                later_scope_rows = [
                    row
                    for row in rows
                    if row[0] == scope_id and row[1] not in expected_revision_ids
                ]
                rows = [
                    *foreign_rows,
                    *(
                        rows_by_revision_id[revision_id]
                        for revision_id in reversed(expected_revision_ids)
                    ),
                    *later_scope_rows,
                ]
                presented_revision_addresses.append(tuple(rows))
            return rows

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class ReverseRevisionAddressConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> ReverseRevisionAddressCursor:
            return ReverseRevisionAddressCursor(
                self._real_connection.cursor(*args, **kwargs)
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> ReverseRevisionAddressConnection:
        return ReverseRevisionAddressConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    listed = store.list_revisions(scope)

    assert listed == expected
    assert len(presented_revision_addresses) == 1
    presented_scope_ids = tuple(
        row[1] for row in presented_revision_addresses[0] if row[0] == scope_id
    )
    assert presented_scope_ids == tuple(reversed(expected_revision_ids))
    revision_id_probe = _SnapshotProbeStr(lower_root.revision_id)
    memory_id_probe = _SnapshotProbeStr(lower_root.memory_id)
    assert store.get_revision(scope, revision_id_probe) == lower_root
    lower_memory = tuple(
        item for item in expected if item.memory_id == lower_root.memory_id
    )
    assert store.list_revisions(scope, memory_id=memory_id_probe) == lower_memory
    assert revision_id_probe.override_calls == 0
    assert memory_id_probe.override_calls == 0
    assert lower_memory == tuple(
        sorted(
            (lower_root, refine, contradict, grandchild),
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    assert store.list_revisions(scope, memory_id="mem_missing") == ()
    for blank in ("", " \t", "\x00"):
        with pytest.raises(RevisionNotFoundError) as raised:
            store.get_revision(scope, blank)
        assert type(raised.value) is RevisionNotFoundError
        assert str(raised.value) == (f"revision {str.__str__(blank)!r} was not found")
        assert store.list_revisions(scope, memory_id=blank) == ()

    with pytest.raises(RevisionNotFoundError) as foreign_error:
        store.get_revision(foreign_scope, lower_root.revision_id)
    with pytest.raises(RevisionNotFoundError) as missing_error:
        store.get_revision(
            MemoryScope("tenant-1", "assistant-memory", "missing-revision-user"),
            lower_root.revision_id,
        )
    assert type(foreign_error.value) is RevisionNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)
    assert store.list_revisions(foreign_scope) == (foreign,)
    assert (
        store.list_revisions(
            foreign_scope,
            memory_id=lower_root.memory_id,
        )
        == ()
    )
    assert (
        store.list_revisions(
            MemoryScope("tenant-1", "assistant-memory", "missing-revision-user"),
            memory_id=lower_root.memory_id,
        )
        == ()
    )

    snapshot = listed
    later_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[5].candidate_id,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=contradict.revision_id,
        idempotency_key="query-later",
    )
    later = store.append_revision(later_proposal)
    assert snapshot == expected
    assert store.list_revisions(scope) == tuple(
        sorted(
            (*expected, later),
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    assert _revision_graph_state(database_path) == ((2, 7, 7, 7, 7), [])

    monkeypatch.undo()

    def transaction_must_not_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validation reached SQLite I/O")

    monkeypatch.setattr(
        sqlite_store_module,
        "_read_transaction",
        transaction_must_not_start,
    )
    monkeypatch.setattr(
        sqlite_store_module,
        "_write_transaction",
        transaction_must_not_start,
    )
    proposal_subclass = _SQLiteRevisionProposalSubclass(
        lower_root_proposal.scope,
        lower_root_proposal.candidate_id,
        lower_root_proposal.operation,
        lower_root_proposal.parent_revision_id,
        lower_root_proposal.idempotency_key,
    )
    scope_subclass = _SQLiteMemoryScopeSubclass(
        scope.tenant_id,
        scope.namespace,
        scope.subject_id,
    )

    with pytest.raises(TypeError, match="proposal must be a RevisionProposal"):
        store.append_revision(proposal_subclass)
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.get_revision(scope_subclass, lower_root.revision_id)
    with pytest.raises(TypeError, match="scope must be a MemoryScope"):
        store.list_revisions(scope_subclass)
    for invalid_id in (None, 7):
        with pytest.raises(TypeError, match="revision_id must be a string"):
            store.get_revision(scope, invalid_id)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="revision_id must be valid UTF-8"):
        store.get_revision(scope, "\ud800")
    for invalid_memory_id in (7, b"memory"):
        with pytest.raises(TypeError, match="memory_id must be a string"):
            store.list_revisions(  # type: ignore[arg-type]
                scope,
                memory_id=invalid_memory_id,
            )
    with pytest.raises(ValueError, match="memory_id must be valid UTF-8"):
        store.list_revisions(scope, memory_id="\ud800")


def test_sqlite_revision_error_precedence_and_relationship_failures_are_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-precedence.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-precedence")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-precedence-foreign",
    )
    source_pairs = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"precedence-{index}",
        )
        for index in range(5)
    )
    candidates = tuple(pair[0] for pair in source_pairs)
    foreign_candidate, _foreign_evidence = _append_sqlite_revision_candidate(
        store,
        foreign_scope,
        index=0,
        key="precedence-foreign",
    )
    owner_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[0].candidate_id,
        idempotency_key="shared-revision-key",
    )
    owner = store.append_revision(owner_proposal)
    foreign_parent = store.append_revision(
        _make_sqlite_revision(
            scope=foreign_scope,
            candidate_id=foreign_candidate.candidate_id,
            idempotency_key="foreign-parent",
        )
    )
    idempotency_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id="cand_missing",
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key=owner.proposal.idempotency_key,
    )
    collision_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id="cand_missing",
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key="collision-before-relationships",
    )
    replacement_suffix = "0" * 40 if owner.content_hash[24:] != "0" * 40 else "1" * 40
    collision_digest = owner.content_hash[:24] + replacement_suffix
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(
            {
                idempotency_attempt.canonical_bytes(): owner.content_hash,
                collision_attempt.canonical_bytes(): collision_digest,
            }
        ),
    )
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((2, 6, 6, 6, 2), [])

    with pytest.raises(RevisionConflictError) as idempotency_error:
        store.append_revision(idempotency_attempt)
    assert type(idempotency_error.value) is RevisionConflictError
    assert str(idempotency_error.value) == (
        "scoped revision idempotency key already refers to different content"
    )
    assert _revision_graph_state(database_path) == baseline_state

    with pytest.raises(RevisionConflictError) as collision_error:
        store.append_revision(collision_attempt)
    assert type(collision_error.value) is RevisionConflictError
    assert str(collision_error.value) == (
        f"revision ID collision for {owner.revision_id!r}"
    )
    assert _revision_graph_state(database_path) == baseline_state

    missing_candidate_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id="cand_missing",
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key="missing-candidate-before-parent",
    )
    with pytest.raises(CandidateNotFoundError) as candidate_error:
        store.append_revision(missing_candidate_attempt)
    assert type(candidate_error.value) is CandidateNotFoundError
    assert str(candidate_error.value) == "candidate 'cand_missing' was not found"
    assert _revision_graph_state(database_path) == baseline_state

    consumed_candidate_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id=owner.proposal.candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key="consumed-candidate-before-parent",
    )
    with pytest.raises(RevisionConflictError) as consumed_error:
        store.append_revision(consumed_candidate_attempt)
    assert type(consumed_error.value) is RevisionConflictError
    assert str(consumed_error.value) == (
        f"candidate {owner.proposal.candidate_id!r} already backs a revision"
    )
    assert _revision_graph_state(database_path) == baseline_state

    missing_parent_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[1].candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id="rev_missing",
        idempotency_key="missing-parent",
    )
    with pytest.raises(RevisionNotFoundError) as parent_error:
        store.append_revision(missing_parent_attempt)
    assert type(parent_error.value) is RevisionNotFoundError
    assert str(parent_error.value) == "revision 'rev_missing' was not found"
    assert _revision_graph_state(database_path) == baseline_state

    foreign_parent_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[2].candidate_id,
        operation=RevisionOperation.CONTRADICT,
        parent_revision_id=foreign_parent.revision_id,
        idempotency_key="foreign-parent-hidden",
    )
    with pytest.raises(RevisionNotFoundError) as foreign_parent_error:
        store.append_revision(foreign_parent_attempt)
    assert type(foreign_parent_error.value) is RevisionNotFoundError
    assert str(foreign_parent_error.value) == (
        f"revision {foreign_parent.revision_id!r} was not found"
    )
    assert _revision_graph_state(database_path) == baseline_state

    foreign_candidate_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id=foreign_candidate.candidate_id,
        idempotency_key="foreign-candidate-hidden",
    )
    with pytest.raises(CandidateNotFoundError) as foreign_candidate_error:
        store.append_revision(foreign_candidate_attempt)
    assert type(foreign_candidate_error.value) is CandidateNotFoundError
    assert str(foreign_candidate_error.value) == (
        f"candidate {foreign_candidate.candidate_id!r} was not found"
    )
    assert _revision_graph_state(database_path) == baseline_state

    recovered_missing_parent = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[1].candidate_id,
            idempotency_key=missing_parent_attempt.idempotency_key,
        )
    )
    recovered_foreign_parent = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[2].candidate_id,
            idempotency_key=foreign_parent_attempt.idempotency_key,
        )
    )
    recovered_collision = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[3].candidate_id,
            idempotency_key=collision_attempt.idempotency_key,
        )
    )
    assert recovered_missing_parent.proposal.candidate_id == candidates[1].candidate_id
    assert recovered_foreign_parent.proposal.candidate_id == candidates[2].candidate_id
    assert recovered_collision.proposal.candidate_id == candidates[3].candidate_id
    assert _revision_graph_state(database_path) == ((2, 6, 6, 6, 5), [])


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
def test_sqlite_revision_collision_is_scoped_atomic_and_loser_candidate_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    database_path = tmp_path / f"revision-{collision_kind}-collision.sqlite3"
    store = SQLiteMemoryStore(database_path)
    first_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"revision-{collision_kind}-first",
    )
    second_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"revision-{collision_kind}-second",
    )
    winner_candidate, _winner_evidence = _append_sqlite_revision_candidate(
        store,
        first_scope,
        index=0,
        key=f"{collision_kind}-winner",
    )
    loser_candidate, _loser_evidence = _append_sqlite_revision_candidate(
        store,
        first_scope,
        index=1,
        key=f"{collision_kind}-loser",
    )
    cross_scope_candidate, _cross_scope_evidence = _append_sqlite_revision_candidate(
        store,
        second_scope,
        index=0,
        key=f"{collision_kind}-cross-scope",
    )
    winner_proposal = _make_sqlite_revision(
        scope=first_scope,
        candidate_id=winner_candidate.candidate_id,
        idempotency_key="winner-revision-key",
    )
    loser_proposal = _make_sqlite_revision(
        scope=first_scope,
        candidate_id=loser_candidate.candidate_id,
        idempotency_key="loser-revision-key",
    )
    replacement_proposal = _make_sqlite_revision(
        scope=first_scope,
        candidate_id=loser_candidate.candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=f"rev_{'a' * 24}",
        idempotency_key=loser_proposal.idempotency_key,
    )
    cross_scope_proposal = _make_sqlite_revision(
        scope=second_scope,
        candidate_id=cross_scope_candidate.candidate_id,
        idempotency_key=loser_proposal.idempotency_key,
    )
    shared_prefix = "a" * 24
    winner_digest = (
        "a" * 64 if collision_kind == "full-hash" else shared_prefix + "b" * 40
    )
    loser_digest = (
        winner_digest if collision_kind == "full-hash" else shared_prefix + "c" * 40
    )
    replacement_digest = "d" * 64
    digest_by_canonical = {
        winner_proposal.canonical_bytes(): winner_digest,
        loser_proposal.canonical_bytes(): loser_digest,
        replacement_proposal.canonical_bytes(): replacement_digest,
        cross_scope_proposal.canonical_bytes(): winner_digest,
    }
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )

    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 0), [])
    winner = store.append_revision(winner_proposal)
    assert winner.content_hash == winner_digest
    assert winner.revision_id == f"rev_{shared_prefix}"
    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 1), [])

    with pytest.raises(RevisionConflictError) as collision_error:
        store.append_revision(loser_proposal)
    assert type(collision_error.value) is RevisionConflictError
    assert str(collision_error.value) == (
        f"revision ID collision for {winner.revision_id!r}"
    )
    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 1), [])

    cross_scope = store.append_revision(cross_scope_proposal)
    assert cross_scope.revision_id == winner.revision_id
    assert store.get_revision(first_scope, winner.revision_id) == winner
    assert store.get_revision(second_scope, winner.revision_id) == cross_scope
    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 2), [])

    recovered_loser = store.append_revision(replacement_proposal)
    assert recovered_loser.revision_id == f"rev_{replacement_digest[:24]}"
    assert recovered_loser.proposal.candidate_id == loser_candidate.candidate_id
    assert recovered_loser.proposal.idempotency_key == loser_proposal.idempotency_key
    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 3), [])

    with pytest.raises(RevisionConflictError) as retry_error:
        store.append_revision(loser_proposal)
    assert type(retry_error.value) is RevisionConflictError
    assert str(retry_error.value) == (
        "scoped revision idempotency key already refers to different content"
    )
    assert store.list_revisions(first_scope) == tuple(
        sorted(
            (winner, recovered_loser),
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    assert store.list_revisions(second_scope) == (cross_scope,)
    assert _revision_graph_state(database_path) == ((2, 3, 3, 3, 3), [])


def test_revision_lineage_rejects_signed64_generation_overflow() -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-overflow-helper")
    parent_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id="cand_parent",
        idempotency_key="overflow-parent",
    )
    parent = MemoryRevision(
        revision_id="rev_parent",
        memory_id="mem_parent",
        generation=2**63 - 1,
        proposal=parent_proposal,
        content_hash="a" * 64,
        created_at=datetime(2026, 7, 8, 4, 5, 6, tzinfo=UTC),
    )
    child_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id="cand_child",
        operation=RevisionOperation.REFINE,
        parent_revision_id=parent.revision_id,
        idempotency_key="overflow-child",
    )
    parent_state = (
        parent.revision_id,
        parent.memory_id,
        parent.generation,
        parent.proposal,
        parent.content_hash,
        parent.created_at,
    )

    with pytest.raises(RevisionConflictError) as raised:
        sqlite_store_module._derive_revision_lineage(
            child_proposal,
            "b" * 64,
            parent,
        )

    assert type(raised.value) is RevisionConflictError
    assert str(raised.value) == "revision generation exceeds the signed-64 range"
    assert (
        parent.revision_id,
        parent.memory_id,
        parent.generation,
        parent.proposal,
        parent.content_hash,
        parent.created_at,
    ) == parent_state


def test_sqlite_revision_overflow_precedes_insert_and_leaves_candidate_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-overflow-append.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-overflow-append")
    parent_candidate, _parent_evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=0,
        key="overflow-append-parent",
    )
    child_candidate, _child_evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=1,
        key="overflow-append-child",
    )
    parent = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=parent_candidate.candidate_id,
            idempotency_key="overflow-append-parent-revision",
        )
    )
    with sqlite_backend._read_transaction(str(database_path)) as cursor:
        snapshot = sqlite_store_module._load_revision_snapshot(cursor)
    (
        scope_by_id,
        candidate_by_address,
        revision_by_address,
        revision_by_idempotency,
        revision_by_candidate,
    ) = snapshot
    scope_id = next(
        stored_scope_id
        for stored_scope_id, stored_scope in scope_by_id.items()
        if stored_scope == scope
    )
    synthetic_parent = MemoryRevision(
        revision_id=parent.revision_id,
        memory_id=parent.memory_id,
        generation=2**63 - 1,
        proposal=parent.proposal,
        content_hash=parent.content_hash,
        created_at=parent.created_at,
    )
    controlled_revision_by_address = dict(revision_by_address)
    controlled_revision_by_address[(scope_id, parent.revision_id)] = synthetic_parent
    controlled_revision_by_idempotency = dict(revision_by_idempotency)
    controlled_revision_by_idempotency[(scope_id, parent.proposal.idempotency_key)] = (
        synthetic_parent
    )
    controlled_revision_by_candidate = dict(revision_by_candidate)
    controlled_revision_by_candidate[(scope_id, parent.proposal.candidate_id)] = (
        synthetic_parent
    )
    snapshot_hits = 0
    real_load_revision_snapshot = sqlite_store_module._load_revision_snapshot

    def controlled_snapshot(
        cursor: sqlite3.Cursor,
    ) -> tuple[
        dict[int, MemoryScope],
        dict[tuple[int, str], MemoryCandidate],
        dict[tuple[int, str], MemoryRevision],
        dict[tuple[int, str], MemoryRevision],
        dict[tuple[int, str], MemoryRevision],
    ]:
        nonlocal snapshot_hits
        snapshot_hits += 1
        assert real_load_revision_snapshot(cursor) == snapshot
        return (
            dict(scope_by_id),
            dict(candidate_by_address),
            dict(controlled_revision_by_address),
            dict(controlled_revision_by_idempotency),
            dict(controlled_revision_by_candidate),
        )

    monkeypatch.setattr(
        sqlite_store_module,
        "_load_revision_snapshot",
        controlled_snapshot,
    )
    sql_plan = _SQLiteFailurePlan(
        after_statement="INSERT INTO memory_revisions",
        after_statement_check=lambda _normalized_sql: None,
    )
    _install_sqlite_failure_proxy(monkeypatch, sql_plan)
    overflow_attempt = _make_sqlite_revision(
        scope=scope,
        candidate_id=child_candidate.candidate_id,
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id=parent.revision_id,
        idempotency_key="overflow-append-attempt",
    )
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((1, 2, 2, 2, 1), [])

    with pytest.raises(RevisionConflictError) as raised:
        store.append_revision(overflow_attempt)

    assert type(raised.value) is RevisionConflictError
    assert str(raised.value) == "revision generation exceeds the signed-64 range"
    assert snapshot_hits == 1
    assert not any(
        event.startswith(
            (
                "attempt:INSERT INTO MEMORY_REVISIONS",
                "executed:INSERT INTO MEMORY_REVISIONS",
            )
        )
        for event in sql_plan.events
    )
    assert _revision_graph_state(database_path) == baseline_state
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        child_rows = connection.execute(
            "SELECT revision_id FROM memory_revisions "
            "WHERE candidate_id = ? OR idempotency_key = ?",
            (
                child_candidate.candidate_id,
                overflow_attempt.idempotency_key,
            ),
        ).fetchall()
    finally:
        connection.close()
    assert child_rows == []

    monkeypatch.undo()
    recovered = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=child_candidate.candidate_id,
            idempotency_key=overflow_attempt.idempotency_key,
        )
    )
    assert recovered.proposal.candidate_id == child_candidate.candidate_id
    assert recovered.proposal.idempotency_key == overflow_attempt.idempotency_key
    assert recovered.generation == 0
    assert _revision_graph_state(database_path) == ((1, 2, 2, 2, 2), [])


def test_sqlite_concurrent_identical_revision_requests_converge(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-concurrent-identical.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-concurrent-identical",
    )
    candidate, _evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=0,
        key="concurrent-identical",
    )
    proposals = tuple(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidate.candidate_id,
            idempotency_key="concurrent-identical-revision",
        )
        for _index in range(_SQLITE_REVISION_RACE_SIZE)
    )

    outcomes = _run_sqlite_revision_race(database_path, proposals)

    assert not any(isinstance(item, MemoryPersistenceError) for item in outcomes)
    assert all(type(item) is MemoryRevision for item in outcomes)
    revisions = tuple(item for item in outcomes if type(item) is MemoryRevision)
    assert len(revisions) == _SQLITE_REVISION_RACE_SIZE
    first = revisions[0]
    assert all(revision == first for revision in revisions)
    assert {revision.revision_id for revision in revisions} == {first.revision_id}
    assert {revision.content_hash for revision in revisions} == {first.content_hash}
    assert {revision.memory_id for revision in revisions} == {first.memory_id}
    assert {revision.generation for revision in revisions} == {0}
    assert {revision.created_at for revision in revisions} == {first.created_at}
    assert all(revision.proposal == proposals[0] for revision in revisions)
    assert _revision_graph_state(database_path) == ((1, 1, 1, 1, 1), [])

    fresh = SQLiteMemoryStore(database_path)
    persisted = fresh.get_revision(scope, first.revision_id)
    assert persisted == first
    assert persisted is not first
    assert fresh.list_revisions(scope) == (persisted,)


def test_sqlite_concurrent_revision_idempotency_conflicts_leave_loser_candidates_reusable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-concurrent-idempotency.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-concurrent-idempotency",
    )
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"concurrent-idempotency-{index}",
        )[0]
        for index in range(_SQLITE_REVISION_RACE_SIZE)
    )
    proposals = tuple(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidate.candidate_id,
            idempotency_key="concurrent-shared-revision-key",
        )
        for candidate in candidates
    )
    derived_revision_ids = {
        f"rev_{hashlib.sha256(proposal.canonical_bytes()).hexdigest()[:24]}"
        for proposal in proposals
    }
    assert len(derived_revision_ids) == _SQLITE_REVISION_RACE_SIZE

    outcomes = _run_sqlite_revision_race(database_path, proposals)

    persistence_errors = tuple(
        item for item in outcomes if isinstance(item, MemoryPersistenceError)
    )
    winners = tuple(item for item in outcomes if type(item) is MemoryRevision)
    conflicts = tuple(item for item in outcomes if type(item) is RevisionConflictError)
    unexpected = tuple(
        item
        for item in outcomes
        if type(item) not in {MemoryRevision, RevisionConflictError}
    )
    assert persistence_errors == ()
    assert unexpected == ()
    assert len(winners) == 1
    assert len(conflicts) == _SQLITE_REVISION_RACE_SIZE - 1
    assert all(
        str(conflict)
        == "scoped revision idempotency key already refers to different content"
        for conflict in conflicts
    )
    winner = winners[0]
    assert _revision_graph_state(database_path) == (
        (
            1,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            1,
        ),
        [],
    )

    loser_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.candidate_id != winner.proposal.candidate_id
    )
    recovery_store = SQLiteMemoryStore(database_path)
    recovered = tuple(
        recovery_store.append_revision(
            _make_sqlite_revision(
                scope=scope,
                candidate_id=candidate.candidate_id,
                idempotency_key=f"concurrent-recovered-candidate-{index}",
            )
        )
        for index, candidate in enumerate(loser_candidates)
    )
    assert len(recovered) == _SQLITE_REVISION_RACE_SIZE - 1
    stored = recovery_store.list_revisions(scope)
    assert len(stored) == _SQLITE_REVISION_RACE_SIZE
    assert {revision.proposal.candidate_id for revision in stored} == {
        candidate.candidate_id for candidate in candidates
    }
    assert _revision_graph_state(database_path) == (
        (
            1,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
        ),
        [],
    )


def test_sqlite_concurrent_different_transitions_using_one_candidate_have_one_winner(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-concurrent-candidate.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-concurrent-candidate",
    )
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"concurrent-candidate-{index}",
        )[0]
        for index in range(_SQLITE_REVISION_RACE_SIZE)
    )
    contested_candidate = candidates[0]
    recovery_candidates = candidates[1:]
    proposals = tuple(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=contested_candidate.candidate_id,
            idempotency_key=f"concurrent-candidate-key-{index}",
        )
        for index in range(_SQLITE_REVISION_RACE_SIZE)
    )
    derived_revision_ids = {
        f"rev_{hashlib.sha256(proposal.canonical_bytes()).hexdigest()[:24]}"
        for proposal in proposals
    }
    assert len(derived_revision_ids) == _SQLITE_REVISION_RACE_SIZE

    outcomes = _run_sqlite_revision_race(database_path, proposals)

    persistence_errors = tuple(
        item for item in outcomes if isinstance(item, MemoryPersistenceError)
    )
    winners = tuple(item for item in outcomes if type(item) is MemoryRevision)
    conflicts = tuple(item for item in outcomes if type(item) is RevisionConflictError)
    unexpected = tuple(
        item
        for item in outcomes
        if type(item) not in {MemoryRevision, RevisionConflictError}
    )
    assert persistence_errors == ()
    assert unexpected == ()
    assert len(winners) == 1
    assert len(conflicts) == _SQLITE_REVISION_RACE_SIZE - 1
    assert all(
        str(conflict)
        == f"candidate {contested_candidate.candidate_id!r} already backs a revision"
        for conflict in conflicts
    )
    winner = winners[0]
    assert _revision_graph_state(database_path) == (
        (
            1,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            1,
        ),
        [],
    )

    loser_keys = tuple(
        proposal.idempotency_key
        for proposal in proposals
        if proposal.idempotency_key != winner.proposal.idempotency_key
    )
    recovery_store = SQLiteMemoryStore(database_path)
    recovered = tuple(
        recovery_store.append_revision(
            _make_sqlite_revision(
                scope=scope,
                candidate_id=candidate.candidate_id,
                idempotency_key=loser_key,
            )
        )
        for candidate, loser_key in zip(
            recovery_candidates,
            loser_keys,
            strict=True,
        )
    )
    assert len(recovered) == _SQLITE_REVISION_RACE_SIZE - 1
    stored = recovery_store.list_revisions(scope)
    assert len(stored) == _SQLITE_REVISION_RACE_SIZE
    assert {revision.proposal.candidate_id for revision in stored} == {
        candidate.candidate_id for candidate in candidates
    }
    assert {revision.proposal.idempotency_key for revision in recovered} == set(
        loser_keys
    )
    assert _revision_graph_state(database_path) == (
        (
            1,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
            _SQLITE_REVISION_RACE_SIZE,
        ),
        [],
    )


def test_sqlite_concurrent_sibling_revisions_all_survive(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-concurrent-siblings.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-concurrent-siblings",
    )
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"concurrent-sibling-{index}",
        )[0]
        for index in range(_SQLITE_REVISION_RACE_SIZE + 1)
    )
    parent = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[0].candidate_id,
            idempotency_key="concurrent-sibling-parent",
        )
    )
    operations = (
        RevisionOperation.REFINE,
        RevisionOperation.SUPERSEDE,
        RevisionOperation.CONTRADICT,
    )
    proposals = tuple(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidate.candidate_id,
            operation=operations[index % len(operations)],
            parent_revision_id=parent.revision_id,
            idempotency_key=f"concurrent-sibling-key-{index}",
        )
        for index, candidate in enumerate(candidates[1:])
    )

    outcomes = _run_sqlite_revision_race(database_path, proposals)

    assert not any(isinstance(item, MemoryPersistenceError) for item in outcomes)
    assert all(type(item) is MemoryRevision for item in outcomes)
    siblings = tuple(item for item in outcomes if type(item) is MemoryRevision)
    assert len(siblings) == _SQLITE_REVISION_RACE_SIZE
    assert len({revision.revision_id for revision in siblings}) == (
        _SQLITE_REVISION_RACE_SIZE
    )
    assert {revision.proposal.candidate_id for revision in siblings} == {
        candidate.candidate_id for candidate in candidates[1:]
    }
    assert {revision.proposal.parent_revision_id for revision in siblings} == {
        parent.revision_id
    }
    assert {revision.memory_id for revision in siblings} == {parent.memory_id}
    assert {revision.generation for revision in siblings} == {parent.generation + 1}

    fresh = SQLiteMemoryStore(database_path)
    stored = fresh.list_revisions(scope)
    assert len(stored) == _SQLITE_REVISION_RACE_SIZE + 1
    assert {revision.revision_id for revision in stored} == {
        parent.revision_id,
        *(revision.revision_id for revision in siblings),
    }
    assert stored == tuple(
        sorted(
            (parent, *siblings),
            key=lambda revision: (
                revision.memory_id,
                revision.generation,
                revision.revision_id,
            ),
        )
    )
    assert _revision_graph_state(database_path) == (
        (
            1,
            _SQLITE_REVISION_RACE_SIZE + 1,
            _SQLITE_REVISION_RACE_SIZE + 1,
            _SQLITE_REVISION_RACE_SIZE + 1,
            _SQLITE_REVISION_RACE_SIZE + 1,
        ),
        [],
    )


def test_sqlite_revision_snapshot_loads_each_scalar_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-scalar-once.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-scalar-once")
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"scalar-once-{index}",
        )[0]
        for index in range(5)
    )
    root = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[0].candidate_id,
            idempotency_key="scalar-once-root",
        )
    )
    left = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[1].candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=root.revision_id,
            idempotency_key="scalar-once-left",
        )
    )
    right = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[2].candidate_id,
            operation=RevisionOperation.CONTRADICT,
            parent_revision_id=root.revision_id,
            idempotency_key="scalar-once-right",
        )
    )
    grandchild = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[3].candidate_id,
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id=left.revision_id,
            idempotency_key="scalar-once-grandchild",
        )
    )
    second_root = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[4].candidate_id,
            idempotency_key="scalar-once-second-root",
        )
    )
    expected = tuple(
        sorted(
            (root, left, right, grandchild, second_root),
            key=lambda item: (item.memory_id, item.generation, item.revision_id),
        )
    )
    revision_select_sql = _normalize_sql(sqlite_store_module._REVISION_SELECT)
    address_scan_sql = _normalize_sql(
        "SELECT scope_id, revision_id FROM memory_revisions"
    )
    scalar_loads: dict[tuple[object, ...], int] = {}
    address_scans = 0
    real_connect = sqlite_backend._connect

    class CountingRevisionCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> CountingRevisionCursor:
            nonlocal address_scans
            normalized = _normalize_sql(sql)
            if normalized == address_scan_sql:
                address_scans += 1
            elif normalized == revision_select_sql:
                assert isinstance(parameters, tuple)
                scalar_loads[parameters] = scalar_loads.get(parameters, 0) + 1
            self._real_cursor.execute(sql, parameters)
            return self

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class CountingRevisionConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> CountingRevisionCursor:
            return CountingRevisionCursor(self._real_connection.cursor(*args, **kwargs))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> CountingRevisionConnection:
        return CountingRevisionConnection(real_connect(path))

    monkeypatch.setattr(sqlite_backend, "_connect", connect)

    assert store.list_revisions(scope) == expected
    assert address_scans == 1
    assert len(scalar_loads) == len(expected)
    assert set(load_count for load_count in scalar_loads.values()) == {1}
    assert {parameters[1] for parameters in scalar_loads} == {
        revision.revision_id for revision in expected
    }


def test_revision_topology_handles_a_chain_deeper_than_recursion_limit() -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-deep-chain")
    chain_length = sys.getrecursionlimit() + 64
    root_hash = "a" * 64
    memory_id = f"mem_{root_hash[:24]}"
    revision_by_address: dict[tuple[int, str], MemoryRevision] = {}
    parent_by_address: dict[tuple[int, str], tuple[int, str] | None] = {}
    parent_revision_id: str | None = None

    for generation in range(chain_length):
        revision_id = f"rev_{generation:024x}"
        operation = (
            RevisionOperation.ADD if generation == 0 else RevisionOperation.REFINE
        )
        proposal = _make_sqlite_revision(
            scope=scope,
            candidate_id=f"cand_{generation:024x}",
            operation=operation,
            parent_revision_id=parent_revision_id,
            idempotency_key=f"deep-chain-{generation}",
        )
        revision = MemoryRevision(
            revision_id=revision_id,
            memory_id=memory_id,
            generation=generation,
            proposal=proposal,
            content_hash=root_hash if generation == 0 else f"{generation:064x}",
            created_at=datetime(2026, 7, 8, tzinfo=UTC),
        )
        address = (1, revision_id)
        revision_by_address[address] = revision
        parent_by_address[address] = (
            None if parent_revision_id is None else (1, parent_revision_id)
        )
        parent_revision_id = revision_id

    revision_by_address = dict(reversed(tuple(revision_by_address.items())))
    parent_by_address = {
        address: parent_by_address[address] for address in revision_by_address
    }
    deepest_address = (1, f"rev_{chain_length - 1:024x}")
    assert next(iter(revision_by_address)) == deepest_address

    sqlite_store_module._validate_revision_topology(
        revision_by_address,
        parent_by_address,
    )

    assert len(revision_by_address) == chain_length
    assert max(item.generation for item in revision_by_address.values()) == (
        chain_length - 1
    )


def test_sqlite_revision_snapshot_rejects_parent_cycles_iteratively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "revision-cycle.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-cycle")
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"cycle-{index}",
        )[0]
        for index in range(2)
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
    finally:
        connection.close()
    first_revision_id = f"rev_{'a' * 24}"
    second_revision_id = f"rev_{'b' * 24}"
    first_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[0].candidate_id,
        operation=RevisionOperation.REFINE,
        parent_revision_id=second_revision_id,
        idempotency_key="cycle-first",
    )
    second_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[1].candidate_id,
        operation=RevisionOperation.CONTRADICT,
        parent_revision_id=first_revision_id,
        idempotency_key="cycle-second",
    )
    first_hash = "a" * 64
    second_hash = "b" * 64
    created_at_text = datetime(2026, 7, 8, 5, 6, 7, tzinfo=UTC).isoformat()
    memory_id = "mem_cycle"

    def virtual_row(
        revision_id: str,
        proposal: RevisionProposal,
        content_hash: str,
    ) -> tuple[object, ...]:
        return (
            revision_id,
            proposal.canonical_bytes(),
            content_hash,
            created_at_text,
            sqlite_store_module._record_storage_hash(
                record_kind="revision",
                scope=scope,
                record_id=revision_id,
                content_hash=content_hash,
                created_at_text=created_at_text,
                memory_id=memory_id,
                generation=1,
            ),
            proposal.candidate_id,
            memory_id,
            1,
            proposal.operation.value,
            proposal.parent_revision_id,
            proposal.idempotency_key,
        )

    rows_by_address = {
        (scope_id, first_revision_id): virtual_row(
            first_revision_id,
            first_proposal,
            first_hash,
        ),
        (scope_id, second_revision_id): virtual_row(
            second_revision_id,
            second_proposal,
            second_hash,
        ),
    }
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(
            {
                first_proposal.canonical_bytes(): first_hash,
                second_proposal.canonical_bytes(): second_hash,
            }
        ),
    )
    address_sql = _normalize_sql("SELECT scope_id, revision_id FROM memory_revisions")
    scalar_sql = _normalize_sql(sqlite_store_module._REVISION_SELECT)
    proxy_hits = {"address": 0, "scalar": 0}
    topology_hits = 0
    real_connect = sqlite_backend._connect
    real_validate_topology = sqlite_store_module._validate_revision_topology

    class VirtualCycleCursor:
        def __init__(self, real_cursor: sqlite3.Cursor) -> None:
            self._real_cursor = real_cursor
            self._last_sql = ""
            self._parameters: object = ()

        def execute(
            self,
            sql: str,
            parameters: object = (),
        ) -> VirtualCycleCursor:
            self._last_sql = _normalize_sql(sql)
            self._parameters = parameters
            self._real_cursor.execute(sql, parameters)
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            rows = [tuple(row) for row in self._real_cursor.fetchall()]
            if self._last_sql == address_sql:
                assert rows == []
                proxy_hits["address"] += 1
                return list(rows_by_address)
            return rows

        def fetchone(self) -> tuple[object, ...] | None:
            row = self._real_cursor.fetchone()
            if self._last_sql == scalar_sql and self._parameters in rows_by_address:
                assert row is None
                proxy_hits["scalar"] += 1
                return rows_by_address[self._parameters]  # type: ignore[index]
            return None if row is None else tuple(row)

        def __iter__(self) -> Any:
            return iter(self._real_cursor)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_cursor, name)

    class VirtualCycleConnection:
        def __init__(self, real_connection: sqlite3.Connection) -> None:
            self._real_connection = real_connection

        def cursor(
            self,
            *args: object,
            **kwargs: object,
        ) -> VirtualCycleCursor:
            return VirtualCycleCursor(self._real_connection.cursor(*args, **kwargs))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real_connection, name)

    def connect(path: str) -> VirtualCycleConnection:
        return VirtualCycleConnection(real_connect(path))

    def observed_topology(
        revision_by_address: dict[tuple[int, str], MemoryRevision],
        parent_by_address: dict[tuple[int, str], tuple[int, str] | None],
    ) -> None:
        nonlocal topology_hits
        topology_hits += 1
        real_validate_topology(revision_by_address, parent_by_address)

    monkeypatch.setattr(sqlite_backend, "_connect", connect)
    monkeypatch.setattr(
        sqlite_store_module,
        "_validate_revision_topology",
        observed_topology,
    )
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((1, 2, 2, 2, 0), [])

    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="revision parent graph contains a cycle",
    ) as raised:
        store.list_revisions(scope)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert raised.value.__cause__ is None
    assert proxy_hits == {"address": 1, "scalar": 2}
    assert topology_hits == 1
    assert _revision_graph_state(database_path) == baseline_state


def test_sqlite_revision_snapshot_validates_unrelated_lower_graph_first(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "revision-unrelated-base.sqlite3"
    store = SQLiteMemoryStore(base_path)
    target_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-unrelated-target",
    )
    target_candidate, _target_evidence = _append_sqlite_revision_candidate(
        store,
        target_scope,
        index=0,
        key="unrelated-target",
    )
    target_proposal = _make_sqlite_revision(
        scope=target_scope,
        candidate_id=target_candidate.candidate_id,
        idempotency_key="unrelated-target-revision",
    )
    target_revision = store.append_revision(target_proposal)
    evidence_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-unrelated-evidence",
    )
    unrelated_evidence = store.append(
        _make_sqlite_evidence(
            scope=evidence_scope,
            payload="unrelated evidence-only scope",
            idempotency_key="unrelated-evidence-only",
        )
    )
    candidate_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-unrelated-candidate",
    )
    unrelated_candidate, _candidate_evidence = _append_sqlite_revision_candidate(
        store,
        candidate_scope,
        index=0,
        key="unrelated-candidate-only",
    )
    revision_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-unrelated-revision",
    )
    revision_candidate, _revision_evidence = _append_sqlite_revision_candidate(
        store,
        revision_scope,
        index=0,
        key="unrelated-revision-only",
    )
    unrelated_revision = store.append_revision(
        _make_sqlite_revision(
            scope=revision_scope,
            candidate_id=revision_candidate.candidate_id,
            idempotency_key="unrelated-revision-only",
        )
    )
    assert _revision_graph_state(base_path) == ((4, 4, 3, 3, 2), [])

    corruption_expectations = {
        "evidence": (
            "stored evidence row failed integrity validation",
            "evidence ID disagrees with its content hash",
        ),
        "candidate": (
            "stored candidate row failed integrity validation",
            "candidate ID disagrees with its content hash",
        ),
        "revision": (
            "stored revision row failed integrity validation",
            "revision ID disagrees with its content hash",
        ),
    }
    for corruption_kind in ("evidence", "candidate", "revision"):
        for operation in ("missing-get", "list-filter", "exact-retry"):
            database_path = tmp_path / (
                f"revision-unrelated-{corruption_kind}-{operation}.sqlite3"
            )
            shutil.copyfile(base_path, database_path)
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                connection.execute("PRAGMA foreign_keys = OFF")
                assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
                if corruption_kind == "evidence":
                    moved_id = f"evd_{'0' * 24}"
                    assert moved_id != unrelated_evidence.evidence_id
                    scope_id, ingest_order = connection.execute(
                        "SELECT scope_id, ingest_order "
                        "FROM memory_evidence_ingest_orders "
                        "WHERE evidence_id = ?",
                        (unrelated_evidence.evidence_id,),
                    ).fetchone()
                    connection.execute(
                        "UPDATE memory_evidence SET evidence_id = ? "
                        "WHERE evidence_id = ?",
                        (moved_id, unrelated_evidence.evidence_id),
                    )
                    connection.execute(
                        "UPDATE memory_evidence_ingest_orders "
                        "SET evidence_id = ?, binding_hash = ? "
                        "WHERE scope_id = ? AND evidence_id = ?",
                        (
                            moved_id,
                            sqlite_backend._evidence_ingest_binding_hash(
                                scope=evidence_scope,
                                evidence_id=moved_id,
                                ingest_order=ingest_order,
                            ),
                            scope_id,
                            unrelated_evidence.evidence_id,
                        ),
                    )
                elif corruption_kind == "candidate":
                    moved_id = f"cand_{'0' * 24}"
                    assert moved_id != unrelated_candidate.candidate_id
                    connection.execute(
                        "UPDATE memory_candidate_evidence SET candidate_id = ? "
                        "WHERE candidate_id = ?",
                        (moved_id, unrelated_candidate.candidate_id),
                    )
                    connection.execute(
                        "UPDATE memory_candidates SET candidate_id = ? "
                        "WHERE candidate_id = ?",
                        (moved_id, unrelated_candidate.candidate_id),
                    )
                else:
                    moved_id = f"rev_{'0' * 24}"
                    assert moved_id != unrelated_revision.revision_id
                    connection.execute(
                        "UPDATE memory_revisions SET revision_id = ? "
                        "WHERE revision_id = ?",
                        (moved_id, unrelated_revision.revision_id),
                    )
            finally:
                connection.close()

            corrupted_state = _revision_graph_state(database_path)
            assert corrupted_state == ((4, 4, 3, 3, 2), [])
            corrupted_rows = _revision_graph_rows(database_path)
            corrupted_store = SQLiteMemoryStore(database_path)
            retry = RevisionProposal(
                scope=target_proposal.scope,
                candidate_id=target_proposal.candidate_id,
                operation=target_proposal.operation,
                parent_revision_id=target_proposal.parent_revision_id,
                idempotency_key=target_proposal.idempotency_key,
            )
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                if operation == "missing-get":
                    corrupted_store.get_revision(target_scope, "rev_missing")
                elif operation == "list-filter":
                    corrupted_store.list_revisions(
                        target_scope,
                        memory_id=target_revision.memory_id,
                    )
                else:
                    corrupted_store.append_revision(retry)

            expected_outer, expected_cause = corruption_expectations[corruption_kind]
            assert type(raised.value) is MemoryPersistenceCorruptionError
            assert str(raised.value) == expected_outer
            assert type(raised.value.__cause__) is ValueError
            assert str(raised.value.__cause__) == expected_cause
            assert _revision_graph_state(database_path) == corrupted_state
            assert _revision_graph_rows(database_path) == corrupted_rows


def test_sqlite_revision_loader_rejects_malformed_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "revision-malformed-duplicate.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-malformed")
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"malformed-{index}",
        )[0]
        for index in range(3)
    )
    root_proposal = _make_sqlite_revision(
        scope=scope,
        candidate_id=candidates[0].candidate_id,
        idempotency_key="malformed-root",
    )
    root = store.append_revision(root_proposal)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
        scalar_row = connection.execute(
            sqlite_store_module._REVISION_SELECT,
            (scope_id, root.revision_id),
        ).fetchone()
    finally:
        connection.close()
    assert scalar_row is not None

    class StaticScalarCursor:
        def __init__(self, row: tuple[object, ...]) -> None:
            self._row = row

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticScalarCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return self._row

    malformed_scalars: list[
        tuple[str, tuple[object, ...], str, type[ValueError] | type[TypeError]]
    ] = [
        (
            "wrong-arity",
            tuple(scalar_row[:-1]),
            "revision row has the wrong field count",
            ValueError,
        )
    ]
    wrong_values = (
        b"revision-id",
        "canonical-not-blob",
        b"content-hash",
        b"created-at",
        b"storage-hash",
        b"candidate-id",
        b"memory-id",
        True,
        b"operation",
        b"parent-id",
        b"idempotency-key",
    )
    for index, wrong_value in enumerate(wrong_values):
        changed_row = list(scalar_row)
        changed_row[index] = wrong_value
        malformed_scalars.append(
            (
                f"wrong-storage-{index}",
                tuple(changed_row),
                f"revision row field {index} has the wrong storage class",
                TypeError,
            )
        )
    overflow_generation_row = list(scalar_row)
    overflow_generation_row[7] = 2**63
    malformed_scalars.append(
        (
            "generation-overflow",
            tuple(overflow_generation_row),
            "generation must fit the non-negative signed-64 range",
            ValueError,
        )
    )

    for case, row, expected_cause, cause_type in malformed_scalars:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_revision(
                StaticScalarCursor(row),  # type: ignore[arg-type]
                scope,
                scope_id,
                root.revision_id,
            )
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == (
            "stored revision row failed integrity validation"
        ), case
        assert type(raised.value.__cause__) is cause_type, case
        assert str(raised.value.__cause__) == expected_cause, case

    with pytest.raises(MemoryPersistenceCorruptionError) as requested_id_error:
        sqlite_store_module._load_revision(
            StaticScalarCursor(tuple(scalar_row)),  # type: ignore[arg-type]
            scope,
            scope_id,
            "rev_requested_id_drift",
        )
    assert type(requested_id_error.value) is MemoryPersistenceCorruptionError
    assert str(requested_id_error.value) == (
        "stored revision row failed integrity validation"
    )
    assert type(requested_id_error.value.__cause__) is ValueError
    assert str(requested_id_error.value.__cause__) == (
        "loaded revision ID differs from requested ID"
    )

    class StaticRowsCursor:
        def __init__(self, rows: tuple[tuple[object, ...], ...]) -> None:
            self._rows = rows

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticRowsCursor:
            return self

        def fetchall(self) -> list[tuple[object, ...]]:
            return list(self._rows)

    address_cases = (
        (
            "wrong-arity",
            ((scope_id,),),
            "revision address row does not contain exactly two values",
        ),
        (
            "boolean-scope",
            ((True, root.revision_id),),
            "revision address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "overflow-scope",
            ((2**63, root.revision_id),),
            "revision address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "missing-scope",
            ((2**63 - 1, root.revision_id),),
            "revision address refers to a missing scope",
        ),
        (
            "nontext-revision",
            ((scope_id, b"revision-id"),),
            "revision address contains a non-text identifier",
        ),
    )
    for case, rows, expected_message in address_cases:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_revision_addresses(
                StaticRowsCursor(rows),  # type: ignore[arg-type]
                {scope_id: scope},
            )
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_message, case
        assert raised.value.__cause__ is None, case

    virtual_proposals = {
        "idempotency": _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[1].candidate_id,
            idempotency_key=root.proposal.idempotency_key,
        ),
        "candidate": _make_sqlite_revision(
            scope=scope,
            candidate_id=root.proposal.candidate_id,
            idempotency_key="duplicate-candidate",
        ),
        "parent": _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[2].candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id="rev_missing",
            idempotency_key="missing-parent",
        ),
    }
    created_at_text = datetime(2026, 7, 8, 6, 7, 8, tzinfo=UTC).isoformat()
    virtual_rows: dict[str, tuple[int, str, tuple[object, ...]]] = {}
    for case, proposal in virtual_proposals.items():
        canonical = proposal.canonical_bytes()
        content_hash = hashlib.sha256(canonical).hexdigest()
        revision_id = f"rev_{content_hash[:24]}"
        generation = 0 if proposal.operation is RevisionOperation.ADD else 1
        memory_id = (
            f"mem_{content_hash[:24]}"
            if proposal.operation is RevisionOperation.ADD
            else root.memory_id
        )
        virtual_rows[case] = (
            scope_id,
            revision_id,
            (
                revision_id,
                canonical,
                content_hash,
                created_at_text,
                sqlite_store_module._record_storage_hash(
                    record_kind="revision",
                    scope=scope,
                    record_id=revision_id,
                    content_hash=content_hash,
                    created_at_text=created_at_text,
                    memory_id=memory_id,
                    generation=generation,
                ),
                proposal.candidate_id,
                memory_id,
                generation,
                proposal.operation.value,
                proposal.parent_revision_id,
                proposal.idempotency_key,
            ),
        )
    expected_messages = {
        "address": "revision address appears multiple times",
        "idempotency": "revision idempotency key appears multiple times in one scope",
        "candidate": "candidate backs multiple revisions in one scope",
        "parent": "revision refers to a missing same-scope parent",
    }
    address_sql = _normalize_sql("SELECT scope_id, revision_id FROM memory_revisions")
    scalar_sql = _normalize_sql(sqlite_store_module._REVISION_SELECT)
    real_connect = sqlite_backend._connect
    baseline_state = _revision_graph_state(database_path)
    assert baseline_state == ((1, 3, 3, 3, 1), [])

    for case in ("address", "idempotency", "candidate", "parent"):
        hits = {"address": 0, "scalar": 0}

        class DuplicateRevisionCursor:
            def __init__(self, real_cursor: sqlite3.Cursor) -> None:
                self._real_cursor = real_cursor
                self._last_sql = ""
                self._parameters: object = ()

            def execute(
                self,
                sql: str,
                parameters: object = (),
            ) -> DuplicateRevisionCursor:
                self._last_sql = _normalize_sql(sql)
                self._parameters = parameters
                self._real_cursor.execute(sql, parameters)
                return self

            def fetchall(self) -> list[tuple[object, ...]]:
                rows = [tuple(row) for row in self._real_cursor.fetchall()]
                if self._last_sql == address_sql:
                    hits["address"] += 1
                    if case == "address":
                        return [*rows, (scope_id, root.revision_id)]
                    virtual_scope_id, virtual_revision_id, _row = virtual_rows[case]
                    return [*rows, (virtual_scope_id, virtual_revision_id)]
                return rows

            def fetchone(self) -> tuple[object, ...] | None:
                row = self._real_cursor.fetchone()
                if case != "address":
                    virtual_scope_id, virtual_revision_id, virtual_row = virtual_rows[
                        case
                    ]
                    if self._last_sql == scalar_sql and self._parameters == (
                        virtual_scope_id,
                        virtual_revision_id,
                    ):
                        assert row is None
                        hits["scalar"] += 1
                        return virtual_row
                return None if row is None else tuple(row)

            def __iter__(self) -> Any:
                return iter(self._real_cursor)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real_cursor, name)

        class DuplicateRevisionConnection:
            def __init__(self, real_connection: sqlite3.Connection) -> None:
                self._real_connection = real_connection

            def cursor(
                self,
                *args: object,
                **kwargs: object,
            ) -> DuplicateRevisionCursor:
                return DuplicateRevisionCursor(
                    self._real_connection.cursor(*args, **kwargs)
                )

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real_connection, name)

        def connect(path: str) -> DuplicateRevisionConnection:
            return DuplicateRevisionConnection(real_connect(path))

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sqlite_backend, "_connect", connect)
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                store.list_revisions(scope)

        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_messages[case], case
        assert raised.value.__cause__ is None, case
        assert hits == {
            "address": 1,
            "scalar": 0 if case == "address" else 1,
        }, case
        assert _revision_graph_state(database_path) == baseline_state, case


def test_sqlite_revision_loader_rejects_projection_lineage_and_storage_drift(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "revision-drift-base.sqlite3"
    store = SQLiteMemoryStore(base_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "revision-drift")
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"drift-{index}",
        )[0]
        for index in range(4)
    )
    root = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[0].candidate_id,
            idempotency_key="drift-root",
        )
    )
    alternate_root = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[1].candidate_id,
            idempotency_key="drift-alternate-root",
        )
    )
    child = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=candidates[2].candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=root.revision_id,
            idempotency_key="drift-child",
        )
    )
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "revision-drift-foreign",
    )
    foreign_candidate, _foreign_evidence = _append_sqlite_revision_candidate(
        store,
        foreign_scope,
        index=0,
        key="drift-foreign",
    )
    foreign_root = store.append_revision(
        _make_sqlite_revision(
            scope=foreign_scope,
            candidate_id=foreign_candidate.candidate_id,
            idempotency_key="drift-foreign-root",
        )
    )
    assert foreign_root.revision_id not in {
        root.revision_id,
        alternate_root.revision_id,
        child.revision_id,
    }
    baseline_state = _revision_graph_state(base_path)
    assert baseline_state == ((2, 5, 5, 5, 4), [])
    canonical_variant = b" \n" + child.proposal.canonical_bytes()
    assert canonical_variant != child.proposal.canonical_bytes()
    assert json.loads(canonical_variant) == json.loads(child.proposal.canonical_bytes())
    hash_suffix = "0" * 40 if child.content_hash[24:] != "0" * 40 else "1" * 40
    changed_content_hash = child.content_hash[:24] + hash_suffix
    assert changed_content_hash != child.content_hash
    scalar_outer = "stored revision row failed integrity validation"
    cases: dict[str, tuple[str, str | None]] = {
        "canonical": (
            scalar_outer,
            "canonical revision bytes disagree with projections",
        ),
        "content-hash": (
            scalar_outer,
            "revision content hash disagrees with canonical bytes",
        ),
        "revision-id": (
            scalar_outer,
            "revision ID disagrees with its content hash",
        ),
        "created-at-z": (
            scalar_outer,
            "revision created_at is not exact UTC isoformat text",
        ),
        "storage-hash": (
            scalar_outer,
            "revision storage hash disagrees with stored metadata",
        ),
        "candidate-id": (
            scalar_outer,
            "canonical revision bytes disagree with projections",
        ),
        "operation": (
            scalar_outer,
            "canonical revision bytes disagree with projections",
        ),
        "parent": (
            scalar_outer,
            "canonical revision bytes disagree with projections",
        ),
        "idempotency": (
            scalar_outer,
            "canonical revision bytes disagree with projections",
        ),
        "root-memory": (
            "ADD revision memory ID disagrees with its content hash",
            None,
        ),
        "root-generation": (
            "ADD revision generation is not zero",
            None,
        ),
        "child-memory": (
            "child revision memory ID differs from parent memory ID",
            None,
        ),
        "generation": (
            "child revision generation is not exactly parent generation plus one",
            None,
        ),
        "missing-parent": (
            "Memory Service SQLite data failed foreign key validation",
            None,
        ),
        "foreign-parent": (
            "Memory Service SQLite data failed foreign key validation",
            None,
        ),
        "invalid-utf8": (
            "SQLite TEXT contains invalid UTF-8",
            "unicode",
        ),
    }

    for case, (expected_outer, expected_cause) in cases.items():
        database_path = tmp_path / f"revision-drift-{case}.sqlite3"
        shutil.copyfile(base_path, database_path)
        case_store = SQLiteMemoryStore(database_path)
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
            target_revision_id = child.revision_id
            if case == "canonical":
                connection.execute(
                    "UPDATE memory_revisions SET canonical = ? WHERE revision_id = ?",
                    (sqlite3.Binary(canonical_variant), child.revision_id),
                )
            elif case == "content-hash":
                connection.execute(
                    "UPDATE memory_revisions SET content_hash = ? "
                    "WHERE revision_id = ?",
                    (changed_content_hash, child.revision_id),
                )
            elif case == "revision-id":
                target_revision_id = f"rev_{'0' * 24}"
                assert target_revision_id != child.revision_id
                connection.execute(
                    "UPDATE memory_revisions SET revision_id = ? WHERE revision_id = ?",
                    (target_revision_id, child.revision_id),
                )
            elif case == "created-at-z":
                connection.execute(
                    "UPDATE memory_revisions SET created_at = ? WHERE revision_id = ?",
                    (
                        child.created_at.isoformat().replace("+00:00", "Z"),
                        child.revision_id,
                    ),
                )
            elif case == "storage-hash":
                connection.execute(
                    "UPDATE memory_revisions SET storage_hash = ? "
                    "WHERE revision_id = ?",
                    ("0" * 64, child.revision_id),
                )
            elif case == "candidate-id":
                connection.execute(
                    "UPDATE memory_revisions SET candidate_id = ? "
                    "WHERE revision_id = ?",
                    (candidates[3].candidate_id, child.revision_id),
                )
            elif case == "operation":
                connection.execute(
                    "UPDATE memory_revisions SET operation = ? WHERE revision_id = ?",
                    (RevisionOperation.CONTRADICT.value, child.revision_id),
                )
            elif case == "parent":
                connection.execute(
                    "UPDATE memory_revisions SET parent_revision_id = ? "
                    "WHERE revision_id = ?",
                    (alternate_root.revision_id, child.revision_id),
                )
            elif case == "idempotency":
                connection.execute(
                    "UPDATE memory_revisions SET idempotency_key = ? "
                    "WHERE revision_id = ?",
                    ("moved-drift-child", child.revision_id),
                )
            elif case == "root-memory":
                target_revision_id = root.revision_id
                connection.execute(
                    "UPDATE memory_revisions SET memory_id = ? WHERE revision_id = ?",
                    ("mem_moved_root", root.revision_id),
                )
            elif case == "root-generation":
                target_revision_id = root.revision_id
                connection.execute(
                    "UPDATE memory_revisions SET generation = 1 WHERE revision_id = ?",
                    (root.revision_id,),
                )
            elif case == "child-memory":
                connection.execute(
                    "UPDATE memory_revisions SET memory_id = ? WHERE revision_id = ?",
                    (alternate_root.memory_id, child.revision_id),
                )
            elif case == "generation":
                connection.execute(
                    "UPDATE memory_revisions SET generation = 2 WHERE revision_id = ?",
                    (child.revision_id,),
                )
            elif case == "missing-parent":
                connection.execute(
                    "UPDATE memory_revisions SET parent_revision_id = ? "
                    "WHERE revision_id = ?",
                    ("rev_missing", child.revision_id),
                )
            elif case == "foreign-parent":
                connection.execute(
                    "UPDATE memory_revisions SET parent_revision_id = ? "
                    "WHERE revision_id = ?",
                    (foreign_root.revision_id, child.revision_id),
                )
            else:
                connection.execute(
                    "UPDATE memory_revisions "
                    "SET idempotency_key = CAST(X'80' AS TEXT) "
                    "WHERE revision_id = ?",
                    (child.revision_id,),
                )

            if case in {
                "root-memory",
                "root-generation",
                "child-memory",
                "generation",
            }:
                row = connection.execute(
                    "SELECT revision_id, content_hash, created_at, memory_id, generation "
                    "FROM memory_revisions WHERE revision_id = ?",
                    (target_revision_id,),
                ).fetchone()
                assert row is not None
                revision_id, content_hash, created_at_text, memory_id, generation = row
                connection.execute(
                    "UPDATE memory_revisions SET storage_hash = ? "
                    "WHERE revision_id = ?",
                    (
                        sqlite_store_module._record_storage_hash(
                            record_kind="revision",
                            scope=scope,
                            record_id=revision_id,
                            content_hash=content_hash,
                            created_at_text=created_at_text,
                            memory_id=memory_id,
                            generation=generation,
                        ),
                        revision_id,
                    ),
                )
        finally:
            connection.close()

        corrupted_state = _revision_graph_state(database_path)
        assert corrupted_state[0] == baseline_state[0]
        if case in {"missing-parent", "foreign-parent"}:
            assert len(corrupted_state[1]) == 1
            table, _rowid, parent_table, _foreign_key_id = corrupted_state[1][0]
            assert table == "memory_revisions"
            assert parent_table == "memory_revisions"
        else:
            assert corrupted_state[1] == []
        corrupted_rows = _revision_graph_rows(database_path)
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            case_store.get_revision(scope, "rev_missing")

        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == expected_outer, case
        if expected_cause is None:
            assert raised.value.__cause__ is None, case
        elif expected_cause == "unicode":
            assert type(raised.value.__cause__) is UnicodeDecodeError, case
            assert raised.value.__cause__.object == b"\x80", case
        else:
            assert type(raised.value.__cause__) is ValueError, case
            assert str(raised.value.__cause__) == expected_cause, case
        assert _revision_graph_state(database_path) == corrupted_state, case
        assert _revision_graph_rows(database_path) == corrupted_rows, case


def test_sqlite_release_reopen_preserves_aliases_member_order_and_provenance(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-reopen.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-reopen-user")
    sources = tuple(
        _append_sqlite_release_root(
            store,
            scope,
            index=index,
            key=f"reopen-{index}",
        )
        for index in range(3)
    )
    ordered_sources = tuple(
        sorted(sources, key=lambda item: item[0].revision_id, reverse=True)
    )
    manifest = ReleaseManifest(
        scope=scope,
        revision_ids=tuple(item[0].revision_id for item in ordered_sources),
    )
    expected_hash = hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    assert manifest.revision_ids != tuple(sorted(manifest.revision_ids))

    first = store.append_release(manifest, idempotency_key="release-reopen-a")
    second = store.append_release(
        ReleaseManifest(scope, tuple(manifest.revision_ids)),
        idempotency_key="release-reopen-b",
    )

    assert first == second
    assert first.release_id == f"rel_{expected_hash[:24]}"
    assert first.content_hash == expected_hash
    assert first.manifest == manifest
    assert first.manifest is not manifest
    assert first.created_at.tzinfo is UTC
    assert store.list_releases(scope) == (first,)
    assert store.get_release_revisions(scope, first.release_id) == tuple(
        item[0] for item in ordered_sources
    )
    assert _release_graph_state(database_path) == (
        (1, 3, 3, 3, 3, 1, 2, 3),
        [],
    )

    del store
    reopened = SQLiteMemoryStore(database_path)
    loaded = reopened.get_release(scope, first.release_id)
    loaded_members = reopened.get_release_revisions(scope, first.release_id)
    retry_a = reopened.append_release(
        ReleaseManifest(scope, tuple(manifest.revision_ids)),
        idempotency_key="release-reopen-a",
    )
    retry_b = reopened.append_release(
        ReleaseManifest(scope, tuple(manifest.revision_ids)),
        idempotency_key="release-reopen-b",
    )

    assert loaded == retry_a == retry_b == first
    assert loaded is not first
    assert loaded.release_id == first.release_id
    assert loaded.content_hash == first.content_hash
    assert loaded.created_at == first.created_at
    assert loaded.manifest.revision_ids == manifest.revision_ids
    assert loaded_members == tuple(item[0] for item in ordered_sources)
    assert reopened.list_releases(scope) == (loaded,)
    for loaded_revision, (_revision, candidate, evidence) in zip(
        loaded_members,
        ordered_sources,
        strict=True,
    ):
        assert loaded_revision.proposal.candidate_id == candidate.candidate_id
        assert reopened.get_candidate_evidence(
            scope,
            loaded_revision.proposal.candidate_id,
        ) == (evidence,)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        member_rows = connection.execute(
            "SELECT position, revision_id, memory_id "
            "FROM memory_release_revisions WHERE release_id = ? "
            "ORDER BY position",
            (first.release_id,),
        ).fetchall()
        alias_rows = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases "
            "ORDER BY idempotency_key",
        ).fetchall()
    finally:
        connection.close()
    assert member_rows == [
        (position, revision.revision_id, revision.memory_id)
        for position, revision in enumerate(loaded_members)
    ]
    assert alias_rows == [
        ("release-reopen-a", first.release_id),
        ("release-reopen-b", first.release_id),
    ]
    assert _release_graph_state(database_path) == (
        (1, 3, 3, 3, 3, 1, 2, 3),
        [],
    )


def test_sqlite_empty_release_creates_scope_reopens_without_members(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-empty.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-empty-user")
    manifest = ReleaseManifest(scope, ())
    expected_hash = hashlib.sha256(manifest.canonical_bytes()).hexdigest()

    assert _release_graph_state(database_path) == ((0, 0, 0, 0, 0, 0, 0, 0), [])
    first = store.append_release(manifest, idempotency_key="memory-off-a")
    second = store.append_release(
        ReleaseManifest(scope, ()),
        idempotency_key="memory-off-b",
    )

    assert first == second
    assert first.release_id == f"rel_{expected_hash[:24]}"
    assert first.content_hash == expected_hash
    assert first.manifest.revision_ids == ()
    assert store.get_release_revisions(scope, first.release_id) == ()
    assert _release_graph_state(database_path) == (
        (1, 0, 0, 0, 0, 1, 2, 0),
        [],
    )

    del store
    reopened = SQLiteMemoryStore(database_path)
    loaded = reopened.get_release(scope, first.release_id)

    assert loaded == first
    assert loaded is not first
    assert loaded.created_at == first.created_at
    assert loaded.manifest == manifest
    assert reopened.list_releases(scope) == (loaded,)
    assert reopened.get_release_revisions(scope, loaded.release_id) == ()
    assert reopened.append_release(manifest, idempotency_key="memory-off-a") == loaded
    assert reopened.append_release(manifest, idempotency_key="memory-off-b") == loaded
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        release_rows = connection.execute(
            "SELECT release_id, content_hash, created_at FROM memory_releases"
        ).fetchall()
        alias_rows = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases "
            "ORDER BY idempotency_key"
        ).fetchall()
        member_rows = connection.execute(
            "SELECT release_id FROM memory_release_revisions"
        ).fetchall()
    finally:
        connection.close()
    assert release_rows == [
        (first.release_id, first.content_hash, first.created_at.isoformat())
    ]
    assert alias_rows == [
        ("memory-off-a", first.release_id),
        ("memory-off-b", first.release_id),
    ]
    assert member_rows == []
    assert _release_graph_state(database_path) == (
        (1, 0, 0, 0, 0, 1, 2, 0),
        [],
    )


def test_sqlite_release_queries_sort_hide_scope_and_snapshot_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-queries.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-query-user")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-query-foreign",
    )
    sources = tuple(
        _append_sqlite_release_root(
            store,
            scope,
            index=index,
            key=f"query-{index}",
        )
        for index in range(4)
    )
    initial_manifests = tuple(
        ReleaseManifest(scope, (revision.revision_id,))
        for revision, _candidate, _evidence in sources[:3]
    )

    def manifest_release_id(manifest: ReleaseManifest) -> str:
        digest = hashlib.sha256(manifest.canonical_bytes()).hexdigest()
        return f"rel_{digest[:24]}"

    sorted_manifests = tuple(sorted(initial_manifests, key=manifest_release_id))
    inserted = tuple(
        store.append_release(
            manifest,
            idempotency_key=f"release-query-{index}",
        )
        for index, manifest in enumerate(reversed(sorted_manifests))
    )
    expected = tuple(sorted(inserted, key=lambda release: release.release_id))

    assert tuple(release.release_id for release in inserted) != tuple(
        release.release_id for release in expected
    )
    listed = store.list_releases(scope)
    assert listed == expected
    assert inspect.signature(SQLiteMemoryStore.append_release) == inspect.signature(
        MemoryReleaseStore.append_release
    )
    assert inspect.signature(SQLiteMemoryStore.get_release) == inspect.signature(
        MemoryReleaseStore.get_release
    )
    assert inspect.signature(
        SQLiteMemoryStore.get_release_revisions
    ) == inspect.signature(MemoryReleaseStore.get_release_revisions)
    assert inspect.signature(SQLiteMemoryStore.list_releases) == inspect.signature(
        MemoryReleaseStore.list_releases
    )

    target = expected[1]
    target_revision = next(
        revision
        for revision, _candidate, _evidence in sources
        if revision.revision_id == target.manifest.revision_ids[0]
    )
    release_id_probe = _SnapshotProbeStr(target.release_id)
    assert store.get_release(scope, release_id_probe) == target
    assert release_id_probe.override_calls == 0

    read_calls = 0
    real_read_transaction = sqlite_store_module._read_transaction

    def counted_read_transaction(database: str) -> object:
        nonlocal read_calls
        read_calls += 1
        return real_read_transaction(database)

    def public_get_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("release member traversal called a public get method")

    with monkeypatch.context() as guarded:
        guarded.setattr(
            sqlite_store_module,
            "_read_transaction",
            counted_read_transaction,
        )
        guarded.setattr(SQLiteMemoryStore, "get_release", public_get_must_not_run)
        guarded.setattr(SQLiteMemoryStore, "get_revision", public_get_must_not_run)
        assert store.get_release_revisions(scope, target.release_id) == (
            target_revision,
        )
    assert read_calls == 1

    snapshot = listed
    later_manifest = ReleaseManifest(scope, (sources[3][0].revision_id,))
    later = store.append_release(
        later_manifest,
        idempotency_key="release-query-later",
    )
    assert snapshot == expected
    assert store.list_releases(scope) == tuple(
        sorted((*expected, later), key=lambda release: release.release_id)
    )

    foreign_revision, _foreign_candidate, _foreign_evidence = (
        _append_sqlite_release_root(
            store,
            foreign_scope,
            index=0,
            key="query-foreign",
        )
    )
    foreign = store.append_release(
        ReleaseManifest(foreign_scope, (foreign_revision.revision_id,)),
        idempotency_key="release-query-foreign",
    )
    missing_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-query-missing",
    )
    before_missing_reads = _release_graph_state(database_path)

    with pytest.raises(ReleaseNotFoundError) as foreign_error:
        store.get_release(foreign_scope, target.release_id)
    with pytest.raises(ReleaseNotFoundError) as missing_error:
        store.get_release(missing_scope, target.release_id)
    assert type(foreign_error.value) is ReleaseNotFoundError
    assert str(foreign_error.value) == str(missing_error.value)
    assert store.list_releases(foreign_scope) == (foreign,)
    assert store.list_releases(missing_scope) == ()
    assert _release_graph_state(database_path) == before_missing_reads

    for absent_id in ("", " \t", "\x00"):
        with pytest.raises(ReleaseNotFoundError) as release_error:
            store.get_release(scope, absent_id)
        with pytest.raises(ReleaseNotFoundError) as members_error:
            store.get_release_revisions(scope, absent_id)
        expected_message = f"release {str.__str__(absent_id)!r} was not found"
        assert str(release_error.value) == expected_message
        assert str(members_error.value) == expected_message

    def transaction_must_not_start(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise AssertionError("validation reached SQLite I/O")

    scope_subclass = _SQLiteMemoryScopeSubclass(
        scope.tenant_id,
        scope.namespace,
        scope.subject_id,
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(
            sqlite_store_module,
            "_read_transaction",
            transaction_must_not_start,
        )
        for operation in (
            lambda: store.get_release(scope_subclass, target.release_id),
            lambda: store.get_release_revisions(scope_subclass, target.release_id),
            lambda: store.list_releases(scope_subclass),
        ):
            with pytest.raises(TypeError, match="scope must be a MemoryScope"):
                operation()
        for invalid_id in (None, 7, b"release"):
            with pytest.raises(TypeError, match="release_id must be a string"):
                store.get_release(scope, invalid_id)  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="release_id must be a string"):
                store.get_release_revisions(  # type: ignore[arg-type]
                    scope,
                    invalid_id,
                )
        with pytest.raises(ValueError, match="release_id must be valid UTF-8"):
            store.get_release(scope, "\ud800")
        with pytest.raises(ValueError, match="release_id must be valid UTF-8"):
            store.get_release_revisions(scope, "\ud800")


def test_sqlite_release_append_rejects_invalid_inputs_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-inputs.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-input-user")
    manifest = ReleaseManifest(scope, ())
    manifest_subclass = _SQLiteReleaseManifestSubclass(scope, ())

    def transaction_must_not_start(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise AssertionError("validation reached SQLite I/O")

    with monkeypatch.context() as guarded:
        guarded.setattr(
            sqlite_store_module,
            "_read_transaction",
            transaction_must_not_start,
        )
        guarded.setattr(
            sqlite_store_module,
            "_write_transaction",
            transaction_must_not_start,
        )
        with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
            store.append_release(
                manifest_subclass,
                idempotency_key="release-input-subclass",
            )
        with pytest.raises(TypeError, match="manifest must be a ReleaseManifest"):
            store.append_release(  # type: ignore[arg-type]
                object(),
                idempotency_key="release-input-object",
            )
        for invalid_key in (None, 7, b"release"):
            with pytest.raises(TypeError, match="idempotency_key must be a string"):
                store.append_release(
                    manifest,
                    idempotency_key=invalid_key,  # type: ignore[arg-type]
                )
        for blank_key in ("", " \t"):
            with pytest.raises(ValueError, match="idempotency_key must not be blank"):
                store.append_release(manifest, idempotency_key=blank_key)
        with pytest.raises(ValueError, match="idempotency_key must be valid UTF-8"):
            store.append_release(manifest, idempotency_key="\ud800")

    missing_manifest = ReleaseManifest(scope, ("rev_missing",))
    ensure_scope_calls = 0
    real_ensure_scope_id = sqlite_store_module._ensure_scope_id

    def counted_ensure_scope_id(
        cursor: sqlite3.Cursor,
        requested_scope: MemoryScope,
    ) -> int:
        nonlocal ensure_scope_calls
        ensure_scope_calls += 1
        return real_ensure_scope_id(cursor, requested_scope)

    with monkeypatch.context() as guarded:
        guarded.setattr(
            sqlite_store_module,
            "_ensure_scope_id",
            counted_ensure_scope_id,
        )
        with pytest.raises(RevisionNotFoundError) as missing_error:
            store.append_release(
                missing_manifest,
                idempotency_key="release-input-valid",
            )
    assert str(missing_error.value) == "revision 'rev_missing' was not found"
    assert ensure_scope_calls == 0
    assert _release_graph_state(database_path) == ((0, 0, 0, 0, 0, 0, 0, 0), [])

    key_probe = _SnapshotProbeStr("release-input-valid")
    release = store.append_release(manifest, idempotency_key=key_probe)

    assert key_probe.override_calls == 0
    assert release.manifest == manifest
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        alias_row = connection.execute(
            "SELECT typeof(idempotency_key), idempotency_key, release_id "
            "FROM memory_release_aliases"
        ).fetchone()
    finally:
        connection.close()
    assert alias_row == ("text", "release-input-valid", release.release_id)
    assert _release_graph_state(database_path) == (
        (1, 0, 0, 0, 0, 1, 1, 0),
        [],
    )


def test_sqlite_release_error_precedence_and_failures_leave_keys_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-precedence.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-precedence")
    foreign_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-precedence-foreign",
    )
    hidden_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-precedence-hidden",
    )
    winner, _winner_candidate, _winner_evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="precedence-winner",
    )
    sibling_sources = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index,
            key=f"precedence-sibling-{index}",
        )
        for index in range(1, 4)
    )
    sibling_parent = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=sibling_sources[0][0].candidate_id,
            idempotency_key="precedence-sibling-parent",
        )
    )
    left_sibling = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=sibling_sources[1][0].candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=sibling_parent.revision_id,
            idempotency_key="precedence-sibling-left",
        )
    )
    right_sibling = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=sibling_sources[2][0].candidate_id,
            operation=RevisionOperation.CONTRADICT,
            parent_revision_id=sibling_parent.revision_id,
            idempotency_key="precedence-sibling-right",
        )
    )
    foreign_revision, _foreign_candidate, _foreign_evidence = (
        _append_sqlite_release_root(
            store,
            foreign_scope,
            index=0,
            key="precedence-foreign",
        )
    )
    assert left_sibling.memory_id == right_sibling.memory_id
    assert left_sibling.revision_id != right_sibling.revision_id

    owner_manifest = ReleaseManifest(scope, (winner.revision_id,))
    owner_key = "release-precedence-owner"
    owner = store.append_release(owner_manifest, idempotency_key=owner_key)
    assert _release_graph_state(database_path) == (
        (2, 5, 5, 5, 5, 1, 1, 1),
        [],
    )

    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        ReleaseManifest(scope, ("rev_missing_same_key",)),
        idempotency_key=owner_key,
        error_type=ReleaseConflictError,
        message=("scoped release idempotency key already refers to different content"),
    )

    missing_after_duplicate_key = "release-missing-after-duplicate"
    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        ReleaseManifest(
            scope,
            (
                left_sibling.revision_id,
                right_sibling.revision_id,
                "rev_missing_after_duplicate",
            ),
        ),
        idempotency_key=missing_after_duplicate_key,
        error_type=RevisionNotFoundError,
        message="revision 'rev_missing_after_duplicate' was not found",
    )

    duplicate_manifest = ReleaseManifest(
        scope,
        (left_sibling.revision_id, right_sibling.revision_id),
    )
    collision_manifest = ReleaseManifest(scope, (left_sibling.revision_id,))
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(
            {
                duplicate_manifest.canonical_bytes(): owner.content_hash,
                collision_manifest.canonical_bytes(): owner.content_hash,
            }
        ),
    )
    duplicate_key = "release-duplicate-before-collision"
    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        duplicate_manifest,
        idempotency_key=duplicate_key,
        error_type=ReleaseConflictError,
        message=(
            "release contains more than one revision for memory_id "
            f"{left_sibling.memory_id!r}"
        ),
    )
    collision_key = "release-plain-collision"
    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        collision_manifest,
        idempotency_key=collision_key,
        error_type=ReleaseConflictError,
        message=f"release ID collision for {owner.release_id!r}",
    )

    foreign_key = "release-foreign-hidden"
    foreign_manifest = ReleaseManifest(
        scope,
        (foreign_revision.revision_id,),
    )
    foreign_error = _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        foreign_manifest,
        idempotency_key=foreign_key,
        error_type=RevisionNotFoundError,
        message=f"revision {foreign_revision.revision_id!r} was not found",
    )
    missing_database_path = tmp_path / "release-genuinely-missing.sqlite3"
    missing_store = SQLiteMemoryStore(missing_database_path)
    missing_owner = missing_store.append_release(
        ReleaseManifest(scope, ()),
        idempotency_key="release-missing-baseline",
    )
    missing_error = _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        missing_store,
        missing_database_path,
        foreign_manifest,
        idempotency_key=foreign_key,
        error_type=RevisionNotFoundError,
        message=f"revision {foreign_revision.revision_id!r} was not found",
    )
    assert type(foreign_error) is RevisionNotFoundError
    assert type(missing_error) is RevisionNotFoundError
    assert str(foreign_error).encode("utf-8") == str(missing_error).encode("utf-8")
    assert _release_graph_state(missing_database_path) == (
        (1, 0, 0, 0, 0, 1, 1, 0),
        [],
    )

    new_scope_key = "release-new-scope-missing"
    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        ReleaseManifest(hidden_scope, ("rev_missing_new_scope",)),
        idempotency_key=new_scope_key,
        error_type=RevisionNotFoundError,
        message="revision 'rev_missing_new_scope' was not found",
    )

    assert store.append_release(owner_manifest, idempotency_key=owner_key) == owner
    for recovered_key in (
        missing_after_duplicate_key,
        duplicate_key,
        collision_key,
    ):
        assert (
            store.append_release(owner_manifest, idempotency_key=recovered_key) == owner
        )
    assert store.append_release(owner_manifest, idempotency_key=foreign_key) == owner
    recovered_hidden = store.append_release(
        ReleaseManifest(hidden_scope, ()),
        idempotency_key=new_scope_key,
    )
    recovered_missing = missing_store.append_release(
        ReleaseManifest(scope, ()),
        idempotency_key=foreign_key,
    )

    assert recovered_hidden.manifest.revision_ids == ()
    assert recovered_missing == missing_owner
    assert store.list_releases(scope) == (owner,)
    assert store.list_releases(hidden_scope) == (recovered_hidden,)
    assert _release_graph_state(database_path) == (
        (3, 5, 5, 5, 5, 2, 6, 1),
        [],
    )
    assert _release_graph_state(missing_database_path) == (
        (1, 0, 0, 0, 0, 1, 2, 0),
        [],
    )


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
def test_sqlite_release_id_collision_is_scoped_atomic_and_loser_key_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    database_path = tmp_path / f"release-{collision_kind}-collision.sqlite3"
    store = SQLiteMemoryStore(database_path)
    first_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"release-{collision_kind}-first",
    )
    second_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"release-{collision_kind}-second",
    )
    winner_revision, _winner_candidate, _winner_evidence = _append_sqlite_release_root(
        store,
        first_scope,
        index=0,
        key=f"{collision_kind}-winner",
    )
    loser_revision, _loser_candidate, _loser_evidence = _append_sqlite_release_root(
        store,
        first_scope,
        index=1,
        key=f"{collision_kind}-loser",
    )
    cross_revision, _cross_candidate, _cross_evidence = _append_sqlite_release_root(
        store,
        second_scope,
        index=0,
        key=f"{collision_kind}-cross-scope",
    )
    winner_manifest = ReleaseManifest(first_scope, (winner_revision.revision_id,))
    loser_manifest = ReleaseManifest(first_scope, (loser_revision.revision_id,))
    cross_manifest = ReleaseManifest(second_scope, (cross_revision.revision_id,))
    shared_prefix = "e" * 24
    winner_digest = shared_prefix + "a" * 40
    loser_digest = (
        winner_digest if collision_kind == "full-hash" else shared_prefix + "b" * 40
    )
    digest_by_canonical = {
        winner_manifest.canonical_bytes(): winner_digest,
        loser_manifest.canonical_bytes(): loser_digest,
        cross_manifest.canonical_bytes(): winner_digest,
    }
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )
    shared_key = "release-shared-scoped-key"
    loser_key = "release-collision-loser"

    winner = store.append_release(winner_manifest, idempotency_key=shared_key)
    cross_scope = store.append_release(cross_manifest, idempotency_key=shared_key)
    assert winner.release_id == cross_scope.release_id == f"rel_{shared_prefix}"
    assert winner.content_hash == winner_digest
    assert cross_scope.content_hash == winner_digest
    before_collision = _release_graph_rows(database_path)
    assert _release_graph_state(database_path) == (
        (2, 3, 3, 3, 3, 2, 2, 2),
        [],
    )

    _assert_sqlite_release_append_failure_is_read_only(
        monkeypatch,
        store,
        database_path,
        loser_manifest,
        idempotency_key=loser_key,
        error_type=ReleaseConflictError,
        message=f"release ID collision for {winner.release_id!r}",
    )
    assert _release_graph_rows(database_path) == before_collision
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        loser_aliases = connection.execute(
            "SELECT scope_id, release_id FROM memory_release_aliases "
            "WHERE idempotency_key = ?",
            (loser_key,),
        ).fetchall()
    finally:
        connection.close()
    assert loser_aliases == []

    loser_canonical = loser_manifest.canonical_bytes()
    real_loser_digest = hashlib.sha256(loser_canonical).hexdigest()
    assert real_loser_digest[:24] != shared_prefix
    assert digest_by_canonical.pop(loser_canonical) == loser_digest
    assert digest_by_canonical == {
        winner_manifest.canonical_bytes(): winner_digest,
        cross_manifest.canonical_bytes(): winner_digest,
    }
    recovered = store.append_release(loser_manifest, idempotency_key=loser_key)

    assert recovered.release_id == f"rel_{real_loser_digest[:24]}"
    assert recovered.content_hash == real_loser_digest
    assert recovered.manifest == loser_manifest
    assert store.append_release(loser_manifest, idempotency_key=loser_key) == recovered
    assert store.get_release(first_scope, winner.release_id) == winner
    assert store.get_release(second_scope, cross_scope.release_id) == cross_scope
    assert store.list_releases(first_scope) == tuple(
        sorted((winner, recovered), key=lambda release: release.release_id)
    )
    assert store.list_releases(second_scope) == (cross_scope,)
    assert _release_graph_state(database_path) == (
        (2, 3, 3, 3, 3, 3, 3, 3),
        [],
    )


def test_sqlite_release_loader_rejects_scalar_projection_and_storage_drift(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "release-scalar-base.sqlite3"
    store = SQLiteMemoryStore(base_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-scalar")
    revision, _candidate, _evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="release-scalar",
    )
    manifest = ReleaseManifest(scope, (revision.revision_id,))
    release = store.append_release(
        manifest,
        idempotency_key="release-scalar-owner",
    )
    connection = sqlite3.connect(base_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
        scalar_row = connection.execute(
            sqlite_store_module._RELEASE_SELECT,
            (scope_id, release.release_id),
        ).fetchone()
    finally:
        connection.close()
    assert scalar_row is not None

    class StaticScalarCursor:
        def __init__(self, row: tuple[object, ...]) -> None:
            self._row = row

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> StaticScalarCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return self._row

    malformed_scalars: list[
        tuple[str, tuple[object, ...], str, type[ValueError] | type[TypeError]]
    ] = [
        (
            "wrong-arity",
            tuple(scalar_row[:-1]),
            "release row has the wrong field count",
            ValueError,
        )
    ]
    wrong_values = (
        b"release-id",
        "canonical-not-blob",
        b"content-hash",
        b"created-at",
        b"storage-hash",
    )
    for index, wrong_value in enumerate(wrong_values):
        changed_row = list(scalar_row)
        changed_row[index] = wrong_value
        malformed_scalars.append(
            (
                f"wrong-storage-{index}",
                tuple(changed_row),
                f"release row field {index} has the wrong storage class",
                TypeError,
            )
        )

    for case, row, expected_cause, cause_type in malformed_scalars:
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            sqlite_store_module._load_release(
                StaticScalarCursor(row),  # type: ignore[arg-type]
                scope,
                scope_id,
                release.release_id,
                (revision,),
            )
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == "stored release row failed integrity validation", (
            case
        )
        assert type(raised.value.__cause__) is cause_type, case
        assert str(raised.value.__cause__) == expected_cause, case

    with pytest.raises(MemoryPersistenceCorruptionError) as requested_id_error:
        sqlite_store_module._load_release(
            StaticScalarCursor(tuple(scalar_row)),  # type: ignore[arg-type]
            scope,
            scope_id,
            "rel_requested_id_drift",
            (revision,),
        )
    assert str(requested_id_error.value) == (
        "stored release row failed integrity validation"
    )
    assert type(requested_id_error.value.__cause__) is ValueError
    assert str(requested_id_error.value.__cause__) == (
        "loaded release ID differs from requested ID"
    )

    canonical_variant = b" \n" + manifest.canonical_bytes()
    assert canonical_variant != manifest.canonical_bytes()
    assert json.loads(canonical_variant) == json.loads(manifest.canonical_bytes())
    hash_suffix = "0" * 40 if release.content_hash[24:] != "0" * 40 else "1" * 40
    changed_content_hash = release.content_hash[:24] + hash_suffix
    moved_release_id = f"rel_{'0' * 24}"
    assert moved_release_id != release.release_id
    drift_cases = {
        "canonical": "canonical release bytes disagree with projections",
        "content-hash": "release content hash disagrees with canonical bytes",
        "release-id": "release ID disagrees with its content hash",
        "created-at-z": "release created_at is not exact UTC isoformat text",
        "storage-hash": "release storage hash disagrees with stored metadata",
    }

    for case, expected_cause in drift_cases.items():
        database_path = tmp_path / f"release-scalar-{case}.sqlite3"
        shutil.copyfile(base_path, database_path)
        connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            if case == "canonical":
                connection.execute(
                    "UPDATE memory_releases SET canonical = ? WHERE release_id = ?",
                    (sqlite3.Binary(canonical_variant), release.release_id),
                )
            elif case == "content-hash":
                connection.execute(
                    "UPDATE memory_releases SET content_hash = ? WHERE release_id = ?",
                    (changed_content_hash, release.release_id),
                )
            elif case == "release-id":
                binding_hash = sqlite_store_module._release_binding_hash(
                    scope=scope,
                    idempotency_key="release-scalar-owner",
                    release_id=moved_release_id,
                )
                connection.execute(
                    "UPDATE memory_release_revisions SET release_id = ? "
                    "WHERE scope_id = ? AND release_id = ?",
                    (moved_release_id, scope_id, release.release_id),
                )
                connection.execute(
                    "UPDATE memory_release_aliases "
                    "SET release_id = ?, binding_hash = ? "
                    "WHERE scope_id = ? AND release_id = ?",
                    (
                        moved_release_id,
                        binding_hash,
                        scope_id,
                        release.release_id,
                    ),
                )
                connection.execute(
                    "UPDATE memory_releases SET release_id = ? "
                    "WHERE scope_id = ? AND release_id = ?",
                    (moved_release_id, scope_id, release.release_id),
                )
            elif case == "created-at-z":
                connection.execute(
                    "UPDATE memory_releases SET created_at = ? WHERE release_id = ?",
                    (
                        release.created_at.isoformat().replace("+00:00", "Z"),
                        release.release_id,
                    ),
                )
            else:
                connection.execute(
                    "UPDATE memory_releases SET storage_hash = ? WHERE release_id = ?",
                    ("0" * 64, release.release_id),
                )
        finally:
            connection.close()

        corrupted_rows = _release_graph_rows(database_path)
        assert corrupted_rows[1] == (), case
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            SQLiteMemoryStore(database_path).get_release(scope, "rel_missing")
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == "stored release row failed integrity validation", (
            case
        )
        assert type(raised.value.__cause__) is ValueError, case
        assert str(raised.value.__cause__) == expected_cause, case
        assert _release_graph_rows(database_path) == corrupted_rows, case


def test_sqlite_release_snapshot_rejects_malformed_duplicate_and_orphan_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-malformed.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-malformed")
    first_root, _first_candidate, _first_evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="release-malformed-first",
    )
    parent, _parent_candidate, _parent_evidence = _append_sqlite_release_root(
        store,
        scope,
        index=1,
        key="release-malformed-parent",
    )
    left_candidate, _left_evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=2,
        key="release-malformed-left",
    )
    right_candidate, _right_evidence = _append_sqlite_revision_candidate(
        store,
        scope,
        index=3,
        key="release-malformed-right",
    )
    left = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=left_candidate.candidate_id,
            operation=RevisionOperation.REFINE,
            parent_revision_id=parent.revision_id,
            idempotency_key="release-malformed-left",
        )
    )
    right = store.append_revision(
        _make_sqlite_revision(
            scope=scope,
            candidate_id=right_candidate.candidate_id,
            operation=RevisionOperation.CONTRADICT,
            parent_revision_id=parent.revision_id,
            idempotency_key="release-malformed-right",
        )
    )
    assert left.memory_id == right.memory_id
    manifest = ReleaseManifest(scope, (first_root.revision_id, left.revision_id))
    release = store.append_release(
        manifest,
        idempotency_key="release-malformed-owner",
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
        member_rows = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT scope_id, release_id, position, revision_id, memory_id "
                "FROM memory_release_revisions"
            ).fetchall()
        )
    finally:
        connection.close()
    assert len(member_rows) == 2
    member_by_revision = {row[3]: row for row in member_rows}
    assert set(member_by_revision) == {first_root.revision_id, left.revision_id}
    baseline_rows = _release_graph_rows(database_path)
    assert baseline_rows[1] == ()
    address_sql = _normalize_sql("SELECT scope_id, release_id FROM memory_releases")
    member_sql = _normalize_sql(
        "SELECT scope_id, release_id, position, revision_id, memory_id "
        "FROM memory_release_revisions"
    )

    def run_case(
        case: str,
        sql: str,
        transform: Callable[
            [tuple[tuple[object, ...], ...], object],
            tuple[tuple[object, ...], ...],
        ],
        message: str,
    ) -> None:
        plan = _SQLiteReadOverridePlan(fetchall_transforms={sql: transform})
        with monkeypatch.context() as guarded:
            _install_sqlite_read_override_proxy(guarded, plan)
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                store.list_releases(scope)
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == message, case
        assert raised.value.__cause__ is None, case
        assert sum(query == sql for query, _parameters in plan.executions) == 1, case
        assert _release_graph_rows(database_path) == baseline_rows, case

    address_cases = (
        (
            "address-arity",
            lambda _rows, _parameters: ((scope_id,),),
            "release address row does not contain exactly two values",
        ),
        (
            "address-bool-scope",
            lambda _rows, _parameters: ((True, release.release_id),),
            "release address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "address-zero-scope",
            lambda _rows, _parameters: ((0, release.release_id),),
            "release address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "address-overflow-scope",
            lambda _rows, _parameters: ((2**63, release.release_id),),
            "release address contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "address-missing-scope",
            lambda _rows, _parameters: ((2**63 - 1, release.release_id),),
            "release address refers to a missing scope",
        ),
        (
            "address-nontext-release",
            lambda _rows, _parameters: ((scope_id, b"release-id"),),
            "release address contains a non-text identifier",
        ),
        (
            "address-duplicate",
            lambda rows, _parameters: (*rows, rows[0]),
            "release address appears multiple times",
        ),
    )
    for case, transform, message in address_cases:
        run_case(case, address_sql, transform, message)

    first_member = member_by_revision[first_root.revision_id]
    right_member = (
        scope_id,
        release.release_id,
        2,
        right.revision_id,
        right.memory_id,
    )
    member_cases = (
        (
            "member-arity",
            lambda _rows, _parameters: (tuple(first_member[:-1]),),
            "release member row does not contain exactly five values",
        ),
        (
            "member-nontext-id",
            lambda _rows, _parameters: (
                (scope_id, release.release_id, 0, b"revision-id", "memory-id"),
            ),
            "release member identifiers must be text",
        ),
        (
            "member-bool-scope",
            lambda _rows, _parameters: ((True, *first_member[1:]),),
            "release member contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "member-zero-scope",
            lambda _rows, _parameters: ((0, *first_member[1:]),),
            "release member contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "member-overflow-scope",
            lambda _rows, _parameters: ((2**63, *first_member[1:]),),
            "release member contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "member-bool-position",
            lambda _rows, _parameters: ((*first_member[:2], True, *first_member[3:]),),
            "release member position is not a non-negative signed 64-bit integer",
        ),
        (
            "member-overflow-position",
            lambda _rows, _parameters: ((*first_member[:2], 2**63, *first_member[3:]),),
            "release member position is not a non-negative signed 64-bit integer",
        ),
        (
            "member-negative-position",
            lambda _rows, _parameters: ((*first_member[:2], -1, *first_member[3:]),),
            "release member position is not a non-negative signed 64-bit integer",
        ),
        (
            "member-duplicate-position",
            lambda rows, _parameters: (
                *rows,
                (*right_member[:2], 0, *right_member[3:]),
            ),
            "release member position appears multiple times",
        ),
        (
            "member-duplicate-revision",
            lambda rows, _parameters: (
                *rows,
                (*first_member[:2], 2, *first_member[3:]),
            ),
            "release contains the same revision multiple times",
        ),
        (
            "member-duplicate-memory",
            lambda rows, _parameters: (*rows, right_member),
            "release contains the same memory multiple times",
        ),
        (
            "member-missing-owner",
            lambda rows, _parameters: (
                *rows,
                (
                    scope_id,
                    "rel_missing_owner",
                    0,
                    first_root.revision_id,
                    first_root.memory_id,
                ),
            ),
            "release member refers to a missing release",
        ),
        (
            "member-missing-revision",
            lambda rows, _parameters: (
                *rows,
                (
                    scope_id,
                    release.release_id,
                    2,
                    "rev_missing_member",
                    "mem_missing_member",
                ),
            ),
            "release member refers to a missing same-scope revision",
        ),
    )
    for case, transform, message in member_cases:
        run_case(case, member_sql, transform, message)

    def run_real_case(
        case: str,
        mutate: Callable[[sqlite3.Connection], None],
        message: str,
    ) -> None:
        case_path = tmp_path / f"release-malformed-{case}.sqlite3"
        shutil.copyfile(database_path, case_path)
        connection = sqlite3.connect(case_path, isolation_level=None)
        try:
            mutate(connection)
        finally:
            connection.close()
        corrupted_rows = _release_graph_rows(case_path)
        assert corrupted_rows != baseline_rows, case
        assert corrupted_rows[1] == (), case
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            SQLiteMemoryStore(case_path).list_releases(scope)
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == message, case
        assert raised.value.__cause__ is None, case
        assert _release_graph_rows(case_path) == corrupted_rows, case

    run_real_case(
        "member-gap",
        lambda connection: connection.execute(
            "UPDATE memory_release_revisions SET position = 3 "
            "WHERE scope_id = ? AND release_id = ? AND position = 0",
            (scope_id, release.release_id),
        ),
        "release member positions are not contiguous from zero",
    )
    run_real_case(
        "release-without-alias",
        lambda connection: connection.execute(
            "DELETE FROM memory_release_aliases WHERE scope_id = ? AND release_id = ?",
            (scope_id, release.release_id),
        ),
        "release exists without an idempotency alias",
    )


def test_sqlite_release_snapshot_rejects_member_relation_and_alias_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-relation-alias-drift.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope("tenant-1", "assistant-memory", "release-relation-drift")
    roots = tuple(
        _append_sqlite_release_root(
            store,
            scope,
            index=index,
            key=f"release-relation-{index}",
        )[0]
        for index in range(3)
    )
    owner_key = "release-relation-owner"
    other_key = "release-relation-other"
    owner_manifest = ReleaseManifest(
        scope,
        (roots[0].revision_id, roots[1].revision_id),
    )
    owner = store.append_release(owner_manifest, idempotency_key=owner_key)
    other = store.append_release(
        ReleaseManifest(scope, (roots[2].revision_id,)),
        idempotency_key=other_key,
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_id = connection.execute(
            "SELECT scope_id FROM memory_scopes WHERE tenant_id = ? "
            "AND namespace = ? AND subject_id = ?",
            (scope.tenant_id, scope.namespace, scope.subject_id),
        ).fetchone()[0]
        member_rows = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT scope_id, release_id, position, revision_id, memory_id "
                "FROM memory_release_revisions"
            ).fetchall()
        )
        alias_rows = tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT scope_id, idempotency_key, release_id, binding_hash "
                "FROM memory_release_aliases"
            ).fetchall()
        )
    finally:
        connection.close()
    owner_members = tuple(row for row in member_rows if row[1] == owner.release_id)
    assert len(owner_members) == 2
    owner_alias = next(row for row in alias_rows if row[1] == owner_key)
    baseline_rows = _release_graph_rows(database_path)
    assert baseline_rows[1] == ()
    member_sql = _normalize_sql(
        "SELECT scope_id, release_id, position, revision_id, memory_id "
        "FROM memory_release_revisions"
    )
    alias_sql = _normalize_sql(
        "SELECT scope_id, idempotency_key, release_id, binding_hash "
        "FROM memory_release_aliases"
    )

    def run_proxy_case(
        case: str,
        sql: str,
        transform: Callable[
            [tuple[tuple[object, ...], ...], object],
            tuple[tuple[object, ...], ...],
        ],
        message: str,
    ) -> None:
        plan = _SQLiteReadOverridePlan(fetchall_transforms={sql: transform})
        with monkeypatch.context() as guarded:
            _install_sqlite_read_override_proxy(guarded, plan)
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                store.get_release(scope, owner.release_id)
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == message, case
        assert raised.value.__cause__ is None, case
        assert sum(query == sql for query, _parameters in plan.executions) == 1, case
        assert _release_graph_rows(database_path) == baseline_rows, case

    def run_real_case(
        case: str,
        mutate: Callable[[sqlite3.Connection], None],
        message: str,
        cause: str | None = None,
    ) -> None:
        case_path = tmp_path / f"release-relation-alias-{case}.sqlite3"
        shutil.copyfile(database_path, case_path)
        connection = sqlite3.connect(case_path, isolation_level=None)
        try:
            mutate(connection)
        finally:
            connection.close()
        corrupted_rows = _release_graph_rows(case_path)
        assert corrupted_rows != baseline_rows, case
        assert corrupted_rows[1] == (), case
        with pytest.raises(MemoryPersistenceCorruptionError) as raised:
            SQLiteMemoryStore(case_path).get_release(scope, owner.release_id)
        assert type(raised.value) is MemoryPersistenceCorruptionError, case
        assert str(raised.value) == message, case
        if cause is None:
            assert raised.value.__cause__ is None, case
        else:
            assert type(raised.value.__cause__) is ValueError, case
            assert str(raised.value.__cause__) == cause, case
        assert _release_graph_rows(case_path) == corrupted_rows, case

    run_real_case(
        "member-deletion",
        lambda connection: connection.execute(
            "DELETE FROM memory_release_revisions "
            "WHERE scope_id = ? AND release_id = ? AND position = 1",
            (scope_id, owner.release_id),
        ),
        "stored release row failed integrity validation",
        "canonical release bytes disagree with projections",
    )
    spare = roots[2]
    run_real_case(
        "member-substitution",
        lambda connection: connection.execute(
            "UPDATE memory_release_revisions "
            "SET revision_id = ?, memory_id = ? "
            "WHERE scope_id = ? AND release_id = ? AND position = 1",
            (spare.revision_id, spare.memory_id, scope_id, owner.release_id),
        ),
        "stored release row failed integrity validation",
        "canonical release bytes disagree with projections",
    )

    def reorder_members(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE memory_release_revisions SET position = position + 100 "
            "WHERE scope_id = ? AND release_id = ?",
            (scope_id, owner.release_id),
        )
        connection.execute(
            "UPDATE memory_release_revisions "
            "SET position = CASE position WHEN 100 THEN 1 WHEN 101 THEN 0 END "
            "WHERE scope_id = ? AND release_id = ?",
            (scope_id, owner.release_id),
        )

    run_real_case(
        "member-reorder",
        reorder_members,
        "stored release row failed integrity validation",
        "canonical release bytes disagree with projections",
    )
    memory_mismatch = (
        *owner_members[0][:4],
        roots[1].memory_id,
    )
    run_proxy_case(
        "member-memory-mismatch",
        member_sql,
        lambda rows, _parameters: tuple(
            memory_mismatch if row == owner_members[0] else row for row in rows
        ),
        "release member memory ID differs from its revision",
    )

    def replace_owner_alias(
        rows: tuple[tuple[object, ...], ...],
        replacement: tuple[object, ...],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(replacement if row[1] == owner_key else row for row in rows)

    alias_cases = (
        (
            "alias-arity",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                tuple(owner_alias[:-1]),
            ),
            "release alias row does not contain exactly four values",
        ),
        (
            "alias-bool-scope",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (True, *owner_alias[1:]),
            ),
            "release alias contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "alias-zero-scope",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (0, *owner_alias[1:]),
            ),
            "release alias contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "alias-overflow-scope",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (2**63, *owner_alias[1:]),
            ),
            "release alias contains an invalid positive signed 64-bit scope ID",
        ),
        (
            "alias-nontext-key",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (scope_id, b"owner-key", *owner_alias[2:]),
            ),
            "release alias values must be text",
        ),
        (
            "alias-nontext-release",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (*owner_alias[:2], b"release-id", owner_alias[3]),
            ),
            "release alias values must be text",
        ),
        (
            "alias-nontext-binding",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (*owner_alias[:3], b"binding"),
            ),
            "release alias values must be text",
        ),
        (
            "alias-duplicate-key",
            lambda rows, _parameters: (*rows, owner_alias),
            "release idempotency key appears multiple times in one scope",
        ),
        (
            "alias-missing-scope",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (2**63 - 1, *owner_alias[1:]),
            ),
            "release alias refers to a missing scope",
        ),
        (
            "alias-missing-target",
            lambda rows, _parameters: replace_owner_alias(
                rows,
                (*owner_alias[:2], "rel_missing_target", owner_alias[3]),
            ),
            "release alias refers to a missing same-scope release",
        ),
    )
    for case, transform, message in alias_cases:
        run_proxy_case(case, alias_sql, transform, message)

    run_real_case(
        "alias-blank-key",
        lambda connection: connection.execute(
            "UPDATE memory_release_aliases SET idempotency_key = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            (" \t", scope_id, owner_key),
        ),
        "stored release alias failed integrity validation",
        "idempotency_key must not be blank",
    )
    run_real_case(
        "alias-changed-key",
        lambda connection: connection.execute(
            "UPDATE memory_release_aliases SET idempotency_key = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            ("release-relation-moved", scope_id, owner_key),
        ),
        "release alias binding hash disagrees with stored metadata",
    )
    run_real_case(
        "alias-target-without-binding",
        lambda connection: connection.execute(
            "UPDATE memory_release_aliases SET release_id = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            (other.release_id, scope_id, owner_key),
        ),
        "release alias binding hash disagrees with stored metadata",
    )
    run_real_case(
        "alias-binding-drift",
        lambda connection: connection.execute(
            "UPDATE memory_release_aliases SET binding_hash = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            ("0" * 64, scope_id, owner_key),
        ),
        "release alias binding hash disagrees with stored metadata",
    )


def test_sqlite_release_snapshot_validates_unrelated_graph_first(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "release-unrelated-base.sqlite3"
    store = SQLiteMemoryStore(base_path)
    target_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-unrelated-target",
    )
    target_revision, _target_candidate, _target_evidence = _append_sqlite_release_root(
        store,
        target_scope,
        index=0,
        key="release-unrelated-target",
    )
    target_manifest = ReleaseManifest(
        target_scope,
        (target_revision.revision_id,),
    )
    target_key = "release-unrelated-target"
    target_release = store.append_release(
        target_manifest,
        idempotency_key=target_key,
    )
    lower_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-unrelated-lower",
    )
    lower_revision, _lower_candidate, _lower_evidence = _append_sqlite_release_root(
        store,
        lower_scope,
        index=0,
        key="release-unrelated-lower",
    )
    graph_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-unrelated-graph",
    )
    graph_revision, _graph_candidate, _graph_evidence = _append_sqlite_release_root(
        store,
        graph_scope,
        index=0,
        key="release-unrelated-graph",
    )
    graph_key = "release-unrelated-graph"
    graph_release = store.append_release(
        ReleaseManifest(graph_scope, (graph_revision.revision_id,)),
        idempotency_key=graph_key,
    )
    assert _release_graph_rows(base_path)[1] == ()
    corruption_expectations: dict[str, tuple[str, str | None]] = {
        "lower": (
            "stored revision row failed integrity validation",
            "revision ID disagrees with its content hash",
        ),
        "release": (
            "stored release row failed integrity validation",
            "release storage hash disagrees with stored metadata",
        ),
        "member": (
            "stored release row failed integrity validation",
            "canonical release bytes disagree with projections",
        ),
        "alias": (
            "release alias binding hash disagrees with stored metadata",
            None,
        ),
    }

    for corruption_kind in ("lower", "release", "member", "alias"):
        for operation in (
            "missing-get",
            "list",
            "member-get",
            "exact-retry",
            "alias-only-append",
        ):
            database_path = tmp_path / (
                f"release-unrelated-{corruption_kind}-{operation}.sqlite3"
            )
            shutil.copyfile(base_path, database_path)
            connection = sqlite3.connect(database_path, isolation_level=None)
            try:
                connection.execute("PRAGMA foreign_keys = OFF")
                if corruption_kind == "lower":
                    moved_id = f"rev_{'0' * 24}"
                    assert moved_id != lower_revision.revision_id
                    connection.execute(
                        "UPDATE memory_revisions SET revision_id = ? "
                        "WHERE revision_id = ?",
                        (moved_id, lower_revision.revision_id),
                    )
                elif corruption_kind == "release":
                    connection.execute(
                        "UPDATE memory_releases SET storage_hash = ? "
                        "WHERE release_id = ?",
                        ("0" * 64, graph_release.release_id),
                    )
                elif corruption_kind == "member":
                    connection.execute(
                        "DELETE FROM memory_release_revisions WHERE release_id = ?",
                        (graph_release.release_id,),
                    )
                else:
                    connection.execute(
                        "UPDATE memory_release_aliases SET binding_hash = ? "
                        "WHERE idempotency_key = ?",
                        ("0" * 64, graph_key),
                    )
            finally:
                connection.close()

            corrupted_rows = _release_graph_rows(database_path)
            assert corrupted_rows[1] == (), (corruption_kind, operation)
            corrupted_store = SQLiteMemoryStore(database_path)
            with pytest.raises(MemoryPersistenceCorruptionError) as raised:
                if operation == "missing-get":
                    corrupted_store.get_release(target_scope, "rel_missing")
                elif operation == "list":
                    corrupted_store.list_releases(target_scope)
                elif operation == "member-get":
                    corrupted_store.get_release_revisions(
                        target_scope,
                        target_release.release_id,
                    )
                elif operation == "exact-retry":
                    corrupted_store.append_release(
                        target_manifest,
                        idempotency_key=target_key,
                    )
                else:
                    corrupted_store.append_release(
                        target_manifest,
                        idempotency_key="release-unrelated-new-alias",
                    )

            expected_outer, expected_cause = corruption_expectations[corruption_kind]
            assert type(raised.value) is MemoryPersistenceCorruptionError
            assert str(raised.value) == expected_outer, (corruption_kind, operation)
            if expected_cause is None:
                assert raised.value.__cause__ is None, (corruption_kind, operation)
            else:
                assert type(raised.value.__cause__) is ValueError
                assert str(raised.value.__cause__) == expected_cause
            assert _release_graph_rows(database_path) == corrupted_rows


def test_sqlite_release_snapshot_loads_each_scalar_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-scalar-once.sqlite3"
    store = SQLiteMemoryStore(database_path)
    first_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-scalar-once-first",
    )
    second_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-scalar-once-second",
    )
    first_roots = tuple(
        _append_sqlite_release_root(
            store,
            first_scope,
            index=index,
            key=f"release-scalar-once-first-{index}",
        )[0]
        for index in range(3)
    )
    second_root, _second_candidate, _second_evidence = _append_sqlite_release_root(
        store,
        second_scope,
        index=0,
        key="release-scalar-once-second",
    )
    first_manifest = ReleaseManifest(
        first_scope,
        (first_roots[0].revision_id, first_roots[1].revision_id),
    )
    first = store.append_release(
        first_manifest,
        idempotency_key="release-scalar-once-first",
    )
    assert (
        store.append_release(
            first_manifest,
            idempotency_key="release-scalar-once-first-alias",
        )
        == first
    )
    second = store.append_release(
        ReleaseManifest(first_scope, (first_roots[2].revision_id,)),
        idempotency_key="release-scalar-once-second-release",
    )
    foreign = store.append_release(
        ReleaseManifest(second_scope, (second_root.revision_id,)),
        idempotency_key="release-scalar-once-foreign",
    )
    expected = tuple(sorted((first, second), key=lambda item: item.release_id))
    release_select_sql = _normalize_sql(sqlite_store_module._RELEASE_SELECT)
    address_sql = _normalize_sql("SELECT scope_id, release_id FROM memory_releases")
    member_sql = _normalize_sql(
        "SELECT scope_id, release_id, position, revision_id, memory_id "
        "FROM memory_release_revisions"
    )
    alias_sql = _normalize_sql(
        "SELECT scope_id, idempotency_key, release_id, binding_hash "
        "FROM memory_release_aliases"
    )
    plan = _SQLiteReadOverridePlan()
    _install_sqlite_read_override_proxy(monkeypatch, plan)

    assert store.list_releases(first_scope) == expected
    assert sum(sql == address_sql for sql, _parameters in plan.executions) == 1
    assert sum(sql == member_sql for sql, _parameters in plan.executions) == 1
    assert sum(sql == alias_sql for sql, _parameters in plan.executions) == 1
    scalar_parameters = tuple(
        parameters for sql, parameters in plan.executions if sql == release_select_sql
    )
    assert len(scalar_parameters) == 3
    assert len(set(scalar_parameters)) == 3
    assert all(type(parameters) is tuple for parameters in scalar_parameters)
    assert {parameters[1] for parameters in scalar_parameters} == {
        first.release_id,
        second.release_id,
        foreign.release_id,
    }


@pytest.mark.parametrize(
    "failure_stage",
    ["core", "second-member", "first-alias", "alias-only"],
)
def test_sqlite_release_insert_failures_roll_back_every_stage_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    database_path = tmp_path / f"release-failure-{failure_stage}.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"release-failure-{failure_stage}",
    )
    revisions = (
        ()
        if failure_stage == "core"
        else tuple(
            _append_sqlite_release_root(
                store,
                scope,
                index=index,
                key=f"release-failure-{failure_stage}-{index}",
            )[0]
            for index in range(3)
        )
    )
    manifest = ReleaseManifest(
        scope,
        tuple(revision.revision_id for revision in revisions),
    )
    owner = None
    idempotency_key = f"release-failure-{failure_stage}-target"
    if failure_stage == "alias-only":
        owner = store.append_release(
            manifest,
            idempotency_key="release-failure-alias-owner",
        )
    baseline_rows = _release_graph_rows(database_path)
    assert baseline_rows[1] == ()
    if failure_stage == "core":
        assert baseline_rows[0] == ((), (), (), (), (), (), (), ())
    marker, occurrence = {
        "core": ("INSERT INTO memory_releases", 1),
        "second-member": ("INSERT INTO memory_release_revisions", 2),
        "first-alias": ("INSERT INTO memory_release_aliases", 1),
        "alias-only": ("INSERT INTO memory_release_aliases", 1),
    }[failure_stage]
    expected_insert_counts = {
        "core": {"core": 1, "member": 0, "alias": 0},
        "second-member": {"core": 1, "member": 2, "alias": 0},
        "first-alias": {"core": 1, "member": 3, "alias": 1},
        "alias-only": {"core": 0, "member": 0, "alias": 1},
    }[failure_stage]
    injected = sqlite3.OperationalError(
        f"injected release failure after {failure_stage}"
    )
    injected.sqlite_errorcode = sqlite3.SQLITE_IOERR
    plan = _SQLiteFailurePlan(
        after_statement=marker,
        after_occurrence=occurrence,
        after_statement_error=injected,
    )
    _install_sqlite_failure_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceError) as raised:
        store.append_release(manifest, idempotency_key=idempotency_key)

    assert type(raised.value) is MemoryPersistenceError
    assert raised.value.__cause__ is injected
    normalized_marker = _normalize_sql(marker)
    assert plan.events[-6].startswith(f"executed:{normalized_marker}")
    assert plan.events[-5].startswith(f"fail-after:{normalized_marker}")
    assert plan.events[-4:] == [
        "attempt:ROLLBACK",
        "executed:ROLLBACK",
        "attempt:CLOSE",
        "executed:CLOSE",
    ]
    assert "attempt:COMMIT" not in plan.events
    assert "executed:COMMIT" not in plan.events
    insert_counts = {
        "core": sum(
            event.startswith("executed:INSERT INTO MEMORY_RELEASES ")
            for event in plan.events
        ),
        "member": sum(
            event.startswith("executed:INSERT INTO MEMORY_RELEASE_REVISIONS ")
            for event in plan.events
        ),
        "alias": sum(
            event.startswith("executed:INSERT INTO MEMORY_RELEASE_ALIASES ")
            for event in plan.events
        ),
    }
    assert insert_counts == expected_insert_counts
    assert _release_graph_rows(database_path) == baseline_rows

    monkeypatch.undo()
    recovered = store.append_release(manifest, idempotency_key=idempotency_key)

    if owner is not None:
        assert recovered == owner
        expected_aliases = 2
    else:
        assert recovered.manifest == manifest
        expected_aliases = 1
    assert store.get_release_revisions(scope, recovered.release_id) == revisions
    assert (
        SQLiteMemoryStore(database_path).get_release(
            scope,
            recovered.release_id,
        )
        == recovered
    )
    expected_state = (
        (1, 0, 0, 0, 0, 1, 1, 0)
        if failure_stage == "core"
        else (1, 3, 3, 3, 3, 1, expected_aliases, 3)
    )
    assert _release_graph_state(database_path) == (expected_state, [])


def test_sqlite_release_post_insert_global_readback_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-post-insert-readback.sqlite3"
    store = SQLiteMemoryStore(database_path)
    unrelated_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-readback-unrelated-alias",
    )
    target_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-readback-target-new",
    )
    unrelated_revision, _unrelated_candidate, _unrelated_evidence = (
        _append_sqlite_release_root(
            store,
            unrelated_scope,
            index=0,
            key="release-readback-unrelated-alias",
        )
    )
    unrelated_key = "release-readback-unrelated-owner"
    unrelated = store.append_release(
        ReleaseManifest(unrelated_scope, (unrelated_revision.revision_id,)),
        idempotency_key=unrelated_key,
    )
    target_revisions = tuple(
        _append_sqlite_release_root(
            store,
            target_scope,
            index=index,
            key=f"release-readback-target-new-{index}",
        )[0]
        for index in range(2)
    )
    target_manifest = ReleaseManifest(
        target_scope,
        tuple(revision.revision_id for revision in target_revisions),
    )
    target_key = "release-readback-target-new"
    target_release_id = (
        f"rel_{hashlib.sha256(target_manifest.canonical_bytes()).hexdigest()[:24]}"
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_rows = connection.execute(
            "SELECT scope_id, subject_id FROM memory_scopes"
        ).fetchall()
        original_alias = connection.execute(
            "SELECT release_id, binding_hash FROM memory_release_aliases "
            "WHERE idempotency_key = ?",
            (unrelated_key,),
        ).fetchone()
    finally:
        connection.close()
    scope_id_by_subject = {subject_id: scope_id for scope_id, subject_id in scope_rows}
    unrelated_scope_id = scope_id_by_subject[unrelated_scope.subject_id]
    target_scope_id = scope_id_by_subject[target_scope.subject_id]
    assert original_alias == (
        unrelated.release_id,
        sqlite_store_module._release_binding_hash(
            scope=unrelated_scope,
            idempotency_key=unrelated_key,
            release_id=unrelated.release_id,
        ),
    )
    baseline_rows = _release_graph_rows(database_path)
    assert baseline_rows[1] == ()

    def tamper_unrelated_alias(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "UPDATE memory_release_aliases SET binding_hash = ? "
            "WHERE scope_id = ? AND idempotency_key = ?",
            ("0" * 64, unrelated_scope_id, unrelated_key),
        )
        assert cursor.rowcount == 1
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []

    plan = _SQLiteReleaseTamperPlan(
        trigger_key=target_key,
        tamper=tamper_unrelated_alias,
        observed_sql={
            _normalize_sql(
                "SELECT scope_id, release_id FROM memory_releases"
            ): "address-scan",
            _normalize_sql(
                "SELECT scope_id, release_id, position, revision_id, memory_id "
                "FROM memory_release_revisions"
            ): "member-scan",
            _normalize_sql(sqlite_store_module._RELEASE_SELECT): "scalar",
            _normalize_sql(
                "SELECT scope_id, idempotency_key, release_id, binding_hash "
                "FROM memory_release_aliases"
            ): "alias-scan",
        },
    )
    _install_sqlite_release_tamper_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.append_release(target_manifest, idempotency_key=target_key)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert str(raised.value) == (
        "release alias binding hash disagrees with stored metadata"
    )
    assert raised.value.__cause__ is None
    assert plan.trigger_hits == plan.tamper_hits == 1
    assert plan.events[:4] == [
        "alias-insert",
        "tamper",
        "address-scan",
        "member-scan",
    ]
    assert plan.events.count("scalar") == 2
    assert plan.events[-3:] == ["alias-scan", "rollback", "close"]
    assert "commit" not in plan.events
    assert _release_graph_rows(database_path) == baseline_rows
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        restored_alias = connection.execute(
            "SELECT release_id, binding_hash FROM memory_release_aliases "
            "WHERE scope_id = ? AND idempotency_key = ?",
            (unrelated_scope_id, unrelated_key),
        ).fetchone()
        target_rows = connection.execute(
            "SELECT release_id FROM memory_releases "
            "WHERE scope_id = ? AND release_id = ?",
            (target_scope_id, target_release_id),
        ).fetchall()
        target_aliases = connection.execute(
            "SELECT release_id FROM memory_release_aliases "
            "WHERE scope_id = ? AND idempotency_key = ?",
            (target_scope_id, target_key),
        ).fetchall()
    finally:
        connection.close()
    assert restored_alias == original_alias
    assert target_rows == []
    assert target_aliases == []

    monkeypatch.undo()
    successful_snapshots: list[Any] = []
    real_load_release_snapshot = sqlite_store_module._load_release_snapshot

    def record_successful_snapshot(cursor: sqlite3.Cursor) -> Any:
        snapshot = real_load_release_snapshot(cursor)
        successful_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        sqlite_store_module,
        "_load_release_snapshot",
        record_successful_snapshot,
    )
    recovered = store.append_release(target_manifest, idempotency_key=target_key)

    assert len(successful_snapshots) == 2
    preflight_releases = successful_snapshots[0][2]
    post_write_releases = successful_snapshots[1][2]
    assert target_release_id not in {
        release.release_id for release in preflight_releases.values()
    }
    post_write_release = next(
        release
        for release in post_write_releases.values()
        if release.release_id == target_release_id
    )
    assert recovered is post_write_release
    assert recovered.release_id == target_release_id
    assert (
        store.append_release(target_manifest, idempotency_key=target_key) == recovered
    )
    assert len(successful_snapshots) == 3
    monkeypatch.undo()
    assert store.get_release_revisions(target_scope, target_release_id) == (
        target_revisions
    )
    assert store.get_release(unrelated_scope, unrelated.release_id) == unrelated
    assert _release_graph_state(database_path) == (
        (2, 3, 3, 3, 3, 2, 2, 3),
        [],
    )


def test_sqlite_release_alias_only_global_readback_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "release-alias-only-readback.sqlite3"
    store = SQLiteMemoryStore(database_path)
    unrelated_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-readback-unrelated-member",
    )
    target_scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-readback-target-alias",
    )
    unrelated_revisions = tuple(
        _append_sqlite_release_root(
            store,
            unrelated_scope,
            index=index,
            key=f"release-readback-unrelated-member-{index}",
        )[0]
        for index in range(2)
    )
    unrelated = store.append_release(
        ReleaseManifest(
            unrelated_scope,
            tuple(revision.revision_id for revision in unrelated_revisions),
        ),
        idempotency_key="release-readback-unrelated-member-owner",
    )
    target_revision, _target_candidate, _target_evidence = _append_sqlite_release_root(
        store,
        target_scope,
        index=0,
        key="release-readback-target-alias",
    )
    target_manifest = ReleaseManifest(target_scope, (target_revision.revision_id,))
    target_owner_key = "release-readback-target-alias-owner"
    target = store.append_release(
        target_manifest,
        idempotency_key=target_owner_key,
    )
    target_alias_key = "release-readback-target-alias-new"
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        scope_rows = connection.execute(
            "SELECT scope_id, subject_id FROM memory_scopes"
        ).fetchall()
        original_members = connection.execute(
            "SELECT position, revision_id, memory_id "
            "FROM memory_release_revisions WHERE release_id = ? "
            "ORDER BY position",
            (unrelated.release_id,),
        ).fetchall()
    finally:
        connection.close()
    scope_id_by_subject = {subject_id: scope_id for scope_id, subject_id in scope_rows}
    unrelated_scope_id = scope_id_by_subject[unrelated_scope.subject_id]
    target_scope_id = scope_id_by_subject[target_scope.subject_id]
    assert original_members == [
        (position, revision.revision_id, revision.memory_id)
        for position, revision in enumerate(unrelated_revisions)
    ]
    baseline_rows = _release_graph_rows(database_path)
    assert baseline_rows[1] == ()

    def tamper_unrelated_member(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "UPDATE memory_release_revisions SET position = 2 "
            "WHERE scope_id = ? AND release_id = ? AND position = 0",
            (unrelated_scope_id, unrelated.release_id),
        )
        assert cursor.rowcount == 1
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchall() == []

    plan = _SQLiteReleaseTamperPlan(
        trigger_key=target_alias_key,
        tamper=tamper_unrelated_member,
        observed_sql={
            _normalize_sql(
                "SELECT scope_id, release_id FROM memory_releases"
            ): "address-scan",
            _normalize_sql(
                "SELECT scope_id, release_id, position, revision_id, memory_id "
                "FROM memory_release_revisions"
            ): "member-scan",
            _normalize_sql(
                "SELECT scope_id, idempotency_key, release_id, binding_hash "
                "FROM memory_release_aliases"
            ): "alias-scan",
        },
    )
    _install_sqlite_release_tamper_proxy(monkeypatch, plan)

    with pytest.raises(MemoryPersistenceCorruptionError) as raised:
        store.append_release(target_manifest, idempotency_key=target_alias_key)

    assert type(raised.value) is MemoryPersistenceCorruptionError
    assert str(raised.value) == (
        "release member positions are not contiguous from zero"
    )
    assert raised.value.__cause__ is None
    assert plan.trigger_hits == plan.tamper_hits == 1
    assert plan.events == [
        "alias-insert",
        "tamper",
        "address-scan",
        "member-scan",
        "rollback",
        "close",
    ]
    assert "alias-scan" not in plan.events
    assert "commit" not in plan.events
    assert _release_graph_rows(database_path) == baseline_rows
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        restored_members = connection.execute(
            "SELECT position, revision_id, memory_id "
            "FROM memory_release_revisions "
            "WHERE scope_id = ? AND release_id = ? ORDER BY position",
            (unrelated_scope_id, unrelated.release_id),
        ).fetchall()
        target_aliases = connection.execute(
            "SELECT release_id FROM memory_release_aliases "
            "WHERE scope_id = ? AND idempotency_key = ?",
            (target_scope_id, target_alias_key),
        ).fetchall()
    finally:
        connection.close()
    assert restored_members == original_members
    assert target_aliases == []

    monkeypatch.undo()
    successful_snapshots: list[Any] = []
    real_load_release_snapshot = sqlite_store_module._load_release_snapshot

    def record_successful_snapshot(cursor: sqlite3.Cursor) -> Any:
        snapshot = real_load_release_snapshot(cursor)
        successful_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        sqlite_store_module,
        "_load_release_snapshot",
        record_successful_snapshot,
    )
    recovered = store.append_release(
        target_manifest,
        idempotency_key=target_alias_key,
    )

    assert len(successful_snapshots) == 2
    preflight_release = next(
        release
        for release in successful_snapshots[0][2].values()
        if release.release_id == target.release_id
    )
    post_write_release = next(
        release
        for release in successful_snapshots[1][2].values()
        if release.release_id == target.release_id
    )
    assert preflight_release is not post_write_release
    assert recovered is post_write_release
    assert recovered is not preflight_release
    assert recovered == target
    assert (
        store.append_release(
            target_manifest,
            idempotency_key=target_alias_key,
        )
        == target
    )
    assert len(successful_snapshots) == 3
    monkeypatch.undo()
    assert (
        store.get_release_revisions(
            unrelated_scope,
            unrelated.release_id,
        )
        == unrelated_revisions
    )
    assert _release_graph_state(database_path) == (
        (2, 3, 3, 3, 3, 2, 3, 3),
        [],
    )


def test_sqlite_concurrent_identical_release_requests_converge(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-concurrent-identical.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-concurrent-identical",
    )
    revision, _candidate, _evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="release-concurrent-identical",
    )
    manifest = ReleaseManifest(scope, (revision.revision_id,))
    requests = tuple(
        (manifest, "release-concurrent-identical-key")
        for _index in range(_SQLITE_RELEASE_RACE_SIZE)
    )

    outcomes = _run_sqlite_release_race(database_path, requests)

    assert not any(isinstance(item, MemoryPersistenceError) for item in outcomes)
    assert all(type(item) is MemoryRelease for item in outcomes)
    releases = tuple(item for item in outcomes if type(item) is MemoryRelease)
    assert len(releases) == _SQLITE_RELEASE_RACE_SIZE
    winner = releases[0]
    assert all(release == winner for release in releases)
    assert {release.release_id for release in releases} == {winner.release_id}
    assert {release.content_hash for release in releases} == {winner.content_hash}
    assert {release.created_at for release in releases} == {winner.created_at}
    assert {release.manifest for release in releases} == {manifest}
    assert _release_graph_state(database_path) == (
        (1, 1, 1, 1, 1, 1, 1, 1),
        [],
    )

    fresh = SQLiteMemoryStore(database_path)
    persisted = fresh.get_release(scope, winner.release_id)
    assert persisted == winner
    assert persisted is not winner
    assert fresh.get_release_revisions(scope, winner.release_id) == (revision,)
    assert fresh.list_releases(scope) == (persisted,)


def test_sqlite_concurrent_alias_keys_bind_one_release(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-concurrent-aliases.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-concurrent-aliases",
    )
    revision, _candidate, _evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="release-concurrent-aliases",
    )
    manifest = ReleaseManifest(scope, (revision.revision_id,))
    keys = tuple(
        f"release-concurrent-alias-{index}"
        for index in range(_SQLITE_RELEASE_RACE_SIZE)
    )
    requests = tuple((manifest, key) for key in keys)

    outcomes = _run_sqlite_release_race(database_path, requests)

    assert not any(isinstance(item, MemoryPersistenceError) for item in outcomes)
    assert all(type(item) is MemoryRelease for item in outcomes)
    releases = tuple(item for item in outcomes if type(item) is MemoryRelease)
    assert len(releases) == _SQLITE_RELEASE_RACE_SIZE
    winner = releases[0]
    assert all(release == winner for release in releases)
    assert _release_graph_state(database_path) == (
        (1, 1, 1, 1, 1, 1, _SQLITE_RELEASE_RACE_SIZE, 1),
        [],
    )

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        alias_rows = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases "
            "ORDER BY idempotency_key"
        ).fetchall()
    finally:
        connection.close()
    assert alias_rows == [(key, winner.release_id) for key in sorted(keys)]

    fresh = SQLiteMemoryStore(database_path)
    assert fresh.list_releases(scope) == (winner,)
    assert fresh.get_release_revisions(scope, winner.release_id) == (revision,)
    for key in keys:
        assert fresh.append_release(manifest, idempotency_key=key) == winner
    assert _release_graph_state(database_path) == (
        (1, 1, 1, 1, 1, 1, _SQLITE_RELEASE_RACE_SIZE, 1),
        [],
    )


def test_sqlite_concurrent_shared_key_has_one_winner_and_losers_recover(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-concurrent-shared-key.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-concurrent-shared-key",
    )
    revisions = tuple(
        _append_sqlite_release_root(
            store,
            scope,
            index=index,
            key=f"release-concurrent-shared-key-{index}",
        )[0]
        for index in range(_SQLITE_RELEASE_RACE_SIZE)
    )
    manifests = tuple(
        ReleaseManifest(scope, (revision.revision_id,)) for revision in revisions
    )
    shared_key = "release-concurrent-shared-key-owner"
    requests = tuple((manifest, shared_key) for manifest in manifests)

    outcomes = _run_sqlite_release_race(database_path, requests)

    persistence_errors = tuple(
        item for item in outcomes if isinstance(item, MemoryPersistenceError)
    )
    winners = tuple(item for item in outcomes if type(item) is MemoryRelease)
    conflicts = tuple(item for item in outcomes if type(item) is ReleaseConflictError)
    unexpected = tuple(
        item
        for item in outcomes
        if type(item) not in {MemoryRelease, ReleaseConflictError}
    )
    assert persistence_errors == ()
    assert unexpected == ()
    assert len(winners) == 1
    assert len(conflicts) == _SQLITE_RELEASE_RACE_SIZE - 1
    assert all(
        str(conflict)
        == "scoped release idempotency key already refers to different content"
        for conflict in conflicts
    )
    winner = winners[0]
    assert _release_graph_state(database_path) == (
        (1, 6, 6, 6, 6, 1, 1, 1),
        [],
    )
    loser_pairs = tuple(
        (manifest, revision)
        for manifest, revision in zip(manifests, revisions, strict=True)
        if manifest != winner.manifest
    )
    assert len(loser_pairs) == _SQLITE_RELEASE_RACE_SIZE - 1

    recovery_store = SQLiteMemoryStore(database_path)
    recovered = tuple(
        recovery_store.append_release(
            manifest,
            idempotency_key=f"release-concurrent-recovered-{index}",
        )
        for index, (manifest, _revision) in enumerate(loser_pairs)
    )

    assert len(recovered) == _SQLITE_RELEASE_RACE_SIZE - 1
    stored = recovery_store.list_releases(scope)
    assert len(stored) == _SQLITE_RELEASE_RACE_SIZE
    assert {release.manifest for release in stored} == set(manifests)
    for manifest, revision in loser_pairs:
        release = next(item for item in stored if item.manifest == manifest)
        assert recovery_store.get_release_revisions(scope, release.release_id) == (
            revision,
        )
    assert _release_graph_state(database_path) == (
        (1, 6, 6, 6, 6, 6, 6, 6),
        [],
    )


def test_sqlite_concurrent_sibling_revisions_survive_in_separate_releases(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "release-concurrent-siblings.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        "release-concurrent-siblings",
    )
    parent, _parent_candidate, _parent_evidence = _append_sqlite_release_root(
        store,
        scope,
        index=0,
        key="release-concurrent-sibling-parent",
    )
    candidates = tuple(
        _append_sqlite_revision_candidate(
            store,
            scope,
            index=index + 1,
            key=f"release-concurrent-sibling-{index}",
        )[0]
        for index in range(_SQLITE_RELEASE_RACE_SIZE)
    )
    operations = (
        RevisionOperation.REFINE,
        RevisionOperation.SUPERSEDE,
        RevisionOperation.CONTRADICT,
    )
    siblings = tuple(
        store.append_revision(
            _make_sqlite_revision(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=operations[index % len(operations)],
                parent_revision_id=parent.revision_id,
                idempotency_key=f"release-concurrent-sibling-revision-{index}",
            )
        )
        for index, candidate in enumerate(candidates)
    )
    assert {revision.memory_id for revision in siblings} == {parent.memory_id}
    manifests = tuple(
        ReleaseManifest(scope, (revision.revision_id,)) for revision in siblings
    )
    requests = tuple(
        (manifest, f"release-concurrent-sibling-key-{index}")
        for index, manifest in enumerate(manifests)
    )

    outcomes = _run_sqlite_release_race(database_path, requests)

    assert not any(isinstance(item, MemoryPersistenceError) for item in outcomes)
    assert all(type(item) is MemoryRelease for item in outcomes)
    releases = tuple(item for item in outcomes if type(item) is MemoryRelease)
    assert len(releases) == _SQLITE_RELEASE_RACE_SIZE
    assert len({release.release_id for release in releases}) == (
        _SQLITE_RELEASE_RACE_SIZE
    )
    assert {release.manifest for release in releases} == set(manifests)
    assert _release_graph_state(database_path) == (
        (1, 7, 7, 7, 7, 6, 6, 6),
        [],
    )

    fresh = SQLiteMemoryStore(database_path)
    stored = fresh.list_releases(scope)
    assert len(stored) == _SQLITE_RELEASE_RACE_SIZE
    assert {release.manifest for release in stored} == set(manifests)
    for release in stored:
        (revision_id,) = release.manifest.revision_ids
        revision = next(item for item in siblings if item.revision_id == revision_id)
        assert fresh.get_release_revisions(scope, release.release_id) == (revision,)
    assert {revision.memory_id for revision in siblings} == {parent.memory_id}


@pytest.mark.parametrize("collision_kind", ["full-hash", "id-prefix"])
def test_sqlite_concurrent_release_id_collision_is_atomic_and_loser_keys_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    database_path = tmp_path / f"release-concurrent-{collision_kind}.sqlite3"
    store = SQLiteMemoryStore(database_path)
    scope = MemoryScope(
        "tenant-1",
        "assistant-memory",
        f"release-concurrent-{collision_kind}",
    )
    revisions = tuple(
        _append_sqlite_release_root(
            store,
            scope,
            index=index,
            key=f"release-concurrent-{collision_kind}-{index}",
        )[0]
        for index in range(_SQLITE_RELEASE_RACE_SIZE)
    )
    manifests = tuple(
        ReleaseManifest(scope, (revision.revision_id,)) for revision in revisions
    )
    keys = tuple(
        f"release-concurrent-{collision_kind}-key-{index}"
        for index in range(_SQLITE_RELEASE_RACE_SIZE)
    )
    shared_prefix = "d" * 24
    digest_by_canonical = {
        manifest.canonical_bytes(): (
            shared_prefix + "a" * 40
            if collision_kind == "full-hash"
            else shared_prefix + f"{index + 1:040x}"
        )
        for index, manifest in enumerate(manifests)
    }
    assert len(set(digest_by_canonical)) == _SQLITE_RELEASE_RACE_SIZE
    if collision_kind == "full-hash":
        assert len(set(digest_by_canonical.values())) == 1
    else:
        assert len(set(digest_by_canonical.values())) == _SQLITE_RELEASE_RACE_SIZE
    monkeypatch.setattr(
        sqlite_store_module,
        "sha256",
        _stable_digest_oracle(digest_by_canonical),
    )
    requests = tuple(zip(manifests, keys, strict=True))

    outcomes = _run_sqlite_release_race(database_path, requests)

    persistence_errors = tuple(
        item for item in outcomes if isinstance(item, MemoryPersistenceError)
    )
    winners = tuple(item for item in outcomes if type(item) is MemoryRelease)
    conflicts = tuple(item for item in outcomes if type(item) is ReleaseConflictError)
    unexpected = tuple(
        item
        for item in outcomes
        if type(item) not in {MemoryRelease, ReleaseConflictError}
    )
    assert persistence_errors == ()
    assert unexpected == ()
    assert len(winners) == 1
    assert len(conflicts) == _SQLITE_RELEASE_RACE_SIZE - 1
    winner = winners[0]
    assert winner.release_id == f"rel_{shared_prefix}"
    assert winner.content_hash == digest_by_canonical[winner.manifest.canonical_bytes()]
    assert all(
        str(conflict) == f"release ID collision for {winner.release_id!r}"
        for conflict in conflicts
    )
    winner_index = manifests.index(winner.manifest)
    winner_key = keys[winner_index]
    assert _release_graph_state(database_path) == (
        (1, 6, 6, 6, 6, 1, 1, 1),
        [],
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        alias_rows = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases"
        ).fetchall()
    finally:
        connection.close()
    assert alias_rows == [(winner_key, winner.release_id)]

    winner_canonical = winner.manifest.canonical_bytes()
    loser_requests = tuple(
        (manifest, key) for manifest, key in requests if manifest != winner.manifest
    )
    for manifest, _key in loser_requests:
        forced_digest = digest_by_canonical.pop(manifest.canonical_bytes())
        assert forced_digest.startswith(shared_prefix)
        assert (
            not hashlib.sha256(manifest.canonical_bytes())
            .hexdigest()
            .startswith(shared_prefix)
        )
    assert digest_by_canonical == {
        winner_canonical: winner.content_hash,
    }
    recovery_store = SQLiteMemoryStore(database_path)
    recovered = tuple(
        recovery_store.append_release(manifest, idempotency_key=key)
        for manifest, key in loser_requests
    )

    assert len(recovered) == _SQLITE_RELEASE_RACE_SIZE - 1
    assert len({release.release_id for release in recovered}) == (
        _SQLITE_RELEASE_RACE_SIZE - 1
    )
    assert all(release.release_id != winner.release_id for release in recovered)
    stored = recovery_store.list_releases(scope)
    assert len(stored) == _SQLITE_RELEASE_RACE_SIZE
    assert {release.manifest for release in stored} == set(manifests)
    for manifest, key in requests:
        expected = next(item for item in stored if item.manifest == manifest)
        assert (
            recovery_store.append_release(
                manifest,
                idempotency_key=key,
            )
            == expected
        )
    assert _release_graph_state(database_path) == (
        (1, 6, 6, 6, 6, 6, 6, 6),
        [],
    )
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        alias_rows = connection.execute(
            "SELECT idempotency_key, release_id FROM memory_release_aliases "
            "ORDER BY idempotency_key"
        ).fetchall()
    finally:
        connection.close()
    expected_alias_rows = sorted(
        (
            key,
            next(
                release.release_id for release in stored if release.manifest == manifest
            ),
        )
        for manifest, key in requests
    )
    assert alias_rows == expected_alias_rows
