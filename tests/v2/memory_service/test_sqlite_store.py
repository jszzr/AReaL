# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable SQLite Memory Service backend."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import areal.v2.memory_service._sqlite_backend as sqlite_backend
import areal.v2.memory_service.sqlite_store as sqlite_store_module
from areal.v2.memory_service.errors import (
    EvidenceConflictError,
    EvidenceNotFoundError,
    MemoryPersistenceBusyError,
    MemoryPersistenceCorruptionError,
    MemoryPersistenceError,
    MemoryPersistenceSchemaError,
    MemoryServiceError,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
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


class _SQLiteMemoryScopeSubclass(MemoryScope):
    pass


class _SQLiteEvidenceEventSubclass(EvidenceEvent):
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

    __hash__ = str.__hash__


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


def test_sqlite_evidence_missing_loader_result_routes_by_known_context(
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
    with pytest.raises(EvidenceNotFoundError, match=record.evidence_id):
        store.get(event.scope, record.evidence_id)
    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="idempotency index refers to a missing row",
    ):
        store.append(event)
    with pytest.raises(
        MemoryPersistenceCorruptionError,
        match="evidence listing refers to a missing row",
    ):
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
                "SELECT EVIDENCE_ID FROM MEMORY_EVIDENCE WHERE SCOPE_ID = ?"
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


def test_sqlite_evidence_scope_lookup_requires_positive_signed_64_bit_id() -> None:
    scope = MemoryScope("tenant-1", "assistant-memory", "scope-user")

    class ScopeCursor:
        def __init__(self, stored_scope_id: object) -> None:
            self._stored_scope_id = stored_scope_id

        def execute(
            self,
            _sql: str,
            _parameters: object = (),
        ) -> ScopeCursor:
            return self

        def fetchone(self) -> tuple[object, ...]:
            return (self._stored_scope_id,)

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
    rows = [None]
    if persisted_scope_id is not None:
        rows.append((persisted_scope_id,))

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


def test_new_database_has_exact_v1_header_catalog_and_metadata(
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
        assert len(sqlite_backend._SCHEMA_DDL) == 11
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
        "PRAGMA USER_VERSION = 1"
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
        "PRAGMA user_version = 1",
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
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
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
        "PRAGMA user_version = 1",
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


@pytest.mark.parametrize("version", [2, 2**31 - 1])
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
        "DROP INDEX idx_memory_revisions_sort",
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


def test_v1_switched_to_wal_is_rejected_without_conversion(tmp_path: Path) -> None:
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
        sqlite_backend._release_binding_hash(
            scope=scope,
            idempotency_key="alias-a",
            release_id="rel_a",
        )
        == "f61c2a3ef34cfe876c675f843c18508e7134c64b3fed251e48ad9885459e71aa"
    )

    assert seen == [
        b'{"content_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","created_at":"2026-07-08T00:00:00+00:00","record_id":"evd_a","record_kind":"evidence","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"content_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","created_at":"2026-07-08T00:00:00+00:00","generation":7,"memory_id":"mem_a","record_id":"rev_a","record_kind":"revision","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
        b'{"idempotency_key":"alias-a","record_kind":"release_alias","release_id":"rel_a","schema_version":1,"scope":{"namespace":"assistant-memory","subject_id":"user-1","tenant_id":"tenant-1"}}',
    ]


@pytest.mark.parametrize("record_kind", ["evidence", "candidate", "release"])
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
