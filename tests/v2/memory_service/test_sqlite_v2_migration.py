# SPDX-License-Identifier: Apache-2.0

"""Failure-oriented tests for exact SQLite v1/v2 to v3 migration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

import areal.v2.memory_service._sqlite_backend as sqlite_backend
from areal.v2.memory_service.errors import MemoryPersistenceSchemaError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore
from areal.v2.memory_service.types import EvidenceEvent, EvidenceKind, MemoryScope

_V1_SCHEMA_SPEC_GOLDEN = (
    "445a839fb37b9db29842f018f75887debcbb96997a1391b73068ea23e2f355c0"
)
_V1_SCHEMA_CATALOG_GOLDEN = (
    "713b6273425eee792360134b25ded70d88b8e4aba92fef256c566d322d41d7a6"
)
_V2_SCHEMA_SPEC_GOLDEN = (
    "28ce8fc78297c4d4b046ccd7be272b5d4de7318005fe277c8383f44e83ea9a0a"
)
_V1_OBJECT_NAMES = frozenset(
    {
        "idx_memory_evidence_sort",
        "idx_memory_revisions_sort",
        "memory_candidate_evidence",
        "memory_candidates",
        "memory_evidence",
        "memory_release_aliases",
        "memory_release_revisions",
        "memory_releases",
        "memory_revisions",
        "memory_schema_metadata",
        "memory_scopes",
    }
)

_CATALOG_SQL = """SELECT type, name, tbl_name, sql
FROM main.sqlite_master
WHERE sql IS NOT NULL
  AND substr(name, 1, 7) <> 'sqlite_'
ORDER BY type COLLATE BINARY, name COLLATE BINARY,
         tbl_name COLLATE BINARY, sql COLLATE BINARY"""


def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _catalog_rows(cursor: sqlite3.Cursor) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row) for row in cursor.execute(_CATALOG_SQL).fetchall())


def _catalog_hash(cursor: sqlite3.Cursor) -> str:
    return hashlib.sha256(_compact_json_bytes(_catalog_rows(cursor))).hexdigest()


@dataclass(frozen=True, slots=True)
class _V1EvidenceSeed:
    scope_id: int
    event: EvidenceEvent
    created_at: datetime

    @property
    def canonical(self) -> bytes:
        return self.event.canonical_bytes()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical).hexdigest()

    @property
    def evidence_id(self) -> str:
        return f"evd_{self.content_hash[:24]}"


def _create_exact_v1(
    database_path: Path,
    seeds: Sequence[_V1EvidenceSeed] = (),
) -> None:
    """Create v1 directly from its frozen DDL, without touching v2 code paths."""

    encoded_ddl = _compact_json_bytes(sqlite_backend._SCHEMA_V1_DDL)
    assert len(sqlite_backend._SCHEMA_V1_DDL) == 11
    assert hashlib.sha256(encoded_ddl).hexdigest() == _V1_SCHEMA_SPEC_GOLDEN
    assert sqlite_backend._SCHEMA_V1_SPEC_HASH == _V1_SCHEMA_SPEC_GOLDEN

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("BEGIN IMMEDIATE")
        for statement in sqlite_backend._SCHEMA_V1_DDL:
            cursor.execute(statement)

        catalog_hash = _catalog_hash(cursor)
        assert catalog_hash == _V1_SCHEMA_CATALOG_GOLDEN
        cursor.execute(
            "INSERT INTO memory_schema_metadata "
            "(singleton, schema_spec_hash, schema_catalog_hash) VALUES (?, ?, ?)",
            (1, _V1_SCHEMA_SPEC_GOLDEN, catalog_hash),
        )
        cursor.execute(
            f"PRAGMA application_id = {sqlite_backend._APPLICATION_ID}"
        )
        cursor.execute(f"PRAGMA user_version = {sqlite_backend._SCHEMA_V1_VERSION}")

        scopes_by_id: dict[int, MemoryScope] = {}
        for seed in seeds:
            existing = scopes_by_id.setdefault(seed.scope_id, seed.event.scope)
            assert existing == seed.event.scope
        for scope_id, scope in sorted(scopes_by_id.items()):
            cursor.execute(
                "INSERT INTO memory_scopes "
                "(scope_id, tenant_id, namespace, subject_id) VALUES (?, ?, ?, ?)",
                (scope_id, scope.tenant_id, scope.namespace, scope.subject_id),
            )
        for seed in seeds:
            created_at_text = seed.created_at.astimezone(UTC).isoformat()
            cursor.execute(
                """INSERT INTO memory_evidence (
    scope_id, evidence_id, canonical, content_hash, created_at,
    storage_hash, session_id, run_id, sequence_no, kind, payload,
    observed_at, idempotency_key
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    seed.scope_id,
                    seed.evidence_id,
                    seed.canonical,
                    seed.content_hash,
                    created_at_text,
                    sqlite_backend._record_storage_hash(
                        record_kind="evidence",
                        scope=seed.event.scope,
                        record_id=seed.evidence_id,
                        content_hash=seed.content_hash,
                        created_at_text=created_at_text,
                    ),
                    seed.event.session_id,
                    seed.event.run_id,
                    seed.event.sequence_no,
                    seed.event.kind.value,
                    seed.event.payload,
                    seed.event.observed_at.isoformat(),
                    seed.event.idempotency_key,
                ),
            )
        cursor.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _create_exact_v2(database_path: Path) -> None:
    """Create an empty exact v2 database without executing v3 additions."""

    assert hashlib.sha256(
        _compact_json_bytes(sqlite_backend._SCHEMA_V2_DDL)
    ).hexdigest() == _V2_SCHEMA_SPEC_GOLDEN
    assert sqlite_backend._SCHEMA_V2_SPEC_HASH == _V2_SCHEMA_SPEC_GOLDEN
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("BEGIN IMMEDIATE")
        for statement in sqlite_backend._SCHEMA_V2_DDL:
            cursor.execute(statement)
        catalog_hash = _catalog_hash(cursor)
        cursor.execute(
            "INSERT INTO memory_schema_metadata "
            "(singleton, schema_spec_hash, schema_catalog_hash) VALUES (?, ?, ?)",
            (1, _V2_SCHEMA_SPEC_GOLDEN, catalog_hash),
        )
        cursor.execute(
            f"PRAGMA application_id = {sqlite_backend._APPLICATION_ID}"
        )
        cursor.execute(f"PRAGMA user_version = {sqlite_backend._SCHEMA_V2_VERSION}")
        cursor.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _create_populated_exact_v2(
    database_path: Path,
    seeds: Sequence[_V1EvidenceSeed],
) -> None:
    """Create populated v2 state while stopping before v3 additions."""

    _create_exact_v1(database_path, seeds)
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("BEGIN EXCLUSIVE")
        sqlite_backend._migrate_v1_to_v2_locked(cursor)
        cursor.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _object_names(cursor: sqlite3.Cursor) -> frozenset[str]:
    return frozenset(row[1] for row in _catalog_rows(cursor))


def _assert_exact_v1(database_path: Path) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        assert cursor.execute("PRAGMA application_id").fetchone() == (
            sqlite_backend._APPLICATION_ID,
        )
        assert cursor.execute("PRAGMA user_version").fetchone() == (1,)
        assert _object_names(cursor) == _V1_OBJECT_NAMES
        assert _catalog_hash(cursor) == _V1_SCHEMA_CATALOG_GOLDEN
        assert cursor.execute(
            "SELECT schema_spec_hash, schema_catalog_hash "
            "FROM memory_schema_metadata WHERE singleton = 1"
        ).fetchone() == (_V1_SCHEMA_SPEC_GOLDEN, _V1_SCHEMA_CATALOG_GOLDEN)
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def _assert_exact_v2(database_path: Path) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        assert cursor.execute("PRAGMA application_id").fetchone() == (
            sqlite_backend._APPLICATION_ID,
        )
        assert cursor.execute("PRAGMA user_version").fetchone() == (2,)
        assert _object_names(cursor) == (
            sqlite_backend._V2_REQUIRED_TABLES
            | sqlite_backend._V2_REQUIRED_INDEXES
        )
        assert cursor.execute(
            "SELECT schema_spec_hash, schema_catalog_hash "
            "FROM memory_schema_metadata WHERE singleton = 1"
        ).fetchone() == (
            sqlite_backend._SCHEMA_V2_SPEC_HASH,
            _catalog_hash(cursor),
        )
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def _assert_exact_v3(database_path: Path) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        assert cursor.execute("PRAGMA application_id").fetchone() == (
            sqlite_backend._APPLICATION_ID,
        )
        assert cursor.execute("PRAGMA user_version").fetchone() == (3,)
        assert _object_names(cursor) == (
            sqlite_backend._REQUIRED_TABLES | sqlite_backend._REQUIRED_INDEXES
        )
        assert cursor.execute(
            "SELECT schema_spec_hash, schema_catalog_hash "
            "FROM memory_schema_metadata WHERE singleton = 1"
        ).fetchone() == (
            sqlite_backend._SCHEMA_SPEC_HASH,
            _catalog_hash(cursor),
        )
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def _event(scope: MemoryScope, index: int) -> EvidenceEvent:
    instant = datetime(2026, 7, 11, 8, 0, tzinfo=UTC) + timedelta(minutes=index)
    return EvidenceEvent(
        scope=scope,
        session_id=f"session-{index}",
        run_id=f"run-{index}",
        sequence_no=index,
        kind=EvidenceKind.USER_MESSAGE if index % 2 == 0 else EvidenceKind.FEEDBACK,
        payload=f"migration evidence {index}",
        observed_at=instant,
        idempotency_key=f"v1-evidence-{index}",
    )


def _read_ingest_rows(database_path: Path) -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT ingest_order, scope_id, evidence_id, binding_hash "
                "FROM memory_evidence_ingest_orders ORDER BY ingest_order"
            ).fetchall()
        )
    finally:
        connection.close()


def test_exact_v1_fixture_is_bound_to_frozen_schema_goldens(tmp_path: Path) -> None:
    database_path = tmp_path / "exact-v1.sqlite3"

    _create_exact_v1(database_path)

    _assert_exact_v1(database_path)


def test_empty_exact_v1_migrates_to_v3_and_reopens(tmp_path: Path) -> None:
    database_path = tmp_path / "empty-v1.sqlite3"
    _create_exact_v1(database_path)

    SQLiteMemoryStore(database_path)

    _assert_exact_v3(database_path)
    assert _read_ingest_rows(database_path) == ()
    SQLiteMemoryStore(database_path)
    _assert_exact_v3(database_path)


def test_empty_exact_v2_migrates_to_v3_and_reopens(tmp_path: Path) -> None:
    database_path = tmp_path / "empty-v2.sqlite3"
    _create_exact_v2(database_path)

    SQLiteMemoryStore(database_path)

    _assert_exact_v3(database_path)
    assert _read_ingest_rows(database_path) == ()
    SQLiteMemoryStore(database_path)
    _assert_exact_v3(database_path)


def test_populated_exact_v2_migrates_without_rewriting_evidence_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "populated-v2.sqlite3"
    scope = MemoryScope("tenant", "agent-memory", "populated-v2")
    instant = datetime(2026, 7, 11, 7, 0, tzinfo=UTC)
    seeds = (
        _V1EvidenceSeed(5, _event(scope, 1), instant),
        _V1EvidenceSeed(5, _event(scope, 2), instant + timedelta(seconds=1)),
    )
    _create_populated_exact_v2(database_path, seeds)
    _assert_exact_v2(database_path)
    before = _read_ingest_rows(database_path)

    store = SQLiteMemoryStore(database_path)

    _assert_exact_v3(database_path)
    assert _read_ingest_rows(database_path) == before
    assert {item.evidence_id for item in store.list(scope)} == {
        seed.evidence_id for seed in seeds
    }


def test_multiscope_backfill_is_deterministic_and_survives_reopen(
    tmp_path: Path,
) -> None:
    scope_a = MemoryScope("tenant-a", "agent-memory", "subject-a")
    scope_b = MemoryScope("tenant-b", "agent-memory", "subject-b")
    created_at = datetime(2026, 7, 11, 9, 0, tzinfo=UTC)
    seeds = (
        _V1EvidenceSeed(41, _event(scope_a, 3), created_at + timedelta(seconds=2)),
        _V1EvidenceSeed(7, _event(scope_b, 1), created_at),
        _V1EvidenceSeed(41, _event(scope_a, 2), created_at),
        _V1EvidenceSeed(7, _event(scope_b, 4), created_at + timedelta(seconds=1)),
    )
    forward_path = tmp_path / "forward.sqlite3"
    reverse_path = tmp_path / "reverse.sqlite3"
    _create_exact_v1(forward_path, seeds)
    _create_exact_v1(reverse_path, tuple(reversed(seeds)))

    SQLiteMemoryStore(forward_path)
    SQLiteMemoryStore(reverse_path)

    forward_rows = _read_ingest_rows(forward_path)
    reverse_rows = _read_ingest_rows(reverse_path)
    assert forward_rows == reverse_rows
    expected_seeds = sorted(
        seeds,
        key=lambda seed: (
            seed.created_at.astimezone(UTC).isoformat(),
            seed.evidence_id,
            seed.scope_id,
        ),
    )
    assert tuple(row[:3] for row in forward_rows) == tuple(
        (ingest_order, seed.scope_id, seed.evidence_id)
        for ingest_order, seed in enumerate(expected_seeds)
    )
    assert tuple(row[3] for row in forward_rows) == tuple(
        sqlite_backend._evidence_ingest_binding_hash(
            scope=seed.event.scope,
            evidence_id=seed.evidence_id,
            ingest_order=ingest_order,
        )
        for ingest_order, seed in enumerate(expected_seeds)
    )

    reopened = SQLiteMemoryStore(forward_path)
    assert _read_ingest_rows(forward_path) == forward_rows
    assert {record.evidence_id for record in reopened.list(scope_a)} == {
        seed.evidence_id for seed in seeds if seed.event.scope == scope_a
    }
    assert {record.evidence_id for record in reopened.list(scope_b)} == {
        seed.evidence_id for seed in seeds if seed.event.scope == scope_b
    }


def test_corrupt_v1_is_rejected_without_partial_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "corrupt-v1.sqlite3"
    _create_exact_v1(database_path)
    corrupt_catalog_hash = "0" * 64
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE memory_schema_metadata SET schema_catalog_hash = ? "
            "WHERE singleton = 1",
            (corrupt_catalog_hash,),
        )
    finally:
        connection.close()

    with pytest.raises(
        MemoryPersistenceSchemaError,
        match="schema catalog fingerprint does not match",
    ):
        SQLiteMemoryStore(database_path)

    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        cursor = connection.cursor()
        assert cursor.execute("PRAGMA user_version").fetchone() == (1,)
        assert _object_names(cursor) == _V1_OBJECT_NAMES
        assert cursor.execute(
            "SELECT schema_spec_hash, schema_catalog_hash "
            "FROM memory_schema_metadata WHERE singleton = 1"
        ).fetchone() == (_V1_SCHEMA_SPEC_GOLDEN, corrupt_catalog_hash)
    finally:
        connection.close()


@dataclass(slots=True)
class _OneShotFailure:
    fired: bool = False
    target_prefix: str = "INSERT INTO MEMORY_EVIDENCE_INGEST_ORDERS"
    message: str = "injected migration failure after ingest insert"


class _FailingCursor:
    def __init__(self, real_cursor: sqlite3.Cursor, state: _OneShotFailure) -> None:
        self._real_cursor = real_cursor
        self._state = state

    def execute(self, sql: str, parameters: object = ()) -> _FailingCursor:
        self._real_cursor.execute(sql, parameters)
        normalized = " ".join(sql.split()).upper()
        if (
            not self._state.fired
            and normalized.startswith(self._state.target_prefix)
        ):
            self._state.fired = True
            raise RuntimeError(self._state.message)
        return self

    def __iter__(self) -> Any:
        return iter(self._real_cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_cursor, name)


class _FailingConnection:
    def __init__(
        self,
        real_connection: sqlite3.Connection,
        state: _OneShotFailure,
    ) -> None:
        self._real_connection = real_connection
        self._state = state

    def cursor(self, *args: object, **kwargs: object) -> _FailingCursor:
        return _FailingCursor(
            self._real_connection.cursor(*args, **kwargs),
            self._state,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_connection, name)


def test_migration_failure_rolls_back_atomically_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "retry-v1.sqlite3"
    scope = MemoryScope("tenant", "agent-memory", "subject")
    instant = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    seeds = (
        _V1EvidenceSeed(9, _event(scope, 1), instant),
        _V1EvidenceSeed(9, _event(scope, 2), instant + timedelta(seconds=1)),
    )
    _create_exact_v1(database_path, seeds)
    state = _OneShotFailure()
    real_connect = sqlite_backend._connect

    def failing_connect(path: str) -> _FailingConnection:
        return _FailingConnection(real_connect(path), state)

    monkeypatch.setattr(sqlite_backend, "_connect", failing_connect)
    with pytest.raises(
        RuntimeError,
        match="injected migration failure after ingest insert",
    ):
        SQLiteMemoryStore(database_path)

    assert state.fired
    _assert_exact_v1(database_path)

    monkeypatch.setattr(sqlite_backend, "_connect", real_connect)
    SQLiteMemoryStore(database_path)
    _assert_exact_v3(database_path)
    assert tuple(row[0] for row in _read_ingest_rows(database_path)) == (0, 1)


@pytest.mark.parametrize(
    "marker",
    tuple(
        " ".join(statement.split()).upper().split(" (")[0]
        for statement in sqlite_backend._SCHEMA_V3_ADDITIONS
    )
    + (
        "UPDATE MEMORY_SCHEMA_METADATA SET SCHEMA_SPEC_HASH = ?",
        "PRAGMA USER_VERSION = 3",
    ),
)
def test_every_v2_to_v3_migration_boundary_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    database_path = tmp_path / f"v2-v3-boundary-{hashlib.sha256(marker.encode()).hexdigest()[:8]}.sqlite3"
    _create_exact_v2(database_path)
    _assert_exact_v2(database_path)
    state = _OneShotFailure(
        target_prefix=marker,
        message=f"injected v2 to v3 failure after {marker}",
    )
    real_connect = sqlite_backend._connect

    def failing_connect(path: str) -> _FailingConnection:
        return _FailingConnection(real_connect(path), state)

    monkeypatch.setattr(sqlite_backend, "_connect", failing_connect)
    with pytest.raises(RuntimeError, match="injected v2 to v3 failure"):
        SQLiteMemoryStore(database_path)

    assert state.fired
    _assert_exact_v2(database_path)
    monkeypatch.setattr(sqlite_backend, "_connect", real_connect)
    SQLiteMemoryStore(database_path)
    _assert_exact_v3(database_path)


def test_concurrent_constructors_converge_on_one_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "concurrent-v1.sqlite3"
    scope_a = MemoryScope("tenant-a", "memory", "subject-a")
    scope_b = MemoryScope("tenant-b", "memory", "subject-b")
    instant = datetime(2026, 7, 11, 11, 0, tzinfo=UTC)
    seeds = (
        _V1EvidenceSeed(3, _event(scope_a, 1), instant),
        _V1EvidenceSeed(8, _event(scope_b, 2), instant + timedelta(seconds=1)),
    )
    _create_exact_v1(database_path, seeds)
    barrier = Barrier(2)

    def construct() -> SQLiteMemoryStore:
        barrier.wait()
        return SQLiteMemoryStore(database_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(construct) for _ in range(2))
        stores = tuple(future.result(timeout=10) for future in futures)

    assert len(stores) == 2
    _assert_exact_v3(database_path)
    rows = _read_ingest_rows(database_path)
    assert tuple(row[0] for row in rows) == (0, 1)
    assert len({(row[1], row[2]) for row in rows}) == 2
    SQLiteMemoryStore(database_path)
    assert _read_ingest_rows(database_path) == rows
