# SPDX-License-Identifier: Apache-2.0

"""Private SQLite mechanics for the durable Memory Service backend."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from areal.v2.memory_service.errors import (
    MemoryPersistenceBusyError,
    MemoryPersistenceCorruptionError,
    MemoryPersistenceError,
    MemoryPersistenceSchemaError,
)
from areal.v2.memory_service.types import MemoryScope

_APPLICATION_ID = 1095912787
_SCHEMA_VERSION = 2
_SCHEMA_V1_VERSION = 1
_MIN_SQLITE_VERSION = (3, 7, 17)
_BUSY_TIMEOUT_MS = 5000
_MAX_SIGNED_64 = 2**63 - 1

_SCHEMA_SHA256 = hashlib.sha256
_RECORD_SHA256 = hashlib.sha256

_SCHEMA_V1_DDL = (
    """CREATE TABLE memory_schema_metadata (
    singleton INTEGER NOT NULL PRIMARY KEY
        CHECK (typeof(singleton) = 'integer') CHECK (singleton = 1),
    schema_spec_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_spec_hash) = 'text')
        CHECK (length(schema_spec_hash) = 64),
    schema_catalog_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_catalog_hash) = 'text')
        CHECK (length(schema_catalog_hash) = 64)
)""",
    """CREATE TABLE memory_scopes (
    scope_id INTEGER NOT NULL PRIMARY KEY CHECK (typeof(scope_id) = 'integer'),
    tenant_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(tenant_id) = 'text'),
    namespace TEXT NOT NULL COLLATE BINARY CHECK (typeof(namespace) = 'text'),
    subject_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(subject_id) = 'text'),
    UNIQUE (tenant_id, namespace, subject_id)
)""",
    """CREATE TABLE memory_evidence (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    evidence_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(evidence_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    session_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(session_id) = 'text'),
    run_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(run_id) = 'text'),
    sequence_no INTEGER NOT NULL CHECK (typeof(sequence_no) = 'integer')
        CHECK (sequence_no BETWEEN 0 AND 9223372036854775807),
    kind TEXT NOT NULL COLLATE BINARY CHECK (typeof(kind) = 'text')
        CHECK (kind IN ('user_message','agent_message','tool_call','tool_result',
                        'environment','feedback','outcome')),
    payload TEXT NOT NULL COLLATE BINARY CHECK (typeof(payload) = 'text'),
    observed_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(observed_at) = 'text'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    PRIMARY KEY (scope_id, evidence_id),
    UNIQUE (scope_id, idempotency_key),
    FOREIGN KEY (scope_id) REFERENCES memory_scopes (scope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_candidates (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    candidate_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(candidate_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    content TEXT NOT NULL COLLATE BINARY CHECK (typeof(content) = 'text'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    PRIMARY KEY (scope_id, candidate_id),
    UNIQUE (scope_id, idempotency_key),
    FOREIGN KEY (scope_id) REFERENCES memory_scopes (scope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_candidate_evidence (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    candidate_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(candidate_id) = 'text'),
    position INTEGER NOT NULL CHECK (typeof(position) = 'integer')
        CHECK (position BETWEEN 0 AND 9223372036854775807),
    evidence_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(evidence_id) = 'text'),
    PRIMARY KEY (scope_id, candidate_id, position),
    UNIQUE (scope_id, candidate_id, evidence_id),
    FOREIGN KEY (scope_id, candidate_id)
        REFERENCES memory_candidates (scope_id, candidate_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (scope_id, evidence_id)
        REFERENCES memory_evidence (scope_id, evidence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_revisions (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    revision_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(revision_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    candidate_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(candidate_id) = 'text'),
    memory_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(memory_id) = 'text'),
    generation INTEGER NOT NULL CHECK (typeof(generation) = 'integer')
        CHECK (generation BETWEEN 0 AND 9223372036854775807),
    operation TEXT NOT NULL COLLATE BINARY CHECK (typeof(operation) = 'text')
        CHECK (operation IN ('add','refine','supersede','contradict')),
    parent_revision_id TEXT COLLATE BINARY
        CHECK (parent_revision_id IS NULL OR typeof(parent_revision_id) = 'text'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    CHECK ((operation = 'add' AND parent_revision_id IS NULL)
        OR (operation IN ('refine','supersede','contradict')
            AND parent_revision_id IS NOT NULL)),
    PRIMARY KEY (scope_id, revision_id),
    UNIQUE (scope_id, idempotency_key),
    UNIQUE (scope_id, candidate_id),
    UNIQUE (scope_id, revision_id, memory_id),
    FOREIGN KEY (scope_id, candidate_id)
        REFERENCES memory_candidates (scope_id, candidate_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (scope_id, parent_revision_id)
        REFERENCES memory_revisions (scope_id, revision_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_releases (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    release_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(release_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    PRIMARY KEY (scope_id, release_id),
    FOREIGN KEY (scope_id) REFERENCES memory_scopes (scope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_release_aliases (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    release_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(release_id) = 'text'),
    binding_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(binding_hash) = 'text') CHECK (length(binding_hash) = 64),
    PRIMARY KEY (scope_id, idempotency_key),
    FOREIGN KEY (scope_id, release_id)
        REFERENCES memory_releases (scope_id, release_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_release_revisions (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    release_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(release_id) = 'text'),
    position INTEGER NOT NULL CHECK (typeof(position) = 'integer')
        CHECK (position BETWEEN 0 AND 9223372036854775807),
    revision_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(revision_id) = 'text'),
    memory_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(memory_id) = 'text'),
    PRIMARY KEY (scope_id, release_id, position),
    UNIQUE (scope_id, release_id, revision_id),
    UNIQUE (scope_id, release_id, memory_id),
    FOREIGN KEY (scope_id, release_id)
        REFERENCES memory_releases (scope_id, release_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (scope_id, revision_id, memory_id)
        REFERENCES memory_revisions (scope_id, revision_id, memory_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE INDEX idx_memory_evidence_sort ON memory_evidence (
    scope_id, session_id, run_id, sequence_no, observed_at, evidence_id
)""",
    """CREATE INDEX idx_memory_revisions_sort ON memory_revisions (
    scope_id, memory_id, generation, revision_id
)""",
)

_SCHEMA_V2_ADDITIONS = (
    """CREATE TABLE memory_evidence_ingest_orders (
    ingest_order INTEGER NOT NULL PRIMARY KEY CHECK (typeof(ingest_order) = 'integer')
        CHECK (ingest_order BETWEEN 0 AND 9223372036854775807),
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    evidence_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(evidence_id) = 'text'),
    binding_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(binding_hash) = 'text') CHECK (length(binding_hash) = 64),
    UNIQUE (scope_id, evidence_id),
    UNIQUE (scope_id, evidence_id, ingest_order),
    FOREIGN KEY (scope_id, evidence_id)
        REFERENCES memory_evidence (scope_id, evidence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_evidence_snapshots (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    snapshot_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(snapshot_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    allowed_kinds_canonical BLOB NOT NULL
        CHECK (typeof(allowed_kinds_canonical) = 'blob'),
    cutoff_utc TEXT NOT NULL COLLATE BINARY CHECK (typeof(cutoff_utc) = 'text'),
    evidence_high_watermark INTEGER NOT NULL
        CHECK (typeof(evidence_high_watermark) = 'integer')
        CHECK (evidence_high_watermark BETWEEN -1 AND 9223372036854775807),
    ordering_policy TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(ordering_policy) = 'text'),
    member_count INTEGER NOT NULL CHECK (typeof(member_count) = 'integer')
        CHECK (member_count BETWEEN 0 AND 9223372036854775807),
    PRIMARY KEY (scope_id, snapshot_id),
    FOREIGN KEY (scope_id) REFERENCES memory_scopes (scope_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE TABLE memory_evidence_snapshot_aliases (
    scope_id INTEGER NOT NULL CHECK (typeof(scope_id) = 'integer'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    snapshot_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(snapshot_id) = 'text'),
    binding_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(binding_hash) = 'text') CHECK (length(binding_hash) = 64),
    PRIMARY KEY (scope_id, idempotency_key),
    FOREIGN KEY (scope_id, snapshot_id)
        REFERENCES memory_evidence_snapshots (scope_id, snapshot_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
    """CREATE INDEX idx_memory_evidence_ingest_scope ON memory_evidence_ingest_orders (
    scope_id, ingest_order
)""",
)

_SCHEMA_DDL = _SCHEMA_V1_DDL + _SCHEMA_V2_ADDITIONS

_V1_REQUIRED_TABLES = frozenset(
    {
        "memory_schema_metadata",
        "memory_scopes",
        "memory_evidence",
        "memory_candidates",
        "memory_candidate_evidence",
        "memory_revisions",
        "memory_releases",
        "memory_release_aliases",
        "memory_release_revisions",
    }
)
_V1_REQUIRED_INDEXES = frozenset(
    {"idx_memory_evidence_sort", "idx_memory_revisions_sort"}
)
_REQUIRED_TABLES = _V1_REQUIRED_TABLES | frozenset(
    {
        "memory_evidence_ingest_orders",
        "memory_evidence_snapshots",
        "memory_evidence_snapshot_aliases",
    }
)
_REQUIRED_INDEXES = _V1_REQUIRED_INDEXES | frozenset(
    {"idx_memory_evidence_ingest_scope"}
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


def _schema_spec_digest(ddl: tuple[str, ...] = _SCHEMA_DDL) -> str:
    return _SCHEMA_SHA256(_compact_json_bytes(ddl)).hexdigest()


_SCHEMA_V1_SPEC_HASH = (
    "445a839fb37b9db29842f018f75887debcbb96997a1391b73068ea23e2f355c0"
)
_SCHEMA_SPEC_HASH = (
    "28ce8fc78297c4d4b046ccd7be272b5d4de7318005fe277c8383f44e83ea9a0a"
)


def _snapshot_database_path(database_path: str | os.PathLike[str]) -> str:
    """Snapshot one string-valued path-like object into one absolute file path."""

    try:
        raw_path = os.fspath(database_path)
    except TypeError as error:
        raise TypeError(
            "database_path must be a string-valued path-like object"
        ) from error
    if not isinstance(raw_path, str):
        raise TypeError("database_path must be a string-valued path-like object")
    path = str.__str__(raw_path)
    if type(path) is not str:
        path = path.encode("utf-8", "strict").decode("utf-8", "strict")
    if not path.strip():
        raise ValueError("database_path must not be blank")
    if "\x00" in path:
        raise ValueError("database_path must not contain NUL")
    try:
        path.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError("database_path must be valid UTF-8") from error
    if path == ":memory:":
        raise ValueError("database_path must name a durable database file")
    return os.path.abspath(path)


def _require_supported_runtime() -> None:
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        raise MemoryPersistenceSchemaError(
            "SQLite 3.7.17 or newer is required for the Memory Service schema"
        )


def _require_delete_journal(cursor: sqlite3.Cursor) -> None:
    row = cursor.execute("PRAGMA journal_mode").fetchone()
    if (
        row is None
        or len(row) != 1
        or not isinstance(row[0], str)
        or row[0].lower() != "delete"
    ):
        raise MemoryPersistenceSchemaError(
            "Memory Service SQLite databases require DELETE journal mode"
        )


def _require_pragma_value(
    cursor: sqlite3.Cursor,
    pragma: str,
    expected: int,
) -> None:
    row = cursor.execute(f"PRAGMA {pragma}").fetchone()
    if row != (expected,):
        raise MemoryPersistenceSchemaError(
            f"SQLite {pragma} readback did not equal {expected}"
        )


def _configure_connection(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA foreign_keys = ON")
    _require_pragma_value(cursor, "foreign_keys", 1)
    cursor.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    _require_pragma_value(cursor, "busy_timeout", _BUSY_TIMEOUT_MS)
    cursor.execute("PRAGMA synchronous = FULL")
    _require_pragma_value(cursor, "synchronous", 2)


def _strict_text_factory(value: bytes) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MemoryPersistenceCorruptionError(
            "SQLite TEXT contains invalid UTF-8"
        ) from error


def _connect(path: str) -> sqlite3.Connection:
    """Open and fully configure one per-operation SQLite connection."""

    _require_supported_runtime()
    try:
        connection = sqlite3.connect(
            path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
    except sqlite3.Error as error:
        raise _map_sqlite_error(error) from error
    try:
        connection.text_factory = _strict_text_factory
        cursor = connection.cursor()
        _require_delete_journal(cursor)
        _configure_connection(cursor)
    except BaseException as error:
        try:
            connection.close()
        except BaseException as close_error:
            _add_cleanup_note(error, "close", close_error)
        if isinstance(error, sqlite3.Error):
            raise _map_sqlite_error(error) from error
        raise
    return connection


def _catalog_rows(cursor: sqlite3.Cursor) -> tuple[tuple[str, str, str, str], ...]:
    rows = cursor.execute(_CATALOG_SQL).fetchall()
    result: list[tuple[str, str, str, str]] = []
    for row in rows:
        if len(row) != 4 or not all(type(value) is str for value in row):
            raise MemoryPersistenceSchemaError(
                "SQLite schema catalog contains an invalid row"
            )
        result.append((row[0], row[1], row[2], row[3]))
    return tuple(result)


def _catalog_hash(cursor: sqlite3.Cursor) -> str:
    return _SCHEMA_SHA256(_compact_json_bytes(_catalog_rows(cursor))).hexdigest()


def _read_integer_pragma(cursor: sqlite3.Cursor, pragma: str) -> int:
    row = cursor.execute(f"PRAGMA {pragma}").fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int:
        raise MemoryPersistenceSchemaError(
            f"SQLite {pragma} did not return one integer"
        )
    return row[0]


def _initialize_v2_locked(cursor: sqlite3.Cursor) -> None:
    for statement in _SCHEMA_DDL:
        cursor.execute(statement)
    catalog_hash = _catalog_hash(cursor)
    cursor.execute(
        "INSERT INTO memory_schema_metadata "
        "(singleton, schema_spec_hash, schema_catalog_hash) VALUES (?, ?, ?)",
        (1, _SCHEMA_SPEC_HASH, catalog_hash),
    )
    cursor.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
    cursor.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    _validate_v2_locked(cursor)


def _validate_schema_locked(
    cursor: sqlite3.Cursor,
    *,
    version: int,
    ddl: tuple[str, ...],
    spec_hash: str,
    required_tables: frozenset[str],
    required_indexes: frozenset[str],
) -> None:
    application_id = _read_integer_pragma(cursor, "application_id")
    if application_id != _APPLICATION_ID:
        raise MemoryPersistenceSchemaError(
            "SQLite application_id does not identify a Memory Service database"
        )
    user_version = _read_integer_pragma(cursor, "user_version")
    if user_version != version:
        raise MemoryPersistenceSchemaError(
            f"unsupported Memory Service SQLite user_version {user_version}"
        )

    current_spec_hash = _schema_spec_digest(ddl)
    if current_spec_hash != spec_hash:
        raise MemoryPersistenceSchemaError(
            "compiled Memory Service schema specification hash changed"
        )

    catalog_rows = _catalog_rows(cursor)
    tables = {name for kind, name, _table_name, _sql in catalog_rows if kind == "table"}
    indexes = {
        name for kind, name, _table_name, _sql in catalog_rows if kind == "index"
    }
    if (
        tables != required_tables
        or indexes != required_indexes
        or len(catalog_rows) != len(required_tables) + len(required_indexes)
    ):
        raise MemoryPersistenceSchemaError(
            f"Memory Service SQLite schema catalog does not match version {version}"
        )

    metadata_rows = cursor.execute("SELECT * FROM memory_schema_metadata").fetchall()
    if len(metadata_rows) != 1 or len(metadata_rows[0]) != 3:
        raise MemoryPersistenceSchemaError(
            "Memory Service SQLite schema metadata must contain one row"
        )
    singleton, stored_spec_hash, stored_catalog_hash = metadata_rows[0]
    if singleton != 1 or stored_spec_hash != spec_hash:
        raise MemoryPersistenceSchemaError(
            "Memory Service schema specification metadata does not match"
        )
    current_catalog_hash = _SCHEMA_SHA256(_compact_json_bytes(catalog_rows)).hexdigest()
    if stored_catalog_hash != current_catalog_hash:
        raise MemoryPersistenceSchemaError(
            "Memory Service SQLite schema catalog fingerprint does not match"
        )
    if cursor.execute("PRAGMA foreign_key_check").fetchall():
        raise MemoryPersistenceCorruptionError(
            "Memory Service SQLite data failed foreign key validation"
        )


def _validate_v1_locked(cursor: sqlite3.Cursor) -> None:
    _validate_schema_locked(
        cursor,
        version=_SCHEMA_V1_VERSION,
        ddl=_SCHEMA_V1_DDL,
        spec_hash=_SCHEMA_V1_SPEC_HASH,
        required_tables=_V1_REQUIRED_TABLES,
        required_indexes=_V1_REQUIRED_INDEXES,
    )


def _validate_ingest_orders_locked(cursor: sqlite3.Cursor) -> None:
    evidence_rows = cursor.execute(
        "SELECT scope_id, evidence_id FROM memory_evidence"
    ).fetchall()
    ingest_rows = cursor.execute(
        "SELECT ingest_order, scope_id, evidence_id, binding_hash "
        "FROM memory_evidence_ingest_orders ORDER BY ingest_order"
    ).fetchall()
    if any(
        len(row) != 2 or type(row[0]) is not int or type(row[1]) is not str
        for row in evidence_rows
    ) or any(
        len(row) != 4
        or type(row[0]) is not int
        or not 0 <= row[0] <= _MAX_SIGNED_64
        or type(row[1]) is not int
        or type(row[2]) is not str
        or type(row[3]) is not str
        for row in ingest_rows
    ):
        raise MemoryPersistenceCorruptionError(
            "evidence ingest-order rows have invalid storage classes"
        )
    evidence_addresses = {(row[0], row[1]) for row in evidence_rows}
    ingest_addresses = {(row[1], row[2]) for row in ingest_rows}
    ingest_orders = tuple(row[0] for row in ingest_rows)
    if (
        evidence_addresses != ingest_addresses
        or len(evidence_addresses) != len(evidence_rows)
        or len(ingest_addresses) != len(ingest_rows)
        or ingest_orders != tuple(range(len(ingest_rows)))
    ):
        raise MemoryPersistenceCorruptionError(
            "evidence ingest-order mapping is incomplete or non-contiguous"
        )
    scope_rows = cursor.execute(
        "SELECT scope_id, tenant_id, namespace, subject_id FROM memory_scopes"
    ).fetchall()
    scopes: dict[int, MemoryScope] = {}
    for row in scope_rows:
        if (
            len(row) != 4
            or type(row[0]) is not int
            or any(type(value) is not str for value in row[1:])
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence ingest-order scope has invalid storage classes"
            )
        try:
            scopes[row[0]] = MemoryScope(
                tenant_id=row[1],
                namespace=row[2],
                subject_id=row[3],
            )
        except (TypeError, ValueError) as error:
            raise MemoryPersistenceCorruptionError(
                "evidence ingest-order scope failed public validation"
            ) from error
    for ingest_order, scope_id, evidence_id, binding_hash in ingest_rows:
        scope = scopes.get(scope_id)
        if scope is None or binding_hash != _evidence_ingest_binding_hash(
            scope=scope,
            evidence_id=evidence_id,
            ingest_order=ingest_order,
        ):
            raise MemoryPersistenceCorruptionError(
                "evidence ingest-order binding hash is invalid"
            )


def _validate_v2_locked(cursor: sqlite3.Cursor) -> None:
    _validate_schema_locked(
        cursor,
        version=_SCHEMA_VERSION,
        ddl=_SCHEMA_DDL,
        spec_hash=_SCHEMA_SPEC_HASH,
        required_tables=_REQUIRED_TABLES,
        required_indexes=_REQUIRED_INDEXES,
    )
    _validate_ingest_orders_locked(cursor)


def _migrate_v1_to_v2_locked(cursor: sqlite3.Cursor) -> None:
    _validate_v1_locked(cursor)
    for statement in _SCHEMA_V2_ADDITIONS:
        cursor.execute(statement)
    rows = cursor.execute(
        "SELECT scope_id, evidence_id, created_at FROM memory_evidence "
        "ORDER BY created_at COLLATE BINARY, evidence_id COLLATE BINARY, scope_id"
    ).fetchall()
    for ingest_order, row in enumerate(rows):
        if (
            len(row) != 3
            or type(row[0]) is not int
            or type(row[1]) is not str
            or type(row[2]) is not str
        ):
            raise MemoryPersistenceCorruptionError(
                "v1 evidence addresses cannot be migrated"
            )
        scope_id, evidence_id, created_at_text = row
        try:
            created_at = datetime.fromisoformat(created_at_text).astimezone(UTC)
        except (TypeError, ValueError, OverflowError) as error:
            raise MemoryPersistenceCorruptionError(
                "v1 evidence created_at cannot be migrated"
            ) from error
        if created_at.isoformat() != created_at_text:
            raise MemoryPersistenceCorruptionError(
                "v1 evidence created_at is not canonical UTC text"
            )
        scope_row = cursor.execute(
            "SELECT tenant_id, namespace, subject_id FROM memory_scopes "
            "WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()
        if (
            scope_row is None
            or len(scope_row) != 3
            or any(type(value) is not str for value in scope_row)
        ):
            raise MemoryPersistenceCorruptionError(
                "v1 evidence scope cannot be migrated"
            )
        try:
            scope = MemoryScope(
                tenant_id=scope_row[0],
                namespace=scope_row[1],
                subject_id=scope_row[2],
            )
        except (TypeError, ValueError) as error:
            raise MemoryPersistenceCorruptionError(
                "v1 evidence scope failed public validation"
            ) from error
        binding_hash = _evidence_ingest_binding_hash(
            scope=scope,
            evidence_id=evidence_id,
            ingest_order=ingest_order,
        )
        cursor.execute(
            "INSERT INTO memory_evidence_ingest_orders "
            "(ingest_order, scope_id, evidence_id, binding_hash) "
            "VALUES (?, ?, ?, ?)",
            (ingest_order, scope_id, evidence_id, binding_hash),
        )
    cursor.execute(
        "UPDATE memory_schema_metadata SET schema_spec_hash = ?, "
        "schema_catalog_hash = ? WHERE singleton = 1",
        (_SCHEMA_SPEC_HASH, _catalog_hash(cursor)),
    )
    cursor.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    _validate_v2_locked(cursor)


def _initialize_or_validate_locked(
    cursor: sqlite3.Cursor,
    prelock_page_count: int,
) -> None:
    application_id = _read_integer_pragma(cursor, "application_id")
    user_version = _read_integer_pragma(cursor, "user_version")
    internal_schema_version = _read_integer_pragma(cursor, "schema_version")
    catalog_rows = _catalog_rows(cursor)

    if application_id == _APPLICATION_ID and user_version == _SCHEMA_VERSION:
        _validate_v2_locked(cursor)
        return
    if application_id == _APPLICATION_ID and user_version == _SCHEMA_V1_VERSION:
        _migrate_v1_to_v2_locked(cursor)
        return
    if (
        prelock_page_count == 0
        and application_id == 0
        and user_version == 0
        and internal_schema_version == 0
        and not catalog_rows
    ):
        _initialize_v2_locked(cursor)
        return
    if application_id not in {0, _APPLICATION_ID}:
        raise MemoryPersistenceSchemaError(
            "SQLite application_id belongs to another application"
        )
    if user_version != _SCHEMA_VERSION:
        if user_version != 0 or application_id == _APPLICATION_ID:
            raise MemoryPersistenceSchemaError(
                f"unsupported Memory Service SQLite user_version {user_version}"
            )
    raise MemoryPersistenceSchemaError(
        "only a truly empty v0 SQLite database may be initialized"
    )


def _map_sqlite_error(error: sqlite3.Error) -> MemoryPersistenceError:
    error_code = getattr(error, "sqlite_errorcode", None)
    if type(error_code) is int and error_code & 0xFF in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }:
        return MemoryPersistenceBusyError("SQLite database is busy or locked")
    return MemoryPersistenceError("SQLite persistence operation failed")


def _add_cleanup_note(
    primary_error: BaseException,
    operation: str,
    cleanup_error: BaseException,
) -> None:
    primary_error.add_note(
        f"{operation} cleanup failed without replacing the primary error: "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )


def _initialize_database(path: str) -> None:
    """Initialize v2, migrate exact v1, or validate one exact v2 database."""

    _require_supported_runtime()
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    transaction_may_be_active = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(path)
        cursor = connection.cursor()
        prelock_page_count = _read_integer_pragma(cursor, "page_count")
        transaction_may_be_active = True
        cursor.execute("BEGIN EXCLUSIVE")
        _require_delete_journal(cursor)
        _initialize_or_validate_locked(cursor, prelock_page_count)
        cursor.execute("COMMIT")
        transaction_may_be_active = False
    except BaseException as error:
        if isinstance(error, sqlite3.Error):
            primary_error = _map_sqlite_error(error)
        else:
            primary_error = error
        if transaction_may_be_active and cursor is not None:
            try:
                cursor.execute("ROLLBACK")
            except BaseException as rollback_error:
                _add_cleanup_note(primary_error, "ROLLBACK", rollback_error)
            else:
                transaction_may_be_active = False
        if primary_error is error:
            raise
        raise primary_error from error
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary_error is not None:
                    _add_cleanup_note(primary_error, "close", close_error)
                elif isinstance(close_error, sqlite3.Error):
                    raise _map_sqlite_error(close_error) from close_error
                else:
                    raise


@contextmanager
def _transaction(
    path: str,
    *,
    begin_sql: str,
    lock_catalog_before_journal: bool,
) -> Iterator[sqlite3.Cursor]:
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    transaction_may_be_active = False
    primary_error: BaseException | None = None

    try:
        connection = _connect(path)
        cursor = connection.cursor()
        transaction_may_be_active = True
        cursor.execute(begin_sql)
        if lock_catalog_before_journal:
            cursor.execute("SELECT name FROM main.sqlite_master LIMIT 1").fetchone()
        _require_delete_journal(cursor)
        _validate_v2_locked(cursor)
        yield cursor
        cursor.execute("COMMIT")
        transaction_may_be_active = False
    except BaseException as error:
        if isinstance(error, sqlite3.Error):
            primary_error = _map_sqlite_error(error)
        else:
            primary_error = error

        if transaction_may_be_active and cursor is not None:
            try:
                cursor.execute("ROLLBACK")
            except BaseException as rollback_error:
                _add_cleanup_note(primary_error, "ROLLBACK", rollback_error)
            else:
                transaction_may_be_active = False

        if primary_error is error:
            raise
        raise primary_error from error
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary_error is not None:
                    _add_cleanup_note(primary_error, "close", close_error)
                elif isinstance(close_error, sqlite3.Error):
                    raise _map_sqlite_error(close_error) from close_error
                else:
                    raise


@contextmanager
def _read_transaction(path: str) -> Iterator[sqlite3.Cursor]:
    with _transaction(
        path,
        begin_sql="BEGIN",
        lock_catalog_before_journal=True,
    ) as cursor:
        yield cursor


@contextmanager
def _write_transaction(path: str) -> Iterator[sqlite3.Cursor]:
    with _transaction(
        path,
        begin_sql="BEGIN IMMEDIATE",
        lock_catalog_before_journal=False,
    ) as cursor:
        yield cursor


def _record_digest(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("record digest payload must be bytes")
    return _RECORD_SHA256(payload).hexdigest()


def _scope_payload(scope: MemoryScope) -> dict[str, str]:
    if type(scope) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
    }


def _record_storage_hash(
    *,
    record_kind: Literal[
        "evidence", "candidate", "revision", "release", "evidence_snapshot"
    ],
    scope: MemoryScope,
    record_id: str,
    content_hash: str,
    created_at_text: str,
    memory_id: str | None = None,
    generation: int | None = None,
) -> str:
    if record_kind not in {
        "evidence",
        "candidate",
        "revision",
        "release",
        "evidence_snapshot",
    }:
        raise ValueError("record_kind is not supported")
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_kind": record_kind,
        "scope": _scope_payload(scope),
        "record_id": record_id,
        "content_hash": content_hash,
        "created_at": created_at_text,
    }
    if record_kind == "revision":
        if memory_id is None or type(generation) is not int:
            raise ValueError("revision storage hashes require memory_id and generation")
        if generation < 0 or generation > _MAX_SIGNED_64:
            raise ValueError("revision generation must fit a signed 64-bit integer")
        payload["memory_id"] = memory_id
        payload["generation"] = generation
    elif memory_id is not None or generation is not None:
        raise ValueError("only revision storage hashes accept memory_id and generation")
    return _record_digest(_compact_json_bytes(payload))


def _release_binding_hash(
    *,
    scope: MemoryScope,
    idempotency_key: str,
    release_id: str,
) -> str:
    payload = {
        "schema_version": 1,
        "record_kind": "release_alias",
        "scope": _scope_payload(scope),
        "idempotency_key": idempotency_key,
        "release_id": release_id,
    }
    return _record_digest(_compact_json_bytes(payload))


def _evidence_ingest_binding_hash(
    *,
    scope: MemoryScope,
    evidence_id: str,
    ingest_order: int,
) -> str:
    if type(ingest_order) is not int or not 0 <= ingest_order <= _MAX_SIGNED_64:
        raise ValueError("ingest_order must fit the non-negative signed-64 range")
    payload = {
        "schema_version": 1,
        "record_kind": "evidence_ingest_order",
        "scope": _scope_payload(scope),
        "evidence_id": evidence_id,
        "ingest_order": ingest_order,
    }
    return _record_digest(_compact_json_bytes(payload))


def _snapshot_binding_hash(
    *,
    scope: MemoryScope,
    idempotency_key: str,
    snapshot_id: str,
) -> str:
    payload = {
        "schema_version": 1,
        "record_kind": "evidence_snapshot_alias",
        "scope": _scope_payload(scope),
        "idempotency_key": idempotency_key,
        "snapshot_id": snapshot_id,
    }
    return _record_digest(_compact_json_bytes(payload))
