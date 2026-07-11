# SPDX-License-Identifier: Apache-2.0

"""Small durable ledger for content-addressed local-update run seals.

One seal is identified by canonical JSON containing its scope and payload.
``created_at`` and the idempotency key are storage metadata, so a retry cannot
change the content identity.  A scoped idempotency alias and its core seal are
published in one ``BEGIN IMMEDIATE`` SQLite transaction.

The hashes detect accidental or partial storage drift.  They are not signatures
against an operator who can rewrite the database and recompute every hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

__all__ = [
    "LocalUpdateRunSeal",
    "LocalUpdateRunSealCommitUnknownError",
    "LocalUpdateRunSealConflictError",
    "LocalUpdateRunSealCorruptionError",
    "LocalUpdateRunSealNotFoundError",
    "LocalUpdateRunSealPersistenceError",
    "LocalUpdateRunSealValidationError",
    "get_local_update_run_seal",
    "seal_local_update_run",
]


class LocalUpdateRunSealValidationError(ValueError):
    """The caller supplied a path, scope, key, or payload outside the contract."""


class LocalUpdateRunSealConflictError(RuntimeError):
    """An idempotency key or truncated content ID has conflicting content."""


class LocalUpdateRunSealNotFoundError(LookupError):
    """No seal is published through the requested scoped alias."""


class LocalUpdateRunSealCorruptionError(RuntimeError):
    """The database, schema, or stored seal graph failed exact validation."""


class LocalUpdateRunSealPersistenceError(RuntimeError):
    """SQLite or the local filesystem could not complete an operation."""


class LocalUpdateRunSealCommitUnknownError(LocalUpdateRunSealPersistenceError):
    """COMMIT was attempted but its durable outcome was not acknowledged.

    Recovery is an exact retry with the same database, scope, key, and payload.
    """


@dataclass(frozen=True, slots=True)
class LocalUpdateRunSeal:
    """One immutable core seal viewed through one scoped idempotency alias."""

    scope: str
    idempotency_key: str
    seal_id: str
    content_hash: str
    canonical: bytes
    created_at: datetime


_APPLICATION_ID = 1095914835
_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 10_000
_MAX_CANONICAL_BYTES = 16 * 1024 * 1024
_MAX_TEXT_BYTES = 4096
_MAX_JSON_DEPTH = 128
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEAL_ID_PATTERN = re.compile(r"lurs_[0-9a-f]{24}")
_CONTENT_DOMAIN = b"areal-memory-local-update-run-seal-content-v1\0"
_STORAGE_DOMAIN = b"areal-memory-local-update-run-seal-storage-v1\0"
_ALIAS_DOMAIN = b"areal-memory-local-update-run-seal-alias-v1\0"
_KIND = "areal-memory-local-update-run-seal-v1"
_DEFAULT_SCOPE = "global-update"

_SCHEMA_DDL = (
    """CREATE TABLE local_update_run_seal_metadata (
    singleton INTEGER NOT NULL PRIMARY KEY
        CHECK (typeof(singleton) = 'integer') CHECK (singleton = 1),
    schema_spec_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_spec_hash) = 'text')
        CHECK (length(schema_spec_hash) = 64),
    schema_catalog_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(schema_catalog_hash) = 'text')
        CHECK (length(schema_catalog_hash) = 64)
)""",
    """CREATE TABLE local_update_run_seals (
    scope TEXT NOT NULL COLLATE BINARY CHECK (typeof(scope) = 'text'),
    seal_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(seal_id) = 'text'),
    canonical BLOB NOT NULL CHECK (typeof(canonical) = 'blob'),
    content_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(content_hash) = 'text') CHECK (length(content_hash) = 64),
    created_at TEXT NOT NULL COLLATE BINARY CHECK (typeof(created_at) = 'text'),
    storage_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(storage_hash) = 'text') CHECK (length(storage_hash) = 64),
    PRIMARY KEY (scope, seal_id)
)""",
    """CREATE TABLE local_update_run_seal_aliases (
    scope TEXT NOT NULL COLLATE BINARY CHECK (typeof(scope) = 'text'),
    idempotency_key TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(idempotency_key) = 'text'),
    seal_id TEXT NOT NULL COLLATE BINARY CHECK (typeof(seal_id) = 'text'),
    binding_hash TEXT NOT NULL COLLATE BINARY
        CHECK (typeof(binding_hash) = 'text') CHECK (length(binding_hash) = 64),
    PRIMARY KEY (scope, idempotency_key),
    FOREIGN KEY (scope, seal_id)
        REFERENCES local_update_run_seals (scope, seal_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)""",
)
_REQUIRED_TABLES = frozenset(
    {
        "local_update_run_seal_metadata",
        "local_update_run_seals",
        "local_update_run_seal_aliases",
    }
)
_CATALOG_SQL = """SELECT type, name, tbl_name, sql
FROM main.sqlite_master
WHERE sql IS NOT NULL AND substr(name, 1, 7) <> 'sqlite_'
ORDER BY type COLLATE BINARY, name COLLATE BINARY,
         tbl_name COLLATE BINARY, sql COLLATE BINARY"""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _schema_spec_hash() -> str:
    return hashlib.sha256(_canonical_json_bytes(list(_SCHEMA_DDL))).hexdigest()


_SCHEMA_SPEC_HASH = _schema_spec_hash()


def _validate_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise LocalUpdateRunSealValidationError(
            f"{name} must be a non-blank exact str without NUL"
        )
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise LocalUpdateRunSealValidationError(
            f"{name} must contain valid UTF-8 text"
        ) from error
    if len(encoded) > _MAX_TEXT_BYTES:
        raise LocalUpdateRunSealValidationError(f"{name} is too large")
    return value


def _validate_json_value(
    value: object,
    *,
    depth: int = 0,
    active: set[int] | None = None,
) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise LocalUpdateRunSealValidationError("payload JSON is nested too deeply")
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise LocalUpdateRunSealValidationError(
                "payload JSON numbers must be finite"
            )
        return
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise LocalUpdateRunSealValidationError(
                "payload JSON strings must contain valid UTF-8 text"
            ) from error
        return
    if type(value) not in {dict, list}:
        raise LocalUpdateRunSealValidationError(
            "payload must contain only exact JSON object, array, and scalar types"
        )
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise LocalUpdateRunSealValidationError("payload JSON must not be cyclic")
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise LocalUpdateRunSealValidationError(
                        "payload JSON object keys must be exact str values"
                    )
                _validate_json_value(key, depth=depth + 1, active=active)
                _validate_json_value(item, depth=depth + 1, active=active)
        else:
            for item in value:
                _validate_json_value(item, depth=depth + 1, active=active)
    finally:
        active.remove(identity)


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON numbers are forbidden")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON keys are forbidden")
        value[key] = item
    return value


def _parse_canonical_json_object(value: object) -> dict[str, object]:
    if type(value) is not bytes or not value or len(value) > _MAX_CANONICAL_BYTES:
        raise ValueError("canonical JSON must be non-empty bounded exact bytes")
    text = value.decode("ascii", errors="strict")
    decoded = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    if type(decoded) is not dict or _canonical_json_bytes(decoded) != value:
        raise ValueError("stored value is not one canonical JSON object")
    return decoded


def _seal_canonical(scope: str, payload: dict[str, object]) -> bytes:
    _validate_json_value(payload)
    try:
        canonical = _canonical_json_bytes(
            {
                "kind": _KIND,
                "payload": payload,
                "schema_version": _SCHEMA_VERSION,
                "scope": scope,
            }
        )
        _parse_canonical_json_object(canonical)
    except LocalUpdateRunSealValidationError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise LocalUpdateRunSealValidationError(
            "payload could not be encoded as canonical JSON"
        ) from error
    if len(canonical) > _MAX_CANONICAL_BYTES:
        raise LocalUpdateRunSealValidationError("canonical payload is too large")
    return canonical


def _content_hash(canonical: bytes) -> str:
    return hashlib.sha256(_CONTENT_DOMAIN + canonical).hexdigest()


def _storage_hash(
    *,
    scope: str,
    seal_id: str,
    content_hash: str,
    created_at_text: str,
) -> str:
    return _digest(
        _STORAGE_DOMAIN,
        {
            "content_hash": content_hash,
            "created_at": created_at_text,
            "record_kind": "core",
            "schema_version": _SCHEMA_VERSION,
            "scope": scope,
            "seal_id": seal_id,
        },
    )


def _alias_hash(
    *,
    scope: str,
    idempotency_key: str,
    seal_id: str,
    content_hash: str,
) -> str:
    return _digest(
        _ALIAS_DOMAIN,
        {
            "content_hash": content_hash,
            "idempotency_key": idempotency_key,
            "record_kind": "alias",
            "schema_version": _SCHEMA_VERSION,
            "scope": scope,
            "seal_id": seal_id,
        },
    )


def _snapshot_database_path(database_path: str | os.PathLike[str]) -> str:
    try:
        raw_path = os.fspath(database_path)
    except TypeError as error:
        raise LocalUpdateRunSealValidationError(
            "database_path must be a string-valued path-like object"
        ) from error
    if not isinstance(raw_path, str):
        raise LocalUpdateRunSealValidationError(
            "database_path must be a string-valued path-like object"
        )
    path = str.__str__(raw_path)
    if type(path) is not str or not path.strip() or "\x00" in path:
        raise LocalUpdateRunSealValidationError("database_path is invalid")
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise LocalUpdateRunSealValidationError(
            "database_path must contain valid UTF-8 text"
        ) from error
    if path == ":memory:" or path.startswith("//"):
        raise LocalUpdateRunSealValidationError(
            "database_path must name one local durable file"
        )
    absolute = os.path.abspath(path)
    parent = os.path.realpath(os.path.dirname(absolute))
    return os.path.join(parent, os.path.basename(absolute))


def _private_regular_file(file_stat: os.stat_result) -> bool:
    return (
        stat.S_ISREG(file_stat.st_mode)
        and file_stat.st_nlink == 1
        and file_stat.st_mode & 0o077 == 0
        and (not hasattr(os, "geteuid") or file_stat.st_uid == os.geteuid())
    )


def _require_private_database(path: str) -> tuple[int, int]:
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise LocalUpdateRunSealPersistenceError(
            "could not inspect the seal database"
        ) from error
    if not _private_regular_file(file_stat):
        raise LocalUpdateRunSealCorruptionError(
            "seal database must be one private, owned, regular file"
        )
    return file_stat.st_dev, file_stat.st_ino


def _fsync_parent_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(os.path.dirname(path), flags)
        os.fsync(descriptor)
    except OSError as error:
        raise LocalUpdateRunSealPersistenceError(
            "could not fsync the seal database directory"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_private_database(path: str) -> tuple[int, int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            return _require_private_database(path)
        file_stat = os.fstat(descriptor)
        if not _private_regular_file(file_stat):
            raise LocalUpdateRunSealCorruptionError(
                "new seal database is not a private regular file"
            )
        os.fsync(descriptor)
        identity = (file_stat.st_dev, file_stat.st_ino)
    except (LocalUpdateRunSealCorruptionError, LocalUpdateRunSealPersistenceError):
        raise
    except OSError as error:
        raise LocalUpdateRunSealPersistenceError(
            "could not create the seal database"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_parent_directory(path)
    if _require_private_database(path) != identity:
        raise LocalUpdateRunSealCorruptionError(
            "seal database path changed during creation"
        )
    return identity


def _strict_text_factory(value: bytes) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise LocalUpdateRunSealCorruptionError(
            "seal database TEXT contains invalid UTF-8"
        ) from error


def _map_sqlite_error(error: sqlite3.Error) -> RuntimeError:
    error_code = getattr(error, "sqlite_errorcode", None)
    base_code = error_code & 0xFF if type(error_code) is int else None
    if base_code in {
        getattr(sqlite3, "SQLITE_CORRUPT", -1),
        getattr(sqlite3, "SQLITE_NOTADB", -1),
    }:
        return LocalUpdateRunSealCorruptionError("seal database is corrupt")
    return LocalUpdateRunSealPersistenceError("seal database operation failed")


def _read_pragma_integer(cursor: sqlite3.Cursor, name: str) -> int:
    row = cursor.execute(f"PRAGMA {name}").fetchone()
    if row is None or len(row) != 1 or type(row[0]) is not int:
        raise LocalUpdateRunSealCorruptionError(
            f"SQLite {name} did not return one integer"
        )
    return row[0]


def _require_pragma(cursor: sqlite3.Cursor, name: str, expected: int) -> None:
    if _read_pragma_integer(cursor, name) != expected:
        raise LocalUpdateRunSealCorruptionError(
            f"SQLite {name} did not retain the required value"
        )


def _connect(path: str, expected_identity: tuple[int, int]) -> sqlite3.Connection:
    connection: sqlite3.Connection | None = None
    try:
        if _require_private_database(path) != expected_identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path identity changed before open"
            )
        connection = sqlite3.connect(
            f"{Path(path).as_uri()}?mode=rw",
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
            uri=True,
        )
        if _require_private_database(path) != expected_identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path identity changed during open"
            )
        connection.text_factory = _strict_text_factory
        cursor = connection.cursor()
        journal = cursor.execute("PRAGMA journal_mode").fetchone()
        if journal is None or len(journal) != 1 or str(journal[0]).lower() != "delete":
            raise LocalUpdateRunSealCorruptionError(
                "seal database requires DELETE journal mode"
            )
        cursor.execute("PRAGMA foreign_keys = ON")
        _require_pragma(cursor, "foreign_keys", 1)
        cursor.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        _require_pragma(cursor, "busy_timeout", _BUSY_TIMEOUT_MS)
        cursor.execute("PRAGMA synchronous = EXTRA")
        _require_pragma(cursor, "synchronous", 3)
        cursor.execute("PRAGMA fullfsync = ON")
        _require_pragma(cursor, "fullfsync", 1)
        return connection
    except BaseException as error:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                error.add_note(
                    "close failed without replacing the open error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        if isinstance(error, sqlite3.Error):
            raise _map_sqlite_error(error) from error
        raise


def _catalog_rows(cursor: sqlite3.Cursor) -> tuple[tuple[str, str, str, str], ...]:
    rows = cursor.execute(_CATALOG_SQL).fetchall()
    result: list[tuple[str, str, str, str]] = []
    for row in rows:
        if len(row) != 4 or any(type(value) is not str for value in row):
            raise LocalUpdateRunSealCorruptionError(
                "seal database schema catalog contains an invalid row"
            )
        result.append((row[0], row[1], row[2], row[3]))
    return tuple(result)


def _catalog_hash(cursor: sqlite3.Cursor) -> str:
    return hashlib.sha256(_canonical_json_bytes(_catalog_rows(cursor))).hexdigest()


def _initialize_schema_locked(cursor: sqlite3.Cursor) -> None:
    for statement in _SCHEMA_DDL:
        cursor.execute(statement)
    catalog_hash = _catalog_hash(cursor)
    cursor.execute(
        "INSERT INTO local_update_run_seal_metadata VALUES (?, ?, ?)",
        (1, _SCHEMA_SPEC_HASH, catalog_hash),
    )
    cursor.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
    cursor.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _validate_schema_locked(cursor: sqlite3.Cursor) -> None:
    if (
        _read_pragma_integer(cursor, "application_id") != _APPLICATION_ID
        or _read_pragma_integer(cursor, "user_version") != _SCHEMA_VERSION
        or _schema_spec_hash() != _SCHEMA_SPEC_HASH
    ):
        raise LocalUpdateRunSealCorruptionError(
            "seal database schema identity does not match"
        )
    catalog = _catalog_rows(cursor)
    tables = {name for kind, name, _table, _sql in catalog if kind == "table"}
    if tables != _REQUIRED_TABLES or len(catalog) != len(_REQUIRED_TABLES):
        raise LocalUpdateRunSealCorruptionError(
            "seal database schema catalog does not match"
        )
    metadata = cursor.execute(
        "SELECT singleton, schema_spec_hash, schema_catalog_hash "
        "FROM local_update_run_seal_metadata"
    ).fetchall()
    if len(metadata) != 1 or metadata[0] != (
        1,
        _SCHEMA_SPEC_HASH,
        _catalog_hash(cursor),
    ):
        raise LocalUpdateRunSealCorruptionError(
            "seal database schema metadata does not match"
        )
    if cursor.execute("PRAGMA foreign_key_check").fetchall():
        raise LocalUpdateRunSealCorruptionError(
            "seal database foreign-key validation failed"
        )
    if cursor.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise LocalUpdateRunSealCorruptionError("seal database quick-check failed")


def _initialize_database(path: str) -> None:
    identity = _ensure_private_database(path)
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    active = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(path, identity)
        cursor = connection.cursor()
        active = True
        cursor.execute("BEGIN EXCLUSIVE")
        application_id = _read_pragma_integer(cursor, "application_id")
        user_version = _read_pragma_integer(cursor, "user_version")
        catalog = _catalog_rows(cursor)
        if application_id == 0 and user_version == 0 and not catalog:
            _initialize_schema_locked(cursor)
        else:
            _validate_schema_locked(cursor)
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed before initialization COMMIT"
            )
        cursor.execute("COMMIT")
        active = False
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed after initialization COMMIT"
            )
    except BaseException as error:
        primary_error = (
            _map_sqlite_error(error) if isinstance(error, sqlite3.Error) else error
        )
        if active and cursor is not None:
            try:
                cursor.execute("ROLLBACK")
            except BaseException as rollback_error:
                primary_error.add_note(
                    "ROLLBACK failed without replacing initialization error: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        if primary_error is error:
            raise
        raise primary_error from error
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "close failed after initialization error: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                else:
                    raise LocalUpdateRunSealPersistenceError(
                        "could not close the seal database after initialization"
                    ) from close_error


def _transaction_fault_hook(_stage: str) -> None:
    """No-op production hook monkeypatched by crash/ACK-loss tests."""


@contextmanager
def _write_transaction(path: str) -> Iterator[sqlite3.Cursor]:
    identity = _require_private_database(path)
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    active = False
    commit_attempted = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(path, identity)
        cursor = connection.cursor()
        active = True
        cursor.execute("BEGIN IMMEDIATE")
        _transaction_fault_hook("after_begin")
        _validate_schema_locked(cursor)
        yield cursor
        _transaction_fault_hook("before_commit")
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed before COMMIT"
            )
        commit_attempted = True
        cursor.execute("COMMIT")
        active = False
        _transaction_fault_hook("before_post_commit_identity_check")
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed after COMMIT"
            )
        _transaction_fault_hook("after_commit")
    except BaseException as error:
        if commit_attempted:
            primary_error = LocalUpdateRunSealCommitUnknownError(
                "seal COMMIT outcome is unknown; retry the exact scoped request"
            )
        elif isinstance(error, sqlite3.Error):
            primary_error = _map_sqlite_error(error)
        else:
            primary_error = error
        if active and cursor is not None:
            try:
                cursor.execute("ROLLBACK")
            except BaseException as rollback_error:
                primary_error.add_note(
                    "ROLLBACK failed without replacing the primary error: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        if primary_error is error:
            raise
        raise primary_error from error
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "close failed after primary error: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                else:
                    raise LocalUpdateRunSealPersistenceError(
                        "could not close the seal database"
                    ) from close_error


@contextmanager
def _read_transaction(path: str) -> Iterator[sqlite3.Cursor]:
    identity = _require_private_database(path)
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    active = False
    primary_error: BaseException | None = None
    try:
        connection = _connect(path, identity)
        cursor = connection.cursor()
        active = True
        cursor.execute("BEGIN")
        cursor.execute("SELECT name FROM main.sqlite_master LIMIT 1").fetchone()
        _validate_schema_locked(cursor)
        yield cursor
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed before read COMMIT"
            )
        cursor.execute("COMMIT")
        active = False
        if _require_private_database(path) != identity:
            raise LocalUpdateRunSealCorruptionError(
                "seal database path changed after read COMMIT"
            )
    except BaseException as error:
        primary_error = (
            _map_sqlite_error(error) if isinstance(error, sqlite3.Error) else error
        )
        if active and cursor is not None:
            try:
                cursor.execute("ROLLBACK")
            except BaseException as rollback_error:
                primary_error.add_note(
                    "ROLLBACK failed without replacing the read error: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        if primary_error is error:
            raise
        raise primary_error from error
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as close_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "close failed after read error: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                else:
                    raise LocalUpdateRunSealPersistenceError(
                        "could not close the seal database after read"
                    ) from close_error


@dataclass(frozen=True, slots=True)
class _StoredCore:
    scope: str
    seal_id: str
    content_hash: str
    canonical: bytes
    created_at: datetime


def _load_core_rows(cursor: sqlite3.Cursor) -> dict[tuple[str, str], _StoredCore]:
    rows = cursor.execute(
        "SELECT scope, seal_id, canonical, content_hash, created_at, storage_hash "
        "FROM local_update_run_seals"
    ).fetchall()
    result: dict[tuple[str, str], _StoredCore] = {}
    for row in rows:
        try:
            if (
                len(row) != 6
                or type(row[0]) is not str
                or type(row[1]) is not str
                or type(row[2]) is not bytes
                or any(type(value) is not str for value in row[3:])
            ):
                raise ValueError("invalid seal row storage classes")
            scope = _validate_text(row[0], "stored scope")
            seal_id, canonical, content_hash, created_at_text, storage_hash = row[1:]
            decoded = _parse_canonical_json_object(canonical)
            if frozenset(decoded) != {
                "kind",
                "payload",
                "schema_version",
                "scope",
            }:
                raise ValueError("canonical seal has unexpected fields")
            if (
                decoded["kind"] != _KIND
                or decoded["schema_version"] != _SCHEMA_VERSION
                or decoded["scope"] != scope
                or type(decoded["payload"]) is not dict
            ):
                raise ValueError("canonical seal projections disagree")
            calculated_hash = _content_hash(canonical)
            if (
                _SHA256_PATTERN.fullmatch(content_hash) is None
                or content_hash != calculated_hash
                or _SEAL_ID_PATTERN.fullmatch(seal_id) is None
                or seal_id != f"lurs_{calculated_hash[:24]}"
            ):
                raise ValueError("seal content identity disagrees")
            created_at = datetime.fromisoformat(created_at_text)
            if (
                created_at.tzinfo is None
                or created_at.utcoffset() != timedelta(0)
                or created_at.isoformat() != created_at_text
            ):
                raise ValueError("seal created_at is not exact aware ISO text")
            if storage_hash != _storage_hash(
                scope=scope,
                seal_id=seal_id,
                content_hash=content_hash,
                created_at_text=created_at_text,
            ):
                raise ValueError("seal storage hash disagrees")
            address = (scope, seal_id)
            if address in result:
                raise ValueError("duplicate seal address")
            result[address] = _StoredCore(
                scope=scope,
                seal_id=seal_id,
                content_hash=content_hash,
                canonical=canonical,
                created_at=created_at,
            )
        except LocalUpdateRunSealCorruptionError:
            raise
        except (
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise LocalUpdateRunSealCorruptionError(
                "stored local-update run seal failed exact validation"
            ) from error
    return result


def _load_state(
    cursor: sqlite3.Cursor,
) -> tuple[
    dict[tuple[str, str], _StoredCore],
    dict[tuple[str, str], _StoredCore],
]:
    cores = _load_core_rows(cursor)
    rows = cursor.execute(
        "SELECT scope, idempotency_key, seal_id, binding_hash "
        "FROM local_update_run_seal_aliases"
    ).fetchall()
    aliases: dict[tuple[str, str], _StoredCore] = {}
    for row in rows:
        try:
            if len(row) != 4 or any(type(value) is not str for value in row):
                raise ValueError("invalid alias row storage classes")
            scope = _validate_text(row[0], "stored alias scope")
            key = _validate_text(row[1], "stored idempotency key")
            seal_id, binding_hash = row[2:]
            core = cores.get((scope, seal_id))
            address = (scope, key)
            if (
                core is None
                or address in aliases
                or binding_hash
                != _alias_hash(
                    scope=scope,
                    idempotency_key=key,
                    seal_id=seal_id,
                    content_hash=core.content_hash,
                )
            ):
                raise ValueError("alias target or binding disagrees")
            aliases[address] = core
        except (TypeError, ValueError, UnicodeError) as error:
            raise LocalUpdateRunSealCorruptionError(
                "stored local-update run seal alias failed exact validation"
            ) from error
    referenced = {(core.scope, core.seal_id) for core in aliases.values()}
    if referenced != set(cores):
        raise LocalUpdateRunSealCorruptionError(
            "local-update run seal exists without an idempotency alias"
        )
    return cores, aliases


def _public_seal(core: _StoredCore, idempotency_key: str) -> LocalUpdateRunSeal:
    return LocalUpdateRunSeal(
        scope=core.scope,
        idempotency_key=idempotency_key,
        seal_id=core.seal_id,
        content_hash=core.content_hash,
        canonical=core.canonical,
        created_at=core.created_at,
    )


def seal_local_update_run(
    database_path: str | os.PathLike[str],
    *,
    idempotency_key: str,
    payload: dict[str, object],
    scope: str = _DEFAULT_SCOPE,
) -> LocalUpdateRunSeal:
    """Atomically publish or exactly replay one scoped local-update run seal."""

    path = _snapshot_database_path(database_path)
    scope = _validate_text(scope, "scope")
    idempotency_key = _validate_text(idempotency_key, "idempotency_key")
    if type(payload) is not dict:
        raise LocalUpdateRunSealValidationError("payload must be an exact dict")
    canonical = _seal_canonical(scope, payload)
    content_hash = _content_hash(canonical)
    if type(content_hash) is not str or _SHA256_PATTERN.fullmatch(content_hash) is None:
        raise LocalUpdateRunSealPersistenceError(
            "content digest did not return lowercase SHA-256"
        )
    seal_id = f"lurs_{content_hash[:24]}"
    _initialize_database(path)

    with _write_transaction(path) as cursor:
        cores, aliases = _load_state(cursor)
        existing_alias = aliases.get((scope, idempotency_key))
        if existing_alias is not None:
            if existing_alias.canonical == canonical:
                return _public_seal(existing_alias, idempotency_key)
            raise LocalUpdateRunSealConflictError(
                "scoped idempotency key already refers to different content"
            )

        existing_core = cores.get((scope, seal_id))
        if existing_core is not None and (
            existing_core.content_hash != content_hash
            or existing_core.canonical != canonical
        ):
            raise LocalUpdateRunSealConflictError(
                f"local-update run seal ID collision for {seal_id!r}"
            )
        if existing_core is None:
            created_at = datetime.now(UTC)
            created_at_text = created_at.isoformat()
            cursor.execute(
                "INSERT INTO local_update_run_seals "
                "(scope, seal_id, canonical, content_hash, created_at, storage_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    scope,
                    seal_id,
                    canonical,
                    content_hash,
                    created_at_text,
                    _storage_hash(
                        scope=scope,
                        seal_id=seal_id,
                        content_hash=content_hash,
                        created_at_text=created_at_text,
                    ),
                ),
            )
            _transaction_fault_hook("after_core_insert")
        cursor.execute(
            "INSERT INTO local_update_run_seal_aliases "
            "(scope, idempotency_key, seal_id, binding_hash) VALUES (?, ?, ?, ?)",
            (
                scope,
                idempotency_key,
                seal_id,
                _alias_hash(
                    scope=scope,
                    idempotency_key=idempotency_key,
                    seal_id=seal_id,
                    content_hash=content_hash,
                ),
            ),
        )
        _transaction_fault_hook("after_alias_insert")
        _cores, reloaded_aliases = _load_state(cursor)
        reloaded = reloaded_aliases.get((scope, idempotency_key))
        if (
            reloaded is None
            or reloaded.content_hash != content_hash
            or reloaded.canonical != canonical
        ):
            raise LocalUpdateRunSealCorruptionError(
                "inserted local-update run seal could not be reloaded exactly"
            )
        _transaction_fault_hook("after_readback")
        return _public_seal(reloaded, idempotency_key)


def get_local_update_run_seal(
    database_path: str | os.PathLike[str],
    *,
    idempotency_key: str,
    scope: str = _DEFAULT_SCOPE,
) -> LocalUpdateRunSeal:
    """Reopen one seal through its exact scoped idempotency alias."""

    path = _snapshot_database_path(database_path)
    scope = _validate_text(scope, "scope")
    idempotency_key = _validate_text(idempotency_key, "idempotency_key")
    try:
        _require_private_database(path)
    except FileNotFoundError as error:
        raise LocalUpdateRunSealNotFoundError(
            "local-update run seal database was not found"
        ) from error
    with _read_transaction(path) as cursor:
        _cores, aliases = _load_state(cursor)
        core = aliases.get((scope, idempotency_key))
        if core is None:
            raise LocalUpdateRunSealNotFoundError(
                "local-update run seal alias was not found"
            )
        return _public_seal(core, idempotency_key)
