# SPDX-License-Identifier: Apache-2.0

"""Tests for the durable SQLite Memory Service backend."""

from __future__ import annotations

from areal.v2.memory_service.errors import (
    MemoryPersistenceBusyError,
    MemoryPersistenceCorruptionError,
    MemoryPersistenceError,
    MemoryPersistenceSchemaError,
    MemoryServiceError,
)


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
