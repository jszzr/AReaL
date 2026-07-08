# SPDX-License-Identifier: Apache-2.0

"""Durable SQLite implementation of the Memory Service contracts."""

from __future__ import annotations

import os

from areal.v2.memory_service._sqlite_backend import (
    _initialize_database,
    _snapshot_database_path,
)


class SQLiteMemoryStore:
    """Local durable backend for immutable Memory Service records."""

    def __init__(self, database_path: str | os.PathLike[str]) -> None:
        self._database_path = _snapshot_database_path(database_path)
        _initialize_database(self._database_path)
