# SPDX-License-Identifier: Apache-2.0

"""Worker and session registries for the Router service.

All state is in-memory (lost on restart). Thread-safe via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class WorkerInfo:
    """A registered data proxy worker."""

    worker_id: str
    worker_addr: str
    is_healthy: bool = True
    active_requests: int = 0
    registered_at: float = field(default_factory=time.time)
    registered_from_worker_id: str | None = field(default=None, repr=False)


class WorkerRegistry:
    """Thread-safe worker registry with health tracking."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}  # worker_addr -> WorkerInfo
        self._id_to_addr: dict[str, str] = {}  # worker_id -> worker_addr
        # The latest accepted incarnation survives unregister. Without this
        # tombstone, a delayed first-registration request could resurrect a
        # retired process at a reused address.
        self._last_worker_ids: dict[str, str] = {}
        # Incarnation IDs are one-shot fencing tokens. Keeping retired IDs
        # prevents an old delayed request from becoming valid at another
        # address after the original owner has been unregistered.
        self._used_worker_ids: set[str] = set()
        self._lock = asyncio.Lock()

    async def register(
        self,
        worker_addr: str,
        worker_id: str,
        expected_worker_id: str | None,
    ) -> str:
        """CAS a caller-generated incarnation into ``worker_addr``.

        Returns ``created``, ``replayed``, or ``replaced``. Invalid or stale
        transitions raise ``ValueError`` without changing registry state.
        """

        async with self._lock:
            current = self._workers.get(worker_addr)
            if current is not None:
                if current.worker_id == worker_id:
                    if current.registered_from_worker_id == expected_worker_id:
                        return "replayed"
                    raise ValueError(f"Registration replay mismatch for {worker_addr}")
            if worker_id in self._used_worker_ids:
                raise ValueError(f"Worker ID {worker_id} has already been used")

            if current is not None:
                if expected_worker_id != current.worker_id:
                    raise ValueError(
                        f"Worker epoch mismatch for {worker_addr}: "
                        f"expected {current.worker_id}, got {expected_worker_id}"
                    )
                action = "replaced"
            else:
                if worker_addr not in self._last_worker_ids:
                    if expected_worker_id is not None:
                        raise ValueError(
                            f"Worker {worker_addr} has no predecessor "
                            f"{expected_worker_id}"
                        )
                    action = "created"
                else:
                    last_worker_id = self._last_worker_ids[worker_addr]
                    if (
                        expected_worker_id != last_worker_id
                        or worker_id == last_worker_id
                    ):
                        raise ValueError(
                            f"Worker epoch mismatch for retired {worker_addr}: "
                            f"expected predecessor {last_worker_id}"
                        )
                    action = "replaced"

            existing_addr = self._id_to_addr.get(worker_id)
            if existing_addr is not None and existing_addr != worker_addr:
                raise ValueError(
                    f"Worker ID {worker_id} is already active at {existing_addr}"
                )

            if current is not None:
                self._id_to_addr.pop(current.worker_id, None)
            self._workers[worker_addr] = WorkerInfo(
                worker_id=worker_id,
                worker_addr=worker_addr,
                registered_from_worker_id=expected_worker_id,
            )
            self._id_to_addr[worker_id] = worker_addr
            self._last_worker_ids[worker_addr] = worker_id
            self._used_worker_ids.add(worker_id)
            return action

    async def unregister(self, worker_addr: str, worker_id: str) -> bool:
        """Remove only the exact active incarnation, preserving its tombstone."""

        async with self._lock:
            current = self._workers.get(worker_addr)
            if current is None or current.worker_id != worker_id:
                return False
            self._workers.pop(worker_addr)
            self._id_to_addr.pop(worker_id, None)
            return True

    async def get_by_id(self, worker_id: str) -> WorkerInfo | None:
        """Look up a worker by its ID."""
        async with self._lock:
            addr = self._id_to_addr.get(worker_id)
            if addr is None:
                return None
            return self._workers.get(addr)

    async def get_by_addr(self, worker_addr: str) -> WorkerInfo | None:
        async with self._lock:
            return self._workers.get(worker_addr)

    async def get_epoch(self, worker_addr: str) -> tuple[str, str | None]:
        """Return ``(status, worker_id)`` for one canonical address."""

        async with self._lock:
            current = self._workers.get(worker_addr)
            if current is not None:
                return "active", current.worker_id
            last_worker_id = self._last_worker_ids.get(worker_addr)
            if last_worker_id is not None:
                return "retired", last_worker_id
            return "unseen", None

    async def update_health(
        self, worker_addr: str, expected_worker_id: str, healthy: bool
    ) -> bool:
        """Set health only if the probe belongs to the active incarnation."""

        async with self._lock:
            w = self._workers.get(worker_addr)
            if w is None or w.worker_id != expected_worker_id:
                return False
            w.is_healthy = healthy
            return True

    async def get_healthy_workers(self) -> list[WorkerInfo]:
        """Return only workers with ``is_healthy == True``."""
        async with self._lock:
            return [w for w in self._workers.values() if w.is_healthy]

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Return all workers regardless of health."""
        async with self._lock:
            return list(self._workers.values())

    async def list_worker_addrs(self) -> list[str]:
        """Return all registered worker addresses."""
        async with self._lock:
            return list(self._workers.keys())

    async def contains(self, worker_addr: str) -> bool:
        async with self._lock:
            return worker_addr in self._workers


@dataclass(frozen=True)
class SessionRoute:
    """Immutable worker epoch captured when a session was registered."""

    worker_addr: str
    worker_id: str


class SessionRegistry:
    """Maps session API keys and session IDs to worker addresses.

    Pinning persists after reward is set (needed for
    ``/export_trajectories``). Cleaned up after export_trajectories or
    when a worker is deleted.
    """

    def __init__(self) -> None:
        self._key_to_worker: dict[str, str] = {}  # session_api_key -> worker_addr
        self._key_to_worker_id: dict[str, str] = {}
        self._key_to_id: dict[str, str] = {}  # session_api_key -> session_id
        self._id_to_worker: dict[str, str] = {}  # session_id -> worker_addr
        self._id_to_worker_id: dict[str, str] = {}
        self._id_to_key: dict[str, str] = {}  # session_id -> session_api_key
        self._lock = asyncio.Lock()

    async def register_session(
        self,
        session_key: str,
        session_id: str,
        worker_addr: str,
        worker_id: str,
    ) -> None:
        """Register one globally unique session mapping idempotently."""
        await self.register_sessions(
            [(session_key, session_id)], worker_addr, worker_id=worker_id
        )

    async def register_sessions(
        self,
        sessions: list[tuple[str, str]],
        worker_addr: str,
        worker_id: str,
    ) -> None:
        """Atomically register a batch, rejecting key or ID ownership changes."""

        async with self._lock:
            batch_ids: dict[str, str] = {}
            batch_keys: dict[str, str] = {}
            keys_to_refresh: dict[str, str] = {}
            for session_key, session_id in sessions:
                if session_id in batch_ids and batch_ids[session_id] != session_key:
                    raise ValueError(f"Session ID {session_id} is duplicated in batch")
                if session_key in batch_keys and batch_keys[session_key] != session_id:
                    raise ValueError(
                        f"Session key for {session_id} is duplicated in batch"
                    )
                batch_ids[session_id] = session_key
                batch_keys[session_key] = session_id

                existing_id_worker = self._id_to_worker.get(session_id)
                existing_id_worker_id = self._id_to_worker_id.get(session_id)
                existing_id_key = self._id_to_key.get(session_id)
                if existing_id_worker is not None and (
                    existing_id_worker != worker_addr
                    or existing_id_key != session_key
                    or existing_id_worker_id != worker_id
                ):
                    raise ValueError(
                        f"Session ID {session_id} is already registered to another owner"
                    )

                existing_key_worker = self._key_to_worker.get(session_key)
                existing_key_worker_id = self._key_to_worker_id.get(session_key)
                existing_key_id = self._key_to_id.get(session_key)
                if existing_key_worker is not None:
                    if (
                        existing_key_worker != worker_addr
                        or existing_key_worker_id != worker_id
                    ):
                        raise ValueError(
                            f"Session key for {session_id} is already registered "
                            "to another owner"
                        )
                    if existing_key_id is not None and existing_key_id != session_id:
                        keys_to_refresh[session_key] = existing_key_id

            for session_key, old_session_id in keys_to_refresh.items():
                # Keep the old ID routable so its ready trajectory can still
                # be exported. Detach only the refreshed key; revoking the old
                # group must not invalidate the new session's credentials.
                self._id_to_key.pop(old_session_id, None)

            for session_key, session_id in sessions:
                self._key_to_worker[session_key] = worker_addr
                self._key_to_worker_id[session_key] = worker_id
                self._key_to_id[session_key] = session_id
                self._id_to_worker[session_id] = worker_addr
                self._id_to_worker_id[session_id] = worker_id
                self._id_to_key[session_id] = session_key

    async def lookup_by_key(self, session_key: str) -> str | None:
        """Return the worker address pinned to a session API key, or None."""
        async with self._lock:
            return self._key_to_worker.get(session_key)

    async def lookup_by_id(self, session_id: str) -> str | None:
        """Return the worker address pinned to a session ID, or None."""
        async with self._lock:
            return self._id_to_worker.get(session_id)

    async def route_by_key(self, session_key: str) -> SessionRoute | None:
        """Return the address and the exact worker epoch stored for a key."""

        async with self._lock:
            worker_addr = self._key_to_worker.get(session_key)
            if worker_addr is None:
                return None
            return SessionRoute(
                worker_addr=worker_addr,
                worker_id=self._key_to_worker_id[session_key],
            )

    async def route_by_id(self, session_id: str) -> SessionRoute | None:
        """Return the address and the exact worker epoch stored for an ID."""

        async with self._lock:
            worker_addr = self._id_to_worker.get(session_id)
            if worker_addr is None:
                return None
            return SessionRoute(
                worker_addr=worker_addr,
                worker_id=self._id_to_worker_id[session_id],
            )

    async def revoke_by_worker(self, worker_addr: str, worker_id: str) -> int:
        """Remove all sessions pinned to an exact worker incarnation.

        Returns the number of session keys removed.
        """
        async with self._lock:
            keys_to_remove = [
                k
                for k, owner_id in self._key_to_worker_id.items()
                if owner_id == worker_id and self._key_to_worker.get(k) == worker_addr
            ]
            ids_to_remove = [
                k
                for k, owner_id in self._id_to_worker_id.items()
                if owner_id == worker_id and self._id_to_worker.get(k) == worker_addr
            ]
            for k in keys_to_remove:
                del self._key_to_worker[k]
                self._key_to_worker_id.pop(k, None)
                self._key_to_id.pop(k, None)
            for k in ids_to_remove:
                self._id_to_key.pop(k, None)
                self._id_to_worker_id.pop(k, None)
                del self._id_to_worker[k]
            return len(keys_to_remove)

    async def revoke_session(self, session_id: str) -> bool:
        """Remove a single session by its ID.

        Removes both the session_id→worker and session_key→worker mappings.
        Called after ``/export_trajectories`` to prevent unbounded growth.

        Returns True if the session was found and removed, False otherwise.
        """
        async with self._lock:
            if session_id not in self._id_to_worker:
                return False
            del self._id_to_worker[session_id]
            self._id_to_worker_id.pop(session_id, None)
            session_key = self._id_to_key.pop(session_id, None)
            if session_key is not None:
                self._key_to_worker.pop(session_key, None)
                self._key_to_worker_id.pop(session_key, None)
                self._key_to_id.pop(session_key, None)
            return True

    async def session_key_for_id(self, session_id: str) -> str | None:
        async with self._lock:
            return self._id_to_key.get(session_id)

    async def count(self) -> int:
        """Return the number of registered session keys."""
        async with self._lock:
            return len(self._key_to_worker)


@dataclass
class ModelInfo:
    """A registered model (internal or external)."""

    name: str
    url: str  # empty string for internal models
    api_key: str | None
    data_proxy_addrs: list[str] = field(default_factory=list)


@dataclass
class GroupInfo:
    """A registered group of sessions."""

    group_id: str
    worker_addr: str
    worker_id: str
    session_ids: list[str] = field(default_factory=list)
    session_api_keys: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe registry for model routing."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        url: str,
        api_key: str | None,
        data_proxy_addrs: list[str],
    ) -> None:
        async with self._lock:
            self._models[name] = ModelInfo(
                name=name,
                url=url,
                api_key=api_key,
                data_proxy_addrs=data_proxy_addrs,
            )

    async def get(self, name: str) -> ModelInfo | None:
        async with self._lock:
            return self._models.get(name)

    async def first(self) -> ModelInfo | None:
        async with self._lock:
            if not self._models:
                return None
            return next(iter(self._models.values()))

    async def list_names(self) -> list[str]:
        async with self._lock:
            return list(self._models.keys())

    async def remove(self, name: str) -> bool:
        async with self._lock:
            return self._models.pop(name, None) is not None


class GroupRegistry:
    """Maps group IDs to worker addresses and member session IDs."""

    def __init__(self) -> None:
        self._groups: dict[str, GroupInfo] = {}
        self._lock = asyncio.Lock()

    async def register_group(
        self,
        group_id: str,
        worker_addr: str,
        session_ids: list[str],
        worker_id: str,
        session_api_keys: list[str] | None = None,
    ) -> bool:
        """Store a group mapping with idempotent retry semantics."""
        resolved_api_keys = list(session_api_keys or [])
        async with self._lock:
            existing = self._groups.get(group_id)
            if existing is not None:
                if (
                    existing.worker_addr == worker_addr
                    and existing.worker_id == worker_id
                    and existing.session_ids == list(session_ids)
                    and existing.session_api_keys == resolved_api_keys
                ):
                    return False
                raise ValueError(
                    f"Group {group_id} already registered with different sessions"
                )
            self._groups[group_id] = GroupInfo(
                group_id=group_id,
                worker_addr=worker_addr,
                worker_id=worker_id,
                session_ids=list(session_ids),
                session_api_keys=resolved_api_keys,
            )
            return True

    async def lookup(self, group_id: str) -> GroupInfo | None:
        """Return the GroupInfo for a group_id, or None."""
        async with self._lock:
            return self._groups.get(group_id)

    async def revoke(self, group_id: str) -> list[str]:
        """Remove a group. Returns the session_ids that were in the group."""
        async with self._lock:
            info = self._groups.pop(group_id, None)
            if info is None:
                return []
            return info.session_ids

    async def revoke_by_worker(self, worker_addr: str, worker_id: str) -> int:
        """Remove groups owned by an exact incarnation and return their count."""

        async with self._lock:
            group_ids = [
                group_id
                for group_id, info in self._groups.items()
                if info.worker_addr == worker_addr and info.worker_id == worker_id
            ]
            for group_id in group_ids:
                self._groups.pop(group_id, None)
            return len(group_ids)
