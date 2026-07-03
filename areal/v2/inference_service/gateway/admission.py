# SPDX-License-Identifier: Apache-2.0

"""Version-bound admission leases for callback-delivered online rollouts."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum


class OnlineLeaseState(str, Enum):
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OnlineLeaseCapacityError(RuntimeError):
    """The registry cannot accept more externally owned lease state."""


@dataclass(frozen=True)
class OnlineLease:
    lease_id: str
    expected_version: int
    callback_url: str = ""
    ttl_seconds: float = 300.0


@dataclass(frozen=True)
class OnlineLeaseBinding:
    admission_id: str
    worker_addr: str
    worker_id: str
    group_id: str
    session_ids: tuple[str, ...]


class RequestWorkerOwnershipState(str, Enum):
    """Lifecycle of one request's externally owned worker mutation."""

    PINNED = "pinned"
    CLEANUP_PENDING = "cleanup_pending"


class RequestWorkerOwnershipCapacityError(RuntimeError):
    """The registry cannot accept another externally owned mutation."""


@dataclass(frozen=True)
class RequestWorkerBinding:
    """Stable worker selected for retrying one logical mutation."""

    fingerprint: str
    worker_addr: str
    worker_id: str


@dataclass(frozen=True)
class RequestWorkerOwnership:
    """Immutable snapshot of one request's current ownership state."""

    binding: RequestWorkerBinding
    state: RequestWorkerOwnershipState
    cleanup_binding: OnlineLeaseBinding | None = None


@dataclass
class _RequestWorkerOwnershipRecord:
    binding: RequestWorkerBinding
    cleanup_binding: OnlineLeaseBinding | None = None


class RequestWorkerOwnershipRegistry:
    """Remember worker mutations until their external ownership is resolved."""

    def __init__(self, max_owned_records: int = 4096) -> None:
        if max_owned_records < 1:
            raise ValueError("max_owned_records must be >= 1")
        self._records: dict[str, _RequestWorkerOwnershipRecord] = {}
        self._lock = asyncio.Lock()
        self._max_owned_records = max_owned_records

    async def remember(
        self, request_id: str, binding: RequestWorkerBinding
    ) -> RequestWorkerBinding:
        async with self._lock:
            record = self._records.get(request_id)
            if record is not None:
                if record.binding != binding:
                    raise ValueError(
                        f"Request {request_id} has a conflicting worker binding"
                    )
                return record.binding
            if len(self._records) >= self._max_owned_records:
                raise RequestWorkerOwnershipCapacityError(
                    "Request worker ownership capacity is exhausted; retry after "
                    "pending mutations or cleanup complete"
                )
            self._records[request_id] = _RequestWorkerOwnershipRecord(binding=binding)
            return binding

    async def recall(
        self, request_id: str, fingerprint: str
    ) -> RequestWorkerOwnership | None:
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return None
            if record.binding.fingerprint != fingerprint:
                raise ValueError(f"Request {request_id} has a conflicting replay")
            state = (
                RequestWorkerOwnershipState.CLEANUP_PENDING
                if record.cleanup_binding is not None
                else RequestWorkerOwnershipState.PINNED
            )
            return RequestWorkerOwnership(
                binding=record.binding,
                state=state,
                cleanup_binding=record.cleanup_binding,
            )

    async def retain_cleanup(
        self,
        request_id: str,
        binding: RequestWorkerBinding,
        cleanup_binding: OnlineLeaseBinding,
    ) -> None:
        """Transfer a pinned mutation to reaper-owned cleanup."""

        async with self._lock:
            record = self._records.get(request_id)
            if record is None or record.binding != binding:
                raise ValueError(f"Request {request_id} has no matching worker owner")
            if (
                cleanup_binding.worker_addr != binding.worker_addr
                or cleanup_binding.worker_id != binding.worker_id
            ):
                raise ValueError(
                    f"Request {request_id} cleanup worker does not match its owner"
                )
            if (
                record.cleanup_binding is not None
                and record.cleanup_binding != cleanup_binding
            ):
                raise ValueError(f"Request {request_id} has a conflicting cleanup")
            record.cleanup_binding = cleanup_binding

    async def pending_cleanups(
        self,
    ) -> list[tuple[str, RequestWorkerBinding, OnlineLeaseBinding]]:
        async with self._lock:
            return [
                (request_id, record.binding, record.cleanup_binding)
                for request_id, record in self._records.items()
                if record.cleanup_binding is not None
            ]

    async def acknowledge_cleanup(
        self,
        request_id: str,
        binding: RequestWorkerBinding,
        cleanup_binding: OnlineLeaseBinding,
    ) -> bool:
        """Remove ownership only when both worker and cleanup bindings match."""

        async with self._lock:
            record = self._records.get(request_id)
            if (
                record is None
                or record.binding != binding
                or record.cleanup_binding != cleanup_binding
            ):
                return False
            self._records.pop(request_id)
            return True

    async def forget(self, request_id: str, binding: RequestWorkerBinding) -> bool:
        """Release a definitively settled worker mutation using exact ownership."""

        async with self._lock:
            record = self._records.get(request_id)
            if (
                record is None
                or record.binding != binding
                or record.cleanup_binding is not None
            ):
                return False
            self._records.pop(request_id)
            return True


@dataclass
class _LeaseRecord:
    lease: OnlineLease
    state: OnlineLeaseState = OnlineLeaseState.AVAILABLE
    failure_reason: str | None = None
    binding: OnlineLeaseBinding | None = None
    terminal_at: float | None = None
    expires_at: float = 0.0
    cleanup_acknowledged: bool = False
    start_in_flight: bool = False


@dataclass(frozen=True)
class ExpiredOnlineLease:
    lease: OnlineLease
    binding: OnlineLeaseBinding | None
    reason: str


class OnlineLeaseRegistry:
    """Single-use online leases shared by every worker behind one gateway."""

    def __init__(
        self,
        max_terminal_records: int = 4096,
        max_owned_records: int = 4096,
    ) -> None:
        if max_terminal_records < 1:
            raise ValueError("max_terminal_records must be >= 1")
        if max_owned_records < 1:
            raise ValueError("max_owned_records must be >= 1")
        self._records: dict[str, _LeaseRecord] = {}
        self._lock = asyncio.Lock()
        self._max_terminal_records = max_terminal_records
        self._max_owned_records = max_owned_records

    @staticmethod
    def _owns_external_state(record: _LeaseRecord) -> bool:
        return (
            record.terminal_at is None
            or record.start_in_flight
            or (record.binding is not None and not record.cleanup_acknowledged)
        )

    def _purge_terminal_records_locked(self) -> None:
        terminal_records = [
            (lease_id, record.terminal_at)
            for lease_id, record in self._records.items()
            if record.terminal_at is not None
            and not record.start_in_flight
            and (record.binding is None or record.cleanup_acknowledged)
        ]
        terminal_records.sort(key=lambda item: item[1])
        excess = max(0, len(terminal_records) - self._max_terminal_records)
        for lease_id, _ in terminal_records[:excess]:
            self._records.pop(lease_id, None)

    async def grant(self, lease: OnlineLease) -> OnlineLease:
        """Publish a lease, treating an identical HTTP replay as idempotent."""

        async with self._lock:
            self._purge_terminal_records_locked()
            existing = self._records.get(lease.lease_id)
            if existing is not None:
                if existing.lease != lease:
                    raise ValueError(f"Lease {lease.lease_id} has a conflicting replay")
                return existing.lease
            owned_records = sum(
                self._owns_external_state(record) for record in self._records.values()
            )
            if owned_records >= self._max_owned_records:
                raise OnlineLeaseCapacityError(
                    "Online lease ownership capacity is exhausted; retry after "
                    "pending starts or cleanup complete"
                )
            if lease.ttl_seconds <= 0:
                raise ValueError("Lease ttl_seconds must be > 0")
            self._records[lease.lease_id] = _LeaseRecord(
                lease=lease,
                expires_at=time.monotonic() + lease.ttl_seconds,
            )
            return lease

    async def try_acquire(self) -> OnlineLease | None:
        """Atomically claim the oldest available lease exactly once."""

        async with self._lock:
            now = time.monotonic()
            for record in self._records.values():
                if (
                    record.state is OnlineLeaseState.AVAILABLE
                    and record.expires_at > now
                ):
                    record.state = OnlineLeaseState.ACQUIRED
                    record.start_in_flight = True
                    return record.lease
            return None

    async def expire_stale(self, now: float | None = None) -> list[ExpiredOnlineLease]:
        """Fail non-terminal leases whose controller-owned lifetime elapsed."""

        resolved_now = time.monotonic() if now is None else now
        expired: list[ExpiredOnlineLease] = []
        async with self._lock:
            for record in self._records.values():
                if record.state in {
                    OnlineLeaseState.COMPLETED,
                    OnlineLeaseState.FAILED,
                    OnlineLeaseState.CANCELLED,
                }:
                    continue
                if record.expires_at > resolved_now:
                    continue
                reason = (
                    f"Online lease expired after {record.lease.ttl_seconds:g} seconds"
                )
                record.state = OnlineLeaseState.FAILED
                record.failure_reason = reason
                record.terminal_at = resolved_now
                expired.append(
                    ExpiredOnlineLease(
                        lease=record.lease,
                        binding=record.binding,
                        reason=reason,
                    )
                )
            self._purge_terminal_records_locked()
        return expired

    async def cancel(self, lease_id: str) -> bool:
        cancelled, _ = await self.cancel_and_take_binding(lease_id)
        return cancelled

    async def cancel_and_take_binding(
        self, lease_id: str
    ) -> tuple[bool, OnlineLeaseBinding | None]:
        """Cancel a lease and return its compensation target atomically."""

        async with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                return False, None
            if record.state in {
                OnlineLeaseState.CANCELLED,
                OnlineLeaseState.FAILED,
            }:
                binding = (
                    None
                    if record.start_in_flight or record.cleanup_acknowledged
                    else record.binding
                )
                return False, binding
            if record.state is OnlineLeaseState.COMPLETED:
                return False, None
            record.state = OnlineLeaseState.CANCELLED
            record.terminal_at = time.monotonic()
            # A start request may already have created the worker session while
            # Router registration is still in flight.  Let that owner reach its
            # commit point before compensation; otherwise cleanup can run first
            # and the late registration would resurrect a stale route.
            binding = None if record.start_in_flight else record.binding
            self._purge_terminal_records_locked()
            return True, binding

    async def fail(self, lease_id: str, reason: str) -> bool:
        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or record.state in {
                OnlineLeaseState.COMPLETED,
                OnlineLeaseState.FAILED,
                OnlineLeaseState.CANCELLED,
            }:
                return False
            record.state = OnlineLeaseState.FAILED
            record.failure_reason = reason
            record.terminal_at = time.monotonic()
            self._purge_terminal_records_locked()
            return True

    async def complete(self, lease_id: str) -> bool:
        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or record.state is not OnlineLeaseState.ACQUIRED:
                return False
            record.state = OnlineLeaseState.COMPLETED
            record.terminal_at = time.monotonic()
            self._purge_terminal_records_locked()
            return True

    async def bind(
        self,
        lease_id: str,
        binding: OnlineLeaseBinding,
    ) -> None:
        """Attach the created session so cancellation can compensate it."""

        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or record.state is not OnlineLeaseState.ACQUIRED:
                raise ValueError(f"Lease {lease_id} is not acquired")
            if record.binding is not None and record.binding != binding:
                raise ValueError(f"Lease {lease_id} is already bound")
            record.binding = binding
            record.cleanup_acknowledged = False

    async def retain_cleanup_binding(
        self,
        lease_id: str,
        binding: OnlineLeaseBinding,
    ) -> None:
        """Persist compensation ownership after a terminal transition won."""

        async with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                raise ValueError(f"Unknown lease {lease_id}")
            if record.binding is not None and record.binding != binding:
                raise ValueError(f"Lease {lease_id} has a conflicting binding")
            record.binding = binding
            record.cleanup_acknowledged = False
            if record.terminal_at is not None:
                record.terminal_at = time.monotonic()

    async def pending_cleanup_bindings(
        self,
    ) -> list[tuple[str, OnlineLeaseBinding]]:
        async with self._lock:
            return [
                (lease_id, record.binding)
                for lease_id, record in self._records.items()
                if record.terminal_at is not None
                and record.binding is not None
                and not record.cleanup_acknowledged
                and not record.start_in_flight
            ]

    async def acknowledge_cleanup(
        self, lease_id: str, binding: OnlineLeaseBinding
    ) -> bool:
        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or record.binding != binding:
                return False
            record.cleanup_acknowledged = True
            self._purge_terminal_records_locked()
            return True

    async def finish_start(
        self, lease_id: str
    ) -> tuple[bool, OnlineLeaseBinding | None]:
        """Release the start pin and return its linearized disposition.

        The boolean is true only when the lease is still acquired at the
        start-session commit point.  A returned binding belongs to a terminal
        lease and must be compensated before reporting start success.
        """

        async with self._lock:
            record = self._records.get(lease_id)
            if record is None:
                return False, None
            record.start_in_flight = False
            active = record.state is OnlineLeaseState.ACQUIRED
            binding = (
                record.binding
                if record.terminal_at is not None
                and record.binding is not None
                and not record.cleanup_acknowledged
                else None
            )
            self._purge_terminal_records_locked()
            return active, binding

    async def get_binding(self, lease_id: str) -> OnlineLeaseBinding | None:
        async with self._lock:
            record = self._records.get(lease_id)
            return None if record is None else record.binding

    async def available_count(self) -> int:
        async with self._lock:
            now = time.monotonic()
            return sum(
                record.state is OnlineLeaseState.AVAILABLE and record.expires_at > now
                for record in self._records.values()
            )


@dataclass(frozen=True)
class ReplayableHTTPResult:
    """Replayable HTTP result for one caller-generated mutation request."""

    status_code: int
    content: bytes
    media_type: str | None = None


@dataclass
class _RequestReplayRecord:
    fingerprint: str
    ready: asyncio.Event
    result: ReplayableHTTPResult | None = None
    terminal_at: float | None = None


@dataclass(frozen=True)
class RequestReplayReservation:
    """Stable handle returned atomically with request ownership."""

    is_owner: bool
    _record: _RequestReplayRecord


class RequestReplayRegistry:
    """Deduplicate caller retries before rerunning a state-changing request."""

    def __init__(self, max_terminal_records: int = 4096) -> None:
        if max_terminal_records < 1:
            raise ValueError("max_terminal_records must be >= 1")
        self._records: dict[str, _RequestReplayRecord] = {}
        self._lock = asyncio.Lock()
        self._max_terminal_records = max_terminal_records

    def _purge_terminal_records_locked(self) -> None:
        terminal = sorted(
            (
                (request_id, record.terminal_at)
                for request_id, record in self._records.items()
                if record.terminal_at is not None
            ),
            key=lambda item: item[1],
        )
        excess = max(0, len(terminal) - self._max_terminal_records)
        for request_id, _ in terminal[:excess]:
            self._records.pop(request_id, None)

    async def reserve(
        self, request_id: str, fingerprint: str
    ) -> RequestReplayReservation:
        """Return an owner/replay handle without a second lookup race."""

        async with self._lock:
            self._purge_terminal_records_locked()
            existing = self._records.get(request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise ValueError(f"Request {request_id} has a conflicting replay")
                return RequestReplayReservation(False, existing)
            record = _RequestReplayRecord(
                fingerprint=fingerprint,
                ready=asyncio.Event(),
            )
            self._records[request_id] = record
            return RequestReplayReservation(True, record)

    async def release_pending(
        self,
        request_id: str,
        fingerprint: str,
        result: ReplayableHTTPResult,
    ) -> None:
        """Wake current duplicates, then allow a future retry to reserve again."""

        async with self._lock:
            record = self._records.get(request_id)
            if (
                record is not None
                and record.fingerprint == fingerprint
                and record.result is None
            ):
                record.result = result
                record.ready.set()
                self._records.pop(request_id, None)

    async def finish(self, request_id: str, result: ReplayableHTTPResult) -> None:
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                raise RuntimeError(f"Unknown request: {request_id}")
            if record.result is not None:
                if record.result != result:
                    raise RuntimeError(
                        f"Request {request_id} was settled inconsistently"
                    )
                return
            record.result = result
            record.terminal_at = time.monotonic()
            record.ready.set()
            self._purge_terminal_records_locked()

    async def wait(
        self, reservation: RequestReplayReservation, timeout: float | None = None
    ) -> ReplayableHTTPResult:
        record = reservation._record
        ready = record.ready
        if timeout is None:
            await ready.wait()
        else:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        if record.result is None:
            raise RuntimeError("Replay reservation settled without a result")
        return record.result
