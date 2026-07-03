# SPDX-License-Identifier: Apache-2.0

"""Version-bound admission leases for callback-delivered online rollouts."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from enum import Enum


class OnlineLeaseState(str, Enum):
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OnlineLeaseCapacityError(RuntimeError):
    """The registry cannot accept more externally owned lease state."""


class RequestReplayCapacityError(RuntimeError):
    """The bounded replay ledger cannot admit another request ID."""


class RequestReplayExpiredError(RuntimeError):
    """A request ID is fenced but its replayable response is unavailable."""


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            hashlib.sha256(self.fingerprint.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def digest_fingerprint(fingerprint: str) -> str:
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


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
        self._records: dict[bytes, _RequestWorkerOwnershipRecord] = {}
        self._lock = asyncio.Lock()
        self._max_owned_records = max_owned_records

    @staticmethod
    def _request_digest(request_id: str | bytes) -> bytes:
        if isinstance(request_id, bytes):
            return request_id
        return hashlib.sha256(request_id.encode("utf-8")).digest()

    async def remember(
        self, request_id: str, binding: RequestWorkerBinding
    ) -> RequestWorkerBinding:
        async with self._lock:
            request_id_digest = self._request_digest(request_id)
            record = self._records.get(request_id_digest)
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
            self._records[request_id_digest] = _RequestWorkerOwnershipRecord(
                binding=binding
            )
            return binding

    async def recall(
        self, request_id: str, fingerprint: str
    ) -> RequestWorkerOwnership | None:
        async with self._lock:
            record = self._records.get(self._request_digest(request_id))
            if record is None:
                return None
            if record.binding.fingerprint != RequestWorkerBinding.digest_fingerprint(
                fingerprint
            ):
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
            record = self._records.get(self._request_digest(request_id))
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
    ) -> list[tuple[bytes, RequestWorkerBinding, OnlineLeaseBinding]]:
        async with self._lock:
            return [
                (request_id_digest, record.binding, record.cleanup_binding)
                for request_id_digest, record in self._records.items()
                if record.cleanup_binding is not None
            ]

    async def acknowledge_cleanup(
        self,
        request_id: str | bytes,
        binding: RequestWorkerBinding,
        cleanup_binding: OnlineLeaseBinding,
    ) -> bool:
        """Remove ownership only when both worker and cleanup bindings match."""

        async with self._lock:
            request_id_digest = self._request_digest(request_id)
            record = self._records.get(request_id_digest)
            if (
                record is None
                or record.binding != binding
                or record.cleanup_binding != cleanup_binding
            ):
                return False
            self._records.pop(request_id_digest)
            return True

    async def forget(self, request_id: str, binding: RequestWorkerBinding) -> bool:
        """Release a definitively settled worker mutation using exact ownership."""

        async with self._lock:
            request_id_digest = self._request_digest(request_id)
            record = self._records.get(request_id_digest)
            if (
                record is None
                or record.binding != binding
                or record.cleanup_binding is not None
            ):
                return False
            self._records.pop(request_id_digest)
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
    export_in_flight: bool = False


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
            or record.export_in_flight
            or (record.binding is not None and not record.cleanup_acknowledged)
        )

    def _purge_terminal_records_locked(self) -> None:
        terminal_records = [
            (lease_id, record.terminal_at)
            for lease_id, record in self._records.items()
            if record.terminal_at is not None
            and not record.start_in_flight
            and not record.export_in_flight
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
                if record.export_in_flight:
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
                    if (
                        record.start_in_flight
                        or record.export_in_flight
                        or record.cleanup_acknowledged
                    )
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
            binding = (
                None
                if record.start_in_flight or record.export_in_flight
                else record.binding
            )
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
            if (
                record is None
                or record.state is not OnlineLeaseState.DELIVERED
                or not record.export_in_flight
            ):
                return False
            record.export_in_flight = False
            record.state = OnlineLeaseState.COMPLETED
            # A successful destructive export consumed the DataProxy session;
            # Router group cleanup is tracked separately by the Gateway export
            # reconciler. The lease binding must not be reaped as cancellation.
            record.cleanup_acknowledged = True
            record.terminal_at = time.monotonic()
            self._purge_terminal_records_locked()
            return True

    async def mark_delivered(
        self,
        lease_id: str,
        ttl_seconds: float,
        *,
        group_id: str | None = None,
        session_ids: tuple[str, ...] | None = None,
    ) -> bool:
        """Enter the bounded export phase and renew its cleanup deadline."""

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or record.state not in {
                OnlineLeaseState.ACQUIRED,
                OnlineLeaseState.DELIVERED,
            }:
                return False
            if record.start_in_flight or record.binding is None:
                return False
            if record.export_in_flight:
                return False
            if group_id is not None and record.binding.group_id != group_id:
                raise ValueError(f"Lease {lease_id} does not own group {group_id}")
            if session_ids is not None and record.binding.session_ids != session_ids:
                raise ValueError(
                    f"Lease {lease_id} does not own the requested sessions"
                )
            record.state = OnlineLeaseState.DELIVERED
            record.export_in_flight = True
            record.expires_at = time.monotonic() + ttl_seconds
            return True

    async def release_export(self, lease_id: str) -> bool:
        """Release an export pin after a non-terminal attempt."""

        async with self._lock:
            record = self._records.get(lease_id)
            if record is None or not record.export_in_flight:
                return False
            record.export_in_flight = False
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
                and not record.export_in_flight
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
    replay_worker_addr: str | None = None
    replay_worker_id: str | None = None


@dataclass
class _RequestReplayRecord:
    fingerprint_digest: bytes
    request_expires_at: float
    ready: asyncio.Event
    result: ReplayableHTTPResult | None = None
    terminal_at: float | None = None
    expired_reason: str | None = None


@dataclass(frozen=True)
class RequestReplayReservation:
    """Stable handle returned atomically with request ownership."""

    is_owner: bool
    _record: _RequestReplayRecord


class RequestReplayRegistry:
    """Bounded in-memory replay ledger for state-changing requests.

    Callers declare a finite absolute retry deadline.  After that deadline the
    request is rejected independently of retained state, so terminal records
    can be reclaimed without making a destructive request executable again.
    Request IDs and semantic fingerprints are retained only as SHA-256 digests.
    """

    def __init__(
        self,
        max_records: int = 4096,
        retry_ttl_seconds: float = 300.0,
        max_result_bytes: int = 16 * 1024 * 1024,
        max_total_result_bytes: int = 64 * 1024 * 1024,
        *,
        max_terminal_records: int | None = None,
    ) -> None:
        if max_terminal_records is not None:
            if max_records != 4096 and max_records != max_terminal_records:
                raise ValueError(
                    "max_records and max_terminal_records must match when both "
                    "are provided"
                )
            # Backward-compatible keyword. Its safety semantics are stronger:
            # the limit now covers pending requests and compact fences too.
            max_records = max_terminal_records
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        if retry_ttl_seconds <= 0:
            raise ValueError("retry_ttl_seconds must be > 0")
        if max_result_bytes < 0:
            raise ValueError("max_result_bytes must be >= 0")
        if max_total_result_bytes < 0:
            raise ValueError("max_total_result_bytes must be >= 0")
        if max_result_bytes > max_total_result_bytes:
            raise ValueError("max_result_bytes must be <= max_total_result_bytes")
        self._records: dict[str, _RequestReplayRecord] = {}
        self._lock = asyncio.Lock()
        self._max_records = max_records
        self._retry_ttl_seconds = retry_ttl_seconds
        self._max_result_bytes = max_result_bytes
        self._max_total_result_bytes = max_total_result_bytes
        self._total_result_bytes = 0

    @staticmethod
    def _digest(value: str) -> bytes:
        return hashlib.sha256(value.encode("utf-8")).digest()

    def _purge_expired_records_locked(self, now: float) -> None:
        expired_ids = [
            request_id_digest
            for request_id_digest, record in self._records.items()
            if record.terminal_at is not None and record.request_expires_at <= now
        ]
        for request_id_digest in expired_ids:
            record = self._records.pop(request_id_digest)
            if record.result is not None:
                self._total_result_bytes -= len(record.result.content)

    async def reserve(
        self,
        request_id: str,
        fingerprint: str,
        *,
        request_expires_at: float,
    ) -> RequestReplayReservation:
        """Return an owner/replay handle without a second lookup race."""

        async with self._lock:
            now = time.time()
            self._purge_expired_records_locked(now)
            if not math.isfinite(request_expires_at) or request_expires_at <= now:
                raise RequestReplayExpiredError("request deadline expired")
            if request_expires_at - now > self._retry_ttl_seconds:
                raise ValueError(
                    "request_expires_at exceeds the maximum replay horizon"
                )
            request_id_digest = self._digest(request_id)
            fingerprint_digest = self._digest(fingerprint)
            existing = self._records.get(request_id_digest)
            if existing is not None:
                if (
                    existing.fingerprint_digest != fingerprint_digest
                    or existing.request_expires_at != request_expires_at
                ):
                    raise ValueError(f"Request {request_id} has a conflicting replay")
                if existing.expired_reason is not None:
                    raise RequestReplayExpiredError(existing.expired_reason)
                return RequestReplayReservation(False, existing)
            if len(self._records) >= self._max_records:
                raise RequestReplayCapacityError(
                    "Request replay capacity is exhausted; retry with an existing "
                    "request_id or restart with a durable replay ledger"
                )
            record = _RequestReplayRecord(
                fingerprint_digest=fingerprint_digest,
                request_expires_at=request_expires_at,
                ready=asyncio.Event(),
            )
            self._records[request_id_digest] = record
            return RequestReplayReservation(True, record)

    async def release_pending(
        self,
        request_id: str,
        fingerprint: str,
        result: ReplayableHTTPResult,
    ) -> None:
        """Wake current duplicates, then allow a future retry to reserve again."""

        async with self._lock:
            request_id_digest = self._digest(request_id)
            record = self._records.get(request_id_digest)
            if (
                record is not None
                and record.fingerprint_digest == self._digest(fingerprint)
                and record.result is None
            ):
                record.result = result
                record.ready.set()
                self._records.pop(request_id_digest, None)

    async def finish(self, request_id: str, result: ReplayableHTTPResult) -> bool:
        """Settle a request and report whether its response body is replayable."""

        async with self._lock:
            now = time.time()
            self._purge_expired_records_locked(now)
            request_id_digest = self._digest(request_id)
            record = self._records.get(request_id_digest)
            if record is None:
                raise RuntimeError(f"Unknown request: {request_id}")
            if record.result is not None:
                if record.result != result:
                    raise RuntimeError(
                        f"Request {request_id} was settled inconsistently"
                    )
                return True
            if record.expired_reason is not None:
                raise RuntimeError(
                    f"Request {request_id} was already fenced: {record.expired_reason}"
                )
            result_bytes = len(result.content)
            replayable = (
                result_bytes <= self._max_result_bytes
                and self._total_result_bytes + result_bytes
                <= self._max_total_result_bytes
            )
            if replayable:
                record.result = result
                self._total_result_bytes += result_bytes
            else:
                record.expired_reason = "response exceeds replay byte limits"
            record.terminal_at = time.monotonic()
            record.ready.set()
            self._purge_expired_records_locked(time.time())
            return replayable

    async def wait(
        self, reservation: RequestReplayReservation, timeout: float | None = None
    ) -> ReplayableHTTPResult:
        record = reservation._record
        ready = record.ready
        if timeout is None:
            await ready.wait()
        else:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        if record.expired_reason is not None:
            raise RequestReplayExpiredError(record.expired_reason)
        if record.result is None:
            raise RuntimeError("Replay reservation settled without a result")
        return record.result
