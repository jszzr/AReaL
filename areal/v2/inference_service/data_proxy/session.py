# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle management for the data proxy."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600
MAX_EXPORT_REPLAY_RECORDS = 4096


# =============================================================================
# Request/Response Models
# =============================================================================


class TrajectoryDeliveryMode(str, Enum):
    """How completed trajectories are delivered to the session consumer."""

    CALLBACK = "callback"
    PULL = "pull"


class StartSessionRequest(BaseModel):
    """Request to start one or more offline RL sessions.

    When ``group_size`` is greater than 1, the data proxy creates multiple
    sessions atomically. The response always contains a flat ``sessions``
    list — single-session is just ``group_size=1``.

    Completed trajectories use callback delivery by default. Set
    ``delivery_mode`` to ``pull`` when the client will explicitly export them.
    """

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)
    group_size: int = 1
    delivery_mode: TrajectoryDeliveryMode = TrajectoryDeliveryMode.CALLBACK
    lease_id: str | None = None
    admission_id: str | None = None
    expected_version: int | None = None
    request_expires_at: float


class SessionCredentials(BaseModel):
    """One session's identity and authentication key."""

    session_id: str
    session_api_key: str


class StartSessionResponse(BaseModel):
    """Response from start_session — always a list of session credentials.

    The inference Gateway adds ``expected_version`` after consuming a callback
    lease. It is the policy version that must have produced the loss-bearing
    trajectory. Pull and direct Data Proxy responses omit it.
    """

    group_id: str
    sessions: list[SessionCredentials]
    expected_version: int | None = Field(
        default=None,
        description=(
            "Policy version bound to the consumed callback lease; omitted for "
            "pull or direct Data Proxy responses."
        ),
    )


class CancelSessionsRequest(BaseModel):
    """Compensating cleanup for an admitted session group."""

    admission_id: str
    session_ids: list[str]


class SetRewardRequest(BaseModel):
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float
    model: str | None = None


class ExportTrajectoriesRequest(BaseModel):
    """Request to export trajectories for one or more sessions.

    All sessions are exported and merged into a single interactions dict.
    ``group_id`` and ``lease_id`` bind destructive cleanup to the session
    group and callback lease that actually own the requested trajectories.
    """

    request_id: str = Field(min_length=1, max_length=256)
    request_expires_at: float
    session_ids: list[str]
    group_id: str | None = None
    lease_id: str | None = None
    trajectory_id: int | None = None
    discount: float = 1.0
    style: str = "individual"
    remove_session: bool = True

    def replay_fingerprint(self) -> str:
        """Return a canonical fingerprint of the export's semantic inputs."""
        payload = self.model_dump(mode="json", exclude={"request_id"})
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ExportTrajectoriesResponse(BaseModel):
    """Response containing merged serialized interactions."""

    traj: dict[str, Any]


class ExportReplayConflictError(ValueError):
    """Raised when one request ID is reused for a different export request."""


class ExportReplayCapacityError(RuntimeError):
    """Raised when the bounded export replay ledger cannot admit a new ID."""


class ExportReplayExpiredError(RuntimeError):
    """Raised when an export ID is fenced after its response retry horizon."""


class ExportReplayResultTooLargeError(RuntimeError):
    """Raised before commit when an export cannot fit in the replay ledger."""


@dataclass(frozen=True)
class ExportReplayReservation:
    """Atomic export replay lookup result."""

    is_owner: bool
    payload: dict[str, Any] | None = None


@dataclass
class _ExportReplayRecord:
    fingerprint_digest: bytes
    request_expires_at: float
    payload_json: str | None = None
    terminal_at: float | None = None
    expired_reason: str | None = None


@dataclass(frozen=True)
class RewardResult:
    """Internal result returned when an online session closes a trajectory."""

    session_id: str
    trajectory_id: int | None
    interaction_count: int
    ready_transition: bool


@dataclass(frozen=True)
class ReadyNotification:
    session_id: str
    trajectory_id: int
    lease_id: str | None = None
    expected_version: int | None = None
    group_id: str | None = None


@dataclass
class ReadyTrajectory:
    """One ready-but-not-yet-exported online trajectory."""

    trajectory_id: int
    interaction_id: str
    completions: InteractionCache
    created_at: float
    delivery_mode: TrajectoryDeliveryMode = TrajectoryDeliveryMode.CALLBACK
    callback_delivered: bool = False

    @property
    def needs_online_callback(self) -> bool:
        return self.delivery_mode is TrajectoryDeliveryMode.CALLBACK


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    """Unified session data for both offline and online modes.

    Maintains ``active_completions`` (the current in-progress interaction
    cache) and ``ready_trajectories`` (reward-bounded, exportable
    trajectories).

    - **Offline**: one session → one trajectory via ``set_reward`` →
      ``export_trajectory``.
    - **Online**: one persistent session → many reward-bounded trajectories
      via repeated ``set_reward`` → ``export_trajectory`` calls.
    """

    def __init__(
        self,
        session_id: str,
        set_reward_finish_timeout: float = 0.0,
        delivery_mode: TrajectoryDeliveryMode = TrajectoryDeliveryMode.CALLBACK,
        lease_id: str | None = None,
        expected_version: int | None = None,
        group_id: str | None = None,
    ):
        self.session_id = session_id
        self.delivery_mode = delivery_mode
        self.lease_id = lease_id
        self.expected_version = expected_version
        self.group_id = group_id
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._last_access_time = time.time()
        self._lock = threading.Lock()
        self._active_completions = InteractionCache()
        self._ready_trajectories: OrderedDict[int, ReadyTrajectory] = OrderedDict()
        self._next_trajectory_id = 0
        self._last_set_reward_time: float | None = None
        self._last_reward_interaction_id: str | None = None

    def update_last_access(self) -> None:
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    @property
    def active_completions(self) -> InteractionCache:
        return self._active_completions

    @property
    def has_ready_trajectories(self) -> bool:
        with self._lock:
            return bool(self._ready_trajectories)

    def _latest_ready_trajectory_locked(self) -> ReadyTrajectory | None:
        """Return the latest ready trajectory.

        Caller must already hold ``self._lock``.
        """
        if not self._ready_trajectories:
            return None
        return next(reversed(self._ready_trajectories.values()))

    def _resolve_duplicate_ready_locked(
        self, interaction_id: str | None
    ) -> ReadyTrajectory | None:
        latest_ready = self._latest_ready_trajectory_locked()
        if latest_ready is None:
            return None
        if len(self._active_completions) != 0:
            return None
        resolved_interaction_id = interaction_id or latest_ready.interaction_id
        if resolved_interaction_id == latest_ready.interaction_id:
            return latest_ready
        return None

    def _mark_active_trajectory_ready_locked(
        self,
        now: float,
    ) -> RewardResult:
        completions = self._active_completions
        if len(completions) == 0:
            raise ValueError("No interactions in session")

        resolved_interaction_id = self._last_reward_interaction_id
        if resolved_interaction_id is None:
            raise ValueError("No reward has been set for the active trajectory")

        trajectory_id = self._next_trajectory_id
        self._next_trajectory_id += 1
        ready = ReadyTrajectory(
            trajectory_id=trajectory_id,
            interaction_id=resolved_interaction_id,
            completions=completions,
            created_at=now,
            delivery_mode=self.delivery_mode,
        )
        self._ready_trajectories[trajectory_id] = ready
        self._active_completions = InteractionCache()
        self._last_set_reward_time = None
        self._last_reward_interaction_id = None

        return RewardResult(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            interaction_count=len(completions),
            ready_transition=True,
        )

    def _finalize_if_reward_timeout_elapsed_locked(
        self,
        now: float,
    ) -> RewardResult | None:
        if self._last_set_reward_time is None:
            return None
        if now - self._last_set_reward_time < self._set_reward_finish_timeout:
            return None
        return self._mark_active_trajectory_ready_locked(now)

    def set_reward(
        self,
        interaction_id: str | None,
        reward: float,
    ) -> RewardResult:
        """Record reward for the active trajectory."""
        with self._lock:
            now = time.time()
            self._last_access_time = now

            duplicate_ready = self._resolve_duplicate_ready_locked(interaction_id)
            if duplicate_ready is not None:
                return RewardResult(
                    session_id=self.session_id,
                    trajectory_id=duplicate_ready.trajectory_id,
                    interaction_count=len(duplicate_ready.completions),
                    ready_transition=False,
                )

            completions = self._active_completions
            if len(completions) == 0:
                raise ValueError("No interactions in session")

            resolved_interaction_id = interaction_id or completions.last_interaction_id
            if resolved_interaction_id not in completions:
                raise ValueError(f"Interaction {resolved_interaction_id} not found")

            completions.set_reward(resolved_interaction_id, reward)
            self._last_reward_interaction_id = resolved_interaction_id
            self._last_set_reward_time = now

            ready_result = self._finalize_if_reward_timeout_elapsed_locked(now)
            if ready_result is not None:
                return ready_result

            return RewardResult(
                session_id=self.session_id,
                trajectory_id=None,
                interaction_count=len(completions),
                ready_transition=False,
            )

    def finalize_if_reward_timeout_elapsed(
        self,
        now: float | None = None,
    ) -> RewardResult | None:
        with self._lock:
            return self._finalize_if_reward_timeout_elapsed_locked(now or time.time())

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        with self._lock:
            return [
                ReadyNotification(
                    session_id=self.session_id,
                    trajectory_id=ready.trajectory_id,
                    lease_id=self.lease_id,
                    expected_version=self.expected_version,
                    group_id=self.group_id,
                )
                for ready in self._ready_trajectories.values()
                if ready.needs_online_callback and not ready.callback_delivered
            ]

    def mark_online_callback_delivered(self, trajectory_id: int) -> bool:
        with self._lock:
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None or not ready.needs_online_callback:
                return False
            if ready.callback_delivered:
                return True
            ready.callback_delivered = True
            return True

    def add_string_interaction(self, messages: list[dict], response: str) -> str:
        interaction_id = str(uuid.uuid4())
        interaction = InteractionWithTokenLogpReward(
            messages=messages,
            output_message_list=[{"role": "assistant", "content": response}],
        )
        interaction._interaction_id = interaction_id
        self._active_completions[interaction_id] = interaction
        self.update_last_access()
        return interaction_id

    def export_trajectory(
        self,
        discount: float,
        style: str,
        trajectory_id: int | None = None,
    ) -> tuple[int, dict[str, InteractionWithTokenLogpReward]]:
        """Export a ready trajectory.

        Parameters
        ----------
        discount : float
            Reward discount factor passed to
            :pymethod:`InteractionCache.export_interactions`.
        style : str
            Export style (``"individual"`` or ``"concat"``).
        trajectory_id : int | None
            Specific trajectory to export.  When ``None``, the latest
            ready trajectory is exported.

        Returns
        -------
        tuple[int, dict[str, InteractionWithTokenLogpReward]]
            ``(trajectory_id, interactions)``

        Raises
        ------
        KeyError
            If no ready trajectories exist, or the requested
            ``trajectory_id`` is not found.
        """
        trajectory_id, interactions = self.prepare_trajectory_export(
            discount=discount,
            style=style,
            trajectory_id=trajectory_id,
        )
        self.commit_trajectory_export(trajectory_id)
        return trajectory_id, interactions

    def prepare_trajectory_export(
        self,
        discount: float,
        style: str,
        trajectory_id: int | None = None,
    ) -> tuple[int, dict[str, InteractionWithTokenLogpReward]]:
        """Build an export without consuming it until the caller commits."""

        with self._lock:
            if not self._ready_trajectories:
                raise KeyError(f"No ready trajectories for session {self.session_id}")

            target_trajectory_id = trajectory_id
            if target_trajectory_id is None:
                target_trajectory_id = next(reversed(self._ready_trajectories))

            ready = self._ready_trajectories.get(target_trajectory_id)
            if ready is None:
                raise KeyError(
                    f"Trajectory {target_trajectory_id} not found for session {self.session_id}"
                )

        interactions = ready.completions.export_interactions(
            style=style,
            reward_discount=discount,
        )
        return ready.trajectory_id, interactions

    def commit_trajectory_export(self, trajectory_id: int) -> None:
        """Consume a previously prepared trajectory after serialization succeeds."""

        with self._lock:
            ready = self._ready_trajectories.pop(trajectory_id, None)
            if ready is None:
                raise KeyError(
                    f"Trajectory {trajectory_id} not found for session {self.session_id}"
                )


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    """Thread-safe store for session lifecycle management."""

    def __init__(
        self,
        set_reward_finish_timeout: float = 0.0,
        max_export_replay_records: int = MAX_EXPORT_REPLAY_RECORDS,
        export_replay_ttl_seconds: float = 300.0,
        max_export_replay_result_bytes: int = 16 * 1024 * 1024,
        max_export_replay_total_bytes: int = 64 * 1024 * 1024,
    ):
        if max_export_replay_records <= 0:
            raise ValueError("max_export_replay_records must be positive")
        if export_replay_ttl_seconds <= 0:
            raise ValueError("export_replay_ttl_seconds must be positive")
        if max_export_replay_result_bytes < 0:
            raise ValueError("max_export_replay_result_bytes must be non-negative")
        if max_export_replay_total_bytes < 0:
            raise ValueError("max_export_replay_total_bytes must be non-negative")
        if max_export_replay_result_bytes > max_export_replay_total_bytes:
            raise ValueError(
                "max_export_replay_result_bytes must be <= "
                "max_export_replay_total_bytes"
            )
        self._sessions: dict[str, SessionData] = {}
        self._api_key_to_session: dict[str, str] = {}
        self._session_to_api_key: dict[str, str] = {}
        self._export_replays: dict[bytes, _ExportReplayRecord] = {}
        self._max_export_replay_records = max_export_replay_records
        self._export_replay_ttl_seconds = export_replay_ttl_seconds
        self._max_export_replay_result_bytes = max_export_replay_result_bytes
        self._max_export_replay_total_bytes = max_export_replay_total_bytes
        self._export_replay_total_bytes = 0
        self._lock = threading.Lock()
        self._admin_api_key: str = "areal-admin-key"
        self._set_reward_finish_timeout = set_reward_finish_timeout

    def set_admin_key(self, key: str) -> None:
        with self._lock:
            self._admin_api_key = key

    @staticmethod
    def _export_replay_conflict(request_id: str) -> ExportReplayConflictError:
        return ExportReplayConflictError(
            f"request_id {request_id} was replayed with a different export request"
        )

    @staticmethod
    def _replay_digest(value: str) -> bytes:
        return hashlib.sha256(value.encode("utf-8")).digest()

    def _purge_expired_export_replays_locked(self, now: float) -> None:
        expired_ids = [
            request_id_digest
            for request_id_digest, record in self._export_replays.items()
            if record.terminal_at is not None and record.request_expires_at <= now
        ]
        for request_id_digest in expired_ids:
            record = self._export_replays.pop(request_id_digest)
            if record.payload_json is not None:
                self._export_replay_total_bytes -= len(
                    record.payload_json.encode("utf-8")
                )

    def reserve_export_replay(
        self,
        request_id: str,
        request_fingerprint: str,
        *,
        request_expires_at: float,
    ) -> ExportReplayReservation:
        """Reserve a new export or return its detached cached response.

        Pending, completed, and compact expired records all count toward the
        same hard capacity.  This is intentional: forgetting an expired ID
        would permit a destructive export to execute again.
        """

        with self._lock:
            now = time.time()
            self._purge_expired_export_replays_locked(now)
            if not math.isfinite(request_expires_at) or request_expires_at <= now:
                raise ExportReplayExpiredError("request deadline expired")
            if request_expires_at - now > self._export_replay_ttl_seconds:
                raise ValueError(
                    "request_expires_at exceeds the maximum replay horizon"
                )
            request_id_digest = self._replay_digest(request_id)
            fingerprint_digest = self._replay_digest(request_fingerprint)
            record = self._export_replays.get(request_id_digest)
            if record is not None:
                if (
                    record.fingerprint_digest != fingerprint_digest
                    or record.request_expires_at != request_expires_at
                ):
                    raise self._export_replay_conflict(request_id)
                if record.expired_reason is not None:
                    raise ExportReplayExpiredError(record.expired_reason)
                if record.payload_json is None:
                    return ExportReplayReservation(is_owner=False)
                return ExportReplayReservation(
                    is_owner=False,
                    payload=json.loads(record.payload_json),
                )
            if len(self._export_replays) >= self._max_export_replay_records:
                raise ExportReplayCapacityError(
                    "Export replay capacity is exhausted; retry an existing "
                    "request_id or restart with a durable replay ledger"
                )
            self._export_replays[request_id_digest] = _ExportReplayRecord(
                fingerprint_digest=fingerprint_digest,
                request_expires_at=request_expires_at,
            )
            return ExportReplayReservation(is_owner=True)

    def release_export_replay(
        self,
        request_id: str,
        request_fingerprint: str,
    ) -> bool:
        """Release an uncommitted reservation after a side-effect-free failure."""

        with self._lock:
            request_id_digest = self._replay_digest(request_id)
            record = self._export_replays.get(request_id_digest)
            if (
                record is None
                or record.fingerprint_digest != self._replay_digest(request_fingerprint)
                or record.terminal_at is not None
                or record.payload_json is not None
            ):
                return False
            self._export_replays.pop(request_id_digest, None)
            return True

    def finish_export_replay(
        self,
        request_id: str,
        request_fingerprint: str,
        serialized_traj: dict[str, Any],
    ) -> dict[str, Any]:
        """Make an export replayable before its trajectories are consumed."""

        payload_json = json.dumps(
            serialized_traj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_bytes = len(payload_json.encode("utf-8"))
        with self._lock:
            now = time.time()
            self._purge_expired_export_replays_locked(now)
            request_id_digest = self._replay_digest(request_id)
            record = self._export_replays.get(request_id_digest)
            if record is None:
                raise RuntimeError(f"Unknown export request_id {request_id}")
            if record.fingerprint_digest != self._replay_digest(request_fingerprint):
                raise self._export_replay_conflict(request_id)
            if record.expired_reason is not None:
                raise ExportReplayExpiredError(record.expired_reason)
            if record.payload_json is not None:
                return json.loads(record.payload_json)
            if (
                payload_bytes > self._max_export_replay_result_bytes
                or self._export_replay_total_bytes + payload_bytes
                > self._max_export_replay_total_bytes
            ):
                # The trajectory is still intact at this point.  Releasing the
                # pending ID is safe and lets the caller retry after capacity
                # changes or use a differently configured proxy.
                self._export_replays.pop(request_id_digest, None)
                raise ExportReplayResultTooLargeError(
                    "Export response exceeds replay byte limits; trajectory was "
                    "not consumed"
                )
            record.payload_json = payload_json
            record.terminal_at = time.monotonic()
            self._export_replay_total_bytes += payload_bytes
            result = json.loads(payload_json)
            self._purge_expired_export_replays_locked(time.time())
            return result

    def get_export_replay(
        self,
        request_id: str,
        request_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Return a detached JSON snapshot for a completed export replay."""
        with self._lock:
            self._purge_expired_export_replays_locked(time.time())
            record = self._export_replays.get(self._replay_digest(request_id))
            if record is None:
                return None
            if record.fingerprint_digest != self._replay_digest(request_fingerprint):
                raise self._export_replay_conflict(request_id)
            if record.expired_reason is not None:
                raise ExportReplayExpiredError(record.expired_reason)
            if record.payload_json is None:
                return None
            payload_json = record.payload_json
        return json.loads(payload_json)

    def record_export_replay(
        self,
        request_id: str,
        request_fingerprint: str,
        serialized_traj: dict[str, Any],
        *,
        request_expires_at: float,
    ) -> dict[str, Any]:
        """Backward-compatible reserve-and-finish helper."""

        reservation = self.reserve_export_replay(
            request_id,
            request_fingerprint,
            request_expires_at=request_expires_at,
        )
        if not reservation.is_owner:
            if reservation.payload is None:
                raise RuntimeError(f"Export request_id {request_id} is still pending")
            return reservation.payload
        return self.finish_export_replay(
            request_id,
            request_fingerprint,
            serialized_traj,
        )

    @property
    def admin_api_key(self) -> str:
        return self._admin_api_key

    def start_session(
        self,
        task_id: str,
        api_key: str | None = None,
        delivery_mode: TrajectoryDeliveryMode = TrajectoryDeliveryMode.CALLBACK,
        lease_id: str | None = None,
        expected_version: int | None = None,
        group_id: str | None = None,
    ) -> tuple[str, str]:
        """Start a new session, returning (session_id, session_api_key).

        If *api_key* is provided the key is reused (refreshed); otherwise a
        fresh opaque key is generated.
        """
        with self._lock:
            session_prefix = f"{task_id}-{group_id}" if group_id else task_id
            idx = 0
            while f"{session_prefix}-{idx}" in self._sessions:
                idx += 1
            session_id = f"{session_prefix}-{idx}"

            if api_key:
                session_api_key = api_key
                existing_sid = self._api_key_to_session.get(session_api_key)
                if existing_sid is not None:
                    existing_session = self._sessions.get(existing_sid)
                    if (
                        existing_session is not None
                        and not existing_session.has_ready_trajectories
                    ):
                        raise ValueError(
                            f"API key is already bound to active session {existing_sid}."
                        )
                    self._remove_api_keys_for_session(existing_sid)
            else:
                session_api_key = secrets.token_urlsafe(32)
                while (
                    session_api_key in self._api_key_to_session
                    or session_api_key == self._admin_api_key
                ):
                    session_api_key = secrets.token_urlsafe(32)

            self._sessions[session_id] = SessionData(
                session_id=session_id,
                set_reward_finish_timeout=self._set_reward_finish_timeout,
                delivery_mode=delivery_mode,
                lease_id=lease_id,
                expected_version=expected_version,
                group_id=group_id,
            )
            self._api_key_to_session[session_api_key] = session_id
            self._session_to_api_key[session_id] = session_api_key

        return (session_id, session_api_key)

    def get_session_by_api_key(self, api_key: str) -> SessionData | None:
        with self._lock:
            session_id = self._api_key_to_session.get(api_key)
            if session_id is None:
                return None
            return self._sessions.get(session_id)

    def get_or_create_hitl_session(self) -> SessionData:
        """Return the persistent HITL session, creating it if needed."""
        with self._lock:
            session = self._sessions.get("__hitl__")
            if session is None:
                session = SessionData(
                    session_id="__hitl__",
                    set_reward_finish_timeout=self._set_reward_finish_timeout,
                    delivery_mode=TrajectoryDeliveryMode.PULL,
                )
                self._sessions["__hitl__"] = session
            return session

    def get_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def session_ids_for_group(self, group_id: str) -> set[str]:
        """Return the current complete membership of one opaque session group."""

        with self._lock:
            return {
                session_id
                for session_id, session in self._sessions.items()
                if session.group_id == group_id
            }

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._remove_api_keys_for_session(session_id)

    def _remove_api_keys_for_session(self, session_id: str) -> None:
        api_key = self._session_to_api_key.pop(session_id, None)
        if api_key:
            self._api_key_to_session.pop(api_key, None)

    def cleanup_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> None:
        with self._lock:
            stale_sessions: list[str] = []
            for sid, session in self._sessions.items():
                if not session.is_stale(timeout_seconds):
                    continue
                if session.has_ready_trajectories:
                    continue
                stale_sessions.append(sid)

            for sid in stale_sessions:
                self._sessions.pop(sid, None)
                self._remove_api_keys_for_session(sid)

    def finalize_rewarded_trajectories(
        self,
        now: float | None = None,
    ) -> list[RewardResult]:
        with self._lock:
            sessions = list(self._sessions.values())

        finalized: list[RewardResult] = []
        resolved_now = time.time() if now is None else now
        for session in sessions:
            ready_result = session.finalize_if_reward_timeout_elapsed(resolved_now)
            if ready_result is not None:
                finalized.append(ready_result)
        return finalized

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        with self._lock:
            sessions = list(self._sessions.values())

        notifications: list[ReadyNotification] = []
        for session in sessions:
            notifications.extend(session.pending_online_callbacks())
        return notifications

    def mark_online_callback_delivered(
        self, session_id: str, trajectory_id: int
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        return session.mark_online_callback_delivered(trajectory_id)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
