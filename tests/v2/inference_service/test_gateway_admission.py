"""Tests for version-bound online admission at the V2 gateway."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from areal.v2.inference_service.gateway import admission as admission_module
from areal.v2.inference_service.gateway.admission import (
    OnlineLease,
    OnlineLeaseBinding,
    OnlineLeaseRegistry,
    ReplayableHTTPResult,
    RequestReplayRegistry,
)
from areal.v2.inference_service.gateway.app import create_app
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.gateway.streaming import (
    RouterDestination,
    RouterSessionRegistrationError,
    RouterUnreachableError,
)
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER

ADMIN_KEY = "test-admin-key"
WORKER_ADDR = "http://worker-1:18082"
WORKER_ID = "worker-1-epoch-1"
MODULE = "areal.v2.inference_service.gateway.app"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def _app_client() -> tuple[object, httpx.AsyncClient]:
    app = create_app(
        GatewayConfig(
            admin_api_key=ADMIN_KEY,
            router_addr="http://mock-router:8081",
        )
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
    return app, client


def _configured_app_client(config: GatewayConfig) -> tuple[object, httpx.AsyncClient]:
    app = create_app(config)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
    return app, client


class TestOnlineLeaseRegistry:
    @pytest.mark.asyncio
    async def test_grant_is_idempotent_but_conflicting_replay_fails(self):
        registry = OnlineLeaseRegistry()
        lease = OnlineLease("lease-1", expected_version=2)

        assert await registry.grant(lease) == lease
        assert await registry.grant(lease) == lease
        with pytest.raises(ValueError, match="conflicting replay"):
            await registry.grant(OnlineLease("lease-1", expected_version=3))

    @pytest.mark.asyncio
    async def test_one_lease_allows_exactly_one_concurrent_acquire(self):
        registry = OnlineLeaseRegistry()
        await registry.grant(OnlineLease("lease-1", expected_version=0))

        results = await asyncio.gather(*[registry.try_acquire() for _ in range(8)])

        assert sum(result is not None for result in results) == 1
        assert next(result for result in results if result is not None).lease_id == (
            "lease-1"
        )

    @pytest.mark.asyncio
    async def test_cancel_is_terminal_and_idempotent(self):
        registry = OnlineLeaseRegistry()
        await registry.grant(OnlineLease("lease-1", expected_version=0))

        assert await registry.cancel("lease-1") is True
        assert await registry.cancel("lease-1") is False
        assert await registry.try_acquire() is None

    @pytest.mark.asyncio
    async def test_identical_grant_replay_is_idempotent_after_terminal_state(self):
        registry = OnlineLeaseRegistry()
        lease = OnlineLease("lease-1", expected_version=0)
        await registry.grant(lease)
        assert await registry.cancel("lease-1") is True

        assert await registry.grant(lease) == lease
        assert await registry.available_count() == 0

    @pytest.mark.asyncio
    async def test_terminal_purge_uses_transition_order_not_grant_order(self):
        registry = OnlineLeaseRegistry(max_terminal_records=2)
        old_active = OnlineLease("old-active", expected_version=0)
        await registry.grant(old_active)
        await registry.try_acquire()
        for lease_id in ("terminal-1", "terminal-2"):
            lease = OnlineLease(lease_id, expected_version=0)
            await registry.grant(lease)
            assert await registry.cancel(lease_id) is True

        # This lease was granted first but became terminal most recently, so
        # it must survive a bounded tombstone purge.
        assert await registry.cancel("old-active") is True
        assert await registry.grant(old_active) == old_active
        assert await registry.available_count() == 0

    @pytest.mark.asyncio
    async def test_expired_acquired_lease_becomes_terminal(self):
        registry = OnlineLeaseRegistry()
        lease = OnlineLease(
            "lease-expired",
            expected_version=7,
            callback_url="http://controller",
            ttl_seconds=10.0,
        )
        await registry.grant(lease)
        assert await registry.try_acquire() == lease

        expired = await registry.expire_stale(now=time.monotonic() + 11.0)

        assert len(expired) == 1
        assert expired[0].lease == lease
        assert "expired" in expired[0].reason
        assert await registry.try_acquire() is None

    @pytest.mark.asyncio
    async def test_terminal_lease_retains_cleanup_until_acknowledged(self):
        registry = OnlineLeaseRegistry()
        lease = OnlineLease("lease-cleanup", expected_version=0)
        binding = OnlineLeaseBinding(
            admission_id="lease-cleanup",
            worker_addr=WORKER_ADDR,
            group_id="grp-cleanup",
            session_ids=("session-1",),
        )
        await registry.grant(lease)
        await registry.try_acquire()
        assert await registry.cancel("lease-cleanup") is True

        # A session may finish creation after cancellation won. Its owner can
        # still attach the compensation target for the background reconciler.
        await registry.retain_cleanup_binding("lease-cleanup", binding)
        await registry.finish_start("lease-cleanup")
        assert await registry.pending_cleanup_bindings() == [("lease-cleanup", binding)]
        cancelled, retry_binding = await registry.cancel_and_take_binding(
            "lease-cleanup"
        )
        assert cancelled is False
        assert retry_binding == binding

        assert await registry.acknowledge_cleanup("lease-cleanup", binding) is True
        assert await registry.pending_cleanup_bindings() == []
        assert await registry.cancel_and_take_binding("lease-cleanup") == (
            False,
            None,
        )

    @pytest.mark.asyncio
    async def test_expired_in_flight_start_is_pinned_until_late_binding_arrives(self):
        registry = OnlineLeaseRegistry(max_terminal_records=1)
        lease = OnlineLease("lease-in-flight", expected_version=0, ttl_seconds=1.0)
        await registry.grant(lease)
        await registry.try_acquire()
        await registry.expire_stale(now=time.monotonic() + 2.0)

        other = OnlineLease("other-terminal", expected_version=0)
        await registry.grant(other)
        await registry.cancel(other.lease_id)
        binding = OnlineLeaseBinding(
            admission_id=lease.lease_id,
            worker_addr=WORKER_ADDR,
            group_id="late-group",
            session_ids=("late-session",),
        )

        # The terminal-cap purge must not drop a lease while its DataProxy
        # start request can still return a late 201.
        await registry.retain_cleanup_binding(lease.lease_id, binding)
        assert await registry.pending_cleanup_bindings() == []
        await registry.finish_start(lease.lease_id)
        assert (lease.lease_id, binding) in await registry.pending_cleanup_bindings()

    @pytest.mark.asyncio
    async def test_unacknowledged_cleanup_survives_terminal_cap_pressure(self):
        registry = OnlineLeaseRegistry(max_terminal_records=1)
        lease = OnlineLease("lease-pending-cleanup", expected_version=0)
        binding = OnlineLeaseBinding(
            admission_id=lease.lease_id,
            worker_addr=WORKER_ADDR,
            group_id="group-pending-cleanup",
            session_ids=("session-pending-cleanup",),
        )
        await registry.grant(lease)
        await registry.try_acquire()
        await registry.bind(lease.lease_id, binding)
        await registry.cancel(lease.lease_id)
        await registry.finish_start(lease.lease_id)

        newer = OnlineLease("newer-terminal", expected_version=0)
        await registry.grant(newer)
        await registry.cancel(newer.lease_id)

        assert await registry.pending_cleanup_bindings() == [(lease.lease_id, binding)]

    @pytest.mark.asyncio
    async def test_pending_cleanup_applies_backpressure_instead_of_growing_unbounded(
        self,
    ):
        with pytest.raises(RuntimeError, match="ownership capacity"):
            registry = OnlineLeaseRegistry(max_owned_records=1)
            lease = OnlineLease("lease-owned", expected_version=0)
            binding = OnlineLeaseBinding(
                admission_id=lease.lease_id,
                worker_addr=WORKER_ADDR,
                group_id="group-owned",
                session_ids=("session-owned",),
            )
            await registry.grant(lease)
            await registry.try_acquire()
            await registry.bind(lease.lease_id, binding)
            await registry.cancel(lease.lease_id)
            await registry.finish_start(lease.lease_id)
            await registry.grant(OnlineLease("lease-overflow", expected_version=0))

        await registry.acknowledge_cleanup(lease.lease_id, binding)
        overflow = OnlineLease("lease-overflow", expected_version=0)
        assert await registry.grant(overflow) == overflow


class TestRequestReplayRegistry:
    @pytest.mark.asyncio
    async def test_replay_handle_survives_release_between_reserve_and_wait(self):
        registry = RequestReplayRegistry()
        owner = await registry.reserve("request-1", "fingerprint")
        replay = await registry.reserve("request-1", "fingerprint")
        transient = ReplayableHTTPResult(429, b'{"error":"no capacity"}')

        assert owner.is_owner is True
        assert replay.is_owner is False
        await registry.release_pending("request-1", "fingerprint", transient)

        assert await registry.wait(replay, timeout=0.1) == transient


class TestRequestWorkerOwnershipRegistry:
    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="max_owned_records must be >= 1"):
            admission_module.RequestWorkerOwnershipRegistry(max_owned_records=0)

    @pytest.mark.asyncio
    async def test_remember_and_recall_pinned_worker(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )

        assert await registry.remember("request-1", binding) == binding

        ownership = await registry.recall("request-1", "fingerprint-1")
        assert ownership is not None
        assert ownership.binding == binding
        assert ownership.state is admission_module.RequestWorkerOwnershipState.PINNED
        assert ownership.cleanup_binding is None

    @pytest.mark.asyncio
    async def test_exact_replay_precedes_capacity_check_without_evicting_owner(self):
        registry = admission_module.RequestWorkerOwnershipRegistry(max_owned_records=1)
        first = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        overflow = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-2",
            worker_addr="http://worker-2:18082",
            worker_id="worker-2-epoch-1",
        )
        await registry.remember("request-1", first)

        assert await registry.remember("request-1", first) == first
        with pytest.raises(ValueError, match="conflicting worker binding"):
            await registry.remember("request-1", overflow)
        with pytest.raises(
            admission_module.RequestWorkerOwnershipCapacityError,
            match="ownership capacity",
        ):
            await registry.remember("request-2", overflow)

        retained = await registry.recall("request-1", "fingerprint-1")
        assert retained is not None
        assert retained.binding == first
        assert await registry.recall("request-2", "fingerprint-2") is None

    @pytest.mark.asyncio
    async def test_recall_rejects_conflicting_fingerprint(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        await registry.remember("request-1", binding)

        with pytest.raises(ValueError, match="conflicting replay"):
            await registry.recall("request-1", "different-fingerprint")

    @pytest.mark.asyncio
    async def test_retain_cleanup_transitions_owner_and_exposes_pending_binding(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        worker_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        cleanup_binding = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="group-1",
            session_ids=("session-1",),
        )
        await registry.remember("request-1", worker_binding)

        await registry.retain_cleanup("request-1", worker_binding, cleanup_binding)
        await registry.retain_cleanup("request-1", worker_binding, cleanup_binding)

        ownership = await registry.recall("request-1", "fingerprint-1")
        assert ownership is not None
        assert ownership.state is (
            admission_module.RequestWorkerOwnershipState.CLEANUP_PENDING
        )
        assert ownership.cleanup_binding == cleanup_binding
        assert await registry.pending_cleanups() == [
            ("request-1", worker_binding, cleanup_binding)
        ]

    @pytest.mark.asyncio
    async def test_retain_cleanup_rejects_stale_owner_and_conflicting_target(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        worker_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        stale_worker_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr="http://worker-2:18082",
            worker_id="worker-2-epoch-1",
        )
        cleanup_binding = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="group-1",
            session_ids=("session-1",),
        )
        wrong_worker_cleanup = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr="http://worker-2:18082",
            group_id="group-1",
            session_ids=("session-1",),
        )
        conflicting_cleanup = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="group-2",
            session_ids=("session-2",),
        )
        await registry.remember("request-1", worker_binding)

        with pytest.raises(ValueError, match="no matching worker owner"):
            await registry.retain_cleanup(
                "request-1", stale_worker_binding, cleanup_binding
            )
        with pytest.raises(ValueError, match="cleanup worker does not match"):
            await registry.retain_cleanup(
                "request-1", worker_binding, wrong_worker_cleanup
            )
        await registry.retain_cleanup("request-1", worker_binding, cleanup_binding)
        with pytest.raises(ValueError, match="conflicting cleanup"):
            await registry.retain_cleanup(
                "request-1", worker_binding, conflicting_cleanup
            )

    @pytest.mark.asyncio
    async def test_cleanup_acknowledgement_is_exact_before_removing_owner(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        worker_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        cleanup_binding = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="group-1",
            session_ids=("session-1",),
        )
        different_cleanup = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="different-group",
            session_ids=("different-session",),
        )
        await registry.remember("request-1", worker_binding)
        await registry.retain_cleanup("request-1", worker_binding, cleanup_binding)

        assert (
            await registry.acknowledge_cleanup(
                "request-1", worker_binding, different_cleanup
            )
            is False
        )
        assert await registry.pending_cleanups() == [
            ("request-1", worker_binding, cleanup_binding)
        ]
        assert (
            await registry.acknowledge_cleanup(
                "request-1", worker_binding, cleanup_binding
            )
            is True
        )
        assert await registry.recall("request-1", "fingerprint-1") is None
        assert (
            await registry.acknowledge_cleanup(
                "request-1", worker_binding, cleanup_binding
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_forget_is_exact_and_releases_capacity(self):
        registry = admission_module.RequestWorkerOwnershipRegistry(max_owned_records=1)
        binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        different_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr="http://worker-2:18082",
            worker_id="worker-2-epoch-1",
        )
        replacement = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-2",
            worker_addr="http://worker-2:18082",
            worker_id="worker-2-epoch-1",
        )
        await registry.remember("request-1", binding)

        assert await registry.forget("request-1", different_binding) is False
        assert await registry.recall("request-1", "fingerprint-1") is not None
        assert await registry.forget("request-1", binding) is True
        assert await registry.forget("request-1", binding) is False
        assert await registry.remember("request-2", replacement) == replacement

    @pytest.mark.asyncio
    async def test_forget_cannot_drop_cleanup_pending_ownership(self):
        registry = admission_module.RequestWorkerOwnershipRegistry()
        worker_binding = admission_module.RequestWorkerBinding(
            fingerprint="fingerprint-1",
            worker_addr=WORKER_ADDR,
            worker_id=WORKER_ID,
        )
        cleanup_binding = OnlineLeaseBinding(
            admission_id="request-1",
            worker_addr=WORKER_ADDR,
            group_id="group-1",
            session_ids=("session-1",),
        )
        await registry.remember("request-1", worker_binding)
        await registry.retain_cleanup("request-1", worker_binding, cleanup_binding)

        assert await registry.forget("request-1", worker_binding) is False
        assert await registry.pending_cleanups() == [
            ("request-1", worker_binding, cleanup_binding)
        ]


class TestGatewayOnlineAdmission:
    def test_request_ownership_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="max_pending_request_owners must be >= 1"):
            create_app(
                GatewayConfig(
                    admin_api_key=ADMIN_KEY,
                    router_addr="http://mock-router:8081",
                    max_pending_request_owners=0,
                )
            )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_zero_lease_returns_429_before_any_downstream_call(
        self, mock_query, mock_forward, mock_register
    ):
        _, client = _app_client()
        async with client:
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-zero-lease",
                },
                headers=_headers(),
            )

        assert response.status_code == 429
        mock_query.assert_not_awaited()
        mock_forward.assert_not_awaited()
        mock_register.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_concurrent_zero_lease_replays_wake_and_can_retry_after_grant(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-after-grant",
                "sessions": [{"session_id": "task-1-0", "session_api_key": "key-1"}],
            },
        )
        _, client = _app_client()
        body = {
            "task_id": "task-1",
            "delivery_mode": "callback",
            "request_id": "request-waits",
        }

        async with client:
            rejected = await asyncio.gather(
                *[
                    client.post("/rl/start_session", json=body, headers=_headers())
                    for _ in range(2)
                ]
            )
            await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-after-grant", "expected_version": 0},
                headers=_headers(),
            )
            admitted = await client.post(
                "/rl/start_session", json=body, headers=_headers()
            )

        assert [response.status_code for response in rejected] == [429, 429]
        assert admitted.status_code == 201
        mock_query.assert_awaited_once()
        mock_forward.assert_awaited_once()
        mock_register.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_callback_session_consumes_lease_and_forwards_lease_metadata(
        self, mock_query, mock_forward, mock_register
    ):
        app, client = _app_client()
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-1",
                "sessions": [{"session_id": "task-1-0", "session_api_key": "key-1"}],
            },
        )

        async with client:
            granted = await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-1", "expected_version": 5},
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-consume",
                },
                headers=_headers(),
            )

        assert granted.status_code == 201
        assert response.status_code == 201
        forwarded = json.loads(mock_forward.call_args.args[1])
        assert forwarded["lease_id"] == "lease-1"
        assert forwarded["admission_id"] == "lease-1"
        assert forwarded["expected_version"] == 5
        assert await app.state.online_lease_registry.available_count() == 0

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_lost_success_response_replays_without_consuming_next_lease(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        response_body = {
            "group_id": "grp-1",
            "sessions": [{"session_id": "task-1-0", "session_api_key": "key-1"}],
        }
        mock_forward.return_value = httpx.Response(201, json=response_body)
        app, client = _app_client()
        request_body = {
            "task_id": "task-1",
            "delivery_mode": "callback",
            "request_id": "producer-request-1",
        }

        async with client:
            for lease_id in ("lease-1", "lease-2"):
                await client.post(
                    "/internal/online_leases",
                    json={"lease_id": lease_id, "expected_version": 0},
                    headers=_headers(),
                )
            first = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            replay = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert first.status_code == replay.status_code == 201
        assert first.json() == replay.json() == response_body
        mock_query.assert_awaited_once()
        mock_forward.assert_awaited_once()
        mock_register.assert_awaited_once()
        assert await app.state.online_lease_registry.available_count() == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_request_id_conflict_does_not_consume_another_lease(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-1",
                "sessions": [{"session_id": "task-1-0", "session_api_key": "key-1"}],
            },
        )
        app, client = _app_client()

        async with client:
            for lease_id in ("lease-1", "lease-2"):
                await client.post(
                    "/internal/online_leases",
                    json={"lease_id": lease_id, "expected_version": 0},
                    headers=_headers(),
                )
            first = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "same-request",
                },
                headers=_headers(),
            )
            conflict = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "different-task",
                    "delivery_mode": "callback",
                    "request_id": "same-request",
                },
                headers=_headers(),
            )

        assert first.status_code == 201
        assert conflict.status_code == 409
        assert await app.state.online_lease_registry.available_count() == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_one_lease_admits_exactly_one_concurrent_start(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-1",
                "sessions": [{"session_id": "task-1-0", "session_api_key": "key-1"}],
            },
        )
        _, client = _app_client()
        async with client:
            await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-1", "expected_version": 0},
                headers=_headers(),
            )
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/rl/start_session",
                        json={
                            "task_id": f"task-{index}",
                            "delivery_mode": "callback",
                            "request_id": f"request-{index}",
                        },
                        headers=_headers(),
                    )
                    for index in range(2)
                ]
            )

        assert sorted(response.status_code for response in responses) == [201, 429]
        mock_query.assert_awaited_once()
        mock_forward.assert_awaited_once()
        mock_register.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_session_bypasses_online_admission(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-pull",
                "sessions": [
                    {"session_id": "task-pull-0", "session_api_key": "key-pull"}
                ],
            },
        )
        _, client = _app_client()

        async with client:
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-pull",
                    "delivery_mode": "pull",
                    "group_size": 2,
                },
                headers=_headers(),
            )

        assert response.status_code == 201
        forwarded = json.loads(mock_forward.call_args.args[1])
        assert "lease_id" not in forwarded
        assert forwarded["admission_id"].startswith("gateway-")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_request_id_replays_same_created_session(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        response_body = {
            "group_id": "grp-pull",
            "sessions": [{"session_id": "task-pull-0", "session_api_key": "key-pull"}],
        }
        mock_forward.return_value = httpx.Response(201, json=response_body)
        _, client = _app_client()
        request_body = {
            "task_id": "task-pull",
            "delivery_mode": "pull",
            "request_id": "pull-request-1",
        }

        async with client:
            first = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            replay = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert first.status_code == replay.status_code == 201
        assert first.json() == replay.json() == response_body
        forwarded = json.loads(mock_forward.await_args.args[1])
        assert forwarded["admission_id"] == "pull-request-1"
        mock_query.assert_awaited_once()
        mock_forward.assert_awaited_once()
        mock_register.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_lost_worker_response_retries_same_worker_and_admission(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.ReadError("response lost"),
            httpx.Response(
                201,
                json={
                    "group_id": "grp-pull",
                    "sessions": [
                        {
                            "session_id": "task-pull-0",
                            "session_api_key": "key-pull",
                        }
                    ],
                },
            ),
        ]
        _, client = _app_client()
        request_body = {
            "task_id": "task-pull",
            "delivery_mode": "pull",
            "request_id": "pull-lost-response",
        }

        async with client:
            first = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            retry = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert first.status_code == 502
        assert retry.status_code == 201
        mock_query.assert_awaited_once()
        assert mock_forward.await_count == 2
        forwarded = [json.loads(call.args[1]) for call in mock_forward.await_args_list]
        assert {item["admission_id"] for item in forwarded} == {"pull-lost-response"}
        assert {
            call.args[2][WORKER_ID_HEADER] for call in mock_forward.await_args_list
        } == {WORKER_ID}
        mock_register.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_ownership_capacity_backpressures_without_evicting_oldest(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.side_effect = [
            RouterDestination(WORKER_ADDR, WORKER_ID),
            RouterDestination("http://worker-2:18082", "worker-2-epoch-1"),
            RouterDestination("http://worker-2:18082", "worker-2-epoch-1"),
        ]
        attempts: dict[str, int] = {}

        async def _forward(_url, body, *_args, **_kwargs):
            admission_id = json.loads(body)["admission_id"]
            attempts[admission_id] = attempts.get(admission_id, 0) + 1
            if admission_id == "capacity-start-a" and attempts[admission_id] == 1:
                raise httpx.ReadError("response lost")
            return httpx.Response(
                201,
                json={
                    "group_id": f"group-{admission_id}",
                    "sessions": [
                        {
                            "session_id": f"session-{admission_id}",
                            "session_api_key": f"key-{admission_id}",
                        }
                    ],
                },
            )

        mock_forward.side_effect = _forward
        app, client = _configured_app_client(
            GatewayConfig(
                admin_api_key=ADMIN_KEY,
                router_addr="http://mock-router:8081",
                max_pending_request_owners=1,
            )
        )
        first_body = {
            "task_id": "capacity-a",
            "delivery_mode": "pull",
            "request_id": "capacity-start-a",
        }
        second_body = {
            "task_id": "capacity-b",
            "delivery_mode": "pull",
            "request_id": "capacity-start-b",
        }

        async with client:
            first = await client.post(
                "/rl/start_session", json=first_body, headers=_headers()
            )
            blocked = await client.post(
                "/rl/start_session", json=second_body, headers=_headers()
            )
            recovered = await client.post(
                "/rl/start_session", json=first_body, headers=_headers()
            )
            admitted = await client.post(
                "/rl/start_session", json=second_body, headers=_headers()
            )

        assert [
            first.status_code,
            blocked.status_code,
            recovered.status_code,
            admitted.status_code,
        ] == [502, 503, 201, 201]
        assert attempts == {"capacity-start-a": 2, "capacity-start-b": 1}
        assert mock_query.await_count == 3
        assert mock_register.await_count == 2
        assert (
            await app.state.start_request_workers.recall(
                "capacity-start-a",
                json.dumps(
                    {"task_id": "capacity-a", "delivery_mode": "pull"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
            is None
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_transient_503_allows_same_request_id_to_retry(
        self, mock_query, mock_forward, mock_register
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(503, json={"error": "temporarily overloaded"}),
            httpx.Response(
                201,
                json={
                    "group_id": "grp-pull-retry",
                    "sessions": [
                        {
                            "session_id": "task-pull-retry-0",
                            "session_api_key": "key-pull-retry",
                        }
                    ],
                },
            ),
        ]
        _, client = _app_client()
        request_body = {
            "task_id": "task-pull-retry",
            "delivery_mode": "pull",
            "request_id": "pull-transient-503",
        }

        async with client:
            first = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            retry = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert first.status_code == 503
        assert retry.status_code == 201
        assert mock_forward.await_count == 2
        mock_register.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_requires_caller_generated_request_id(self):
        _, client = _app_client()
        async with client:
            response = await client.post(
                "/rl/start_session",
                json={"task_id": "task-1", "delivery_mode": "callback"},
                headers=_headers(),
            )

        assert response.status_code == 422
        assert "request_id" in response.text

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_callback_group_is_rejected_without_consuming_lease(
        self, mock_query, mock_forward, mock_register
    ):
        app, client = _app_client()
        async with client:
            await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-1", "expected_version": 0},
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "group_size": 2,
                    "request_id": "request-group",
                },
                headers=_headers(),
            )

        assert response.status_code == 422
        assert await app.state.online_lease_registry.available_count() == 1
        mock_query.assert_not_awaited()
        mock_forward.assert_not_awaited()
        mock_register.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_routing_failure_notifies_the_reserved_waiter(
        self, mock_query, mock_forward, mock_register, mock_notify
    ):
        mock_query.side_effect = RouterUnreachableError("router down")
        _, client = _app_client()
        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-1",
                    "expected_version": 2,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-routing-failure",
                },
                headers=_headers(),
            )

        assert response.status_code == 502
        mock_notify.assert_awaited_once()
        assert mock_notify.call_args.args[0] == "http://controller"
        assert mock_notify.call_args.args[2] == "lease-1"
        mock_forward.assert_not_awaited()
        mock_register.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_registration_failure_compensates_created_session(
        self, mock_query, mock_forward, mock_register, mock_revoke, mock_notify
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(
                201,
                json={
                    "group_id": "grp-1",
                    "sessions": [
                        {"session_id": "task-1-0", "session_api_key": "key-1"}
                    ],
                },
            ),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        mock_register.side_effect = RouterUnreachableError("register failed")
        _, client = _app_client()

        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-1",
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-registration-failure",
                },
                headers=_headers(),
            )

        assert response.status_code == 502
        assert mock_forward.await_count == 2
        cleanup_url = mock_forward.await_args_list[1].args[0]
        cleanup_body = json.loads(mock_forward.await_args_list[1].args[1])
        assert cleanup_url == f"{WORKER_ADDR}/rl/cancel_sessions"
        assert cleanup_body == {
            "admission_id": "lease-1",
            "session_ids": ["task-1-0"],
        }
        mock_revoke.assert_awaited_once()
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_callback_registration_conflict_is_compensated_and_terminal(
        self, mock_query, mock_forward, mock_register, mock_revoke, mock_notify
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(
                201,
                json={
                    "group_id": "grp-stale-callback",
                    "sessions": [
                        {
                            "session_id": "task-stale-callback-0",
                            "session_api_key": "key-stale-callback",
                        }
                    ],
                },
            ),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        mock_register.side_effect = RouterSessionRegistrationError(
            409, "worker incarnation is stale"
        )
        mock_revoke.return_value = True
        app, client = _app_client()
        request_body = {
            "task_id": "task-stale-callback",
            "delivery_mode": "callback",
            "request_id": "request-stale-callback",
        }

        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-stale-callback",
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            replay = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert response.status_code == replay.status_code == 409
        assert response.json()["error"] == "worker incarnation is stale"
        assert mock_forward.await_count == 2
        cleanup_body = json.loads(mock_forward.await_args_list[1].args[1])
        assert cleanup_body == {
            "admission_id": "lease-stale-callback",
            "session_ids": ["task-stale-callback-0"],
        }
        mock_revoke.assert_awaited_once()
        mock_notify.assert_awaited_once()
        assert (
            app.state.online_lease_registry._records["lease-stale-callback"].state.value
            == "failed"
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_registration_conflict_cleans_and_forgets_recall(
        self, mock_query, mock_forward, mock_register, mock_revoke
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(
                201,
                json={
                    "group_id": "grp-stale-pull",
                    "sessions": [
                        {
                            "session_id": "task-stale-pull-0",
                            "session_api_key": "key-stale-pull",
                        }
                    ],
                },
            ),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        mock_register.side_effect = RouterSessionRegistrationError(
            409, "worker incarnation is stale"
        )
        mock_revoke.return_value = True
        app, client = _app_client()
        request_body = {
            "task_id": "task-stale-pull",
            "delivery_mode": "pull",
            "request_id": "request-stale-pull",
        }

        async with client:
            response = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            replay = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert response.status_code == replay.status_code == 409
        assert response.json()["error"] == "worker incarnation is stale"
        assert mock_query.await_count == 1
        assert mock_forward.await_count == 2
        cleanup_body = json.loads(mock_forward.await_args_list[1].args[1])
        assert cleanup_body == {
            "admission_id": "request-stale-pull",
            "session_ids": ["task-stale-pull-0"],
        }
        mock_revoke.assert_awaited_once()
        assert (
            await app.state.start_request_workers.recall(
                "request-stale-pull",
                json.dumps(
                    {"task_id": "task-stale-pull", "delivery_mode": "pull"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
            is None
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_failed_conflict_cleanup_is_retained_until_reaper_acknowledges(
        self, mock_query, mock_forward, mock_register, mock_revoke
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        cleanup_retried = asyncio.Event()
        cleanup_attempts = 0

        async def _forward(url, *_args, **_kwargs):
            nonlocal cleanup_attempts
            if url.endswith("/rl/start_session"):
                return httpx.Response(
                    201,
                    json={
                        "group_id": "group-pending-pull-cleanup",
                        "sessions": [
                            {
                                "session_id": "session-pending-pull-cleanup",
                                "session_api_key": "key-pending-pull-cleanup",
                            }
                        ],
                    },
                )
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                return httpx.Response(500, json={"error": "cleanup unavailable"})
            cleanup_retried.set()
            return httpx.Response(200, json={"status": "ok", "removed": 1})

        mock_forward.side_effect = _forward
        mock_register.side_effect = RouterSessionRegistrationError(
            409, "worker incarnation is stale"
        )
        mock_revoke.return_value = True
        app, client = _app_client()
        request_body = {
            "task_id": "task-pending-pull-cleanup",
            "delivery_mode": "pull",
            "request_id": "request-pending-pull-cleanup",
        }
        fingerprint = json.dumps(
            {
                "task_id": "task-pending-pull-cleanup",
                "delivery_mode": "pull",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        async with app.router.lifespan_context(app):
            async with client:
                response = await client.post(
                    "/rl/start_session", json=request_body, headers=_headers()
                )
                pending = await app.state.start_request_workers.recall(
                    "request-pending-pull-cleanup", fingerprint
                )
                assert pending is not None
                assert pending.state is (
                    admission_module.RequestWorkerOwnershipState.CLEANUP_PENDING
                )
                await asyncio.wait_for(cleanup_retried.wait(), timeout=2.5)
                await asyncio.sleep(0)

        assert response.status_code == 409
        assert (
            await app.state.start_request_workers.recall(
                "request-pending-pull-cleanup", fingerprint
            )
            is None
        )
        assert cleanup_attempts == 2
        assert mock_query.await_count == 1
        assert mock_register.await_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_pull_cancelled_owner_with_failed_cleanup_blocks_same_id_forward(
        self, mock_query, mock_forward, mock_register, mock_revoke
    ):
        registration_started = asyncio.Event()
        never_finish_registration = asyncio.Event()
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)

        async def _forward(url, *_args, **_kwargs):
            if url.endswith("/rl/start_session"):
                return httpx.Response(
                    201,
                    json={
                        "group_id": "group-cancelled-pull-cleanup",
                        "sessions": [
                            {
                                "session_id": "session-cancelled-pull-cleanup",
                                "session_api_key": "key-cancelled-pull-cleanup",
                            }
                        ],
                    },
                )
            return httpx.Response(500, json={"error": "cleanup unavailable"})

        async def _register(*_args, **_kwargs):
            registration_started.set()
            await never_finish_registration.wait()

        mock_forward.side_effect = _forward
        mock_register.side_effect = _register
        mock_revoke.return_value = True
        app, client = _app_client()
        request_body = {
            "task_id": "task-cancelled-pull-cleanup",
            "delivery_mode": "pull",
            "request_id": "request-cancelled-pull-cleanup",
        }
        fingerprint = json.dumps(
            {
                "task_id": "task-cancelled-pull-cleanup",
                "delivery_mode": "pull",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        async with client:
            start_task = asyncio.create_task(
                client.post("/rl/start_session", json=request_body, headers=_headers())
            )
            await registration_started.wait()
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task
            retry = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert retry.status_code == 503
        pending = await app.state.start_request_workers.recall(
            "request-cancelled-pull-cleanup", fingerprint
        )
        assert pending is not None
        assert pending.state is (
            admission_module.RequestWorkerOwnershipState.CLEANUP_PENDING
        )
        assert mock_query.await_count == 1
        assert mock_forward.await_count == 2
        assert mock_register.await_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_delete_during_router_registration_cannot_leave_stale_route(
        self,
        mock_query,
        mock_forward,
        mock_register,
        mock_revoke,
        mock_notify,
    ):
        events: list[str] = []
        registration_started = asyncio.Event()
        allow_registration = asyncio.Event()
        cancellation_committed = asyncio.Event()

        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)

        async def _forward(url, *_args, **_kwargs):
            if url.endswith("/rl/start_session"):
                events.append("worker-created")
                return httpx.Response(
                    201,
                    json={
                        "group_id": "group-racing-delete",
                        "sessions": [
                            {
                                "session_id": "session-racing-delete",
                                "session_api_key": "key-racing-delete",
                            }
                        ],
                    },
                )
            events.append("worker-cleanup")
            return httpx.Response(200, json={"status": "ok", "removed": 1})

        async def _register(*_args, **_kwargs):
            events.append("router-register-started")
            registration_started.set()
            await allow_registration.wait()
            events.append("router-register-committed")

        async def _revoke(*_args, **_kwargs):
            events.append("router-cleanup")
            return True

        mock_forward.side_effect = _forward
        mock_register.side_effect = _register
        mock_revoke.side_effect = _revoke
        app, client = _app_client()
        registry = app.state.online_lease_registry
        original_cancel = registry.cancel_and_take_binding

        async def _observable_cancel(*args, **kwargs):
            result = await original_cancel(*args, **kwargs)
            cancellation_committed.set()
            return result

        async with client:
            await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-racing-delete", "expected_version": 0},
                headers=_headers(),
            )
            start_task = asyncio.create_task(
                client.post(
                    "/rl/start_session",
                    json={
                        "task_id": "task-racing-delete",
                        "delivery_mode": "callback",
                        "request_id": "request-racing-delete",
                    },
                    headers=_headers(),
                )
            )
            await registration_started.wait()
            with patch.object(
                registry,
                "cancel_and_take_binding",
                side_effect=_observable_cancel,
            ):
                delete_task = asyncio.create_task(
                    client.delete(
                        "/internal/online_leases/lease-racing-delete",
                        headers=_headers(),
                    )
                )
                await cancellation_committed.wait()
                await asyncio.sleep(0)
                allow_registration.set()
                started, deleted = await asyncio.gather(start_task, delete_task)

        assert deleted.status_code == 200
        assert started.status_code != 201
        if "router-register-committed" in events:
            assert events.index("router-register-committed") < max(
                i for i, event in enumerate(events) if event == "router-cleanup"
            )
        mock_notify.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("delivery_mode", ["callback", "pull"])
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_cancel_during_router_registration_releases_owned_start(
        self,
        mock_query,
        mock_forward,
        mock_register,
        mock_revoke,
        mock_notify,
        delivery_mode,
    ):
        registration_started = asyncio.Event()
        never_finish_registration = asyncio.Event()
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)

        async def _forward(url, *_args, **_kwargs):
            if url.endswith("/rl/start_session"):
                return httpx.Response(
                    201,
                    json={
                        "group_id": f"group-cancel-{delivery_mode}",
                        "sessions": [
                            {
                                "session_id": f"session-cancel-{delivery_mode}",
                                "session_api_key": f"key-cancel-{delivery_mode}",
                            }
                        ],
                    },
                )
            return httpx.Response(200, json={"status": "ok", "removed": 1})

        async def _register(*_args, **_kwargs):
            registration_started.set()
            await never_finish_registration.wait()

        mock_forward.side_effect = _forward
        mock_register.side_effect = _register
        mock_revoke.return_value = True
        app, client = _app_client()
        lease_id = f"lease-cancel-{delivery_mode}"
        request_id = f"request-cancel-{delivery_mode}"

        async with client:
            if delivery_mode == "callback":
                await client.post(
                    "/internal/online_leases",
                    json={
                        "lease_id": lease_id,
                        "expected_version": 0,
                        "callback_url": "http://controller",
                    },
                    headers=_headers(),
                )
            start_task = asyncio.create_task(
                client.post(
                    "/rl/start_session",
                    json={
                        "task_id": f"task-cancel-{delivery_mode}",
                        "delivery_mode": delivery_mode,
                        "request_id": request_id,
                    },
                    headers=_headers(),
                )
            )
            await registration_started.wait()
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task

        request_record = app.state.start_request_registry._records.get(request_id)
        assert request_record is None or request_record.result is not None
        if delivery_mode == "callback":
            lease_record = app.state.online_lease_registry._records[lease_id]
            assert lease_record.start_in_flight is False
        cleanup_urls = [
            call.args[0]
            for call in mock_forward.await_args_list
            if call.args[0].endswith("/rl/cancel_sessions")
        ]
        assert cleanup_urls == [f"{WORKER_ADDR}/rl/cancel_sessions"]
        mock_revoke.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_cancel_during_failure_notification_releases_start_ownership(
        self,
        mock_query,
        mock_forward,
        mock_notify,
    ):
        notification_started = asyncio.Event()
        never_finish_notification = asyncio.Event()
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(409, json={"error": "version mismatch"}),
            httpx.Response(500, json={"error": "cleanup unavailable"}),
        ]

        async def _notify(*_args, **_kwargs):
            notification_started.set()
            await never_finish_notification.wait()

        mock_notify.side_effect = _notify
        app, client = _app_client()
        lease_id = "lease-cancel-during-notify"
        request_id = "request-cancel-during-notify"

        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": lease_id,
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            start_task = asyncio.create_task(
                client.post(
                    "/rl/start_session",
                    json={
                        "task_id": "task-cancel-during-notify",
                        "delivery_mode": "callback",
                        "request_id": request_id,
                    },
                    headers=_headers(),
                )
            )
            await notification_started.wait()
            start_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start_task

        lease_record = app.state.online_lease_registry._records[lease_id]
        assert lease_record.start_in_flight is False
        request_record = app.state.start_request_registry._records.get(request_id)
        assert request_record is None or request_record.result is not None
        retryable_cleanup = OnlineLeaseBinding(
            admission_id=lease_id,
            worker_addr=WORKER_ADDR,
            group_id="",
            session_ids=(),
        )
        assert (
            lease_id,
            retryable_cleanup,
        ) in await app.state.online_lease_registry.pending_cleanup_bindings()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_cancel_during_non_201_binding_releases_start_ownership(
        self,
        mock_query,
        mock_forward,
        mock_notify,
    ):
        """A response-side cancellation must not leave an immortal start pin."""

        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(409, json={"error": "version mismatch"}),
            httpx.Response(200, json={"status": "ok", "removed": 0}),
        ]
        app, client = _app_client()
        registry = app.state.online_lease_registry
        original_retain = registry.retain_cleanup_binding
        retain_started = asyncio.Event()
        allow_retain = asyncio.Event()

        async def _blocking_retain(*args, **kwargs):
            retain_started.set()
            await allow_retain.wait()
            return await original_retain(*args, **kwargs)

        lease_id = "lease-cancel-non-201-bind"
        request_id = "request-cancel-non-201-bind"
        async with client:
            granted = await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": lease_id,
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            assert granted.status_code == 201

            with patch.object(
                registry,
                "retain_cleanup_binding",
                side_effect=_blocking_retain,
            ):
                start_task = asyncio.create_task(
                    client.post(
                        "/rl/start_session",
                        json={
                            "task_id": "task-cancel-non-201-bind",
                            "delivery_mode": "callback",
                            "request_id": request_id,
                        },
                        headers=_headers(),
                    )
                )
                await retain_started.wait()
                start_task.cancel()
                await asyncio.sleep(0)
                allow_retain.set()
                with pytest.raises(asyncio.CancelledError):
                    await start_task

        lease_record = registry._records[lease_id]
        assert lease_record.start_in_flight is False
        request_record = app.state.start_request_registry._records.get(request_id)
        assert request_record is None or request_record.result is not None
        cleanup_urls = [
            call.args[0]
            for call in mock_forward.await_args_list
            if call.args[0].endswith("/rl/cancel_sessions")
        ]
        assert cleanup_urls == [f"{WORKER_ADDR}/rl/cancel_sessions"]
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_second_cancel_during_ambiguous_forward_releases_start_ownership(
        self,
        mock_query,
        mock_forward,
        mock_notify,
    ):
        """Cleanup preparation stays owned after the forward error is caught."""

        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.ReadError("response lost after request was sent"),
            httpx.Response(200, json={"status": "ok", "removed": 0}),
        ]
        app, client = _app_client()
        registry = app.state.online_lease_registry
        original_retain = registry.retain_cleanup_binding
        retain_started = asyncio.Event()
        allow_retain = asyncio.Event()

        async def _blocking_retain(*args, **kwargs):
            retain_started.set()
            await allow_retain.wait()
            return await original_retain(*args, **kwargs)

        lease_id = "lease-second-cancel-forward"
        request_id = "request-second-cancel-forward"
        async with client:
            granted = await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": lease_id,
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            assert granted.status_code == 201

            with (
                patch.object(
                    registry,
                    "bind",
                    side_effect=ValueError("terminal transition won"),
                ),
                patch.object(
                    registry,
                    "retain_cleanup_binding",
                    side_effect=_blocking_retain,
                ),
            ):
                start_task = asyncio.create_task(
                    client.post(
                        "/rl/start_session",
                        json={
                            "task_id": "task-second-cancel-forward",
                            "delivery_mode": "callback",
                            "request_id": request_id,
                        },
                        headers=_headers(),
                    )
                )
                await retain_started.wait()
                start_task.cancel()
                await asyncio.sleep(0)
                allow_retain.set()
                with pytest.raises(asyncio.CancelledError):
                    await start_task

        lease_record = registry._records[lease_id]
        assert lease_record.start_in_flight is False
        request_record = app.state.start_request_registry._records.get(request_id)
        assert request_record is None or request_record.result is not None
        cleanup_urls = [
            call.args[0]
            for call in mock_forward.await_args_list
            if call.args[0].endswith("/rl/cancel_sessions")
        ]
        assert cleanup_urls == [f"{WORKER_ADDR}/rl/cancel_sessions"]
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_second_cancel_during_registration_failure_retention_releases_start(
        self,
        mock_query,
        mock_forward,
        mock_register,
        mock_notify,
    ):
        """A new cancellation cannot interrupt generic registration cleanup."""

        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(
                201,
                json={
                    "group_id": "group-registration-failure-retain",
                    "sessions": [
                        {
                            "session_id": "session-registration-failure-retain",
                            "session_api_key": "key-registration-failure-retain",
                        }
                    ],
                },
            ),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        mock_register.side_effect = RuntimeError("unexpected registration failure")
        app, client = _app_client()
        registry = app.state.online_lease_registry
        original_retain = registry.retain_cleanup_binding
        retain_started = asyncio.Event()
        allow_retain = asyncio.Event()

        async def _blocking_retain(*args, **kwargs):
            retain_started.set()
            await allow_retain.wait()
            return await original_retain(*args, **kwargs)

        lease_id = "lease-second-cancel-registration-failure"
        request_id = "request-second-cancel-registration-failure"
        async with client:
            granted = await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": lease_id,
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            assert granted.status_code == 201

            with patch.object(
                registry,
                "retain_cleanup_binding",
                side_effect=_blocking_retain,
            ):
                start_task = asyncio.create_task(
                    client.post(
                        "/rl/start_session",
                        json={
                            "task_id": "task-second-cancel-registration-failure",
                            "delivery_mode": "callback",
                            "request_id": request_id,
                        },
                        headers=_headers(),
                    )
                )
                await retain_started.wait()
                start_task.cancel()
                await asyncio.sleep(0)
                allow_retain.set()
                with pytest.raises(asyncio.CancelledError):
                    await start_task

        lease_record = registry._records[lease_id]
        assert lease_record.start_in_flight is False
        request_record = app.state.start_request_registry._records.get(request_id)
        assert request_record is None or request_record.result is not None
        cleanup_urls = [
            call.args[0]
            for call in mock_forward.await_args_list
            if call.args[0].endswith("/rl/cancel_sessions")
        ]
        assert cleanup_urls == [f"{WORKER_ADDR}/rl/cancel_sessions"]
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_callback_transient_502_allows_same_request_id_with_new_lease(
        self, mock_query, mock_forward, mock_register, mock_notify
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.ReadError("response lost after request was sent"),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
            httpx.Response(
                201,
                json={
                    "group_id": "grp-callback-retry",
                    "sessions": [
                        {
                            "session_id": "task-callback-retry-0",
                            "session_api_key": "key-callback-retry",
                        }
                    ],
                },
            ),
        ]
        app, client = _app_client()
        request_body = {
            "task_id": "task-callback-retry",
            "delivery_mode": "callback",
            "request_id": "callback-transient-502",
        }

        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-callback-retry-1",
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            first = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-callback-retry-2",
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            retry = await client.post(
                "/rl/start_session", json=request_body, headers=_headers()
            )

        assert first.status_code == 502
        assert retry.status_code == 201
        assert mock_forward.await_count == 3
        assert await app.state.online_lease_registry.available_count() == 0
        mock_register.assert_awaited_once()
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.notify_online_lease_failure", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_ambiguous_forward_failure_cancels_by_admission_id(
        self, mock_query, mock_forward, mock_register, mock_notify
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.ReadError("response lost after request was sent"),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        _, client = _app_client()

        async with client:
            await client.post(
                "/internal/online_leases",
                json={
                    "lease_id": "lease-1",
                    "expected_version": 0,
                    "callback_url": "http://controller",
                },
                headers=_headers(),
            )
            response = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-ambiguous",
                },
                headers=_headers(),
            )

        assert response.status_code == 502
        assert mock_forward.await_count == 2
        cleanup_body = json.loads(mock_forward.await_args_list[1].args[1])
        assert cleanup_body == {"admission_id": "lease-1", "session_ids": []}
        mock_register.assert_not_awaited()
        mock_notify.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_lease_cancellation_cleans_bound_session(
        self, mock_query, mock_forward, mock_register, mock_revoke
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.side_effect = [
            httpx.Response(
                201,
                json={
                    "group_id": "grp-1",
                    "sessions": [
                        {"session_id": "task-1-0", "session_api_key": "key-1"}
                    ],
                },
            ),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        _, client = _app_client()

        async with client:
            await client.post(
                "/internal/online_leases",
                json={"lease_id": "lease-1", "expected_version": 0},
                headers=_headers(),
            )
            started = await client.post(
                "/rl/start_session",
                json={
                    "task_id": "task-1",
                    "delivery_mode": "callback",
                    "request_id": "request-cancel",
                },
                headers=_headers(),
            )
            cancelled = await client.delete(
                "/internal/online_leases/lease-1", headers=_headers()
            )

        assert started.status_code == 201
        assert cancelled.status_code == 200
        assert mock_forward.await_count == 2
        cleanup_body = json.loads(mock_forward.await_args_list[1].args[1])
        assert cleanup_body == {
            "admission_id": "lease-1",
            "session_ids": ["task-1-0"],
        }
        mock_revoke.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_lease_cancellation_atomically_takes_bound_session(
        self, mock_forward, mock_revoke
    ):
        """A stale pre-cancel read must not lose a concurrently bound session."""

        app, client = _app_client()
        registry = app.state.online_lease_registry
        await registry.grant(OnlineLease("lease-race", expected_version=0))
        await registry.try_acquire()
        await registry.bind(
            "lease-race",
            OnlineLeaseBinding(
                admission_id="lease-race",
                worker_addr=WORKER_ADDR,
                group_id="grp-race",
                session_ids=("task-race-0",),
            ),
        )
        await registry.finish_start("lease-race")
        # Model the old get_binding() -> cancel() race: the first read saw no
        # binding, even though one existed by the time cancellation committed.
        registry.get_binding = AsyncMock(return_value=None)
        mock_forward.return_value = httpx.Response(
            200, json={"status": "ok", "removed": 1}
        )

        async with client:
            cancelled = await client.delete(
                "/internal/online_leases/lease-race", headers=_headers()
            )

        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
        cleanup_body = json.loads(mock_forward.await_args.args[1])
        assert cleanup_body == {
            "admission_id": "lease-race",
            "session_ids": ["task-race-0"],
        }
        mock_revoke.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_failed_cleanup_keeps_binding_for_delete_retry(
        self, mock_forward, mock_revoke
    ):
        app, client = _app_client()
        registry = app.state.online_lease_registry
        await registry.grant(OnlineLease("lease-retry", expected_version=0))
        await registry.try_acquire()
        await registry.bind(
            "lease-retry",
            OnlineLeaseBinding(
                admission_id="lease-retry",
                worker_addr=WORKER_ADDR,
                group_id="grp-retry",
                session_ids=("task-retry-0",),
            ),
        )
        await registry.finish_start("lease-retry")
        mock_forward.side_effect = [
            httpx.Response(500, json={"error": "temporary"}),
            httpx.Response(200, json={"status": "ok", "removed": 1}),
        ]
        mock_revoke.return_value = True

        async with client:
            first = await client.delete(
                "/internal/online_leases/lease-retry", headers=_headers()
            )
            second = await client.delete(
                "/internal/online_leases/lease-retry", headers=_headers()
            )

        assert first.status_code == 502
        assert second.status_code == 200
        assert mock_forward.await_count == 2
        assert mock_revoke.await_count == 2


class TestGatewayExportCleanupCapacity:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_ownership_capacity_backpressures_without_evicting_oldest(
        self, mock_query, mock_forward
    ):
        mock_query.side_effect = [
            RouterDestination(WORKER_ADDR, WORKER_ID),
            RouterDestination("http://worker-2:18082", "worker-2-epoch-1"),
            RouterDestination("http://worker-2:18082", "worker-2-epoch-1"),
        ]
        attempts: dict[str, int] = {}

        async def _forward(_url, body, *_args, **_kwargs):
            request_id = json.loads(body)["request_id"]
            attempts[request_id] = attempts.get(request_id, 0) + 1
            if request_id == "capacity-export-a" and attempts[request_id] == 1:
                raise httpx.ReadError("response lost")
            return httpx.Response(200, json={"traj": {"request_id": request_id}})

        mock_forward.side_effect = _forward
        app, client = _configured_app_client(
            GatewayConfig(
                admin_api_key=ADMIN_KEY,
                router_addr="http://mock-router:8081",
                max_pending_request_owners=1,
            )
        )
        first_body = {
            "request_id": "capacity-export-a",
            "session_ids": ["session-a"],
        }
        second_body = {
            "request_id": "capacity-export-b",
            "session_ids": ["session-b"],
        }

        async with client:
            first = await client.post(
                "/export_trajectories", json=first_body, headers=_headers()
            )
            blocked = await client.post(
                "/export_trajectories", json=second_body, headers=_headers()
            )
            recovered = await client.post(
                "/export_trajectories", json=first_body, headers=_headers()
            )
            admitted = await client.post(
                "/export_trajectories", json=second_body, headers=_headers()
            )

        assert [
            first.status_code,
            blocked.status_code,
            recovered.status_code,
            admitted.status_code,
        ] == [502, 503, 200, 200]
        assert attempts == {"capacity-export-a": 2, "capacity-export-b": 1}
        assert mock_query.await_count == 3
        assert (
            await app.state.export_request_workers.recall(
                "capacity-export-a",
                json.dumps(
                    {"session_ids": ["session-a"]},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
            is None
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure_stage", ["router", "forward_exception", "worker_5xx"]
    )
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_cancel_while_releasing_failed_export_settles_all_ownership(
        self, mock_query, mock_forward, failure_stage
    ):
        """Every occupied cleanup-slot exit settles replay and capacity together."""

        if failure_stage == "router":
            mock_query.side_effect = [
                RouterUnreachableError("router temporarily unavailable"),
                RouterDestination(WORKER_ADDR, WORKER_ID),
            ]
            mock_forward.return_value = httpx.Response(
                400, json={"error": "deterministic second request"}
            )
        else:
            mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
            first_outcome = (
                httpx.ReadError("worker response lost")
                if failure_stage == "forward_exception"
                else httpx.Response(503, json={"error": "temporarily overloaded"})
            )
            mock_forward.side_effect = [
                first_outcome,
                httpx.Response(400, json={"error": "deterministic second request"}),
            ]

        app = create_app(
            GatewayConfig(
                admin_api_key=ADMIN_KEY,
                router_addr="http://mock-router:8081",
                forward_timeout=0.01,
                max_pending_export_cleanups=1,
            )
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        registry = app.state.export_request_registry
        original_release = registry.release_pending
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        first_request_id = f"export-cancel-release-{failure_stage}"

        async def _blocking_release(request_id, fingerprint, result):
            if request_id == first_request_id:
                release_started.set()
                await allow_release.wait()
            await original_release(request_id, fingerprint, result)

        first_body = {
            "request_id": first_request_id,
            "session_ids": [f"session-{failure_stage}"],
            "group_id": f"group-{failure_stage}",
        }
        second_body = {
            "request_id": f"export-after-{failure_stage}",
            "session_ids": [f"session-after-{failure_stage}"],
            "group_id": f"group-after-{failure_stage}",
        }

        async with client:
            with patch.object(
                registry, "release_pending", side_effect=_blocking_release
            ):
                export_task = asyncio.create_task(
                    client.post(
                        "/export_trajectories",
                        json=first_body,
                        headers=_headers(),
                    )
                )
                await release_started.wait()
                export_task.cancel()
                await asyncio.sleep(0)
                allow_release.set()
                with pytest.raises(asyncio.CancelledError):
                    await export_task

            request_record = registry._records.get(first_request_id)
            assert request_record is None or request_record.result is not None
            admitted = await client.post(
                "/export_trajectories", json=second_body, headers=_headers()
            )

        assert admitted.status_code == 400

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_cancel_after_destructive_export_hands_cleanup_to_reaper(
        self, mock_query, mock_forward, mock_revoke
    ):
        """A cached 200 must never strand its cleanup slot as reserved."""

        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(200, json={"traj": []})
        # The cancellation-safe finalizer may fail its immediate cleanup.  The
        # cached replay must still observe the reaper-owned pending group and
        # recover it before admitting the next group.
        mock_revoke.side_effect = [False, True, True]
        app = create_app(
            GatewayConfig(
                admin_api_key=ADMIN_KEY,
                router_addr="http://mock-router:8081",
                max_pending_export_cleanups=1,
            )
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        registry = app.state.export_request_registry
        original_finish = registry.finish
        export_settled = asyncio.Event()
        allow_finish_return = asyncio.Event()
        first_request_id = "export-cancel-after-200"

        async def _blocking_finish(request_id, result):
            await original_finish(request_id, result)
            if request_id == first_request_id and not export_settled.is_set():
                export_settled.set()
                await allow_finish_return.wait()

        first_body = {
            "request_id": first_request_id,
            "session_ids": ["session-cancelled-export"],
            "group_id": "group-cancelled-export",
        }
        second_body = {
            "request_id": "export-after-cancelled-export",
            "session_ids": ["session-after-cancelled-export"],
            "group_id": "group-after-cancelled-export",
        }

        async with client:
            with patch.object(registry, "finish", side_effect=_blocking_finish):
                export_task = asyncio.create_task(
                    client.post(
                        "/export_trajectories",
                        json=first_body,
                        headers=_headers(),
                    )
                )
                await export_settled.wait()
                export_task.cancel()
                await asyncio.sleep(0)
                allow_finish_return.set()
                with pytest.raises(asyncio.CancelledError):
                    await export_task

                replay = await client.post(
                    "/export_trajectories", json=first_body, headers=_headers()
                )
                admitted = await client.post(
                    "/export_trajectories", json=second_body, headers=_headers()
                )

        assert replay.status_code == 200
        assert admitted.status_code == 200
        assert mock_revoke.await_count == 3

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_failed_router_cleanup_backpressures_new_export_groups(
        self, mock_query, mock_forward, mock_revoke
    ):
        mock_query.return_value = RouterDestination(WORKER_ADDR, WORKER_ID)
        mock_forward.return_value = httpx.Response(200, json={"traj": []})
        mock_revoke.side_effect = [False, True, True]
        app = create_app(
            GatewayConfig(
                admin_api_key=ADMIN_KEY,
                router_addr="http://mock-router:8081",
                max_pending_export_cleanups=1,
            )
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        first_body = {
            "request_id": "export-cleanup-1",
            "session_ids": ["session-1"],
            "group_id": "group-1",
        }
        second_body = {
            "request_id": "export-cleanup-2",
            "session_ids": ["session-2"],
            "group_id": "group-2",
        }

        async with client:
            first = await client.post(
                "/export_trajectories", json=first_body, headers=_headers()
            )
            blocked = await client.post(
                "/export_trajectories", json=second_body, headers=_headers()
            )
            first_replay = await client.post(
                "/export_trajectories", json=first_body, headers=_headers()
            )
            admitted = await client.post(
                "/export_trajectories", json=second_body, headers=_headers()
            )

        assert first.status_code == 200
        assert blocked.status_code == 503
        assert first_replay.status_code == 200
        assert admitted.status_code == 200
        assert mock_forward.await_count == 2
