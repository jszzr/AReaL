"""Unit tests for the Router service (Plan 2, Task 3).

Tests worker registry, session registry, routing strategies,
and all router endpoints.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.router import app as router_app
from areal.v2.inference_service.router.app import create_app
from areal.v2.inference_service.router.config import RouterConfig
from areal.v2.inference_service.router.state import (
    SessionRegistry,
    WorkerInfo,
    WorkerRegistry,
)
from areal.v2.inference_service.router.strategies import (
    RoundRobinStrategy,
    get_strategy,
)

# =============================================================================
# Constants
# =============================================================================

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:18082"
WORKER_2 = "http://worker-2:18082"
WORKER_3 = "http://worker-3:18082"
WORKER_ID_1 = "worker-1-epoch-1"
WORKER_ID_2 = "worker-2-epoch-1"
WORKER_ID_3 = "worker-3-epoch-1"


def worker_id_for(worker_addr: str) -> str:
    return {
        WORKER_1: WORKER_ID_1,
        WORKER_2: WORKER_ID_2,
        WORKER_3: WORKER_ID_3,
    }[worker_addr]


def register_payload(
    worker_addr: str,
    worker_id: str | None = None,
    expected_worker_id: str | None = None,
) -> dict[str, str | None]:
    return {
        "worker_addr": worker_addr,
        "worker_id": worker_id or worker_id_for(worker_addr),
        "expected_worker_id": expected_worker_id,
    }


# =============================================================================
# WorkerRegistry unit tests
# =============================================================================


class TestWorkerRegistry:
    @pytest.mark.asyncio
    async def test_register_worker(self):
        reg = WorkerRegistry()
        action = await reg.register(WORKER_1, WORKER_ID_1, None)
        assert action == "created"
        workers = await reg.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_addr == WORKER_1
        assert workers[0].worker_id == WORKER_ID_1
        assert workers[0].is_healthy is True

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self):
        reg = WorkerRegistry()
        assert await reg.register(WORKER_1, WORKER_ID_1, None) == "created"
        assert await reg.register(WORKER_1, WORKER_ID_1, None) == "replayed"
        workers = await reg.get_all_workers()
        assert len(workers) == 1

    @pytest.mark.asyncio
    async def test_unregister_worker_compare_and_delete(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        assert await reg.unregister(WORKER_1, WORKER_ID_1) is True
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_unregister_unknown_noop(self):
        reg = WorkerRegistry()
        assert await reg.unregister("http://unknown:9999", "unknown-id") is False
        assert len(await reg.get_all_workers()) == 0

    @pytest.mark.asyncio
    async def test_health_update(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        assert await reg.update_health(WORKER_1, WORKER_ID_1, False) is True
        workers = await reg.get_all_workers()
        assert workers[0].is_healthy is False

    @pytest.mark.asyncio
    async def test_get_healthy_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        await reg.register(WORKER_2, WORKER_ID_2, None)
        await reg.update_health(WORKER_1, WORKER_ID_1, False)
        healthy = await reg.get_healthy_workers()
        assert len(healthy) == 1
        assert healthy[0].worker_addr == WORKER_2

    @pytest.mark.asyncio
    async def test_get_all_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        await reg.register(WORKER_2, WORKER_ID_2, None)
        await reg.update_health(WORKER_1, WORKER_ID_1, False)
        all_w = await reg.get_all_workers()
        assert len(all_w) == 2

    @pytest.mark.asyncio
    async def test_list_worker_addrs(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        await reg.register(WORKER_2, WORKER_ID_2, None)
        addrs = await reg.list_worker_addrs()
        assert set(addrs) == {WORKER_1, WORKER_2}

    @pytest.mark.asyncio
    async def test_unregister_stale_id_is_noop(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        assert await reg.unregister(WORKER_1, "stale-id") is False
        workers = await reg.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == WORKER_ID_1

    @pytest.mark.asyncio
    async def test_stale_health_completion_cannot_poison_successor(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        await reg.register(WORKER_1, "worker-1-epoch-2", WORKER_ID_1)

        applied = await reg.update_health(WORKER_1, WORKER_ID_1, False)

        assert applied is False
        current = await reg.get_by_addr(WORKER_1)
        assert current is not None
        assert current.worker_id == "worker-1-epoch-2"
        assert current.is_healthy is True

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, WORKER_ID_1, None)
        info = await reg.get_by_id(WORKER_ID_1)
        assert info is not None
        assert info.worker_addr == WORKER_1
        assert info.worker_id == WORKER_ID_1

    @pytest.mark.asyncio
    async def test_get_by_id_unknown_returns_none(self):
        reg = WorkerRegistry()
        result = await reg.get_by_id("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_retired_worker_id_cannot_be_reused_at_another_address(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1, "one-shot-id", None)
        assert await reg.unregister(WORKER_1, "one-shot-id") is True

        with pytest.raises(ValueError, match="already been used"):
            await reg.register(WORKER_2, "one-shot-id", None)


# =============================================================================
# SessionRegistry unit tests
# =============================================================================


class TestSessionRegistry:
    @pytest.mark.asyncio
    async def test_register_session(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_key(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_id(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_unknown_key(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_key("nonexistent") is None

    @pytest.mark.asyncio
    async def test_lookup_unknown_id(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_revoke_by_worker(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        await reg.register_session("key-2", "id-2", WORKER_1, WORKER_ID_1)
        await reg.register_session("key-3", "id-3", WORKER_1, "worker-1-epoch-2")
        count = await reg.revoke_by_worker(WORKER_1, WORKER_ID_1)
        assert count == 2
        assert await reg.lookup_by_key("key-1") is None
        assert await reg.lookup_by_key("key-2") is None
        assert await reg.lookup_by_id("id-1") is None
        assert await reg.lookup_by_id("id-2") is None
        # A successor at the same address is a different owner and is untouched.
        assert await reg.lookup_by_key("key-3") == WORKER_1

    @pytest.mark.asyncio
    async def test_count(self):
        reg = SessionRegistry()
        assert await reg.count() == 0
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        assert await reg.count() == 1
        await reg.register_session("key-2", "id-2", WORKER_2, WORKER_ID_2)
        assert await reg.count() == 2

    @pytest.mark.asyncio
    async def test_cross_worker_rebind_is_rejected(self):
        """A live session identity cannot silently move between workers."""
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1, WORKER_ID_1)
        with pytest.raises(ValueError, match="another owner"):
            await reg.register_session("key-1", "id-1", WORKER_2, WORKER_ID_2)
        assert await reg.lookup_by_key("key-1") == WORKER_1
        assert await reg.lookup_by_id("id-1") == WORKER_1


# =============================================================================
# Routing strategies unit tests
# =============================================================================


class TestRoutingStrategies:
    def test_round_robin_cycling(self):
        s = RoundRobinStrategy()
        w1 = WorkerInfo(worker_id="w1", worker_addr=WORKER_1)
        w2 = WorkerInfo(worker_id="w2", worker_addr=WORKER_2)
        w3 = WorkerInfo(worker_id="w3", worker_addr=WORKER_3)
        workers = [w1, w2, w3]
        picks = [s.pick(workers).worker_addr for _ in range(6)]  # type: ignore[union-attr]
        assert picks == [
            WORKER_1,
            WORKER_2,
            WORKER_3,
            WORKER_1,
            WORKER_2,
            WORKER_3,
        ]

    def test_round_robin_empty(self):
        s = RoundRobinStrategy()
        assert s.pick([]) is None

    def test_get_strategy_round_robin(self):
        s = get_strategy("round_robin")
        assert isinstance(s, RoundRobinStrategy)

    def test_get_strategy_least_busy_not_implemented(self):
        with pytest.raises(NotImplementedError, match="least_busy"):
            get_strategy("least_busy")

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            get_strategy("random")


# =============================================================================
# Router endpoint tests
# =============================================================================


@pytest.fixture
def config():
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,  # effectively disable polling in tests
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def client(config):
    """Create router app and yield an httpx async client.
    Bypasses lifespan (no background health poller) by setting state directly.
    """
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


async def register_workers(client, *worker_addrs: str) -> None:
    for worker_addr in worker_addrs:
        response = await client.post(
            "/register",
            json=register_payload(worker_addr),
            headers=admin_headers(),
        )
        assert response.status_code == 200


class TestRouterEndpoints:
    # ----- Health -----

    @pytest.mark.asyncio
    async def test_health_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] == 0
        assert data["sessions"] == 0
        assert data["strategy"] == "round_robin"

    @pytest.mark.asyncio
    async def test_health_poll_rejects_200_from_different_worker_epoch(
        self, config, monkeypatch
    ):
        probe_finished = asyncio.Event()

        class FakeHealthClient:
            async def get(self, _url):
                probe_finished.set()
                return httpx.Response(
                    200,
                    json={"status": "ok", "worker_id": "successor-epoch"},
                )

            async def aclose(self):
                return None

        fake_client = FakeHealthClient()
        monkeypatch.setattr(
            router_app.httpx,
            "AsyncClient",
            lambda *args, **kwargs: fake_client,
        )
        app = create_app(config)
        await app.state.worker_registry.register(WORKER_1, WORKER_ID_1, None)

        async with app.router.lifespan_context(app):
            await asyncio.wait_for(probe_finished.wait(), timeout=1.0)
            await asyncio.sleep(0)
            worker = await app.state.worker_registry.get_by_addr(WORKER_1)

        assert worker is not None
        assert worker.is_healthy is False

    @pytest.mark.asyncio
    async def test_hitl_route_binds_first_use(self, client):
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )

        resp = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

        pinned = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert pinned.status_code == 200
        assert pinned.json()["worker_addr"] == WORKER_1

    # ----- Worker registration -----

    @pytest.mark.asyncio
    async def test_register_worker_admin_key(self, client):
        resp = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["worker_id"] == WORKER_ID_1
        assert resp.json()["action"] == "created"

        # Verify via /health
        health = (await client.get("/health")).json()
        assert health["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_worker_no_auth_401(self, client):
        resp = await client.post("/register", json=register_payload(WORKER_1))
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_worker_wrong_key_403(self, client):
        resp = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_worker_epoch_reports_unseen_active_and_retired_state(self, client):
        unseen = await client.get(
            "/worker_epoch",
            params={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert unseen.status_code == 200
        assert unseen.json() == {
            "worker_addr": WORKER_1,
            "status": "unseen",
            "worker_id": None,
        }

        assert (
            await client.post(
                "/register", json=register_payload(WORKER_1), headers=admin_headers()
            )
        ).status_code == 200
        active = await client.get(
            "/worker_epoch",
            params={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert active.json() == {
            "worker_addr": WORKER_1,
            "status": "active",
            "worker_id": WORKER_ID_1,
        }

        assert (
            await client.post(
                "/unregister",
                json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
                headers=admin_headers(),
            )
        ).json()["removed"] is True
        retired = await client.get(
            "/worker_epoch",
            params={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert retired.json() == {
            "worker_addr": WORKER_1,
            "status": "retired",
            "worker_id": WORKER_ID_1,
        }

        successor = await client.post(
            "/register",
            json=register_payload(
                WORKER_1,
                worker_id="worker-1-epoch-2",
                expected_worker_id=retired.json()["worker_id"],
            ),
            headers=admin_headers(),
        )
        assert successor.status_code == 200
        assert successor.json()["action"] == "replaced"

    @pytest.mark.asyncio
    async def test_worker_epoch_requires_admin_key(self, client):
        assert (
            await client.get("/worker_epoch", params={"worker_addr": WORKER_1})
        ).status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.parametrize("worker_id", ["", "x" * 129])
    async def test_register_rejects_invalid_worker_id_schema(self, client, worker_id):
        response = await client.post(
            "/register",
            json={
                "worker_addr": WORKER_1,
                "worker_id": worker_id,
                "expected_worker_id": None,
            },
            headers=admin_headers(),
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_requires_nullable_expected_worker_id_field(self, client):
        missing = await client.post(
            "/register",
            json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
            headers=admin_headers(),
        )
        explicit_null = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )

        assert missing.status_code == 422
        assert explicit_null.status_code == 200

    @pytest.mark.asyncio
    async def test_exact_register_replay_does_not_revoke_sessions(self, client):
        registration = register_payload(WORKER_1)
        assert (
            await client.post("/register", json=registration, headers=admin_headers())
        ).status_code == 200
        assert (
            await client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "replay-key", "session_id": "replay-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": WORKER_ID_1,
                    "group_id": "replay-group",
                },
                headers=admin_headers(),
            )
        ).status_code == 200

        replay = await client.post(
            "/register", json=registration, headers=admin_headers()
        )
        routed = await client.post(
            "/route",
            json={"api_key": "replay-key"},
            headers=admin_headers(),
        )

        assert replay.status_code == 200
        assert replay.json() == {
            "status": "ok",
            "worker_id": WORKER_ID_1,
            "action": "replayed",
        }
        assert routed.status_code == 200
        assert routed.json()["worker_id"] == WORKER_ID_1

    @pytest.mark.asyncio
    async def test_crash_successor_cas_replaces_worker_and_revokes_old_owner(
        self, client
    ):
        assert (
            await client.post(
                "/register", json=register_payload(WORKER_1), headers=admin_headers()
            )
        ).status_code == 200
        assert (
            await client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "old-key", "session_id": "old-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": WORKER_ID_1,
                    "group_id": "old-group",
                },
                headers=admin_headers(),
            )
        ).status_code == 200

        successor = await client.post(
            "/register",
            json=register_payload(
                WORKER_1,
                worker_id="worker-1-epoch-2",
                expected_worker_id=WORKER_ID_1,
            ),
            headers=admin_headers(),
        )

        assert successor.status_code == 200
        assert successor.json()["action"] == "replaced"
        assert successor.json()["worker_id"] == "worker-1-epoch-2"
        assert (
            await client._transport.app.state.session_registry.lookup_by_key("old-key")
            is None
        )
        assert (
            await client._transport.app.state.group_registry.lookup("old-group") is None
        )

    @pytest.mark.asyncio
    async def test_cancelled_replacement_finishes_exact_owner_cascade(
        self, client, monkeypatch
    ):
        await register_workers(client, WORKER_1)
        assert (
            await client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "old-key", "session_id": "old-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": WORKER_ID_1,
                    "group_id": "old-group",
                },
                headers=admin_headers(),
            )
        ).status_code == 200

        app = client._transport.app
        session_registry = app.state.session_registry
        original_revoke = session_registry.revoke_by_worker
        cascade_entered = asyncio.Event()
        allow_cascade = asyncio.Event()

        async def pause_before_session_cascade(*args, **kwargs):
            cascade_entered.set()
            await allow_cascade.wait()
            return await original_revoke(*args, **kwargs)

        monkeypatch.setattr(
            session_registry, "revoke_by_worker", pause_before_session_cascade
        )
        replacement = asyncio.create_task(
            client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id="worker-1-epoch-2",
                    expected_worker_id=WORKER_ID_1,
                ),
                headers=admin_headers(),
            )
        )
        await asyncio.wait_for(cascade_entered.wait(), timeout=1.0)
        replacement.cancel()
        allow_cascade.set()
        with pytest.raises(asyncio.CancelledError):
            await replacement

        current = await app.state.worker_registry.get_by_addr(WORKER_1)
        assert current is not None
        assert current.worker_id == "worker-1-epoch-2"
        assert await session_registry.lookup_by_key("old-key") is None
        assert await app.state.group_registry.lookup("old-group") is None

        replay = await client.post(
            "/register",
            json=register_payload(
                WORKER_1,
                worker_id="worker-1-epoch-2",
                expected_worker_id=WORKER_ID_1,
            ),
            headers=admin_headers(),
        )
        assert replay.status_code == 200
        assert replay.json()["action"] == "replayed"

    @pytest.mark.asyncio
    async def test_delayed_old_register_is_rejected_after_replacement(self, client):
        old_registration = register_payload(WORKER_1)
        assert (
            await client.post(
                "/register", json=old_registration, headers=admin_headers()
            )
        ).status_code == 200
        assert (
            await client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id="worker-1-epoch-2",
                    expected_worker_id=WORKER_ID_1,
                ),
                headers=admin_headers(),
            )
        ).status_code == 200

        delayed = await client.post(
            "/register", json=old_registration, headers=admin_headers()
        )

        assert delayed.status_code == 409
        current = await client._transport.app.state.worker_registry.get_by_addr(
            WORKER_1
        )
        assert current is not None
        assert current.worker_id == "worker-1-epoch-2"

    @pytest.mark.asyncio
    async def test_concurrent_successors_from_same_epoch_have_single_winner(
        self, client
    ):
        assert (
            await client.post(
                "/register", json=register_payload(WORKER_1), headers=admin_headers()
            )
        ).status_code == 200

        responses = await asyncio.gather(
            *[
                client.post(
                    "/register",
                    json=register_payload(
                        WORKER_1,
                        worker_id=successor_id,
                        expected_worker_id=WORKER_ID_1,
                    ),
                    headers=admin_headers(),
                )
                for successor_id in ("worker-1-epoch-2", "worker-1-epoch-3")
            ]
        )

        assert sorted(response.status_code for response in responses) == [200, 409]
        winner = next(
            response.json()["worker_id"]
            for response in responses
            if response.status_code == 200
        )
        current = await client._transport.app.state.worker_registry.get_by_addr(
            WORKER_1
        )
        assert current is not None
        assert current.worker_id == winner

    @pytest.mark.asyncio
    async def test_retired_worker_accepts_only_cas_successor(self, client):
        await register_workers(client, WORKER_1)
        assert (
            await client.post(
                "/unregister",
                json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
                headers=admin_headers(),
            )
        ).json()["removed"] is True

        delayed = await client.post(
            "/register", json=register_payload(WORKER_1), headers=admin_headers()
        )
        successor = await client.post(
            "/register",
            json=register_payload(
                WORKER_1,
                worker_id="worker-1-epoch-2",
                expected_worker_id=WORKER_ID_1,
            ),
            headers=admin_headers(),
        )

        assert delayed.status_code == 409
        assert successor.status_code == 200
        assert successor.json()["action"] == "replaced"

    # ----- Worker deletion (with cascade) -----

    @pytest.mark.asyncio
    async def test_delete_worker_cascades(self, client):
        # Register worker + session pinned to it
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "grp-test-1",
            },
            headers=admin_headers(),
        )

        # Delete the worker
        resp = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["removed"] is True
        assert resp.json()["sessions_revoked"] == 1
        assert resp.json()["groups_revoked"] == 1

        # Session key should no longer route
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancelled_unregister_finishes_exact_owner_cascade(
        self, client, monkeypatch
    ):
        await register_workers(client, WORKER_1)
        assert (
            await client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "old-key", "session_id": "old-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": WORKER_ID_1,
                    "group_id": "old-group",
                },
                headers=admin_headers(),
            )
        ).status_code == 200

        app = client._transport.app
        session_registry = app.state.session_registry
        original_revoke = session_registry.revoke_by_worker
        cascade_entered = asyncio.Event()
        allow_cascade = asyncio.Event()

        async def pause_before_session_cascade(*args, **kwargs):
            cascade_entered.set()
            await allow_cascade.wait()
            return await original_revoke(*args, **kwargs)

        monkeypatch.setattr(
            session_registry, "revoke_by_worker", pause_before_session_cascade
        )
        unregister = asyncio.create_task(
            client.post(
                "/unregister",
                json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
                headers=admin_headers(),
            )
        )
        await asyncio.wait_for(cascade_entered.wait(), timeout=1.0)
        unregister.cancel()
        allow_cascade.set()
        with pytest.raises(asyncio.CancelledError):
            await unregister

        assert await app.state.worker_registry.get_by_addr(WORKER_1) is None
        assert await session_registry.lookup_by_key("old-key") is None
        assert await app.state.group_registry.lookup("old-group") is None

        replay = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
            headers=admin_headers(),
        )
        assert replay.status_code == 200
        assert replay.json()["removed"] is False

    @pytest.mark.asyncio
    async def test_delayed_unregister_cannot_delete_successor_or_its_sessions(
        self, client
    ):
        await register_workers(client, WORKER_1)
        successor_id = "worker-1-epoch-2"
        assert (
            await client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id=successor_id,
                    expected_worker_id=WORKER_ID_1,
                ),
                headers=admin_headers(),
            )
        ).status_code == 200
        assert (
            await client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "new-key", "session_id": "new-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": successor_id,
                    "group_id": "new-group",
                },
                headers=admin_headers(),
            )
        ).status_code == 200

        delayed = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
            headers=admin_headers(),
        )
        routed = await client.post(
            "/route", json={"api_key": "new-key"}, headers=admin_headers()
        )

        assert delayed.status_code == 200
        assert delayed.json()["removed"] is False
        assert delayed.json()["sessions_revoked"] == 0
        assert delayed.json()["groups_revoked"] == 0
        assert routed.status_code == 200
        assert routed.json()["worker_id"] == successor_id
        assert (
            await client._transport.app.state.group_registry.lookup("new-group")
            is not None
        )

    @pytest.mark.asyncio
    async def test_retired_id_cannot_be_reused_at_another_addr(self, client):
        reused_worker_id = "reused-worker-id"
        assert (
            await client.post(
                "/register",
                json=register_payload(WORKER_1, worker_id=reused_worker_id),
                headers=admin_headers(),
            )
        ).status_code == 200
        assert (
            await client.post(
                "/unregister",
                json={"worker_addr": WORKER_1, "worker_id": reused_worker_id},
                headers=admin_headers(),
            )
        ).json()["removed"] is True
        reused = await client.post(
            "/register",
            json=register_payload(WORKER_2, worker_id=reused_worker_id),
            headers=admin_headers(),
        )

        assert reused.status_code == 409
        assert "already been used" in reused.json()["detail"]

    # ----- /route — admin key -----

    @pytest.mark.asyncio
    async def test_route_admin_key_sticky_hitl(self, client):
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json=register_payload(WORKER_2),
            headers=admin_headers(),
        )

        resp1 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp1.status_code == 200
        addr1 = resp1.json()["worker_addr"]

        resp2 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp2.status_code == 200
        addr2 = resp2.json()["worker_addr"]

        assert addr1 == addr2

    @pytest.mark.asyncio
    async def test_route_new_sessions_are_unpinned_and_round_robin(self, client):
        for worker_addr in (WORKER_1, WORKER_2):
            await client.post(
                "/register",
                json=register_payload(worker_addr),
                headers=admin_headers(),
            )

        responses = [
            await client.post(
                "/route",
                json={"new_session": True},
                headers=admin_headers(),
            )
            for _ in range(2)
        ]

        assert [response.status_code for response in responses] == [200, 200]
        assert {response.json()["worker_addr"] for response in responses} == {
            WORKER_1,
            WORKER_2,
        }
        assert await client._transport.app.state.session_registry.count() == 0

    # ----- /route — session key -----

    @pytest.mark.asyncio
    async def test_route_session_key_pinned(self, client):
        # Register worker + session
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "grp-test-2",
            },
            headers=admin_headers(),
        )

        # Route with session key → pinned to WORKER_1
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — unknown key -----

    @pytest.mark.asyncio
    async def test_route_unknown_key_404(self, client):
        resp = await client.post(
            "/route",
            json={"api_key": "unknown-key", "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404
        assert "Unknown API key" in resp.json()["detail"]

    # ----- /route — no healthy workers -----

    @pytest.mark.asyncio
    async def test_route_no_healthy_workers_503(self, client):
        """Admin key but no workers registered → 503."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No registered workers" in resp.json()["detail"]

    # ----- /route — pinned worker unhealthy -----

    @pytest.mark.asyncio
    async def test_route_pinned_worker_unhealthy_still_routes(self, client):
        # Register worker + session
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "grp-test-4",
            },
            headers=admin_headers(),
        )

        # Mark worker unhealthy
        wr: WorkerRegistry = client._transport.app.state.worker_registry  # type: ignore[attr-defined]
        await wr.update_health(WORKER_1, WORKER_ID_1, False)

        # Routing still returns the pinned worker even if unhealthy
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — session_id lookup -----

    @pytest.mark.asyncio
    async def test_route_by_session_id(self, client):
        worker = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        assert worker.status_code == 200
        worker_id = worker.json()["worker_id"]

        registered = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": worker_id,
                "group_id": "grp-test-3",
            },
            headers=admin_headers(),
        )
        assert registered.status_code == 200

        resp = await client.post(
            "/route",
            json={"session_id": "task-0-0"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1
        assert resp.json()["worker_id"] == worker_id

    @pytest.mark.asyncio
    async def test_route_by_session_id_unknown_404(self, client):
        resp = await client.post(
            "/route",
            json={"session_id": "nonexistent"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    # ----- /route — missing both api_key and session_id -----

    @pytest.mark.asyncio
    async def test_route_missing_both_keys_422(self, client):
        resp = await client.post(
            "/route",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /route — no auth -----

    @pytest.mark.asyncio
    async def test_route_no_auth_401(self, client):
        """Route without auth → 401."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
        )
        assert resp.status_code == 401

    # ----- /register_session — no auth -----

    @pytest.mark.asyncio
    async def test_register_session_no_auth_401(self, client):
        """Register session without auth → 401."""
        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "grp-test-5",
            },
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_session_missing_worker_id_is_422_with_no_side_effects(
        self, client
    ):
        await register_workers(client, WORKER_1)

        response = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "missing-key", "session_id": "missing-id"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "missing-group",
            },
            headers=admin_headers(),
        )

        assert response.status_code == 422
        assert await client._transport.app.state.session_registry.count() == 0
        assert (
            await client._transport.app.state.group_registry.lookup("missing-group")
            is None
        )

    @pytest.mark.asyncio
    async def test_register_session_stale_worker_id_is_409_with_no_side_effects(
        self, client
    ):
        await register_workers(client, WORKER_1)
        assert (
            await client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id="worker-1-epoch-2",
                    expected_worker_id=WORKER_ID_1,
                ),
                headers=admin_headers(),
            )
        ).status_code == 200

        response = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "stale-key", "session_id": "stale-id"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "stale-group",
            },
            headers=admin_headers(),
        )

        assert response.status_code == 409
        assert await client._transport.app.state.session_registry.count() == 0
        assert (
            await client._transport.app.state.group_registry.lookup("stale-group")
            is None
        )

    @pytest.mark.asyncio
    async def test_register_session_and_replacement_are_linearized(
        self, client, monkeypatch
    ):
        await register_workers(client, WORKER_1)
        registry = client._transport.app.state.session_registry
        original_register_sessions = registry.register_sessions
        registration_paused = asyncio.Event()
        allow_registration = asyncio.Event()

        async def pause_registration(*args, **kwargs):
            registration_paused.set()
            await allow_registration.wait()
            return await original_register_sessions(*args, **kwargs)

        monkeypatch.setattr(registry, "register_sessions", pause_registration)
        session_task = asyncio.create_task(
            client.post(
                "/register_session",
                json={
                    "sessions": [
                        {"session_api_key": "racing-key", "session_id": "racing-id"}
                    ],
                    "worker_addr": WORKER_1,
                    "worker_id": WORKER_ID_1,
                    "group_id": "racing-group",
                },
                headers=admin_headers(),
            )
        )
        await asyncio.wait_for(registration_paused.wait(), timeout=1.0)
        replacement_task = asyncio.create_task(
            client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id="worker-1-epoch-2",
                    expected_worker_id=WORKER_ID_1,
                ),
                headers=admin_headers(),
            )
        )
        await asyncio.sleep(0)
        assert replacement_task.done() is False

        allow_registration.set()
        session_response, replacement_response = await asyncio.gather(
            session_task, replacement_task
        )

        assert session_response.status_code == 200
        assert replacement_response.status_code == 200
        assert await registry.lookup_by_key("racing-key") is None
        assert (
            await client._transport.app.state.group_registry.lookup("racing-group")
            is None
        )

    # ----- /register_session -----

    @pytest.mark.asyncio
    async def test_register_session(self, client):
        """Register a session, then verify pinned routing works."""
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )

        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": WORKER_ID_1,
                "group_id": "grp-test-6",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify: /route with session key returns pinned worker
        route_resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert route_resp.status_code == 200
        assert route_resp.json()["worker_addr"] == WORKER_1
        assert route_resp.json()["worker_id"] == WORKER_ID_1

    @pytest.mark.asyncio
    async def test_register_session_identical_retry_is_idempotent(self, client):
        await register_workers(client, WORKER_1)
        payload = {
            "sessions": [{"session_api_key": "sess-key-1", "session_id": "task-0-0"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-retry",
        }

        first = await client.post(
            "/register_session", json=payload, headers=admin_headers()
        )
        replay = await client.post(
            "/register_session", json=payload, headers=admin_headers()
        )

        assert first.status_code == 200
        assert replay.status_code == 200
        assert await client._transport.app.state.session_registry.count() == 1

    @pytest.mark.asyncio
    async def test_register_session_conflicting_retry_preserves_original(self, client):
        await register_workers(client, WORKER_1, WORKER_2)
        original = {
            "sessions": [{"session_api_key": "sess-key-1", "session_id": "task-0-0"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-retry",
        }
        conflict = {
            "sessions": [{"session_api_key": "sess-key-2", "session_id": "task-0-1"}],
            "worker_addr": WORKER_2,
            "worker_id": WORKER_ID_2,
            "group_id": "grp-retry",
        }

        assert (
            await client.post(
                "/register_session", json=original, headers=admin_headers()
            )
        ).status_code == 200
        replay = await client.post(
            "/register_session", json=conflict, headers=admin_headers()
        )

        assert replay.status_code == 409
        group = await client._transport.app.state.group_registry.lookup("grp-retry")
        assert group.worker_addr == WORKER_1
        assert group.session_ids == ["task-0-0"]
        assert (
            await client._transport.app.state.session_registry.lookup_by_id("task-0-1")
            is None
        )

    @pytest.mark.asyncio
    async def test_register_session_rejects_api_key_change_on_retry(self, client):
        await register_workers(client, WORKER_1)
        original = {
            "sessions": [{"session_api_key": "sess-key-1", "session_id": "task-0-0"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-retry",
        }
        changed_key = {
            **original,
            "sessions": [{"session_api_key": "sess-key-2", "session_id": "task-0-0"}],
        }

        assert (
            await client.post(
                "/register_session", json=original, headers=admin_headers()
            )
        ).status_code == 200
        replay = await client.post(
            "/register_session", json=changed_key, headers=admin_headers()
        )

        assert replay.status_code == 409
        assert (
            await client._transport.app.state.session_registry.lookup_by_key(
                "sess-key-2"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_register_session_rejects_global_id_collision_across_groups(
        self, client
    ):
        await register_workers(client, WORKER_1, WORKER_2)
        first = {
            "sessions": [{"session_api_key": "sess-key-1", "session_id": "same-id"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-worker-1",
        }
        collision = {
            "sessions": [{"session_api_key": "sess-key-2", "session_id": "same-id"}],
            "worker_addr": WORKER_2,
            "worker_id": WORKER_ID_2,
            "group_id": "grp-worker-2",
        }

        assert (
            await client.post("/register_session", json=first, headers=admin_headers())
        ).status_code == 200
        response = await client.post(
            "/register_session", json=collision, headers=admin_headers()
        )

        assert response.status_code == 409
        assert (
            await client._transport.app.state.session_registry.lookup_by_id("same-id")
            == WORKER_1
        )
        assert (
            await client._transport.app.state.group_registry.lookup("grp-worker-2")
            is None
        )

    @pytest.mark.asyncio
    async def test_register_session_refreshes_key_on_same_worker(self, client):
        await register_workers(client, WORKER_1)
        first = {
            "sessions": [{"session_api_key": "stable-key", "session_id": "old-id"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-old",
        }
        refreshed = {
            "sessions": [{"session_api_key": "stable-key", "session_id": "new-id"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-new",
        }

        assert (
            await client.post("/register_session", json=first, headers=admin_headers())
        ).status_code == 200
        assert (
            await client.post(
                "/register_session", json=refreshed, headers=admin_headers()
            )
        ).status_code == 200

        registry = client._transport.app.state.session_registry
        assert await registry.lookup_by_id("old-id") == WORKER_1
        assert await registry.lookup_by_id("new-id") == WORKER_1
        assert await registry.lookup_by_key("stable-key") == WORKER_1

        # Export cleanup for the old group removes only the old ID, not the
        # refreshed key that now authenticates the new session.
        removed = await client.post(
            "/remove_session",
            json={"group_id": "grp-old"},
            headers=admin_headers(),
        )
        assert removed.status_code == 200
        assert await registry.lookup_by_id("old-id") is None
        assert await registry.lookup_by_id("new-id") == WORKER_1
        assert await registry.lookup_by_key("stable-key") == WORKER_1

    @pytest.mark.asyncio
    async def test_register_session_rejects_key_refresh_across_workers(self, client):
        await register_workers(client, WORKER_1, WORKER_2)
        first = {
            "sessions": [{"session_api_key": "stable-key", "session_id": "old-id"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "grp-old",
        }
        moved = {
            "sessions": [{"session_api_key": "stable-key", "session_id": "new-id"}],
            "worker_addr": WORKER_2,
            "worker_id": WORKER_ID_2,
            "group_id": "grp-new",
        }

        assert (
            await client.post("/register_session", json=first, headers=admin_headers())
        ).status_code == 200
        response = await client.post(
            "/register_session", json=moved, headers=admin_headers()
        )

        assert response.status_code == 409
        registry = client._transport.app.state.session_registry
        assert await registry.lookup_by_id("old-id") == WORKER_1
        assert await registry.lookup_by_id("new-id") is None

    @pytest.mark.asyncio
    async def test_unregister_removes_groups_and_delayed_replay_cannot_revive_worker(
        self, client
    ):
        await register_workers(client, WORKER_1)
        payload = {
            "sessions": [{"session_api_key": "dead-key", "session_id": "dead-id"}],
            "worker_addr": WORKER_1,
            "worker_id": WORKER_ID_1,
            "group_id": "dead-group",
        }
        assert (
            await client.post(
                "/register_session", json=payload, headers=admin_headers()
            )
        ).status_code == 200

        removed = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": WORKER_ID_1},
            headers=admin_headers(),
        )
        replay = await client.post(
            "/register_session", json=payload, headers=admin_headers()
        )
        route = await client.post(
            "/route",
            json={"api_key": "dead-key", "path": "/chat/completions"},
            headers=admin_headers(),
        )

        assert removed.status_code == 200
        assert replay.status_code == 409
        assert route.status_code == 404
        assert (
            await client._transport.app.state.group_registry.lookup("dead-group")
            is None
        )

    @pytest.mark.asyncio
    async def test_old_worker_epoch_cannot_replay_into_reused_address(self, client):
        first_registration = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        old_worker_id = first_registration.json()["worker_id"]
        payload = {
            "sessions": [{"session_api_key": "old-key", "session_id": "old-id"}],
            "worker_addr": WORKER_1,
            "worker_id": old_worker_id,
            "group_id": "old-group",
        }
        assert (
            await client.post(
                "/register_session", json=payload, headers=admin_headers()
            )
        ).status_code == 200
        await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": old_worker_id},
            headers=admin_headers(),
        )
        new_registration = await client.post(
            "/register",
            json=register_payload(
                WORKER_1,
                worker_id="worker-1-epoch-2",
                expected_worker_id=old_worker_id,
            ),
            headers=admin_headers(),
        )
        assert new_registration.json()["worker_id"] != old_worker_id

        replay = await client.post(
            "/register_session", json=payload, headers=admin_headers()
        )

        assert replay.status_code == 409
        assert "epoch mismatch" in replay.text
        assert (
            await client._transport.app.state.session_registry.lookup_by_key("old-key")
            is None
        )

    @pytest.mark.asyncio
    async def test_route_never_splices_stale_session_with_new_worker_epoch(
        self, client, monkeypatch
    ):
        first_registration = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        assert first_registration.status_code == 200
        old_worker_id = first_registration.json()["worker_id"]

        session_registration = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "stale-key", "session_id": "stale-id"}
                ],
                "worker_addr": WORKER_1,
                "worker_id": old_worker_id,
                "group_id": "stale-group",
            },
            headers=admin_headers(),
        )
        assert session_registration.status_code == 200

        session_registry = client._transport.app.state.session_registry
        original_lookup = session_registry.route_by_key
        stale_mapping_read = asyncio.Event()
        allow_route_to_continue = asyncio.Event()

        async def pause_after_stale_mapping_read(session_key: str):
            session_route = await original_lookup(session_key)
            stale_mapping_read.set()
            await allow_route_to_continue.wait()
            return session_route

        monkeypatch.setattr(
            session_registry, "route_by_key", pause_after_stale_mapping_read
        )

        route_task = asyncio.create_task(
            client.post(
                "/route",
                json={"api_key": "stale-key", "path": "/rl/start_session"},
                headers=admin_headers(),
            )
        )
        await asyncio.wait_for(stale_mapping_read.wait(), timeout=1.0)
        replacement_task = asyncio.create_task(
            client.post(
                "/register",
                json=register_payload(
                    WORKER_1,
                    worker_id="worker-1-epoch-2",
                    expected_worker_id=old_worker_id,
                ),
                headers=admin_headers(),
            )
        )
        try:
            await asyncio.sleep(0)
            assert replacement_task.done() is False
        finally:
            allow_route_to_continue.set()

        routed, replacement = await asyncio.gather(route_task, replacement_task)
        assert routed.status_code == 200
        assert routed.json()["worker_id"] == old_worker_id
        assert replacement.status_code == 200
        assert replacement.json()["worker_id"] == "worker-1-epoch-2"

    # ----- /workers -----

    @pytest.mark.asyncio
    async def test_workers_list_admin_key(self, client):
        await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json=register_payload(WORKER_2),
            headers=admin_headers(),
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["workers"]) == 2
        addrs = {w["addr"] for w in data["workers"]}
        assert addrs == {WORKER_1, WORKER_2}
        assert all(w["healthy"] is True for w in data["workers"])
        assert all("worker_id" in w for w in data["workers"])
        assert all(isinstance(w["worker_id"], str) for w in data["workers"])

    @pytest.mark.asyncio
    async def test_workers_list_no_auth_401(self, client):
        resp = await client.get("/workers")
        assert resp.status_code == 401

    # ----- /unregister by worker_id -----

    @pytest.mark.asyncio
    async def test_delete_worker_by_id(self, client):
        """Delete a worker by worker_id instead of worker_addr."""
        # Register
        reg_resp = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        # Delete by worker_id
        resp = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": worker_id},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["removed"] is True

        # Verify worker is gone
        health = (await client.get("/health")).json()
        assert health["workers"] == 0

    @pytest.mark.asyncio
    async def test_delete_worker_by_id_not_found_is_idempotent(self, client):
        """A stale compare-and-delete is an idempotent no-op."""
        resp = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1, "worker_id": "nonexistent-id"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["removed"] is False
        assert resp.json()["sessions_revoked"] == 0
        assert resp.json()["groups_revoked"] == 0

    @pytest.mark.asyncio
    async def test_delete_worker_missing_both_422(self, client):
        """Delete without worker_id or worker_addr → 422."""
        resp = await client.post(
            "/unregister",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /resolve_worker/{worker_id} -----

    @pytest.mark.asyncio
    async def test_resolve_worker_200(self, client):
        """Resolve a registered worker by ID → 200 with worker_id and worker_addr."""
        reg_resp = await client.post(
            "/register",
            json=register_payload(WORKER_1),
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        resp = await client.get(
            f"/resolve_worker/{worker_id}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_id"] == worker_id
        assert data["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_resolve_worker_not_found_404(self, client):
        """Resolve an unknown worker_id → 404."""
        resp = await client.get(
            "/resolve_worker/nonexistent-id",
            headers=admin_headers(),
        )
        assert resp.status_code == 404
