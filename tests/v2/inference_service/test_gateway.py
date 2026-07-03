"""Unit tests for the inference gateway.

Tests gateway endpoints with mocked Router and data proxy workers.
Uses ``unittest.mock.patch`` to mock ``streaming.py`` functions at module level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.gateway.app import create_app
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.gateway.streaming import (
    RouterDestination,
    RouterKeyRejectedError,
    RouterSessionRegistrationError,
    RouterUnreachableError,
    register_session_in_router,
)
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER

# =============================================================================
# Constants & Config
# =============================================================================

ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key-abc123"
WORKER_ADDR = "http://worker-1:18082"
WORKER_ADDR_2 = "http://worker-2:18082"
WORKER_ADDR_3 = "http://worker-3:18082"

MODULE = "areal.v2.inference_service.gateway.app"


@pytest.fixture
def config():
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8081",
        router_timeout=2.0,
        forward_timeout=30.0,
    )


@pytest_asyncio.fixture
async def client(config):
    """Create gateway app and yield an httpx async client."""
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers():
    return {"Authorization": f"Bearer {SESSION_KEY}"}


# =============================================================================
# Health endpoint
# =============================================================================


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


# =============================================================================
# Auth rejection
# =============================================================================


class TestAuthRejection:
    @pytest.mark.asyncio
    async def test_missing_auth_chat_401(self, client):
        """POST /chat/completions without auth → 401."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_session_key_on_admin_endpoint_rejected(self, client):
        """POST /rl/start_session with session key → 403."""
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t1"},
            headers=session_headers(),
        )
        assert resp.status_code == 403


# =============================================================================
# Admin endpoints
# =============================================================================


class TestAdminEndpoints:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_non_streaming(
        self, mock_query_router, mock_forward, client
    ):
        """Admin key → /chat/completions (non-streaming) → response forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        # Simulate data proxy response
        mock_resp = httpx.Response(
            200,
            json={"id": "chatcmpl-1", "choices": [{"message": {"content": "Hi"}}]},
        )
        mock_forward.return_value = mock_resp

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-1"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_sse_stream")
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_streaming(
        self, mock_query_router, mock_forward_sse, client
    ):
        """Admin key → /chat/completions (streaming) → SSE forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        async def _stream():
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_forward_sse.return_value = _stream()

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_creates_and_registers(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """Admin key → /rl/start_session → forwarded, response intercepted, session registered."""
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        session_resp_data = {
            "group_id": "grp-test-1",
            "sessions": [{"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}],
        }
        mock_forward.return_value = httpx.Response(201, json=session_resp_data)

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "task-1", "delivery_mode": "pull"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["group_id"] == "grp-test-1"
        assert data["sessions"] == [
            {"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}
        ]

        # Verify router registration
        mock_register.assert_called_once()
        reg_args = mock_register.call_args
        assert reg_args.args[0] == "http://mock-router:8081"  # router_addr
        assert reg_args.args[1] == [
            {"session_api_key": "sess-key-xyz", "session_id": "task-1-0"}
        ]  # sessions_list
        assert reg_args.args[2] == WORKER_ADDR  # worker_addr
        assert reg_args.kwargs["worker_id"] == "worker-epoch-1"
        assert mock_query_router.call_args.kwargs["new_session"] is True

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_refresh_routes_to_existing_key_owner(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-refresh",
                "sessions": [
                    {"session_id": "task-refresh-1", "session_api_key": "old-key"}
                ],
            },
        )

        response = await client.post(
            "/rl/start_session",
            json={
                "task_id": "task-refresh",
                "api_key": "old-key",
                "delivery_mode": "pull",
            },
            headers=admin_headers(),
        )

        assert response.status_code == 201
        assert mock_query_router.call_args.args[1] == "old-key"
        assert mock_query_router.call_args.kwargs["new_session"] is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_rejects_legacy_string_router_result(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-legacy-route",
                "sessions": [
                    {"session_id": "task-legacy-0", "session_api_key": "key-legacy"}
                ],
            },
        )

        response = await client.post(
            "/rl/start_session",
            json={"task_id": "task-legacy", "delivery_mode": "pull"},
            headers=admin_headers(),
        )

        assert response.status_code == 502
        assert "worker incarnation" in response.json()["error"]
        mock_forward.assert_not_awaited()
        mock_register.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_router_registration_fails(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """If router registration fails after session creation → 502."""
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-2",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )
        mock_register.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t", "delivery_mode": "pull"},
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "registration failed" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_admin_export_trajectories(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            200, json={"traj": {"interactions": []}}
        )

        resp = await client.post(
            "/export_trajectories",
            json={
                "request_id": "admin-export",
                "session_ids": ["task-1-0"],
                "group_id": "grp-test",
                "discount": 1.0,
                "style": "sft",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_query_router.assert_called_once()
        assert mock_query_router.call_args.kwargs["session_id"] == "task-1-0"
        mock_revoke.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_export_response_replays_after_router_mapping_is_removed(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        response_body = {"traj": {"rewards": [1.0]}}
        mock_forward.return_value = httpx.Response(200, json=response_body)
        mock_revoke.return_value = True
        request_body = {
            "request_id": "export-replay-1",
            "session_ids": ["task-1-0"],
            "group_id": "grp-test",
        }

        first = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )
        replay = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )

        assert first.status_code == replay.status_code == 200
        assert first.json() == replay.json() == response_body
        mock_query_router.assert_awaited_once()
        mock_forward.assert_awaited_once()
        mock_revoke.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_export_lost_worker_response_retries_recalled_worker(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR,
            worker_id="worker-epoch-1",
        )
        mock_forward.side_effect = [
            httpx.ReadError("response lost after export committed"),
            httpx.Response(200, json={"traj": {"rewards": [1.0]}}),
        ]
        request_body = {
            "request_id": "export-lost-worker-response",
            "session_ids": ["task-1-0"],
        }

        first = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )
        retry = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )

        assert first.status_code == 502
        assert retry.status_code == 200
        mock_query_router.assert_awaited_once()
        assert mock_forward.await_count == 2
        assert {
            call.args[2][WORKER_ID_HEADER] for call in mock_forward.await_args_list
        } == {"worker-epoch-1"}
        mock_revoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_unexpected_router_error_releases_pending_owner(
        self, mock_query_router, client
    ):
        mock_query_router.side_effect = [
            ValueError("malformed router response"),
            RouterKeyRejectedError("session not found", 404),
        ]
        request_body = {
            "request_id": "export-router-error",
            "session_ids": ["task-1-0"],
        }

        first = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )
        retry = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )

        assert first.status_code == 502
        assert retry.status_code == 401
        assert mock_query_router.await_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_export_router_cleanup_failure_retries_without_losing_result(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            200, json={"traj": {"rewards": [1.0]}}
        )
        mock_revoke.side_effect = [False, True]
        request_body = {
            "request_id": "export-cleanup-retry",
            "session_ids": ["task-1-0"],
            "group_id": "grp-test",
        }

        first = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )
        retry = await client.post(
            "/export_trajectories", json=request_body, headers=admin_headers()
        )

        assert first.status_code == 200
        assert retry.status_code == 200
        assert mock_query_router.await_count == 1
        assert mock_forward.await_count == 1
        assert mock_revoke.await_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_online_export_without_group_id_skips_revoke(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            200, json={"traj": {"interactions": []}}
        )

        resp = await client.post(
            "/export_trajectories",
            json={
                "request_id": "online-export",
                "session_ids": ["__hitl__"],
                "trajectory_id": 0,
                "discount": 1.0,
                "style": "individual",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_revoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_trajectories_missing_session_id(self, client):
        """Admin key → /export_trajectories without session_id → 400."""
        resp = await client.post(
            "/export_trajectories",
            json={"request_id": "missing-session-export", "discount": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 400
        assert "session_ids" in resp.json()["error"]


# =============================================================================
# Session endpoints
# =============================================================================


class TestSessionEndpoints:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_chat_completions(
        self, mock_query_router, mock_forward, client
    ):
        """Session key → /chat/completions → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"id": "chatcmpl-2", "choices": [{"message": {"content": "OK"}}]},
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "chatcmpl-2"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward_finish(
        self, mock_query_router, mock_forward, client
    ):
        """Session key → /rl/set_reward (finish=True) → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"message": "success", "interaction_count": 5, "finished": True},
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 0.0, "finish": True},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward(self, mock_query_router, mock_forward, client):
        """Session key → /rl/set_reward → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"message": "success"})

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_returns_ready_transition_without_router_notification(
        self, mock_query_router, mock_forward, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": True,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_duplicate_ready_transition_is_forwarded_as_is(
        self, mock_query_router, mock_forward, client
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": False,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is False


# =============================================================================
# Broadcast endpoints
# =============================================================================


class TestBroadcast:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_generation_targets_worker(
        self, mock_resolve, mock_forward, client
    ):
        """Admin key → /pause_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"message": "paused"})

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["ok"] is True
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_continue_generation_targets_worker(
        self, mock_resolve, mock_forward, client
    ):
        """Admin key → /continue_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"message": "continued"})

        resp = await client.post(
            "/continue_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_worker_broadcast_failure(
        self, mock_resolve, mock_forward, client
    ):
        """Worker returns error → response shows ok=False for that worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_forward.side_effect = httpx.ConnectError("Connection refused")

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["ok"] is False


class TestTargetedControlWorkerIdentity:
    @staticmethod
    async def _request_through_replaced_worker(
        config: GatewayConfig,
        *,
        method: str,
        gateway_path: str,
        expected_worker_path: str,
        worker_status: int,
    ) -> tuple[httpx.Response, list[httpx.Request]]:
        """Resolve E1 to A, then make the process at A behave as successor E2."""

        worker_requests: list[httpx.Request] = []

        async def _router_then_successor(request: httpx.Request) -> httpx.Response:
            if request.url.host == "mock-router":
                assert request.url.path == "/resolve_worker/epoch-e1"
                return httpx.Response(200, json={"worker_addr": WORKER_ADDR})

            assert request.url.host == "worker-1"
            assert request.url.path == expected_worker_path
            worker_requests.append(request)
            supplied_worker_id = request.headers.get(WORKER_ID_HEADER)
            if supplied_worker_id != "epoch-e2":
                return httpx.Response(
                    worker_status,
                    json={"detail": "Data Proxy incarnation mismatch"},
                )
            return httpx.Response(200, json={"message": "successor mutated"})

        app = create_app(config)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_router_then_successor)
        ) as upstream_client:
            app.state.http_client = upstream_client
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway",
            ) as gateway_client:
                response = await gateway_client.request(
                    method,
                    gateway_path,
                    json={"version": 7} if method == "POST" else None,
                    headers=admin_headers(),
                )

        return response, worker_requests

    @pytest.mark.asyncio
    async def test_pause_generation_forwards_resolved_identity_and_preserves_409(
        self, config
    ):
        response, worker_requests = await self._request_through_replaced_worker(
            config,
            method="POST",
            gateway_path="/pause_generation/epoch-e1",
            expected_worker_path="/pause_generation",
            worker_status=409,
        )

        assert len(worker_requests) == 1
        assert worker_requests[0].headers[WORKER_ID_HEADER] == "epoch-e1"
        assert response.status_code == 409
        assert response.json() == {"detail": "Data Proxy incarnation mismatch"}

    @pytest.mark.asyncio
    async def test_set_version_forwards_resolved_identity_and_preserves_409(
        self, config
    ):
        response, worker_requests = await self._request_through_replaced_worker(
            config,
            method="POST",
            gateway_path="/set_version/epoch-e1",
            expected_worker_path="/set_version",
            worker_status=409,
        )

        assert len(worker_requests) == 1
        assert worker_requests[0].headers[WORKER_ID_HEADER] == "epoch-e1"
        assert response.status_code == 409
        assert response.json() == {"detail": "Data Proxy incarnation mismatch"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "gateway_path", "worker_path"),
        [
            (
                "POST",
                "/continue_generation/epoch-e1",
                "/continue_generation",
            ),
            (
                "POST",
                "/release_memory_occupation/epoch-e1",
                "/release_memory_occupation",
            ),
            (
                "POST",
                "/resume_memory_occupation/epoch-e1",
                "/resume_memory_occupation",
            ),
            ("GET", "/get_version/epoch-e1", "/get_version"),
        ],
    )
    async def test_targeted_controls_forward_path_worker_identity(
        self, config, method, gateway_path, worker_path
    ):
        response, worker_requests = await self._request_through_replaced_worker(
            config,
            method=method,
            gateway_path=gateway_path,
            expected_worker_path=worker_path,
            worker_status=409,
        )

        assert len(worker_requests) == 1
        assert worker_requests[0].headers[WORKER_ID_HEADER] == "epoch-e1"
        assert response.status_code == 409
        assert response.json() == {"detail": "Data Proxy incarnation mismatch"}


# =============================================================================
# Router errors
# =============================================================================


class TestRouterErrors:
    @pytest.mark.asyncio
    async def test_register_session_preserves_router_409_details(self):
        async def _stale_registration(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={"detail": "worker incarnation worker-epoch-1 is stale"},
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_stale_registration)
        ) as router_client:
            with pytest.raises(RouterSessionRegistrationError) as exc_info:
                await register_session_in_router(
                    "http://router",
                    [{"session_id": "session-1", "session_api_key": "key-1"}],
                    WORKER_ADDR,
                    2.0,
                    admin_api_key=ADMIN_KEY,
                    group_id="group-1",
                    worker_id="worker-epoch-1",
                    client=router_client,
                )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "worker incarnation worker-epoch-1 is stale"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route_payload", [{}, {"worker_id": ""}])
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_route_without_worker_id_fails_closed_before_forwarding(
        self, mock_forward, config, route_payload
    ):
        mock_forward.return_value = httpx.Response(200, json={"id": "unexpected"})

        async def _route(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"worker_addr": WORKER_ADDR, **route_payload},
            )

        app = create_app(config)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_route),
            base_url=config.router_addr,
        ) as router_client:
            app.state.http_client = router_client
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway",
            ) as gateway_client:
                response = await gateway_client.post(
                    "/chat/completions",
                    json={
                        "model": "sglang",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=admin_headers(),
                )

        assert response.status_code == 502
        assert "worker_id" in response.json()["error"]
        mock_forward.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_router_unreachable_502(self, mock_query_router, client):
        """Router down → 502 JSON error."""
        mock_query_router.side_effect = RouterUnreachableError(
            "Router unreachable: connect timeout"
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "Router unreachable" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_no_healthy_workers_503(self, mock_query_router, client):
        """Router returns 503 (no healthy workers) → gateway returns 503."""
        mock_query_router.side_effect = RouterKeyRejectedError(
            "No healthy workers", 503
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No healthy workers" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_router_unreachable_on_pause(self, mock_resolve, client):
        """Router unreachable when resolving worker for pause → 502."""
        mock_resolve.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_not_found_for_export(self, mock_query_router, client):
        """Session not found for export_trajectories → 401."""
        mock_query_router.side_effect = RouterKeyRejectedError("Session not found", 404)

        resp = await client.post(
            "/export_trajectories",
            json={
                "request_id": "missing-route-export",
                "session_ids": ["nonexistent"],
                "discount": 1.0,
                "style": "sft",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 401
        assert "Session not found" in resp.json()["error"]


# =============================================================================
# Capacity enforcement — /rl/start_session with capacity checks
# =============================================================================


class TestStartSessionCapacity:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_full_flow(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        """A controller lease admits one callback session end to end."""
        mock_query_router.return_value = RouterDestination(
            worker_addr=WORKER_ADDR, worker_id="worker-epoch-1"
        )
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-3",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )

        grant = await client.post(
            "/internal/online_leases",
            json={"lease_id": "lease-test", "expected_version": 0},
            headers=admin_headers(),
        )
        assert grant.status_code == 201

        resp = await client.post(
            "/rl/start_session",
            json={
                "task_id": "t",
                "delivery_mode": "callback",
                "request_id": "request-full-flow",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 201

        # Verify call order: route → forward → register
        mock_query_router.assert_called_once()
        mock_forward.assert_called_once()
        mock_register.assert_called_once()
