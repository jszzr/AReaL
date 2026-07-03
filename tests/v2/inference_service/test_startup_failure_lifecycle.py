from __future__ import annotations

import asyncio
import concurrent.futures
import socket
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.rpc.guard.app import GuardState, create_app
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.http import (
    HttpxAsyncClientCleanup,
    close_httpx_client_from_sync,
    close_httpx_client_on_owner_loop,
)
from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.inference_service.controller.controller import RolloutControllerV2


def _controller() -> RolloutControllerV2:
    return RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-key",
            setup_timeout=1.0,
            workers_ready_timeout=1.0,
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )


@pytest.mark.parametrize("returncode", [None, -3, 0])
def test_guard_reports_forked_child_status(returncode: int | None) -> None:
    state = GuardState()
    child = MagicMock()
    child.poll.return_value = returncode
    state.forked_children_map[("inf-server", 0)] = child

    response = (
        create_app(state)
        .test_client()
        .get(
            "/forked_worker_status",
            query_string={"role": "inf-server", "worker_index": 0},
        )
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "role": "inf-server",
        "worker_index": 0,
        "running": returncode is None,
        "returncode": returncode,
    }


@pytest.mark.parametrize("returncode", [-3, 0])
@pytest.mark.asyncio
async def test_health_wait_fails_fast_when_guard_reports_child_exit(
    returncode: int,
) -> None:
    controller = _controller()
    client = AsyncMock()
    exited = MagicMock(status_code=200)
    exited.json.return_value = {"running": False, "returncode": returncode}

    async def get(url: str, **_kwargs):
        if url == "http://guard/forked_worker_status":
            return exited
        raise httpx.ConnectError("server is not listening")

    client.get.side_effect = get

    try:
        with (
            patch.object(controller, "_get_async_client", return_value=client),
            pytest.raises(
                RuntimeError,
                match=rf"InfServer-0.*inf-server/0.*exit code {returncode}",
            ),
        ):
            await controller._async_wait_for_service(
                "http://inference/health",
                "InfServer-0",
                timeout=1.0,
                process_owners=(("http://guard", "inf-server", 0),),
            )
    finally:
        controller.destroy()


@pytest.mark.asyncio
async def test_health_wait_observes_non_head_rank_exit() -> None:
    controller = _controller()
    client = AsyncMock()
    running = MagicMock(status_code=200)
    running.json.return_value = {"running": True, "returncode": None}
    exited = MagicMock(status_code=200)
    exited.json.return_value = {"running": False, "returncode": 0}
    healthy = MagicMock(status_code=200)

    async def get(url: str, *, params=None, **_kwargs):
        if url == "http://guard-0/forked_worker_status":
            assert params == {"role": "inf-server", "worker_index": 0}
            return running
        if url == "http://guard-1/forked_worker_status":
            assert params == {"role": "inf-server", "worker_index": 1}
            return exited
        assert url == "http://head-inference/health"
        return healthy

    client.get.side_effect = get

    try:
        with (
            patch.object(controller, "_get_async_client", return_value=client),
            pytest.raises(RuntimeError, match=r"InfServer-0.*inf-server/1.*code 0"),
        ):
            await controller._async_wait_for_service(
                "http://head-inference/health",
                "InfServer-0",
                timeout=1.0,
                process_owners=(
                    ("http://guard-0", "inf-server", 0),
                    ("http://guard-1", "inf-server", 1),
                ),
            )
    finally:
        controller.destroy()


@pytest.mark.asyncio
async def test_fork_ownership_is_published_after_ack_before_health_wait() -> None:
    controller = _controller()
    client = AsyncMock()
    allocated = MagicMock()
    allocated.json.return_value = {"host": "127.0.0.1", "ports": [18080]}
    forked = MagicMock()
    owner = ("http://guard", "router", 0)

    async def post(url: str, **_kwargs):
        if url.endswith("/alloc_ports"):
            return allocated
        assert url.endswith("/fork")
        assert owner not in controller._forked_services
        return forked

    async def wait_for_health(*_args, **_kwargs) -> None:
        assert owner in controller._forked_services

    client.post.side_effect = post

    try:
        with (
            patch.object(controller, "_get_async_client", return_value=client),
            patch.object(
                controller,
                "_async_wait_for_service",
                side_effect=wait_for_health,
            ),
        ):
            await controller._async_fork_on_guard(
                guard_addr="http://guard",
                role="router",
                worker_index=0,
                raw_cmd=["python", "-m", "router"],
            )
    finally:
        with patch.object(controller, "_kill_forked_service"):
            controller.destroy()


@pytest.mark.asyncio
async def test_failed_fork_response_does_not_publish_ownership() -> None:
    controller = _controller()
    client = AsyncMock()
    allocated = MagicMock()
    allocated.json.return_value = {"host": "127.0.0.1", "ports": [18080]}
    rejected = MagicMock()
    rejected.raise_for_status.side_effect = httpx.HTTPStatusError(
        "fork rejected",
        request=httpx.Request("POST", "http://guard/fork"),
        response=httpx.Response(500),
    )
    client.post.side_effect = [allocated, rejected]

    try:
        with (
            patch.object(controller, "_get_async_client", return_value=client),
            pytest.raises(httpx.HTTPStatusError, match="fork rejected"),
        ):
            await controller._async_fork_on_guard(
                guard_addr="http://guard",
                role="router",
                worker_index=0,
                raw_cmd=["python", "-m", "router"],
            )
        assert controller._forked_services == []
    finally:
        controller.destroy()


def test_rollout_initialization_failure_rolls_back_in_reverse_order() -> None:
    controller = _controller()
    primary = RuntimeError("inference server exited during startup")
    events: list[str] = []

    def fail_after_partial_startup() -> None:
        controller._service_roles.extend(["rollout-inf", "rollout-extra"])
        controller._forked_services.extend(
            [
                ("http://guard", "inf-server", 0),
                ("http://guard", "router", 0),
            ]
        )
        raise primary

    def kill(_guard: str, role: str, _index: int) -> None:
        events.append(f"kill:{role}")

    def delete(*, role: str) -> None:
        events.append(f"delete:{role}")

    def fail_callback_cleanup() -> None:
        events.append("stop:callback")
        raise OSError("callback cleanup failed")

    controller.scheduler.delete_workers.side_effect = delete

    with (
        patch.object(controller, "_bg_initialize", fail_after_partial_startup),
        patch.object(
            controller,
            "_stop_online_callback_server",
            side_effect=fail_callback_cleanup,
        ),
        patch.object(controller, "_kill_forked_service", side_effect=kill),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controller._guarded_bg_initialize()

    assert exc_info.value is primary
    assert events == [
        "stop:callback",
        "kill:router",
        "kill:inf-server",
        "delete:rollout-extra",
        "delete:rollout-inf",
    ]
    assert controller._forked_services == []
    assert controller._service_roles == []
    assert any(
        "callback cleanup failed" in note for note in getattr(primary, "__notes__", [])
    )

    with pytest.raises(RuntimeError) as later_exc_info:
        _ = controller.inference_worker_urls
    assert later_exc_info.value is primary


def test_destroy_retains_fork_owner_when_guard_rejects_kill() -> None:
    controller = _controller()
    owner = ("http://guard", "inf-server", 0)
    controller._forked_services.append(owner)
    rejected = httpx.Response(
        500,
        text="kill failed",
        request=httpx.Request("POST", "http://guard/kill_forked_worker"),
    )

    with (
        patch.object(controller._sync_client, "post", return_value=rejected),
        pytest.raises(httpx.HTTPStatusError),
    ):
        controller.destroy()

    assert controller._forked_services == [owner]
    assert controller._destroyed is False


def test_guard_deletion_does_not_mask_unconfirmed_direct_kill() -> None:
    controller = _controller()
    owner = ("http://guard", "inf-server", 0)
    controller._forked_services.append(owner)
    controller._service_roles.append("rollout-inf")

    kill = MagicMock(side_effect=[OSError("direct kill failed"), None])
    with (
        patch.object(controller, "_kill_forked_service", kill),
        pytest.raises(OSError, match="direct kill failed"),
    ):
        controller.destroy()

    controller.scheduler.delete_workers.assert_called_once_with(role="rollout-inf")
    assert controller._forked_services == [owner]
    assert controller._service_roles == []
    assert controller._destroyed is False

    with patch.object(controller, "_kill_forked_service", kill):
        controller.destroy()

    assert kill.call_count == 2
    assert controller.scheduler.delete_workers.call_count == 1
    assert controller._forked_services == []
    assert controller._destroyed is True


def test_failed_direct_kill_and_guard_deletion_retain_owners_for_retry() -> None:
    controller = _controller()
    owner = ("http://guard", "inf-server", 0)
    controller._forked_services.append(owner)
    controller._service_roles.append("rollout-inf")
    kill = MagicMock(side_effect=[OSError("direct kill failed"), None])
    controller.scheduler.delete_workers.side_effect = [
        OSError("guard delete failed"),
        None,
    ]

    with (
        patch.object(controller, "_kill_forked_service", kill),
        pytest.raises(OSError, match="direct kill failed"),
    ):
        controller.destroy()

    assert controller._forked_services == [owner]
    assert controller._service_roles == ["rollout-inf"]
    assert controller._destroyed is False

    with patch.object(controller, "_kill_forked_service", kill):
        controller.destroy()

    assert kill.call_count == 2
    assert controller.scheduler.delete_workers.call_count == 2
    assert controller._forked_services == []
    assert controller._service_roles == []
    assert controller._destroyed is True


def test_guard_fallback_does_not_mask_direct_kill_keyboard_interrupt() -> None:
    controller = _controller()
    owner = ("http://guard", "inf-server", 0)
    primary = KeyboardInterrupt("direct kill interrupted")
    controller._forked_services.append(owner)
    controller._service_roles.append("rollout-inf")

    kill = MagicMock(side_effect=[primary, None])
    with (
        patch.object(controller, "_kill_forked_service", kill),
        pytest.raises(KeyboardInterrupt) as exc_info,
    ):
        controller.destroy()

    assert exc_info.value is primary
    assert controller._forked_services == [owner]
    assert controller._service_roles == []

    with patch.object(controller, "_kill_forked_service", kill):
        controller.destroy()

    assert controller._forked_services == []


def test_terminal_cleanup_reopens_closed_sync_client_for_late_owner() -> None:
    controller = _controller()
    owner = ("http://guard", "late-inf-server", 0)
    observed_requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        observed_requests.append(request)
        return httpx.Response(200, json={"status": "success"})

    controller._sync_client.close()
    controller._forked_services.append(owner)
    real_httpx_client = httpx.Client

    def make_sync_client(*_args, **_kwargs) -> httpx.Client:
        return real_httpx_client(transport=httpx.MockTransport(handle))

    with patch(
        "areal.v2.inference_service.controller.controller.httpx.Client",
        side_effect=make_sync_client,
    ):
        controller.destroy()

    assert len(observed_requests) == 1
    assert observed_requests[0].url == httpx.URL("http://guard/kill_forked_worker")
    assert controller._forked_services == []
    assert controller._destroyed is True


def test_destroy_rejects_open_async_client_after_owner_loop_closed() -> None:
    controller = _controller()
    owner_loop = MagicMock()
    owner_loop.is_closed.return_value = True
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock()
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    controller._async_client = client
    controller._async_client_loop = owner_loop
    controller._async_client_cleanup = cleanup

    with pytest.raises(RuntimeError, match="owner event loop closed"):
        controller.destroy()

    client.aclose.assert_not_awaited()
    assert controller._async_client is client
    assert controller._async_client_loop is owner_loop
    assert controller._async_client_cleanup is cleanup
    assert controller._destroyed is False


def test_async_client_closes_on_owner_loop_shutdown_before_destroy() -> None:
    controller = _controller()
    client = MagicMock()
    client.is_closed = False
    close_loops: list[asyncio.AbstractEventLoop] = []

    async def close_client() -> None:
        close_loops.append(asyncio.get_running_loop())
        client.is_closed = True

    client.aclose = AsyncMock(side_effect=close_client)
    loop = asyncio.new_event_loop()
    try:
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            assert loop.run_until_complete(controller._get_async_client()) is client
    finally:
        loop.close()

    client.aclose.assert_awaited_once_with()
    assert close_loops == [loop]
    assert client.is_closed is True
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None

    controller.destroy()

    client.aclose.assert_awaited_once_with()
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None
    assert controller._destroyed is True


def test_sequential_run_async_task_clients_close_on_their_own_loops() -> None:
    controller = _controller()
    clients: list[MagicMock] = []
    close_loops: list[asyncio.AbstractEventLoop] = []

    for _ in range(2):
        client = MagicMock()
        client.is_closed = False

        async def close_client(client=client) -> None:
            close_loops.append(asyncio.get_running_loop())
            client.is_closed = True

        client.aclose = AsyncMock(side_effect=close_client)
        clients.append(client)

    with patch(
        "areal.v2.inference_service.controller.controller.create_httpx_client",
        side_effect=clients,
    ):
        assert run_async_task(controller._get_async_client) is clients[0]
        assert controller._async_client is None
        assert run_async_task(controller._get_async_client) is clients[1]

    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None
    assert close_loops[0] is not close_loops[1]
    for client in clients:
        client.aclose.assert_awaited_once_with()


def test_run_async_task_closes_real_keepalive_transport_before_loop_exit() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    peer_saw_eof = threading.Event()
    server_done = threading.Event()
    server_errors: list[BaseException] = []

    def serve_one_keepalive_response() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2.0)
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    chunk = conn.recv(4096)
                    if not chunk:
                        raise AssertionError("client closed before request headers")
                    request.extend(chunk)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 2\r\n"
                    b"Connection: keep-alive\r\n\r\nok"
                )
                if conn.recv(1) == b"":
                    peer_saw_eof.set()
        except BaseException as exc:
            server_errors.append(exc)
        finally:
            server_done.set()

    thread = threading.Thread(target=serve_one_keepalive_response, daemon=True)
    thread.start()
    controller = _controller()
    old_policy = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    try:

        async def make_request() -> None:
            client = await controller._get_async_client()
            response = await client.get(f"http://127.0.0.1:{listener.getsockname()[1]}")
            assert response.status_code == 200
            assert response.content == b"ok"

        run_async_task(make_request)
        assert peer_saw_eof.wait(timeout=2.0)
        assert server_done.wait(timeout=2.0)
        assert server_errors == []
        assert controller._async_client is None
        assert controller._async_client_loop is None
        assert controller._async_client_cleanup is None
    finally:
        asyncio.set_event_loop_policy(old_policy)
        listener.close()
        thread.join(timeout=2.0)
        controller.destroy()


def test_destroy_retains_async_client_when_live_loop_close_fails() -> None:
    controller = _controller()
    owner_loop = MagicMock()
    primary = RuntimeError("async transport close failed")
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock(side_effect=primary)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    controller._async_client = client
    controller._async_client_loop = owner_loop
    controller._async_client_cleanup = cleanup

    with (
        patch(
            "areal.v2.inference_service.controller.controller."
            "close_httpx_client_from_sync",
            side_effect=primary,
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controller.destroy()

    assert exc_info.value is primary
    assert controller._async_client is client
    assert controller._async_client_loop is owner_loop
    assert controller._async_client_cleanup is cleanup
    assert controller._destroyed is False


def test_transport_close_failure_is_never_mistaken_for_httpx_closed() -> None:
    primary = RuntimeError("transport resource is still open")

    class FailingCloseTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.resource_open = True
            self.close_calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            self.close_calls += 1
            raise primary

    controller = _controller()
    transport = FailingCloseTransport()
    client = httpx.AsyncClient(transport=transport)
    loop = asyncio.new_event_loop()
    try:
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            assert loop.run_until_complete(controller._get_async_client()) is client
            cleanup = controller._async_client_cleanup
            assert cleanup is not None
    finally:
        # The registered callback runs transport cleanup on this owner loop.
        # Its failure is logged by the generic loop-cleanup registry.
        loop.close()

    assert client.is_closed is True
    assert transport.resource_open is True
    assert transport.close_calls == 1
    assert cleanup.succeeded is False
    assert cleanup.error is primary
    assert controller._async_client is client
    assert controller._async_client_cleanup is cleanup

    for _ in range(2):
        with pytest.raises(RuntimeError) as exc_info:
            controller.destroy()
        assert exc_info.value is primary
        assert controller._async_client is client
        assert controller._async_client_cleanup is cleanup
        assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_async_client_close_propagates_failure_on_owner_loop() -> None:
    owner_loop = asyncio.get_running_loop()
    primary = RuntimeError("async transport close failed")
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock(side_effect=primary)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)

    with pytest.raises(RuntimeError) as exc_info:
        await close_httpx_client_on_owner_loop(cleanup)

    assert exc_info.value is primary
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_async_client_close_is_single_flight() -> None:
    owner_loop = asyncio.get_running_loop()
    primary = RuntimeError("single transport close failed")
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    client = MagicMock()

    async def close_client() -> None:
        close_started.set()
        await release_close.wait()
        raise primary

    client.aclose = AsyncMock(side_effect=close_client)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    first = asyncio.create_task(cleanup.close())
    await close_started.wait()
    second = asyncio.create_task(cleanup.close())
    await asyncio.sleep(0)
    release_close.set()

    results = await asyncio.gather(first, second, return_exceptions=True)

    assert results == [primary, primary]
    client.aclose.assert_awaited_once_with()
    assert cleanup.succeeded is False
    assert cleanup.error is primary


@pytest.mark.asyncio
async def test_cancelled_close_follower_does_not_cancel_shared_completion() -> None:
    owner_loop = asyncio.get_running_loop()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    client = MagicMock()

    async def close_client() -> None:
        close_started.set()
        await release_close.wait()

    client.aclose = AsyncMock(side_effect=close_client)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    leader = asyncio.create_task(cleanup.close())
    await close_started.wait()
    follower = asyncio.create_task(cleanup.close())
    await asyncio.sleep(0)

    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    release_close.set()
    await leader

    client.aclose.assert_awaited_once_with()
    assert cleanup.succeeded is True
    assert cleanup.error is None


def test_async_client_close_runs_on_live_owner_loop() -> None:
    owner_loop = asyncio.new_event_loop()
    owner_started = threading.Event()
    close_loops: list[asyncio.AbstractEventLoop] = []
    thread_errors: list[BaseException] = []

    def run_owner_loop() -> None:
        try:
            asyncio.set_event_loop(owner_loop)
            owner_started.set()
            owner_loop.run_forever()
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            owner_loop.close()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert owner_started.wait(timeout=1.0)

    client = MagicMock()
    client.is_closed = False

    async def close_client() -> None:
        close_loops.append(asyncio.get_running_loop())
        client.is_closed = True

    client.aclose = AsyncMock(side_effect=close_client)
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)
    try:
        asyncio.run(close_httpx_client_on_owner_loop(cleanup))
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1.0)

    assert not owner_thread.is_alive()
    assert thread_errors == []
    assert close_loops == [owner_loop]
    client.aclose.assert_awaited_once_with()


def test_get_async_client_rejects_concurrent_foreign_event_loop() -> None:
    controller = _controller()
    owner_loop = asyncio.new_event_loop()
    owner_started = threading.Event()
    owner_errors: list[BaseException] = []
    close_loops: list[asyncio.AbstractEventLoop] = []
    client = MagicMock()

    async def close_client() -> None:
        close_loops.append(asyncio.get_running_loop())

    client.aclose = AsyncMock(side_effect=close_client)

    def run_owner_loop() -> None:
        try:
            asyncio.set_event_loop(owner_loop)
            owner_started.set()
            owner_loop.run_forever()
        except BaseException as exc:
            owner_errors.append(exc)
        finally:
            owner_loop.close()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert owner_started.wait(timeout=1.0)
    try:
        with patch(
            "areal.v2.inference_service.controller.controller.create_httpx_client",
            return_value=client,
        ):
            owner_client = asyncio.run_coroutine_threadsafe(
                controller._get_async_client(), owner_loop
            ).result(timeout=1.0)
            assert owner_client is client

            with pytest.raises(RuntimeError, match="active on another event loop"):
                asyncio.run(controller._get_async_client())

            client.aclose.assert_not_awaited()
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=1.0)

    assert not owner_thread.is_alive()
    assert owner_errors == []
    assert close_loops == [owner_loop]
    client.aclose.assert_awaited_once_with()
    assert controller._async_client is None
    assert controller._async_client_loop is None
    assert controller._async_client_cleanup is None


@pytest.mark.asyncio
async def test_sync_async_client_close_rejects_owner_loop_reentrancy() -> None:
    owner_loop = asyncio.get_running_loop()
    client = MagicMock()
    client.is_closed = False
    client.aclose = AsyncMock()
    cleanup = HttpxAsyncClientCleanup(client, owner_loop)

    with pytest.raises(RuntimeError, match="from its owner event loop"):
        close_httpx_client_from_sync(cleanup)

    client.aclose.assert_not_awaited()


def test_initialization_thread_reaps_owner_published_after_concurrent_destroy() -> None:
    controller = _controller()
    initialization_started = threading.Event()
    release_late_publish = threading.Event()
    killed_owners: list[tuple[str, str, int]] = []
    thread_errors: list[BaseException] = []
    owner = ("http://guard", "late-inf-server", 0)

    def publish_after_destroy() -> None:
        initialization_started.set()
        assert release_late_publish.wait(timeout=1.0)
        controller._forked_services.append(owner)

    def run_initialization() -> None:
        try:
            controller._guarded_bg_initialize()
        except BaseException as exc:
            thread_errors.append(exc)

    with (
        patch.object(controller, "_bg_initialize", publish_after_destroy),
        patch.object(
            controller,
            "_kill_forked_service",
            side_effect=lambda *args: killed_owners.append(args),
        ),
    ):
        init_thread = threading.Thread(target=run_initialization)
        init_thread.start()
        assert initialization_started.wait(timeout=1.0)

        controller.destroy()
        release_late_publish.set()
        init_thread.join(timeout=1.0)

    assert not init_thread.is_alive()
    assert thread_errors == []
    assert killed_owners == [owner]
    assert controller._forked_services == []
    assert controller._destroyed is True


def test_workers_ready_timeout_requests_shutdown_and_reaps_late_owner() -> None:
    controller = _controller()
    controller.config.workers_ready_timeout = 0.01
    initialization_started = threading.Event()
    release_late_publish = threading.Event()
    owner = ("http://guard", "late-inf-server", 0)
    killed_owners: list[tuple[str, str, int]] = []
    deleted_roles: list[str] = []
    terminal_cleanup_done = threading.Event()

    def publish_after_timeout(*_args, **_kwargs) -> None:
        initialization_started.set()
        assert release_late_publish.wait(timeout=1.0)
        controller._forked_services.append(owner)
        controller._service_roles.append("rollout-inf")

    def delete_role(*, role: str) -> None:
        deleted_roles.append(role)

    guarded_initialize = controller._guarded_bg_initialize

    def observe_terminal_cleanup(*args, **kwargs) -> None:
        try:
            guarded_initialize(*args, **kwargs)
        finally:
            terminal_cleanup_done.set()

    controller.scheduler.delete_workers.side_effect = delete_role

    with (
        patch.object(controller, "_bg_initialize", publish_after_timeout),
        patch.object(
            controller,
            "_guarded_bg_initialize",
            side_effect=observe_terminal_cleanup,
        ),
        patch.object(controller, "_start_online_callback_server"),
        patch.object(controller, "_stop_online_callback_server"),
        patch.object(
            controller,
            "_kill_forked_service",
            side_effect=lambda *args: killed_owners.append(args),
        ),
    ):
        with pytest.raises(TimeoutError, match="Worker creation timed out"):
            controller.initialize("rollout")
        assert initialization_started.is_set()
        assert controller._shutdown_requested.is_set()
        release_late_publish.set()
        assert terminal_cleanup_done.wait(timeout=1.0)

    assert killed_owners == [owner]
    assert deleted_roles == ["rollout-inf"]
    assert controller._forked_services == []
    assert controller._service_roles == []
    assert controller._destroyed is True


def test_workers_ready_deadline_win_cleans_owner_before_caller_resumes() -> None:
    controller = _controller()
    owner = ("http://guard", "completed-inf-server", 0)
    killed_owners: list[tuple[str, str, int]] = []
    deleted_roles: list[str] = []

    def publish_before_deadline_caller_resumes(*_args, **_kwargs) -> None:
        controller._forked_services.append(owner)
        controller._service_roles.append("completed-rollout-inf")

    def return_timed_out_after_background_finished(*, timeout: float) -> bool:
        assert timeout == controller.config.workers_ready_timeout
        future = controller._init_future
        assert future is not None
        future.result(timeout=1.0)
        assert controller._workers_ready.is_set()
        return False

    controller.scheduler.delete_workers.side_effect = (
        lambda *, role: deleted_roles.append(role)
    )

    with (
        patch.object(
            controller,
            "_bg_initialize",
            side_effect=publish_before_deadline_caller_resumes,
        ),
        patch.object(controller, "_start_online_callback_server"),
        patch.object(controller, "_stop_online_callback_server"),
        patch.object(
            controller._workers_ready,
            "wait",
            side_effect=return_timed_out_after_background_finished,
        ),
        patch.object(
            controller,
            "_kill_forked_service",
            side_effect=lambda *args: killed_owners.append(args),
        ),
    ):
        with pytest.raises(TimeoutError, match="Worker creation timed out"):
            controller.initialize("rollout")

    assert killed_owners == [owner]
    assert deleted_roles == ["completed-rollout-inf"]
    assert controller._forked_services == []
    assert controller._service_roles == []
    assert controller._destroyed is True


def test_workers_ready_wait_interrupt_requests_terminal_cleanup() -> None:
    controller = _controller()
    initialization_started = threading.Event()
    release_late_publish = threading.Event()
    owner = ("http://guard", "interrupted-inf-server", 0)
    captured_futures: list[concurrent.futures.Future] = []
    killed_owners: list[tuple[str, str, int]] = []
    deleted_roles: list[str] = []

    def publish_after_interrupt(*_args, **_kwargs) -> None:
        initialization_started.set()
        assert release_late_publish.wait(timeout=1.0)
        controller._forked_services.append(owner)
        controller._service_roles.append("interrupted-rollout-inf")

    def interrupt_wait(*, timeout: float) -> bool:
        assert timeout == controller.config.workers_ready_timeout
        assert initialization_started.wait(timeout=1.0)
        future = controller._init_future
        assert future is not None
        captured_futures.append(future)
        raise KeyboardInterrupt("startup wait interrupted")

    controller.scheduler.delete_workers.side_effect = (
        lambda *, role: deleted_roles.append(role)
    )

    with (
        patch.object(controller, "_bg_initialize", side_effect=publish_after_interrupt),
        patch.object(controller, "_start_online_callback_server"),
        patch.object(controller, "_stop_online_callback_server"),
        patch.object(controller._workers_ready, "wait", side_effect=interrupt_wait),
        patch.object(
            controller,
            "_kill_forked_service",
            side_effect=lambda *args: killed_owners.append(args),
        ),
    ):
        with pytest.raises(KeyboardInterrupt, match="startup wait interrupted"):
            controller.initialize("rollout")
        assert controller._shutdown_requested.is_set()
        release_late_publish.set()
        captured_futures[0].result(timeout=1.0)

    assert killed_owners == [owner]
    assert deleted_roles == ["interrupted-rollout-inf"]
    assert controller._forked_services == []
    assert controller._service_roles == []
    assert controller._destroyed is True


def test_callback_start_failure_is_rolled_back() -> None:
    controller = _controller()
    primary = OSError("callback thread start failed after bind")

    def fail_after_publishing_callback_owner() -> None:
        controller._callback_server = MagicMock()
        controller._callback_server_thread = MagicMock()
        raise primary

    with (
        patch.object(
            controller,
            "_start_online_callback_server",
            side_effect=fail_after_publishing_callback_owner,
        ),
        patch.object(controller, "_stop_online_callback_server") as stop_callback,
        pytest.raises(OSError) as exc_info,
    ):
        controller.initialize("rollout")

    assert exc_info.value is primary
    assert controller._shutdown_requested.is_set()
    stop_callback.assert_called_once_with()


def test_callback_thread_pre_ready_failure_keeps_primary_and_closes_owner() -> None:
    controller = _controller()
    primary = OSError("callback event loop creation failed")
    server = MagicMock()

    with (
        patch("werkzeug.serving.make_server", return_value=server),
        patch(
            "areal.v2.inference_service.controller.controller.asyncio.new_event_loop",
            side_effect=primary,
        ),
        pytest.raises(OSError) as exc_info,
    ):
        controller.initialize("rollout")

    assert exc_info.value is primary
    server.shutdown.assert_not_called()
    server.server_close.assert_called_once_with()
    assert controller._callback_server is None
    assert controller._callback_server_thread is None
    assert controller._shutdown_requested.is_set()


def test_callback_thread_never_ready_times_out_and_closes_without_shutdown() -> None:
    controller = _controller()
    controller.config.setup_timeout = 0.01
    release_server = threading.Event()
    server = MagicMock()
    server.serve_forever.side_effect = lambda **_kwargs: release_server.wait(
        timeout=1.0
    )
    server.server_close.side_effect = release_server.set

    with (
        patch("werkzeug.serving.make_server", return_value=server),
        pytest.raises(TimeoutError, match="did not enter its serving loop"),
    ):
        controller.initialize("rollout")

    server.shutdown.assert_not_called()
    server.server_close.assert_called_once_with()
    assert controller._callback_server is None
    assert controller._callback_server_thread is None
    assert controller._shutdown_requested.is_set()


def test_executor_submit_failure_keeps_primary_when_rollback_fails() -> None:
    controller = _controller()
    primary = RuntimeError("executor submit failed")
    cleanup_error = OSError("callback cleanup failed")
    executor = MagicMock()
    executor.submit.side_effect = primary

    with (
        patch("areal.infra.utils.concurrent.get_executor", return_value=executor),
        patch.object(controller, "_start_online_callback_server"),
        patch.object(controller, "destroy", side_effect=cleanup_error),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controller.initialize("rollout")

    assert exc_info.value is primary
    assert any(
        "callback cleanup failed" in note for note in getattr(primary, "__notes__", [])
    )


def test_fast_background_failure_is_not_masked_by_shutdown_wakeup() -> None:
    controller = _controller()
    primary = RuntimeError("background startup failed")
    future = MagicMock()
    future.done.return_value = False
    executor = MagicMock()

    def publish_failure_before_returning_future(*_args, **_kwargs):
        controller._initialization_error = primary
        controller._shutdown_requested.set()
        controller._workers_ready.set()
        return future

    executor.submit.side_effect = publish_failure_before_returning_future

    with (
        patch("areal.infra.utils.concurrent.get_executor", return_value=executor),
        patch.object(controller, "_start_online_callback_server"),
        patch.object(controller, "_stop_online_callback_server"),
        pytest.raises(RuntimeError) as exc_info,
    ):
        controller.initialize("rollout")

    assert exc_info.value is primary


def test_initialize_and_destroy_are_linearized_across_callback_start() -> None:
    controller = _controller()
    callback_start_entered = threading.Event()
    release_callback_start = threading.Event()
    destroy_called = threading.Event()
    destroy_finished = threading.Event()
    deleted_roles: list[str] = []
    initialize_errors: list[BaseException] = []
    destroy_errors: list[BaseException] = []

    def start_callback() -> None:
        callback_start_entered.set()
        assert release_callback_start.wait(timeout=1.0)

    def publish_role(*_args, **_kwargs) -> None:
        controller._service_roles.append("late-role")

    def initialize() -> None:
        try:
            controller.initialize("rollout")
        except BaseException as exc:
            initialize_errors.append(exc)

    def destroy() -> None:
        destroy_called.set()
        try:
            controller.destroy()
        except BaseException as exc:
            destroy_errors.append(exc)
        finally:
            destroy_finished.set()

    controller.scheduler.delete_workers.side_effect = (
        lambda *, role: deleted_roles.append(role)
    )

    with (
        patch.object(
            controller, "_start_online_callback_server", side_effect=start_callback
        ),
        patch.object(controller, "_stop_online_callback_server"),
        patch.object(controller, "_bg_initialize", side_effect=publish_role),
    ):
        initialize_thread = threading.Thread(target=initialize)
        initialize_thread.start()
        assert callback_start_entered.wait(timeout=1.0)

        destroy_thread = threading.Thread(target=destroy)
        destroy_thread.start()
        assert destroy_called.wait(timeout=1.0)
        assert not destroy_finished.wait(timeout=0.05)

        release_callback_start.set()
        initialize_thread.join(timeout=1.0)
        destroy_thread.join(timeout=1.0)

    assert not initialize_thread.is_alive()
    assert not destroy_thread.is_alive()
    assert len(initialize_errors) <= 1
    if initialize_errors:
        assert isinstance(
            initialize_errors[0],
            (RuntimeError, concurrent.futures.CancelledError),
        )
    assert destroy_errors == []
    assert controller._shutdown_requested.is_set()
    assert controller._service_roles == []
    assert deleted_roles in ([], ["late-role"])
    assert controller._destroyed is True


@pytest.mark.parametrize(
    "primary",
    [
        pytest.param(RuntimeError("rollout initialization failed"), id="runtime"),
        pytest.param(KeyboardInterrupt("startup interrupted"), id="interrupt"),
    ],
)
def test_ppo_constructor_failure_rolls_back_owned_resources_and_keeps_primary(
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, name: str, cleanup_error: BaseException | None = None):
            self.name = name
            self.cleanup_error = cleanup_error

        def destroy(self) -> None:
            events.append(f"destroy:{self.name}")
            if self.cleanup_error is not None:
                raise self.cleanup_error

    class Finalizer:
        def __init__(self, name: str):
            self.name = name

        def finalize(self) -> None:
            events.append(f"finalize:{self.name}")

    class Closer:
        def __init__(self, name: str):
            self.name = name

        def close(self) -> None:
            events.append(f"close:{self.name}")

    def fail_initialize(self, *_args, **_kwargs) -> None:
        self.actor = Resource("actor")
        self.critic = Resource("critic")
        self.ref = Resource("ref")
        self.teacher = Resource("teacher")
        self.rollout = Resource("rollout", RuntimeError("rollout cleanup failed"))
        self.eval_rollout = Resource("eval-rollout")
        self.data_controller = Resource("data")
        self.saver = Finalizer("saver")
        self.stats_logger = Closer("stats")
        self._train_rdataset = None
        self._valid_rdataset = None
        raise primary

    monkeypatch.setattr(PPOTrainer, "_initialize_impl", fail_initialize, raising=False)

    with pytest.raises(type(primary)) as exc_info:
        PPOTrainer(config=MagicMock())

    assert exc_info.value is primary
    assert events == [
        "close:stats",
        "finalize:saver",
        "destroy:data",
        "destroy:eval-rollout",
        "destroy:rollout",
        "destroy:teacher",
        "destroy:ref",
        "destroy:critic",
        "destroy:actor",
    ]
    assert any(
        "rollout cleanup failed" in note for note in getattr(primary, "__notes__", [])
    )


def test_ppo_close_finishes_actor_cleanup_before_raising_rollout_error() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    primary = RuntimeError("rollout scheduler cleanup failed")
    events: list[str] = []

    trainer.saver = MagicMock()
    trainer._train_rdataset = None
    trainer._valid_rdataset = None
    trainer.data_controller = None
    trainer.stats_logger = MagicMock()
    trainer.eval_rollout = None
    trainer.rollout = MagicMock()
    trainer.rollout.destroy.side_effect = primary
    trainer.teacher = None
    trainer.ref = None
    trainer.critic = None
    trainer.actor = MagicMock()
    trainer.actor.destroy.side_effect = lambda: events.append("destroy:actor")

    with (
        patch(
            "areal.trainer.rl_trainer.perf_tracer.save",
            side_effect=lambda **_kwargs: events.append("save:perf"),
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        trainer.close()

    assert exc_info.value is primary
    assert events == ["destroy:actor", "save:perf"]


def test_ppo_context_exit_keeps_training_error_when_cleanup_fails() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    training_error = RuntimeError("training failed")
    cleanup_error = RuntimeError("cleanup failed")
    trainer.close = MagicMock(side_effect=cleanup_error)

    result = trainer.__exit__(RuntimeError, training_error, None)

    assert result is False
    assert any(
        "cleanup failed" in note for note in getattr(training_error, "__notes__", [])
    )
