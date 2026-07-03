# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import select
import subprocess
import sys
from unittest.mock import MagicMock

import httpx
import pytest
import requests

from areal.infra.utils.proc import kill_process_tree
from areal.v2.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.v2.weight_update.controller.controller import (
    WeightUpdateController,
)
from areal.v2.weight_update.gateway.config import WeightUpdateResult

GATEWAY_URL = "http://localhost:7080"


def _spawn_gateway_with_inherited_stdout() -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == b"ready\n"
    return process


def _force_reap_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        kill_process_tree(process.pid)
    process.wait(timeout=5)
    if process.stdout is not None:
        process.stdout.close()


class _ScriptedGatewayProcess:
    def __init__(
        self,
        wait_effects: list[int | BaseException],
        *,
        kill_error: BaseException | None = None,
    ) -> None:
        self.pid = 12345
        self.wait_effects = wait_effects
        self.kill_error = kill_error
        self.wait_timeouts: list[float | None] = []
        self.kill_count = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        effect = self.wait_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect

    def kill(self) -> None:
        self.kill_count += 1
        if self.kill_error is not None:
            raise self.kill_error


@pytest.fixture()
def ctrl() -> WeightUpdateController:
    c = WeightUpdateController(
        config=WeightUpdateControllerConfig(
            admin_api_key="test-admin-key",
            request_timeout=10.0,
        )
    )
    c._gateway_url = GATEWAY_URL
    c._session = MagicMock(spec=requests.Session)
    return c


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp,
        )
    return resp


class TestHealthCheck:
    def test_health_check_success_returns_true(self, ctrl):
        ctrl._session.get.return_value = _mock_response(200, {"status": "healthy"})

        assert ctrl.health_check() is True
        ctrl._session.get.assert_called_once_with(f"{GATEWAY_URL}/health", timeout=10.0)

    def test_health_check_connect_error_returns_false(self, ctrl):
        ctrl._session.get.side_effect = httpx.ConnectError("refused")

        assert ctrl.health_check() is False


class TestConnect:
    def test_connect_stores_pair_name(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])

        assert ctrl._pair_name == "pair0"

    def test_connect_sends_correct_request(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000", "http://train2:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect("pair0", train_urls, infer_urls)

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "awex",
                "save_path": "",
                "use_lora": False,
                "lora_name": "",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )

    def test_connect_disk_mode_sends_disk_fields(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect(
            "pair0",
            train_urls,
            infer_urls,
            mode="disk",
            save_path="/shared/weights",
            use_lora=True,
            lora_name="my-lora",
        )

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "disk",
                "save_path": "/shared/weights",
                "use_lora": True,
                "lora_name": "my-lora",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )


class TestUpdateWeights:
    def test_update_weights_returns_result(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200,
            {"status": "ok", "version": 5, "duration_ms": 123.4, "error": None},
        )

        result = ctrl.update_weights(version=5)

        assert isinstance(result, WeightUpdateResult)
        assert result.status == "ok"
        assert result.version == 5
        assert result.duration_ms == 123.4
        assert result.error is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/update_weights",
            json={"pair_name": "pair0", "version": 5},
            timeout=10.0,
        )

    def test_update_weights_raises_when_not_connected(self, ctrl):
        with pytest.raises(RuntimeError, match="Not connected"):
            ctrl.update_weights(version=1)


class TestDisconnect:
    def test_disconnect_clears_state(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200, {"status": "ok", "pair_name": "pair0"}
        )

        ctrl.disconnect()

        assert ctrl._pair_name is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/disconnect",
            json={"pair_name": "pair0"},
            timeout=10.0,
        )

    def test_disconnect_noop_when_not_connected(self, ctrl):
        ctrl.disconnect()
        assert ctrl._pair_name is None


class TestLifecycle:
    @pytest.mark.parametrize(
        ("failure_stage", "error_message"),
        [
            ("http_client", "http client failed"),
            ("health_check", "health check failed"),
        ],
    )
    def test_initialize_failure_reaps_started_gateway(
        self, monkeypatch, failure_stage, error_message
    ):
        from areal.v2.weight_update.controller import controller as controller_module

        real_popen = subprocess.Popen
        spawned: list[subprocess.Popen[bytes]] = []

        def spawn_test_gateway(*_args, **_kwargs):
            process = real_popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            spawned.append(process)
            return process

        controller = WeightUpdateController()
        monkeypatch.setattr(controller_module.subprocess, "Popen", spawn_test_gateway)
        if failure_stage == "http_client":
            monkeypatch.setattr(
                controller_module.httpx,
                "Client",
                MagicMock(side_effect=RuntimeError(error_message)),
            )
        else:
            monkeypatch.setattr(
                controller,
                "_wait_for_health",
                MagicMock(side_effect=RuntimeError(error_message)),
            )

        try:
            with pytest.raises(RuntimeError, match=error_message):
                controller.initialize()

            assert len(spawned) == 1
            assert spawned[0].returncode is not None
            assert controller._gateway_proc is None
            assert controller._session is None
        finally:
            for process in spawned:
                _force_reap_process(process)

    def test_initialize_preserves_primary_error_when_rollback_fails(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as controller_module

        process = _spawn_gateway_with_inherited_stdout()
        primary_error = RuntimeError("health check failed")
        cleanup_error = RuntimeError("cleanup failed")
        cleanup_attempted = False

        def fail_health_check():
            raise primary_error

        def fail_cleanup():
            nonlocal cleanup_attempted
            cleanup_attempted = True
            raise cleanup_error

        controller = WeightUpdateController()
        monkeypatch.setattr(
            controller_module.subprocess,
            "Popen",
            lambda *_args, **_kwargs: process,
        )
        monkeypatch.setattr(controller, "_wait_for_health", fail_health_check)
        monkeypatch.setattr(controller, "destroy", fail_cleanup)

        try:
            with pytest.raises(RuntimeError) as exc_info:
                controller.initialize()

            assert cleanup_attempted is True
            assert exc_info.value is primary_error
            assert any(
                "cleanup failed" in note
                for note in getattr(exc_info.value, "__notes__", [])
            )
        finally:
            try:
                WeightUpdateController.destroy(controller)
            finally:
                _force_reap_process(process)

    def test_destroy_reaps_gateway_and_releases_inherited_output(self):
        controller = WeightUpdateController()
        process = _spawn_gateway_with_inherited_stdout()
        controller._gateway_proc = process

        try:
            controller.destroy()

            assert process.returncode is not None
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], 1.0)
            assert readable == [process.stdout]
            assert process.stdout.read() == b""
        finally:
            _force_reap_process(process)

    def test_destroy_is_idempotent(self):
        controller = WeightUpdateController()
        process = _spawn_gateway_with_inherited_stdout()
        controller._gateway_proc = process

        try:
            controller.destroy()
            controller.destroy()

            assert process.returncode is not None
            assert controller._gateway_proc is None
        finally:
            _force_reap_process(process)

    def test_destroy_reaps_gateway_when_http_session_close_fails(self):
        controller = WeightUpdateController()
        session = MagicMock(spec=httpx.Client)
        session.close.side_effect = RuntimeError("session close failed")
        process = _spawn_gateway_with_inherited_stdout()
        controller._session = session
        controller._gateway_proc = process

        try:
            controller.destroy()

            assert process.returncode is not None
            assert controller._session is None
            assert controller._gateway_proc is None
            session.close.assert_called_once_with()
        finally:
            _force_reap_process(process)

    def test_destroy_uses_blocking_wait_after_kill_when_tree_cleanup_fails(
        self, monkeypatch
    ):
        from areal.v2.weight_update.controller import controller as controller_module

        tree_cleanup_attempted = False

        def fail_tree_cleanup(_pid):
            nonlocal tree_cleanup_attempted
            tree_cleanup_attempted = True
            raise RuntimeError("tree cleanup failed")

        process = _ScriptedGatewayProcess(
            [subprocess.TimeoutExpired(cmd="gateway", timeout=1), -9]
        )
        controller = WeightUpdateController()
        controller._gateway_proc = process
        monkeypatch.setattr(controller_module, "kill_process_tree", fail_tree_cleanup)

        controller.destroy()

        assert tree_cleanup_attempted is True
        assert process.wait_timeouts == [1, None]
        assert process.kill_count == 1
        assert controller._gateway_proc is None

    def test_destroy_waits_after_process_exits_before_kill(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as controller_module

        process = _ScriptedGatewayProcess(
            [subprocess.TimeoutExpired(cmd="gateway", timeout=1), -9],
            kill_error=ProcessLookupError(),
        )
        controller = WeightUpdateController()
        controller._gateway_proc = process
        monkeypatch.setattr(controller_module, "kill_process_tree", lambda _pid: None)

        controller.destroy()

        assert process.wait_timeouts == [1, None]
        assert process.kill_count == 1
        assert controller._gateway_proc is None

    def test_destroy_retains_gateway_owner_when_final_wait_fails(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as controller_module

        wait_error = OSError("wait failed")
        process = _ScriptedGatewayProcess(
            [
                subprocess.TimeoutExpired(cmd="gateway", timeout=1),
                wait_error,
                -9,
            ]
        )
        controller = WeightUpdateController()
        controller._gateway_proc = process
        monkeypatch.setattr(controller_module, "kill_process_tree", lambda _pid: None)

        with pytest.raises(OSError, match="wait failed") as exc_info:
            controller.destroy()

        assert exc_info.value is wait_error
        assert controller._gateway_proc is process

        controller.destroy()

        assert process.wait_timeouts == [1, None, 1]
        assert process.kill_count == 1
        assert controller._gateway_proc is None

    def test_full_lifecycle(self, ctrl):
        connect_resp = _mock_response(200, {"pair_name": "pair0"})
        update_resp = _mock_response(
            200, {"status": "ok", "version": 1, "duration_ms": 50.0, "error": None}
        )
        disconnect_resp = _mock_response(200, {"status": "ok", "pair_name": "pair0"})
        ctrl._session.post.side_effect = [connect_resp, update_resp, disconnect_resp]

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])
        assert ctrl._pair_name == "pair0"

        result = ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        ctrl.disconnect()
        assert ctrl._pair_name is None

    def test_gateway_error_raises_http_error(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(500, {"error": "internal"})

        with pytest.raises(requests.HTTPError):
            ctrl.update_weights(version=1)
