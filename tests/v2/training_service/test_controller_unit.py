from __future__ import annotations

import concurrent.futures
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.api.io_struct import WeightUpdateMeta
from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.inference_service.controller.controller import RolloutControllerV2
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
    _disk_gateway_save_root,
)
from areal.v2.weight_update.controller.controller import WeightUpdateController
from areal.v2.weight_update.gateway.config import WeightUpdateResult

MODULE = "areal.v2.training_service.controller.controller"


def _make_response(method: str, url: str, *, json=None) -> httpx.Response:
    return httpx.Response(
        200,
        json=json,
        request=httpx.Request(method, url),
    )


def _make_controller(scheduler: MagicMock | None = None) -> GatewayTrainController:
    return GatewayTrainController(
        train_engine="areal.engine.FSDPEngine",
        scheduler=scheduler or MagicMock(),
        config=TrainEngineConfig(
            experiment_name="test-exp",
            trial_name="trial-0",
            backend="fsdp:d2",
            scheduling_spec=(
                SchedulingSpec(
                    cpu=1,
                    gpu=1,
                    mem=1024,
                    port_count=1,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
            ),
            admin_api_key="test-admin-key",
            request_timeout=5.0,
            setup_timeout=5.0,
        ),
    )


def _make_rollout() -> RolloutControllerV2:
    rollout = RolloutControllerV2.__new__(RolloutControllerV2)
    rollout._init_future = None
    rollout._inf_addrs = ["http://inference-worker"]
    rollout.config = SimpleNamespace(api_url=None)
    rollout.rollout_alloc = SimpleNamespace(backend="sglang")
    return rollout


def _disk_meta() -> SimpleNamespace:
    return SimpleNamespace(
        type="disk",
        path="",
        use_lora=False,
        lora_name="",
        lora_keep_versions=0,
    )


class _FakeAsyncClient:
    def __init__(self, responses_or_errors):
        self._responses_or_errors = list(responses_or_errors)
        self.get = AsyncMock(side_effect=self._get)
        self.post = AsyncMock(side_effect=self._post)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def _get(self, _url: str):
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    async def _post(self, _url: str, json=None, **kwargs):
        _ = json
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _FailOnceSession:
    def __init__(self, error: BaseException):
        self.error = error
        self.close_count = 0
        self.raised_traceback = None

    def close(self) -> None:
        self.close_count += 1
        if self.close_count == 1:
            try:
                raise self.error
            except BaseException as exc:
                self.raised_traceback = exc.__traceback__
                raise


class TestGatewayTrainControllerInitialization:
    def test_workers_ready_timeout_requests_shutdown_and_reaps_late_owner(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        controller.config.workers_ready_timeout = 0.01
        initialization_started = threading.Event()
        release_late_publish = threading.Event()
        owner = ("http://guard", "late-train-worker", 0)
        killed_owners: list[tuple[str, str, int]] = []
        deleted_roles: list[str] = []
        terminal_cleanup_done = threading.Event()

        def publish_after_timeout(*_args, **_kwargs) -> None:
            initialization_started.set()
            assert release_late_publish.wait(timeout=1.0)
            controller._forked_services.append(owner)
            controller._service_roles.append("actor-guard")

        def delete_role(*, role: str) -> None:
            deleted_roles.append(role)

        guarded_initialize = controller._guarded_bg_initialize

        def observe_terminal_cleanup(*args, **kwargs) -> None:
            try:
                guarded_initialize(*args, **kwargs)
            finally:
                terminal_cleanup_done.set()

        scheduler.delete_workers.side_effect = delete_role

        with (
            patch.object(controller, "_bg_initialize", publish_after_timeout),
            patch.object(
                controller,
                "_guarded_bg_initialize",
                side_effect=observe_terminal_cleanup,
            ),
            patch.object(
                controller,
                "_kill_forked_service",
                side_effect=lambda *args: killed_owners.append(args),
            ),
        ):
            with pytest.raises(TimeoutError, match="Worker creation timed out"):
                controller.initialize("actor")

            assert initialization_started.is_set()
            assert controller._shutdown_requested.is_set()
            release_late_publish.set()
            assert terminal_cleanup_done.wait(timeout=1.0)

        assert killed_owners == [owner]
        assert deleted_roles == ["actor-guard"]
        assert controller._forked_services == []
        assert controller._service_roles == []

    def test_workers_ready_deadline_win_cleans_owner_before_caller_resumes(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        owner = ("http://guard", "completed-train-worker", 0)
        killed_owners: list[tuple[str, str, int]] = []
        deleted_roles: list[str] = []

        def publish_before_deadline_caller_resumes(*_args, **_kwargs) -> None:
            controller._forked_services.append(owner)
            controller._service_roles.append("completed-actor-guard")

        def return_timed_out_after_background_finished(*, timeout: float) -> bool:
            assert timeout == controller.config.workers_ready_timeout
            future = controller._init_future
            assert future is not None
            future.result(timeout=1.0)
            assert controller._workers_ready.is_set()
            return False

        scheduler.delete_workers.side_effect = lambda *, role: deleted_roles.append(
            role
        )

        with (
            patch.object(
                controller,
                "_bg_initialize",
                side_effect=publish_before_deadline_caller_resumes,
            ),
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
                controller.initialize("actor")

        assert killed_owners == [owner]
        assert deleted_roles == ["completed-actor-guard"]
        assert controller._forked_services == []
        assert controller._service_roles == []

    def test_workers_ready_wait_interrupt_requests_terminal_cleanup(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        initialization_started = threading.Event()
        release_late_publish = threading.Event()
        owner = ("http://guard", "interrupted-train-worker", 0)
        captured_futures: list[concurrent.futures.Future] = []
        killed_owners: list[tuple[str, str, int]] = []
        deleted_roles: list[str] = []

        def publish_after_interrupt(*_args, **_kwargs) -> None:
            initialization_started.set()
            assert release_late_publish.wait(timeout=1.0)
            controller._forked_services.append(owner)
            controller._service_roles.append("interrupted-actor-guard")

        def interrupt_wait(*, timeout: float) -> bool:
            assert timeout == controller.config.workers_ready_timeout
            assert initialization_started.wait(timeout=1.0)
            future = controller._init_future
            assert future is not None
            captured_futures.append(future)
            raise KeyboardInterrupt("startup wait interrupted")

        scheduler.delete_workers.side_effect = lambda *, role: deleted_roles.append(
            role
        )

        with (
            patch.object(
                controller, "_bg_initialize", side_effect=publish_after_interrupt
            ),
            patch.object(controller._workers_ready, "wait", side_effect=interrupt_wait),
            patch.object(
                controller,
                "_kill_forked_service",
                side_effect=lambda *args: killed_owners.append(args),
            ),
        ):
            with pytest.raises(KeyboardInterrupt, match="startup wait interrupted"):
                controller.initialize("actor")
            assert controller._shutdown_requested.is_set()
            release_late_publish.set()
            captured_futures[0].result(timeout=1.0)

        assert killed_owners == [owner]
        assert deleted_roles == ["interrupted-actor-guard"]
        assert controller._forked_services == []
        assert controller._service_roles == []

    def test_executor_submit_failure_keeps_primary_when_rollback_fails(self):
        controller = _make_controller()
        primary = RuntimeError("executor submit failed")
        cleanup_error = OSError("training cleanup failed")
        executor = MagicMock()
        executor.submit.side_effect = primary

        with (
            patch(f"{MODULE}.get_executor", return_value=executor),
            patch.object(controller, "destroy", side_effect=cleanup_error),
            pytest.raises(RuntimeError) as exc_info,
        ):
            controller.initialize("actor")

        assert exc_info.value is primary
        assert any(
            "training cleanup failed" in note
            for note in getattr(primary, "__notes__", [])
        )

    def test_fast_background_failure_is_not_masked_by_shutdown_wakeup(self):
        controller = _make_controller()
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
            patch(f"{MODULE}.get_executor", return_value=executor),
            pytest.raises(RuntimeError) as exc_info,
        ):
            controller.initialize("actor")

        assert exc_info.value is primary

    def test_ensure_initialized_surfaces_terminal_background_failure(self):
        controller = _make_controller()
        primary = RuntimeError("background startup failed after readiness")
        controller._init_future = None
        controller._initialization_error = primary

        with pytest.raises(RuntimeError) as exc_info:
            controller._ensure_initialized()

        assert exc_info.value is primary

    @pytest.mark.asyncio
    async def test_async_startup_keeps_primary_when_local_rollback_fails(self):
        scheduler = MagicMock()
        scheduler.exp_config = None
        primary = RuntimeError("guard discovery failed")
        scheduler.get_workers.side_effect = primary
        scheduler.delete_workers.side_effect = OSError("guard cleanup failed")
        controller = _make_controller(scheduler)

        with pytest.raises(RuntimeError) as exc_info:
            await controller._async_initialize("actor")

        assert exc_info.value is primary
        assert controller._service_roles == ["actor-guard"]
        assert any(
            "guard cleanup failed" in note for note in getattr(primary, "__notes__", [])
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_seed", [-1, 2**32, True])
    async def test_invalid_base_seed_fails_before_guard_creation(self, invalid_seed):
        scheduler = MagicMock()
        scheduler.exp_config = None
        scheduler.create_workers.side_effect = AssertionError(
            "guard creation must not start for an invalid seed"
        )
        controller = _make_controller(scheduler)

        with pytest.raises(ValueError, match="base_seed.*unsigned 32-bit"):
            await controller._async_initialize(role="actor", base_seed=invalid_seed)

        scheduler.create_workers.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("experiment_seed", [0, 20260703])
    async def test_async_initialize_offloads_scheduler_and_uses_async_helpers(
        self, experiment_seed
    ):
        worker0 = MagicMock(ip="127.0.0.1", worker_ports=[18000], id="guard-0")
        worker1 = MagicMock(ip="127.0.0.1", worker_ports=[18001], id="guard-1")

        scheduler = MagicMock()
        scheduler.create_workers.return_value = ["guard-0", "guard-1"]
        scheduler.get_workers.return_value = [worker0, worker1]

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = SimpleNamespace(
            seed=experiment_seed,
            scheduler=SimpleNamespace(type="local"),
        )

        def make_scheduler(*, exp_config):
            scheduler.exp_config = exp_config
            return scheduler

        with patch.object(
            rl_trainer, "LocalScheduler", side_effect=make_scheduler
        ) as scheduler_cls:
            scheduler = trainer._init_scheduler()

        scheduler_cls.assert_called_once_with(exp_config=trainer.config)

        controller = _make_controller(scheduler)
        controller._role = "train-role"

        port_client = _FakeAsyncClient(
            [
                _make_response(
                    "POST",
                    "http://127.0.0.1:18000/alloc_ports",
                    json={"ports": [29500]},
                )
            ]
        )

        async def _run_in_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("httpx.AsyncClient", return_value=port_client),
            patch(
                f"{MODULE}.asyncio.to_thread", side_effect=_run_in_thread
            ) as mock_to_thread,
            patch.object(
                controller, "_async_set_guards_env", new_callable=AsyncMock
            ) as mock_set_env,
            patch.object(
                controller,
                "_async_fork_on_guard",
                new_callable=AsyncMock,
                side_effect=[
                    ("127.0.0.1", 19001),
                    ("127.0.0.1", 19002),
                    ("127.0.0.1", 18081),
                    ("127.0.0.1", 18082),
                    ("127.0.0.1", 18080),
                ],
            ) as mock_async_fork,
            patch.object(controller, "_fork_on_guard", autospec=True) as mock_sync_fork,
            patch.object(
                controller, "_create_engine_on_worker", new_callable=AsyncMock
            ) as mock_create_engine,
            patch.object(
                controller,
                "_call_worker_engine_endpoint",
                new_callable=AsyncMock,
            ) as mock_call_engine,
            patch.object(
                controller, "_register_in_router", new_callable=AsyncMock
            ) as mock_register,
        ):
            await controller._async_initialize(role="train-role")

        assert mock_to_thread.await_count == 2
        create_call = mock_to_thread.await_args_list[0]
        get_call = mock_to_thread.await_args_list[1]
        assert create_call.args[0] is scheduler.create_workers
        assert get_call.args[0] is scheduler.get_workers
        assert get_call.kwargs == {
            "role": "train-role-guard",
            "timeout": 5,
        }

        mock_set_env.assert_awaited_once()
        assert mock_async_fork.await_count == 5
        mock_sync_fork.assert_not_called()

        for rank, fork_call in enumerate(mock_async_fork.await_args_list[:2]):
            assert fork_call.kwargs["raw_cmd"][-6:] == [
                "--seed",
                str(experiment_seed),
                "--seed-role",
                "train-role",
                "--seed-rank",
                str(rank),
            ]

        assert mock_create_engine.await_count == 2
        assert mock_call_engine.await_count == 4
        mock_register.assert_awaited_once_with(
            "http://127.0.0.1:18081",
            "http://127.0.0.1:18082",
            controller.api_key,
        )

        assert controller._worker_addrs == [
            "http://127.0.0.1:19001",
            "http://127.0.0.1:19002",
        ]
        assert controller._router_addr == "http://127.0.0.1:18081"
        assert controller._model_addr == "http://127.0.0.1:18082"
        assert controller._gateway_addr == "http://127.0.0.1:18080"
        assert controller.api_key is not None
        assert controller.api_key.startswith("ak-train-role-")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_seed", [True, "bad", 1.5])
    async def test_invalid_scheduler_seed_fails_before_guard_creation(
        self, invalid_seed
    ):
        scheduler = MagicMock()
        scheduler.exp_config = SimpleNamespace(seed=invalid_seed)
        scheduler.create_workers.side_effect = AssertionError(
            "guard creation must not start for an invalid seed"
        )
        controller = _make_controller(scheduler)

        with pytest.raises(ValueError, match="base_seed.*unsigned 32-bit"):
            await controller._async_initialize(role="actor")

        scheduler.create_workers.assert_not_called()


class TestGatewayTrainControllerWeightUpdate:
    @staticmethod
    def _prepare_controller(result=None, error=None):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = MagicMock()
        controller._weight_update_ctrl.update_weights.side_effect = error
        if error is None:
            controller._weight_update_ctrl.update_weights.return_value = result
        return controller

    def test_failed_result_raises_and_keeps_generation_paused(self):
        result = WeightUpdateResult(
            status="error",
            version=1,
            duration_ms=10,
            error="inference load failed",
        )
        controller = self._prepare_controller(result=result)
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="inference load failed"):
            controller.update_weights(meta)

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_not_called()

    def test_gateway_exception_keeps_generation_paused(self):
        controller = self._prepare_controller(error=TimeoutError("gateway timeout"))
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(TimeoutError, match="gateway timeout"):
            controller.update_weights(meta)

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_not_called()

    def test_pause_exception_attempts_resume(self):
        controller = self._prepare_controller()
        controller.rollout.pause_generation.side_effect = RuntimeError("pause failed")
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="pause failed"):
            controller.update_weights(meta)

        controller.rollout.continue_generation.assert_called_once_with()

    def test_resume_exception_does_not_mask_pause_exception(self):
        controller = self._prepare_controller()
        controller.rollout.pause_generation.side_effect = RuntimeError("pause failed")
        controller.rollout.continue_generation.side_effect = RuntimeError(
            "resume failed"
        )
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="pause failed"):
            controller.update_weights(meta)

    def test_successful_disk_update_removes_versioned_checkpoint(self, tmp_path):
        checkpoint_path = tmp_path / "weight_update_v1"
        checkpoint_path.mkdir()
        result = WeightUpdateResult(
            status="ok",
            version=1,
            duration_ms=10,
        )
        controller = self._prepare_controller(result=result)
        controller._disk_weight_update_root = str(tmp_path)
        meta = WeightUpdateMeta(
            type="disk",
            path=str(checkpoint_path),
            version=1,
            clear_checkpoint_after_load=True,
        )

        controller.update_weights(meta)

        assert not checkpoint_path.exists()
        controller.rollout.continue_generation.assert_called_once_with()

    def test_disk_cleanup_uses_connected_root_not_update_meta_path(self, tmp_path):
        checkpoint_path = tmp_path / "weight_update_v1"
        checkpoint_path.mkdir()
        unrelated_path = tmp_path / "unrelated"
        unrelated_path.mkdir()
        result = WeightUpdateResult(
            status="ok",
            version=1,
            duration_ms=10,
        )
        controller = self._prepare_controller(result=result)
        controller._disk_weight_update_root = str(tmp_path)
        meta = WeightUpdateMeta(
            type="disk",
            path=str(unrelated_path),
            version=1,
            clear_checkpoint_after_load=True,
        )

        controller.update_weights(meta)

        assert not checkpoint_path.exists()
        assert unrelated_path.exists()

    def test_disk_gateway_save_root_matches_versioned_meta_path(self, tmp_path):
        meta = WeightUpdateMeta.from_disk(
            experiment_name="test-exp",
            trial_name="trial-0",
            file_root=str(tmp_path),
        )

        save_root = _disk_gateway_save_root(meta)

        assert os.path.join(save_root, "weight_update_v3") == meta.with_version(3).path

    def test_destroy_waits_for_inflight_weight_update(self):
        update_entered = threading.Event()
        allow_update = threading.Event()
        destroy_entered = threading.Event()

        class BlockingWeightUpdateController:
            def update_weights(self, *, version):
                update_entered.set()
                if not allow_update.wait(timeout=2.0):
                    raise TimeoutError("test did not release weight update")
                return WeightUpdateResult(
                    status="ok",
                    version=version,
                    duration_ms=10,
                )

            def destroy(self):
                destroy_entered.set()

        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = BlockingWeightUpdateController()
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )
        update_errors: list[BaseException] = []
        destroy_errors: list[BaseException] = []

        def update_weights() -> None:
            try:
                controller.update_weights(meta)
            except BaseException as exc:
                update_errors.append(exc)

        def destroy() -> None:
            try:
                controller.destroy()
            except BaseException as exc:
                destroy_errors.append(exc)

        update_thread = threading.Thread(target=update_weights)
        destroy_thread = threading.Thread(target=destroy)
        update_thread.start()
        assert update_entered.wait(timeout=1.0)
        destroy_thread.start()
        assert controller._shutdown_requested.wait(timeout=1.0)
        destroy_advanced_past_update = destroy_entered.wait(timeout=0.2)
        allow_update.set()
        update_thread.join(timeout=2.0)
        destroy_thread.join(timeout=2.0)

        assert not destroy_advanced_past_update
        assert not update_thread.is_alive()
        assert not destroy_thread.is_alive()
        assert update_errors == []
        assert destroy_errors == []
        assert destroy_entered.is_set()
        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_called_once_with()

    def test_weight_update_rejects_after_shutdown_is_requested(self):
        controller = self._prepare_controller()
        controller._shutdown_requested.set()
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="after shutdown was requested"):
            controller.update_weights(meta)

        controller.rollout.pause_generation.assert_not_called()
        controller._weight_update_ctrl.update_weights.assert_not_called()

    def test_connect_engine_uses_canonical_disk_save_root(self, tmp_path):
        controller = _make_controller()
        controller._worker_addrs = ["http://train:8000"]
        rollout = RolloutControllerV2.__new__(RolloutControllerV2)
        rollout.config = SimpleNamespace(api_url=None)
        rollout.rollout_alloc = SimpleNamespace(backend="sglang")
        rollout._init_future = None
        rollout._inf_addrs = ["http://infer:9000"]
        meta = WeightUpdateMeta.from_disk(
            experiment_name="test-exp",
            trial_name="trial-0",
            file_root=str(tmp_path),
        )

        with (
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.initialize"
            ),
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.connect"
            ) as mock_connect,
        ):
            controller.connect_engine(rollout, meta)

        expected_root = os.path.dirname(meta.path)
        assert controller._disk_weight_update_root == expected_root
        assert mock_connect.call_args.kwargs["save_path"] == expected_root

    @pytest.mark.parametrize(
        ("backend", "api_url"),
        [("vllm", None), (None, "https://example.com/v1")],
    )
    def test_connect_engine_rejects_unsupported_disk_rollout(self, backend, api_url):
        controller = _make_controller()
        rollout = RolloutControllerV2.__new__(RolloutControllerV2)
        rollout.config = SimpleNamespace(api_url=api_url)
        rollout.rollout_alloc = (
            SimpleNamespace(backend=backend) if backend is not None else None
        )
        rollout._init_future = None
        rollout._inf_addrs = []
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update")

        with (
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.initialize"
            ),
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.connect"
            ),
            pytest.raises(ValueError, match="local SGLang"),
        ):
            controller.connect_engine(rollout, meta)


class TestGatewayTrainControllerLifecycle:
    def test_guard_deletion_does_not_mask_unconfirmed_direct_kill(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        owner = ("http://guard", "train-worker", 0)
        controller._forked_services.append(owner)
        controller._service_roles.append("actor-guard")

        rejected = MagicMock()
        rejected.raise_for_status.side_effect = requests.HTTPError("direct kill failed")

        accepted = MagicMock()
        with (
            patch("requests.post", return_value=rejected),
            pytest.raises(requests.HTTPError, match="direct kill failed"),
        ):
            controller.destroy()

        scheduler.delete_workers.assert_called_once_with(role="actor-guard")
        assert controller._forked_services == [owner]
        assert controller._service_roles == []

        with patch("requests.post", return_value=accepted):
            controller.destroy()

        assert scheduler.delete_workers.call_count == 1
        assert controller._forked_services == []

    def test_failed_direct_kill_and_guard_deletion_retain_owners_for_retry(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        owner = ("http://guard", "train-worker", 0)
        controller._forked_services.append(owner)
        controller._service_roles.append("actor-guard")
        rejected = MagicMock()
        rejected.raise_for_status.side_effect = requests.HTTPError("direct kill failed")
        accepted = MagicMock()
        scheduler.delete_workers.side_effect = [
            OSError("guard delete failed"),
            None,
        ]

        with (
            patch("requests.post", return_value=rejected),
            pytest.raises(requests.HTTPError, match="direct kill failed"),
        ):
            controller.destroy()

        assert controller._forked_services == [owner]
        assert controller._service_roles == ["actor-guard"]

        with patch("requests.post", return_value=accepted) as post:
            controller.destroy()

        assert post.call_count == 1
        assert scheduler.delete_workers.call_count == 2
        assert controller._forked_services == []
        assert controller._service_roles == []

    def test_guard_fallback_does_not_mask_direct_kill_keyboard_interrupt(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        owner = ("http://guard", "train-worker", 0)
        primary = KeyboardInterrupt("direct kill interrupted")
        controller._forked_services.append(owner)
        controller._service_roles.append("actor-guard")

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

    def test_destroy_releases_owned_weight_update_controller_once(self):
        class OwnedWeightUpdateController:
            def __init__(self):
                self.destroy_count = 0

            def destroy(self):
                self.destroy_count += 1

        controller = _make_controller()
        weight_update_controller = OwnedWeightUpdateController()
        controller._weight_update_ctrl = weight_update_controller

        controller.destroy()
        controller.destroy()

        assert weight_update_controller.destroy_count == 1
        assert controller._weight_update_ctrl is None

    def test_destroy_continues_worker_cleanup_when_weight_update_destroy_fails(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        session_error = RuntimeError("weight update cleanup failed")
        session = _FailOnceSession(session_error)
        weight_update_controller = WeightUpdateController()
        weight_update_controller._session = session
        controller._weight_update_ctrl = weight_update_controller
        controller._service_roles = ["actor"]

        controller.destroy()

        assert session.close_count == 1
        scheduler.delete_workers.assert_called_once_with(role="actor")
        assert controller._weight_update_ctrl is weight_update_controller
        assert controller._service_roles == []

        controller.destroy()

        assert session.close_count == 2
        assert weight_update_controller._session is None
        assert controller._weight_update_ctrl is None

    def test_destroy_defers_weight_update_keyboard_interrupt_until_cleanup_finishes(
        self, monkeypatch
    ):
        from areal.v2.weight_update.controller import controller as wu_module

        primary_error = KeyboardInterrupt("session close interrupted")
        process_error = OSError("gateway wait failed")
        session = _FailOnceSession(primary_error)
        gateway_process = MagicMock()
        gateway_process.pid = 12345
        gateway_process.wait.side_effect = [process_error, 0]
        weight_update_controller = WeightUpdateController()
        weight_update_controller._session = session
        weight_update_controller._gateway_proc = gateway_process
        weight_update_controller._gateway_url = "http://weight-gateway"
        monkeypatch.setattr(wu_module, "kill_process_tree", lambda _pid: None)
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        graceful_shutdown = MagicMock()
        kill_forked_service = MagicMock()
        controller._graceful_shutdown_workers = graceful_shutdown
        controller._kill_forked_service = kill_forked_service
        controller._weight_update_ctrl = weight_update_controller
        controller._worker_addrs = ["http://train-worker"]
        controller._forked_services = [("http://guard", "router", 0)]
        controller._service_roles = ["actor"]
        controller.api_key = "test-api-key"

        try:
            with pytest.raises(KeyboardInterrupt) as exc_info:
                controller.destroy()

            assert exc_info.value is primary_error
            traceback_cursor = exc_info.tb
            assert session.raised_traceback is not None
            traceback_nodes = []
            while traceback_cursor is not None:
                traceback_nodes.append(traceback_cursor)
                traceback_cursor = traceback_cursor.tb_next
            assert session.raised_traceback in traceback_nodes
            assert any(
                "OSError: gateway wait failed" in note
                for note in getattr(primary_error, "__notes__", [])
            )

            graceful_shutdown.assert_called_once_with()
            kill_forked_service.assert_called_once_with("http://guard", "router", 0)
            scheduler.delete_workers.assert_called_once_with(role="actor")
            assert controller._weight_update_ctrl is weight_update_controller
            assert weight_update_controller._session is session
            assert weight_update_controller._gateway_proc is gateway_process
            assert weight_update_controller.gateway_url == "http://weight-gateway"
            assert controller._worker_addrs == []
            assert controller._forked_services == []
            assert controller._service_roles == []
            assert controller.api_key is None

            controller.destroy()

            assert session.close_count == 2
            assert weight_update_controller._session is None
            assert weight_update_controller._gateway_proc is None
            assert weight_update_controller.gateway_url == ""
            assert controller._weight_update_ctrl is None
        finally:
            if controller._weight_update_ctrl is not None:
                controller.destroy()

    def test_destroy_keeps_weight_update_base_exception_primary_when_cleanup_fails(
        self,
    ):
        primary_error = KeyboardInterrupt("session close interrupted")
        secondary_error = RuntimeError("worker cleanup failed")
        session = _FailOnceSession(primary_error)
        weight_update_controller = WeightUpdateController()
        weight_update_controller._session = session
        scheduler = MagicMock()
        scheduler.delete_workers.side_effect = [secondary_error, None]
        controller = _make_controller(scheduler)
        controller._weight_update_ctrl = weight_update_controller
        controller._service_roles = ["actor"]

        try:
            with pytest.raises(KeyboardInterrupt) as exc_info:
                controller.destroy()

            assert exc_info.value is primary_error
            assert any(
                "RuntimeError: worker cleanup failed" in note
                for note in getattr(primary_error, "__notes__", [])
            )
            scheduler.delete_workers.assert_called_once_with(role="actor")
            assert controller._service_roles == ["actor"]
            assert controller._weight_update_ctrl is weight_update_controller

            controller.destroy()

            assert session.close_count == 2
            assert controller._weight_update_ctrl is None
            assert controller._service_roles == []
        finally:
            if controller._weight_update_ctrl is not None:
                controller.destroy()

    def test_connect_engine_failure_releases_weight_update_controller(
        self, monkeypatch
    ):
        from areal.v2.weight_update.controller import controller as wu_module

        class FailingWeightUpdateController:
            def __init__(self, _config):
                self.initialized = False
                self.destroyed = False

            def initialize(self):
                self.initialized = True

            def connect(self, **_kwargs):
                raise RuntimeError("connect failed")

            def destroy(self):
                self.destroyed = True

        resource = FailingWeightUpdateController(None)
        monkeypatch.setattr(
            wu_module,
            "WeightUpdateController",
            lambda _config: resource,
        )

        controller = _make_controller()

        with pytest.raises(RuntimeError, match="connect failed"):
            controller.connect_engine(_make_rollout(), _disk_meta())

        assert resource.initialized is True
        assert resource.destroyed is True
        assert controller._weight_update_ctrl is None

    @pytest.mark.parametrize(
        "cleanup_error",
        [
            pytest.param(RuntimeError("cleanup failed"), id="runtime-error"),
            pytest.param(
                KeyboardInterrupt("cleanup interrupted"), id="keyboard-interrupt"
            ),
        ],
    )
    def test_connect_engine_preserves_primary_error_and_failed_cleanup_owner(
        self, monkeypatch, cleanup_error
    ):
        from areal.v2.weight_update.controller import controller as wu_module

        primary_error = RuntimeError("connect failed")

        class FailingWeightUpdateController:
            def initialize(self):
                return None

            def connect(self, **_kwargs):
                raise primary_error

            def destroy(self):
                raise cleanup_error

        resource = FailingWeightUpdateController()
        monkeypatch.setattr(
            wu_module,
            "WeightUpdateController",
            lambda _config: resource,
        )
        controller = _make_controller()

        observed_error: BaseException | None = None
        try:
            controller.connect_engine(_make_rollout(), _disk_meta())
        except BaseException as exc:
            observed_error = exc

        assert observed_error is primary_error
        assert any(
            type(cleanup_error).__name__ in note
            for note in getattr(primary_error, "__notes__", [])
        )
        assert controller._weight_update_ctrl is resource

    def test_connect_engine_rejects_replacing_existing_owner(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as wu_module

        class ConnectedWeightUpdateController:
            def initialize(self):
                return None

            def connect(self, **_kwargs):
                return None

            def destroy(self):
                return None

        resources: list[ConnectedWeightUpdateController] = []

        def create_resource(_config):
            resource = ConnectedWeightUpdateController()
            resources.append(resource)
            return resource

        monkeypatch.setattr(wu_module, "WeightUpdateController", create_resource)
        controller = _make_controller()
        rollout = _make_rollout()
        meta = _disk_meta()

        controller.connect_engine(rollout, meta)

        with pytest.raises(RuntimeError, match="already connected"):
            controller.connect_engine(rollout, meta)

        assert len(resources) == 1
        assert controller._weight_update_ctrl is resources[0]

    def test_connect_engine_rejects_after_destroy_returns(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as wu_module

        resources_created = 0

        def create_resource(_config):
            nonlocal resources_created
            resources_created += 1
            return MagicMock()

        monkeypatch.setattr(wu_module, "WeightUpdateController", create_resource)
        controller = _make_controller()

        controller.destroy()

        with pytest.raises(RuntimeError, match="shutdown was requested"):
            controller.connect_engine(_make_rollout(), _disk_meta())

        assert resources_created == 0
        assert controller._weight_update_ctrl is None

    def test_destroy_cannot_return_before_inflight_connect_is_owned(self, monkeypatch):
        from areal.v2.weight_update.controller import controller as wu_module

        connect_entered = threading.Event()
        allow_connect = threading.Event()
        worker_cleanup_entered = threading.Event()

        class BlockingWeightUpdateController:
            def __init__(self):
                self.destroy_count = 0

            def initialize(self):
                return None

            def connect(self, **_kwargs):
                connect_entered.set()
                if not allow_connect.wait(timeout=2.0):
                    raise TimeoutError("test did not release connect")

            def destroy(self):
                self.destroy_count += 1

        resource = BlockingWeightUpdateController()
        monkeypatch.setattr(
            wu_module,
            "WeightUpdateController",
            lambda _config: resource,
        )
        scheduler = MagicMock()
        scheduler.delete_workers.side_effect = (
            lambda **_kwargs: worker_cleanup_entered.set()
        )
        controller = _make_controller(scheduler)
        controller._service_roles = ["actor"]
        rollout = _make_rollout()
        meta = _disk_meta()
        connect_errors: list[BaseException] = []
        destroy_errors: list[BaseException] = []

        def connect() -> None:
            try:
                controller.connect_engine(rollout, meta)
            except BaseException as exc:
                connect_errors.append(exc)

        def destroy() -> None:
            try:
                controller.destroy()
            except BaseException as exc:
                destroy_errors.append(exc)

        connect_thread = threading.Thread(target=connect)
        destroy_thread = threading.Thread(target=destroy)
        connect_thread.start()
        assert connect_entered.wait(timeout=1.0)
        destroy_thread.start()
        assert controller._shutdown_requested.wait(timeout=1.0)
        destroy_advanced_past_wu = worker_cleanup_entered.wait(timeout=0.2)
        allow_connect.set()
        connect_thread.join(timeout=2.0)
        destroy_thread.join(timeout=2.0)

        assert not destroy_advanced_past_wu
        assert not connect_thread.is_alive()
        assert not destroy_thread.is_alive()
        assert connect_errors == []
        assert destroy_errors == []
        assert resource.destroy_count == 1
        assert controller._weight_update_ctrl is None
