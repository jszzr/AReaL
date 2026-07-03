"""Tests for RolloutControllerV2."""

from __future__ import annotations

import asyncio
import re
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.utils import stats_tracker
from areal.v2.inference_service.controller import workflow as workflow_module
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.v2.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
    validate_trajectory_policy_version,
)
from areal.v2.inference_service.data_proxy.session import TrajectoryDeliveryMode
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER


def _make_scheduler(n_gpus_per_node: int = 8) -> MagicMock:
    scheduler = MagicMock()
    scheduler.n_gpus_per_node = n_gpus_per_node
    return scheduler


# =============================================================================
# InferenceEngineConfig
# =============================================================================


class TestInferenceEngineConfigForInferenceService:
    def test_defaults(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.admin_api_key == "areal-admin-key"
        assert cfg.model == "default"
        assert cfg.consumer_batch_size == 1
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False
        assert cfg.agent is not None
        assert (
            cfg.agent.agent_cls_path
            == "areal.experimental.openai.proxy.online_agent._OnlineAgent"
        )

    def test_custom_values(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="custom-key",
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=3.0,
            ),
        )
        assert cfg.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5
        assert cfg.agent is not None
        assert cfg.agent.set_reward_finish_timeout == 3.0

    def test_scheduling_fields(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            request_timeout=60.0,
            setup_timeout=600.0,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.setup_timeout == 600.0

    def test_dump_to_file_defaults_to_false(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.dump_to_file is False


# =============================================================================
# RolloutControllerV2 — workflow resolution helpers
# =============================================================================


class TestControllerWorkflowResolution:
    def test_resolve_workflow_with_instance(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        with pytest.raises(TypeError, match=r"callable run\(\) method"):
            controller._resolve_workflow(12345)

    def test_resolve_workflow_none_creates_online_inference_service_workflow(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        resolved = controller._resolve_workflow(
            None,
            workflow_kwargs={"timeout": 3.0},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.controller is controller
        assert resolved.agent is None
        assert resolved.timeout == 3.0

    def test_online_workflow_defaults_to_finite_request_timeout(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
            request_timeout=47.0,
        )
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        controller._gateway_addr = "http://test:8080"

        resolved = controller._resolve_workflow(None, workflow_kwargs={})

        assert resolved.timeout == 47.0

    def test_resolve_workflow_agent_class_creates_offline_workflow(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert isinstance(resolved.agent, MockAgent)

    def test_resolve_should_accept_fn_none(self):
        assert RolloutControllerV2._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert RolloutControllerV2._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in InferenceServiceWorkflow."""
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert hasattr(resolved, "arun_episode")

    def test_resolve_workflow_agent_class_without_gateway_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        with pytest.raises(ValueError, match="Gateway address is unavailable"):
            controller._resolve_workflow(MockAgent, workflow_kwargs={})

    def test_resolve_workflow_rollout_workflow_instance_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow instances are not supported",
        ):
            controller._resolve_workflow(workflow)

    def test_resolve_workflow_rollout_workflow_class_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow classes are not supported",
        ):
            controller._resolve_workflow(
                "areal.v2.inference_service.controller.workflow.InferenceServiceWorkflow"
            )


class TestSubmitPolicyVersionSnapshot:
    @staticmethod
    def _make_controller(version: int = 7) -> RolloutControllerV2:
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"
        controller._workflow_executor = MagicMock()
        controller._version = version
        return controller

    def test_eval_submit_snapshots_current_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(data={}, workflow=None, is_eval=True)

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        controller._version = 8
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.expected_policy_version == 7

    def test_training_submit_has_no_expected_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(data={}, workflow=None, is_eval=False)

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        assert resolved.expected_policy_version is None

    def test_workflow_kwargs_cannot_spoof_expected_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(
            data={},
            workflow=None,
            workflow_kwargs={"expected_policy_version": 999},
            is_eval=True,
        )

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        assert resolved.expected_policy_version == 7


# =============================================================================
# RolloutControllerV2 — API surface
# =============================================================================


class TestRolloutControllerV2APISurface:
    def test_has_all_public_methods(self):
        methods = [
            "initialize",
            "destroy",
            "submit",
            "wait",
            "rollout_batch",
            "prepare_batch",
            "chat_completion",
            "set_version",
            "get_version",
            "get_capacity",
            "pause",
            "resume",
            "export_stats",
            "pause_generation",
            "continue_generation",
            "config_perf_tracer",
            "save_perf_tracer",
        ]
        for m in methods:
            assert hasattr(RolloutControllerV2, m), f"Missing method: {m}"

    def test_has_properties(self):
        properties = [
            "staleness_manager",
            "workflow_executor",
            "proxy_gateway_addr",
            "worker_ids",
        ]
        for p in properties:
            assert hasattr(RolloutControllerV2, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        """RolloutControllerV2 must NOT be a subclass of RolloutController."""
        # Verify it doesn't inherit from any class except object
        bases = RolloutControllerV2.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# RolloutControllerV2 — construction + state
# =============================================================================


class TestRolloutControllerV2Construction:
    def test_admin_api_key_none_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        cfg.admin_api_key = ""
        with pytest.raises(ValueError, match="admin_api_key must be set"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_model_empty_raises(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1", admin_api_key="test-key", model=""
        )
        with pytest.raises(ValueError, match="model must not be empty"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_constructor(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None
        assert controller._worker_ids == {}
        assert controller._desired_worker_ids == {}
        assert controller._predecessor_worker_ids == {}
        assert controller.worker_ids == {}

    def test_admin_api_key_defaults(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        assert controller.config.admin_api_key == "test-key"

    def test_version_management_without_services(self):
        """set_version / get_version work even without gateway services."""
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_export_stats_drains_local_workflow_metrics(self):
        stats_tracker.export_all(reset=True)
        try:
            stats_tracker.get("rollout").scalar(reward=0.75)
            controller = RolloutControllerV2(
                config=InferenceEngineConfig(
                    backend="sglang:d1", admin_api_key="test-key"
                ),
                scheduler=MagicMock(n_gpus_per_node=8),
            )

            assert controller.export_stats() == {
                "rollout/reward": 0.75,
                "rollout/reward__count": 1,
            }
            assert controller.export_stats() == {}
        finally:
            stats_tracker.export_all(reset=True)

    def test_proxy_gateway_addr(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Before initialize, proxy_gateway_addr returns the empty _gateway_addr
        assert controller.proxy_gateway_addr == ""

    def test_callback_addr_formats_ipv6_hostport(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "2001:db8::10"
        controller._callback_port = 19000

        assert controller.callback_addr == "[2001:db8::10]:19000"

    def test_workflow_executor_raises_before_init(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller.workflow_executor

    def test_config_perf_tracer_is_noop(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.config_perf_tracer()
        controller.save_perf_tracer()

    @pytest.mark.asyncio
    async def test_async_initialize_passes_callback_reward_timeout_and_worker_identity(
        self,
    ):
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker = MagicMock()
        worker.ip = "127.0.0.1"
        worker.worker_ports = [18000]

        scheduler = MagicMock(n_gpus_per_node=8)
        scheduler.get_workers.return_value = [worker]

        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            tokenizer_path="mock-tokenizer",
            request_timeout=15.0,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=7.5,
            ),
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=1,
                    cmd="python -m areal.v2.inference_service.guard",
                ),
            ),
            admin_api_key="test-admin-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000
        borrowed_server_info = LocalInfServerInfo(
            host="127.0.0.1", port=30000, process=MagicMock()
        )
        server_infos = [borrowed_server_info]

        with (
            patch.object(controller, "_async_fork_on_guard") as mock_fork,
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                return_value="epoch-e1",
            ),
        ):
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=server_infos,
            )

        assert controller.server_infos == server_infos
        assert controller.server_infos is not server_infos
        controller.server_infos.clear()
        assert server_infos == [borrowed_server_info]

        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1
        data_proxy_cmd = data_proxy_calls[0].kwargs["raw_cmd"]
        assert "--set-reward-finish-timeout" in data_proxy_cmd
        assert "7.5" in data_proxy_cmd
        assert "--callback-server-addr" in data_proxy_cmd
        assert "http://127.0.0.1:19000" in data_proxy_cmd
        assert "--worker-id" in data_proxy_cmd
        assert data_proxy_cmd[data_proxy_cmd.index("--worker-id") + 1] == "epoch-e1"

        response = MagicMock()
        response.json.return_value = {"worker_id": "epoch-e1"}
        client = MagicMock()
        client.post.return_value = response
        with patch(
            "areal.v2.inference_service.controller.controller.httpx.Client"
        ) as client_cls:
            client_cls.return_value.__enter__.return_value = client
            controller._register_data_proxies_in_router()

        assert client.post.call_args.kwargs["json"]["worker_id"] == "epoch-e1"


class TestControllerDirectControlIdentity:
    @staticmethod
    def _controller() -> RolloutControllerV2:
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._data_proxy_addrs = ["http://data-proxy:18081"]
        controller._worker_ids = {"http://data-proxy:18081": "epoch-e2"}
        return controller

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "args", "endpoint", "payload"),
        [
            ("_async_set_version", (7,), "/set_version", {"version": 7}),
            (
                "_async_offload",
                (),
                "/release_memory_occupation",
                {},
            ),
            (
                "_async_onload",
                (["weights"],),
                "/resume_memory_occupation",
                {"tags": ["weights"]},
            ),
            ("_async_pause_generation", (), "/pause_generation", {}),
            ("_async_continue_generation", (), "/continue_generation", {}),
        ],
    )
    async def test_control_request_carries_confirmed_worker_identity(
        self,
        method_name: str,
        args: tuple[object, ...],
        endpoint: str,
        payload: dict[str, object],
    ) -> None:
        controller = self._controller()
        response = MagicMock(status_code=200)
        client = MagicMock()
        client.post = AsyncMock(return_value=response)

        with patch.object(controller, "_get_async_client", return_value=client):
            await getattr(controller, method_name)(*args)

        client.post.assert_awaited_once_with(
            f"http://data-proxy:18081{endpoint}",
            json=payload,
            headers={WORKER_ID_HEADER: "epoch-e2"},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("_async_set_version", (7,)),
            ("_async_offload", ()),
            ("_async_onload", (None,)),
            ("_async_pause_generation", ()),
            ("_async_continue_generation", ()),
        ],
    )
    async def test_control_request_without_confirmed_identity_fails_closed(
        self,
        method_name: str,
        args: tuple[object, ...],
    ) -> None:
        controller = self._controller()
        controller._worker_ids.clear()

        with (
            patch.object(controller, "_get_async_client") as get_client,
            pytest.raises(RuntimeError, match="router-confirmed worker identity"),
        ):
            await getattr(controller, method_name)(*args)

        get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_control_request_uses_one_atomic_identity_snapshot(self) -> None:
        controller = self._controller()
        controller._data_proxy_addrs.append("http://data-proxy:18082")
        controller._worker_ids["http://data-proxy:18082"] = "epoch-e2-b"
        observed: list[tuple[str, str]] = []

        async def capture(
            addr: str,
            worker_id: str,
            _endpoint: str,
            _payload: dict[str, object],
        ) -> None:
            observed.append((addr, worker_id))
            if addr == "http://data-proxy:18081":
                with controller._registration_state_lock:
                    controller._worker_ids["http://data-proxy:18082"] = "epoch-e3-b"

        with patch.object(controller, "_async_data_proxy_post", side_effect=capture):
            await controller._async_pause_generation()

        assert observed == [
            ("http://data-proxy:18081", "epoch-e2"),
            ("http://data-proxy:18082", "epoch-e2-b"),
        ]


class TestRouterRegistrationIncarnations:
    @staticmethod
    def _controller() -> RolloutControllerV2:
        return RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )

    @staticmethod
    def _mock_registration_response(worker_id: str) -> tuple[MagicMock, MagicMock]:
        response = MagicMock()
        response.json.return_value = {"worker_id": worker_id}
        client = MagicMock()
        client.post.return_value = response
        return client, response

    def test_register_sends_caller_generated_incarnation_cas_payload(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]
        client, response = self._mock_registration_response("desired-worker-id")

        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                return_value="desired-worker-id",
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client"
            ) as client_cls,
        ):
            client_cls.return_value.__enter__.return_value = client
            controller._register_data_proxies_in_router()

        client.post.assert_called_once_with(
            "http://router:18080/register",
            json={
                "worker_addr": "http://data-proxy:18081",
                "worker_id": "desired-worker-id",
                "expected_worker_id": None,
            },
            headers={"Authorization": "Bearer test-admin-key"},
            timeout=5,
        )
        response.raise_for_status.assert_called_once_with()
        assert controller.worker_ids == {"http://data-proxy:18081": "desired-worker-id"}

    def test_register_method_retry_reuses_exact_incarnation_cas_payload(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]
        client, _ = self._mock_registration_response("desired-worker-id")

        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                return_value="desired-worker-id",
            ) as uuid4,
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client"
            ) as client_cls,
        ):
            client_cls.return_value.__enter__.return_value = client
            controller._register_data_proxies_in_router()
            controller._register_data_proxies_in_router()

        assert uuid4.call_count == 1
        assert client.post.call_count == 2
        assert (
            client.post.call_args_list[0].kwargs["json"]
            == (client.post.call_args_list[1].kwargs["json"])
        )
        assert client.post.call_args_list[0].kwargs["json"] == {
            "worker_addr": "http://data-proxy:18081",
            "worker_id": "desired-worker-id",
            "expected_worker_id": None,
        }

    def test_concurrent_register_calls_freeze_one_payload_and_retry_reuses_it(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]

        first_uuid_entered = threading.Event()
        second_register_started = threading.Event()
        second_uuid_entered = threading.Event()
        uuid_calls: list[str] = []
        uuid_calls_lock = threading.Lock()
        payloads: list[dict[str, object]] = []
        payloads_lock = threading.Lock()
        errors: list[BaseException] = []

        def generate_worker_id() -> str:
            with uuid_calls_lock:
                worker_id = f"desired-worker-{len(uuid_calls)}"
                uuid_calls.append(worker_id)
                call_index = len(uuid_calls) - 1
            if call_index == 0:
                first_uuid_entered.set()
                assert second_register_started.wait(timeout=2.0)
                # A correct registration-state lock keeps the second caller out;
                # the buggy check-then-set reaches uuid4() a second time.
                second_uuid_entered.wait(timeout=0.5)
            else:
                second_uuid_entered.set()
            return worker_id

        class RegistrationClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def post(
                self,
                _url: str,
                *,
                json: dict[str, object],
                **_kwargs: object,
            ) -> MagicMock:
                with payloads_lock:
                    payloads.append(dict(json))
                response = MagicMock()
                response.json.return_value = {"worker_id": json["worker_id"]}
                return response

        def register() -> None:
            try:
                controller._register_data_proxies_in_router()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        first = threading.Thread(target=register)

        def register_second() -> None:
            second_register_started.set()
            register()

        second = threading.Thread(target=register_second)

        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                side_effect=generate_worker_id,
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client",
                side_effect=RegistrationClient,
            ),
        ):
            first.start()
            assert first_uuid_entered.wait(timeout=2.0)
            second.start()
            first.join(timeout=3.0)
            second.join(timeout=3.0)
            assert not first.is_alive()
            assert not second.is_alive()
            controller._register_data_proxies_in_router()

        assert errors == []
        assert uuid_calls == ["desired-worker-0"]
        assert len(payloads) == 3
        assert payloads[0] == payloads[1] == payloads[2]
        assert controller.worker_ids == {"http://data-proxy:18081": "desired-worker-0"}

    def test_successor_launch_waits_for_in_flight_registration_commit(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]

        first_post_entered = threading.Event()
        release_first_response = threading.Event()
        successor_record_started = threading.Event()
        successor_record_finished = threading.Event()
        payloads: list[dict[str, object]] = []
        payloads_lock = threading.Lock()
        errors: list[BaseException] = []

        class RegistrationClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def post(
                self,
                _url: str,
                *,
                json: dict[str, object],
                **_kwargs: object,
            ) -> MagicMock:
                with payloads_lock:
                    payloads.append(dict(json))
                if json["worker_id"] == "incarnation-e1":
                    first_post_entered.set()
                    assert release_first_response.wait(timeout=3.0)
                response = MagicMock()
                response.json.return_value = {"worker_id": json["worker_id"]}
                return response

        def register() -> None:
            try:
                controller._register_data_proxies_in_router()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def record_successor() -> None:
            successor_record_started.set()
            try:
                controller._record_data_proxy_launch("http://data-proxy:18081")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                successor_record_finished.set()

        register_thread = threading.Thread(target=register)
        successor_thread = threading.Thread(target=record_successor)
        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                side_effect=["incarnation-e1", "incarnation-e2"],
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client",
                side_effect=RegistrationClient,
            ),
        ):
            register_thread.start()
            assert first_post_entered.wait(timeout=2.0)
            successor_thread.start()
            assert successor_record_started.wait(timeout=2.0)
            successor_finished_before_commit = successor_record_finished.wait(
                timeout=0.5
            )
            release_first_response.set()
            register_thread.join(timeout=3.0)
            successor_thread.join(timeout=3.0)
            controller._register_data_proxies_in_router()

        assert not register_thread.is_alive()
        assert not successor_thread.is_alive()
        assert errors == []
        assert not successor_finished_before_commit
        assert controller._data_proxy_addrs == ["http://data-proxy:18081"]
        assert len(payloads) == 2
        assert payloads[0] == {
            "worker_addr": "http://data-proxy:18081",
            "worker_id": "incarnation-e1",
            "expected_worker_id": None,
        }
        assert all(
            payload
            == {
                "worker_addr": "http://data-proxy:18081",
                "worker_id": "incarnation-e2",
                "expected_worker_id": "incarnation-e1",
            }
            for payload in payloads[1:]
        )
        assert controller.worker_ids == {"http://data-proxy:18081": "incarnation-e2"}

    def test_new_launch_at_same_addr_uses_successor_with_active_predecessor(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._worker_ids["http://data-proxy:18081"] = "predecessor-id"
        client, _ = self._mock_registration_response("successor-id")

        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                return_value="successor-id",
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client"
            ) as client_cls,
        ):
            client_cls.return_value.__enter__.return_value = client
            controller._record_data_proxy_launch("http://data-proxy:18081")
            controller._register_data_proxies_in_router()

        assert client.post.call_args.kwargs["json"] == {
            "worker_addr": "http://data-proxy:18081",
            "worker_id": "successor-id",
            "expected_worker_id": "predecessor-id",
        }
        assert controller.worker_ids == {"http://data-proxy:18081": "successor-id"}

    def test_register_rejects_response_for_different_incarnation(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]
        client, _ = self._mock_registration_response("different-worker-id")

        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                return_value="desired-worker-id",
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client"
            ) as client_cls,
        ):
            client_cls.return_value.__enter__.return_value = client
            with pytest.raises(RuntimeError, match="different-worker-id"):
                controller._register_data_proxies_in_router()

        assert controller.worker_ids == {}

    def test_destroy_clears_incarnation_registration_state(self):
        controller = self._controller()
        controller._worker_ids["http://data-proxy:18081"] = "active-id"
        controller._desired_worker_ids["http://data-proxy:18081"] = "desired-id"
        controller._predecessor_worker_ids["http://data-proxy:18081"] = "active-id"

        controller.destroy()

        assert controller._worker_ids == {}
        assert controller._desired_worker_ids == {}
        assert controller._predecessor_worker_ids == {}

    def test_destroy_invalidates_in_flight_registration_response(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"
        controller._data_proxy_addrs = ["http://data-proxy:18081"]

        post_entered = threading.Event()
        release_response = threading.Event()
        errors: list[BaseException] = []

        class BlockingRegistrationClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def post(
                self,
                _url: str,
                *,
                json: dict[str, object],
                **_kwargs: object,
            ) -> MagicMock:
                post_entered.set()
                assert release_response.wait(timeout=3.0)
                response = MagicMock()
                response.json.return_value = {"worker_id": json["worker_id"]}
                return response

        def register() -> None:
            try:
                controller._register_data_proxies_in_router()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        thread = threading.Thread(target=register)
        with patch(
            "areal.v2.inference_service.controller.controller.httpx.Client",
            side_effect=BlockingRegistrationClient,
        ):
            thread.start()
            assert post_entered.wait(timeout=2.0)
            try:
                controller.destroy()
            finally:
                release_response.set()
                thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert errors == []
        assert controller._data_proxy_addrs == []
        assert controller._worker_ids == {}
        assert controller._desired_worker_ids == {}
        assert controller._predecessor_worker_ids == {}

    def test_launch_record_after_destroy_cannot_repopulate_registration_state(self):
        controller = self._controller()
        controller.destroy()

        with patch(
            "areal.v2.inference_service.controller.controller.uuid.uuid4"
        ) as uuid4:
            controller._record_data_proxy_launch("http://data-proxy:18081")

        uuid4.assert_not_called()
        assert controller._data_proxy_addrs == []
        assert controller._worker_ids == {}
        assert controller._desired_worker_ids == {}
        assert controller._predecessor_worker_ids == {}

    def test_launch_record_and_registration_snapshot_are_atomic(self):
        controller = self._controller()
        controller._router_addr = "http://router:18080"

        record_uuid_entered = threading.Event()
        snapshot_uuid_entered = threading.Event()
        release_record_uuid = threading.Event()
        uuid_calls: list[str] = []
        uuid_calls_lock = threading.Lock()
        payloads: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def generate_worker_id() -> str:
            with uuid_calls_lock:
                call_index = len(uuid_calls)
                worker_id = "launch-worker-id" if call_index == 0 else "snapshot-id"
                uuid_calls.append(worker_id)
            if call_index == 0:
                record_uuid_entered.set()
                assert release_record_uuid.wait(timeout=3.0)
            else:
                snapshot_uuid_entered.set()
            return worker_id

        class RegistrationClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def post(
                self,
                _url: str,
                *,
                json: dict[str, object],
                **_kwargs: object,
            ) -> MagicMock:
                payloads.append(dict(json))
                response = MagicMock()
                response.json.return_value = {"worker_id": json["worker_id"]}
                return response

        def record_launch() -> None:
            try:
                controller._record_data_proxy_launch("http://data-proxy:18081")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def register() -> None:
            try:
                controller._register_data_proxies_in_router()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        record_thread = threading.Thread(target=record_launch)
        register_thread = threading.Thread(target=register)
        with (
            patch(
                "areal.v2.inference_service.controller.controller.uuid.uuid4",
                side_effect=generate_worker_id,
            ),
            patch(
                "areal.v2.inference_service.controller.controller.httpx.Client",
                side_effect=RegistrationClient,
            ),
        ):
            record_thread.start()
            assert record_uuid_entered.wait(timeout=2.0)
            register_thread.start()
            snapshot_raced_partial_record = snapshot_uuid_entered.wait(timeout=0.5)
            release_record_uuid.set()
            record_thread.join(timeout=3.0)
            register_thread.join(timeout=3.0)

        assert not record_thread.is_alive()
        assert not register_thread.is_alive()
        assert errors == []
        assert not snapshot_raced_partial_record
        assert uuid_calls == ["launch-worker-id"]
        assert payloads == [
            {
                "worker_addr": "http://data-proxy:18081",
                "worker_id": "launch-worker-id",
                "expected_worker_id": None,
            }
        ]
        assert controller.worker_ids == {"http://data-proxy:18081": "launch-worker-id"}
        assert controller._desired_worker_ids == {
            "http://data-proxy:18081": "launch-worker-id"
        }
        assert controller._predecessor_worker_ids == {"http://data-proxy:18081": None}


class TestOnlineCallbackFlow:
    def test_controller_has_no_unowned_completed_result_buffer(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )

        assert not hasattr(controller, "_completed_online_results")

    @pytest.mark.asyncio
    async def test_online_callback_without_matching_lease_is_rejected(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="Unknown or cancelled lease"):
            await controller._handle_online_ready_callback(
                {
                    "session_id": "agent-a",
                    "trajectory_id": 0,
                    "lease_id": "unknown",
                    "expected_version": 0,
                }
            )

    @pytest.mark.asyncio
    async def test_out_of_order_callbacks_settle_the_reserved_waiter(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._version = 3

        lease_1, version_1 = controller.reserve_online_trajectory()
        lease_2, version_2 = controller.reserve_online_trajectory()
        waiter_1 = asyncio.create_task(
            controller.wait_for_online_trajectory(lease_1, timeout=1.0)
        )
        waiter_2 = asyncio.create_task(
            controller.wait_for_online_trajectory(lease_2, timeout=1.0)
        )
        await asyncio.sleep(0)

        await controller._handle_online_ready_callback(
            {
                "session_id": "agent-b",
                "trajectory_id": 2,
                "lease_id": lease_2,
                "expected_version": version_2,
            }
        )
        await controller._handle_online_ready_callback(
            {
                "session_id": "agent-a",
                "trajectory_id": 1,
                "lease_id": lease_1,
                "expected_version": version_1,
            }
        )
        assert await waiter_1 == {
            "session_id": "agent-a",
            "trajectory_id": 1,
            "lease_id": lease_1,
            "expected_version": 3,
        }
        assert await waiter_2 == {
            "session_id": "agent-b",
            "trajectory_id": 2,
            "lease_id": lease_2,
            "expected_version": 3,
        }

    @pytest.mark.asyncio
    async def test_identical_callback_replay_is_idempotent(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        lease_id, expected_version = controller.reserve_online_trajectory()
        payload = {
            "session_id": "agent-a",
            "trajectory_id": 1,
            "lease_id": lease_id,
            "expected_version": expected_version,
        }

        first = await controller._handle_online_ready_callback(payload)
        replay = await controller._handle_online_ready_callback(payload)
        result = await controller.wait_for_online_trajectory(lease_id, timeout=1.0)

        assert first == replay
        assert result == payload

    @pytest.mark.asyncio
    async def test_ready_callback_replay_after_waiter_consumption_is_idempotent(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        lease_id, expected_version = controller.reserve_online_trajectory()
        payload = {
            "session_id": "agent-a",
            "trajectory_id": 1,
            "lease_id": lease_id,
            "expected_version": expected_version,
            "group_id": "group-a",
        }

        first = await controller._handle_online_ready_callback(payload)
        assert await controller.wait_for_online_trajectory(lease_id, timeout=1.0) == {
            **payload,
        }
        assert lease_id not in controller._online_waiters

        replay = await controller._handle_online_ready_callback(payload)
        assert replay == first

        with pytest.raises(RuntimeError, match="already settled"):
            await controller._handle_online_ready_callback(
                {**payload, "trajectory_id": 2}
            )

    @pytest.mark.asyncio
    async def test_failed_callback_replay_after_waiter_consumption_is_idempotent(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        lease_id, expected_version = controller.reserve_online_trajectory()
        payload = {
            "lease_id": lease_id,
            "expected_version": expected_version,
            "reason": "router registration failed",
        }

        first = await controller._handle_online_failed_callback(payload)
        with pytest.raises(RuntimeError, match="router registration failed"):
            await controller.wait_for_online_trajectory(lease_id, timeout=1.0)
        assert lease_id not in controller._online_waiters

        replay = await controller._handle_online_failed_callback(payload)
        assert replay == first

        with pytest.raises(RuntimeError, match="already settled"):
            await controller._handle_online_failed_callback(
                {**payload, "reason": "worker forward failed"}
            )

    @pytest.mark.asyncio
    async def test_callback_settlement_tombstones_are_bounded_in_settlement_order(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        assert controller._online_settlement_limit >= 4096
        controller._online_settlement_limit = 2

        settled: list[tuple[str, dict[str, object]]] = []
        for trajectory_id in range(3):
            lease_id, expected_version = controller.reserve_online_trajectory()
            payload: dict[str, object] = {
                "session_id": f"agent-{trajectory_id}",
                "trajectory_id": trajectory_id,
                "lease_id": lease_id,
                "expected_version": expected_version,
            }
            await controller._handle_online_ready_callback(payload)
            await controller.wait_for_online_trajectory(lease_id, timeout=1.0)
            settled.append((lease_id, payload))

        assert list(controller._online_settlements) == [
            settled[1][0],
            settled[2][0],
        ]
        with pytest.raises(RuntimeError, match="Unknown or cancelled lease"):
            await controller._handle_online_ready_callback(settled[0][1])
        assert (await controller._handle_online_ready_callback(settled[1][1]))[
            "status"
        ] == "ok"

    @pytest.mark.asyncio
    async def test_destroy_clears_callback_settlement_tombstones(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        lease_id, expected_version = controller.reserve_online_trajectory()
        await controller._handle_online_ready_callback(
            {
                "session_id": "agent-a",
                "trajectory_id": 1,
                "lease_id": lease_id,
                "expected_version": expected_version,
            }
        )
        await controller.wait_for_online_trajectory(lease_id, timeout=1.0)
        assert controller._online_settlements

        controller.destroy()

        assert not controller._online_settlements

    @pytest.mark.asyncio
    async def test_wrong_version_callback_keeps_reserved_waiter_pending(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        lease_id, expected_version = controller.reserve_online_trajectory()
        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(lease_id, timeout=1.0)
        )
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="expects policy version"):
            await controller._handle_online_ready_callback(
                {
                    "session_id": "agent-a",
                    "trajectory_id": 0,
                    "lease_id": lease_id,
                    "expected_version": expected_version + 1,
                }
            )
        assert not waiter_task.done()
        waiter_task.cancel()

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_accept_late_callback(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        lease_id, expected_version = controller.reserve_online_trajectory()
        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(lease_id, timeout=1.0)
        )
        await asyncio.sleep(0)
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        with pytest.raises(RuntimeError, match="Unknown or cancelled lease"):
            await controller._handle_online_ready_callback(
                {
                    "session_id": "agent-a",
                    "trajectory_id": 0,
                    "lease_id": lease_id,
                    "expected_version": expected_version,
                }
            )

    @pytest.mark.asyncio
    async def test_gateway_failure_rejects_the_reserved_workflow(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1", admin_api_key="test-admin-key"
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        lease_id, expected_version = controller.reserve_online_trajectory()
        waiter = asyncio.create_task(
            controller.wait_for_online_trajectory(lease_id, timeout=1.0)
        )
        await asyncio.sleep(0)

        result = await controller._handle_online_failed_callback(
            {
                "lease_id": lease_id,
                "expected_version": expected_version,
                "reason": "router registration failed",
            }
        )

        assert result["status"] == "ok"
        with pytest.raises(RuntimeError, match="router registration failed"):
            await waiter


class TestInferenceServiceWorkflow:
    @staticmethod
    def _versioned_trajectory(
        policy_version: int, reward: float = 1.25
    ) -> dict[str, torch.Tensor]:
        return {
            "rewards": torch.tensor([0.0, reward]),
            "versions": torch.tensor([-1, policy_version], dtype=torch.int32),
            "loss_mask": torch.tensor([0, 1], dtype=torch.int32),
        }

    @staticmethod
    def _versioned_rtensor_trajectory(
        policy_version: int, reward: float = 1.25
    ) -> dict[str, RTensor]:
        node_addr = "storage.test:9999"
        return {
            "rewards": RTensor(
                shard=TensorShardInfo(shard_id="rewards", node_addr=node_addr),
                data=torch.tensor([0.0, reward]),
            ),
            "versions": RTensor(
                shard=TensorShardInfo(shard_id="versions", node_addr=node_addr),
                data=torch.tensor([-1, policy_version], dtype=torch.int32),
            ),
            "loss_mask": RTensor(
                shard=TensorShardInfo(shard_id="loss-mask", node_addr=node_addr),
                data=torch.tensor([0, 1], dtype=torch.int32),
            ),
        }

    @pytest.mark.asyncio
    async def test_grant_client_error_is_not_retried(self):
        controller = MagicMock(callback_addr="127.0.0.1:19000")
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            timeout=5.0,
        )
        response = MagicMock(status=409)
        response.text = AsyncMock(return_value="conflicting replay")
        response.raise_for_status = MagicMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=context)

        with pytest.raises(ValueError, match="HTTP 409"):
            await workflow._grant_online_lease(session, "lease-1", 0)

        session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_serializes_pull_delivery(self):
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "group_id": "grp",
                "sessions": [{"session_id": "s", "session_api_key": "k"}],
            }
        )
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http_session = MagicMock()
        mock_http_session.post = MagicMock(return_value=mock_cm)

        with patch(
            "areal.v2.inference_service.controller.workflow.uuid.uuid4",
            return_value="request-1",
        ):
            result = await workflow._start_session(
                mock_http_session,
                "42",
                group_size=1,
                delivery_mode=TrajectoryDeliveryMode.PULL,
            )

        assert result == ("grp", [("s", "k")])
        mock_http_session.post.assert_called_once_with(
            "http://test:8080/rl/start_session",
            json={
                "task_id": "42",
                "request_id": "request-1",
                "group_size": 1,
                "delivery_mode": "pull",
            },
            headers={"Authorization": "Bearer test-key"},
        )

    @pytest.mark.asyncio
    async def test_export_uses_stable_caller_generated_request_id(self):
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"traj": {"value": 1}})
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http_session = MagicMock()
        mock_http_session.post = MagicMock(return_value=mock_cm)

        with patch(
            "areal.v2.inference_service.controller.workflow.uuid.uuid4",
            return_value="export-request-1",
        ):
            result = await workflow._export_interactions(
                mock_http_session,
                ["session-1"],
                group_id="group-1",
                trajectory_id=3,
            )

        assert result == {"value": 1}
        mock_http_session.post.assert_called_once_with(
            "http://test:8080/export_trajectories",
            json={
                "request_id": "export-request-1",
                "session_ids": ["session-1"],
                "group_id": "group-1",
                "trajectory_id": 3,
                "discount": 1.0,
                "style": "individual",
                "remove_session": True,
            },
            headers={"Authorization": "Bearer test-key"},
        )

    @pytest.mark.asyncio
    async def test_online_mode_reserves_waiter_before_publishing_lease(self):
        events: list[str] = []
        controller = MagicMock()
        controller.reserve_online_trajectory = MagicMock(
            side_effect=lambda: (events.append("reserve") or ("lease-1", 4))
        )
        controller.wait_for_online_trajectory = AsyncMock(
            side_effect=lambda *args, **kwargs: (
                events.append("wait")
                or {
                    "session_id": "sess-1",
                    "trajectory_id": 7,
                    "lease_id": "lease-1",
                    "expected_version": 4,
                }
            )
        )

        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            timeout=3.0,
        )
        workflow._grant_online_lease = AsyncMock(
            side_effect=lambda *args, **kwargs: events.append("grant")
        )
        workflow._cancel_online_lease = AsyncMock()
        workflow._export_interactions = AsyncMock(
            return_value={
                "rewards": torch.tensor([0.0, 1.0]),
                "versions": torch.tensor([-1, 4], dtype=torch.int32),
                "loss_mask": torch.tensor([0, 1], dtype=torch.int32),
            }
        )

        with patch(
            "areal.v2.inference_service.controller.workflow.stats_tracker"
        ) as mock_st:
            mock_st.get.return_value = MagicMock()
            result = await workflow._run_online(AsyncMock())

        assert result is not None
        assert events == ["reserve", "grant", "wait"]
        controller.wait_for_online_trajectory.assert_awaited_once_with(
            "lease-1", timeout=3.0
        )
        workflow._cancel_online_lease.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_online_mode_cancels_remote_lease_when_wait_times_out(self):
        controller = MagicMock()
        controller.reserve_online_trajectory.return_value = ("lease-1", 4)
        controller.wait_for_online_trajectory = AsyncMock(
            side_effect=TimeoutError("producer did not finish")
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._grant_online_lease = AsyncMock()
        workflow._cancel_online_lease = AsyncMock()

        with pytest.raises(TimeoutError, match="producer did not finish"):
            await workflow._run_online(AsyncMock())

        workflow._cancel_online_lease.assert_awaited_once()
        controller.cancel_online_trajectory.assert_called_once_with("lease-1")

    @pytest.mark.asyncio
    async def test_external_online_mode_accepts_interaction_only_provenance(self):
        controller = MagicMock()
        controller.external_mode = True
        controller.reserve_online_trajectory.return_value = ("lease-external", 0)
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={
                "session_id": "external-session",
                "trajectory_id": 0,
                "lease_id": "lease-external",
                "expected_version": 0,
            }
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._grant_online_lease = AsyncMock()
        workflow._cancel_online_lease = AsyncMock()
        workflow._export_interactions = AsyncMock(
            return_value={"interactions": [{"reward": 1.0}]}
        )

        with patch(
            "areal.v2.inference_service.controller.workflow.stats_tracker"
        ) as mock_st:
            mock_st.get.return_value = MagicMock()
            result = await workflow._run_online(AsyncMock())

        assert result == {"interactions": [{"reward": 1.0}]}
        workflow._cancel_online_lease.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_offline_mode_runs_agent(self):
        controller = MagicMock()

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        mock_interaction = MagicMock(reward=1.0)
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._start_session = AsyncMock(
            return_value=("grp-test-1", [("sess-1", "sess-api-key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": mock_interaction}
        )

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        workflow._start_session.assert_awaited_once_with(
            mock_http_session,
            "42",
            group_size=1,
            delivery_mode=TrajectoryDeliveryMode.PULL,
        )
        workflow._set_last_reward.assert_awaited_once()
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session, ["sess-1"], group_id="grp-test-1"
        )

    @pytest.mark.asyncio
    async def test_offline_mode_records_reward_with_expected_policy_version(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.25

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        to_thread = AsyncMock(side_effect=lambda fn, *args: fn(*args))
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(workflow_module.asyncio, "to_thread", to_thread),
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_offline(MagicMock(), {})

        assert result is not None
        to_thread.assert_awaited_once_with(
            workflow_module.validate_trajectory_policy_version,
            traj,
            7,
        )
        clear_node.assert_not_awaited()
        tracker.scalar.assert_called_once_with(reward=1.25, policy_version=7)

    @pytest.mark.asyncio
    async def test_offline_mode_rejects_version_before_recording_metrics(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.25

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(6)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            with pytest.raises(ValueError, match="expected policy version 7"):
                await workflow._run_offline(MagicMock(), {})

        tracker.scalar.assert_not_called()
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_offline_mode_clears_exported_trajectory_when_group_fails(self):
        class FailingAgent:
            async def run(self, data, **kwargs):
                raise RuntimeError("agent failed")

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=FailingAgent(),
            gateway_addr="http://test:8080",
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())

            result = await workflow._run_offline(MagicMock(), {})

        assert result is None
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_offline_mode_without_expectation_accepts_missing_versions(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.25

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            expected_policy_version=None,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(return_value={"input_ids": []})
        tracker = MagicMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_offline(MagicMock(), {})

        assert result is not None
        tracker.scalar.assert_called_once_with(reward=1.25)

    @pytest.mark.asyncio
    async def test_online_mode_records_reward_with_expected_policy_version(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        to_thread = AsyncMock(side_effect=lambda fn, *args: fn(*args))
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(workflow_module.asyncio, "to_thread", to_thread),
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_online(MagicMock())

        assert result is not None
        to_thread.assert_awaited_once_with(
            workflow_module.validate_trajectory_policy_version,
            traj,
            7,
        )
        clear_node.assert_not_awaited()
        tracker.scalar.assert_called_once_with(reward=1.25, policy_version=7)

    @pytest.mark.asyncio
    async def test_online_mode_rejects_version_before_recording_metrics(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        traj = self._versioned_rtensor_trajectory(6)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            with pytest.raises(ValueError, match="expected policy version 7"):
                await workflow._run_online(MagicMock())

        tracker.scalar.assert_not_called()
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_online_mode_clears_exported_trajectory_when_reward_is_missing(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )
        traj = {
            "input_ids": RTensor(
                shard=TensorShardInfo(
                    shard_id="input-ids", node_addr="storage.test:9999"
                ),
                data=torch.tensor([1, 2], dtype=torch.int64),
            )
        }
        workflow._export_interactions = AsyncMock(return_value=traj)
        clear_node = AsyncMock()

        with patch.object(RTensor, "clear_node", clear_node):
            result = await workflow._run_online(MagicMock())

        assert result is None
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["input-ids"],
        )

    @pytest.mark.asyncio
    async def test_online_mode_without_expectation_accepts_missing_versions(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=None,
        )
        workflow._export_interactions = AsyncMock(
            return_value={"rewards": torch.tensor([0.0, 1.25])}
        )
        tracker = MagicMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_online(MagicMock())

        assert result is not None
        tracker.scalar.assert_called_once_with(reward=1.25)


class TestValidateTrajectoryPolicyVersion:
    @staticmethod
    def _trajectory(
        versions: list[int], loss_mask: list[int]
    ) -> dict[str, torch.Tensor]:
        return {
            "versions": torch.tensor(versions, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
        }

    def test_validate_trajectory_policy_version_accepts_expected_loss_tokens(self):
        traj = self._trajectory(
            versions=[-1, -1, 7, 7],
            loss_mask=[0, 0, 1, 1],
        )

        workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_ignores_untrained_positions(self):
        traj = self._trajectory(
            versions=[999, -1, 7, 123],
            loss_mask=[0, 0, 1, 0],
        )

        workflow_module.validate_trajectory_policy_version(traj, 7)

    @pytest.mark.parametrize(
        "versions, observed",
        [
            ([-1, 7, 6], "[6, 7]"),
            ([-1, 6, 6], "[6]"),
        ],
        ids=["mixed", "all-stale"],
    )
    def test_validate_trajectory_policy_version_rejects_mixed_or_stale_versions(
        self, versions: list[int], observed: str
    ):
        traj = self._trajectory(versions=versions, loss_mask=[0, 1, 1])

        with pytest.raises(
            ValueError,
            match=rf"expected policy version 7.*observed {re.escape(observed)}",
        ):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    @pytest.mark.parametrize("missing_key", ["versions", "loss_mask"])
    def test_validate_trajectory_policy_version_rejects_missing_field(
        self, missing_key: str
    ):
        traj = self._trajectory(versions=[-1, 7], loss_mask=[0, 1])
        del traj[missing_key]

        with pytest.raises(ValueError, match=rf"missing.*{missing_key}"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_shape_mismatch(self):
        traj = self._trajectory(versions=[-1, 7, 7], loss_mask=[0, 1])

        with pytest.raises(ValueError, match="same shape"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_floating_versions(self):
        traj = {
            "versions": torch.tensor([-1.0, 7.0], dtype=torch.float32),
            "loss_mask": torch.tensor([0, 1], dtype=torch.int32),
        }

        with pytest.raises(ValueError, match="signed integer dtype"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_no_loss_tokens(self):
        traj = self._trajectory(versions=[-1, 7], loss_mask=[0, 0])

        with pytest.raises(ValueError, match="no loss-bearing tokens"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_accepts_local_rtensors(self):
        traj = {
            "versions": RTensor(
                shard=TensorShardInfo(shard_id="versions", node_addr="unused"),
                data=torch.tensor([-1, 7, 7], dtype=torch.int32),
            ),
            "loss_mask": RTensor(
                shard=TensorShardInfo(shard_id="loss-mask", node_addr="unused"),
                data=torch.tensor([0, 1, 1], dtype=torch.int32),
            ),
        }

        with patch(
            "areal.infra.rpc.rtensor.get_backend",
            side_effect=AssertionError("local RTensor must not fetch"),
        ):
            workflow_module.validate_trajectory_policy_version(traj, 7)


class TestValidateTrajectoryPolicyVersion:
    @staticmethod
    def _trajectory(versions: list[int], loss_mask: list[int]):
        return {
            "versions": torch.tensor(versions, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
        }

    def test_accepts_expected_loss_bearing_tokens(self):
        validate_trajectory_policy_version(
            self._trajectory([999, -1, 4, 4], [0, 0, 1, 1]),
            4,
        )

    def test_rejects_stale_or_mixed_loss_bearing_tokens(self):
        with pytest.raises(ValueError, match=r"expected policy version 4.*\[3, 4\]"):
            validate_trajectory_policy_version(
                self._trajectory([-1, 4, 3], [0, 1, 1]),
                4,
            )

    @pytest.mark.parametrize("missing", ["versions", "loss_mask"])
    def test_rejects_missing_provenance(self, missing):
        trajectory = self._trajectory([-1, 4], [0, 1])
        del trajectory[missing]
        with pytest.raises(ValueError, match=missing):
            validate_trajectory_policy_version(trajectory, 4)

    def test_rejects_trajectory_without_loss_tokens(self):
        with pytest.raises(ValueError, match="no loss-bearing tokens"):
            validate_trajectory_policy_version(
                self._trajectory([-1, 4], [0, 0]),
                4,
            )


# =============================================================================
# Multi-node inference configuration
# =============================================================================


class TestMultiNodeConfig:
    def test_scheduler_zero_gpus_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 0
        with pytest.raises(ValueError, match="n_gpus_per_node must be >= 1"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=0))

    def test_gpus_not_divisible_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 3
        with pytest.raises(ValueError, match="must be divisible by n_gpus_per_node"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=3))

    def test_single_node_backward_compat(self):
        cfg = InferenceEngineConfig(backend="sglang:d2t4", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 1

    def test_multi_node_valid_config(self):
        # tp=16, n_gpus_per_node=8 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(backend="sglang:d1t16", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 2

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_worker_count(self):
        """With multi-node and pre-existing server_infos, should create dp_size workers."""
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),  # router
                ("127.0.0.1", 18082),  # data proxy (only 1, on head)
                ("127.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="10.0.0.1", port=30000, process=MagicMock()
                    ),
                ],
            )

        # With server_infos, total_workers = dp_size = 1 (not dp_size * nnodes_per_instance)
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 1

        # 3 forks: router + data-proxy + gateway (all on head worker)
        assert mock_fork.call_count == 3
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_fork_path(self):
        """Exercise the full multi-node fork path (server_infos=None)."""
        from areal.api.cli_args import SchedulingSpec

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0, worker1]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        # Track async client .post calls to /alloc_ports and /fork
        alloc_port_counter = 0
        fork_calls = []

        async def mock_async_post(url, json=None, timeout=None):
            nonlocal alloc_port_counter
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/alloc_ports" in url:
                alloc_port_counter += 1
                resp.json.return_value = {
                    "status": "success",
                    "host": url.split("//")[1].split(":")[0],
                    "ports": [30000 + alloc_port_counter],
                }
            elif "/fork" in url:
                fork_calls.append(json)
                resp.json.return_value = {"status": "success"}
            return resp

        mock_async_client = AsyncMock()
        mock_async_client.post = mock_async_post

        with (
            patch.object(
                controller, "_get_async_client", return_value=mock_async_client
            ),
            patch.object(controller, "_async_fork_on_guard") as mock_fork,
            patch.object(controller, "_async_wait_for_service"),
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            mock_fork.side_effect = [
                ("10.0.0.1", 18081),  # router
                ("10.0.0.1", 18082),  # data proxy
                ("10.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=None,
            )

        # dp_size=1, nnodes_per_instance=2: total_workers = 2
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 2

        # Async client .post calls for inf server fork:
        # 1 rendezvous alloc (nnodes_per_instance > 1) + 2 node allocs + 2 forks = 5
        assert alloc_port_counter == 3  # 1 rendezvous + 2 per-node
        assert len(fork_calls) == 2  # 1 per node in the group

        # Verify fork payloads have correct worker_index and role
        assert fork_calls[0]["role"] == "inf-server"
        assert fork_calls[0]["worker_index"] == 0
        assert fork_calls[1]["role"] == "inf-server"
        assert fork_calls[1]["worker_index"] == 1

        # Verify dist_init_addr propagated to fork commands
        for fc in fork_calls:
            cmd_str = " ".join(fc["raw_cmd"])
            assert "--dist-init-addr" in cmd_str or "--dist_init_addr" in cmd_str

        # Only 1 data proxy (dp_size=1, on head worker only)
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1
