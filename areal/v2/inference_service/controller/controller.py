# SPDX-License-Identifier: Apache-2.0

"""RolloutControllerV2 — parallel implementation to RolloutController.

Routes inference and pause/continue traffic through the gateway HTTP stack
(Gateway → Router → Data Proxy → inference backend).
All servers are launched as worker processes via the scheduler.  Inference
server processes are forked through RPCGuard (a lightweight process manager).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from areal.infra.utils.http import (
    HttpxAsyncClientCleanup,
    HttpxAsyncClientCleanupState,
    async_http_retry,
    close_httpx_client_from_sync,
    create_httpx_client,
    register_httpx_client_loop_cleanup,
)

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler, Worker

from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import LocalInfServerInfo
from areal.utils import logging, stats_tracker
from areal.utils.network import format_hostport
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER

logger = logging.getLogger("RolloutControllerV2")

_MAX_ONLINE_SETTLEMENT_TOMBSTONES = 4096
_DEFAULT_SERVICE_LOG_LEVEL = "warning"


@dataclass
class _OnlineWaiter:
    lease_id: str
    expected_version: int
    future: asyncio.Future
    delivery: dict[str, Any] | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class _OnlineSettlement:
    """Immutable callback fingerprint retained only for idempotent retries."""

    callback_kind: str
    expected_version: int
    details: tuple[Any, ...]


@dataclass(frozen=True)
class _DataProxyRegistrationIntent:
    data_proxy_addr: str
    desired_worker_id: str
    expected_worker_id: str | None


@dataclass(frozen=True)
class _DataProxyRegistrationSnapshot:
    generation: int
    intents: tuple[_DataProxyRegistrationIntent, ...]


class _OnlineLeaseConflict(RuntimeError):
    """A callback does not match a live version-bound online lease."""


class _DummyDataLoader:
    """Minimal dataloader that yields a single batch of empty dicts.

    Used by :meth:`RolloutControllerV2.prepare_batch` when
    ``dataloader`` is ``None`` (online-agent mode).
    """

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __iter__(self):
        yield [{} for _ in range(self.batch_size)]


class RolloutControllerV2:
    """Inference controller that routes everything through the gateway HTTP stack.

    This is a **parallel** implementation to ``RolloutController`` (NOT a
    subclass).  It is duck-type compatible: the trainer can use either one
    without code changes.

    All servers (inference backend, Router, Data Proxy, Gateway) are launched
    as worker sub-processes via the scheduler.  The controller talks to them
    directly over HTTP — no engine creation or RPC calls on workers.

    The inference backend is determined from ``config.backend``
    (``"sglang"`` and ``"vllm"`` are supported).
    """

    # Worker role suffix for RPCGuard workers
    _INF_SUFFIX = "-inf"

    def __init__(
        self,
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ) -> None:
        if config.admin_api_key is None or not config.admin_api_key.strip():
            raise ValueError(
                "InferenceEngineConfig.admin_api_key must be set (not None or empty)"
            )
        if not config.model:
            raise ValueError("InferenceEngineConfig.model must not be empty")
        if config.use_lora and (
            not isinstance(config.lora_name, str) or not config.lora_name.strip()
        ):
            raise ValueError(
                "InferenceEngineConfig.lora_name must be set when use_lora=True"
            )
        self.config = config
        self.scheduler = scheduler

        if config.api_url is not None:
            self.rollout_alloc = None
        else:
            from areal.api.alloc_mode import ModelAllocation

            self.rollout_alloc = ModelAllocation.from_str(config.backend)

        # Multi-node: derive nnodes_per_instance from scheduler's n_gpus_per_node.
        # External mode has no local inference servers, so always single-node.
        if self.rollout_alloc is None:
            nnodes_per_instance = 1
        else:
            total_gpus = (
                self.rollout_alloc.parallel.tp_size
                * self.rollout_alloc.parallel.pp_size
            )
            n_gpus_per_node = self.scheduler.n_gpus_per_node
            if n_gpus_per_node < 1:
                raise ValueError(f"n_gpus_per_node must be >= 1, got {n_gpus_per_node}")
            if total_gpus <= n_gpus_per_node:
                nnodes_per_instance = 1
            elif total_gpus % n_gpus_per_node != 0:
                raise ValueError(
                    f"tp_size * pp_size ({total_gpus}) must be divisible "
                    f"by n_gpus_per_node ({n_gpus_per_node})"
                )
            else:
                nnodes_per_instance = total_gpus // n_gpus_per_node
        self._nnodes_per_instance = nnodes_per_instance

        # Worker management
        self.workers: list[Worker] = []
        self._server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str = ""

        # Addresses resolved after initialization
        self._inf_addrs: list[str] = []
        self._router_addr: str = ""
        self._data_proxy_addrs: list[str] = []
        self._gateway_addr: str = ""

        # Active worker ID mapping (data proxy addr → router-confirmed worker_id)
        self._worker_ids: dict[str, str] = {}  # data_proxy_addr -> worker_id
        # Registration intent is stable for the lifetime of one launched data proxy.
        # _worker_ids remains the public map of router-confirmed active incarnations.
        self._desired_worker_ids: dict[str, str] = {}
        self._predecessor_worker_ids: dict[str, str | None] = {}
        self._registration_operation_lock = Lock()
        self._registration_state_lock = Lock()
        # Serialize initialize/destroy state transitions.  Re-entrancy is
        # required when executor submission fails while initialize owns the
        # lifecycle lock and must roll back the callback server immediately.
        self._destroy_lock = threading.RLock()
        self._registration_generation = 0
        self._destroyed = False

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # WorkflowExecutor (created in initialize)
        self._workflow_executor = None

        # Staleness manager (created in initialize)
        self._staleness_manager = None

        # Online callback server / waiter state
        self._online_waiters: dict[str, _OnlineWaiter] = {}
        self._online_waiters_lock = Lock()
        self._online_settlements: OrderedDict[str, _OnlineSettlement] = OrderedDict()
        self._online_settlement_limit = _MAX_ONLINE_SETTLEMENT_TOMBSTONES
        self._callback_app = None
        self._callback_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()
        self._callback_serving = threading.Event()
        self._callback_start_error: BaseException | None = None

        # Track which service roles were created for cleanup
        self._service_roles: list[str] = []

        # Track services forked directly via RPCGuard /fork (raw_cmd mode).
        # Each entry: (guard_addr, role, worker_index) for /kill_forked_worker.
        self._forked_services: list[tuple[str, str, int]] = []

        # Shared HTTP clients
        self._sync_client = httpx.Client(timeout=30.0)
        self._async_client: httpx.AsyncClient | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        self._async_client_cleanup: HttpxAsyncClientCleanup | None = None
        self._async_client_lock = Lock()

        # Proxy compatibility (no-ops — gateway IS the proxy)
        self._proxy_started = False
        self.proxy_workers: list = []
        self.proxy_addrs: list[str] = []

        # Pipelined initialization state
        self._init_future: concurrent.futures.Future | None = None
        self._initialization_error: BaseException | None = None
        self._init_lock = threading.Lock()
        self._workers_ready = threading.Event()
        self._shutdown_requested = threading.Event()

    # -- Initialize --------------------------------------------------------

    def _rollback_initialization(self, primary_error: BaseException) -> None:
        """Fence startup and reap every owner without masking its root cause."""
        if self._initialization_error is None:
            self._initialization_error = primary_error
        self._shutdown_requested.set()
        try:
            self.destroy()
        except BaseException as cleanup_error:
            primary_error.add_note(
                "Startup rollback cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    def initialize(
        self,
        role: str,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args: Any,
        wait: bool = False,
        **kwargs: Any,
    ) -> concurrent.futures.Future | None:
        from areal.infra.utils.concurrent import get_executor

        with self._destroy_lock:
            if self._destroyed or self._shutdown_requested.is_set():
                raise RuntimeError("initialize() called after controller shutdown")
            if self._init_future is not None:
                raise RuntimeError(
                    "initialize() called while a previous initialization is in progress"
                )

            self._worker_role = role
            self._workers_ready.clear()
            self._initialization_error = None
            try:
                self._start_online_callback_server()
                future = get_executor("ctrl_init").submit(
                    self._guarded_bg_initialize,
                    server_args,
                    server_infos,
                    *args,
                    **kwargs,
                )
            except BaseException as exc:
                self._rollback_initialization(exc)
                raise
            # Holding the lifecycle lock prevents a fast background failure
            # from destroying the controller before this owner is published.
            self._init_future = future

        try:
            ready_timeout = self.config.workers_ready_timeout
            if not self._workers_ready.wait(timeout=ready_timeout):
                raise TimeoutError(f"Worker creation timed out after {ready_timeout}s")
            if future.done():
                future.result()
            if self._shutdown_requested.is_set():
                if self._initialization_error is not None:
                    raise self._initialization_error
                raise RuntimeError("controller initialization cancelled by shutdown")

            if wait:
                self._ensure_initialized()
                return None
            return future
        except BaseException as exc:
            self._rollback_initialization(exc)
            raise

    def _guarded_bg_initialize(self, *args: Any, **kwargs: Any) -> None:
        """Ensure _workers_ready is signaled even if _bg_initialize fails."""
        primary_error: BaseException | None = None
        try:
            self._bg_initialize(*args, **kwargs)
        except BaseException as exc:
            primary_error = exc
            logger.error(
                "RolloutControllerV2 initialization failed, rolling back",
                exc_info=True,
            )
            self._rollback_initialization(exc)
            raise
        finally:
            self._workers_ready.set()
            if primary_error is None and self._shutdown_requested.is_set():
                # A concurrent destroy may finish before an in-flight scheduler
                # or HTTP call publishes its acknowledged owner. The initializer
                # performs the terminal pass once it regains control.
                self.destroy()

    def _bg_initialize(
        self,
        server_args: dict[str, Any] | None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from areal.infra.utils.concurrent import run_async_task

        if self._shutdown_requested.is_set():
            return
        run_async_task(
            self._async_initialize, server_args, server_infos, *args, **kwargs
        )

        if self._shutdown_requested.is_set():
            return

        self._register_data_proxies_in_router()

        if self._shutdown_requested.is_set():
            return

        from areal.infra.remote_inf_engine import RemoteInfEngine
        from areal.infra.staleness_manager import StalenessManager
        from areal.infra.workflow_executor import WorkflowExecutor

        max_concurrent = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent,
            consumer_batch_size=self.config.consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        self._workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=cast(RemoteInfEngine, self),
            staleness_manager=self._staleness_manager,
        )
        self._workflow_executor.initialize()

        if self._shutdown_requested.is_set():
            return

        logger.info("RolloutControllerV2 initialized (role=%s)", self._worker_role)

        if self.config.model:
            # Call _register_model_impl directly to avoid _ensure_initialized()
            # which would deadlock: the main thread holds _init_lock waiting for
            # this future to complete, but _ensure_initialized() also needs _init_lock.
            self._register_model_impl(
                model=self.config.model,
                url=self.config.api_url or "",
                api_key=self.config.provider_api_key,
            )
        if self.external_mode:
            logger.info(
                "External model mode: url=%s, model=%s",
                self.config.api_url,
                self.config.model,
            )

    def _ensure_initialized(self) -> None:
        if self._init_future is None:
            # Some direct/legacy callers construct a lightweight controller
            # with ``__new__`` and provide pre-existing inference addresses.
            # Treat the absence of the pipelined-initialization field as the
            # historical "no pending initialization" state.
            initialization_error = getattr(self, "_initialization_error", None)
            if initialization_error is not None:
                raise initialization_error
            return
        with self._init_lock:
            future = self._init_future
            if future is None:
                return
            future.result(timeout=self.config.setup_timeout)
            self._init_future = None

    async def _async_initialize(
        self,
        server_args: dict[str, Any] | None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
        from areal.api.scheduler_api import Job

        cfg = self.config
        admin_api_key = self.config.admin_api_key
        agent_cfg = self._agent_config

        if self.external_mode:
            dp_size = 1
            inf_backend = None
        else:
            alloc = self.rollout_alloc
            dp_size = alloc.parallel.dp_size
            inf_backend = alloc.backend

        # ==================================================================
        # Step 0: Create RPCGuard workers (dp_size × nnodes_per_instance)
        # ==================================================================
        if self.external_mode:
            inf_spec = SchedulingSpec(
                task_type="worker",
                port_count=2,
                gpu=0,
                mem=8,
                cmd="python -m areal.v2.inference_service.guard",
            )
            total_workers = dp_size
        else:
            inf_spec = SchedulingSpec(**asdict(cfg.scheduling_spec[0]))
            instance_size = alloc.parallel.tp_size * alloc.parallel.pp_size
            nnodes_per_instance = self._nnodes_per_instance
            gpus_per_worker = instance_size // nnodes_per_instance

            if server_infos is not None:
                total_workers = dp_size
                inf_spec.gpu = 0
            else:
                total_workers = dp_size * nnodes_per_instance
                inf_spec.cpu *= gpus_per_worker
                inf_spec.mem *= gpus_per_worker
                if inf_spec.gpu > 0:
                    inf_spec.gpu = gpus_per_worker

            inf_spec.cmd = "python -m areal.v2.inference_service.guard"

        inf_role = f"{self._worker_role}{self._INF_SUFFIX}"
        inf_job = Job(
            replicas=total_workers,
            tasks=[inf_spec for _ in range(total_workers)],
            scheduling_strategy=SchedulingStrategy(),
            role=inf_role,
        )

        self.scheduler.create_workers(job=inf_job)
        self._service_roles.append(inf_role)
        inf_workers = self.scheduler.get_workers(role=inf_role)
        if len(inf_workers) != total_workers:
            raise RuntimeError(
                f"Expected {total_workers} workers for role {inf_role!r}, "
                f"got {len(inf_workers)}"
            )
        self.workers = inf_workers
        logger.info("RPCGuard workers ready: %s", [w.id for w in inf_workers])

        self._workers_ready.set()

        if self._shutdown_requested.is_set():
            return

        guard_addr_0 = f"http://{format_hostport(self.workers[0].ip, int(self.workers[0].worker_ports[0]))}"

        # ==================================================================
        # Step 1+2: Launch inference servers AND fork Router in PARALLEL
        # Router does not depend on inference servers.
        # ==================================================================
        router_cmd = [
            sys.executable,
            "-m",
            "areal.v2.inference_service.router",
            "--admin-api-key",
            admin_api_key,
            "--routing-strategy",
            cfg.routing_strategy,
            "--poll-interval",
            str(cfg.poll_interval),
            "--log-level",
            _DEFAULT_SERVICE_LOG_LEVEL,
        ]
        router_task = asyncio.ensure_future(
            self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="router",
                worker_index=0,
                raw_cmd=router_cmd,
            )
        )

        if self.external_mode:
            logger.info("External mode — skipping inference server launch")
        elif server_infos is not None:
            self._server_infos = list(server_infos)
            self._inf_addrs = [
                f"http://{format_hostport(info.host, info.port)}"
                for info in server_infos
            ]
            logger.info(
                "Using %d pre-existing server_infos, skipping inference server fork",
                len(server_infos),
            )
        else:
            assert alloc is not None and inf_backend is not None
            await self._async_fork_inf_servers(
                cfg,
                alloc,
                inf_backend,
                inf_workers,
                dp_size,
                nnodes_per_instance,
                server_args,
            )
        logger.info("Inference servers: %s", self._inf_addrs)

        router_host, router_port = await router_task
        self._router_addr = f"http://{format_hostport(router_host, router_port)}"
        logger.info("Router: %s", self._router_addr)

        if self._shutdown_requested.is_set():
            return

        # ==================================================================
        # Step 3+4: Fork Data Proxies AND Gateway in PARALLEL
        # Data Proxies need _inf_addrs (ready); Gateway needs _router_addr (ready).
        # ==================================================================
        data_proxy_base_cmd = [
            sys.executable,
            "-m",
            "areal.v2.inference_service.data_proxy",
            "--tokenizer-path",
            cfg.tokenizer_path,
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            _DEFAULT_SERVICE_LOG_LEVEL,
            "--request-timeout",
            str(cfg.request_timeout),
            "--set-reward-finish-timeout",
            str(cfg.agent.set_reward_finish_timeout),
            "--callback-server-addr",
            f"http://{self.callback_addr}",
            "--tool-call-parser",
            agent_cfg.tool_call_parser,
            "--reasoning-parser",
            agent_cfg.reasoning_parser,
            "--chat-template-type",
            agent_cfg.chat_template_type,
        ]
        if agent_cfg.engine_max_tokens is not None:
            data_proxy_base_cmd += [
                "--engine-max-tokens",
                str(agent_cfg.engine_max_tokens),
            ]
        if cfg.use_lora:
            data_proxy_base_cmd += ["--use-lora", "--lora-name", cfg.lora_name]

        data_proxy_worker_ids = [str(uuid.uuid4()) for _ in range(dp_size)]

        async def _fork_data_proxy(group_idx: int) -> tuple[str, int, str]:
            worker_id = data_proxy_worker_ids[group_idx]
            if self.external_mode:
                head_worker = inf_workers[group_idx]
            else:
                head_worker = inf_workers[
                    group_idx
                    if server_infos is not None
                    else group_idx * nnodes_per_instance
                ]
            guard_addr = f"http://{format_hostport(head_worker.ip, int(head_worker.worker_ports[0]))}"
            if self.external_mode:
                dp_cmd = data_proxy_base_cmd + [
                    "--worker-id",
                    worker_id,
                    "--backend-addr",
                    "",
                ]
            else:
                dp_cmd = data_proxy_base_cmd + [
                    "--worker-id",
                    worker_id,
                    "--backend-addr",
                    self._inf_addrs[group_idx],
                    "--backend-type",
                    inf_backend or "sglang",
                ]
            host, port = await self._async_fork_on_guard(
                guard_addr=guard_addr,
                role="data-proxy",
                worker_index=group_idx,
                raw_cmd=dp_cmd,
            )
            return host, port, worker_id

        gw_cmd = [
            sys.executable,
            "-m",
            "areal.v2.inference_service.gateway",
            "--admin-api-key",
            admin_api_key,
            "--router-addr",
            self._router_addr,
            "--forward-timeout",
            str(cfg.request_timeout),
            "--log-level",
            _DEFAULT_SERVICE_LOG_LEVEL,
        ]

        gw_task = asyncio.ensure_future(
            self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="gateway",
                worker_index=0,
                raw_cmd=gw_cmd,
            )
        )
        dp_results = await asyncio.gather(
            *[_fork_data_proxy(i) for i in range(dp_size)]
        )

        for dp_host, dp_port, worker_id in dp_results:
            self._record_data_proxy_launch(
                f"http://{format_hostport(dp_host, dp_port)}", worker_id
            )
        logger.info("Data proxies: %s", self._data_proxy_addrs)

        gw_host, gw_port = await gw_task
        self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
        logger.info("Gateway: %s", self._gateway_addr)

    async def _async_fork_inf_servers(
        self,
        cfg: Any,
        alloc: Any,
        inf_backend: str,
        inf_workers: list,
        dp_size: int,
        nnodes_per_instance: int,
        server_args: dict[str, Any] | None,
    ) -> None:
        if inf_backend == "sglang":
            from areal.api.cli_args import SGLangConfig

            _build_launch_cmd = SGLangConfig.build_cmd_from_args
        elif inf_backend == "vllm":
            from areal.api.cli_args import vLLMConfig

            _build_launch_cmd = vLLMConfig.build_cmd_from_args
        else:
            raise ValueError(f"Unsupported inference backend: {inf_backend!r}")

        async def _fork_group(
            group_idx: int,
        ) -> tuple[str, int, list[tuple[str, str, int]]]:
            group_workers = inf_workers[
                group_idx * nnodes_per_instance : (group_idx + 1) * nnodes_per_instance
            ]
            head_worker = group_workers[0]
            head_guard_addr = f"http://{format_hostport(head_worker.ip, int(head_worker.worker_ports[0]))}"
            client = await self._get_async_client()

            dist_init_addr = None
            if nnodes_per_instance > 1:
                resp = await client.post(
                    f"{head_guard_addr}/alloc_ports",
                    json={"count": 1},
                    timeout=30.0,
                )
                resp.raise_for_status()
                rendezvous_data = resp.json()
                rendezvous_host = rendezvous_data["host"]
                rendezvous_port = rendezvous_data["ports"][0]
                dist_init_addr = format_hostport(rendezvous_host, rendezvous_port)

            async def _fork_node(node_rank: int, worker: Any) -> tuple[str, int, str]:
                guard_addr = (
                    f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
                )

                resp = await client.post(
                    f"{guard_addr}/alloc_ports",
                    json={"count": 1},
                    timeout=30.0,
                )
                resp.raise_for_status()
                port_data = resp.json()
                inf_host: str = port_data["host"]
                inf_port: int = port_data["ports"][0]

                local_args = {
                    **(server_args or {}),
                    "host": inf_host,
                    "port": inf_port,
                    "nnodes": nnodes_per_instance,
                    "node_rank": node_rank,
                    "dist_init_addr": dist_init_addr,
                }
                cmd = _build_launch_cmd(local_args)

                fork_payload: dict[str, Any] = {
                    "role": "inf-server",
                    "worker_index": group_idx * nnodes_per_instance + node_rank,
                    "raw_cmd": cmd,
                }
                if inf_backend == "vllm":
                    from areal.infra.utils.launcher import (
                        TRITON_CACHE_PATH as _TRITON_CACHE,
                    )
                    from areal.infra.utils.launcher import (
                        VLLM_CACHE_ROOT as _VLLM_CACHE,
                    )

                    fork_payload["env"] = {
                        "TRITON_CACHE_PATH": os.path.join(
                            os.environ.get("TRITON_CACHE_PATH", _TRITON_CACHE),
                            str(uuid.uuid4()),
                        ),
                        "VLLM_CACHE_ROOT": os.path.join(
                            os.environ.get("VLLM_CACHE_ROOT", _VLLM_CACHE),
                            str(uuid.uuid4()),
                        ),
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                    }

                resp = await client.post(
                    f"{guard_addr}/fork",
                    json=fork_payload,
                    timeout=30.0,
                )
                resp.raise_for_status()
                owner = (
                    guard_addr,
                    "inf-server",
                    group_idx * nnodes_per_instance + node_rank,
                )
                self._forked_services.append(owner)
                return inf_host, inf_port, guard_addr

            node_results = await asyncio.gather(
                *[_fork_node(rank, w) for rank, w in enumerate(group_workers)]
            )

            head_inf_host, head_inf_port, _ = node_results[0]
            forked: tuple[tuple[str, str, int], ...] = tuple(
                (guard_addr, "inf-server", group_idx * nnodes_per_instance + rank)
                for rank, (_, _, guard_addr) in enumerate(node_results)
            )
            return (head_inf_host, head_inf_port, forked)

        group_results = await asyncio.gather(*[_fork_group(i) for i in range(dp_size)])

        health_targets: list[tuple[str, tuple[tuple[str, str, int], ...]]] = []
        for host, port, forked in group_results:
            addr = f"http://{format_hostport(host, port)}"
            self._inf_addrs.append(addr)
            self._server_infos.append(
                LocalInfServerInfo(
                    host=host,
                    port=port,
                    process=None,  # type: ignore[arg-type]
                )
            )
            health_targets.append((addr, forked))

        # Wait for all inference servers to be healthy in parallel
        await asyncio.gather(
            *[
                self._async_wait_for_service(
                    f"{addr}/health",
                    f"InfServer-{i}",
                    timeout=cfg.setup_timeout,
                    process_owners=process_owners,
                )
                for i, (addr, process_owners) in enumerate(health_targets)
            ]
        )

    # -- Service health checks & registration ------------------------------

    def _record_data_proxy_launch(
        self, data_proxy_addr: str, worker_id: str | None = None
    ) -> None:
        """Track one newly launched process and freeze its registration CAS."""
        data_proxy_addr = data_proxy_addr.rstrip("/")
        with self._registration_operation_lock:
            with self._registration_state_lock:
                if self._destroyed or self._shutdown_requested.is_set():
                    return
                if worker_id is not None and data_proxy_addr in self._data_proxy_addrs:
                    raise ValueError(
                        "duplicate canonical data-proxy address returned by launcher: "
                        f"{data_proxy_addr}"
                    )
                desired_worker_id = worker_id or str(uuid.uuid4())
                expected_worker_id = self._worker_ids.get(data_proxy_addr)
                if data_proxy_addr not in self._data_proxy_addrs:
                    self._data_proxy_addrs.append(data_proxy_addr)
                self._desired_worker_ids[data_proxy_addr] = desired_worker_id
                self._predecessor_worker_ids[data_proxy_addr] = expected_worker_id
                self._registration_generation += 1

    def _freeze_data_proxy_registration_snapshot(
        self,
    ) -> _DataProxyRegistrationSnapshot | None:
        """Atomically freeze the CAS payload for one registration attempt."""
        with self._registration_state_lock:
            if (
                self._destroyed
                or self._shutdown_requested.is_set()
                or not self._data_proxy_addrs
            ):
                return None

            state_changed = False
            for data_proxy_addr in self._data_proxy_addrs:
                if data_proxy_addr not in self._desired_worker_ids:
                    self._desired_worker_ids[data_proxy_addr] = str(uuid.uuid4())
                    state_changed = True
                if data_proxy_addr not in self._predecessor_worker_ids:
                    self._predecessor_worker_ids[data_proxy_addr] = (
                        self._worker_ids.get(data_proxy_addr)
                    )
                    state_changed = True
            if state_changed:
                self._registration_generation += 1

            return _DataProxyRegistrationSnapshot(
                generation=self._registration_generation,
                intents=tuple(
                    _DataProxyRegistrationIntent(
                        data_proxy_addr=data_proxy_addr,
                        desired_worker_id=self._desired_worker_ids[data_proxy_addr],
                        expected_worker_id=self._predecessor_worker_ids[
                            data_proxy_addr
                        ],
                    )
                    for data_proxy_addr in self._data_proxy_addrs
                ),
            )

    def _wait_for_service(
        self,
        url: str,
        name: str,
        timeout: float | None = None,
        process_owners: tuple[tuple[str, str, int], ...] = (),
    ) -> None:
        """Wait for a service to become healthy."""
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._shutdown_requested.is_set():
                raise RuntimeError(f"{name} startup cancelled by controller shutdown")
            is_healthy = False
            try:
                resp = self._sync_client.get(url, timeout=2)
                is_healthy = resp.status_code == 200
            except httpx.HTTPError:
                pass
            for process_owner in process_owners:
                returncode = self._forked_worker_returncode(process_owner)
                if returncode is not None:
                    _, role, worker_index = process_owner
                    raise RuntimeError(
                        f"{name} ({role}/{worker_index}) exited before becoming "
                        f"healthy with exit code {returncode}"
                    )
            if is_healthy:
                logger.info("%s is ready at %s", name, url)
                return
            time.sleep(0.1)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    async def _async_wait_for_service(
        self,
        url: str,
        name: str,
        timeout: float | None = None,
        process_owners: tuple[tuple[str, str, int], ...] = (),
    ) -> None:
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        client = await self._get_async_client()
        while time.monotonic() < deadline:
            if self._shutdown_requested.is_set():
                raise RuntimeError(f"{name} startup cancelled by controller shutdown")
            is_healthy = False
            try:
                resp = await client.get(url, timeout=2.0)
                is_healthy = resp.status_code == 200
            except Exception:
                pass
            for process_owner in process_owners:
                returncode = await self._async_forked_worker_returncode(
                    client, process_owner
                )
                if returncode is not None:
                    _, role, worker_index = process_owner
                    raise RuntimeError(
                        f"{name} ({role}/{worker_index}) exited before becoming "
                        f"healthy with exit code {returncode}"
                    )
            if is_healthy:
                logger.info("%s is ready at %s", name, url)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    def _forked_worker_returncode(
        self, process_owner: tuple[str, str, int]
    ) -> int | None:
        guard_addr, role, worker_index = process_owner
        try:
            response = self._sync_client.get(
                f"{guard_addr}/forked_worker_status",
                params={"role": role, "worker_index": worker_index},
                timeout=2.0,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            if payload.get("running", True):
                return None
            return int(payload["returncode"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    async def _async_forked_worker_returncode(
        client: httpx.AsyncClient,
        process_owner: tuple[str, str, int],
    ) -> int | None:
        guard_addr, role, worker_index = process_owner
        try:
            response = await client.get(
                f"{guard_addr}/forked_worker_status",
                params={"role": role, "worker_index": worker_index},
                timeout=2.0,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            if payload.get("running", True):
                return None
            return int(payload["returncode"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None

    def _register_data_proxies_in_router(self) -> None:
        """Register all data proxy workers in the router and store their worker IDs."""
        with self._registration_operation_lock:
            self._register_data_proxies_in_router_serialized()

    def _register_data_proxies_in_router_serialized(self) -> None:
        """Run one registration snapshot, request, and commit without a new launch."""
        snapshot = self._freeze_data_proxy_registration_snapshot()
        if snapshot is None:
            return

        from concurrent.futures import ThreadPoolExecutor

        admin_key = self.config.admin_api_key
        router_addr = self._router_addr

        def _register_one(
            intent: _DataProxyRegistrationIntent,
        ) -> tuple[_DataProxyRegistrationIntent, str]:
            # Each thread gets its own httpx.Client because httpx.Client
            # is not thread-safe and must not be shared across threads.
            with httpx.Client() as client:
                resp = client.post(
                    f"{router_addr}/register",
                    json={
                        "worker_addr": intent.data_proxy_addr,
                        "worker_id": intent.desired_worker_id,
                        "expected_worker_id": intent.expected_worker_id,
                    },
                    headers={"Authorization": f"Bearer {admin_key}"},
                    timeout=5,
                )
            resp.raise_for_status()
            worker_id = resp.json().get("worker_id")
            if worker_id != intent.desired_worker_id:
                raise RuntimeError(
                    "Router registered a different data proxy incarnation for "
                    f"{intent.data_proxy_addr}: expected "
                    f"{intent.desired_worker_id}, got {worker_id}"
                )
            logger.info(
                "Registered data proxy %s in router (worker_id=%s)",
                intent.data_proxy_addr,
                worker_id,
            )
            return intent, worker_id

        with ThreadPoolExecutor(max_workers=len(snapshot.intents)) as pool:
            results = list(pool.map(_register_one, snapshot.intents))

        with self._registration_state_lock:
            if (
                self._destroyed
                or self._registration_generation != snapshot.generation
                or tuple(self._data_proxy_addrs)
                != tuple(intent.data_proxy_addr for intent in snapshot.intents)
            ):
                return
            for intent, _worker_id in results:
                if (
                    self._desired_worker_ids.get(intent.data_proxy_addr)
                    != intent.desired_worker_id
                    or intent.data_proxy_addr not in self._predecessor_worker_ids
                    or self._predecessor_worker_ids[intent.data_proxy_addr]
                    != intent.expected_worker_id
                ):
                    return
            for intent, worker_id in results:
                self._worker_ids[intent.data_proxy_addr] = worker_id

    def register_model(
        self,
        model: str,
        url: str = "",
        api_key: str | None = None,
        data_proxy_addrs: list[str] | None = None,
    ) -> None:
        self._ensure_initialized()
        self._register_model_impl(model, url, api_key, data_proxy_addrs)

    def _register_model_impl(
        self,
        model: str,
        url: str = "",
        api_key: str | None = None,
        data_proxy_addrs: list[str] | None = None,
    ) -> None:
        if data_proxy_addrs is None:
            data_proxy_addrs = self._data_proxy_addrs
        resp = self._sync_client.post(
            f"{self._gateway_addr}/register_model",
            json={
                "model": model,
                "url": url,
                "api_key": api_key,
                "data_proxy_addrs": data_proxy_addrs,
            },
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()

    @property
    def external_mode(self) -> bool:
        return self.config.api_url is not None

    def _start_online_callback_server(self) -> None:
        """Start callback server used by the router to deliver ready trajectories."""
        if self._callback_server is not None:
            return

        from flask import Flask, jsonify, request
        from werkzeug.serving import make_server

        from areal.utils.network import find_free_ports, gethostip

        app = Flask("online_rollout_callback")

        @app.route("/callback/online_ready", methods=["POST"])
        def online_ready():
            if request.headers.get("Authorization") != (
                f"Bearer {self.config.admin_api_key}"
            ):
                return jsonify({"error": "Invalid admin API key"}), 403
            payload = request.get_json() or {}
            try:
                if self._callback_loop is None:
                    raise RuntimeError("Callback loop not ready")
                result = self._callback_loop.run_until_complete(
                    self._handle_online_ready_callback(payload)
                )
                return jsonify(result)
            except _OnlineLeaseConflict as exc:
                return jsonify({"error": str(exc)}), 409
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 425
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Online callback handler error: %s", exc, exc_info=True)
                return jsonify({"error": str(exc)}), 500

        @app.route("/callback/online_failed", methods=["POST"])
        def online_failed():
            if request.headers.get("Authorization") != (
                f"Bearer {self.config.admin_api_key}"
            ):
                return jsonify({"error": "Invalid admin API key"}), 403
            payload = request.get_json() or {}
            try:
                if self._callback_loop is None:
                    raise RuntimeError("Callback loop not ready")
                result = self._callback_loop.run_until_complete(
                    self._handle_online_failed_callback(payload)
                )
                return jsonify(result)
            except _OnlineLeaseConflict as exc:
                return jsonify({"error": str(exc)}), 409
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 425
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Online failure handler error: %s", exc, exc_info=True)
                return jsonify({"error": str(exc)}), 500

        self._callback_port = int(find_free_ports(1)[0])
        self._callback_host = gethostip()
        self._callback_app = app
        assert self._callback_host is not None
        assert self._callback_port is not None
        self._callback_server = make_server(
            self._callback_host,
            self._callback_port,
            app,
            threaded=False,
        )
        self._callback_server.RequestHandlerClass.log_request = (  # type: ignore[attr-defined]
            lambda self, *args, **kwargs: None
        )

        self._callback_loop_ready.clear()
        self._callback_serving.clear()
        self._callback_start_error = None
        original_service_actions = self._callback_server.service_actions

        def mark_serving() -> None:
            original_service_actions()
            self._callback_serving.set()
            self._callback_loop_ready.set()

        # BaseServer invokes service_actions only after serve_forever has entered
        # its polling loop.  Publishing readiness there makes shutdown safe: a
        # mere Thread.start() acknowledgement is too early for BaseServer.shutdown.
        self._callback_server.service_actions = mark_serving

        def serve_forever() -> None:
            try:
                self._callback_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._callback_loop)
                logger.info(
                    "Online callback server starting on %s",
                    format_hostport(self._callback_host, self._callback_port),
                )
                assert self._callback_server is not None
                self._callback_server.serve_forever(poll_interval=0.05)
            except BaseException as exc:
                self._callback_start_error = exc
                self._callback_loop_ready.set()
            finally:
                # Wake a waiter even if a custom server exits without entering
                # BaseServer's polling loop and without raising.
                self._callback_loop_ready.set()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()
        startup_timeout = self.config.setup_timeout
        if not self._callback_loop_ready.wait(timeout=startup_timeout):
            raise TimeoutError(
                "Online callback server did not enter its serving loop within "
                f"{startup_timeout}s"
            )
        if self._callback_start_error is not None:
            raise self._callback_start_error
        thread = self._callback_server_thread
        if (
            not self._callback_serving.is_set()
            or thread is None
            or not thread.is_alive()
        ):
            # Read the error once more after observing thread termination so an
            # exception published immediately after the readiness wake wins.
            if self._callback_start_error is not None:
                raise self._callback_start_error
            raise RuntimeError("Online callback server exited before becoming ready")

    def _stop_online_callback_server(self) -> None:
        if self._callback_server is not None:
            logger.info("Stopping online callback server...")
            server = self._callback_server
            thread = self._callback_server_thread
            serving = (
                self._callback_serving.is_set()
                and thread is not None
                and thread.is_alive()
            )
            if serving:
                server.shutdown()
            else:
                # BaseServer.shutdown deadlocks if serve_forever never entered.
                server.server_close()
            if thread is not None and thread.ident is not None:
                thread.join(timeout=5.0)
            if serving:
                server.server_close()
            if self._callback_loop is not None:
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()
            self._callback_serving.clear()
            self._callback_start_error = None

    @property
    def callback_addr(self) -> str:
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return format_hostport(self._callback_host, self._callback_port)

    def reserve_online_trajectory(self) -> tuple[str, int]:
        """Register a version-bound waiter before its lease becomes visible.

        Registering first closes the race where a fast external producer could
        complete before the controller had a future ready to receive it.
        """

        lease_id = str(uuid.uuid4())
        expected_version = self.get_version()
        future = asyncio.get_running_loop().create_future()
        with self._online_waiters_lock:
            self._online_waiters[lease_id] = _OnlineWaiter(
                lease_id=lease_id,
                expected_version=expected_version,
                future=future,
            )
        return lease_id, expected_version

    def cancel_online_trajectory(self, lease_id: str) -> None:
        """Remove and cancel a reservation without buffering a late callback."""

        with self._online_waiters_lock:
            waiter = self._online_waiters.pop(lease_id, None)
        if waiter is not None and not waiter.future.done():
            waiter.future.cancel()

    def _record_online_settlement_locked(
        self, lease_id: str, settlement: _OnlineSettlement
    ) -> None:
        """Remember a settled callback fingerprint while holding the waiter lock."""

        existing = self._online_settlements.get(lease_id)
        if existing is not None:
            if existing != settlement:  # pragma: no cover - internal invariant
                raise RuntimeError(
                    f"Conflicting settlement for online lease {lease_id}"
                )
            return
        self._online_settlements[lease_id] = settlement
        while len(self._online_settlements) > self._online_settlement_limit:
            self._online_settlements.popitem(last=False)

    async def wait_for_online_trajectory(
        self, lease_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        with self._online_waiters_lock:
            waiter = self._online_waiters.get(lease_id)
        if waiter is None:
            raise RuntimeError(f"Unknown online lease: {lease_id}")
        try:
            if timeout is None:
                return await waiter.future
            return await asyncio.wait_for(waiter.future, timeout=timeout)
        finally:
            with self._online_waiters_lock:
                current = self._online_waiters.get(lease_id)
                if current is waiter:
                    self._online_waiters.pop(lease_id, None)

    async def _handle_online_ready_callback(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        session_id = payload.get("session_id")
        trajectory_id = payload.get("trajectory_id")
        lease_id = payload.get("lease_id")
        expected_version = payload.get("expected_version")
        if (
            not session_id
            or trajectory_id is None
            or not lease_id
            or expected_version is None
        ):
            raise _OnlineLeaseConflict(
                "Missing session_id, trajectory_id, lease_id, or expected_version"
            )

        export_request = {
            "session_id": session_id,
            "trajectory_id": int(trajectory_id),
            "lease_id": str(lease_id),
            "expected_version": int(expected_version),
        }
        group_id = payload.get("group_id")
        if group_id is not None:
            export_request["group_id"] = str(group_id)

        callback_settlement = _OnlineSettlement(
            callback_kind="ready",
            expected_version=int(expected_version),
            details=(
                str(session_id),
                int(trajectory_id),
                str(group_id) if group_id is not None else None,
            ),
        )
        callback_ack = {
            "status": "ok",
            "session_id": session_id,
            "trajectory_id": int(trajectory_id),
            "lease_id": str(lease_id),
        }

        with self._online_waiters_lock:
            settled = self._online_settlements.get(str(lease_id))
            if settled is not None:
                if settled == callback_settlement:
                    return callback_ack
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} is already settled by another callback"
                )
            waiter = self._online_waiters.get(str(lease_id))
            if waiter is None or waiter.future.cancelled():
                raise _OnlineLeaseConflict(f"Unknown or cancelled lease: {lease_id}")
            if waiter.expected_version != int(expected_version):
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} expects policy version "
                    f"{waiter.expected_version}, got {expected_version}"
                )
            if waiter.delivery is not None:
                if waiter.delivery == export_request:
                    return callback_ack
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} was already settled by another trajectory"
                )
            if waiter.future.done():
                raise _OnlineLeaseConflict(f"Lease {lease_id} is already settled")
            waiter.delivery = export_request

        delivery_ack: concurrent.futures.Future[None] = concurrent.futures.Future()

        def _deliver() -> None:
            try:
                with self._online_waiters_lock:
                    current = self._online_waiters.get(str(lease_id))
                    if (
                        current is not waiter
                        or waiter.future.cancelled()
                        or waiter.future.done()
                    ):
                        raise _OnlineLeaseConflict(
                            f"Lease {lease_id} was cancelled before delivery"
                        )
                    waiter.future.set_result(export_request)
                    self._record_online_settlement_locked(
                        str(lease_id), callback_settlement
                    )
                delivery_ack.set_result(None)
            except BaseException as exc:
                delivery_ack.set_exception(exc)

        waiter.future.get_loop().call_soon_threadsafe(_deliver)
        try:
            await asyncio.wrap_future(delivery_ack)
        except BaseException:
            with self._online_waiters_lock:
                current = self._online_waiters.get(str(lease_id))
                if current is waiter and waiter.future.cancelled():
                    self._online_waiters.pop(str(lease_id), None)
            raise
        return callback_ack

    async def _handle_online_failed_callback(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        lease_id = payload.get("lease_id")
        expected_version = payload.get("expected_version")
        reason = payload.get("reason")
        if not lease_id or expected_version is None or not reason:
            raise _OnlineLeaseConflict(
                "Missing lease_id, expected_version, or failure reason"
            )

        callback_settlement = _OnlineSettlement(
            callback_kind="failed",
            expected_version=int(expected_version),
            details=(str(reason),),
        )
        callback_ack = {"status": "ok", "lease_id": str(lease_id)}

        with self._online_waiters_lock:
            settled = self._online_settlements.get(str(lease_id))
            if settled is not None:
                if settled == callback_settlement:
                    return callback_ack
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} is already settled by another callback"
                )
            waiter = self._online_waiters.get(str(lease_id))
            if waiter is None or waiter.future.cancelled():
                raise _OnlineLeaseConflict(f"Unknown or cancelled lease: {lease_id}")
            if waiter.expected_version != int(expected_version):
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} expects policy version "
                    f"{waiter.expected_version}, got {expected_version}"
                )
            if waiter.delivery is not None:
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} already delivered a trajectory"
                )
            if waiter.failure_reason is not None:
                if waiter.failure_reason == str(reason):
                    return callback_ack
                raise _OnlineLeaseConflict(
                    f"Lease {lease_id} already failed for another reason"
                )
            if waiter.future.done():
                raise _OnlineLeaseConflict(f"Lease {lease_id} is already settled")
            waiter.failure_reason = str(reason)

        failure_ack: concurrent.futures.Future[None] = concurrent.futures.Future()

        def _fail() -> None:
            try:
                with self._online_waiters_lock:
                    current = self._online_waiters.get(str(lease_id))
                    if (
                        current is not waiter
                        or waiter.future.cancelled()
                        or waiter.future.done()
                    ):
                        raise _OnlineLeaseConflict(
                            f"Lease {lease_id} was cancelled before failure delivery"
                        )
                    waiter.future.set_exception(
                        RuntimeError(f"Online lease {lease_id} failed: {reason}")
                    )
                    self._record_online_settlement_locked(
                        str(lease_id), callback_settlement
                    )
                failure_ack.set_result(None)
            except BaseException as exc:
                failure_ack.set_exception(exc)

        waiter.future.get_loop().call_soon_threadsafe(_fail)
        await asyncio.wrap_future(failure_ack)
        return callback_ack

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        """Tear down all services and release resources."""
        with self._destroy_lock:
            with self._registration_state_lock:
                has_runtime_owners = bool(
                    self._callback_server is not None
                    or self._callback_server_thread is not None
                    or self._workflow_executor is not None
                    or self._forked_services
                    or self._service_roles
                    or self.workers
                    or self._async_client is not None
                    or self._async_client_cleanup is not None
                )
                if self._destroyed and not has_runtime_owners:
                    return
                # Fence late registration before any blocking cleanup starts.
                self._destroyed = True
                self._registration_generation += 1
                self._data_proxy_addrs.clear()
                self._worker_ids.clear()
                self._desired_worker_ids.clear()
                self._predecessor_worker_ids.clear()

            cleanup_errors: list[tuple[str, BaseException]] = []
            fork_cleanup_errors: list[
                tuple[tuple[str, str, int], str, BaseException]
            ] = []
            role_cleanup_errors: list[tuple[str, BaseException]] = []

            def record_cleanup_error(stage: str, exc: BaseException) -> None:
                logger.error("Cleanup failed at %s", stage, exc_info=True)
                cleanup_errors.append((stage, exc))

            self._shutdown_requested.set()
            future = self._init_future
            self._init_future = None
            cancelled = False
            if future is not None:
                cancelled = future.cancel()
            # A future cancelled before it starts cannot execute the guarded
            # initializer's finally block. Running initializers publish their
            # own error before waking the waiter.
            if future is None or cancelled:
                self._workers_ready.set()

            try:
                self._stop_online_callback_server()
            except BaseException as exc:
                record_cleanup_error("online callback server", exc)

            if self._workflow_executor is not None:
                try:
                    self._workflow_executor.destroy()
                except BaseException as exc:
                    record_cleanup_error("workflow executor", exc)
                else:
                    self._workflow_executor = None

            # Kill services forked directly via RPCGuard /fork in reverse order.
            for owner in reversed(list(self._forked_services)):
                guard_addr, role, worker_index = owner
                try:
                    self._kill_forked_service(guard_addr, role, worker_index)
                except BaseException as exc:
                    stage = f"forked service {role}/{worker_index}"
                    logger.error("Cleanup failed at %s", stage, exc_info=True)
                    fork_cleanup_errors.append((owner, stage, exc))
                else:
                    self._forked_services.remove(owner)

            # Close shared HTTP clients after all direct kill requests.
            try:
                self._sync_client.close()
            except BaseException as exc:
                record_cleanup_error("sync HTTP client", exc)
            with self._async_client_lock:
                async_client = self._async_client
                async_client_loop = self._async_client_loop
                async_client_cleanup = self._async_client_cleanup
            if async_client is not None and async_client_cleanup is None:
                record_cleanup_error(
                    "async HTTP client",
                    RuntimeError("async HTTP client cleanup state is missing"),
                )
            elif async_client is None and (
                async_client_loop is not None or async_client_cleanup is not None
            ):
                record_cleanup_error(
                    "async HTTP client",
                    RuntimeError("async HTTP client cleanup state is inconsistent"),
                )
            elif async_client_cleanup is not None:
                try:
                    close_httpx_client_from_sync(async_client_cleanup)
                except BaseException as exc:
                    record_cleanup_error("async HTTP client", exc)
                else:
                    with self._async_client_lock:
                        if self._async_client_cleanup is async_client_cleanup:
                            self._async_client = None
                            self._async_client_loop = None
                            self._async_client_cleanup = None

            for role in reversed(list(self._service_roles)):
                try:
                    self.scheduler.delete_workers(role=role)
                except BaseException as exc:
                    stage = f"worker role {role}"
                    logger.error("Cleanup failed at %s", stage, exc_info=True)
                    role_cleanup_errors.append((stage, exc))
                else:
                    logger.info("Workers deleted for role: %s", role)
                    self._service_roles.remove(role)

            # Scheduler deletion is best effort and does not currently prove
            # that every guard exited. Keep any direct owner whose kill was not
            # acknowledged so a later destroy() can retry it explicitly.
            for _owner, stage, exc in fork_cleanup_errors:
                record_cleanup_error(stage, exc)
            for stage, exc in role_cleanup_errors:
                record_cleanup_error(stage, exc)

            self.workers.clear()
            self._server_infos.clear()
            try:
                with self._online_waiters_lock:
                    for waiter in self._online_waiters.values():
                        if not waiter.future.done():
                            waiter.future.cancel()
                    self._online_waiters.clear()
                    self._online_settlements.clear()
            except BaseException as exc:
                record_cleanup_error("online waiters", exc)
            self._inf_addrs.clear()
            self._router_addr = ""
            self._gateway_addr = ""
            self._staleness_manager = None

            if cleanup_errors:
                with self._registration_state_lock:
                    # Permit an explicit retry for any owner retained above.
                    self._destroyed = False
                first_stage, first_error = cleanup_errors[0]
                for stage, error in cleanup_errors[1:]:
                    first_error.add_note(
                        f"{stage} also failed: {type(error).__name__}: {error}"
                    )
                first_error.add_note(f"First cleanup failure occurred at {first_stage}")
                raise first_error

    # -- Version management ------------------------------------------------

    def set_version(self, version: int) -> None:
        """Set version locally and broadcast to all data proxy workers."""
        from areal.infra.utils.concurrent import run_async_task

        self._ensure_initialized()

        with self._version_lock:
            self._version = version

        if not self._gateway_addr:
            return

        run_async_task(self._async_set_version, version)

    async def _async_set_version(self, version: int) -> None:
        payload = {"version": version}
        targets = self._snapshot_data_proxy_control_targets()
        results = await asyncio.gather(
            *[
                self._async_data_proxy_post(addr, worker_id, "/set_version", payload)
                for addr, worker_id in targets
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        for r in failed:
            logger.error("Failed to set version on a worker: %s", r)
        if failed and len(failed) == len(results):
            raise RuntimeError(
                f"set_version({version}) failed on ALL {len(failed)} workers"
            )

    def get_version(self) -> int:
        """Return the local version (compatible with VersionProvider protocol)."""
        with self._version_lock:
            return self._version

    # -- Capacity ----------------------------------------------------------

    def get_capacity(self) -> int:
        if self.staleness_manager is None:
            raise RuntimeError("RolloutControllerV2.initialize() must be called first")
        return self.staleness_manager.get_capacity()

    # -- Submit / Wait / Batch ---------------------------------------------

    def submit(
        self,
        data: dict[str, Any],
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
    ) -> int:
        self._ensure_initialized()
        expected_policy_version = self.get_version() if is_eval else None
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
        )
        resolved_workflow.expected_policy_version = expected_policy_version
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        self._ensure_initialized()
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def wait_for_task(
        self,
        task_id: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> dict[str, Any] | None:
        return self.workflow_executor.wait_for_task(
            task_id,
            timeout=timeout,
            raise_timeout=raise_timeout,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]] | None,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Submit a batch of data items and wait for all results.

        Parameters
        ----------
        data : list[dict[str, Any]] | None
            A list of data dicts to submit for rollout.  When ``None``
            (online-agent mode), a list of ``batch_size`` empty dicts is
            used automatically; ``batch_size`` **must** be provided in
            this case.
        workflow : Any
            Agent instance, agent class, import-path string, or ``None``
            for online mode.
        workflow_kwargs : dict[str, Any] | None
            Keyword arguments forwarded to the workflow/agent constructor.
        should_accept_fn : Any
            Optional predicate ``(trajectory_dict) -> bool`` used to
            filter results.
        group_size : int
            Number of times to run the workflow per input (default ``1``).
        batch_size : int | None
            Expected batch size.  **Required** when ``data`` is ``None``;
            when ``data`` is provided, an optional consistency check
            ensures ``len(data) == batch_size``.  Pass ``None`` (default)
            to skip the check.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dicts (one per completed rollout).
        """
        self._ensure_initialized()
        if not self._gateway_addr:
            raise RuntimeError("RolloutControllerV2.initialize() must be called first")
        if data is None:
            if batch_size is None:
                raise ValueError(
                    "batch_size must be specified when data is None (online-agent mode)"
                )
            data = [{} for _ in range(batch_size)]
        elif batch_size is not None and len(data) != batch_size:
            raise ValueError(
                f"len(data)={len(data)} does not match batch_size={batch_size}"
            )
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        for item in data:
            self.workflow_executor.submit(
                data=item,
                workflow=resolved_workflow,
                should_accept_fn=resolved_accept_fn,
            )
        results = self.workflow_executor.wait(count=len(data))
        # Return list of trajectories (matching RolloutController API)
        return [r for r in results if r is not None]

    def prepare_batch(
        self,
        dataloader: Any,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare a full training batch by consuming data from a dataloader.

        Parameters
        ----------
        dataloader : Any | None
            An iterable that yields batches of data dicts and exposes a
            ``batch_size`` attribute.  When ``None`` (online-agent mode),
            an internal dummy dataloader is used that produces a single
            batch of empty dicts sized by ``batch_size``.
        workflow : Any
            Agent instance, agent class, import-path string, or ``None``
            for online mode.
        workflow_kwargs : dict[str, Any] | None
            Keyword arguments forwarded to the workflow/agent constructor.
        should_accept_fn : Any
            Optional predicate ``(trajectory_dict) -> bool`` used to
            filter results.
        group_size : int
            Number of times to run the workflow per input (default ``1``).
        dynamic_bs : bool
            Enable dynamic batch sizing (default ``False``).
        batch_size : int | None
            Batch size for the dummy dataloader when ``dataloader`` is
            ``None``.  **Required** when ``dataloader`` is ``None``.
            Ignored when ``dataloader`` is not ``None``.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dicts (matching ``RolloutController`` API).
        """
        self._ensure_initialized()
        if not self._gateway_addr:
            raise RuntimeError("RolloutControllerV2.initialize() must be called first")
        if dataloader is None:
            if batch_size is None:
                raise ValueError(
                    "batch_size must be specified when dataloader is None "
                    "(online-agent mode)"
                )
            dataloader = _DummyDataLoader(batch_size=batch_size)
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        results = self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            dynamic_bs=dynamic_bs,
        )
        # Return list of trajectories (matching RolloutController API)
        return [r for r in results if r is not None]

    async def chat_completion(
        self,
        messages: list[dict],
        session_api_key: str | None = None,
        **kwargs,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """Send a chat completion request through the gateway HTTP stack.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style chat messages.
        session_api_key : str | None
            If provided, authenticate as this session; otherwise use the
            admin API key from the OpenAI proxy config.
        **kwargs
            Optional overrides: ``temperature``, ``top_p``,
            ``max_completion_tokens``, ``stream``.

        Returns
        -------
        ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]
            When ``stream=False`` (default): parsed OpenAI ChatCompletion object.
            When ``stream=True``: async generator yielding ChatCompletionChunk.
        """
        self._ensure_initialized()
        import aiohttp

        stream = kwargs.get("stream", False)
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "max_completion_tokens": kwargs.get("max_completion_tokens", 512),
            "stream": stream,
        }
        # Forward extra body params (e.g. chat_template_kwargs)
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        body["model"] = self.config.model
        api_key = (
            session_api_key
            if session_api_key is not None
            else self.config.admin_api_key
        )
        url = f"{self._gateway_addr}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if stream:
            return self._stream_chat_completion(url, body, headers)

        # Non-streaming path
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        ) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Gateway /chat/completions returned {resp.status}: {text}"
                    )
                resp_json = await resp.json()

        return ChatCompletion.model_validate(resp_json)

    async def _stream_chat_completion(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Parse SSE stream from the gateway into ChatCompletionChunk objects."""
        import aiohttp

        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        try:
            resp = await session.post(url, json=body, headers=headers)
            if resp.status != 200:
                text = await resp.text()
                await resp.release()
                await session.close()
                raise RuntimeError(
                    f"Gateway /chat/completions returned {resp.status}: {text}"
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                payload = decoded[len("data: ") :]
                if payload == "[DONE]":
                    break
                import json as _json

                chunk_data = _json.loads(payload)
                yield ChatCompletionChunk.model_validate(chunk_data)

            await resp.release()
        finally:
            await session.close()

    # -- Pause / Resume / Offload -------------------------------------------

    def pause(self) -> None:
        """Pause dispatcher + pause all workers."""
        self._ensure_initialized()
        assert self._workflow_executor is not None
        self._workflow_executor.pause()

    def resume(self) -> None:
        """Resume all workers + resume dispatcher."""
        self._ensure_initialized()
        assert self._workflow_executor is not None
        self._workflow_executor.resume()

    def offload(self) -> None:
        """Offload model memory on all inference workers."""
        from areal.infra.utils.concurrent import run_async_task

        self._ensure_initialized()
        run_async_task(self._async_offload)

    async def _async_offload(self) -> None:
        targets = self._snapshot_data_proxy_control_targets()
        if not targets:
            return
        results = await asyncio.gather(
            *(
                self._async_data_proxy_post(
                    addr, worker_id, "/release_memory_occupation", {}
                )
                for addr, worker_id in targets
            ),
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        for r in failed:
            logger.error("Failed to offload a worker: %s", r)
        if failed and len(failed) == len(results):
            raise RuntimeError(f"offload failed on ALL {len(failed)} workers")

    def onload(self, tags: list[str] | None = None) -> None:
        """Reload model memory on all inference workers."""
        from areal.infra.utils.concurrent import run_async_task

        self._ensure_initialized()
        run_async_task(self._async_onload, tags)

    async def _async_onload(self, tags: list[str] | None = None) -> None:
        targets = self._snapshot_data_proxy_control_targets()
        if not targets:
            return
        payload: dict = {"tags": tags} if tags is not None else {}
        results = await asyncio.gather(
            *(
                self._async_data_proxy_post(
                    addr, worker_id, "/resume_memory_occupation", payload
                )
                for addr, worker_id in targets
            ),
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        for r in failed:
            logger.error("Failed to onload a worker: %s", r)
        if failed and len(failed) == len(results):
            raise RuntimeError(f"onload failed on ALL {len(failed)} workers")

    def pause_generation(self) -> None:
        """Pause generation on all workers."""
        from areal.infra.utils.concurrent import run_async_task

        self._ensure_initialized()
        run_async_task(self._async_pause_generation)

    async def _async_pause_generation(self) -> None:
        targets = self._snapshot_data_proxy_control_targets()
        if not targets:
            return
        results = await asyncio.gather(
            *[
                self._async_data_proxy_post(addr, worker_id, "/pause_generation", {})
                for addr, worker_id in targets
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        for r in failed:
            logger.error("Failed to pause generation on a worker: %s", r)
        if failed and len(failed) == len(results):
            raise RuntimeError(f"pause_generation failed on ALL {len(failed)} workers")

    def continue_generation(self) -> None:
        """Continue generation on all workers."""
        from areal.infra.utils.concurrent import run_async_task

        self._ensure_initialized()
        run_async_task(self._async_continue_generation)

    async def _async_continue_generation(self) -> None:
        targets = self._snapshot_data_proxy_control_targets()
        if not targets:
            return
        results = await asyncio.gather(
            *[
                self._async_data_proxy_post(addr, worker_id, "/continue_generation", {})
                for addr, worker_id in targets
            ],
            return_exceptions=True,
        )
        failed = [r for r in results if isinstance(r, Exception)]
        for r in failed:
            logger.error("Failed to continue generation on a worker: %s", r)
        if failed and len(failed) == len(results):
            raise RuntimeError(
                f"continue_generation failed on ALL {len(failed)} workers"
            )

    # -- Stats -------------------------------------------------------------

    def export_stats(self) -> dict[str, float]:
        """Export and reset statistics recorded by the local workflow executor."""
        return stats_tracker.export_all()

    def config_perf_tracer(self, config: Any = None, role: str = "") -> None:
        """No-op — gateway does not have per-worker perf tracing."""

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        """No-op."""

    # -- Proxy compatibility (gateway IS the proxy) ------------------------

    @property
    def proxy_gateway_addr(self) -> str:
        self._ensure_initialized()
        return self._gateway_addr

    # -- Properties --------------------------------------------------------

    @property
    def inference_worker_urls(self) -> list[str]:
        self._ensure_initialized()
        return list(self._inf_addrs)

    @property
    def inference_guard_addrs(self) -> list[str]:
        self._ensure_initialized()
        return [
            f"http://{format_hostport(w.ip, int(w.worker_ports[0]))}"
            for w in self.workers
        ]

    @property
    def server_infos(self) -> list[LocalInfServerInfo]:
        self._ensure_initialized()
        return self._server_infos

    @property
    def worker_ids(self) -> dict[str, str]:
        """Return mapping from data proxy address to active worker_id."""
        self._ensure_initialized()
        with self._registration_state_lock:
            return dict(self._worker_ids)

    @property
    def staleness_manager(self):
        self._ensure_initialized()
        return self._staleness_manager

    @property
    def workflow_executor(self):
        self._ensure_initialized()
        if self._workflow_executor is None:
            raise RuntimeError("RolloutControllerV2.initialize() must be called first")
        return self._workflow_executor

    # -- Workflow resolution helpers ----------------------------------------

    def _wrap_agent(self, agent: Any, group_size: int = 1):
        """Wrap an agent in an InferenceServiceWorkflow.

        Parameters
        ----------
        agent : Any
            The agent to wrap (any object with an async ``run()`` method).
        group_size : int
            Number of parallel trajectories per episode.
        """
        from areal.v2.inference_service.controller.workflow import (
            InferenceServiceWorkflow,
        )

        if not self._gateway_addr:
            raise ValueError(
                "Gateway address is unavailable; initialize the controller first"
            )

        agent_cfg = self._agent_config
        admin_api_key = self.config.admin_api_key
        turn_discount = agent_cfg.turn_discount
        export_style = agent_cfg.export_style

        return InferenceServiceWorkflow(
            controller=self,
            agent=agent,
            gateway_addr=self._gateway_addr,
            admin_api_key=admin_api_key,
            discount=turn_discount,
            export_style=export_style,
            group_size=group_size,
        )

    def _resolve_workflow(
        self,
        workflow,
        workflow_kwargs=None,
        group_size=1,
    ):
        """Resolve a workflow-like input to an InferenceServiceWorkflow.

        Unlike ``RolloutController._resolve_workflow``, this method does
        **not** accept ``RolloutWorkflow`` instances or subclasses directly.
        It accepts agent objects/classes with an async ``run()`` method, or
        ``None`` for online mode.

        Parameters
        ----------
        workflow : Any
            An agent instance, agent class, import-path string, or ``None``.
        workflow_kwargs : dict, optional
            Keyword arguments passed to the agent constructor.
        group_size : int
            Number of times to run the workflow per input.
        """
        from areal.api.workflow_api import RolloutWorkflow
        from areal.utils.dynamic_import import import_from_string

        # External mode only supports online mode (workflow=None)
        if self.external_mode and workflow is not None:
            raise ValueError(
                "External model mode only supports online mode (workflow=None). "
                "Agent-based workflows are not supported with external models."
            )

        if self.external_mode and group_size > 1:
            raise ValueError(
                "External model mode requires group_size=1, "
                f"got group_size={group_size}."
            )

        # (a) None → online mode: create InferenceServiceWorkflow without agent
        if workflow is None:
            if group_size > 1:
                raise ValueError(
                    "Online mode (workflow=None) does not support group_size > 1. "
                    f"Got group_size={group_size}."
                )

            from areal.v2.inference_service.controller.workflow import (
                InferenceServiceWorkflow,
            )

            online_kwargs = dict(workflow_kwargs or {})
            online_kwargs.pop("controller", None)
            online_kwargs.setdefault("timeout", self.config.request_timeout)
            return InferenceServiceWorkflow(
                controller=self,
                agent=None,
                gateway_addr=self._gateway_addr,
                admin_api_key=self.config.admin_api_key,
                **online_kwargs,
            )

        # (b) Resolve workflow input (string import path, class, or instance).
        #     Defer instantiation until after the RolloutWorkflow guard.
        if isinstance(workflow, str):
            agent = import_from_string(workflow)
        else:
            agent = workflow

        # (c) Reject RolloutWorkflow classes and instances
        if isinstance(agent, type) and issubclass(agent, RolloutWorkflow):
            raise TypeError(
                "RolloutControllerV2 only accepts agent classes or instances with a "
                "run() method or None for online mode; direct RolloutWorkflow "
                "classes are not supported"
            )
        if isinstance(agent, RolloutWorkflow):
            raise TypeError(
                "RolloutControllerV2 only accepts agent classes or instances with a "
                "run() method or None for online mode; direct RolloutWorkflow "
                "instances are not supported"
            )

        if isinstance(agent, type):
            agent = agent(**(workflow_kwargs or {}))
        if not callable(getattr(agent, "run", None)):
            raise TypeError(
                f"workflow must be an agent with a callable run() method. "
                f"Got workflow={workflow!r}"
            )

        # (d) Wrap the agent in InferenceServiceWorkflow (with group_size)
        resolved = self._wrap_agent(agent, group_size=group_size)

        return resolved

    @staticmethod
    def _resolve_should_accept_fn(
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None,
    ) -> Callable[[dict[str, Any]], bool] | None:
        """Resolve should_accept_fn to a callable or None."""
        if should_accept_fn is None:
            return None
        if callable(should_accept_fn):
            return should_accept_fn
        if isinstance(should_accept_fn, str):
            from areal.utils.dynamic_import import import_from_string

            func = import_from_string(should_accept_fn)
            if not callable(func):
                raise TypeError(f"Imported {should_accept_fn!r} is not callable")
            return cast(Callable[[dict[str, Any]], bool], func)
        raise TypeError(f"Invalid should_accept_fn type: {type(should_accept_fn)}")

    @property
    def _agent_config(self):
        return self.config.agent

    # -- Internal HTTP helpers ---------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
    ) -> tuple[str, int]:
        """Fork a process on a RPCGuard worker via ``/fork`` with ``raw_cmd``.

        Returns ``(host, port)`` of the forked service and records the entry
        in ``_forked_services`` for cleanup.
        """
        resp = self._sync_client.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        resp = self._sync_client.post(
            f"{guard_addr}/fork",
            json={
                "role": role,
                "worker_index": worker_index,
                "raw_cmd": cmd,
            },
        )
        resp.raise_for_status()
        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(
            f"{addr}{health_path}",
            role,
            process_owners=((guard_addr, role, worker_index),),
        )

        return host, port

    async def _async_fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
    ) -> tuple[str, int]:
        """Async fork a process on a RPCGuard worker via ``/fork``.

        Returns ``(host, port)`` of the forked service. Ownership is recorded
        immediately after the guard acknowledges the fork and before readiness
        polling begins.
        """
        client = await self._get_async_client()
        resp = await client.post(
            f"{guard_addr}/alloc_ports", json={"count": 1}, timeout=30.0
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]
        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }

        resp = await client.post(f"{guard_addr}/fork", json=fork_payload, timeout=30.0)
        resp.raise_for_status()
        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        await self._async_wait_for_service(
            f"{addr}{health_path}",
            role,
            process_owners=((guard_addr, role, worker_index),),
        )

        return host, port

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        if getattr(self._sync_client, "is_closed", False) is True:
            # A concurrent destroy may close the client before the initializer
            # publishes a late owner.  The initializer's terminal cleanup pass
            # must still be able to reap that acknowledged child.
            self._sync_client = httpx.Client(timeout=30.0)
        try:
            resp = self._sync_client.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Killed forked service %s/%d", role, worker_index)
        except httpx.HTTPError as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )
            raise

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client, recreating it when the event loop changes.

        ``run_async_task`` creates a fresh event loop per invocation via
        ``asyncio.run()``.  TCP connections are tied to the loop that opened
        them, so we must recreate the client when the loop changes.  Within
        a single loop lifetime all callers (e.g. concurrent ``asyncio.gather``
        calls) share one client and its connection pool.
        """
        current_loop = asyncio.get_running_loop()
        with self._async_client_lock:
            client = self._async_client
            cleanup = self._async_client_cleanup
            owner_loop = self._async_client_loop
            if client is not None:
                if cleanup is None or owner_loop is not cleanup.owner_loop:
                    raise RuntimeError(
                        "async HTTP client cleanup state is inconsistent"
                    )
                state, error = cleanup.snapshot()
                if state is HttpxAsyncClientCleanupState.FAILED:
                    assert error is not None
                    raise error
                if state is HttpxAsyncClientCleanupState.SUCCEEDED:
                    self._async_client = None
                    self._async_client_loop = None
                    self._async_client_cleanup = None
                elif cleanup.owner_loop is current_loop:
                    if state is HttpxAsyncClientCleanupState.CLOSING:
                        raise RuntimeError(
                            "async HTTP client cleanup is already in progress"
                        )
                    return client
                elif cleanup.owner_loop.is_running():
                    raise RuntimeError(
                        "async HTTP client is active on another event loop"
                    )
                elif cleanup.owner_loop.is_closed():
                    raise RuntimeError(
                        "async HTTP client owner event loop closed before cleanup "
                        "succeeded"
                    )
                else:
                    raise RuntimeError(
                        "async HTTP client owner event loop is not running"
                    )
            elif cleanup is not None or owner_loop is not None:
                raise RuntimeError("async HTTP client cleanup state is inconsistent")

            client = create_httpx_client(timeout=self.config.request_timeout)
            cleanup = HttpxAsyncClientCleanup(client, current_loop)

            def clear_client() -> None:
                with self._async_client_lock:
                    if self._async_client_cleanup is cleanup:
                        self._async_client = None
                        self._async_client_loop = None
                        self._async_client_cleanup = None

            register_httpx_client_loop_cleanup(cleanup, on_closed=clear_client)
            self._async_client = client
            self._async_client_loop = current_loop
            self._async_client_cleanup = cleanup
            return client

    def _snapshot_data_proxy_control_targets(self) -> tuple[tuple[str, str], ...]:
        """Freeze direct-control destinations with their confirmed incarnations.

        Address-only forwarding is unsafe: the process listening at an address may
        have been replaced after Router registration.  Copy both fields under the
        registration-state lock so every request in one broadcast remains pinned to
        the same confirmed incarnation.  An unconfirmed address fails the whole
        broadcast before any worker can be mutated.
        """
        with self._registration_state_lock:
            targets: list[tuple[str, str]] = []
            for addr in self._data_proxy_addrs:
                worker_id = self._worker_ids.get(addr)
                if not worker_id:
                    raise RuntimeError(
                        "Cannot send a direct Data Proxy control request: "
                        f"{addr} has no router-confirmed worker identity"
                    )
                targets.append((addr, worker_id))
            return tuple(targets)

    @async_http_retry
    async def _async_data_proxy_post(
        self,
        addr: str,
        worker_id: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> None:
        """POST directly to a data proxy, bypassing gateway/router resolution."""
        url = f"{addr}{endpoint}"
        try:
            client = await self._get_async_client()
            resp = await client.post(
                url,
                json=payload,
                headers={WORKER_ID_HEADER: worker_id},
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Data proxy {url} returned {resp.status_code}: {resp.text}"
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to POST {url}: {exc}") from exc
