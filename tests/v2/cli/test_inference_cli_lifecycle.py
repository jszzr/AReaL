# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import click
import pytest

from areal.v2.cli.inference.scheduler import TaskAllocation, TaskHandle
from areal.v2.cli.inference.state import (
    ModelEntry,
    ModelReplica,
    ModelState,
    ServiceState,
)


def _service_state(gateway_pid: int = 30, router_pid: int = 40) -> ServiceState:
    return ServiceState(
        service="svc",
        backend="local",
        gateway_handle=TaskHandle(
            host="127.0.0.1", ports=[8080], gpu_devices=[], ref={"pid": gateway_pid}
        ),
        router_handle=TaskHandle(
            host="127.0.0.1", ports=[9000], gpu_devices=[], ref={"pid": router_pid}
        ),
        admin_api_key="admin",
        started_at=1.0,
    )


def _replica(
    *, worker_pid: int, proxy_pid: int, router_worker_id: str | None = None
) -> ModelReplica:
    return ModelReplica(
        data_proxy=TaskHandle(
            host="127.0.0.1", ports=[5001], gpu_devices=[], ref={"pid": proxy_pid}
        ),
        worker=TaskHandle(
            host="127.0.0.1", ports=[5000], gpu_devices=[0], ref={"pid": worker_pid}
        ),
        router_worker_id=router_worker_id,
    )


class _Scheduler:
    def __init__(self, handles: list[TaskHandle]) -> None:
        self._handles = iter(handles)
        self.submitted_specs = []

    def submit(self, spec):
        self.submitted_specs.append(spec)
        return next(self._handles)


def _worker_handle(*, rank: int) -> TaskHandle:
    return TaskHandle(
        host="127.0.0.1",
        ports=[6000 + rank],
        gpu_devices=[rank],
        ref={"pid": 100 + rank},
    )


def _proxy_handle(*, rank: int) -> TaskHandle:
    return TaskHandle(
        host="127.0.0.1",
        ports=[7000 + rank],
        gpu_devices=[],
        ref={"pid": 200 + rank},
    )


def _register_internal(common, tmp_path, *, backend, scheduler, router, gateway):
    return common.register_internal(
        model="model",
        backend=backend,
        model_path="/model",
        tokenizer_path="/model",
        engine_extra=[],
        proxy_extra=[],
        model_health_timeout=1.0,
        log_level="info",
        admin_api_key="admin",
        gateway=gateway,
        router=router,
        log_dir=tmp_path,
        scheduler=scheduler,
    )


def test_register_internal_preserves_worker_identity_from_proxy_launch_to_router(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    uuid_calls = []

    def fake_uuid4():
        uuid_calls.append(None)
        return "proxy-incarnation-1"

    router = SimpleNamespace(register_worker=lambda *args: None)
    router_calls = []

    def register_worker(addr, worker_id, expected_worker_id):
        router_calls.append((addr, worker_id, expected_worker_id))
        return {"status": "ok", "worker_id": worker_id}

    router.register_worker = register_worker
    gateway_calls = []
    gateway = SimpleNamespace(
        register_model=lambda payload: gateway_calls.append(payload)
    )
    scheduler = _Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)])
    monkeypatch.setattr(common, "uuid4", fake_uuid4, raising=False)
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)

    replicas = _register_internal(
        common,
        tmp_path,
        backend="sglang:d1",
        scheduler=scheduler,
        router=router,
        gateway=gateway,
    )

    assert len(uuid_calls) == 1
    proxy_cmd = scheduler.submitted_specs[1].cmd_builder(
        TaskAllocation(host="127.0.0.1", ports=[7000], gpu_devices=[])
    )
    assert "--worker-id" in proxy_cmd
    assert proxy_cmd[proxy_cmd.index("--worker-id") + 1] == "proxy-incarnation-1"
    assert router_calls == [("http://127.0.0.1:7000", "proxy-incarnation-1", None)]
    assert replicas[0].router_worker_id == "proxy-incarnation-1"
    assert gateway_calls == [
        {
            "model": "model",
            "url": "",
            "api_key": "",
            "data_proxy_addrs": ["http://127.0.0.1:7000"],
        }
    ]


def test_data_proxy_worker_identity_cannot_be_overridden_by_proxy_extra(tmp_path):
    from areal.v2.cli.inference.launcher import build_data_proxy_task_spec

    spec = build_data_proxy_task_spec(
        name="data_proxy/model/0",
        worker_id="proxy-incarnation-1",
        backend_addr="http://127.0.0.1:6000",
        backend_type="sglang",
        tokenizer_path="/model",
        admin_api_key="admin",
        log_level="info",
        extra_args=["--worker-id", "spoofed-incarnation"],
        log_file=tmp_path / "proxy.log",
    )

    proxy_cmd = spec.cmd_builder(
        TaskAllocation(host="127.0.0.1", ports=[7000], gpu_devices=[])
    )

    # argparse uses the final occurrence, so the launcher's immutable identity
    # must be appended after all user-supplied proxy arguments.
    assert proxy_cmd[-2:] == ["--worker-id", "proxy-incarnation-1"]


def test_register_internal_rejects_mismatched_router_worker_id_echo(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    router = SimpleNamespace(
        register_worker=lambda addr, worker_id, expected_worker_id: {
            "status": "ok",
            "worker_id": "different-incarnation",
        }
    )
    gateway_calls = []
    gateway = SimpleNamespace(
        register_model=lambda payload: gateway_calls.append(payload)
    )
    kill_calls = []
    scheduler = _Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)])
    monkeypatch.setattr(common, "uuid4", lambda: "proxy-incarnation-1")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: kill_calls.append((list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="unexpected worker_id"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=scheduler,
            router=router,
            gateway=gateway,
        )

    assert gateway_calls == []
    assert kill_calls == [([100, 200], 10.0)]


def test_register_internal_rolls_back_exact_router_owner_before_killing_on_gateway_failure(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []

    def register_worker(addr, worker_id, expected_worker_id):
        events.append(("register", addr, worker_id, expected_worker_id))
        return {"status": "ok", "worker_id": worker_id}

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        raise RuntimeError("malformed rollback response")

    def register_model(payload):
        events.append(("gateway", payload))
        raise common.ServiceUnreachable("gateway down")

    router = SimpleNamespace(
        register_worker=register_worker, unregister_worker=unregister_worker
    )
    gateway = SimpleNamespace(register_model=register_model)
    scheduler = _Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)])
    monkeypatch.setattr(common, "uuid4", lambda: "proxy-incarnation-1")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="gateway register_model failed"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=scheduler,
            router=router,
            gateway=gateway,
        )

    assert events[0] == (
        "register",
        "http://127.0.0.1:7000",
        "proxy-incarnation-1",
        None,
    )
    assert events[-2:] == [
        ("unregister", "http://127.0.0.1:7000", "proxy-incarnation-1"),
        ("kill", [100, 200], 10.0),
    ]


def test_register_internal_rolls_back_only_confirmed_owner_on_partial_registration_failure(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []
    worker_ids = iter(["proxy-incarnation-1", "proxy-incarnation-2"])

    def register_worker(addr, worker_id, expected_worker_id):
        events.append(("register", addr, worker_id, expected_worker_id))
        if worker_id == "proxy-incarnation-2":
            raise common.ServiceHTTPError(503, "router unavailable")
        return {"status": "ok", "worker_id": worker_id}

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        return {"status": "ok", "removed": True}

    router = SimpleNamespace(
        register_worker=register_worker, unregister_worker=unregister_worker
    )
    gateway = SimpleNamespace(
        register_model=lambda payload: events.append(("gateway", payload))
    )
    scheduler = _Scheduler(
        [
            _worker_handle(rank=0),
            _proxy_handle(rank=0),
            _worker_handle(rank=1),
            _proxy_handle(rank=1),
        ]
    )
    monkeypatch.setattr(common, "uuid4", lambda: next(worker_ids))
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="router register_worker"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d2",
            scheduler=scheduler,
            router=router,
            gateway=gateway,
        )

    assert [event for event in events if event[0] == "unregister"] == [
        ("unregister", "http://127.0.0.1:7000", "proxy-incarnation-1")
    ]
    assert events[-1] == ("kill", [100, 200, 101, 201], 10.0)
    assert not any(event[0] == "gateway" for event in events)


def test_deregister_uses_exact_router_worker_id_before_killing(tmp_path, monkeypatch):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="proxy-incarnation-1",
    )
    models = {
        "model": ModelEntry(backend="sglang:d1", replicas=[replica]),
    }
    model_state = SimpleNamespace(
        models=models,
        save=lambda: events.append(("save",)),
    )
    state = SimpleNamespace(
        service="svc",
        models=models,
        model_state=model_state,
        router_url="http://router",
        admin_api_key="admin",
    )

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=lambda addr, worker_id: events.append(
            ("unregister", addr, worker_id)
        ),
    )
    monkeypatch.setattr(
        deregister.inf_lifecycle, "load_running_state", lambda service: state
    )
    monkeypatch.setattr(deregister, "RouterClient", lambda *args: router)
    monkeypatch.setattr(
        deregister,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids), grace_s)),
    )

    result = deregister.do_deregister("model", 7.0, False, service="svc")

    assert result == 0
    assert events == [
        ("remove_model", "model"),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("kill", [20], 7.0),
        ("kill", [10], 7.0),
        ("save",),
    ]
    assert models == {}


def test_deregister_legacy_replica_warns_and_skips_router_unregister(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    warnings = []
    replica = _replica(worker_pid=10, proxy_pid=20, router_worker_id=None)
    models = {
        "model": ModelEntry(backend="sglang:d1", replicas=[replica]),
    }
    model_state = SimpleNamespace(
        models=models,
        save=lambda: events.append(("save",)),
    )
    state = SimpleNamespace(
        service="svc",
        models=models,
        model_state=model_state,
        router_url="http://router",
        admin_api_key="admin",
    )
    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=lambda addr, worker_id: events.append(
            ("unregister", addr, worker_id)
        ),
    )
    monkeypatch.setattr(
        deregister.inf_lifecycle, "load_running_state", lambda service: state
    )
    monkeypatch.setattr(deregister, "RouterClient", lambda *args: router)
    monkeypatch.setattr(
        deregister,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids), grace_s)),
    )
    monkeypatch.setattr(
        deregister.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    result = deregister.do_deregister("model", 7.0, False, service="svc")

    assert result == 0
    assert not any(event[0] == "unregister" for event in events)
    assert len(warnings) == 1
    assert "router_worker_id" in warnings[0][0]
    assert warnings[0][1] == ("http://127.0.0.1:5001",)
    assert events[-3:] == [
        ("kill", [20], 7.0),
        ("kill", [10], 7.0),
        ("save",),
    ]
    assert models == {}


def test_deregister_holds_model_state_lock_through_load_unregister_kill_and_save(
    monkeypatch,
):
    from areal.v2.cli.inference.commands import deregister

    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="proxy-incarnation-1",
    )
    models = {
        "model": ModelEntry(backend="sglang:d1", replicas=[replica]),
    }
    model_state = SimpleNamespace(
        models=models,
        save=lambda: events.append(("save",)),
    )
    state = SimpleNamespace(
        service="svc",
        models=models,
        model_state=model_state,
        router_url="http://router",
        admin_api_key="admin",
    )

    @contextmanager
    def lock_model_state(service):
        events.append(("lock_enter", service))
        try:
            yield
        finally:
            events.append(("lock_exit", service))

    fake_store = SimpleNamespace(lock_model_state=lock_model_state)

    def resolve_service_name(service):
        events.append(("resolve", service))
        return "svc"

    def load_running_state(service):
        events.append(("load", service))
        return state

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=lambda addr, worker_id: events.append(
            ("unregister", addr, worker_id)
        ),
    )
    monkeypatch.setattr(
        deregister.inf_lifecycle, "resolve_service_name", resolve_service_name
    )
    monkeypatch.setattr(
        deregister.inf_lifecycle, "load_running_state", load_running_state
    )
    monkeypatch.setattr(deregister, "store", fake_store, raising=False)
    monkeypatch.setattr(deregister, "RouterClient", lambda *args: router)
    monkeypatch.setattr(
        deregister,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids), grace_s)),
    )

    result = deregister.do_deregister("model", 7.0, False, service=None)

    assert result == 0
    assert events == [
        ("resolve", None),
        ("lock_enter", "svc"),
        ("load", "svc"),
        ("remove_model", "model"),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("kill", [20], 7.0),
        ("kill", [10], 7.0),
        ("save",),
        ("lock_exit", "svc"),
    ]


def test_terminate_runtime_state_kills_in_order(monkeypatch):
    from areal.v2.cli.inference import common
    from areal.v2.cli.inference.state import RuntimeState

    service_state = _service_state(gateway_pid=30, router_pid=40)
    model_state = ModelState(
        service="svc",
        models={
            "m": ModelEntry(
                backend="sglang:d1",
                replicas=[_replica(worker_pid=10, proxy_pid=20)],
            )
        },
    )
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(common, "kill_pids", fake_kill_pids)

    common.terminate_runtime_state(
        RuntimeState(service_state=service_state, model_state=model_state),
        grace_s=7.0,
    )

    # Data-flow order: data_proxy → worker → gateway → router
    assert calls == [([20], 7.0), ([10], 7.0), ([30], 7.0), ([40], 7.0)]


def test_prepare_service_slot_force_recovers_raw_pids(tmp_path, monkeypatch):
    from areal.v2.cli.inference import lifecycle as inf_lifecycle_mod
    from areal.v2.cli.inference.lifecycle import inf_lifecycle
    from areal.v2.cli.inference.state import store

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    # Omit the ``backend`` field so ServiceState.load raises KeyError and
    # force_replace_slot falls back to recover_pids_from_raw_state.
    store.service_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "gateway_handle": {
                    "host": "127.0.0.1",
                    "ports": [8080],
                    "gpu_devices": [],
                    "ref": {"pid": 100},
                },
                "router_handle": {
                    "host": "127.0.0.1",
                    "ports": [9000],
                    "gpu_devices": [],
                    "ref": {"pid": 101},
                },
                "admin_api_key": "admin",
                "started_at": 1.0,
            }
        )
    )
    store.models_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "models": {
                    "m": {
                        "backend": "sglang:d1",
                        "replicas": [
                            {
                                "data_proxy": {
                                    "host": "127.0.0.1",
                                    "ports": [5001],
                                    "gpu_devices": [],
                                    "ref": {"pid": 300},
                                },
                                "worker": {
                                    "host": "127.0.0.1",
                                    "ports": [5000],
                                    "gpu_devices": [0],
                                    "ref": {"pid": 200},
                                },
                            }
                        ],
                    }
                },
            }
        )
    )
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(inf_lifecycle_mod, "kill_pids", fake_kill_pids)

    inf_lifecycle.force_replace_slot("svc", grace_s=5.0)

    assert calls == [([100, 101, 300, 200], 5.0)]
    assert not store.service_state_path("svc").exists()
    assert not store.models_state_path("svc").exists()


def test_foreground_cleanup_uses_latest_model_state(tmp_path, monkeypatch):
    from areal.v2.cli.inference import common
    from areal.v2.cli.inference.commands import run

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    service_state = _service_state(gateway_pid=30, router_pid=40)
    latest = ModelState(service="svc")
    latest.models["later"] = ModelEntry(
        backend="sglang:d1",
        replicas=[_replica(worker_pid=10, proxy_pid=20)],
    )
    latest.save()
    calls = []

    def fake_kill_pids(pids, grace_s):
        calls.append((list(pids), grace_s))

    monkeypatch.setattr(common, "kill_pids", fake_kill_pids)

    run._cleanup_runtime("svc", service_state, ModelState(service="svc"), grace_s=3.0)

    # Same data-flow order as terminate_runtime_state.
    assert calls == [([20], 3.0), ([10], 3.0), ([30], 3.0), ([40], 3.0)]
