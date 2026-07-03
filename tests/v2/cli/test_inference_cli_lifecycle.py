# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
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


def _unseen_epoch(addr: str) -> dict[str, str | None]:
    return {"worker_addr": addr, "status": "unseen", "worker_id": None}


def _register_internal(
    common,
    tmp_path,
    *,
    backend,
    scheduler,
    router,
    gateway,
    persist_entry=None,
):
    kwargs = {}
    if persist_entry is not None:
        kwargs["persist_entry"] = persist_entry
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
        **kwargs,
    )


def test_register_internal_preserves_worker_identity_from_proxy_launch_to_router(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    uuid_calls = []

    def fake_uuid4():
        uuid_calls.append(None)
        return "proxy-incarnation-1"

    router = SimpleNamespace(
        get_worker_epoch=_unseen_epoch,
        register_worker=lambda *args: None,
        unregister_worker=lambda *args: {"removed": False},
    )
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


def test_register_internal_reads_retired_predecessor_once_after_proxy_health(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []

    def get_worker_epoch(addr):
        events.append(("epoch", addr))
        return {
            "worker_addr": addr,
            "status": "retired",
            "worker_id": "retired-incarnation",
        }

    def register_worker(addr, worker_id, expected_worker_id):
        events.append(("register", addr, worker_id, expected_worker_id))
        return {"status": "ok", "worker_id": worker_id}

    router = SimpleNamespace(
        get_worker_epoch=get_worker_epoch,
        register_worker=register_worker,
        unregister_worker=lambda *args: {"removed": False},
    )
    gateway = SimpleNamespace(register_model=lambda payload: None)
    scheduler = _Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)])
    monkeypatch.setattr(common, "uuid4", lambda: "new-incarnation")
    monkeypatch.setattr(
        common,
        "wait_http_health",
        lambda url, **kwargs: events.append(
            ("health", url, kwargs.get("expected_worker_id"))
        ),
    )

    _register_internal(
        common,
        tmp_path,
        backend="sglang:d1",
        scheduler=scheduler,
        router=router,
        gateway=gateway,
    )

    assert events == [
        ("health", "http://127.0.0.1:6000", None),
        ("health", "http://127.0.0.1:7000", "new-incarnation"),
        ("epoch", "http://127.0.0.1:7000"),
        (
            "register",
            "http://127.0.0.1:7000",
            "new-incarnation",
            "retired-incarnation",
        ),
    ]


def test_register_internal_does_not_reread_epoch_or_retry_after_cas_conflict(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []

    def get_worker_epoch(addr):
        events.append(("epoch", addr))
        return {
            "worker_addr": addr,
            "status": "active",
            "worker_id": "observed-predecessor",
        }

    def register_worker(addr, worker_id, expected_worker_id):
        events.append(("register", addr, worker_id, expected_worker_id))
        raise common.ServiceHTTPError(409, "epoch changed")

    router = SimpleNamespace(
        get_worker_epoch=get_worker_epoch,
        register_worker=register_worker,
        unregister_worker=lambda addr, worker_id: events.append(
            ("unregister", addr, worker_id)
        ),
    )
    gateway = SimpleNamespace(register_model=lambda payload: None)
    scheduler = _Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)])
    monkeypatch.setattr(common, "uuid4", lambda: "new-incarnation")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(common, "kill_pids", lambda *args, **kwargs: None)

    with pytest.raises(click.ClickException, match="router register_worker"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=scheduler,
            router=router,
            gateway=gateway,
        )

    assert [event for event in events if event[0] == "epoch"] == [
        ("epoch", "http://127.0.0.1:7000"),
        ("epoch", "http://127.0.0.1:7000"),
    ]
    assert [event for event in events if event[0] == "register"] == [
        (
            "register",
            "http://127.0.0.1:7000",
            "new-incarnation",
            "observed-predecessor",
        )
    ]
    assert (
        "unregister",
        "http://127.0.0.1:7000",
        "new-incarnation",
    ) in events


def test_register_internal_rejects_duplicate_data_proxy_addresses(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []
    worker_ids = iter(["incarnation-1", "incarnation-2"])
    router = SimpleNamespace(
        get_worker_epoch=lambda addr: events.append(("epoch", addr))
        or _unseen_epoch(addr),
        register_worker=lambda *args: events.append(("register", *args)),
        unregister_worker=lambda addr, worker_id: events.append(
            ("unregister", addr, worker_id)
        ),
    )
    gateway = SimpleNamespace(register_model=lambda payload: None)
    scheduler = _Scheduler(
        [
            _worker_handle(rank=0),
            _proxy_handle(rank=0),
            _worker_handle(rank=1),
            _proxy_handle(rank=0),
        ]
    )
    monkeypatch.setattr(common, "uuid4", lambda: next(worker_ids))
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(common, "kill_pids", lambda *args, **kwargs: None)

    with pytest.raises(click.ClickException, match="duplicate data-proxy address"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d2",
            scheduler=scheduler,
            router=router,
            gateway=gateway,
        )

    assert not any(event[0] == "register" for event in events)
    assert [event for event in events if event[0] == "unregister"] == [
        ("unregister", "http://127.0.0.1:7000", "incarnation-2"),
        ("unregister", "http://127.0.0.1:7000", "incarnation-1"),
    ]


def test_register_internal_rejects_mismatched_router_worker_id_echo(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    router = SimpleNamespace(
        get_worker_epoch=_unseen_epoch,
        register_worker=lambda addr, worker_id, expected_worker_id: {
            "status": "ok",
            "worker_id": "different-incarnation",
        },
        unregister_worker=lambda *args: {"removed": False},
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
        get_worker_epoch=_unseen_epoch,
        register_worker=register_worker,
        unregister_worker=unregister_worker,
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
    assert events[-1] == (
        "unregister",
        "http://127.0.0.1:7000",
        "proxy-incarnation-1",
    )
    assert not any(event[0] == "kill" for event in events)


def test_register_internal_rolls_back_every_generated_owner_on_ambiguous_registration_failure(
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
        get_worker_epoch=_unseen_epoch,
        register_worker=register_worker,
        unregister_worker=unregister_worker,
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
        ("unregister", "http://127.0.0.1:7001", "proxy-incarnation-2"),
        ("unregister", "http://127.0.0.1:7000", "proxy-incarnation-1"),
    ]
    assert events[-1] == ("kill", [100, 200, 101, 201], 10.0)
    assert not any(event[0] == "gateway" for event in events)


def test_register_response_loss_rolls_back_then_next_launch_cas_reuses_address(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    class StatefulRouter:
        def __init__(self):
            self.status = "unseen"
            self.worker_id = None
            self.drop_first_register_response = True
            self.register_payloads = []

        def get_worker_epoch(self, addr):
            return {
                "worker_addr": addr,
                "status": self.status,
                "worker_id": self.worker_id,
            }

        def register_worker(self, addr, worker_id, expected_worker_id):
            self.register_payloads.append((addr, worker_id, expected_worker_id))
            assert expected_worker_id == self.worker_id
            self.status = "active"
            self.worker_id = worker_id
            if self.drop_first_register_response:
                self.drop_first_register_response = False
                raise common.ServiceUnreachable("response lost after commit")
            return {"status": "ok", "worker_id": worker_id}

        def unregister_worker(self, addr, worker_id):
            if self.status == "active" and self.worker_id == worker_id:
                self.status = "retired"
                return {"status": "ok", "removed": True}
            return {"status": "ok", "removed": False}

    router = StatefulRouter()
    gateway_calls = []
    gateway = SimpleNamespace(
        register_model=lambda payload: gateway_calls.append(payload)
    )
    worker_ids = iter(["incarnation-e1", "incarnation-e2"])
    monkeypatch.setattr(common, "uuid4", lambda: next(worker_ids))
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(common, "kill_pids", lambda *args, **kwargs: None)

    with pytest.raises(click.ClickException, match="response lost after commit"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=_Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)]),
            router=router,
            gateway=gateway,
        )
    assert router.status == "retired"
    assert router.worker_id == "incarnation-e1"

    replicas = _register_internal(
        common,
        tmp_path,
        backend="sglang:d1",
        scheduler=_Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)]),
        router=router,
        gateway=gateway,
    )

    assert router.register_payloads == [
        ("http://127.0.0.1:7000", "incarnation-e1", None),
        ("http://127.0.0.1:7000", "incarnation-e2", "incarnation-e1"),
    ]
    assert router.status == "active"
    assert router.worker_id == "incarnation-e2"
    assert replicas[0].router_worker_id == "incarnation-e2"
    assert len(gateway_calls) == 1


def test_register_response_loss_with_unconfirmed_cleanup_persists_pending_state(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    class AmbiguousRouter:
        def __init__(self):
            self.worker_id = None

        def get_worker_epoch(self, addr):
            return {
                "worker_addr": addr,
                "status": "active" if self.worker_id else "unseen",
                "worker_id": self.worker_id,
            }

        def register_worker(self, addr, worker_id, expected_worker_id):
            self.worker_id = worker_id
            raise common.ServiceUnreachable("register response lost")

        def unregister_worker(self, addr, worker_id):
            raise common.ServiceUnreachable("cleanup response lost")

    snapshots = []
    kills = []
    monkeypatch.setattr(common, "uuid4", lambda: "ambiguous-incarnation")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: kills.append((list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="register response lost"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=_Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)]),
            router=AmbiguousRouter(),
            gateway=SimpleNamespace(register_model=lambda payload: None),
            persist_entry=lambda entry: snapshots.append(deepcopy(entry)),
        )

    assert kills == []
    pending = snapshots[-1]
    assert pending is not None
    assert pending.lifecycle_state == "CLEANUP_PENDING"
    assert pending.gateway_model_cleanup_pending is False
    assert pending.replicas[0].router_cleanup_pending is True
    assert pending.replicas[0].router_registration_ambiguous is True


def test_partial_multi_replica_registration_keeps_only_unconfirmed_owner_pending(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    worker_ids = iter(["confirmed-incarnation", "ambiguous-incarnation"])
    active_ids = {}

    def register_worker(addr, worker_id, expected_worker_id):
        active_ids[addr] = worker_id
        if worker_id == "ambiguous-incarnation":
            raise common.ServiceUnreachable("second response lost")
        return {"worker_id": worker_id}

    def unregister_worker(addr, worker_id):
        if worker_id == "confirmed-incarnation":
            active_ids.pop(addr, None)
            return {"removed": True}
        raise common.ServiceUnreachable("second cleanup uncertain")

    router = SimpleNamespace(
        get_worker_epoch=lambda addr: {
            "worker_addr": addr,
            "status": "active" if addr in active_ids else "unseen",
            "worker_id": active_ids.get(addr),
        },
        register_worker=register_worker,
        unregister_worker=unregister_worker,
    )
    snapshots = []
    kills = []
    monkeypatch.setattr(common, "uuid4", lambda: next(worker_ids))
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: kills.append((list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="second response lost"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d2",
            scheduler=_Scheduler(
                [
                    _worker_handle(rank=0),
                    _proxy_handle(rank=0),
                    _worker_handle(rank=1),
                    _proxy_handle(rank=1),
                ]
            ),
            router=router,
            gateway=SimpleNamespace(register_model=lambda payload: None),
            persist_entry=lambda entry: snapshots.append(deepcopy(entry)),
        )

    assert kills == []
    pending = snapshots[-1]
    assert pending.lifecycle_state == "CLEANUP_PENDING"
    assert [r.router_cleanup_pending for r in pending.replicas] == [False, True]
    assert [r.router_registration_ambiguous for r in pending.replicas] == [
        False,
        True,
    ]


def test_confirmed_registration_rollback_kills_before_deleting_durable_entry(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []
    router = SimpleNamespace(
        get_worker_epoch=lambda addr: {
            "worker_addr": addr,
            "status": "active",
            "worker_id": "concurrent-successor",
        },
        register_worker=lambda addr, worker_id, expected: (_ for _ in ()).throw(
            common.ServiceHTTPError(409, "epoch changed")
        ),
        unregister_worker=lambda addr, worker_id: {"removed": False},
    )
    monkeypatch.setattr(common, "uuid4", lambda: "rejected-incarnation")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: events.append(("kill", list(pids))),
    )

    with pytest.raises(click.ClickException, match="epoch changed"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=_Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)]),
            router=router,
            gateway=SimpleNamespace(register_model=lambda payload: None),
            persist_entry=lambda entry: events.append(
                ("persist", None if entry is None else entry.lifecycle_state)
            ),
        )

    assert events[-2:] == [("kill", [100, 200]), ("persist", None)]


def test_registration_cleanup_rejects_malformed_active_epoch():
    from areal.v2.cli.inference import common

    router = SimpleNamespace(
        unregister_worker=lambda addr, worker_id: {"removed": False},
        get_worker_epoch=lambda addr: {
            "worker_addr": addr,
            "status": "active",
            "worker_id": None,
        },
    )

    assert (
        common._confirm_worker_cleanup(
            router, "http://127.0.0.1:7000", "generated-incarnation"
        )
        is False
    )


def test_gateway_response_loss_retains_model_cleanup_pending_until_confirmed(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference import common

    events = []
    router = SimpleNamespace(
        get_worker_epoch=_unseen_epoch,
        register_worker=lambda addr, worker_id, expected: {"worker_id": worker_id},
        unregister_worker=lambda addr, worker_id: {"removed": True},
        remove_model=lambda name: (_ for _ in ()).throw(
            common.ServiceUnreachable("model cleanup uncertain")
        ),
    )

    def register_model(payload):
        events.append(("gateway-committed", payload))
        raise common.ServiceUnreachable("gateway response lost")

    snapshots = []
    kills = []
    monkeypatch.setattr(common, "uuid4", lambda: "proxy-incarnation-1")
    monkeypatch.setattr(common, "wait_http_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        common,
        "kill_pids",
        lambda pids, grace_s: kills.append((list(pids), grace_s)),
    )

    with pytest.raises(click.ClickException, match="gateway response lost"):
        _register_internal(
            common,
            tmp_path,
            backend="sglang:d1",
            scheduler=_Scheduler([_worker_handle(rank=0), _proxy_handle(rank=0)]),
            router=router,
            gateway=SimpleNamespace(register_model=register_model),
            persist_entry=lambda entry: snapshots.append(deepcopy(entry)),
        )

    assert events and kills == []
    pending = snapshots[-1]
    assert pending.lifecycle_state == "CLEANUP_PENDING"
    assert pending.gateway_model_cleanup_pending is True
    assert pending.replicas[0].router_cleanup_pending is False


def test_do_register_persists_cleanup_pending_entry_before_returning_error(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import register

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    models = {}
    saves = []
    model_state = SimpleNamespace(
        models=models,
        occupied_gpus=lambda: set(),
        save=lambda: saves.append(deepcopy(models)),
    )
    state = SimpleNamespace(
        service="svc",
        models=models,
        model_state=model_state,
        gateway_url="http://gateway",
        router_url="http://router",
        admin_api_key="admin",
        backend="local",
    )

    def register_model(*, persist_entry, **kwargs):
        persist_entry(
            ModelEntry(
                backend="sglang:d1",
                replicas=[],
                lifecycle_state="CLEANUP_PENDING",
                gateway_model_cleanup_pending=True,
            )
        )
        raise click.ClickException("registration cleanup pending")

    monkeypatch.setattr(
        register.inf_lifecycle, "load_running_state", lambda service: state
    )
    monkeypatch.setattr(register, "register_model", register_model)
    monkeypatch.setattr(register, "GatewayClient", lambda *args: object())
    monkeypatch.setattr(register, "RouterClient", lambda *args: object())

    with pytest.raises(click.ClickException, match="cleanup pending"):
        register.do_register("model", {}, service="svc")

    assert models["model"].lifecycle_state == "CLEANUP_PENDING"
    assert saves and saves[-1]["model"].gateway_model_cleanup_pending is True


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

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        return {"status": "ok", "removed": True}

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=unregister_worker,
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
        ("save",),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("remove_model", "model"),
        ("save",),
        ("kill", [20], 7.0),
        ("kill", [10], 7.0),
        ("save",),
    ]
    assert models == {}


def test_deregister_legacy_replica_fails_with_migration_hint_and_keeps_state(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
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

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        return {"status": "ok", "removed": True}

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=unregister_worker,
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

    with pytest.raises(click.ClickException, match="legacy.*router_worker_id"):
        deregister.do_deregister("model", 7.0, False, service="svc")

    assert not any(event[0] == "unregister" for event in events)
    assert not any(event[0] == "kill" for event in events)
    assert events == [("save",)]
    assert models["model"].lifecycle_state == "CLEANUP_PENDING"
    assert models["model"].gateway_model_cleanup_pending is True
    assert replica.router_cleanup_pending is True


def test_deregister_unregister_failure_marks_pending_and_keeps_processes_and_state(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="proxy-incarnation-1",
    )
    models = {"model": ModelEntry(backend="sglang:d1", replicas=[replica])}
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

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        raise deregister.ServiceUnreachable("router down")

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=unregister_worker,
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

    with pytest.raises(click.ClickException, match="cleanup is pending"):
        deregister.do_deregister("model", 7.0, False, service="svc")

    assert events == [
        ("save",),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("save",),
    ]
    assert "model" in models
    assert replica.router_cleanup_pending is True
    assert models["model"].lifecycle_state == "CLEANUP_PENDING"
    assert models["model"].gateway_model_cleanup_pending is True


def test_deregister_model_cleanup_failure_keeps_identity_and_processes(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="proxy-incarnation-1",
    )
    models = {"model": ModelEntry(backend="sglang:d1", replicas=[replica])}
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

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        return {"removed": True}

    def remove_model(name):
        events.append(("remove_model", name))
        raise deregister.ServiceUnreachable("router down")

    router = SimpleNamespace(
        unregister_worker=unregister_worker,
        remove_model=remove_model,
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

    with pytest.raises(click.ClickException, match="model cleanup is pending"):
        deregister.do_deregister("model", 7.0, False, service="svc")

    assert events == [
        ("save",),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("remove_model", "model"),
        ("save",),
    ]
    assert "model" in models
    assert replica.router_cleanup_pending is False
    assert models["model"].lifecycle_state == "CLEANUP_PENDING"
    assert models["model"].gateway_model_cleanup_pending is True


def test_deregister_retry_accepts_exact_retired_tombstone_after_response_loss(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="proxy-incarnation-1",
    )
    replica.router_cleanup_pending = True
    models = {"model": ModelEntry(backend="sglang:d1", replicas=[replica])}
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
        unregister_worker=lambda addr, worker_id: {"removed": False},
        get_worker_epoch=lambda addr: {
            "worker_addr": addr,
            "status": "retired",
            "worker_id": "proxy-incarnation-1",
        },
        remove_model=lambda name: events.append(("remove_model", name)),
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
    assert models == {}
    assert events == [
        ("save",),
        ("remove_model", "model"),
        ("save",),
        ("kill", [20], 7.0),
        ("kill", [10], 7.0),
        ("save",),
    ]


def test_deregister_registering_entry_accepts_unseen_epoch_as_never_committed(
    tmp_path, monkeypatch
):
    from areal.v2.cli.inference.commands import deregister

    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    events = []
    replica = _replica(
        worker_pid=10,
        proxy_pid=20,
        router_worker_id="never-committed-incarnation",
    )
    replica.router_cleanup_pending = True
    models = {
        "model": ModelEntry(
            backend="sglang:d1",
            replicas=[replica],
            lifecycle_state="REGISTERING",
        )
    }
    state = SimpleNamespace(
        service="svc",
        models=models,
        model_state=SimpleNamespace(
            models=models, save=lambda: events.append(("save",))
        ),
        router_url="http://router",
        admin_api_key="admin",
    )
    router = SimpleNamespace(
        unregister_worker=lambda addr, worker_id: {"removed": False},
        get_worker_epoch=lambda addr: {
            "worker_addr": addr,
            "status": "unseen",
            "worker_id": None,
        },
        remove_model=lambda name: (_ for _ in ()).throw(
            deregister.ServiceHTTPError(404, "not found")
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

    assert deregister.do_deregister("model", 7.0, False, service="svc") == 0
    assert models == {}
    assert ("kill", [20], 7.0) in events
    assert ("kill", [10], 7.0) in events


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

    def unregister_worker(addr, worker_id):
        events.append(("unregister", addr, worker_id))
        return {"status": "ok", "removed": True}

    router = SimpleNamespace(
        remove_model=lambda name: events.append(("remove_model", name)),
        unregister_worker=unregister_worker,
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
        ("save",),
        ("unregister", "http://127.0.0.1:5001", "proxy-incarnation-1"),
        ("remove_model", "model"),
        ("save",),
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
