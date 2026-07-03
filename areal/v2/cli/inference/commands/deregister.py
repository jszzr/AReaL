# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.v2.cli.inference.client import RouterClient
from areal.v2.cli.inference.common import logger
from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.inference.state import store
from areal.v2.cli.process import kill_pids


@click.command(name="deregister", help="Deregister a model and tear down its workers.")
@click.option("--model-name", required=True, help="Model name to deregister.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL workers immediately.")
def deregister_cmd(
    model_name: str, service: str | None, grace: float, force: bool
) -> None:
    raise SystemExit(do_deregister(model_name, grace, force, service=service) or 0)


def do_deregister(
    model_name: str, grace: float, force: bool, *, service: str | None = None
) -> int:
    service_name = inf_lifecycle.resolve_service_name(service)
    with store.lock_model_state(service_name):
        state = inf_lifecycle.load_running_state(service_name)
        if model_name not in state.models:
            raise click.ClickException(
                f"model {model_name!r} is not registered in service {state.service!r}"
            )
        entry = state.models[model_name]
        router = RouterClient(state.router_url, state.admin_api_key)

        # Router unregister → kill data-proxies → kill workers (same data-flow
        # order as terminate_runtime_state).
        legacy_replicas = [r for r in entry.replicas if r.router_worker_id is None]
        if legacy_replicas:
            for replica in legacy_replicas:
                replica.router_cleanup_pending = True
            state.model_state.save()
            legacy_addrs = ", ".join(r.data_proxy.addr for r in legacy_replicas)
            raise click.ClickException(
                "legacy model state has no router_worker_id for data-proxy "
                f"{legacy_addrs}; refusing unsafe per-model deregistration. "
                "State was retained. Stop and restart the whole service with "
                f"`areal inf stop --service {state.service}` to migrate."
            )

        cleanup_errors: list[str] = []
        for r in entry.replicas:
            assert r.router_worker_id is not None
            try:
                response = router.unregister_worker(
                    r.data_proxy.addr, r.router_worker_id
                )
            except (ServiceHTTPError, ServiceUnreachable) as exc:
                r.router_cleanup_pending = True
                cleanup_errors.append(
                    f"{r.data_proxy.addr} ({r.router_worker_id}): {exc}"
                )
                continue

            if response.get("removed") is True:
                r.router_cleanup_pending = False
                continue

            # A previous exact unregister may have committed even if its
            # response was lost. Accept only a Router tombstone for this exact
            # identity; never infer success from a different active epoch.
            try:
                epoch = router.get_worker_epoch(r.data_proxy.addr)
            except (ServiceHTTPError, ServiceUnreachable) as exc:
                r.router_cleanup_pending = True
                cleanup_errors.append(
                    f"{r.data_proxy.addr} ({r.router_worker_id}): "
                    f"could not verify cleanup after removed=false: {exc}"
                )
                continue
            if (
                epoch.get("status") == "retired"
                and epoch.get("worker_id") == r.router_worker_id
            ):
                r.router_cleanup_pending = False
                continue

            r.router_cleanup_pending = True
            cleanup_errors.append(
                f"{r.data_proxy.addr} ({r.router_worker_id}): exact owner was not "
                f"removed; Router epoch is {epoch!r}"
            )

        if cleanup_errors:
            state.model_state.save()
            details = "; ".join(cleanup_errors)
            raise click.ClickException(
                "router cleanup is pending; local processes and model state were "
                f"retained for a safe retry: {details}"
            )

        model_cleanup_error: Exception | None = None
        try:
            router.remove_model(model_name)
        except ServiceHTTPError as exc:
            if exc.status != 404:
                model_cleanup_error = exc
        except ServiceUnreachable as exc:
            model_cleanup_error = exc

        if model_cleanup_error is not None:
            for r in entry.replicas:
                r.router_cleanup_pending = True
            state.model_state.save()
            raise click.ClickException(
                "router model cleanup is pending; local processes and model state "
                f"were retained for a safe retry: {model_cleanup_error}"
            )

        proxy_pids = [r.data_proxy.pid for r in entry.replicas if r.data_proxy.pid > 0]
        worker_pids = [r.worker.pid for r in entry.replicas if r.worker.pid > 0]
        effective_grace = 0.0 if force else grace
        for pids in (proxy_pids, worker_pids):
            if pids:
                kill_pids(pids, grace_s=effective_grace)

        del state.model_state.models[model_name]
        state.model_state.save()
        logger.info("deregistered model %r from service %r", model_name, state.service)
    return 0
