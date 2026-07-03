# SPDX-License-Identifier: Apache-2.0

"""Router service — stateful routing, session pinning, worker registry.

The Router is a separate FastAPI service from the Gateway.
It owns worker health state, session→worker mappings, and routing strategy.
It never proxies traffic — it only answers routing queries.

Endpoint names are aligned with
``areal.v2.agent_service.router.app``:
``/register``, ``/unregister``, ``/route``, ``/remove_session``.
"""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from areal.utils import logging
from areal.v2.inference_service.router.config import RouterConfig
from areal.v2.inference_service.router.state import (
    GroupRegistry,
    ModelRegistry,
    SessionRegistry,
    WorkerRegistry,
)
from areal.v2.inference_service.router.strategies import get_strategy

logger = logging.getLogger("InferenceRouter")


# =============================================================================
# Auth helpers (same pattern as data proxy)
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header.",
    )


def _require_admin_key(request: Request, admin_key: str) -> str:
    """Validate that the request carries the admin API key."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


# =============================================================================
# Request models
# =============================================================================

WorkerId = Annotated[str, Field(min_length=1, max_length=128)]


class RegisterWorkerRequest(BaseModel):
    worker_addr: str
    worker_id: WorkerId
    expected_worker_id: WorkerId | None


class UnregisterWorkerRequest(BaseModel):
    worker_addr: str
    worker_id: WorkerId


class RouteRequest(BaseModel):
    api_key: str | None = None
    path: str | None = None
    session_id: str | None = None
    model: str | None = None
    new_session: bool = False


class SessionEntry(BaseModel):
    session_api_key: str
    session_id: str


class RegisterSessionRequest(BaseModel):
    sessions: list[SessionEntry]
    worker_addr: str
    worker_id: WorkerId
    group_id: str


class RemoveSessionRequest(BaseModel):
    group_id: str


class RegisterModelRequest(BaseModel):
    model: str
    url: str = ""
    api_key: str | None = None
    data_proxy_addrs: list[str] = []


class RemoveModelRequest(BaseModel):
    name: str


# =============================================================================
# Response models
# =============================================================================


class StatusResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    workers: int
    sessions: int
    strategy: str


class RegisterWorkerResponse(BaseModel):
    status: str
    worker_id: str
    action: str


class UnregisterWorkerResponse(BaseModel):
    status: str
    removed: bool
    sessions_revoked: int
    groups_revoked: int


class RouteResponse(BaseModel):
    worker_addr: str
    worker_id: str
    url: str | None = None
    api_key: str | None = None


class RemoveSessionResponse(BaseModel):
    status: str
    removed: bool


class WorkerInfo(BaseModel):
    worker_id: str
    addr: str
    healthy: bool
    active_requests: int


class WorkersResponse(BaseModel):
    workers: list[WorkerInfo]


class RegisterModelResponse(BaseModel):
    status: str
    model: str
    data_proxy_addrs: list[str]


class ModelsResponse(BaseModel):
    models: list[str]


class RemoveModelResponse(BaseModel):
    status: str
    name: str


class ResolveWorkerResponse(BaseModel):
    worker_id: str
    worker_addr: str


class WorkerEpochResponse(BaseModel):
    worker_addr: str
    status: str
    worker_id: str | None


# =============================================================================
# App factory
# =============================================================================


def create_app(config: RouterConfig) -> FastAPI:
    """Factory that creates the router FastAPI app."""

    worker_registry = WorkerRegistry()
    session_registry = SessionRegistry()
    model_registry = ModelRegistry()
    group_registry = GroupRegistry()
    ownership_lock = asyncio.Lock()
    strategy = get_strategy(config.routing_strategy)

    async def _poll_workers() -> None:
        """Background task: periodically poll worker /health endpoints."""
        while True:
            workers = await worker_registry.get_all_workers()

            async def _check(w):
                try:
                    resp = await app.state.http_client.get(f"{w.worker_addr}/health")
                    payload = resp.json() if resp.status_code == 200 else {}
                    healthy = (
                        isinstance(payload, dict)
                        and payload.get("worker_id") == w.worker_id
                    )
                    await worker_registry.update_health(
                        w.worker_addr, w.worker_id, healthy
                    )
                except Exception:
                    await worker_registry.update_health(
                        w.worker_addr, w.worker_id, False
                    )

            await asyncio.gather(*[_check(w) for w in workers])
            await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Router starting — strategy=%s, poll_interval=%.1fs",
            config.routing_strategy,
            config.poll_interval,
        )
        app.state.http_client = httpx.AsyncClient(timeout=config.worker_health_timeout)
        poll_task = asyncio.create_task(_poll_workers())
        app.state.worker_registry = worker_registry
        app.state.session_registry = session_registry
        app.state.model_registry = model_registry
        app.state.group_registry = group_registry
        app.state.strategy = strategy
        try:
            yield
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
            await app.state.http_client.aclose()
            logger.info("Router shutting down")

    app = FastAPI(title="AReaL Router", lifespan=lifespan)

    # Expose registries on app.state for tests that bypass lifespan
    app.state.worker_registry = worker_registry
    app.state.session_registry = session_registry
    app.state.model_registry = model_registry
    app.state.group_registry = group_registry
    app.state.strategy = strategy
    app.state.ownership_lock = ownership_lock

    async def _await_ownership_transition(task: asyncio.Task):
        """Let an in-memory ownership transition settle before propagating cancel."""

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except BaseException as exc:
                logger.error(
                    "Ownership transition failed while its request was cancelled: %s",
                    exc,
                )
            raise

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health():
        all_workers = await worker_registry.get_all_workers()
        session_count = await session_registry.count()
        return HealthResponse(
            status="ok",
            workers=len(all_workers),
            sessions=session_count,
            strategy=config.routing_strategy,
        )

    # =========================================================================
    # Worker management (admin key required)
    # =========================================================================

    @app.get("/worker_epoch", response_model=WorkerEpochResponse)
    async def worker_epoch(worker_addr: str, request: Request):
        """Return the active or last-retired epoch for one worker address."""

        _require_admin_key(request, config.admin_api_key)
        status, worker_id = await worker_registry.get_epoch(worker_addr)
        return WorkerEpochResponse(
            worker_addr=worker_addr,
            status=status,
            worker_id=worker_id,
        )

    @app.post("/register", response_model=RegisterWorkerResponse)
    async def register(body: RegisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        async def _transition() -> str:
            async with ownership_lock:
                action = await worker_registry.register(
                    body.worker_addr,
                    body.worker_id,
                    body.expected_worker_id,
                )
                if action == "replaced":
                    assert body.expected_worker_id is not None
                    await session_registry.revoke_by_worker(
                        body.worker_addr, body.expected_worker_id
                    )
                    await group_registry.revoke_by_worker(
                        body.worker_addr, body.expected_worker_id
                    )
                return action

        try:
            action = await _await_ownership_transition(
                asyncio.create_task(_transition())
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "Worker registration %s: %s (id=%s)",
            action,
            body.worker_addr,
            body.worker_id,
        )
        return RegisterWorkerResponse(
            status="ok", worker_id=body.worker_id, action=action
        )

    @app.post("/unregister", response_model=UnregisterWorkerResponse)
    async def unregister(body: UnregisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        async def _transition() -> tuple[bool, int, int]:
            async with ownership_lock:
                removed = await worker_registry.unregister(
                    body.worker_addr, body.worker_id
                )
                if removed:
                    sessions_revoked = await session_registry.revoke_by_worker(
                        body.worker_addr, body.worker_id
                    )
                    groups_revoked = await group_registry.revoke_by_worker(
                        body.worker_addr, body.worker_id
                    )
                else:
                    sessions_revoked = 0
                    groups_revoked = 0
                return removed, sessions_revoked, groups_revoked

        removed, sessions_revoked, groups_revoked = await _await_ownership_transition(
            asyncio.create_task(_transition())
        )
        logger.info(
            "Worker unregister: %s (id=%s, removed=%s, sessions=%d, groups=%d)",
            body.worker_addr,
            body.worker_id,
            removed,
            sessions_revoked,
            groups_revoked,
        )
        return UnregisterWorkerResponse(
            status="ok",
            removed=removed,
            sessions_revoked=sessions_revoked,
            groups_revoked=groups_revoked,
        )

    # =========================================================================
    # Routing (admin key required)
    # =========================================================================

    @app.post("/route", response_model=RouteResponse)
    async def route(body: RouteRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        # Step A: resolve model → candidate worker addrs
        model_addrs: list[str] | None = None
        if body.model is not None:
            info = await model_registry.get(body.model)
            if info is not None:
                model_addrs = info.data_proxy_addrs
        if model_addrs is None:
            first = await model_registry.first()
            if first is not None:
                model_addrs = first.data_proxy_addrs

        def _filter_by_model(workers: list, addrs: list[str] | None) -> list:
            if addrs is None:
                return workers
            addr_set = set(addrs)
            return [w for w in workers if w.worker_addr in addr_set]

        if body.new_session:
            if body.api_key is not None or body.session_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail="new_session cannot be combined with api_key or session_id",
                )
            all_workers = await worker_registry.get_all_workers()
            candidates = _filter_by_model(all_workers, model_addrs)
            if not candidates:
                raise HTTPException(status_code=503, detail="No registered workers")
            worker = strategy.pick(candidates)
            if worker is None:
                raise HTTPException(status_code=503, detail="No registered workers")
            return RouteResponse(
                worker_addr=worker.worker_addr,
                worker_id=worker.worker_id,
            )

        # Step B: session_id lookup
        if body.session_id is not None:
            async with ownership_lock:
                session_route = await session_registry.route_by_id(body.session_id)
                if session_route is not None:
                    owner = await worker_registry.get_by_addr(session_route.worker_addr)
                    if owner is None or owner.worker_id != session_route.worker_id:
                        raise HTTPException(
                            status_code=409,
                            detail="Session owner is no longer active",
                        )
                    return RouteResponse(
                        worker_addr=session_route.worker_addr,
                        worker_id=session_route.worker_id,
                    )
            if model_addrs is None:
                raise HTTPException(status_code=404, detail="Session not found")

        # Step C: model-only routing (no api_key/session_id)
        if body.api_key is None and model_addrs is not None:
            all_workers = await worker_registry.get_all_workers()
            candidates = _filter_by_model(all_workers, model_addrs)
            if not candidates:
                raise HTTPException(status_code=503, detail="No registered workers")
            worker = strategy.pick(candidates)
            if worker is None:
                raise HTTPException(status_code=503, detail="No registered workers")
            info = (
                await model_registry.get(body.model)
                if body.model
                else await model_registry.first()
            )
            return RouteResponse(
                worker_addr=worker.worker_addr,
                worker_id=worker.worker_id,
                url=info.url if info else None,
                api_key=info.api_key if info else None,
            )

        if body.api_key is None and body.model is not None and model_addrs is None:
            raise HTTPException(
                status_code=404, detail=f"Model '{body.model}' not found"
            )

        if body.api_key is None:
            raise HTTPException(
                status_code=422,
                detail="Either 'api_key' or 'session_id' must be provided",
            )

        # Step C: Session key → pinned worker
        async with ownership_lock:
            pinned = await session_registry.route_by_key(body.api_key)
            if pinned is not None:
                owner = await worker_registry.get_by_addr(pinned.worker_addr)
                if owner is None or owner.worker_id != pinned.worker_id:
                    raise HTTPException(
                        status_code=409,
                        detail="Session owner is no longer active",
                    )
                return RouteResponse(
                    worker_addr=pinned.worker_addr,
                    worker_id=pinned.worker_id,
                )

        # Step D: Admin key → pick from model addrs
        if hmac.compare_digest(body.api_key, config.admin_api_key):
            async with ownership_lock:
                # Another first-use request may have pinned the key after the
                # optimistic lookup above.
                pinned = await session_registry.route_by_key(body.api_key)
                if pinned is not None:
                    owner = await worker_registry.get_by_addr(pinned.worker_addr)
                    if owner is None or owner.worker_id != pinned.worker_id:
                        raise HTTPException(
                            status_code=409,
                            detail="Session owner is no longer active",
                        )
                    return RouteResponse(
                        worker_addr=pinned.worker_addr,
                        worker_id=pinned.worker_id,
                    )
                all_workers = await worker_registry.get_all_workers()
                candidates = _filter_by_model(all_workers, model_addrs)
                if not candidates:
                    raise HTTPException(status_code=503, detail="No registered workers")
                worker = strategy.pick(candidates)
                if worker is None:
                    raise HTTPException(status_code=503, detail="No registered workers")
                await session_registry.register_session(
                    body.api_key,
                    "__hitl__",
                    worker.worker_addr,
                    worker.worker_id,
                )
                return RouteResponse(
                    worker_addr=worker.worker_addr,
                    worker_id=worker.worker_id,
                )

        # Step E: Unknown key
        raise HTTPException(status_code=404, detail="Unknown API key")

    # =========================================================================
    # Session registration (admin key required)
    # =========================================================================

    @app.post("/register_session", response_model=StatusResponse)
    async def register_session(body: RegisterSessionRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        session_ids = [e.session_id for e in body.sessions]
        async with ownership_lock:
            worker = await worker_registry.get_by_addr(body.worker_addr)
            if worker is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Worker {body.worker_addr} is not registered",
                )
            if body.worker_id != worker.worker_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Worker epoch mismatch for {body.worker_addr}: "
                        f"expected {worker.worker_id}, got {body.worker_id}"
                    ),
                )
            group_created = False
            try:
                # Register the immutable group key first. A conflicting HTTP
                # replay is rejected before any session mapping can be mutated.
                group_created = await group_registry.register_group(
                    body.group_id,
                    body.worker_addr,
                    session_ids,
                    body.worker_id,
                    [entry.session_api_key for entry in body.sessions],
                )
                await session_registry.register_sessions(
                    [
                        (entry.session_api_key, entry.session_id)
                        for entry in body.sessions
                    ],
                    body.worker_addr,
                    worker_id=body.worker_id,
                )
            except ValueError as exc:
                if group_created:
                    await group_registry.revoke(body.group_id)
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        return StatusResponse(status="ok")

    # =========================================================================
    # Session cleanup (admin key required)
    # =========================================================================

    @app.post("/remove_session", response_model=RemoveSessionResponse)
    async def remove_session(body: RemoveSessionRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        async with ownership_lock:
            session_ids = await group_registry.revoke(body.group_id)
            for sid in session_ids:
                await session_registry.revoke_session(sid)
        return RemoveSessionResponse(
            status="ok",
            removed=len(session_ids) > 0,
        )

    # =========================================================================
    # Worker listing (admin key required)
    # =========================================================================

    @app.get("/workers", response_model=WorkersResponse)
    async def list_workers(request: Request):
        _require_admin_key(request, config.admin_api_key)
        all_workers = await worker_registry.get_all_workers()
        return WorkersResponse(
            workers=[
                WorkerInfo(
                    worker_id=w.worker_id,
                    addr=w.worker_addr,
                    healthy=w.is_healthy,
                    active_requests=w.active_requests,
                )
                for w in all_workers
            ]
        )

    @app.post("/register_model", response_model=RegisterModelResponse)
    async def register_model(body: RegisterModelRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        addrs = body.data_proxy_addrs
        if not addrs:
            healthy = await worker_registry.get_healthy_workers()
            if not healthy:
                raise HTTPException(status_code=503, detail="No healthy workers")
            addrs = [w.worker_addr for w in healthy]
        await model_registry.register(
            body.model,
            body.url,
            body.api_key,
            addrs,
        )
        logger.info(
            "Model registered: model=%s url=%s data_proxy_addrs=%s",
            body.model,
            body.url or "(internal)",
            addrs,
        )
        return RegisterModelResponse(
            status="ok",
            model=body.model,
            data_proxy_addrs=addrs,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def list_models(request: Request):
        _require_admin_key(request, config.admin_api_key)
        names = await model_registry.list_names()
        return ModelsResponse(models=names)

    @app.post("/remove_model", response_model=RemoveModelResponse)
    async def remove_model(body: RemoveModelRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        removed = await model_registry.remove(body.name)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"External model '{body.name}' not found",
            )
        logger.info("External model removed: name=%s", body.name)
        return RemoveModelResponse(status="ok", name=body.name)

    # =========================================================================
    # Worker resolution by ID (admin key required)
    # =========================================================================

    @app.get("/resolve_worker/{worker_id}", response_model=ResolveWorkerResponse)
    async def resolve_worker(worker_id: str, request: Request):
        """Resolve a worker_id to its address."""
        _require_admin_key(request, config.admin_api_key)
        worker = await worker_registry.get_by_id(worker_id)
        if worker is None:
            raise HTTPException(
                status_code=404, detail=f"Worker ID {worker_id} not found"
            )
        return ResolveWorkerResponse(
            worker_id=worker.worker_id, worker_addr=worker.worker_addr
        )

    return app
