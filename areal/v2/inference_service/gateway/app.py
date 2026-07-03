# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway — thin HTTP proxy with auth, routing, and forwarding.

The gateway holds only ``admin_api_key`` and ``router_addr``. All worker state,
session pinning, and routing strategies live in the Router service.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.v2.inference_service.gateway.admission import (
    OnlineLease,
    OnlineLeaseBinding,
    OnlineLeaseCapacityError,
    OnlineLeaseRegistry,
    ReplayableHTTPResult,
    RequestReplayRegistry,
    RequestWorkerBinding,
    RequestWorkerOwnership,
    RequestWorkerOwnershipCapacityError,
    RequestWorkerOwnershipRegistry,
    RequestWorkerOwnershipState,
)
from areal.v2.inference_service.gateway.auth import (
    extract_bearer_token,
    require_admin_key,
)
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.gateway.streaming import (
    RouterDestination,
    RouterKeyRejectedError,
    RouterSessionRegistrationError,
    RouterUnreachableError,
    _forwarding_headers,
    forward_request,
    forward_sse_stream,
    list_models_from_router,
    notify_online_lease_failure,
    query_router,
    register_model_in_router,
    register_session_in_router,
    remove_model_from_router,
    resolve_worker_addr,
    revoke_session_in_router,
)
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER

logger = logging.getLogger("InferenceGateway")


# =============================================================================
# Response models
# =============================================================================


class GatewayHealthResponse(BaseModel):
    status: str
    router_addr: str
    available_online_leases: int


class GrantOnlineLeaseRequest(BaseModel):
    lease_id: str
    expected_version: int
    callback_url: str = ""
    ttl_seconds: float = 300.0


class GatewayModelsResponse(BaseModel):
    models: list[str]


class BroadcastResultItem(BaseModel):
    worker_addr: str
    status: int
    ok: bool
    error: str | None = None


class BroadcastResponse(BaseModel):
    results: list[BroadcastResultItem]


def _router_error_response(exc: Exception) -> JSONResponse:
    """Convert router exceptions to HTTP responses."""
    if isinstance(exc, RouterUnreachableError):
        return JSONResponse({"error": str(exc)}, status_code=502)
    if isinstance(exc, RouterKeyRejectedError):
        status = 401 if exc.status_code == 404 else exc.status_code
        return JSONResponse({"error": exc.detail}, status_code=status)
    return JSONResponse({"error": str(exc)}, status_code=500)


def create_app(config: GatewayConfig) -> FastAPI:
    """Factory that creates the inference gateway FastAPI app."""

    if config.max_pending_export_cleanups < 1:
        raise ValueError("max_pending_export_cleanups must be >= 1")
    if config.max_pending_request_owners < 1:
        raise ValueError("max_pending_request_owners must be >= 1")
    online_lease_registry = OnlineLeaseRegistry()
    start_request_registry = RequestReplayRegistry()
    export_request_registry = RequestReplayRegistry()
    pending_export_group_cleanups: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    reserved_export_group_cleanups: set[str] = set()
    export_group_cleanup_lock = asyncio.Lock()
    start_request_workers = RequestWorkerOwnershipRegistry(
        max_owned_records=config.max_pending_request_owners
    )
    export_request_workers = RequestWorkerOwnershipRegistry(
        max_owned_records=config.max_pending_request_owners
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = create_httpx_client(timeout=config.router_timeout)
        lease_reaper = asyncio.create_task(_online_lease_reaper())
        try:
            yield
        finally:
            lease_reaper.cancel()
            try:
                await lease_reaper
            except asyncio.CancelledError:
                pass
            await app.state.http_client.aclose()
            if _fallback_client is not None:
                await _fallback_client.aclose()

    app = FastAPI(title="AReaL Inference Gateway", lifespan=lifespan)
    app.state.online_lease_registry = online_lease_registry
    app.state.start_request_registry = start_request_registry
    app.state.export_request_registry = export_request_registry
    app.state.pending_export_group_cleanups = pending_export_group_cleanups
    app.state.start_request_workers = start_request_workers
    app.state.export_request_workers = export_request_workers

    # Fallback client for tests that bypass the lifespan.
    # NOTE: When testing without ASGI transport (which runs the lifespan),
    # this client will NOT be cleaned up automatically. Tests should prefer
    # using httpx.AsyncClient(transport=ASGITransport(app=app)) to ensure
    # proper lifespan management and client cleanup.
    _fallback_client: httpx.AsyncClient | None = None

    def _client() -> httpx.AsyncClient:
        nonlocal _fallback_client
        try:
            return app.state.http_client
        except AttributeError:
            if _fallback_client is None:
                _fallback_client = create_httpx_client(timeout=config.router_timeout)
            return _fallback_client

    async def _await_critical_task(task: asyncio.Task, operation: str):
        """Finish a state transition even if its request is cancelled repeatedly."""

        request_cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                request_cancelled = True
                continue
            except BaseException:
                break

        if request_cancelled:
            try:
                task.result()
            except BaseException as exc:
                logger.error(
                    "%s failed while its request was cancelled: %s",
                    operation,
                    exc,
                )
            raise asyncio.CancelledError
        return task.result()

    def _targeted_control_headers(request: Request, worker_id: str) -> dict[str, str]:
        """Forward auth/content headers while pinning the resolved incarnation."""

        headers = _forwarding_headers(dict(request.headers))
        # The path identity is authoritative.  A caller-supplied header must not
        # be able to retarget the command after Router resolution.
        headers[WORKER_ID_HEADER] = worker_id
        return headers

    async def _forward_targeted_control(
        worker_addr: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> BroadcastResponse | Response:
        """Forward one control command without ever selecting another worker."""

        try:
            resp = await forward_request(
                f"{worker_addr}{path}",
                body,
                headers,
                config.forward_timeout,
                client=_client(),
            )
        except Exception as exc:
            return BroadcastResponse(
                results=[
                    BroadcastResultItem(
                        worker_addr=worker_addr,
                        status=502,
                        ok=False,
                        error=str(exc),
                    )
                ]
            )

        # A 409 is the Data Proxy's incarnation fence.  Preserve it verbatim so
        # callers cannot mistake a stale command for a successful broadcast.
        if resp.status_code == 409:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

        return BroadcastResponse(
            results=[
                BroadcastResultItem(
                    worker_addr=worker_addr,
                    status=resp.status_code,
                    ok=resp.status_code < 400,
                )
            ]
        )

    async def _remember_start_worker(
        request_id: str,
        fingerprint: str,
        worker_addr: str,
        worker_id: str,
    ) -> RequestWorkerBinding:
        return await start_request_workers.remember(
            request_id,
            RequestWorkerBinding(
                fingerprint=fingerprint,
                worker_addr=worker_addr,
                worker_id=worker_id,
            ),
        )

    async def _recalled_start_worker(
        request_id: str, fingerprint: str
    ) -> RequestWorkerOwnership | None:
        return await start_request_workers.recall(request_id, fingerprint)

    async def _forget_start_worker(
        request_id: str, binding: RequestWorkerBinding
    ) -> bool:
        return await start_request_workers.forget(request_id, binding)

    async def _remember_export_worker(
        request_id: str,
        fingerprint: str,
        worker_addr: str,
        worker_id: str,
    ) -> RequestWorkerBinding:
        return await export_request_workers.remember(
            request_id,
            RequestWorkerBinding(
                fingerprint=fingerprint,
                worker_addr=worker_addr,
                worker_id=worker_id,
            ),
        )

    async def _recalled_export_worker(
        request_id: str, fingerprint: str
    ) -> RequestWorkerOwnership | None:
        return await export_request_workers.recall(request_id, fingerprint)

    async def _forget_export_worker(
        request_id: str, binding: RequestWorkerBinding
    ) -> bool:
        return await export_request_workers.forget(request_id, binding)

    async def _cleanup_online_binding(binding: OnlineLeaseBinding) -> bool:
        """Best-effort compensation for a created callback session."""

        worker_cleaned = False
        try:
            response = await forward_request(
                f"{binding.worker_addr}/rl/cancel_sessions",
                json.dumps(
                    {
                        "admission_id": binding.admission_id,
                        "session_ids": list(binding.session_ids),
                    }
                ).encode(),
                {
                    "Authorization": f"Bearer {config.admin_api_key}",
                    "Content-Type": "application/json",
                    WORKER_ID_HEADER: binding.worker_id,
                },
                min(config.forward_timeout, 10.0),
                client=_client(),
            )
            if response.status_code == 409:
                # The old incarnation is already gone.  Its in-memory session
                # state cannot be cleaned any further, but the Router's exact
                # ownership record still must be compared and removed below.
                worker_cleaned = True
            elif response.status_code < 200 or response.status_code >= 300:
                logger.warning(
                    "Failed to cancel sessions for group %s on %s: HTTP %d %s",
                    binding.group_id,
                    binding.worker_addr,
                    response.status_code,
                    response.text,
                )
            else:
                worker_cleaned = True
        except Exception as exc:
            logger.warning(
                "Failed to cancel sessions for group %s on %s: %s",
                binding.group_id,
                binding.worker_addr,
                exc,
            )
        router_cleaned = True
        if binding.group_id:
            router_cleaned = await revoke_session_in_router(
                config.router_addr,
                config.admin_api_key,
                binding.group_id,
                binding.worker_addr,
                binding.worker_id,
                binding.session_ids,
                timeout=config.router_timeout,
                client=_client(),
            )
        return worker_cleaned and router_cleaned

    async def _cleanup_and_ack_online_binding(
        lease_id: str, binding: OnlineLeaseBinding
    ) -> bool:
        cleaned = await _cleanup_online_binding(binding)
        if cleaned:
            await online_lease_registry.acknowledge_cleanup(lease_id, binding)
        return cleaned

    async def _cleanup_and_ack_start_binding(
        request_id: str,
        owner_binding: RequestWorkerBinding,
        cleanup_binding: OnlineLeaseBinding,
    ) -> bool:
        cleaned = await _cleanup_online_binding(cleanup_binding)
        if cleaned:
            await start_request_workers.acknowledge_cleanup(
                request_id, owner_binding, cleanup_binding
            )
        return cleaned

    async def _fail_online_lease(lease: OnlineLease, reason: str) -> None:
        if not await online_lease_registry.fail(lease.lease_id, reason):
            return
        try:
            await notify_online_lease_failure(
                lease.callback_url,
                config.admin_api_key,
                lease.lease_id,
                lease.expected_version,
                reason,
                config.router_timeout,
                client=_client(),
            )
        except Exception as exc:
            logger.error(
                "Failed to notify controller that online lease %s failed: %s",
                lease.lease_id,
                exc,
            )

    async def _ensure_export_group_cleanup(group_id: str) -> bool:
        async with export_group_cleanup_lock:
            binding = pending_export_group_cleanups.get(group_id)
        if binding is None:
            return True
        worker_addr, worker_id, session_ids = binding
        revoked = await revoke_session_in_router(
            config.router_addr,
            config.admin_api_key,
            group_id,
            worker_addr,
            worker_id,
            session_ids,
            timeout=config.router_timeout,
            client=_client(),
        )
        if revoked:
            async with export_group_cleanup_lock:
                if pending_export_group_cleanups.get(group_id) == binding:
                    pending_export_group_cleanups.pop(group_id, None)
        return revoked

    async def _handoff_export_group_cleanup(
        group_id: str,
        worker_addr: str,
        worker_id: str,
        session_ids: tuple[str, ...],
    ) -> None:
        """Make Router cleanup reaper-owned before any later await can cancel."""

        async with export_group_cleanup_lock:
            binding = (worker_addr, worker_id, session_ids)
            existing = pending_export_group_cleanups.get(group_id)
            if existing is not None and existing != binding:
                raise RuntimeError(
                    f"Export group {group_id} has conflicting cleanup ownership"
                )
            reserved_export_group_cleanups.discard(group_id)
            pending_export_group_cleanups[group_id] = binding

    async def _reserve_export_group_cleanup(group_id: str) -> tuple[bool, bool]:
        """Reserve bounded ownership for cleanup after a destructive export."""

        async with export_group_cleanup_lock:
            if group_id in pending_export_group_cleanups:
                return True, False
            if group_id in reserved_export_group_cleanups:
                return True, False
            owned = len(pending_export_group_cleanups) + len(
                reserved_export_group_cleanups
            )
            if owned >= config.max_pending_export_cleanups:
                return False, False
            reserved_export_group_cleanups.add(group_id)
            return True, True

    async def _release_export_group_cleanup_reservation(group_id: str | None) -> None:
        if group_id is None:
            return
        async with export_group_cleanup_lock:
            reserved_export_group_cleanups.discard(group_id)

    async def _retry_export_group_cleanup_if_pending(group_id: str) -> None:
        async with export_group_cleanup_lock:
            is_pending = group_id in pending_export_group_cleanups
        if is_pending:
            await _ensure_export_group_cleanup(group_id)

    async def _reap_expired_online_leases() -> None:
        expired_leases = await online_lease_registry.expire_stale()

        async def _notify_expired(expired) -> None:
            try:
                await notify_online_lease_failure(
                    expired.lease.callback_url,
                    config.admin_api_key,
                    expired.lease.lease_id,
                    expired.lease.expected_version,
                    expired.reason,
                    config.router_timeout,
                    client=_client(),
                )
            except BaseException as exc:
                logger.warning(
                    "Failed to notify controller that expired online lease %s: %s",
                    expired.lease.lease_id,
                    exc,
                )

        await asyncio.gather(
            *[_notify_expired(expired) for expired in expired_leases],
            return_exceptions=True,
        )
        pending = await online_lease_registry.pending_cleanup_bindings()
        await asyncio.gather(
            *[
                _cleanup_and_ack_online_binding(lease_id, binding)
                for lease_id, binding in pending
            ],
            return_exceptions=True,
        )
        pending_starts = await start_request_workers.pending_cleanups()
        await asyncio.gather(
            *[
                _cleanup_and_ack_start_binding(
                    request_id, owner_binding, cleanup_binding
                )
                for request_id, owner_binding, cleanup_binding in pending_starts
            ],
            return_exceptions=True,
        )

    async def _online_lease_reaper() -> None:
        while True:
            await asyncio.sleep(1.0)
            await _reap_expired_online_leases()
            async with export_group_cleanup_lock:
                pending_groups = list(pending_export_group_cleanups)
            await asyncio.gather(
                *[
                    _ensure_export_group_cleanup(group_id)
                    for group_id in pending_groups
                ],
                return_exceptions=True,
            )

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=GatewayHealthResponse)
    async def health():
        return GatewayHealthResponse(
            status="ok",
            router_addr=config.router_addr,
            available_online_leases=await online_lease_registry.available_count(),
        )

    @app.post("/internal/online_leases", status_code=201)
    async def grant_online_lease(body: GrantOnlineLeaseRequest, request: Request):
        require_admin_key(request, config.admin_api_key)
        if body.ttl_seconds <= 0:
            return JSONResponse({"error": "ttl_seconds must be > 0"}, status_code=422)
        try:
            lease = await online_lease_registry.grant(
                OnlineLease(
                    lease_id=body.lease_id,
                    expected_version=body.expected_version,
                    callback_url=body.callback_url,
                    ttl_seconds=body.ttl_seconds,
                )
            )
        except OnlineLeaseCapacityError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return {
            "status": "ok",
            "lease_id": lease.lease_id,
            "expected_version": lease.expected_version,
        }

    @app.delete("/internal/online_leases/{lease_id}")
    async def cancel_online_lease(lease_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        cancelled, binding = await online_lease_registry.cancel_and_take_binding(
            lease_id
        )
        if binding is not None:
            cleaned = await _cleanup_and_ack_online_binding(lease_id, binding)
            if not cleaned:
                return JSONResponse(
                    {
                        "error": "Lease cancelled but session cleanup failed; retry DELETE",
                        "cancelled": cancelled,
                    },
                    status_code=502,
                )
        return {"status": "ok", "cancelled": cancelled}

    # =========================================================================
    # POST /chat/completions — admin OR session key, streaming or non-streaming
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model_name = None
        is_streaming = False
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
            is_streaming = body_json.get("stream", False) or False
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            route_result = await query_router(
                config.router_addr,
                token,
                "/chat/completions",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model_name,
                return_destination=True,
                client=_client(),
            )
            if not isinstance(route_result, RouterDestination):
                raise RouterUnreachableError(
                    "Router chat route did not return a worker incarnation"
                )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)
        worker_addr = route_result.worker_addr
        headers[WORKER_ID_HEADER] = route_result.worker_id

        if is_streaming:
            return StreamingResponse(
                forward_sse_stream(
                    f"{worker_addr}/chat/completions",
                    body,
                    headers,
                    config.forward_timeout,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        resp = await forward_request(
            f"{worker_addr}/chat/completions",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @app.post("/register_model")
    async def register_model(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.json()
        model = body.get("model")
        url = body.get("url", "")
        api_key = body.get("api_key")
        data_proxy_addrs = body.get("data_proxy_addrs", [])
        if not model:
            return JSONResponse({"error": "model is required"}, status_code=400)
        try:
            result = await register_model_in_router(
                config.router_addr,
                model,
                url,
                api_key,
                data_proxy_addrs,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resolved_addrs = result.get("data_proxy_addrs", data_proxy_addrs)
        headers = _forwarding_headers(dict(request.headers))

        # Phase 3: parallelize data proxy registration
        async def _register_one(addr: str) -> httpx.Response:
            return await forward_request(
                f"{addr}/register_model",
                json.dumps(
                    {
                        "name": model,
                        "url": url,
                        "model": model,
                        "api_key": api_key,
                    }
                ).encode(),
                headers,
                config.forward_timeout,
                client=_client(),
            )

        responses = await asyncio.gather(
            *[_register_one(addr) for addr in resolved_addrs],
            return_exceptions=True,
        )
        # Check for failures after all tasks have completed
        failed = False
        errors = []
        for resp in responses:
            if isinstance(resp, Exception):
                logger.error("Data proxy registration raised: %s", resp)
                errors.append(str(resp))
                failed = True
            elif resp.status_code != 200:
                logger.error(
                    "Data proxy registration failed with status %d: %s",
                    resp.status_code,
                    resp.text,
                )
                errors.append(f"status {resp.status_code}: {resp.text}")
                failed = True
        if failed:
            await remove_model_from_router(
                config.router_addr,
                model,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
            return JSONResponse(
                {"error": "Data proxy registration failed", "details": errors},
                status_code=502,
            )
        return result

    @app.get("/models")
    async def list_models(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            names = await list_models_from_router(
                config.router_addr,
                config.admin_api_key,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)
        return GatewayModelsResponse(models=names)

    # =========================================================================
    # POST /rl/start_session — admin key ONLY, intercept response
    # =========================================================================

    @app.post("/rl/start_session")
    async def start_session(request: Request):
        require_admin_key(request, config.admin_api_key)

        body = await request.body()
        try:
            body_json = json.loads(body)
            if not isinstance(body_json, dict):
                raise ValueError("request body must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return JSONResponse({"error": f"Invalid JSON body: {exc}"}, status_code=400)

        delivery_mode = body_json.get("delivery_mode", "callback")
        if delivery_mode not in {"callback", "pull"}:
            return JSONResponse(
                {"error": f"Unsupported delivery_mode: {delivery_mode}"},
                status_code=422,
            )

        request_id = body_json.pop("request_id", None)
        legacy_admission_id = body_json.pop("admission_id", None)
        if request_id is None:
            request_id = legacy_admission_id
        elif legacy_admission_id is not None and legacy_admission_id != request_id:
            return JSONResponse(
                {"error": "request_id and admission_id must match when both are set"},
                status_code=422,
            )
        if request_id is not None and (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > 256
        ):
            return JSONResponse(
                {"error": "request_id must be a non-empty string up to 256 characters"},
                status_code=422,
            )

        lease: OnlineLease | None = None
        if delivery_mode == "callback":
            if request_id is None:
                return JSONResponse(
                    {
                        "error": "Callback delivery requires a caller-generated request_id"
                    },
                    status_code=422,
                )
            try:
                group_size = max(int(body_json.get("group_size", 1)), 1)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "group_size must be an integer"}, status_code=422
                )
            if group_size != 1:
                return JSONResponse(
                    {
                        "error": "Callback delivery currently requires group_size=1; "
                        "use pull delivery for grouped sessions"
                    },
                    status_code=422,
                )

        # Pull callers may omit the key for backwards compatibility. The
        # gateway-generated value still makes Gateway→DataProxy retries safe;
        # callers that retry the outer request should provide their own ID.
        resolved_request_id = request_id or f"gateway-{uuid.uuid4()}"
        fingerprint = json.dumps(
            body_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        try:
            reservation = await start_request_registry.reserve(
                resolved_request_id, fingerprint
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        if not reservation.is_owner:
            try:
                cached = await start_request_registry.wait(
                    reservation, timeout=config.forward_timeout
                )
            except TimeoutError:
                return JSONResponse(
                    {"error": f"Start request {resolved_request_id} is still pending"},
                    status_code=504,
                )
            return Response(
                content=cached.content,
                status_code=cached.status_code,
                media_type=cached.media_type,
            )

        async def _finish(result: ReplayableHTTPResult) -> Response:
            await start_request_registry.finish(resolved_request_id, result)
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )

        async def _finish_json(payload: dict, status_code: int) -> Response:
            return await _finish(
                ReplayableHTTPResult(
                    status_code=status_code,
                    content=json.dumps(payload).encode(),
                    media_type="application/json",
                )
            )

        async def _release_json(payload: dict, status_code: int) -> Response:
            result = ReplayableHTTPResult(
                status_code=status_code,
                content=json.dumps(payload).encode(),
                media_type="application/json",
            )
            return await _release(result)

        async def _release(result: ReplayableHTTPResult) -> Response:
            await start_request_registry.release_pending(
                resolved_request_id, fingerprint, result
            )
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )

        async def _settle_callback_failure(
            owned_lease: OnlineLease,
            *,
            reason: str,
            result: ReplayableHTTPResult,
            binding: OnlineLeaseBinding | None = None,
        ) -> Response:
            """Fail one lease, then finalize ownership despite cancellation."""

            notification_cancelled = False
            try:
                await _fail_online_lease(owned_lease, reason)
            except asyncio.CancelledError:
                notification_cancelled = True

            async def _critical_finalize() -> Response:
                if binding is not None:
                    # Persist compensation ownership inside the shielded task.
                    # Callers may themselves be handling a cancellation and
                    # can receive another one at any preparatory await.
                    await online_lease_registry.retain_cleanup_binding(
                        owned_lease.lease_id, binding
                    )
                    await _cleanup_and_ack_online_binding(owned_lease.lease_id, binding)
                await online_lease_registry.finish_start(owned_lease.lease_id)
                if result.status_code >= 500:
                    return await _release(result)
                return await _finish(result)

            finalize_task = asyncio.create_task(_critical_finalize())
            response = await _await_critical_task(
                finalize_task, "Callback failure settlement"
            )
            if notification_cancelled:
                raise asyncio.CancelledError
            return response

        downstream_admission_id = resolved_request_id
        if delivery_mode == "callback":
            lease = await online_lease_registry.try_acquire()
            if lease is None:
                no_capacity = ReplayableHTTPResult(
                    status_code=429,
                    content=json.dumps(
                        {"error": "No admitted online rollout is waiting"}
                    ).encode(),
                    media_type="application/json",
                )
                await start_request_registry.release_pending(
                    resolved_request_id, fingerprint, no_capacity
                )
                return Response(
                    content=no_capacity.content,
                    status_code=no_capacity.status_code,
                    media_type=no_capacity.media_type,
                )
            downstream_admission_id = lease.lease_id
            body_json.update(
                {
                    "lease_id": lease.lease_id,
                    "admission_id": downstream_admission_id,
                    "expected_version": lease.expected_version,
                }
            )
        else:
            body_json["admission_id"] = downstream_admission_id
        body = json.dumps(body_json).encode()

        refresh_api_key = body_json.get("api_key")
        start_worker_binding: RequestWorkerBinding | None = None
        try:
            destination = None
            if lease is None:
                recalled = await _recalled_start_worker(
                    resolved_request_id, fingerprint
                )
                if recalled is not None:
                    if recalled.state is RequestWorkerOwnershipState.CLEANUP_PENDING:
                        return await _release_json(
                            {
                                "error": (
                                    "A previous start attempt is still awaiting "
                                    "compensating cleanup"
                                )
                            },
                            503,
                        )
                    start_worker_binding = recalled.binding
                    destination = RouterDestination(
                        worker_addr=recalled.binding.worker_addr,
                        worker_id=recalled.binding.worker_id,
                    )
            if destination is None:
                route_result = await query_router(
                    config.router_addr,
                    refresh_api_key,
                    "/rl/start_session",
                    config.router_timeout,
                    admin_api_key=config.admin_api_key,
                    new_session=refresh_api_key is None,
                    return_destination=True,
                    client=_client(),
                )
                if not isinstance(route_result, RouterDestination):
                    raise RouterUnreachableError(
                        "Router start route did not return a worker incarnation"
                    )
                destination = route_result
            worker_addr = destination.worker_addr
            worker_id = destination.worker_id
            if lease is None and start_worker_binding is None:
                start_worker_binding = await _remember_start_worker(
                    resolved_request_id,
                    fingerprint,
                    worker_addr,
                    worker_id,
                )
        except RequestWorkerOwnershipCapacityError as exc:
            return await _release_json({"error": str(exc)}, 503)
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            response = _router_error_response(exc)
            result = ReplayableHTTPResult(
                status_code=response.status_code,
                content=response.body,
                media_type=response.media_type,
            )
            if lease is None:
                return await _release(result)
            return await _settle_callback_failure(
                lease,
                reason=str(exc),
                result=result,
            )
        except BaseException as exc:
            result = ReplayableHTTPResult(
                status_code=502,
                content=json.dumps(
                    {"error": f"Router selection failed: {exc}"}
                ).encode(),
                media_type="application/json",
            )
            if lease is not None:
                response = await _settle_callback_failure(
                    lease,
                    reason=str(exc),
                    result=result,
                )
            else:
                response = await _release(result)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return response

        headers = _forwarding_headers(dict(request.headers))
        headers[WORKER_ID_HEADER] = worker_id

        try:
            resp = await forward_request(
                f"{worker_addr}/rl/start_session",
                body,
                headers,
                config.forward_timeout,
                client=_client(),
            )
        except BaseException as exc:
            if lease is not None:
                fallback_binding = OnlineLeaseBinding(
                    admission_id=lease.lease_id,
                    worker_addr=worker_addr,
                    worker_id=worker_id,
                    group_id="",
                    session_ids=(),
                )
                response = await _settle_callback_failure(
                    lease,
                    reason=str(exc),
                    binding=fallback_binding,
                    result=ReplayableHTTPResult(
                        status_code=502,
                        content=json.dumps(
                            {"error": f"Data proxy start_session failed: {exc}"}
                        ).encode(),
                        media_type="application/json",
                    ),
                )
            else:
                # Keep the worker binding so a retry reaches the same
                # DataProxy admission cache and can recover a lost 201.
                response = await _release_json(
                    {"error": f"Data proxy start_session failed: {exc}"}, 502
                )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return response

        group_id: str | None = None
        sessions: list[dict[str, str]] = []
        start_success_settled = False
        if resp.status_code == 201:
            try:
                resp_data = resp.json()
                group_id = resp_data["group_id"]
                sessions = resp_data.get("sessions", [])

                binding: OnlineLeaseBinding | None = None
                if lease is not None:
                    binding = OnlineLeaseBinding(
                        admission_id=lease.lease_id,
                        worker_addr=worker_addr,
                        worker_id=worker_id,
                        group_id=group_id,
                        session_ids=tuple(str(item["session_id"]) for item in sessions),
                    )
                    # Persist the compensation target before Router mutation.
                    # A cancellation or registration failure can then retry
                    # cleanup from the lease tombstone.
                    await online_lease_registry.bind(lease.lease_id, binding)

                await register_session_in_router(
                    config.router_addr,
                    sessions,
                    worker_addr,
                    config.router_timeout,
                    admin_api_key=config.admin_api_key,
                    group_id=group_id,
                    worker_id=worker_id,
                    client=_client(),
                )

                async def _finalize_registered_start() -> Response:
                    nonlocal start_success_settled
                    if lease is None:
                        assert start_worker_binding is not None
                        response = await _finish_json(resp_data, 201)
                        await _forget_start_worker(
                            resolved_request_id, start_worker_binding
                        )
                    else:
                        (
                            active,
                            deferred_binding,
                        ) = await online_lease_registry.finish_start(lease.lease_id)
                        cleanup_ok = True
                        if deferred_binding is not None:
                            cleanup_ok = await _cleanup_and_ack_online_binding(
                                lease.lease_id, deferred_binding
                            )
                        if not active:
                            status_code = 409 if cleanup_ok else 502
                            response = await _finish_json(
                                {
                                    "error": (
                                        "Online lease ended while start_session was "
                                        "still registering"
                                    )
                                },
                                status_code,
                            )
                        else:
                            response = await _finish_json(resp_data, 201)
                    start_success_settled = True
                    return response

                return await _await_critical_task(
                    asyncio.create_task(_finalize_registered_start()),
                    "Registered start settlement",
                )
            except asyncio.CancelledError:
                if start_success_settled:
                    raise
                cancelled_binding = OnlineLeaseBinding(
                    admission_id=downstream_admission_id,
                    worker_addr=worker_addr,
                    worker_id=worker_id,
                    group_id=group_id or "",
                    session_ids=tuple(str(item["session_id"]) for item in sessions),
                )

                async def _compensate_cancelled_start() -> None:
                    if lease is not None:
                        await online_lease_registry.retain_cleanup_binding(
                            lease.lease_id, cancelled_binding
                        )
                        await _fail_online_lease(
                            lease, "start_session owner was cancelled"
                        )
                        await _cleanup_and_ack_online_binding(
                            lease.lease_id, cancelled_binding
                        )
                        await online_lease_registry.finish_start(lease.lease_id)
                    else:
                        assert start_worker_binding is not None
                        await start_request_workers.retain_cleanup(
                            resolved_request_id,
                            start_worker_binding,
                            cancelled_binding,
                        )
                        await _cleanup_and_ack_start_binding(
                            resolved_request_id,
                            start_worker_binding,
                            cancelled_binding,
                        )
                    await start_request_registry.release_pending(
                        resolved_request_id,
                        fingerprint,
                        ReplayableHTTPResult(
                            status_code=503,
                            content=b'{"error":"start_session owner cancelled; retry"}',
                            media_type="application/json",
                        ),
                    )

                compensation_task = asyncio.create_task(_compensate_cancelled_start())
                await _await_critical_task(
                    compensation_task, "Cancelled start compensation"
                )
                raise
            except RouterSessionRegistrationError as exc:
                conflict_binding = OnlineLeaseBinding(
                    admission_id=downstream_admission_id,
                    worker_addr=worker_addr,
                    worker_id=worker_id,
                    group_id=group_id or "",
                    session_ids=tuple(str(item["session_id"]) for item in sessions),
                )
                result = ReplayableHTTPResult(
                    status_code=exc.status_code,
                    content=json.dumps({"error": exc.detail}).encode(),
                    media_type="application/json",
                )
                if lease is not None:
                    return await _settle_callback_failure(
                        lease,
                        reason=exc.detail,
                        binding=conflict_binding,
                        result=result,
                    )

                async def _settle_pull_conflict() -> Response:
                    assert start_worker_binding is not None
                    await start_request_workers.retain_cleanup(
                        resolved_request_id,
                        start_worker_binding,
                        conflict_binding,
                    )
                    await _cleanup_and_ack_start_binding(
                        resolved_request_id,
                        start_worker_binding,
                        conflict_binding,
                    )
                    return await _finish(result)

                settle_task = asyncio.create_task(_settle_pull_conflict())
                return await _await_critical_task(
                    settle_task, "Pull registration conflict settlement"
                )
            except Exception as exc:
                failed_binding: OnlineLeaseBinding | None = None
                if lease is not None:
                    failed_binding = OnlineLeaseBinding(
                        admission_id=lease.lease_id,
                        worker_addr=worker_addr,
                        worker_id=worker_id,
                        group_id=group_id or "",
                        session_ids=tuple(str(item["session_id"]) for item in sessions),
                    )
                logger.error("Failed to register session in router: %s", exc)
                traceback.print_exc()
                result = ReplayableHTTPResult(
                    status_code=502,
                    content=json.dumps(
                        {
                            "error": (
                                "Session created on worker but router registration "
                                f"failed: {exc}"
                            )
                        }
                    ).encode(),
                    media_type="application/json",
                )
                if lease is not None:
                    assert failed_binding is not None
                    return await _settle_callback_failure(
                        lease,
                        reason=str(exc),
                        binding=failed_binding,
                        result=result,
                    )
                # Keep both the worker and DataProxy admission binding.  A
                # retry can replay the same 201 and safely retry Router's
                # idempotent group registration.
                return await _release(result)

        result = ReplayableHTTPResult(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
        if lease is not None:
            fallback_binding = OnlineLeaseBinding(
                admission_id=lease.lease_id,
                worker_addr=worker_addr,
                worker_id=worker_id,
                group_id="",
                session_ids=(),
            )
            return await _settle_callback_failure(
                lease,
                reason=f"Data proxy returned HTTP {resp.status_code}",
                binding=fallback_binding,
                result=result,
            )
        if result.status_code >= 500:
            return await _release(result)
        assert start_worker_binding is not None

        async def _finalize_terminal_pull_start() -> Response:
            response = await _finish(result)
            await _forget_start_worker(resolved_request_id, start_worker_binding)
            return response

        return await _await_critical_task(
            asyncio.create_task(_finalize_terminal_pull_start()),
            "Terminal pull start settlement",
        )

    # =========================================================================
    # POST /rl/set_reward — session key or admin key (HITL)
    # =========================================================================

    @app.post("/rl/set_reward")
    async def set_reward(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model = None
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            route_result = await query_router(
                config.router_addr,
                token,
                "/rl/set_reward",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model,
                return_destination=True,
                client=_client(),
            )
            if not isinstance(route_result, RouterDestination):
                raise RouterUnreachableError(
                    "Router reward route did not return a worker incarnation"
                )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)
        worker_addr = route_result.worker_addr
        headers[WORKER_ID_HEADER] = route_result.worker_id

        resp = await forward_request(
            f"{worker_addr}/rl/set_reward",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /pause_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/pause_generation/{worker_id}", response_model=BroadcastResponse)
    async def pause_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _targeted_control_headers(request, worker_id)
        return await _forward_targeted_control(
            worker_addr, "/pause_generation", body, headers
        )

    # =========================================================================
    # POST /continue_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/continue_generation/{worker_id}", response_model=BroadcastResponse)
    async def continue_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _targeted_control_headers(request, worker_id)
        return await _forward_targeted_control(
            worker_addr, "/continue_generation", body, headers
        )

    # =========================================================================
    # POST /release_memory_occupation/{worker_id} — admin key ONLY
    # =========================================================================

    @app.post(
        "/release_memory_occupation/{worker_id}", response_model=BroadcastResponse
    )
    async def release_memory_occupation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _targeted_control_headers(request, worker_id)
        return await _forward_targeted_control(
            worker_addr, "/release_memory_occupation", body, headers
        )

    # =========================================================================
    # POST /resume_memory_occupation/{worker_id} — admin key ONLY
    # =========================================================================

    @app.post("/resume_memory_occupation/{worker_id}", response_model=BroadcastResponse)
    async def resume_memory_occupation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _targeted_control_headers(request, worker_id)
        return await _forward_targeted_control(
            worker_addr, "/resume_memory_occupation", body, headers
        )

    # =========================================================================
    # POST /export_trajectories — admin key ONLY, route by session_ids
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.body()

        try:
            body_json = json.loads(body)
        except (json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        session_ids: list[str] = body_json.get("session_ids") or []
        group_id: str | None = body_json.get("group_id")

        if not session_ids:
            return JSONResponse({"error": "session_ids is required"}, status_code=400)

        request_id = body_json.get("request_id")
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > 256
        ):
            return JSONResponse(
                {"error": "request_id must be a non-empty string up to 256 characters"},
                status_code=422,
            )
        fingerprint_payload = dict(body_json)
        fingerprint_payload.pop("request_id", None)
        fingerprint = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        try:
            reservation = await export_request_registry.reserve(request_id, fingerprint)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        if not reservation.is_owner:
            try:
                cached = await export_request_registry.wait(
                    reservation, timeout=config.forward_timeout
                )
            except TimeoutError:
                return JSONResponse(
                    {"error": f"Export request {request_id} is still pending"},
                    status_code=504,
                )
            if group_id is not None:
                await _retry_export_group_cleanup_if_pending(group_id)
            return Response(
                content=cached.content,
                status_code=cached.status_code,
                media_type=cached.media_type,
            )

        cleanup_slot_reserved = False
        if group_id is not None:
            admitted, cleanup_slot_reserved = await _reserve_export_group_cleanup(
                group_id
            )
            if not admitted:
                result = ReplayableHTTPResult(
                    status_code=503,
                    content=json.dumps(
                        {
                            "error": (
                                "Export cleanup capacity is exhausted; retry after "
                                "pending Router cleanup completes"
                            )
                        }
                    ).encode(),
                    media_type="application/json",
                )
                await export_request_registry.release_pending(
                    request_id, fingerprint, result
                )
                return Response(
                    content=result.content,
                    status_code=result.status_code,
                    media_type=result.media_type,
                )

        export_worker_binding: RequestWorkerBinding | None = None

        async def _finalize_export_ownership(
            result: ReplayableHTTPResult,
            *,
            terminal: bool,
            cleanup_required: bool,
        ) -> None:
            """Settle replay and cleanup-slot ownership as one critical section."""

            if cleanup_required:
                assert group_id is not None
                await _handoff_export_group_cleanup(
                    group_id,
                    worker_addr,
                    worker_id,
                    tuple(session_ids),
                )
            elif cleanup_slot_reserved:
                await _release_export_group_cleanup_reservation(group_id)

            if terminal:
                await export_request_registry.finish(request_id, result)
                assert export_worker_binding is not None
                await _forget_export_worker(request_id, export_worker_binding)
            else:
                await export_request_registry.release_pending(
                    request_id, fingerprint, result
                )

            if cleanup_required:
                await _ensure_export_group_cleanup(group_id)

        async def _settle_export_ownership(
            result: ReplayableHTTPResult,
            *,
            terminal: bool,
            cleanup_required: bool = False,
        ) -> None:
            finalize_task = asyncio.create_task(
                _finalize_export_ownership(
                    result,
                    terminal=terminal,
                    cleanup_required=cleanup_required,
                )
            )
            await _await_critical_task(finalize_task, "Export ownership settlement")

        try:
            recalled = await _recalled_export_worker(request_id, fingerprint)
            if recalled is not None:
                export_worker_binding = recalled.binding
            else:
                route_result = await query_router(
                    config.router_addr,
                    timeout=config.router_timeout,
                    session_id=session_ids[0],
                    admin_api_key=config.admin_api_key,
                    return_destination=True,
                    client=_client(),
                )
                if not isinstance(route_result, RouterDestination):
                    raise RouterUnreachableError(
                        "Router export route did not return a worker incarnation"
                    )
                export_worker_binding = await _remember_export_worker(
                    request_id,
                    fingerprint,
                    route_result.worker_addr,
                    route_result.worker_id,
                )
            worker_addr = export_worker_binding.worker_addr
            worker_id = export_worker_binding.worker_id
        except RequestWorkerOwnershipCapacityError as exc:
            result = ReplayableHTTPResult(
                status_code=503,
                content=json.dumps({"error": str(exc)}).encode(),
                media_type="application/json",
            )
            await _settle_export_ownership(result, terminal=False)
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            response = _router_error_response(exc)
            await _settle_export_ownership(
                ReplayableHTTPResult(
                    status_code=response.status_code,
                    content=response.body,
                    media_type=response.media_type,
                ),
                terminal=False,
            )
            return response
        except BaseException as exc:
            result = ReplayableHTTPResult(
                status_code=502,
                content=json.dumps(
                    {"error": f"Router selection failed: {exc}"}
                ).encode(),
                media_type="application/json",
            )
            await _settle_export_ownership(result, terminal=False)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )

        headers = _forwarding_headers(dict(request.headers))
        headers[WORKER_ID_HEADER] = worker_id
        try:
            resp = await forward_request(
                f"{worker_addr}/export_trajectories",
                body,
                headers,
                config.forward_timeout,
                client=_client(),
            )
        except BaseException as exc:
            # The worker may have committed the destructive export before the
            # transport failed. Release only the gateway reservation so a
            # caller retry can recover the worker's cached response.
            result = ReplayableHTTPResult(
                status_code=502,
                content=json.dumps(
                    {"error": f"Data proxy export failed: {exc}"}
                ).encode(),
                media_type="application/json",
            )
            await _settle_export_ownership(result, terminal=False)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )

        result = ReplayableHTTPResult(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
        if resp.status_code >= 500:
            await _settle_export_ownership(result, terminal=False)
            return Response(
                content=result.content,
                status_code=result.status_code,
                media_type=result.media_type,
            )

        await _settle_export_ownership(
            result,
            terminal=True,
            cleanup_required=resp.status_code == 200 and group_id is not None,
        )
        return Response(
            content=result.content,
            status_code=result.status_code,
            media_type=result.media_type,
        )

    # =========================================================================
    # POST /set_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/set_version/{worker_id}")
    async def set_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _targeted_control_headers(request, worker_id)
        resp = await forward_request(
            f"{worker_addr}/set_version",
            body,
            headers,
            config.forward_timeout,
            client=_client(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # GET /get_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.get("/get_version/{worker_id}")
    async def get_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
                client=_client(),
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        try:
            resp = await _client().get(
                f"{worker_addr}/get_version",
                headers=_targeted_control_headers(request, worker_id),
                timeout=config.forward_timeout,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    # =========================================================================
    # Compatibility aliases for RolloutCallback — map /callback/* to endpoints
    # =========================================================================
    # RolloutCallback uses /callback/* prefixed paths for generation control.
    # Gateway implements the actual handlers at unprefixed paths.  These aliases
    # register the SAME handler functions on both routes.
    # POST /callback/pause_generation/{worker_id} → pause_generation
    app.add_api_route(
        "/callback/pause_generation/{worker_id}",
        pause_generation,
        methods=["POST"],
    )

    # POST /callback/continue_generation/{worker_id} → continue_generation
    app.add_api_route(
        "/callback/continue_generation/{worker_id}",
        continue_generation,
        methods=["POST"],
    )

    # =========================================================================
    # OpenAI / OpenRouter compatibility aliases — /v1/* prefixed routes
    # =========================================================================
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        "/v1/models",
        list_models,
        methods=["GET"],
    )

    return app
