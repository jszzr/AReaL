# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import Response as RawResponse
from fastapi.responses import StreamingResponse
from flask import Flask
from pydantic import BaseModel

from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    concat_string_interactions,
)
from areal.infra.rpc.guard.data_blueprint import (
    data_bp,
)
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import serialize_value
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import (
    CancelSessionsRequest,
    ExportReplayCapacityError,
    ExportReplayConflictError,
    ExportReplayExpiredError,
    ExportReplayResultTooLargeError,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    ReadyNotification,
    SessionCredentials,
    SessionData,
    SessionStore,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    TrajectoryDeliveryMode,
)
from areal.v2.inference_service.data_proxy.tokenizer_proxy import (
    TokenizerProxy,
)
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend
from areal.v2.inference_service.worker_identity import WORKER_ID_HEADER

logger = logging.getLogger("InferenceDataProxy")

_MAX_ADMISSION_TOMBSTONES = 4096


# =============================================================================
# Response models
# =============================================================================


class DataProxyHealthResponse(BaseModel):
    status: str
    worker_id: str | None
    backend: str | None
    sessions: int
    paused: bool
    version: int


class DataProxyStatusResponse(BaseModel):
    status: str


class PauseGenerationResponse(BaseModel):
    status: str
    paused: bool


class SetVersionResponse(BaseModel):
    status: str
    version: int


class GetVersionResponse(BaseModel):
    version: int


class SetRewardResponse(BaseModel):
    message: str
    interaction_count: int
    session_id: str
    trajectory_id: int | None
    trajectory_ready: bool
    ready_transition: bool


class RegisterModelResponse(BaseModel):
    status: str
    name: str


class ConfigureBackendResponse(BaseModel):
    status: str
    backend_addr: str


# =============================================================================
# API Key helpers (for RL control-plane endpoints only)
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
    )


def _require_admin_key(request: Request, store: SessionStore) -> str:
    """Validate that the request carries the admin API key."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, store.admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request, store: SessionStore) -> str:
    """Resolve session_id from the session API key in the Authorization header."""
    token = _extract_bearer_token(request)
    session = store.get_session_by_api_key(token)
    if session is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session API key."
        )
    return session.session_id


def _require_worker_identity(request: Request, config: DataProxyConfig) -> None:
    """Reject a control request addressed to another process incarnation."""
    if config.worker_id is None:
        return
    if request.headers.get(WORKER_ID_HEADER) != config.worker_id:
        raise HTTPException(
            status_code=409,
            detail="Data Proxy incarnation mismatch",
        )


def _resolve_session_from_token(
    token: str | None,
    store: SessionStore,
    *,
    create_hitl: bool = True,
) -> SessionData | None:
    """Resolve a session from the bearer token.

    Session key → lookup by API key.
    Admin key → persistent HITL session.
    """
    if token is None:
        return None
    session = store.get_session_by_api_key(token)
    if session is not None:
        return session
    if create_hitl and hmac.compare_digest(token, store.admin_api_key):
        return store.get_or_create_hitl_session()
    return None


def _try_extract_bearer_token(request: Request) -> str | None:
    """Extract bearer token if present. Returns None if missing/malformed.

    Unlike _extract_bearer_token, this never raises — it's for endpoints
    that accept requests with or without auth.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def _create_inf_bridge(
    backend_addr: str,
    pause_state: PauseState,
    config: DataProxyConfig,
) -> InfBridge:
    """Create an InfBridge instance from proxy config."""
    if config.backend_type == "sglang":
        backend = SGLangBridgeBackend()
    elif config.backend_type == "vllm":
        backend = VLLMBridgeBackend()
    else:
        raise ValueError(f"Unsupported backend_type: {config.backend_type!r}")

    return InfBridge(
        backend=backend,
        backend_addr=backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )


def _create_areal_client(
    inf_bridge: InfBridge,
    tok: TokenizerProxy,
    config: DataProxyConfig,
) -> ArealOpenAI:
    """Create an ArealOpenAI client backed by the given InfBridge."""
    return ArealOpenAI(
        engine=inf_bridge,
        tokenizer=tok._tok,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        engine_max_tokens=config.engine_max_tokens,
        chat_template_type=config.chat_template_type,
    )


async def _post_online_ready_callback(
    callback_server_addr: str,
    admin_api_key: str,
    notification: ReadyNotification,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    if not callback_server_addr:
        return False

    callback_base = callback_server_addr.rstrip("/")
    try:

        async def _do(c: httpx.AsyncClient) -> httpx.Response:
            return await c.post(
                f"{callback_base}/callback/online_ready",
                json={
                    "session_id": notification.session_id,
                    "trajectory_id": notification.trajectory_id,
                    "lease_id": notification.lease_id,
                    "expected_version": notification.expected_version,
                    "group_id": notification.group_id,
                },
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )

        if client is not None:
            resp = await _do(client)
        else:
            async with create_httpx_client(timeout=timeout) as c:
                resp = await _do(c)

        if resp.status_code >= 400:
            logger.warning(
                "Online ready callback failed for %s/%s with %d: %s",
                notification.session_id,
                notification.trajectory_id,
                resp.status_code,
                resp.text,
            )
            return False
        return True
    except Exception as exc:
        logger.warning(
            "Online ready callback unreachable for %s/%s: %s",
            notification.session_id,
            notification.trajectory_id,
            exc,
        )
        return False


async def _flush_ready_trajectories(app: FastAPI) -> None:
    store: SessionStore = app.state.session_store
    config: DataProxyConfig = app.state.config
    http_client: httpx.AsyncClient | None = getattr(app.state, "http_client", None)

    for ready_result in store.finalize_rewarded_trajectories():
        logger.info(
            "Trajectory ready: session=%s trajectory=%s interactions=%s",
            ready_result.session_id,
            ready_result.trajectory_id,
            ready_result.interaction_count,
        )

    pending_notifications = store.pending_online_callbacks()

    async def _deliver(
        notification: ReadyNotification,
    ) -> tuple[ReadyNotification, bool]:
        try:
            delivered = await _post_online_ready_callback(
                config.callback_server_addr,
                config.admin_api_key,
                notification,
                config.request_timeout,
                client=http_client,
            )
        except BaseException as exc:
            logger.warning(
                "Callback delivery failed for %s/%s: %s",
                notification.session_id,
                notification.trajectory_id,
                exc,
            )
            delivered = False
        return notification, delivered

    results = await asyncio.gather(*[_deliver(n) for n in pending_notifications])
    for notification, delivered in results:
        if delivered:
            store.mark_online_callback_delivered(
                notification.session_id,
                notification.trajectory_id,
            )


async def _ready_trajectory_loop(app: FastAPI) -> None:
    while True:
        await _flush_ready_trajectories(app)
        await asyncio.sleep(0.1)


def create_app(config: DataProxyConfig) -> FastAPI:
    """Factory that creates the FastAPI app with lifespan-managed resources."""

    admission_lock = asyncio.Lock()
    export_lock = asyncio.Lock()
    admission_records: dict[str, tuple[str, StartSessionResponse]] = {}
    cancelled_admissions: dict[str, None] = {}

    def _tombstone_admission_locked(admission_id: str) -> None:
        cancelled_admissions[admission_id] = None
        if len(cancelled_admissions) > _MAX_ADMISSION_TOMBSTONES:
            oldest = next(iter(cancelled_admissions))
            cancelled_admissions.pop(oldest, None)

    async def _close_admissions_for_sessions(session_ids: set[str]) -> None:
        """Bound admission replay state to the lifetime of its sessions."""

        if not session_ids:
            return
        async with admission_lock:
            for admission_id, (_, response) in list(admission_records.items()):
                if any(
                    credential.session_id in session_ids
                    for credential in response.sessions
                ):
                    admission_records.pop(admission_id, None)
                    _tombstone_admission_locked(admission_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Data proxy starting — backend=%s, tokenizer=%s",
            config.backend_addr or "(none)",
            config.tokenizer_path,
        )

        pause_state = PauseState()
        app.state.pause_state = pause_state
        app.state.config = config
        app.state.session_store = SessionStore(
            set_reward_finish_timeout=config.set_reward_finish_timeout,
            max_export_replay_records=config.max_export_replay_records,
            export_replay_ttl_seconds=config.export_replay_ttl_seconds,
            max_export_replay_result_bytes=config.max_export_replay_result_bytes,
            max_export_replay_total_bytes=config.max_export_replay_total_bytes,
        )
        app.state.session_store.set_admin_key(config.admin_api_key)
        app.state.version = 0
        app.state.http_client = create_httpx_client(timeout=config.request_timeout)

        if not config.backend_addr:
            app.state.tokenizer = None
            app.state.inf_bridge = None
            app.state.areal_client = None
        else:
            tok = TokenizerProxy(config.tokenizer_path)
            inf_bridge = _create_inf_bridge(config.backend_addr, pause_state, config)
            areal_client = _create_areal_client(inf_bridge, tok, config)
            app.state.tokenizer = tok
            app.state.inf_bridge = inf_bridge
            app.state.areal_client = areal_client

        ready_task = asyncio.create_task(_ready_trajectory_loop(app))
        try:
            yield
        finally:
            ready_task.cancel()
            try:
                await ready_task
            except asyncio.CancelledError:
                pass
            if app.state.inf_bridge is not None:
                await app.state.inf_bridge.aclose()
            await app.state.http_client.aclose()
        logger.info("Data proxy shutting down")

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)
    app.state.admission_records = admission_records
    app.state.cancelled_admissions = cancelled_admissions
    _registered_models: dict[str, dict[str, str | None]] = {}

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=DataProxyHealthResponse)
    async def health():
        store: SessionStore = app.state.session_store
        pause_state: PauseState = app.state.pause_state
        return DataProxyHealthResponse(
            status="ok",
            worker_id=config.worker_id,
            backend=config.backend_addr,
            sessions=store.session_count,
            paused=await pause_state.is_paused(),
            version=app.state.version,
        )

    @app.post("/configure", response_model=DataProxyStatusResponse)
    async def configure():
        return DataProxyStatusResponse(status="ok")

    # =========================================================================
    # Pause/Resume — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/pause_generation", response_model=PauseGenerationResponse)
    async def pause_generation(request: Request):
        _require_worker_identity(request, config)
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.pause()
        return PauseGenerationResponse(status="ok", paused=True)

    @app.post("/continue_generation", response_model=PauseGenerationResponse)
    async def continue_generation(request: Request):
        _require_worker_identity(request, config)
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.resume()
        return PauseGenerationResponse(status="ok", paused=False)

    @app.post("/release_memory_occupation")
    async def release_memory_occupation(request: Request):
        _require_worker_identity(request, config)
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.offload()
        return {"status": "ok"}

    @app.post("/resume_memory_occupation")
    async def resume_memory_occupation(request: Request):
        _require_worker_identity(request, config)
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        body = await request.json() if await request.body() else {}
        tags = body.get("tags")
        await inf_bridge.onload(tags=tags)
        return {"status": "ok"}

    # =========================================================================
    # Version management — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/set_version", response_model=SetVersionResponse)
    async def set_version(request: Request):
        _require_worker_identity(request, config)
        body = await request.json()
        version = body.get("version")
        if version is None or not isinstance(version, int):
            raise HTTPException(status_code=400, detail="'version' (int) is required")
        app.state.version = version

        # Propagate version to InfBridge so it stamps correct versions on generated tokens
        if app.state.inf_bridge is not None:
            app.state.inf_bridge.set_version(version)

        return SetVersionResponse(status="ok", version=version)

    @app.get("/get_version", response_model=GetVersionResponse)
    async def get_version(request: Request):
        _require_worker_identity(request, config)
        return GetVersionResponse(version=app.state.version)

    # =========================================================================
    # Session management (admin key / session key required)
    # =========================================================================

    @app.post("/rl/start_session", status_code=201)
    async def start_session(
        body: StartSessionRequest, request: Request
    ) -> StartSessionResponse:
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        _require_worker_identity(request, config)

        group_size = max(body.group_size, 1)
        if body.delivery_mode is TrajectoryDeliveryMode.CALLBACK:
            if group_size != 1:
                raise HTTPException(
                    status_code=422,
                    detail="Callback delivery currently requires group_size=1",
                )
            missing = [
                name
                for name, value in (
                    ("lease_id", body.lease_id),
                    ("admission_id", body.admission_id),
                    ("expected_version", body.expected_version),
                )
                if value is None
            ]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Callback delivery requires {', '.join(missing)}",
                )
            if body.lease_id != body.admission_id:
                raise HTTPException(
                    status_code=422,
                    detail="lease_id and admission_id must match",
                )
            if body.expected_version != app.state.version:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Worker policy version is {app.state.version}, but lease "
                        f"expects {body.expected_version}"
                    ),
                )

        fingerprint = json.dumps(body.model_dump(mode="json"), sort_keys=True)

        async def _create() -> StartSessionResponse:
            group_id = f"grp-{uuid.uuid4()}"
            credentials: list[SessionCredentials] = []
            try:
                for i in range(group_size):
                    session_id, session_api_key = store.start_session(
                        body.task_id,
                        body.api_key if i == 0 else None,
                        delivery_mode=body.delivery_mode,
                        lease_id=body.lease_id,
                        expected_version=body.expected_version,
                        group_id=group_id,
                    )
                    credentials.append(
                        SessionCredentials(
                            session_id=session_id,
                            session_api_key=session_api_key,
                        )
                    )
            except ValueError as exc:
                for credential in credentials:
                    store.remove_session(credential.session_id)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return StartSessionResponse(group_id=group_id, sessions=credentials)

        if body.admission_id is None:
            return await _create()

        async with admission_lock:
            if body.admission_id in cancelled_admissions:
                raise HTTPException(
                    status_code=410,
                    detail=f"Admission {body.admission_id} is closed or cancelled",
                )
            existing = admission_records.get(body.admission_id)
            if existing is not None:
                existing_fingerprint, response = existing
                if existing_fingerprint != fingerprint:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Admission {body.admission_id} was replayed with a "
                            "different request"
                        ),
                    )
                return response

            response = await _create()
            admission_records[body.admission_id] = (fingerprint, response)
            return response

    @app.post("/rl/cancel_sessions")
    async def cancel_sessions(body: CancelSessionsRequest, request: Request):
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        _require_worker_identity(request, config)
        async with admission_lock:
            _tombstone_admission_locked(body.admission_id)
            session_ids = set(body.session_ids)
            existing = admission_records.pop(body.admission_id, None)
            if existing is not None:
                session_ids.update(
                    credential.session_id for credential in existing[1].sessions
                )
            removed = 0
            for session_id in session_ids:
                if store.get_session(session_id) is not None:
                    store.remove_session(session_id)
                    removed += 1
        return {"status": "ok", "removed": removed}

    @app.post("/rl/set_reward", response_model=SetRewardResponse)
    async def set_reward(body: SetRewardRequest, request: Request):
        store: SessionStore = app.state.session_store
        token = _extract_bearer_token(request)
        session = _resolve_session_from_token(token, store, create_hitl=False)
        is_admin = hmac.compare_digest(token, store.admin_api_key)
        if session is None and not is_admin:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session API key."
            )
        _require_worker_identity(request, config)
        if session is None:
            session = store.get_or_create_hitl_session()

        try:
            reward_result = session.set_reward(
                interaction_id=body.interaction_id,
                reward=body.reward,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SetRewardResponse(
            message="success",
            interaction_count=reward_result.interaction_count,
            session_id=reward_result.session_id,
            trajectory_id=reward_result.trajectory_id,
            trajectory_ready=reward_result.trajectory_id is not None,
            ready_transition=reward_result.ready_transition,
        )

    # =========================================================================
    # Chat completions — OpenAI-compatible
    #
    # If the bearer token is a known session key, use session cache.
    # Otherwise (no token, admin key, unknown key) → standalone mode.
    # Data proxy never rejects requests on /chat/completions.
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        raw_body = await request.body()
        try:
            body_json = json.loads(raw_body)
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        model_name = body_json.get("model")
        store: SessionStore = app.state.session_store

        token = _try_extract_bearer_token(request)
        session = _resolve_session_from_token(token, store, create_hitl=False)
        _require_worker_identity(request, config)
        if (
            session is None
            and token is not None
            and hmac.compare_digest(token, store.admin_api_key)
        ):
            session = store.get_or_create_hitl_session()
        if session is not None:
            session.update_last_access()

        # -----------------------------------------------------------------
        # External model path: model is a registered external model name
        # -----------------------------------------------------------------
        ext_info = _registered_models.get(model_name) if model_name else None
        if ext_info is not None and ext_info.get("url"):
            ext_url = (ext_info["url"] or "").rstrip("/")
            ext_model = ext_info["model"]
            provider_api_key = ext_info.get("api_key")

            forward_body = dict(body_json)
            if ext_model is not None:
                forward_body["model"] = ext_model
            else:
                forward_body.pop("model", None)

            _skip = {"host", "content-length", "transfer-encoding", "authorization"}
            forward_headers = {
                k: v for k, v in dict(request.headers).items() if k.lower() not in _skip
            }
            if provider_api_key:
                forward_headers["authorization"] = f"Bearer {provider_api_key}"

            is_streaming = forward_body.get("stream", False) or False
            messages = body_json.get("messages", [])

            if is_streaming:
                collected_chunks: list[str] = []

                async def _stream_and_cache():
                    success = False
                    try:
                        async with create_httpx_client(
                            timeout=httpx.Timeout(config.request_timeout)
                        ) as stream_client:
                            async with stream_client.stream(
                                "POST",
                                f"{ext_url}/chat/completions",
                                json=forward_body,
                                headers=forward_headers,
                            ) as resp:
                                if resp.status_code != 200:
                                    error_body = await resp.aread()
                                    yield (
                                        f"data: {json.dumps({'error': error_body.decode()})}\n\n".encode()
                                    )
                                    return
                                async for chunk in resp.aiter_bytes():
                                    decoded = chunk.decode("utf-8", errors="replace")
                                    collected_chunks.append(decoded)
                                    yield chunk
                                success = True
                    except Exception as exc:
                        logger.error(
                            "External stream error for %s: %s", model_name, exc
                        )
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
                    finally:
                        if success and collected_chunks and session is not None:
                            session.add_string_interaction(
                                messages,
                                "".join(collected_chunks),
                            )

                return StreamingResponse(
                    _stream_and_cache(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            full_url = f"{ext_url}/chat/completions"
            try:
                resp = await app.state.http_client.post(
                    full_url,
                    json=forward_body,
                    headers=forward_headers,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"External API error: {exc}"
                )

            if resp.status_code != 200:
                logger.error(
                    "External API returned %d for %s: %s",
                    resp.status_code,
                    full_url,
                    resp.text[:500],
                )

            response_str = resp.text

            if resp.status_code == 200 and session is not None:
                session.add_string_interaction(messages, response_str)

            return RawResponse(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

        # -----------------------------------------------------------------
        # Internal model path: use AReaL inference server
        # -----------------------------------------------------------------
        areal_client: ArealOpenAI = app.state.areal_client

        if session is not None:
            areal_cache: Any = session.active_completions
        else:
            areal_cache = None

        # Build kwargs from request body
        kwargs = dict(body_json)
        # Remove model (ArealOpenAI ignores it)
        kwargs.pop("model", None)

        # Determine streaming
        is_streaming = kwargs.get("stream", False) or False

        # Apply defaults for temperature/top_p if not set
        if "temperature" not in kwargs:
            kwargs["temperature"] = 1.0
        if "top_p" not in kwargs:
            kwargs["top_p"] = 1.0

        create_fn: Any = areal_client.chat.completions.create

        try:
            result = await create_fn(
                areal_cache=areal_cache,
                **kwargs,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        if is_streaming:
            # result is an async generator of ChatCompletionChunk

            async def _sse_stream():
                async for chunk in result:
                    yield f"data: {chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return result

    @app.post("/register_model", response_model=RegisterModelResponse)
    async def register_model(request: Request):
        body = await request.json()
        name = body.get("name") or body.get("model")
        url = body.get("url", "")
        model = body.get("model", name)
        api_key = body.get("api_key")
        if not name:
            raise HTTPException(status_code=400, detail="model name is required")
        _registered_models[name] = {"url": url, "model": model, "api_key": api_key}
        logger.info("Model registered: name=%s url=%s", name, url or "(internal)")
        return RegisterModelResponse(status="ok", name=name)

    # =========================================================================
    # Trajectory export (admin key required)
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(
        body: ExportTrajectoriesRequest, request: Request
    ) -> ExportTrajectoriesResponse:
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        _require_worker_identity(request, config)

        if not body.session_ids:
            raise HTTPException(
                status_code=400,
                detail="session_ids must be a non-empty list",
            )

        request_fingerprint = body.replay_fingerprint()
        try:
            reservation = store.reserve_export_replay(
                body.request_id, request_fingerprint
            )
        except ExportReplayConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ExportReplayExpiredError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except ExportReplayCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not reservation.is_owner:
            if reservation.payload is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Export request_id {body.request_id} is still pending",
                )
            return ExportTrajectoriesResponse(traj=reservation.payload)

        try:
            await export_lock.acquire()
        except BaseException:
            store.release_export_replay(body.request_id, request_fingerprint)
            raise
        try:
            try:
                if len(set(body.session_ids)) != len(body.session_ids):
                    raise HTTPException(
                        status_code=422,
                        detail="session_ids must contain unique values",
                    )

                sessions: list[SessionData] = []
                missing_session_ids: list[str] = []
                for sid in body.session_ids:
                    session = store.get_session(sid)
                    if session is None:
                        missing_session_ids.append(sid)
                    else:
                        sessions.append(session)
                if missing_session_ids:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            "Export sessions not found: "
                            + ", ".join(missing_session_ids)
                        ),
                    )

                session_group_ids = {session.group_id for session in sessions}
                if len(session_group_ids) != 1:
                    raise HTTPException(
                        status_code=409,
                        detail="All export sessions must belong to the same group",
                    )
                actual_group_id = next(iter(session_group_ids))
                if body.group_id is not None and body.group_id != actual_group_id:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"group_id {body.group_id} does not own the requested "
                            "sessions"
                        ),
                    )
                if body.remove_session and body.group_id is not None:
                    group_members = store.session_ids_for_group(body.group_id)
                    if set(body.session_ids) != group_members:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Destructive group export must include every session "
                                f"owned by group_id {body.group_id}"
                            ),
                        )

                session_lease_ids = {session.lease_id for session in sessions}
                if len(session_lease_ids) != 1:
                    raise HTTPException(
                        status_code=409,
                        detail="All export sessions must belong to the same lease",
                    )
                actual_lease_id = next(iter(session_lease_ids))
                if body.lease_id != actual_lease_id:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"lease_id {body.lease_id} does not own the requested "
                            "sessions"
                        ),
                    )

                merged: dict[str, InteractionWithTokenLogpReward] = {}
                prepared: list[tuple[SessionData, int]] = []
                for session in sessions:
                    try:
                        prepared_id, interactions = session.prepare_trajectory_export(
                            discount=body.discount,
                            style=body.style,
                            trajectory_id=body.trajectory_id,
                        )
                    except KeyError as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Session {session.session_id} is not ready: {exc}",
                        ) from exc
                    prepared.append((session, prepared_id))
                    merged.update(interactions)

                if all(v.has_tensor_data for v in merged.values()):
                    traj = concat_padded_tensors(
                        [v.to_tensor_dict() for v in merged.values()]
                    )
                    traj = RTensor.remotize(traj, node_addr=config.serving_addr)
                else:
                    traj = concat_string_interactions(merged)

                serialized = serialize_value(traj)
                try:
                    replayable = store.finish_export_replay(
                        body.request_id,
                        request_fingerprint,
                        serialized,
                    )
                except ExportReplayResultTooLargeError as exc:
                    raise HTTPException(status_code=507, detail=str(exc)) from exc

                for session, prepared_id in prepared:
                    session.commit_trajectory_export(prepared_id)

                if body.remove_session:
                    for sid in body.session_ids:
                        store.remove_session(sid)
                    await _close_admissions_for_sessions(set(body.session_ids))
                return ExportTrajectoriesResponse(traj=replayable)
            except BaseException:
                store.release_export_replay(body.request_id, request_fingerprint)
                raise
        finally:
            export_lock.release()

    # =========================================================================
    # Runtime backend reconfiguration (for fork-based deployment)
    # =========================================================================

    @app.post("/configure_backend", response_model=ConfigureBackendResponse)
    async def configure_backend(request: Request):
        """Reconfigure the inference backend address after process start."""
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        body = await request.json()
        new_addr = body.get("backend_addr")
        if not new_addr:
            raise HTTPException(status_code=400, detail="backend_addr is required")
        pause_state: PauseState = app.state.pause_state
        tok: TokenizerProxy = app.state.tokenizer

        old_inf_bridge: InfBridge | None = app.state.inf_bridge

        # Recreate InfBridge + ArealOpenAI with new backend address
        new_inf_bridge = _create_inf_bridge(new_addr, pause_state, app.state.config)
        try:
            new_areal_client = _create_areal_client(
                new_inf_bridge, tok, app.state.config
            )
        except Exception:
            await new_inf_bridge.aclose()
            raise

        # Build updated config copy, then swap all three state fields.
        # Swap references first so new requests immediately use the new bridge.
        from dataclasses import replace as _dc_replace

        new_config = _dc_replace(app.state.config, backend_addr=new_addr)
        app.state.config = new_config
        app.state.inf_bridge = new_inf_bridge
        app.state.areal_client = new_areal_client

        # Close old InfBridge after the swap so in-flight requests that already
        # hold a reference to the old bridge can still finish their HTTP calls.
        if old_inf_bridge is not None:
            await old_inf_bridge.aclose()

        logger.info("Backend reconfigured to %s", new_addr)
        return ConfigureBackendResponse(status="ok", backend_addr=new_addr)

    # =========================================================================
    # RTensor data storage endpoints
    #
    # These endpoints are now mounted from the legacy Flask data_blueprint
    # (areal.infra.rpc.guard.data_blueprint) to ensure a single source of
    # truth for RTensor storage logic.
    #
    # This mount provides:
    # - POST   /data/batch
    # - PUT    /data/<shard_id>
    # - GET    /data/<shard_id>
    # - DELETE /data/clear
    # =========================================================================
    flask_shim = Flask("data_proxy_shim")
    flask_shim.register_blueprint(data_bp)
    app.mount("/", WSGIMiddleware(flask_shim))

    return app
