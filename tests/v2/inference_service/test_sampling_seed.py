"""Tests for per-request sampling seed propagation through the v2 stack."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.api.cli_args import GenerationHyperparameters, SGLangConfig
from areal.api.io_struct import ModelRequest
from areal.experimental.openai.client import ArealOpenAI
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import SessionStore
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


def _sglang_response(
    token_logprobs: list[tuple[float, int]], finish_reason: str = "stop"
) -> dict:
    return {
        "meta_info": {
            "finish_reason": {"type": finish_reason},
            "output_token_logprobs": token_logprobs,
        }
    }


@pytest_asyncio.fixture
async def data_proxy_stack():
    """Build the real DataProxy -> ArealOpenAI -> SGLang bridge stack."""
    config = DataProxyConfig(
        host="127.0.0.1",
        port=0,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=1.0,
    )
    app = create_app(config)

    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        resubmit_wait=0.0,
    )
    backend_payloads: list[dict] = []

    async def capture_request(http_req, **_kwargs):
        backend_payloads.append(deepcopy(http_req.payload))
        return _sglang_response([(-0.1, 99), (-0.2, 2)])

    inf_bridge._send_request = capture_request

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [10, 11]}
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.decode.return_value = "answer"

    areal_client = ArealOpenAI(engine=inf_bridge, tokenizer=tokenizer)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.config = config
    app.state.pause_state = pause_state
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = areal_client
    app.state.tokenizer = tokenizer
    app.state.session_store = store
    app.state.version = 0

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, backend_payloads

    await areal_client.close()
    await inf_bridge.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream, seed", [(False, 0), (True, 2**63 - 1)])
async def test_chat_seed_reaches_sglang_for_streaming_and_non_streaming(
    data_proxy_stack, stream, seed
):
    """The JSON seed must survive every real in-process request boundary."""
    client, backend_payloads = data_proxy_stack

    response = await client.post(
        "/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 8,
            "seed": seed,
            "stream": stream,
        },
    )

    assert response.status_code == 200, response.text
    assert backend_payloads[-1]["sampling_params"]["sampling_seed"] == seed


@pytest.mark.asyncio
async def test_omitted_chat_seed_keeps_sglang_sampling_params_unchanged(
    data_proxy_stack,
):
    """Omitting seed must retain SGLang's existing unseeded request behavior."""
    client, backend_payloads = data_proxy_stack

    response = await client.post(
        "/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 8,
        },
    )

    assert response.status_code == 200, response.text
    assert "sampling_seed" not in backend_payloads[-1]["sampling_params"]


@pytest.mark.asyncio
async def test_invalid_chat_seed_is_rejected_before_backend_dispatch(data_proxy_stack):
    """Invalid seed errors identify the field and do not reach SGLang."""
    client, backend_payloads = data_proxy_stack

    response = await client.post(
        "/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 8,
            "seed": True,
        },
    )

    assert response.status_code >= 400
    assert "seed" in response.json()["detail"]
    assert backend_payloads == []


@pytest.mark.asyncio
async def test_sglang_abort_retry_preserves_sampling_seed():
    """Abort resubmission may shrink token budget but must not change seed."""
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr="http://mock-sglang:30000",
        pause_state=PauseState(),
        resubmit_wait=0.0,
    )
    backend_payloads: list[dict] = []

    async def capture_request(http_req, **_kwargs):
        backend_payloads.append(deepcopy(http_req.payload))
        if len(backend_payloads) == 1:
            return _sglang_response([(-0.1, 99)], "abort")
        return _sglang_response([(-0.2, 100)], "stop")

    inf_bridge._send_request = capture_request
    request = ModelRequest(
        input_ids=[10, 11],
        gconfig=GenerationHyperparameters(
            max_new_tokens=8,
            max_tokens=16,
            seed=987654321,
        ),
    )

    try:
        await inf_bridge.agenerate(request)
    finally:
        await inf_bridge.aclose()

    assert [
        payload["sampling_params"]["sampling_seed"] for payload in backend_payloads
    ] == [987654321, 987654321]


def test_generation_seed_survives_copy_and_openai_serialization():
    """Dataclass copies and OpenAI chat serialization retain an explicit seed."""
    request = ModelRequest(gconfig=GenerationHyperparameters(seed=42))

    copied = request.copy()
    openai_args = copied.gconfig.to_openai_completions_args_dict()

    assert copied.gconfig.seed == 42
    assert openai_args["seed"] == 42


def test_omitted_generation_seed_is_not_serialized_to_openai():
    """The new optional field must not alter existing outbound requests."""
    openai_args = GenerationHyperparameters().to_openai_completions_args_dict()

    assert "seed" not in openai_args


@pytest.mark.parametrize("api_format", ["responses", "openai-agents"])
def test_generation_seed_is_not_serialized_to_unsupported_openai_apis(api_format):
    """Only Chat Completions supports seed in the installed OpenAI APIs."""
    openai_args = GenerationHyperparameters(seed=42).to_openai_args_dict(
        api_format=api_format
    )

    assert "seed" not in openai_args


@pytest.mark.parametrize(
    "invalid_seed",
    [True, False, -1, 2**63, 1.5, "1"],
)
def test_generation_seed_rejects_invalid_values(invalid_seed):
    """Seeds are non-bool, non-negative signed 64-bit integers."""
    with pytest.raises(ValueError, match="seed.*non-negative signed 64-bit integer"):
        GenerationHyperparameters(seed=invalid_seed)


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_generation_seed_accepts_signed_64_bit_boundaries(seed):
    """Both supported signed 64-bit boundaries are accepted."""
    assert GenerationHyperparameters(seed=seed).seed == seed


def test_new_seed_fields_preserve_existing_positional_parameter_order():
    """New public fields are appended instead of shifting existing positions."""
    generation_fields = [field.name for field in fields(GenerationHyperparameters)]
    sglang_fields = [field.name for field in fields(SGLangConfig)]

    assert generation_fields[-3:] == ["lora_name", "use_beam_search", "seed"]
    assert sglang_fields[-3:] == [
        "enable_multithread_load",
        "enable_deterministic_inference",
        "enable_return_routed_experts",
    ]


def _build_sglang_launch(enable_deterministic_inference: bool | None):
    config_kwargs = {}
    if enable_deterministic_inference is not None:
        config_kwargs["enable_deterministic_inference"] = enable_deterministic_inference
    config = SGLangConfig(model_path="test-model", **config_kwargs)
    with patch(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        return_value=True,
    ):
        args = SGLangConfig.build_args(
            sglang_config=config,
            tp_size=1,
            base_gpu_id=0,
        )
        cmd = SGLangConfig.build_cmd(
            sglang_config=config,
            tp_size=1,
            base_gpu_id=0,
        )
    return config, args, cmd


def test_sglang_deterministic_inference_is_opt_in_by_default():
    """Existing launches remain unseeded unless users explicitly opt in."""
    config, args, cmd = _build_sglang_launch(None)

    assert config.enable_deterministic_inference is False
    assert args["enable_deterministic_inference"] is False
    assert "--enable-deterministic-inference" not in cmd


def test_sglang_deterministic_inference_reaches_server_cli_when_enabled():
    """The opt-in emits SGLang's flag that makes sampling_seed effective."""
    config, args, cmd = _build_sglang_launch(True)

    assert config.enable_deterministic_inference is True
    assert args["enable_deterministic_inference"] is True
    assert cmd.count("--enable-deterministic-inference") == 1
