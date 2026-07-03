"""Tests for environment-only SGLang server configuration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig, SGLangConfig
from areal.v2.inference_service.controller.controller import RolloutControllerV2

SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM = (
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"
)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (None, {}),
        (
            False,
            {SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM: "0"},
        ),
        (
            True,
            {SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM: "1"},
        ),
    ],
)
def test_sglang_batch_invariant_deepgemm_builds_server_env(setting, expected):
    config = SGLangConfig(
        model_path="test-model",
        enable_batch_invariant_ops_mm_deepgemm=setting,
    )

    assert SGLangConfig.build_server_env(config) == expected


@pytest.mark.parametrize(
    "setting",
    [False, True],
)
def test_sglang_batch_invariant_deepgemm_is_not_forwarded_as_cli_arg(setting):
    config = SGLangConfig(
        model_path="test-model",
        enable_batch_invariant_ops_mm_deepgemm=setting,
    )

    with patch(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        return_value=True,
    ):
        args = SGLangConfig.build_args(
            sglang_config=config,
            tp_size=1,
            base_gpu_id=0,
        )
    cmd = SGLangConfig.build_cmd_from_args(args)

    assert "enable_batch_invariant_ops_mm_deepgemm" not in args
    assert "--enable-batch-invariant-ops-mm-deepgemm" not in cmd


async def _capture_inf_server_forks(
    *, backend: str, server_env: dict[str, str] | None
) -> list[dict]:
    backend_spec = f"{backend}:d1"
    config = InferenceEngineConfig(
        backend=backend_spec,
        admin_api_key="test-key",
    )
    controller = RolloutControllerV2(
        config=config,
        scheduler=MagicMock(n_gpus_per_node=8),
    )
    worker = SimpleNamespace(ip="10.0.0.1", worker_ports=[18000])
    fork_calls: list[dict] = []

    async def post(url, json=None, timeout=None):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url.endswith("/alloc_ports"):
            response.json.return_value = {
                "host": "10.0.0.1",
                "ports": [30000],
            }
        elif url.endswith("/fork"):
            fork_calls.append(json)
        return response

    client = AsyncMock()
    client.post.side_effect = post
    with (
        patch.object(
            controller,
            "_get_async_client",
            new=AsyncMock(return_value=client),
        ),
        patch.object(
            controller,
            "_async_wait_for_service",
            new=AsyncMock(),
        ),
    ):
        await controller._async_fork_inf_servers(
            cfg=SimpleNamespace(setup_timeout=1.0),
            alloc=ModelAllocation.from_str(backend_spec),
            inf_backend=backend,
            inf_workers=[worker],
            dp_size=1,
            nnodes_per_instance=1,
            server_args={},
            server_env=server_env,
        )
    return fork_calls


@pytest.mark.asyncio
async def test_default_sglang_fork_does_not_inject_server_env():
    fork_calls = await _capture_inf_server_forks(
        backend="sglang",
        server_env=None,
    )

    assert len(fork_calls) == 1
    assert "env" not in fork_calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "expected_value"),
    [(False, "0"), (True, "1")],
)
async def test_explicit_sglang_server_env_reaches_inf_server_fork(
    setting, expected_value
):
    server_env = SGLangConfig.build_server_env(
        SGLangConfig(enable_batch_invariant_ops_mm_deepgemm=setting)
    )

    fork_calls = await _capture_inf_server_forks(
        backend="sglang",
        server_env=server_env,
    )

    assert len(fork_calls) == 1
    assert fork_calls[0]["env"] == {
        SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM: expected_value
    }


@pytest.mark.asyncio
async def test_sglang_server_env_does_not_change_vllm_fork_env():
    fork_calls = await _capture_inf_server_forks(
        backend="vllm",
        server_env={SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM: "0"},
    )

    assert len(fork_calls) == 1
    assert set(fork_calls[0]["env"]) == {
        "TRITON_CACHE_PATH",
        "VLLM_CACHE_ROOT",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
    }
