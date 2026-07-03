# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import InferenceEngineConfig, SGLangConfig, vLLMConfig
from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer, _build_v2_weight_update_meta

SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM = (
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"
)


def _make_config(*, use_lora: bool, weight_update_mode: str):
    return SimpleNamespace(
        experiment_name="test-experiment",
        trial_name="test-trial",
        cluster=SimpleNamespace(fileroot="/tmp/areal-tests"),
        actor=SimpleNamespace(
            _version="v2",
            use_lora=use_lora,
            weight_update_mode=weight_update_mode,
            path="test-model",
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
        gconfig=SimpleNamespace(lora_name="test-adapter"),
        rollout=SimpleNamespace(
            _version="v2",
            api_url=None,
            backend="sglang:d1",
            max_head_offpolicyness=2,
            return_routed_experts=False,
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
        enable_offload=True,
    )


def test_v2_full_parameter_explicit_disk_mode_is_honored():
    config = _make_config(use_lora=False, weight_update_mode="disk")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "disk"
    assert meta.use_lora is False


def test_v2_full_parameter_xccl_mode_uses_awex():
    config = _make_config(use_lora=False, weight_update_mode="xccl")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "awex"


def test_v2_lora_uses_disk_regardless_of_configured_mode():
    config = _make_config(use_lora=True, weight_update_mode="xccl")

    meta = _build_v2_weight_update_meta(config)

    assert meta.type == "disk"
    assert meta.use_lora is True
    assert meta.lora_name == "test-adapter"
    assert meta.lora_keep_versions == 4


def test_v2_disk_mode_rejects_vllm_backend():
    config = _make_config(use_lora=False, weight_update_mode="disk")
    config.rollout.backend = "vllm:d1"

    with pytest.raises(ValueError, match="local SGLang"):
        _build_v2_weight_update_meta(config)


def test_v2_disk_mode_rejects_external_model():
    config = _make_config(use_lora=False, weight_update_mode="disk")
    config.rollout.backend = None
    config.rollout.api_url = "https://example.com/v1"

    with pytest.raises(ValueError, match="local SGLang"):
        _build_v2_weight_update_meta(config)


def test_v2_megatron_lora_error_recommends_supported_backend_pair():
    config = _make_config(use_lora=True, weight_update_mode="xccl")
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = config
    trainer.actor_alloc = SimpleNamespace(backend="megatron")
    trainer.rollout_alloc = SimpleNamespace(backend="sglang")
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._should_offload_critic = False
    trainer._should_offload_ref = False
    trainer._should_offload_teacher = False

    with pytest.raises(ValueError, match="FSDP actor with local SGLang"):
        trainer._validate_cfg()


def _make_rollout_init_trainer(
    *,
    backend: str,
    rollout_version: str = "v2",
    enable_mm_deepgemm: bool | None = None,
) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        rollout=InferenceEngineConfig(
            backend=backend,
            _version=rollout_version,
            admin_api_key="test-key",
        ),
        actor=SimpleNamespace(use_lora=False),
        gconfig=SimpleNamespace(lora_name="test-adapter"),
        sglang=SGLangConfig(
            model_path="test-model",
            enable_batch_invariant_ops_mm_deepgemm=enable_mm_deepgemm,
        ),
        vllm=vLLMConfig(model="test-model"),
        teacher=SimpleNamespace(path=""),
    )
    trainer.rollout_alloc = ModelAllocation.from_str(backend)
    trainer.scheduler = MagicMock(n_gpus_per_node=8)
    return trainer


def _assert_sglang_server_command(server_args, expected_value):
    cmd = SGLangConfig.build_cmd_from_args(server_args)
    if expected_value is None:
        assert cmd[:3] == [
            "python3",
            "-m",
            "areal.v2.inference_service.sglang.launch_server",
        ]
    else:
        assert cmd[:3] == [
            "env",
            f"{SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM}={expected_value}",
            "python3",
        ]


@pytest.mark.parametrize("rollout_version", ["v1", "v2"])
@pytest.mark.parametrize(
    ("setting", "expected_value"),
    [(None, None), (False, "0"), (True, "1")],
)
def test_sglang_rollout_command_covers_v1_and_v2(
    rollout_version, setting, expected_value
):
    trainer = _make_rollout_init_trainer(
        backend="sglang:d1",
        rollout_version=rollout_version,
        enable_mm_deepgemm=setting,
    )
    controller = MagicMock()

    with (
        patch.object(rl_trainer, "is_single_controller", return_value=True),
        patch.object(
            rl_trainer,
            "RolloutControllerV2",
            return_value=controller,
        ),
        patch.object(
            rl_trainer.RemoteSGLangEngine,
            "as_controller",
            return_value=controller,
        ),
        patch(
            "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
            return_value=True,
        ),
    ):
        result = trainer._init_rollout(trainer.config.rollout)

    assert result is controller
    initialize_kwargs = controller.initialize.call_args.kwargs
    assert "server_env" not in initialize_kwargs
    _assert_sglang_server_command(
        initialize_kwargs["server_args"],
        expected_value,
    )


@pytest.mark.parametrize("rollout_version", ["v1", "v2"])
@pytest.mark.parametrize(("setting", "expected_value"), [(False, "0"), (True, "1")])
def test_sglang_teacher_command_covers_v1_and_v2(
    rollout_version, setting, expected_value
):
    trainer = _make_rollout_init_trainer(
        backend="sglang:d1",
        rollout_version=rollout_version,
        enable_mm_deepgemm=setting,
    )
    trainer.teacher_alloc = ModelAllocation.from_str("sglang:d1")
    trainer.actor_alloc = ModelAllocation.from_str("fsdp:d1")
    controller = MagicMock()
    teacher_config = InferenceEngineConfig(
        backend="sglang:d1",
        _version=rollout_version,
        admin_api_key="test-key",
    )

    with (
        patch.object(rl_trainer, "is_single_controller", return_value=True),
        patch.object(
            rl_trainer.RemoteSGLangEngine,
            "as_controller",
            return_value=controller,
        ),
        patch(
            "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
            return_value=True,
        ),
    ):
        result = trainer._init_teacher_rollout(teacher_config)

    assert result is controller
    initialize_kwargs = controller.initialize.call_args.kwargs
    assert "server_env" not in initialize_kwargs
    _assert_sglang_server_command(
        initialize_kwargs["server_args"],
        expected_value,
    )


def test_v2_vllm_rollout_does_not_pass_sglang_server_env():
    trainer = _make_rollout_init_trainer(
        backend="vllm:d1",
        enable_mm_deepgemm=False,
    )
    controller = MagicMock()

    with (
        patch.object(rl_trainer, "is_single_controller", return_value=True),
        patch.object(
            rl_trainer,
            "RolloutControllerV2",
            return_value=controller,
        ),
    ):
        result = trainer._init_rollout(trainer.config.rollout)

    assert result is controller
    assert "server_env" not in controller.initialize.call_args.kwargs
