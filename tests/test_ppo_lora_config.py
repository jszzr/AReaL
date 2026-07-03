# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    load_expr_config,
)

_MINIMAL_CONFIG = """
experiment_name: lora-validation
trial_name: trial0
actor:
  experiment_name: lora-validation
  trial_name: trial0
  backend: fsdp:d1
  _version: v2
  use_lora: false
rollout:
  backend: sglang:d1
  _version: v2
  use_lora: false
  lora_name: ""
gconfig:
  lora_name: canonical-adapter
train_dataset:
  path: dummy-dataset
  type: rl
saver:
  experiment_name: lora-validation
  trial_name: trial0
  fileroot: /tmp/areal
evaluator:
  experiment_name: lora-validation
  trial_name: trial0
  fileroot: /tmp/areal
recover:
  experiment_name: lora-validation
  trial_name: trial0
  fileroot: /tmp/areal
stats_logger:
  experiment_name: lora-validation
  trial_name: trial0
  fileroot: /tmp/areal
"""


def _ppo_config(
    *,
    actor_use_lora: bool = True,
    rollout_use_lora: bool = True,
    rollout_lora_name: str = "canonical-adapter",
    generation_lora_name: str = "canonical-adapter",
    eval_lora_name: str | None = None,
    rollout_backend: str = "sglang:d1",
    controller_version: str = "v2",
) -> PPOConfig:
    eval_gconfig = (
        None
        if eval_lora_name is None
        else GenerationHyperparameters(lora_name=eval_lora_name)
    )
    return PPOConfig(
        experiment_name="lora-validation",
        trial_name="trial0",
        actor=PPOActorConfig(
            backend="fsdp:d1",
            _version=controller_version,
            use_lora=actor_use_lora,
        ),
        rollout=InferenceEngineConfig(
            backend=rollout_backend,
            _version=controller_version,
            use_lora=rollout_use_lora,
            lora_name=rollout_lora_name,
        ),
        gconfig=GenerationHyperparameters(lora_name=generation_lora_name),
        eval_gconfig=eval_gconfig,
    )


@pytest.mark.parametrize(
    ("actor_use_lora", "rollout_use_lora"),
    [(True, False), (False, True)],
)
def test_ppo_config_rejects_actor_rollout_lora_flag_mismatch(
    actor_use_lora: bool,
    rollout_use_lora: bool,
):
    with pytest.raises(ValueError, match=r"actor\.use_lora.*rollout\.use_lora"):
        _ppo_config(
            actor_use_lora=actor_use_lora,
            rollout_use_lora=rollout_use_lora,
        )


def test_ppo_config_rejects_explicit_rollout_adapter_name_mismatch():
    with pytest.raises(ValueError, match=r"rollout\.lora_name.*gconfig\.lora_name"):
        _ppo_config(rollout_lora_name="wrong-adapter")


def test_ppo_config_rejects_empty_generation_adapter_name():
    with pytest.raises(ValueError, match=r"gconfig\.lora_name.*non-empty"):
        _ppo_config(rollout_lora_name="", generation_lora_name="")


def test_ppo_config_rejects_eval_adapter_name_mismatch():
    with pytest.raises(
        ValueError, match=r"eval_gconfig\.lora_name.*gconfig\.lora_name"
    ):
        _ppo_config(eval_lora_name="wrong-eval-adapter")


def test_ppo_config_rejects_vllm_lora_for_controller_v2():
    with pytest.raises(ValueError, match=r"vLLM.*LoRA.*controller v2"):
        _ppo_config(rollout_backend="vllm:d1")


def test_ppo_config_allows_sglang_lora_and_fills_rollout_name():
    config = _ppo_config(rollout_lora_name="")

    assert config.rollout.lora_name == "canonical-adapter"
    assert config.eval_gconfig is not None
    assert config.eval_gconfig.lora_name == "canonical-adapter"


def test_ppo_config_keeps_legacy_vllm_lora_available_for_controller_v1():
    config = _ppo_config(rollout_backend="vllm:d1", controller_version="v1")

    assert config.rollout.use_lora is True


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        (
            ["actor.use_lora=true", "rollout.use_lora=false"],
            r"actor\.use_lora.*rollout\.use_lora",
        ),
        (
            ["actor.use_lora=false", "rollout.use_lora=true"],
            r"actor\.use_lora.*rollout\.use_lora",
        ),
        (
            [
                "actor.use_lora=true",
                "rollout.use_lora=true",
                "rollout.lora_name=wrong-adapter",
            ],
            r"rollout\.lora_name.*gconfig\.lora_name",
        ),
        (
            [
                "actor.use_lora=true",
                "rollout.use_lora=true",
                "rollout.backend=vllm:d1",
            ],
            r"vLLM.*LoRA.*controller v2",
        ),
    ],
    ids=["actor-only", "rollout-only", "explicit-wrong-name", "v2-vllm"],
)
def test_load_expr_config_rejects_lora_mismatch_before_loader_side_effects(
    tmp_path: Path,
    overrides: list[str],
    error_match: str,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_CONFIG)

    with (
        patch("areal.api.cli_args.name_resolve.reconfigure") as mock_reconfigure,
        patch("areal.api.cli_args.save_config") as mock_save_config,
        pytest.raises(ValueError, match=error_match),
    ):
        load_expr_config(["--config", str(config_path), *overrides], PPOConfig)

    mock_reconfigure.assert_not_called()
    mock_save_config.assert_not_called()
