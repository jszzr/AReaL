# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from areal.utils import seeding


@pytest.mark.parametrize("base_seed", [20260703, 2**32 - 1])
def test_set_random_seed_uses_stable_uint32_derivation(monkeypatch, base_seed):
    key = "actor0"
    expected = (base_seed + seeding._seed_from_key(key)) & 0xFFFFFFFF
    monkeypatch.setenv("PYTHONHASHSEED", "original")

    with (
        patch.object(seeding.transformers, "set_seed") as transformers_seed,
        patch.object(seeding.random, "seed") as python_seed,
        patch.object(seeding.np.random, "seed") as numpy_seed,
        patch.object(seeding.torch, "manual_seed") as torch_seed,
    ):
        seeding.set_random_seed(base_seed, key)

    assert seeding.get_seed() == expected
    assert seeding.os.environ["PYTHONHASHSEED"] == str(expected)
    transformers_seed.assert_called_once_with(expected)
    python_seed.assert_called_once_with(expected)
    numpy_seed.assert_called_once_with(expected)
    torch_seed.assert_called_once_with(expected)


@pytest.mark.parametrize("invalid_seed", [-1, 2**32, True])
def test_set_random_seed_rejects_values_outside_uint32(invalid_seed):
    with pytest.raises(ValueError, match="base_seed.*unsigned 32-bit"):
        seeding.set_random_seed(invalid_seed, "actor0")


@pytest.mark.parametrize(
    ("configured_seed", "expected_seed"),
    [(20260703, 20260703), (None, 42)],
)
def test_megatron_model_parallel_rng_uses_worker_seed_before_process_group(
    configured_seed, expected_seed
):
    pytest.importorskip("mbridge")
    from areal.engine.megatron_engine import MegatronEngine

    engine = MegatronEngine.__new__(MegatronEngine)
    get_seed_patch = (
        patch(
            "areal.engine.megatron_engine.get_seed",
            side_effect=ValueError("seed is unset"),
        )
        if configured_seed is None
        else patch(
            "areal.engine.megatron_engine.get_seed", return_value=configured_seed
        )
    )
    with (
        get_seed_patch,
        patch(
            "areal.engine.megatron_engine.tensor_parallel."
            "model_parallel_cuda_manual_seed"
        ) as manual_seed,
    ):
        engine._seed_model_parallel_rng()

    assert engine.seed == expected_seed
    manual_seed.assert_called_once_with(expected_seed)
