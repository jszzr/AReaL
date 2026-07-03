# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import random

import numpy as np
import torch
import transformers

from areal.infra.platforms import current_platform

_SEED = None
_BASE_SEED = None
_SHUFFLER = None


def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def validate_base_seed(base_seed: int) -> int:
    """Return a valid unsigned 32-bit experiment seed or raise ``ValueError``."""
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not 0 <= base_seed < 2**32
    ):
        raise ValueError(
            f"base_seed must be an unsigned 32-bit integer, got {base_seed!r}"
        )
    return base_seed


def derive_seed(base_seed: int, key: str) -> int:
    """Derive the stable uint32 RNG seed for one logical role/rank key."""
    return (validate_base_seed(base_seed) + _seed_from_key(key)) & 0xFFFFFFFF


def set_random_seed(base_seed: int, key: str) -> None:
    """Seed all supported RNGs from a stable uint32 ``base_seed`` and key.

    The effective seed is ``(base_seed + sha256(key).low32) mod 2**32``.
    Keeping this derivation here gives controllers and workers one auditable
    cross-process contract instead of backend-specific seed arithmetic.
    """
    global _SEED, _BASE_SEED
    base_seed = validate_base_seed(base_seed)
    _BASE_SEED = base_seed
    seed = derive_seed(base_seed, key)
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # FIXME: seeding initializes CUDA
    try:
        current_platform.manual_seed(seed)
    except AttributeError:
        pass


def get_seed() -> int:
    global _SEED
    if _SEED is None:
        raise ValueError("Random seed is not set. Please call set_random_seed first.")
    return _SEED


class Shuffler:
    def __init__(self, key="default"):
        self.cnt = 0
        self.base_key = key

    def next_shuffle(self) -> int:
        shuffle_key = f"{self.base_key}_{self.cnt}"
        self.cnt += 1
        return _seed_from_key(shuffle_key)


def get_shuffle_seed() -> int:
    global _BASE_SEED, _SHUFFLER
    if _SHUFFLER is None:
        _SHUFFLER = Shuffler(f"AReaL-seed{_BASE_SEED}")
    return _SHUFFLER.next_shuffle()
