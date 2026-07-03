# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from areal.utils.seeding import validate_base_seed


@dataclass
class TrainWorkerConfig:
    """Train-worker process settings.

    ``seed``, ``seed_role``, and ``seed_rank`` form an optional all-or-none
    cross-process contract. The worker passes them to
    ``set_random_seed(seed, f"{seed_role}{seed_rank}")`` before constructing
    the training engine.
    """

    host: str = "0.0.0.0"
    port: int = 0
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    seed: int | None = None
    seed_role: str | None = None
    seed_rank: int | None = None

    def __post_init__(self) -> None:
        seed_fields = (self.seed, self.seed_role, self.seed_rank)
        supplied = tuple(value is not None for value in seed_fields)
        if any(supplied) and not all(supplied):
            raise ValueError("seed, seed_role, and seed_rank must be provided together")
        if self.seed is None:
            return
        self.seed = validate_base_seed(self.seed)
        if not isinstance(self.seed_role, str) or not self.seed_role.strip():
            raise ValueError(
                f"seed_role must be a non-empty string, got {self.seed_role!r}"
            )
        if (
            isinstance(self.seed_rank, bool)
            or not isinstance(self.seed_rank, int)
            or self.seed_rank < 0
        ):
            raise ValueError(
                f"seed_rank must be a non-negative integer, got {self.seed_rank!r}"
            )
