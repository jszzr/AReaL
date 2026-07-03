# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Inference Gateway service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    """Configuration for the inference gateway.

    The gateway only needs ``admin_api_key`` and ``router_addr`` —
    all worker state and session pinning live in the Router service.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    admin_api_key: str = "areal-admin-key"
    router_addr: str = "http://localhost:8081"
    router_timeout: float = 2.0  # seconds for /route call
    forward_timeout: float = 120.0  # seconds for forwarding to data proxy
    max_pending_request_owners: int = 4096
    max_pending_export_cleanups: int = 4096
    max_request_replay_records: int = 4096
    request_replay_ttl_seconds: float = 300.0
    max_request_replay_result_bytes: int = 16 * 1024 * 1024
    max_request_replay_total_bytes: int = 64 * 1024 * 1024
    log_level: str = "warning"
