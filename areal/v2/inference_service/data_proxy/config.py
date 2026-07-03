# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    backend_addr: str = "http://localhost:30000"  # co-located SGLang/vLLM
    backend_type: str = "sglang"
    use_lora: bool = False
    lora_name: str = ""
    tokenizer_path: str = ""
    log_level: str = "warning"
    request_timeout: float = 120.0  # seconds per SGLang call
    set_reward_finish_timeout: float = 0.0
    max_resubmit_retries: int = 20  # max abort/resubmit cycles before giving up
    resubmit_wait: float = 0.5  # seconds between is_paused polls
    admin_api_key: str = "areal-admin-key"  # admin key for authentication
    callback_server_addr: str = ""
    max_export_replay_records: int = 4096
    export_replay_ttl_seconds: float = 300.0
    max_export_replay_result_bytes: int = 16 * 1024 * 1024
    max_export_replay_total_bytes: int = 64 * 1024 * 1024
    # Resolved serving address (host:port) used as node_addr for RTensor shards.
    # Set at startup by __main__.py after the host is resolved.
    serving_addr: str = ""
    # Immutable process incarnation used to fence delayed control requests.
    worker_id: str | None = None

    # ArealOpenAI client parameters (forwarded from AgentConfig)
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    engine_max_tokens: int | None = None
    chat_template_type: str = "hf"

    def __post_init__(self) -> None:
        if self.use_lora and (
            not isinstance(self.lora_name, str) or not self.lora_name.strip()
        ):
            raise ValueError("lora_name must be set when use_lora=True")
