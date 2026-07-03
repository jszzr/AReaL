# V2 LoRA Request Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every V2 internal-model generation request explicitly select the
versioned LoRA adapter loaded for its request policy version.

**Architecture:** Carry the existing `InferenceEngineConfig.use_lora` and `lora_name`
values through the controller's DataProxy command into `DataProxyConfig`. DataProxy
constructs `ArealOpenAI` with the configured adapter base name and `InfBridge` with an
explicit LoRA-enabled flag; `InfBridge` snapshots the request version and asks either
backend to emit the versioned adapter selector. Non-LoRA requests remain unmodified, and
LoRA configuration fails before service-launch side effects when the adapter name is
empty.

**Tech Stack:** Python 3.12, dataclasses, asyncio/httpx, pytest/pytest-asyncio, SGLang
`/generate`, vLLM OpenAI-compatible requests.

______________________________________________________________________

### Task 1: Fail Closed and Forward Controller Configuration

**Files:**

- Modify: `tests/v2/inference_service/test_controller.py`

- Modify: `areal/v2/inference_service/controller/controller.py`

- [ ] **Step 1: Write the failing constructor test**

```python
def test_lora_requires_non_empty_adapter_name_before_startup():
    cfg = InferenceEngineConfig(backend="sglang:d1", use_lora=True, lora_name="")
    scheduler = MagicMock(n_gpus_per_node=8)
    with pytest.raises(ValueError, match="lora_name"):
        RolloutControllerV2(config=cfg, scheduler=scheduler)
    scheduler.create_workers.assert_not_called()
```

- [ ] **Step 2: Write the failing DataProxy command test**

Extend the existing mocked `_async_initialize` test with `use_lora=True` and
`lora_name="online-gsm8k-lora"`, then assert the DataProxy command contains exactly
`--use-lora --lora-name online-gsm8k-lora`. Add a non-LoRA assertion that neither flag
is emitted.

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
pytest -q tests/v2/inference_service/test_controller.py -k lora
```

Expected: missing validation and missing command flags fail.

- [ ] **Step 4: Implement the minimum controller behavior**

Validate `config.lora_name.strip()` in `RolloutControllerV2.__init__` whenever
`config.use_lora` is true. Append the two LoRA CLI flags only for enabled LoRA.

- [ ] **Step 5: Run tests to verify GREEN**

Run the controller test file and confirm no non-LoRA command regression.

### Task 2: Configure DataProxy and Its Internal Client/Bridge

**Files:**

- Modify: `tests/v2/inference_service/test_data_proxy_standalone.py`

- Modify: `areal/v2/inference_service/data_proxy/config.py`

- Modify: `areal/v2/inference_service/data_proxy/__main__.py`

- Modify: `areal/v2/inference_service/data_proxy/app.py`

- [ ] **Step 1: Write failing configuration and constructor tests**

Test that `DataProxyConfig(use_lora=True, lora_name="")` raises, that
`_create_inf_bridge` receives `use_lora=True`, and that `_create_areal_client` passes
the configured name into `ArealOpenAI`. Also assert the non-LoRA path uses an empty name
and keeps `InfBridge.use_lora` false.

- [ ] **Step 2: Run tests to verify RED**

Run the new standalone tests and confirm missing fields/arguments cause the failures.

- [ ] **Step 3: Implement the minimum DataProxy wiring**

Add defaulted `use_lora` and `lora_name` fields plus `__post_init__` validation, parse
`--use-lora` and `--lora-name`, pass them into `DataProxyConfig`, then pass the flag to
`InfBridge` and the base name to `ArealOpenAI` only when enabled.

- [ ] **Step 4: Run tests to verify GREEN**

Run standalone, chat, and version endpoint tests.

### Task 3: Select and Bind Versioned Adapters in InfBridge

**Files:**

- Modify: `tests/v2/inference_service/test_inf_bridge.py`

- Modify: `areal/v2/inference_service/inf_bridge.py`

- [ ] **Step 1: Write failing SGLang and vLLM tests**

For each backend, construct `InfBridge(use_lora=True)`, send a request whose
`gconfig.lora_name` is `online-gsm8k-lora`, and capture the outgoing payload. Assert
SGLang sends `lora_path=online-gsm8k-lora-v0`; vLLM sends `model=online-gsm8k-lora-v0`.
After `set_version(1)`, a new request must select `-v1`. With `use_lora=False`, assert
neither selector is present even if the generation dataclass contains its legacy default
name.

- [ ] **Step 2: Write the failing abort/resubmit binding test**

Make the first backend response return `abort`, call `bridge.set_version(1)` inside the
first mocked send, and return `stop` on the second send. Assert both captured payloads
still select `-v0` and every output token remains stamped version 0.

- [ ] **Step 3: Run tests to verify RED**

Run only the new InfBridge tests and confirm `use_lora` is unsupported or selectors are
missing.

- [ ] **Step 4: Implement request-version binding**

Store `use_lora` on `InfBridge`. At `agenerate` entry, snapshot `_version`, pass the
snapshot plus `with_lora=self.use_lora` to the backend, and use the snapshot for LoRA
token version attribution throughout retries. Continue current per-attempt attribution
for full-model updates.

- [ ] **Step 5: Run tests to verify GREEN**

Run the entire InfBridge suite for both backends.

### Task 4: Verify and Commit

**Files:**

- Verify all files above.

- [ ] **Step 1: Run focused CPU tests**

Run controller, DataProxy standalone/chat/version, InfBridge, and online-stack tests.

- [ ] **Step 2: Run the broad V2 inference-service CPU suite**

Run `tests/v2/inference_service` excluding hardware-only tests only when their markers
require unavailable services; report every skip.

- [ ] **Step 3: Run pre-commit**

Activate the repository environment and run `pre-commit run --all-files` as required by
`AGENTS.md`; rerun affected tests after any formatter changes.

- [ ] **Step 4: Inspect the diff and commit**

Confirm only the isolated worktree changed, inspect `git diff --check` and the complete
diff, then create conventional logical commits without amend or push.
