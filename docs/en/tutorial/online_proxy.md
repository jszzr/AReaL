# V2 Online Proxy

The V2 inference gateway lets an external agent produce trajectories while an AReaL
trainer consumes them online. The important property is ownership: one trainer waiter
publishes one finite lease, one producer request consumes it, and the exported
loss-bearing tokens must come from the policy version captured by that lease.

```text
external agent -> Gateway -> Router -> Data Proxy -> SGLang/vLLM
                     |          |          |
                 global gate  worker pin  session + token provenance
                     |
                  Controller waiter -> export -> training
```

This page describes only the V2 API. V1 proxy clients and response shapes are not
interchangeable with it.

## Start a V2 training service

Use a configuration with both rollout and actor explicitly set to V2. The Hermes example
is a working reference:

```bash
uv run python3 examples/hermes/train.py \
  --config examples/hermes/config.yaml \
  actor.path=/path/to/your_model \
  rollout.admin_api_key=my-secret-admin-key \
  actor.admin_api_key=my-actor-key
```

The relevant configuration is:

```yaml
rollout:
  _version: v2
  backend: sglang:d1
  admin_api_key: my-secret-admin-key
  request_timeout: 300
  agent:
    mode: online
    export_style: individual

actor:
  _version: v2
```

Use `rollout.admin_api_key` for the inference gateway. The legacy
`rollout.agent.admin_api_key` field is not the V2 gateway credential.

After initialization, the trainer logs the gateway address.

## Callback-delivered online episode

### 1. Create a session

```bash
curl -X POST http://<gateway>/rl/start_session \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-admin-key' \
  -d '{
    "task_id": "gsm8k-17",
    "request_id": "run-42:gsm8k-17",
    "delivery_mode": "callback",
    "group_size": 1
  }'
```

`request_id` is a caller-generated idempotency key for one logical creation. Reuse it
unchanged after a timeout or lost response. An identical replay returns the original
credentials without consuming another lease; the same ID with different parameters
returns `409`.

The outcomes are:

- `201`: a session was created and bound to one trainer lease;
- `429`: no trainer waiter is available, and no session was created;
- `409`: an identity, worker epoch, or replay invariant was violated;
- `410`: the request ID is known, but its response retry horizon has elapsed;
- `503`: the bounded replay/ownership ledger cannot admit a new request ID;
- `422`: the request is invalid, for example callback mode without `request_id`.

Retry `429` with bounded backoff and the same request ID. Callback delivery currently
requires `group_size=1`.

A successful response is:

```json
{
  "group_id": "grp-2a61...",
  "sessions": [
    {
      "session_id": "gsm8k-17-grp-2a61...-0",
      "session_api_key": "opaque-token"
    }
  ]
}
```

Both IDs and keys are opaque. Do not parse their textual form or infer worker placement
from it.

### 2. Run the agent

Use the returned session key for OpenAI-compatible chat calls:

```bash
curl http://<gateway>/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer opaque-token' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 12 * 15 + 3?"}],
    "temperature": 0.7
  }'
```

The Data Proxy records the interaction, token IDs, log probabilities, loss mask, and
local policy version.

### 3. Set the reward

```bash
curl -X POST http://<gateway>/rl/set_reward \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer opaque-token' \
  -d '{"interaction_id": null, "reward": 1.0}'
```

When the reward boundary becomes ready, the Data Proxy sends a version-bound callback to
the exact controller waiter. The controller then exports the trajectory; the external
producer does not call an `end_session` endpoint.

For local SGLang/vLLM policies, AReaL verifies that every loss-bearing token has the
lease's expected policy version. A stale or mixed-version trajectory is rejected and its
remote tensor shards are cleared. External API providers do not expose this token
provenance, so external-mode interaction records are not proof of a provider model
revision.

## Pull delivery and explicit export

Use pull mode for controller-driven agents, grouped sessions, or a client that will
explicitly export:

```json
{
  "task_id": "manual-episode",
  "request_id": "run-42:manual-episode",
  "delivery_mode": "pull",
  "group_size": 2
}
```

Export is destructive, so it also requires a stable request ID:

```bash
curl -X POST http://<gateway>/export_trajectories \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-admin-key' \
  -d '{
    "request_id": "run-42:manual-export-0",
    "session_ids": ["opaque-session-id"],
    "group_id": "grp-2a61...",
    "trajectory_id": 0,
    "discount": 1.0,
    "style": "individual",
    "remove_session": true
  }'
```

If the response is lost, replay the identical body. The Gateway remembers the selected
worker and the Data Proxy replays the original serialized result instead of popping the
trajectory twice. Large successful exports are stored only at the Data Proxy; the
Gateway keeps a compact worker-incarnation marker and delegates an identical replay to
that proxy. Use a new ID for different parameters or a genuinely new export.

Group export is all-or-nothing. `session_ids` must be unique, every session must exist
and have the requested ready trajectory, and every session must belong to the same
`group_id`. A destructive group export must include every current member of that group,
so Router cleanup cannot orphan an omitted session. Validation happens before any
trajectory is consumed. Callback exports also carry the owning `lease_id`; the
controller adds it automatically.

Replay ledgers have an explicit in-process horizon. By default, response bytes are
retained for 300 seconds; afterward the request ID remains as a payload-free fence and
returns `410` instead of executing again. Pending requests, replayable responses, and
expired fences share a fixed record capacity; a new ID receives `503` when it is full.
Per-result and total response-byte limits prevent replay data from growing without a
bound. A Data Proxy returns `507` before consuming a trajectory if its serialized export
cannot fit its replay byte budget.

The relevant programmatic settings are `GatewayConfig.max_request_replay_records`,
`GatewayConfig.request_replay_ttl_seconds`,
`GatewayConfig.max_request_replay_result_bytes`,
`GatewayConfig.max_request_replay_total_bytes`, and the corresponding
`DataProxyConfig.max_export_replay_*` / `export_replay_ttl_seconds` fields.

## Failure and concurrency guarantees

- Admission is global across all Data Proxy workers behind one Gateway.
- New sessions are distributed independently; session keys remain pinned afterward.
- Session IDs include a globally unique group identity, and Router registration rejects
  conflicting ID/key ownership.
- Router registration is bound to a worker registration epoch, preventing a delayed
  response from reviving sessions after a process restarts at the same address.
- Data Proxy health replies include that immutable worker ID. Router health probes
  accept a `200` only when the returned ID matches the registered epoch. A new worker
  remains unroutable until that check succeeds; unpinned and new-session routes select
  only healthy workers.
- The inference CLI reads the admin-only `/worker_epoch` snapshot once after a new proxy
  becomes healthy and uses it as the registration CAS predecessor. A `409` fails the
  launch instead of rereading and overwriting a concurrent successor.
- CLI model state records `REGISTERING`, `ACTIVE`, or `CLEANUP_PENDING`. If a Router or
  Gateway success response is lost and exact cleanup cannot be proven, the CLI retains
  both the local processes and their worker IDs so `areal inf deregister` can safely
  retry cleanup.
- An unclaimed lease expires after the controller-owned timeout. Once callback export
  begins, the lease enters a delivered phase whose deadline covers the bounded Gateway
  forward; a successful destructive export completes it. Cleanup independently retries
  worker cancellation and Router revocation.
- Start/export request IDs are never silently evicted within one process. Response
  payloads compact to `410` fences, and capacity exhaustion backpressures new IDs.

## Fixed Held-Out Evaluation (V2 Only)

Online training data is driven by external applications, but evaluation should use a
fixed dataset that is never admitted to the PPO training FIFO. Pass that dataset to
`PPOTrainer` and provide a separate inline evaluation agent:

This integrity-checked path requires both the actor and rollout V2 controllers:

```yaml
actor:
  _version: v2
rollout:
  _version: v2
  agent:
    mode: online
```

The validation dataset must contain at least one batch. AReaL fails before starting the
online proxy when V2 is not enabled or the configured validation dataloader is empty.

```python
valid_dataset = get_custom_dataset(
    split="test",
    dataset_config=config.valid_dataset,
)

with PPOTrainer(
    config,
    train_dataset=None,
    valid_dataset=valid_dataset,
) as trainer:
    trainer.train(
        workflow=None,
        eval_workflow=MyEvaluationAgent,
        eval_workflow_kwargs={"temperature": 0.0},
    )
```

The main online controller continues to receive externally completed trajectories by
callback. Evaluation uses a separate controller with its own Router, Data Proxy,
Gateway, and session state; only the inference servers are shared. Evaluation sessions
are explicitly pulled by the evaluation workflow, so their trajectories cannot satisfy
an online training waiter.

Evaluation metrics are version checked. When a held-out task is submitted, AReaL
captures the controller's current policy version. A reward is recorded only if every
generated token selected by `loss_mask` carries that exact version in the exported
trajectory. Missing, mixed, or stale token versions reject the trajectory, and any
rejected item aborts the complete online evaluation round instead of publishing a metric
over a smaller, selected subset.

> **Version-0 baseline:** `eval_before_train` schedules the first evaluation when the
> PPO loop first reaches its evaluation point, which is after the first optimizer and
> weight-update step. If an experiment needs an untrained version-0 baseline, run a
> separate frozen evaluation arm with the same validation split, decoding settings, and
> reward function before starting online training.

These are orchestration guarantees on a trusted control plane. The internal lease API
uses the same admin credential, so the gate is not a security boundary against a
malicious holder of `rollout.admin_api_key`.

Replay and ownership ledgers are currently in memory. A Gateway or Data Proxy process
restart loses their records, so the protocol does **not** yet provide exactly-once
destructive export across process restarts. Production deployments that need that
guarantee must add a durable shared ledger (or keep the services alive and reconcile the
job externally) before retrying an ambiguous request after a restart.

## FAQ

## Health

`GET /health` returns Gateway state, not a worker count:

```json
{
  "status": "ok",
  "router_addr": "http://127.0.0.1:8081",
  "available_online_leases": 1
}
```

The lease count is useful for observation only. Producers should use the atomic
`start_session` response (`201` or `429`) rather than polling health as a reservation.
