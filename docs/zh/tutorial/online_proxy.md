# V2 在线代理

V2 推理网关允许外部智能体生成轨迹，同时由 AReaL 训练器在线消费。核心不是简单地“把请求转发到模型”，而是建立 明确的所有权：一个训练 waiter 发布一个有限期
lease，一次 producer 请求消费该 lease，最终参与 loss 的 token 必须来自 lease 捕获的策略版本。

```text
外部智能体 -> Gateway -> Router -> Data Proxy -> SGLang/vLLM
                 |          |          |
              全局准入门   worker 固定   会话与 token 来源
                 |
              Controller waiter -> 导出 -> 训练
```

本文只描述 V2 API；V1 的客户端和响应结构不能与之混用。

## 启动 V2 训练服务

rollout 和 actor 都必须显式设为 V2。Hermes 示例是一份可运行的参考：

```bash
uv run python3 examples/hermes/train.py \
  --config examples/hermes/config.yaml \
  actor.path=/path/to/your_model \
  rollout.admin_api_key=my-secret-admin-key \
  actor.admin_api_key=my-actor-key
```

关键配置如下：

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

推理网关使用 `rollout.admin_api_key`。旧的 `rollout.agent.admin_api_key` 不是 V2 网关凭据。

## callback 在线 episode

### 1. 创建会话

```bash
curl -X POST http://<gateway>/rl/start_session \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-admin-key' \
  -d '{
    "task_id": "gsm8k-17",
    "request_id": "run-42:gsm8k-17",
    "request_expires_at": 1783075200,
    "delivery_mode": "callback",
    "group_size": 1
  }'
```

`request_id` 是调用方生成的幂等键，`request_expires_at` 是有限的 Unix
时间重试截止点。请求超时或响应丢失后必须原样复用两者。相同请求会重放第一次 返回的凭据，不会再次消费 lease；同一个 ID 搭配不同参数会返回 `409`。

主要响应是：

- `201`：会话已创建，并绑定到一个训练 lease；
- `429`：当前没有训练 waiter，且没有创建会话；
- `409`：请求身份、worker epoch 或重放约束冲突；
- `410`：系统仍记得该 request ID，但响应重试窗口已经结束；
- `503`：有界重放/所有权账本已满，不能接收新的 request ID；
- `422`：请求无效，例如 callback 模式缺少 `request_id`。

收到 `429` 后使用有上限的退避和相同 request ID 重试。callback 目前要求 `group_size=1`。

成功响应示例：

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

ID 和密钥都是不透明值，不要解析其文本格式，也不要据此推断 worker 位置。

### 2. 运行智能体

使用返回的会话密钥调用 OpenAI 兼容接口：

```bash
curl http://<gateway>/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer opaque-token' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "12 * 15 + 3 等于多少？"}],
    "temperature": 0.7
  }'
```

Data Proxy 会记录交互、token ID、对数概率、loss mask 和本地策略版本。

### 3. 设置奖励

```bash
curl -X POST http://<gateway>/rl/set_reward \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer opaque-token' \
  -d '{"interaction_id": null, "reward": 1.0}'
```

奖励边界就绪后，Data Proxy 会向准确匹配的 controller waiter 发送带版本的 callback，再由 controller 导出轨迹。 外部
producer 不需要调用 `end_session` 端点。

对于本地 SGLang/vLLM 策略，AReaL 会验证每个参与 loss 的 token 是否具有 lease 所期望的策略版本。陈旧或混合 版本的轨迹会被拒绝，其远程
tensor shard 也会被清理。外部 API 不暴露这类 token provenance，因此 external mode 的交互记录不能证明 provider
的具体模型 revision。

## pull 交付与显式导出

controller 驱动的智能体、分组会话或自行导出的客户端应使用 pull：

```json
{
  "task_id": "manual-episode",
  "request_id": "run-42:manual-episode",
  "request_expires_at": 1783075200,
  "delivery_mode": "pull",
  "group_size": 2
}
```

导出会消费轨迹，因此也必须提供稳定的 request ID：

```bash
curl -X POST http://<gateway>/export_trajectories \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-admin-key' \
  -d '{
    "request_id": "run-42:manual-export-0",
    "request_expires_at": 1783075200,
    "session_ids": ["opaque-session-id"],
    "group_id": "grp-2a61...",
    "trajectory_id": 0,
    "discount": 1.0,
    "style": "individual",
    "remove_session": true
  }'
```

响应丢失后应原样重放请求。Gateway 会记住选中的 worker，Data Proxy 会重放第一次序列化的结果，而不是再次 弹出轨迹。对于较大的成功导出，完整
payload 只保存在 Data Proxy；Gateway 只保存带 worker incarnation 的紧凑 标记，并把相同重放交给该 Data
Proxy。只有参数变化或真正的新导出才使用新的 ID。

分组导出是 all-or-nothing：`session_ids` 必须唯一；每个 session 都必须存在、具有指定的 ready trajectory，并且
全部属于同一个 `group_id`。破坏性分组导出必须包含该组当前的所有成员，避免 Router 清理使未列出的 session 失去
路由。所有检查都在消费任何轨迹之前完成。callback 导出还会携带所属 `lease_id`，controller 会自动补上这个字段。

重放账本具有明确的进程内窗口。每个变更请求都携带有限的 `request_expires_at`，默认最多在未来 300 秒。超过截止点后请求本身恒返回
`410`，因此终态记录可以安全回收，旧 ID 仍不会静默重做。pending 请求和有效期内的可重放响应共用固定记录容量；容量满时，新 ID 返回
`503`。单条与总响应字节上限保证重放数据有界。如果序列化导出无法放入 Data Proxy 的 重放预算，Data Proxy 会在消费轨迹之前返回 `507`。

对应的程序化配置包括 `GatewayConfig.max_request_replay_records`、
`GatewayConfig.request_replay_ttl_seconds`、
`GatewayConfig.max_request_replay_result_bytes`、
`GatewayConfig.max_request_replay_total_bytes`，以及 Data Proxy 中对应的 `max_export_replay_*`
/ `export_replay_ttl_seconds` 字段。

## 失败与并发保证

- 准入容量在同一个 Gateway 后面的所有 Data Proxy worker 之间全局共享。
- 新会话独立分配；创建后，会话密钥固定路由到所属 worker。
- session ID 包含全局唯一的 group 身份，Router 会拒绝 ID 或密钥所有权冲突。
- Router 注册绑定 worker 的注册 epoch，避免进程在同一地址重启后被延迟响应复活旧会话。
- Data Proxy 的健康响应会返回不可变 worker ID；Router 只会把 ID 与已注册 epoch 相同的 `200` 视为健康。新 worker
  在通过该检查前不会参与路由；未固定和新会话路由只选择健康 worker。
- 新 proxy 健康后，推理 CLI 只读取一次管理员端点 `/worker_epoch`，并将其作为注册 CAS 的 predecessor。 如果返回
  `409`，本次启动直接失败，不会重读后覆盖并发 successor。
- CLI 模型状态会记录 `REGISTERING`、`ACTIVE` 或 `CLEANUP_PENDING`。如果 Router 或 Gateway 的成功响应丢失，且无法
  证明精确清理已经完成，CLI 会保留本地进程及其 worker ID，使 `areal inf deregister` 能够安全重试清理。
- 尚未领取的 lease 在 controller 持有的超时后过期。callback 导出开始后，lease 进入 delivered 阶段，其新期限 覆盖有界的
  Gateway 转发；破坏性导出成功后进入 completed。worker 取消和 Router 撤销会独立重试。
- start/export 在声明的截止点前可安全重放；之后终态记录可回收，而过期请求本身仍确定返回 `410`。

## 固定留出集评测（仅限 V2）

在线训练数据由外部应用驱动，但评测应使用固定数据集，并且这些样本绝不能进入 PPO 训练 FIFO。将该数据集传给
`PPOTrainer`，同时提供一个独立的进程内评测智能体：

这条带完整性校验的评测链路要求 actor 与 rollout 同时使用 V2 控制器：

```yaml
actor:
  _version: v2
rollout:
  _version: v2
  agent:
    mode: online
```

验证数据集必须至少包含一个 batch。未启用 V2 或已配置的验证 dataloader 为空时，AReaL 会在启动在线代理前直接失败。

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

在线训练主控制器仍通过 callback 接收外部完成的轨迹。评测使用单独的控制器，拥有独立的 Router、Data Proxy、Gateway
和会话状态；两者只共享推理服务器。评测工作流会主动 pull 自己的会话轨迹，因此评测样本不会满足在线训练的等待队列。

评测指标会经过策略版本校验。提交留出集任务时，AReaL 会捕获控制器当时的策略版本。只有导出轨迹中所有被 `loss_mask` 选中的生成 token
都携带这个精确版本时，reward 才会被记录。缺失、混合或过期的 token
版本都会使该轨迹被拒绝；只要有一个样本被拒绝，整个在线评测轮次都会失败，而不会在一个更小、经过选择的子集上发布指标。

> **版本 0 基线：** `eval_before_train` 会让 PPO 循环第一次到达评测点时执行评测，但这个评测点位于第一次优化器更新和权重更新之后。
> 如果实验需要未训练的版本 0 基线，应在在线训练开始前单独运行一个冻结评测组，并保持验证集、解码设置和奖励函数完全一致。

这些是可信控制平面上的编排保证。内部 lease API 与 producer 共用管理员凭据，所以它不是针对恶意
`rollout.admin_api_key` 持有者的安全边界。

## FAQ

目前 replay 与 ownership 账本只在内存中。Gateway 或 Data Proxy 进程重启会丢失这些记录，因此该协议尚不能 保证跨进程重启的破坏性导出
exactly-once。需要这一保证的生产部署，必须先接入持久化共享账本（或保持服务存活并 在外部完成作业对账），再对重启后的歧义请求进行重试。

## 健康检查

`GET /health` 返回网关状态，而不是 worker 数量：

```json
{
  "status": "ok",
  "router_addr": "http://127.0.0.1:8081",
  "available_online_leases": 1
}
```

lease 数量只适合观测。producer 应以原子的 `start_session` 响应（`201` 或 `429`）作为准入结果，而不是先轮询 health
再自行预留。
