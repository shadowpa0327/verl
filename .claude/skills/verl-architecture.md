---
name: verl-architecture
description: Explains verl's 3-layer architecture, engine/rollout registries, worker hierarchy, dispatch/collect pattern, GPU time-multiplexing, WorkerGroup internals, and the current RayPPOTrainer launch + training flow (including teacher log_prob, AgentLoopManager, RewardLoopManager, CheckpointEngineManager, and RolloutReplica modes). Use when onboarding, exploring the codebase, understanding how components connect, or answering QA on verl's design.
user-invocable: true
---

# verl Architecture Reference

> Scope: official `RayPPOTrainer` paths on the current `main`-aligned tree. The in-development EAGLE drafter co-training sub-pipeline (`ActorRolloutRefDrafterWorker`, `HSCollectorManager`, `FSDPDrafterEngine`) is **out of scope** here — see `claude_docs/project-guide.md`.

## 3-Layer Composition

verl uses a 3-layer composition model where each layer adds RL-specific orchestration on top of pure computation:

```
Layer 3: ActorRolloutRefWorker    — RL-aware, composes training workers + rollout
Layer 2: TrainingWorker           — Batch orchestration, metrics, loss injection
Layer 1: BaseEngine / BaseRollout — Pure computation (two separate hierarchies)
```

Orthogonal to Layer 3, the trainer now owns three Ray-actor-level coordinators: `AgentLoopManager`, `RewardLoopManager`, `CheckpointEngineManager` — see [Launch Flow](#actorrolloutref-launch-flow).

### Layer 1a: BaseEngine (`verl/workers/engine/base.py:29-227`)

Abstract base class for gradient computation: forward/backward/optimizer.

**Key interface methods:**
- `initialize()` (line 37) — Load model, optimizer, scheduler
- `is_param_offload_enabled` / `is_optimizer_offload_enabled` (47, 53) — properties
- `train_mode()` / `eval_mode()` (57, 67) — Context managers for mode switching
- `forward_backward_batch(data, loss_function, forward_only)` (98) — Core fwd/bwd pass
- `train_batch(data, loss_function)` (112) — Complete training step
- `infer_batch(data, loss_function=None)` (133) — Inference-only step
- `get_per_tensor_param(layered_summon=..., base_sync_done=...) -> tuple[Generator[(name, tensor)], Optional[dict]]` (150) — **Returns a tuple**; second element is `peft_config` for LoRA weight sync.
- `to(device, model, optimizer, grad)` (169) — Selective device movement
- `save_checkpoint` / `load_checkpoint` (182, 202)
- `is_mp_src_rank_with_outputs()` (216)
- `disable_adapter() -> ContextManager` (222) — `nullcontext` by default (LoRA "eval-as-base" hook)

**Implementations** (`verl/workers/engine/`):
```
BaseEngine
├── FSDPEngine          (engine/fsdp/transformer_impl.py)     — PyTorch FSDP2
├── MegatronEngine      (engine/megatron/transformer_impl.py) — Megatron parallelism
├── VeOmniEngine        (engine/veomni/transformer_impl.py)   — Custom parallel backend
└── MindspeedEngine     (engine/mindspeed/transformer_impl.py) — NPU acceleration
```

Selected via **EngineRegistry** (`base.py:264-336`):
```python
# Registration: key = (model_type, backend, device)
EngineRegistry.register("language_model", "fsdp", "cuda")

# Usage in TrainingWorker:
self.engine = EngineRegistry.new(
    model_type=config.model_type,
    backend=engine_config.strategy,  # "fsdp", "megatron", etc.
    ...
)
```

### Layer 1b: BaseRollout (`verl/workers/rollout/base.py:29-105`)

Abstract base class for autoregressive token generation via inference servers.

**Key interface methods:**
- `async resume(tags: list[str])` — Resume weights and/or KV cache in GPU memory
- `async update_weights(weights, peft_config=None, base_sync_done=True, global_steps=None)` — Push new weights from training engine
- `async release()` — Release GPU memory
- `generate_sequences(prompts: DataProto)` — Synchronous batch generation

**Registry** (`base.py:83-105`) — async-only; sync mode is deprecated:
```
("vllm", "async")       → verl.workers.rollout.vllm_rollout.ServerAdapter
("vllm_omni", "async")  → verl.workers.rollout.vllm_rollout.ServerAdapter
("sglang", "async")     → verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter
("trtllm", "async")     → verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter
```

`hf_rollout` and `naive_rollout` are **no longer in the registry**. Separate replica instantiation goes through [`RolloutReplicaRegistry`](#rolloutreplica--server-mode).

### Layer 2: TrainingWorker (`verl/workers/engine_workers.py:74-433`)

Wraps one `BaseEngine`. Adds dispatch/collect decorators, micro-batching, loss injection, and metrics aggregation.

**Public (dispatched) methods:**
| Method | Line | Dispatch |
|---|---|---|
| `to` | 156 | ONE_TO_ALL |
| `set_loss_fn` | 166 | ONE_TO_ALL |
| `reset` | 170 | ONE_TO_ALL |
| `train_mini_batch` | 236 | ND mesh (train), blocking=False |
| `train_batch` | 326 | ND mesh (train), blocking=False |
| `infer_batch` | 381 | ND mesh (train), blocking=False |
| `save_checkpoint` / `load_checkpoint` | 427 / 431 | ONE_TO_ALL |

Each dispatched method automatically chunks `DataProto` across DP ranks.

### Layer 3: ActorRolloutRefWorker (`verl/workers/engine_workers.py:435-733`)

The "hybrid worker" — composes multiple `TrainingWorker`s + `BaseRollout`:

```python
self.actor:   TrainingWorker  → BaseEngine   # policy gradient
self.ref:     TrainingWorker  → BaseEngine   # frozen reference (optional)
self.rollout: BaseRollout                    # token generation server
```

**Role modes**: `"actor"` | `"rollout"` | `"ref"` | `"actor_rollout"` | `"actor_rollout_ref"`

**Public (dispatched) methods:**
| Method | Line | Dispatch |
|---|---|---|
| `set_loss_fn` | 484 | ONE_TO_ALL |
| `to` | 488 | ONE_TO_ALL |
| `init_model` | 493 | ONE_TO_ALL |
| `compute_ref_log_prob` | 627 | ND mesh (`ref`) |
| `compute_log_prob` | 634 | ND mesh (`actor`) |
| `update_actor` | 642 | ND mesh (`actor`) |
| `load_checkpoint` / `save_checkpoint` | 647 / 652 | ONE_TO_ALL |
| `update_weights` | 657 | ONE_TO_ALL, async (blocking=False) |
| `execute_checkpoint_engine` | 723 | DP_COMPUTE, blocking=False |

> **Not on this worker:** `generate_sequences` has moved to `AgentLoopManager` (see [Launch Flow](#actorrolloutref-launch-flow)). The trainer calls `self.async_rollout_manager.generate_sequences(...)`, not the worker group.

---

## Dispatch/Collect Pattern

The **Single Controller** pattern means `RayPPOTrainer` on the driver process controls all Ray actors via RPC. Data distribution is handled transparently by decorators.

**How it works** (`verl/single_controller/base/decorator.py`):

```
Driver calls:  actor_rollout_wg.compute_log_prob(batch)  [size 128, DP=4]
  → Dispatch:  chunk batch into 4 parts of size 32
  → Execute:   4 workers compute in parallel
  → Collect:   concatenate 4 outputs back to size 128
  → Return:    unified output to driver
```

**Predefined dispatch modes** (`decorator.py:37-46`):
- `RANK_ZERO` — Only rank 0 executes
- `ONE_TO_ALL` — Broadcast to all workers (e.g., `to("cpu")`, `init_model`)
- `ALL_TO_ALL` — Pairwise
- `DP_COMPUTE` — Per-worker args, no chunking
- `DP_COMPUTE_PROTO` — Auto-chunk `DataProto` across DP ranks, auto-concat results
- `DP_COMPUTE_PROTO_WITH_FUNC` — Same, with a function arg
- `DP_COMPUTE_METRIC` — Metric-aware variant
- `DIRECT_ROLLOUT_METHOD` — Special path for vLLM `ExternalRayDistributedExecutor`

**Named-mesh dispatch**: `make_nd_compute_dataproto_dispatch_fn(mesh_name)` at `decorator.py:299-303` returns `{"dispatch_fn", "collect_fn"}` bound to a device-mesh name (e.g., `"actor"`, `"ref"`, `"train"`).

### DataProto quick reference (`verl/protocol.py`)

Methods used in the RL loop:

| Method | Line | Purpose |
|---|---|---|
| `chunk(chunks)` | 863 | Split batch across workers |
| `concat(list)` | 916 | Merge worker outputs |
| `select(batch_keys, non_tensor_batch_keys, meta_info_keys)` | 599 | Filter keys |
| `pop(batch_keys, ...)` | 720 | Extract & remove |
| `union(other)` | 780 | Merge batch + meta_info |
| `slice(start, end, step)` | 674 | Sub-range |
| `select_idxs(idxs)` | 634 | Index-select |
| `repeat(repeat_times, interleave)` | 970 | Duplicate (used for `rollout.n`) |
| `meta_info` | 327 | Dict passed through every op |

---

## GPU Time-Multiplexing (Weight Sync)

Training engine and rollout engine share the same GPU but never run simultaneously. The `update_weights()` method (`engine_workers.py:656-720`) bridges them.

### Disaggregated (async) path
If `config.rollout.checkpoint_engine.backend != "naive"` (standalone server):
```python
per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
await self.checkpoint_engine.send_weights(per_tensor_param)
return   # no local rollout update; the CheckpointEngineManager handles the transfer
```

### Colocated (sync) path

```
Timeline:
 ┌──────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐
 │ Training │ → │ Weight Sync      │ → │ Rollout  │ → │ Training │
 │ (GPU)    │   │                  │   │ (GPU)    │   │ (GPU)    │
 │          │   │ 1. resume        │   │ generate │   │ Actor    │
 │ Actor    │   │    weights       │   │ sequence │   │ update   │
 │ update   │   │ 2. get_per_      │   │          │   │          │
 │          │   │    tensor_param  │   │          │   │          │
 │          │   │ 2a.(LoRA) base   │   │          │   │          │
 │          │   │    sync first    │   │          │   │          │
 │          │   │ 3. update_       │   │          │   │          │
 │          │   │    weights       │   │          │   │          │
 │          │   │ 4. offload       │   │          │   │          │
 │          │   │ 5. resume        │   │          │   │          │
 │          │   │    kv_cache      │   │          │   │          │
 └──────────┘   └──────────────────┘   └──────────┘   └──────────┘
```

Step-by-step (with line anchors):
1. `rollout.resume(tags=["weights"])` — allocate GPU for rollout model (681).
2. `per_tensor_param, peft_config = actor.engine.get_per_tensor_param(layered_summon=True, base_sync_done=True)` (685).
3. **LoRA branch** — if `not peft_merge and peft_config is not None and not base_sync_done`: do a separate base-weights pass first (`get_per_tensor_param(base_sync_done=False)` → `rollout.update_weights(..., base_sync_done=False)`), then the adapter pass (695-701).
4. `rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps)` (703).
5. `actor.engine.to("cpu", model=True, optimizer=False, grad=False)` if `is_param_offload_enabled` (710); `aggressive_empty_cache()`.
6. `rollout.resume(tags=["kv_cache"])` — allocate KV cache for inference (716).

---

## RolloutReplica — server mode

`verl/workers/rollout/replica.py:85-302` — the abstract server-instance lifecycle used by async rollouts.

**`RolloutMode` enum** (`replica.py:69-82`):
- **`HYBRID`** — rollout engine fused with trainer process; weight sync via pause/resume. On-policy training.
- **`COLOCATED`** — rollout in a separate process in the same Ray placement group; no weight sync needed (GRM/LLM-as-judge).
- **`STANDALONE`** — disaggregated rollout with its own GPU resource pool; weight sync via `CheckpointEngineManager` (off-policy training).

**Init entry points**: `init_hybrid` (144), `init_hybrid_colocated` (156), `init_colocated` (173), `init_standalone` (202). All end by calling `launch_servers()` (252).

**Server lifecycle**: `wake_up` (275), `sleep` (279), `abort_all_requests` (283), `resume_generation` (287), `clear_kv_cache` (291), `start_profile` (295), `stop_profile` (299).

**`RolloutReplicaRegistry`** (`replica.py:304-398`) is separate from `_ROLLOUT_REGISTRY`. Registered loaders: `vllm`, `sglang`, `trtllm`, `vllm_omni` (389-392). Lookup via `get_rollout_replica_class(rollout: str)` (396).

---

## Resource Pool Manager (`verl/single_controller/ray/base.py`)

Manages GPU allocation and colocation:

```python
ResourcePoolManager(
    resource_pool_spec={"pool0": [8]},   # 8 GPUs per node
    mapping={0: "pool0", 1: "pool0"},    # actor + critic share pool
)
```

- Each pool supports up to 3 colocated WorkerGroups (`max_colocate_count=3`)
- Colocated roles (actor, critic, ref) share GPUs in the same resource pool
- Uses Ray placement groups for resource isolation

---

## WorkerGroup Internals

### Base Classes

- **`WorkerGroup`** (`verl/single_controller/base/worker_group.py`) — Abstract base managing a list of distributed workers. Provides method binding, aliveness checking, and execution semantics.
- **`Worker`** (`verl/single_controller/base/worker.py`) — Base for individual workers; each gets rank/world_size/master_addr/master_port from env vars.
- **`RayWorkerGroup`** (`verl/single_controller/ray/base.py`) — Ray-specific implementation. Creates Ray actors via placement groups, binds methods with dispatch/collect semantics.

### Supporting Infrastructure

- **`RayResourcePool`** — wraps Ray placement groups, manages GPU allocation per node.
- **`SubRayResourcePool`** — a subset of bundles from a parent pool (for multi-model training).
- **`ResourcePoolManager`** — maps roles to resource pools, manages colocation.
- **`RayClassWithInitArgs`** — wraps a `ray.remote` class with its constructor arguments, scheduling strategy, and resource requirements.

### Method Binding and Execution

Worker methods decorated with `@register(dispatch_mode=..., execute_mode=...)` are auto-bound to the WorkerGroup:

```
User calls: worker_group.method_name(*args, **kwargs)
  → Functor.__call__()
  → dispatch_fn(wg, *args, **kwargs)      # split/broadcast data
  → execute_fn(method_name, *dispatched)   # invoke on Ray actors
  → (if blocking) ray.get(output)          # wait for results
  → collect_fn(wg, output)                 # aggregate results
```

| Execute method | Async | Target |
|---|---|---|
| `execute_all_sync` | No | All workers |
| `execute_all_async` | Yes | All workers |
| `execute_rank_zero_sync` | No | Rank 0 only |
| `execute_rank_zero_async` | Yes | Rank 0 only |

### Colocated Workers (Hybrid Engine)

`create_colocated_worker_cls()` creates a single Ray remote `WorkerDict` class that internally instantiates multiple component workers (actor, critic, ref, …) in the **same process**:

```python
class WorkerDict(Worker):
    def __init__(self):
        self.worker_dict = {}
        for key, cls in cls_dict.items():
            self.worker_dict[key] = cls(...)  # instantiate each component
```

Methods from all inner classes are monkey-patched onto `WorkerDict` with prefixed names. This enables:
- Shared GPU memory between actor and rollout
- Fast weight sync via NCCL (no network serialization)
- GPU time-multiplexing between training and inference

`wg_dict.spawn(prefix_set=...)` then creates per-role `RayWorkerGroup` views over the same underlying `WorkerDict` actors, each exposing only that role's methods.

---

## ActorRolloutRef Launch Flow

### End-to-End Sequence

```
main_ppo.py: main(config)
  → run_ppo(config)
    → ray.init()
    → TaskRunner.remote()
      → TaskRunner.run(config)
        ├── add_actor_rollout_worker(config)          # Step 1a
        ├── add_critic_worker(config)                 # Step 1b
        ├── add_reward_model_resource_pool(config)    # Step 1c (pool only, no worker group)
        ├── add_teacher_model_resource_pool(config)   # Step 1d (distillation)
        ├── add_ref_policy_worker(config, ...)        # Step 1e (legacy-impl only)
        ├── init_resource_pool_mgr(config)            # Step 2
        ├── RayPPOTrainer(...)                        # Step 3
        ├── trainer.init_workers()                    # Step 4
        └── trainer.fit()                             # Step 5
```

### Step 1: Worker Registration (`verl/trainer/main_ppo.py:109-298`)

- `add_actor_rollout_worker` (124) — picks `Role.ActorRolloutRef` vs `Role.ActorRollout`:
  ```python
  if need_reference_policy(config) and not ref_in_actor:
      role = Role.ActorRolloutRef          # fused ref inside actor worker
  else:
      role = Role.ActorRollout
  ```
  `ref_in_actor` is true when LoRA is on (adapter-disable provides the "ref" forward).
- `add_critic_worker` (178)
- `add_reward_model_resource_pool` (262) — **resource-pool mapping only**; no separate `RewardModelWorker` is spawned. Reward compute is routed through `RewardLoopManager` (see below).
- `add_teacher_model_resource_pool` (274) — distillation teacher pool.
- `add_ref_policy_worker` (286) — only for legacy impl; new impl folds ref into `ActorRolloutRefWorker`.

### Role enum (`verl/trainer/ppo/utils.py:27-41`)

```
Actor = 0, Rollout = 1, ActorRollout = 2, Critic = 3,
RefPolicy = 4, RewardModel = 5, ActorRolloutRef = 6, Env = 7,
TeacherModel = 8    # distillation teacher
```

### Role-selection helpers (`utils.py:75-106`)

```python
need_reference_policy(config) = config.algorithm.use_kl_in_reward
                              or config.actor_rollout_ref.actor.use_kl_loss
need_teacher_policy(config)   = is_distillation_enabled(config.get("distillation"))
need_reward_model(config)     = config.reward.reward_model.enable
need_critic(config)           = critic.enable (else True iff adv_estimator == GAE)
```

### Step 4: `init_workers()` (`verl/trainer/ppo/ray_trainer.py:700-889`)

```python
def init_workers(self):
    # 1. Create resource pools (Ray placement groups)
    self.resource_pool_manager.create_resource_pool()                # 707

    # 2. Register worker classes per resource pool
    actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in ... else Role.ActorRollout
    actor_rollout_cls = RayClassWithInitArgs(...)                    # 715
    # (critic, ref on legacy impl) — same pattern

    # 3. Create colocated WorkerDict and spawn per-role groups
    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict)    # 788
        wg_dict = self.ray_worker_group_cls(resource_pool, worker_dict_cls, ...)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())       # 794
        all_wg.update(spawn_wg)

    # 4. init_model on each spawned group
    self.actor_rollout_wg = all_wg[str(actor_role)]
    self.actor_rollout_wg.init_model()                               # 822

    # 5. Reward coordinator
    self.reward_loop_manager = RewardLoopManager(                    # 834-837
        config=self.config, rm_resource_pool=...)

    # 6. Teacher coordinator (if distillation on)
    if self.use_teacher_policy:
        self.teacher_model_manager = TeacherModelManager(            # 848-851
            config=self.config.distillation,
            resource_pool=self.resource_pool_manager.get_resource_pool(Role.TeacherModel),
        )

    # 7. Async rollout manager (owns generate_sequences)
    from verl.experimental.agent_loop import AgentLoopManager
    self.async_rollout_manager = AgentLoopManager.create(            # 874-880
        config=self.config,
        worker_group=self.actor_rollout_wg,
        rollout_resource_pool=actor_rollout_resource_pool,
        reward_loop_worker_handles=reward_loop_worker_handles,
        teacher_model_manager=self.teacher_model_manager,
    )

    # 8. Checkpoint/weight-sync coordinator (standalone-server path)
    self.checkpoint_manager = CheckpointEngineManager(               # 882-886
        config=checkpoint_engine_config,
        trainer=self.actor_rollout_wg,
        replicas=self.async_rollout_manager.rollout_replicas,
    )
    self.checkpoint_manager.sleep_replicas()                         # 889
```

#### The three new manager layers

| Manager | File | Role |
|---|---|---|
| **`AgentLoopManager`** | `verl/experimental/agent_loop/` | Wraps async rollout. **Owns `generate_sequences()`** (previously on `ActorRolloutRefWorker`). Holds `rollout_replicas`. Custom subclass supported via `config.actor_rollout_ref.rollout.agent.agent_loop_manager_class`. |
| **`RewardLoopManager`** | `verl/experimental/reward_loop/` | Colocated reward computation. Hosts reward-loop workers. Replaces the old separate `RewardModelWorker`. When enabled, can stream reward with rollout. |
| **`CheckpointEngineManager`** | coordinates `trainer` ↔ `rollout_replicas` | Drives `rollout_replicas.sleep()` / `wake_up()` and forwards weights for the disaggregated (standalone-server) path via `checkpoint_engine.send_weights()`. |

#### `init_model()` inside each worker (`engine_workers.py:493+`)

Inside each Ray actor, `init_model()` conditionally builds components based on role:

```python
if "ref" in self.role:
    self.ref = TrainingWorker(config=ref_training_config)
    self.ref.reset()  # → EngineRegistry.new() → FSDPEngine/MegatronEngine
if "actor" in self.role:
    self.actor = TrainingWorker(config=actor_training_config)
    self.actor.reset()
    self.actor.set_loss_fn(self.loss_fn)
if "rollout" in self.role:
    self.rollout = get_rollout_class(name, "async")(...)
```

---

## Training Loop — `RayPPOTrainer.fit()`

(`verl/trainer/ppo/ray_trainer.py:1284-1689`). Per-step ordering (in the order they run inside one `batch_dict` iteration):

| # | Stage | Call / anchor |
|---|---|---|
| 0 | Pre-loop: `val_before_train` | `_validate()` (1313) |
| 0 | Pre-loop: initial weight push | `checkpoint_manager.update_weights(0)` (1306) |
| 1 | **Generate** (async) | `async_rollout_manager.generate_sequences(gen_batch_output)` (1375), bracketed by `start_profile` / `stop_profile` and followed by `checkpoint_manager.sleep_replicas()` (1376). Optional `RolloutSkip` wrapper (1320-1322). |
| 1b | **REMAX baseline** (if `adv_estimator == REMAX`) | second `generate_sequences` pass with `do_sample=False`, then `_compute_reward_colocate` to produce `reward_baselines` (1383-1410). |
| 2 | **Teacher log_prob** (if distillation enabled, colocated) | `_compute_teacher_colocate(batch)` (1414) → `teacher_model_manager.compute_logprobs(batch)` (body at 516-522). |
| 3 | **Response mask** | `batch["response_mask"] = compute_response_mask(batch)` (1419-1420). |
| 4 | **Balance batch** (if `trainer.balance_batch`) | `_balance_batch(batch, ...)` (1425-1426). |
| 5 | **Reward** | inside `marked_timer("reward")` (1437): if `use_rm and "rm_scores" not in batch`, `_compute_reward_colocate(batch)` (1440); then `reward_tensor, reward_extra_infos_dict = extract_reward(batch)` (1444). |
| 6 | **Old log_prob + entropy** (unless `bypass_mode`) | `_compute_old_log_prob(batch)` (1462); inline entropy aggregation (1463-1476). Bypass mode uses `rollout_log_probs` directly (1452-1459). |
| 7 | **Ref log_prob** (if `use_reference_policy`) | `_compute_ref_log_prob(batch)` (1498). |
| 8 | **Values** (if `use_critic`) | `_compute_values(batch)` (1504). |
| 9 | **KL penalty** (if `use_kl_in_reward`) | `apply_kl_penalty(batch, kl_ctrl_in_reward, kl_penalty)` (1517). Else `token_level_rewards = token_level_scores` (1522). |
| 10 | **Rollout correction** (if `rollout_correction` configured, non-bypass) | `compute_rollout_correction_and_add_to_batch(batch, cfg)` (1535) — IS weights, rejection sampling, off-policy metrics. |
| 11 | **Advantage** | `compute_advantage(batch, adv_estimator, gamma, lam, ...)` (1544). |
| 12 | **Update critic** (if `use_critic`) | `_update_critic(batch)` (1557). |
| 12b | Critic warmup | if still warming up, only `checkpoint_manager.update_weights()` (1564), skip actor update. |
| 13 | **Update actor** | `_update_actor(batch)` (1568). |
| 14 | **Save checkpoint** (frequency + ESI) | `_save_checkpoint()` (1590). |
| 15 | **Weight sync** (actor → rollout) | `checkpoint_manager.update_weights(self.global_steps)` (1594). |
| 16 | **Validation** (periodic) | `_validate(merged=True)` (1609). |

Bracketing ops: `_start_profiling`, `marked_timer`, `async_rollout_manager.start_profile/stop_profile`, `checkpoint_manager.sleep_replicas()`.

### Advantage estimators (`verl/trainer/ppo/core_algos.py`)

`AdvantageEstimator` enum (88-110):
`gae`, `grpo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `rloo`, `opo`, `grpo_passk`, `gpg`, `rloo_vectorized`, `grpo_vectorized`, `optimal_token_baseline`, `tir_optimal_token_baseline`, `gdpo`.

Registered via `@register_adv_est` into `ADV_ESTIMATOR_REGISTRY`. Lookup: `get_adv_estimator_fn(name_or_enum)` (137).

### Policy loss registry (`core_algos.py:50`)

```python
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

@register_policy_loss("vanilla")                 # PPO  (1278)
@register_policy_loss("dppo_tv")   # (1372)
@register_policy_loss("dppo_kl")   # (1453)
@register_policy_loss("gspo")      # (1538)
@register_policy_loss("sapo")      # (1614)
@register_policy_loss("gpg")       # (1699)
@register_policy_loss("clip_cov")  # (1735)
@register_policy_loss("kl_cov")    # (1840)
@register_policy_loss("geo_mean")  # (1920)
@register_policy_loss("cispo")     # (2006)
```

Lookup: `get_policy_loss_fn(name)` (70). Additional losses (e.g., `flow_grpo` in `diffusion_algos.py`) register into the same dict.

### Teacher log_prob — distillation details

- **Trigger**: `self.use_teacher_policy and self._should_compute_teacher_colocate(batch)`; config at `config.distillation` (see `utils.need_teacher_policy`).
- **Pool registration**: `TaskRunner.add_teacher_model_resource_pool()` at `main_ppo.py:274-284`.
- **Manager**: `verl.experimental.teacher_loop.TeacherModelManager`, instantiated at `ray_trainer.py:848-851`.
- **Call site**: `RayPPOTrainer._compute_teacher_colocate(batch)` at `ray_trainer.py:516-522` → `teacher_model_manager.compute_logprobs(batch)`, invoked from `fit()` at 1416 and unioned back into `batch` (1417).
- **Stream mode**: `AgentLoopManager.create()` accepts `teacher_model_manager=...` (879) so that teacher-loop workers can sleep/wake alongside rollout workers for streaming distillation.

### Validation loop (`ray_trainer.py:524-613`, invoked at 1313 and 1609)

1. Iterate `val_dataloader`.
2. `async_rollout_manager.generate_sequences(...)`.
3. If no RM scores, `_compute_reward_colocate(test_batch)`.
4. `reward_tensor, reward_extra_info = extract_reward(test_batch)` (601) and aggregate metrics.

---

## Key Files Quick Reference

| File | Purpose |
|---|---|
| `verl/trainer/main_ppo.py` | Entry point, `TaskRunner`, worker/pool registration |
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer` — `init_workers()`, `fit()`, compute/update helpers |
| `verl/trainer/ppo/utils.py` | `Role` enum, `need_reference_policy`, `need_teacher_policy`, `need_reward_model`, `need_critic` |
| `verl/trainer/ppo/core_algos.py` | `AdvantageEstimator`, `ADV_ESTIMATOR_REGISTRY`, `POLICY_LOSS_REGISTRY`, KL controllers |
| `verl/workers/engine/base.py` | `BaseEngine`, `BaseEngineCtx`, `EngineRegistry` |
| `verl/workers/rollout/base.py` | `BaseRollout` + `_ROLLOUT_REGISTRY` (async-only ServerAdapters) |
| `verl/workers/rollout/replica.py` | `RolloutReplica`, `RolloutMode` (HYBRID/COLOCATED/STANDALONE), `RolloutReplicaRegistry` |
| `verl/workers/engine_workers.py` | `TrainingWorker` + `ActorRolloutRefWorker` (new impl) |
| `verl/workers/fsdp_workers.py` | Legacy FSDP-specific workers / `AsyncActorRolloutRefWorker` |
| `verl/experimental/agent_loop/` | `AgentLoopManager` (owns `generate_sequences`) |
| `verl/experimental/reward_loop/` | `RewardLoopManager` (colocated reward compute) |
| `verl/experimental/teacher_loop/` | `TeacherModelManager` (distillation teacher log_prob) |
| `verl/single_controller/base/worker_group.py` | Base `WorkerGroup` abstraction |
| `verl/single_controller/base/worker.py` | Base `Worker` class (rank, world_size) |
| `verl/single_controller/base/decorator.py` | `@register`, dispatch modes, `make_nd_compute_dataproto_dispatch_fn` |
| `verl/single_controller/ray/base.py` | `RayWorkerGroup`, `RayResourcePool`, `ResourcePoolManager`, `create_colocated_worker_cls` |
| `verl/protocol.py` | `DataProto` (chunk/concat/select/pop/union/repeat/meta_info) |

---

> For the in-development EAGLE drafter co-training sub-pipeline (`ActorRolloutRefDrafterWorker`, `HSCollectorManager`, `FSDPDrafterEngine`, Mooncake HS transfer, Eagle3 loss), see **`claude_docs/project-guide.md`**.
