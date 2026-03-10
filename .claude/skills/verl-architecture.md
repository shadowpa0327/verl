---
name: verl-architecture
description: Explains verl's 3-layer architecture, engine/rollout registries, worker hierarchy, dispatch/collect pattern, GPU time-multiplexing, WorkerGroup internals, and ActorRolloutRef launch flow. Use when onboarding, exploring the codebase, understanding how components connect, or asking about verl's design.
user-invocable: true
---

# verl Architecture Reference

## 3-Layer Composition

verl uses a 3-layer composition model where each layer adds RL-specific orchestration on top of pure computation:

```
Layer 3: ActorRolloutRefWorker    — RL-aware, composes workers + rollout
Layer 2: TrainingWorker           — Batch orchestration, metrics, loss injection
Layer 1: BaseEngine / BaseRollout — Pure computation (two separate hierarchies)
```

### Layer 1a: BaseEngine (`verl/workers/engine/base.py:29-227`)

Abstract base class for gradient computation: forward/backward/optimizer.

**Key interface methods:**
- `initialize()` — Load model, optimizer, scheduler
- `train_mode()` / `eval_mode()` — Context managers for mode switching
- `forward_backward_batch(data, loss_function, forward_only)` — Core forward/backward pass
- `train_batch(data, loss_function)` — Complete training step
- `infer_batch(data, loss_function=None)` — Inference-only step
- `get_per_tensor_param()` — Generator yielding `(name, tensor)` for weight sync
- `to(device, model, optimizer, grad)` — Selective device movement

**Implementations** (`verl/workers/engine/`):
```
BaseEngine
├── FSDPEngine          (engine/fsdp/transformer_impl.py:79)     — PyTorch FSDP2
├── MegatronEngine      (engine/megatron/transformer_impl.py:65) — Megatron parallelism
├── VeOmniEngine        (engine/veomni/transformer_impl.py)      — Custom parallel backend
└── MindspeedEngine     (engine/mindspeed/transformer_impl.py)   — NPU acceleration
```

Selected via **EngineRegistry** (`base.py:266-336`):
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

### Layer 1b: BaseRollout (`verl/workers/rollout/base.py:29-102`)

Abstract base class for autoregressive token generation via inference servers.

**Key interface methods:**
- `async resume(tags: list[str])` — Resume weights or KV cache in GPU memory
- `async update_weights(weights: Generator)` — Push new weights from training engine
- `async release()` — Release GPU memory
- `generate_sequences(prompts: DataProto)` — Synchronous batch generation

**Implementations** (`verl/workers/rollout/`):
```
BaseRollout
├── SGLang ServerAdapter   (sglang_rollout/sglang_rollout.py)  — HTTP client
├── vLLM ServerAdapter     (vllm_rollout/vllm_rollout.py)      — Ray actor client
├── TRT-LLM ServerAdapter  (trtllm_rollout/trtllm_rollout.py)  — HTTP client
├── HFRollout              (hf_rollout.py)                      — Sync HuggingFace
└── NaiveRollout           (naive/naive_rollout.py)             — Single-GPU PyTorch
```

Selected via **_ROLLOUT_REGISTRY** (`base.py:81-102`):
```python
# Key = (engine_name, mode)
rollout_cls = get_rollout_class("sglang", "async")
self.rollout = rollout_cls(config=rollout_config, ...)
```

### Layer 2: TrainingWorker (`verl/workers/engine_workers.py:68-410`)

Wraps one BaseEngine. Adds dispatch/collect decorators, micro-batching, loss injection, and metrics aggregation.

**Key methods:**
- `train_mini_batch(data)` — Split batch into N mini-batches over PPO epochs
- `train_batch(data)` — Single training step with loss injection
- `infer_batch(data)` — Inference step in eval mode
- `set_loss_fn(loss_fn)` — Inject custom loss function
- `to(device, model, optimizer, grad)` — Device control

Each method is decorated with dispatch modes that automatically split data across DP ranks.

### Layer 3: ActorRolloutRefWorker (`verl/workers/engine_workers.py:412-689`)

The "hybrid worker" — composes multiple TrainingWorkers + BaseRollout:

```python
self.actor:   TrainingWorker  → BaseEngine   # policy gradient
self.ref:     TrainingWorker  → BaseEngine   # frozen reference (optional)
self.rollout: BaseRollout                    # token generation
```

**Role modes** (line 426): `"actor"` | `"rollout"` | `"ref"` | `"actor_rollout"` | `"actor_rollout_ref"`

**Key methods:**
- `compute_log_prob(data)` — Actor forward in eval mode
- `compute_ref_log_prob(data)` — Reference policy forward
- `update_actor(data)` — Actor gradient update via `actor.train_mini_batch()`
- `update_weights()` — Sync actor weights to rollout engine (see GPU time-multiplexing)
- `generate_sequences(data)` — Delegate to `rollout.generate_sequences()`

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

**Dispatch modes:**
- `Dispatch.ONE_TO_ALL` — Broadcast to all workers (e.g., `to("cpu")`)
- `Dispatch.DP_COMPUTE_PROTO` — Auto-chunk DataProto across DP ranks, auto-concat results
- `make_nd_compute_dataproto_dispatch_fn(mesh_name)` — Named mesh dispatch

---

## GPU Time-Multiplexing (Weight Sync)

Training engine and rollout engine share the same GPU but never run simultaneously. The `update_weights()` method (`engine_workers.py:616-676`) bridges them:

```
Timeline:
 ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
 │ Training │ → │ Weight     │ → │ Rollout  │ → │ Training │
 │ (GPU)    │   │ Sync       │   │ (GPU)    │   │ (GPU)    │
 │          │   │            │   │          │   │          │
 │ Actor    │   │ 1. resume  │   │ generate │   │ Actor    │
 │ update   │   │    weights │   │ sequence │   │ update   │
 │          │   │ 2. extract │   │          │   │          │
 │          │   │ 3. push    │   │          │   │          │
 │          │   │ 4. offload │   │          │   │          │
 │          │   │ 5. resume  │   │          │   │          │
 │          │   │    kv_cache│   │          │   │          │
 └──────────┘   └────────────┘   └──────────┘   └──────────┘
```

Detailed steps:
1. `rollout.resume(tags=["weights"])` — Allocate GPU for rollout model
2. `actor.engine.get_per_tensor_param()` — Extract trained weights
3. `rollout.update_weights(params)` — Push weights to inference server
4. `actor.engine.to("cpu")` — Offload training model to CPU
5. `rollout.resume(tags=["kv_cache"])` — Allocate KV cache for inference

---

## Resource Pool Manager (`verl/single_controller/ray/base.py:167-222`)

Manages GPU allocation and colocation:

```python
ResourcePoolManager(
    resource_pool_spec={"pool0": [8]},   # 8 GPUs per node
    mapping={0: "pool0", 1: "pool0"},    # actor + critic share pool
)
```

- Each pool supports up to 3 colocated WorkerGroups (`max_colocate_count=3`)
- Colocated roles (actor, critic, ref) share GPUs in same resource pool
- Uses Ray placement groups for resource isolation

---

## WorkerGroup Internals

### Base Classes

- **`WorkerGroup`** (`verl/single_controller/base/worker_group.py:123-256`) — Abstract base managing a list of distributed workers. Provides method binding, aliveness checking, and execution semantics.
- **`Worker`** (`verl/single_controller/base/worker.py:76-349`) — Base class for individual workers. Each gets rank, world_size, master_addr, master_port from env vars.
- **`RayWorkerGroup`** (`verl/single_controller/ray/base.py:411-904`) — Ray-specific implementation. Creates Ray actors via placement groups, binds methods with dispatch/collect semantics.

### Supporting Infrastructure

- **`RayResourcePool`** (`ray/base.py:112-161`) — Wraps Ray placement groups, manages GPU allocation per node.
- **`SubRayResourcePool`** (`ray/base.py:163-179`) — A subset of bundles from a parent pool (for multi-model training).
- **`ResourcePoolManager`** (`ray/base.py:181-236`) — Maps roles to resource pools, manages colocation.
- **`RayClassWithInitArgs`** (`ray/base.py:331-409`) — Wraps a `ray.remote` class with its constructor arguments, scheduling strategy, and resource requirements.

### RayWorkerGroup Initialization Paths

1. **New workers from resource pool** (`_init_with_resource_pool`, lines 531-574):
   - Creates placement groups → spawns Ray actors with env vars (WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT)
   - Each actor assigned to a bundle in the placement group
2. **Attach to detached workers** (`_init_with_detached_workers`, lines 504-511):
   - Attaches to existing named Ray actors (used by `from_detached()` / `spawn()`)
3. **Sub-resource pool** (`_init_with_subresource_pool`, lines 576-614):
   - Splits a resource pool into sub-pools, each managing a subset of bundles

### Method Binding and Execution

Worker methods decorated with `@register(dispatch_mode=..., execute_mode=...)` are auto-bound to the WorkerGroup:

```
User calls: worker_group.method_name(*args, **kwargs)
  → Functor.__call__()  (ray/base.py:48-66)
  → dispatch_fn(wg, *args, **kwargs)      # split/broadcast data
  → execute_fn(method_name, *dispatched)   # invoke on Ray actors
  → (if blocking) ray.get(output)          # wait for results
  → collect_fn(wg, output)                 # aggregate results
```

**Execution methods** (`ray/base.py:775-887`):
| Method | Async | Target |
|--------|-------|--------|
| `execute_all_sync()` | No | All workers |
| `execute_all_async()` | Yes | All workers |
| `execute_rank_zero_sync()` | No | Rank 0 only |
| `execute_rank_zero_async()` | Yes | Rank 0 only |

### Colocated Workers (Hybrid Engine)

**`create_colocated_worker_cls()`** (`ray/base.py:981-1022`) creates a single Ray remote `WorkerDict` class that internally instantiates multiple component workers (actor, critic, ref, etc.) in the **same process**:

```python
class WorkerDict(Worker):
    def __init__(self):
        self.worker_dict = {}
        for key, cls in cls_dict.items():
            self.worker_dict[key] = cls(...)  # instantiate each component
```

Methods from all inner classes are monkey-patched onto `WorkerDict` with prefixed names (`"{cls_name}_fwmn_{method_name}"`). This enables:
- Shared GPU memory between actor and rollout
- Fast weight sync via NCCL (no network serialization)
- GPU time-multiplexing between training and inference

**`spawn()`** (`ray/base.py:711-744`) then creates per-role `RayWorkerGroup` views over the same underlying `WorkerDict` actors, each exposing only that role's methods.

---

## ActorRolloutRef Launch Flow

### End-to-End Sequence

```
main_ppo.py: main(config)
  → run_ppo(config)
    → ray.init()
    → TaskRunner.remote()
      → TaskRunner.run(config)
        ├── add_actor_rollout_worker(config)     # Step 1: Register worker class
        ├── add_critic_worker(config)            # Step 2: Register critic
        ├── init_resource_pool_mgr(config)       # Step 3: GPU allocation
        ├── RayPPOTrainer(...)                   # Step 4: Create trainer
        ├── trainer.init_workers()               # Step 5: Spawn Ray actors
        └── trainer.fit()                        # Step 6: Training loop
```

### Step 1: Worker Registration (`main_ppo.py:123-149`)

```python
def add_actor_rollout_worker(self, config):
    from verl.workers.engine_workers import ActorRolloutRefWorker
    # Determine role based on whether reference policy is needed
    role = Role.ActorRolloutRef if need_reference_policy(config) else Role.ActorRollout
    self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
```

**Role enum** (`verl/trainer/ppo/utils.py:26-69`): `Actor=0, Rollout=1, ActorRollout=2, Critic=3, RefPolicy=4, RewardModel=5, ActorRolloutRef=6, Env=7`

### Step 5: Worker Spawning (`ray_trainer.py:674-849`)

```python
def init_workers(self):
    # 1. Create resource pool (Ray placement groups)
    self.resource_pool_manager.create_resource_pool()

    # 2. Wrap worker class with config
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[actor_role],
        config=self.config.actor_rollout_ref,
        role=str(actor_role),
    )

    # 3. Create colocated WorkerDict (actor + critic in same process)
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

    # 4. Create RayWorkerGroup → spawns N Ray actors across GPUs
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
    )

    # 5. Spawn per-role worker group views
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    # Result: {'actor_rollout_ref': RayWorkerGroup, 'critic': RayWorkerGroup}

    # 6. Initialize models on remote workers
    self.actor_rollout_wg = all_wg[str(actor_role)]
    self.actor_rollout_wg.init_model()  # → calls ActorRolloutRefWorker.init_model() on each actor
```

### Step 5.6: Model Init on Each Worker (`engine_workers.py:469-588`)

Inside each Ray actor, `init_model()` conditionally creates components based on role:

```python
def init_model(self):
    if "ref" in self.role:
        self.ref = TrainingWorker(config=ref_training_config)
        self.ref.reset()  # → EngineRegistry.new() → FSDPEngine/MegatronEngine
    if "actor" in self.role:
        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()
        self.actor.set_loss_fn(self.loss_fn)
    if "rollout" in self.role:
        rollout_cls = get_rollout_class(rollout_config.name, rollout_config.mode)
        self.rollout = rollout_cls(config=rollout_config, model_config=model_config)
```

### Training Loop (`ray_trainer.py: fit()`)

```python
for step in range(num_steps):
    # 1. Generate trajectories
    rollout_data = async_rollout_manager.generate_sequences(batch)
    # 2. Compute log probs (actor + ref)
    actor_output = actor_rollout_wg.compute_log_prob(rollout_data)
    ref_output = actor_rollout_wg.compute_ref_log_prob(rollout_data)  # or ref_policy_wg
    # 3. Compute values
    values = critic_wg.compute_values(rollout_data)
    # 4. Compute advantages (on driver)
    advantages = compute_advantage(values, rewards)
    # 5. Update actor
    actor_rollout_wg.update_actor(batch_with_advantages)
    # 6. Update critic
    critic_wg.update_critic(batch_with_advantages)
    # 7. Sync weights: actor → rollout (NCCL, in-place)
    actor_rollout_wg.update_weights(global_step)
```

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `verl/trainer/main_ppo.py` | Entry point, TaskRunner, worker registration |
| `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer orchestrator, init_workers(), fit() |
| `verl/trainer/ppo/utils.py` | Role enum, need_reference_policy() |
| `verl/workers/engine/base.py` | BaseEngine + EngineRegistry |
| `verl/workers/rollout/base.py` | BaseRollout + rollout registry |
| `verl/workers/engine_workers.py` | TrainingWorker + ActorRolloutRefWorker |
| `verl/workers/fsdp_workers.py` | Legacy FSDP-specific workers |
| `verl/workers/rollout/replica.py` | RolloutReplica (server lifecycle) |
| `verl/single_controller/base/worker_group.py` | Base WorkerGroup abstraction |
| `verl/single_controller/base/worker.py` | Base Worker class (rank, world_size) |
| `verl/single_controller/base/decorator.py` | @register, dispatch/collect modes |
| `verl/single_controller/ray/base.py` | RayWorkerGroup, RayResourcePool, ResourcePoolManager, create_colocated_worker_cls |
