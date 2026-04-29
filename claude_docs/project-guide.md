# verl — RL Training Framework

## Project Context
This is the verl codebase, an RL training framework (PPO/GRPO) being used as a migration target for FastRL (EAGLE co-training with speculative decoding).

**Active work:** EAGLE drafter co-training pipeline — porting TorchSpec's hidden states collection + Mooncake transport + Eagle3 training into verl's architecture. Branch: `feat/drafter-cotraining`.

## Drafter Co-Training (Active Migration)

### What We're Building

EAGLE drafter co-training: train a tiny draft model (~2% of target params) alongside the RL policy. The drafter learns to predict the target model's output distribution, enabling speculative decoding at inference time.

```
RL Step with Drafter Co-Training:

  generate_sequences()          → vLLM rollout
  collect_hidden_states()       → vLLM HS collector (prefill-only, KV connector → Mooncake)
  dispatch drafter data         → DrafterDataController on driver
  update_drafter()              → FSDPDrafterEngine (Eagle3 Forward KL loss)
  compute_log_prob()            → Actor engine
  update_actor()                → Actor engine
  update_weights()              → actor + drafter → rollout + HS collector
```

### Architecture

```
RayDrafterCTPPOTrainer (driver)
├── DrafterDataController              ← owns sample_pool
│     drain_as_dataproto() → mesh dispatch per DP rank
├── HSCollectorManager                 ← colocated vLLM replicas + KV connector
│     compute_hidden_states(batch) — wake → prefill → Mooncake → sleep
│
└── actor_rollout_wg
    └── ActorRolloutRefDrafterWorker
            ├── actor     (TrainingWorker → FSDPEngine)       [inherited]
            ├── ref       (TrainingWorker → FSDPEngine)       [inherited]
            ├── rollout   (BaseRollout → vLLM)                [inherited]
            └── drafter   (TrainingWorker → FSDPDrafterEngine) [NEW]
```

### Key Design Decisions

- **vLLM for HS collection** — uses `extract_hidden_states` speculative config + custom `MooncakeHiddenStatesConnector` (KV connector). Zero patching of vLLM internals.
- **Mooncake** as data plane — heavy tensors (hidden states) flow through Mooncake KV store. The 3-level pipeline only routes lightweight metadata (keys, shapes).
- **DrafterDataController on driver** — mirrors verl's single-controller pattern. Levels 1 & 2 are global; Level 3 dispatch via `DataProto.chunk()` + drafter mesh.
- **No unanimity gate** — the centralized controller + mesh dispatch guarantees all ranks get data or none do. Consensus by construction.
- **Pre-norm handling** — vLLM captures `last_hidden_states` before the final RMSNorm (pre-norm). `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` (a frozen copy of the actor's `model.norm`) to normalize before target distribution computation. Note: the loss kernel's built-in RMSNorm is for the *draft* model's norm, not the verifier — these are separate.

### Where to make changes — recipe submodule first

**All drafter-private code lives in the `recipe/drafter_cotraining/`
submodule** (a git submodule pointing at `shadowpa0327/verl-recipe`,
branch `feat/drafter-cotraining`). The in-tree `verl/...` mirrors that
existed during the migration have been **deleted** as of 2026-04-23
(see `migration-status.md` "cleanup" entry).

**Default rule for new work:**

> *Put changes in `recipe/drafter_cotraining/` whenever it can fit there.
> Touch `verl/...` (parent) only when you genuinely need to extend
> verl-side base infrastructure or when the parent change is so small
> that mirroring through the submodule would be more friction than
> value.*

Concrete guidance:

| Change kind | Where it goes |
|---|---|
| Drafter worker / engine logic | `recipe/drafter_cotraining/engine/{workers,drafter_engine}.py` |
| Eagle3 model / loss / draft | `recipe/drafter_cotraining/eagle3/...` |
| Mooncake transfer / KV connector | `recipe/drafter_cotraining/mooncake/...` |
| HS collector manager | `recipe/drafter_cotraining/hs_collector/...` |
| Drafter trainers / data controller / launchers | `recipe/drafter_cotraining/{main_drafter_ct,main_drafter_pretrain}.py` + `trainer/{ray_trainer,pretrain_trainer}.py` + `data/controller.py` |
| Test / smoke scripts | `recipe/drafter_cotraining/scripts/...` |
| Drafter unit tests | `recipe/drafter_cotraining/tests/...` |
| Drafter YAML config / draft model JSONs | `recipe/drafter_cotraining/config/...` |
| Top-level smoke wrappers | `scripts/run_drafter_*.sh` (parent — they shell out to recipe scripts) |
| Drafter→rollout weight sync (TODO 4) | parent `verl/workers/rollout/vllm_rollout/...` (parent must grow the receiving end) |
| New verl base-class hooks (FSDPEngine, RayPPOTrainer, dispatch modes) | parent `verl/...` |
| Design / status docs | parent `claude_docs/...` |

**Submodule workflow** for recipe changes:

```bash
cd recipe/drafter_cotraining
# edit, test, commit on feat/drafter-cotraining (or topic branch)
git push origin2 feat/drafter-cotraining

cd /root/verl                           # parent
git add recipe                          # stage gitlink bump
git commit -m "[drafter] bump recipe submodule: <topic>"
git push                                # parent stays on feat/drafter-cotraining (or your topic branch)
```

**When in doubt** — start in the recipe. If you find yourself needing
to subclass / monkey-patch a verl base class repeatedly, that's a
signal that a small parent-side hook would be the cleaner fix; promote
it to parent then.

### Key Documents

| Doc | Role | What |
|---|---|---|
| `claude_docs/migration-status.md` | **Current state** | What's done, what's TODO (ordered), frozen module sync, verification checklist |
| `claude_docs/drafter-design.md` | **As-built design** | Architecture, data lifecycle, training-step shape, engine-choice rationale |
| `claude_docs/torchspec-to-verl-migration-map.md` | **Reference** | TorchSpec internals + file-by-file connection map to verl |

### Testing

Smoke wrappers (live in parent — shell into the recipe-side tests):

```bash
# Rollout + HS collection + mesh dispatch (no drafter training)
./scripts/run_drafter_rollout_hs.sh

# Close-loop drafter training (rollout → HS → forward → backward → opt step)
./scripts/run_drafter_training.sh                    # MAX_STEPS=32 default
MAX_STEPS=16 ./scripts/run_drafter_training.sh       # quick smoke
```

Recipe-side test/diag scripts (run directly):

```bash
# Mooncake store put/get/remove cycle (needs mooncake_master)
python recipe/drafter_cotraining/scripts/test_mooncake_store.py

# Mooncake round-trip with configurable shapes (needs mooncake_master)
python recipe/drafter_cotraining/scripts/test_vllm_hs_collector.py --seq-len 256 --hidden-dim 3584 --num-samples 5

# Single-controller simulation of drafter pipeline
python recipe/drafter_cotraining/scripts/test_single_controller_hs.py

# Standalone HSCollectorManager end-to-end: vLLM → Mooncake → reader (needs GPU + mooncake_master)
python recipe/drafter_cotraining/scripts/test_hs_collector.py --model Qwen/Qwen2.5-0.5B-Instruct

# Eagle3 loss kernel unit tests
pytest recipe/drafter_cotraining/tests/test_eagle3_loss.py
```

---

## Core Architecture

### 3-Layer Composition

```
Layer 3: ActorRolloutRefWorker    — RL-aware, composes workers + rollout
Layer 2: TrainingWorker           — Batch orchestration, metrics, loss injection
Layer 1: BaseEngine / BaseRollout — Pure computation (two separate hierarchies)
```

### Training Engine (BaseEngine — `verl/workers/engine/base.py`)
Handles gradient computation: forward/backward/optimizer.

```
BaseEngine
├── FSDPEngine           (engine/fsdp/transformer_impl.py)
├── FSDPDrafterEngine    (recipe/drafter_cotraining/engine/drafter_engine.py)   ← submodule
├── MegatronEngine       (engine/megatron/transformer_impl.py)
├── VeOmniEngine         (engine/veomni/transformer_impl.py)
└── MindspeedEngine      (engine/mindspeed/transformer_impl.py)
```

Selected via `EngineRegistry` with key `(model_type, backend, device)`.

Key interface: `train_batch()`, `infer_batch()`, `get_per_tensor_param()`, `train_mode()`/`eval_mode()`, `to(device)`

### Rollout Engine (BaseRollout — `verl/workers/rollout/base.py`)
Handles autoregressive token generation via inference servers.

```
BaseRollout
├── ServerAdapter (SGLang)   — HTTP client
├── ServerAdapter (vLLM)     — Ray actor client
├── ServerAdapter (TRT-LLM)  — HTTP client
├── HFRollout                — Sync HuggingFace
└── NaiveRollout             — Single-GPU PyTorch
```

Selected via `_ROLLOUT_REGISTRY` with key `(engine_name, mode)`.

Key interface: `async resume()`, `async update_weights()`, `async release()`, `generate_sequences()`

### TrainingWorker (`verl/workers/engine_workers.py:68`)
Wraps one BaseEngine. Adds dispatch/collect decorators, micro-batching, loss injection, metrics aggregation.

### ActorRolloutRefWorker (`verl/workers/engine_workers.py:412`)
The "hybrid worker" — composes multiple TrainingWorkers + BaseRollout:

```python
self.actor:   TrainingWorker  → BaseEngine   # policy gradient
self.ref:     TrainingWorker  → BaseEngine   # frozen reference (optional)
self.rollout: BaseRollout                    # token generation
```

Roles: "actor" | "rollout" | "ref" | "actor_rollout" | "actor_rollout_ref"

### Engine ↔ Rollout Connection (Weight Sync)
`update_weights()` bridges the two engine types:
1. `rollout.resume(tags=["weights"])` — wake rollout GPU memory
2. `actor.engine.get_per_tensor_param()` — extract trained weights
3. `rollout.update_weights(params)` — push to inference server
4. `actor.engine.to("cpu")` — offload training model
5. `rollout.resume(tags=["kv_cache"])` — allocate KV cache

GPU time-multiplexing: training engine and rollout engine share GPU but never run simultaneously.

### Two Worker Implementations
- **`engine_workers.py`** (new) — Engine-agnostic via EngineRegistry
- **`fsdp_workers.py`** (legacy) — FSDP-specific, directly manages FSDP model/optimizer

## Training Loop (RayPPOTrainer.fit)

```
① generate_sequences()  → Rollout Engine
② collect_hidden_states()→ vLLM HS Collector (NEW — drafter pipeline)
③ dispatch drafter data  → DrafterDataController (NEW)
④ update_drafter()       → FSDPDrafterEngine (NEW)
⑤ compute_reward()       → Reward workers
⑥ compute_values()       → Critic Training Engine
⑦ compute_log_prob()     → Actor Training Engine (eval mode)
⑧ compute_advantage()    → Driver (GAE/GRPO)
⑨ update_critic()        → Critic Training Engine (train mode)
⑩ update_actor()         → Actor Training Engine (train mode)
⑪ update_weights()       → Sync actor + drafter weights → Rollout + HS Collector
```

## Key Patterns
- **Registry pattern**: `EngineRegistry` and `_ROLLOUT_REGISTRY` for pluggable backends
- **Dispatch/Collect**: RayWorkerGroup transparently splits data across DP ranks
- **Single Controller**: RayPPOTrainer on driver controls all Ray actors via RPC
- **Colocated workers**: Multiple roles (actor, critic, ref) can share GPUs via ResourcePoolManager

## Key Files

| File | Purpose |
|------|---------|
| `verl/trainer/ppo/ray_trainer.py` | Main orchestrator (RayPPOTrainer) |
| `verl/workers/engine_workers.py` | TrainingWorker + ActorRolloutRefWorker (new) |
| `verl/workers/drafter_workers.py` | ActorRolloutRefDrafterWorker (NEW — drafter co-training) |
| `verl/workers/fsdp_workers.py` | Legacy FSDP workers (ActorRolloutRefWorker + CriticWorker) |
| `verl/workers/engine/base.py` | BaseEngine + EngineRegistry |
| `verl/workers/engine/fsdp/drafter_impl.py` | FSDPDrafterEngine (NEW — "drafter_model") |
| `verl/workers/rollout/base.py` | BaseRollout + rollout registry |
| `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` | VllmHSCollector (NEW — HS extraction) |
| `verl/utils/mooncake/` | Mooncake KV store + KV connector (NEW) |
| `verl/models/eagle3/` | Eagle3 model + Forward KL loss (NEW) |
| `verl/trainer/drafter/controller.py` | DrafterDataController (NEW — driver-side) |
| `verl/single_controller/ray/base.py` | RayWorkerGroup (dispatch/collect) |

## Workflow Orchestration

See **[`claude_docs/workflow-orchestration.md`](./claude_docs/workflow-orchestration.md)** for the full guide. Key rules:

1. **Plan mode** for any non-trivial task (3+ steps). Re-plan if things go sideways.
2. **Subagents** liberally — offload research, exploration, parallel analysis. One task per subagent.
3. **Self-improvement** — after any user correction, capture the lesson.
4. **Verify before done** — never mark complete without proving it works (tests, logs, diffs).
5. **Demand elegance** — pause on non-trivial changes to ask "is there a more elegant way?" Skip for simple fixes.
6. **Autonomous bug fixing** — given a bug report, just fix it. Zero hand-holding.

Task tracking: use Claude Code tasks to plan, mark progress, and track results.

Core principles: **Simplicity First** | **No Laziness** (root causes only) | **Minimal Impact**

## Development Notes

- Don't over-engineer — this is research code, can be experimental.
- Don't use too many try-catch unless necessary.
- Use `tasks` to maintain todos.
- No unanimity gate needed — controller pattern guarantees consensus by construction.
- vLLM captures pre-norm last_hidden_states — `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` before target construction. The loss kernel's RMSNorm is for the draft model's own norm (separate concern).
- Don't confused yourself with the legacy worker implementation. For instance (`verl/workers/actor/dp_actor.py`). This is supposed to be deprecated soonly 