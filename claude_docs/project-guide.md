# verl — RL Training Framework

## Project Context
This is the verl codebase, an RL training framework (PPO/GRPO). The active work is EAGLE drafter **pretraining** — a standalone pipeline that trains a tiny draft model on hidden states extracted from a target model. Branch: `feat/draft-model-train`.

**Co-training (RL + drafter) is deferred** — see `migration-status.md` "Deferred: Co-Training".

## Drafter Pretrain (Active Scope)

### What We're Building

EAGLE drafter pretraining: train a tiny draft model (~2% of target params) on hidden states extracted from a frozen target model via a colocated vLLM HS collector. No RL loop, no actor training, no rollout weight sync.

```
Pretrain Loop:

  load parquet data              -> tokenize conversations, build loss masks
  compute_hidden_states()        -> vLLM HS collector (prefill-only, KV connector -> Mooncake)
  dispatch drafter data          -> per-rank mesh dispatch
  update_drafter()               -> FSDPDrafterEngine (Eagle3 Forward KL loss)
  evaluate_drafter()             -> (periodic) eval on held-out data
  save_drafter_checkpoint()      -> (periodic) save checkpoint
```

### Architecture

```
DraftModelPretrainTrainer (driver, CPU)
|-- HSCollectorManager                 <- colocated vLLM replicas + KV connector
|     compute_hidden_states(batch) -- wake -> prefill -> Mooncake -> sleep
|
`-- drafter_wg (RayWorkerGroup, GPU)
    `-- DrafterPretrainWorker          <- extends Worker (NOT ActorRolloutRefWorker)
            `-- drafter   (TrainingWorker -> FSDPDrafterEngine -> Eagle3Model)
            [no actor, no ref, no rollout]
```

### Key Design Decisions

- **Zero verl core changes** — all drafter code lives in `recipe/drafter_cotraining/`. The `verl/` directory is used as-is from upstream.
- **DrafterPretrainWorker extends bare `Worker`** — not `ActorRolloutRefWorker`. Sets `self.rollout = None`, `self.actor = None`, `self.ref = None`. Reuses `ActorRolloutRefDrafterWorker` methods via class-level attribute aliasing.
- **Frozen weights from disk** — `embed_tokens`, `target_lm_head_weight`, `verifier_norm` are loaded from `target_model_path` on disk at init. No live actor to sync from (no-op `_sync_drafter_frozen_modules`).
- **vLLM for HS collection** — uses `extract_hidden_states` speculative config + custom `MooncakeHiddenStatesConnector` (KV connector). Zero patching of vLLM internals.
- **Mooncake** as data plane — heavy tensors (hidden states) flow through Mooncake KV store. The pipeline only routes lightweight metadata (keys, shapes).
- **Pre-norm handling** — vLLM captures `last_hidden_states` before the final RMSNorm (pre-norm). `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` (a frozen copy of the target's `model.norm`) to normalize before target distribution computations.

### Where to make changes — recipe submodule only

**All drafter code lives in `recipe/drafter_cotraining/`** (a git submodule).
The parent `verl/` is used as-is — **zero verl core changes**.

**Default rule for new work:**

> *Put all changes in `recipe/drafter_cotraining/`. Touch `verl/...` (parent)
> only if the change is a generic upstream improvement that benefits all verl
> users and could be upstreamed as a separate PR.*

Concrete guidance:

| Change kind | Where it goes |
|---|---|
| Drafter worker / engine logic | `recipe/drafter_cotraining/workers/{engine_workers,drafter_engine}.py` |
| Eagle3 model / loss / draft | `recipe/drafter_cotraining/eagle3/...` |
| Mooncake transfer / KV connector | `recipe/drafter_cotraining/mooncake/...` |
| HS collector manager | `recipe/drafter_cotraining/hs_collector/...` |
| Pretrain trainer / launcher | `recipe/drafter_cotraining/{main_drafter_pretrain.py,trainer/pretrain_trainer.py}` |
| Data pipeline / collator | `recipe/drafter_cotraining/{data/,utils/}` |
| Test / smoke scripts | `recipe/drafter_cotraining/scripts/...` |
| Drafter unit tests | `recipe/drafter_cotraining/tests/...` |
| Drafter YAML config / draft model JSONs | `recipe/drafter_cotraining/config/...` |
| Top-level smoke wrappers | `scripts/run_drafter_*.sh` (parent — they shell out to recipe scripts) |
| Design / status docs | parent `claude_docs/...` |

**Submodule workflow** for recipe changes:

```bash
cd recipe/drafter_cotraining
# edit, test, commit on feat/drafter-cotraining (or topic branch)
git push origin2 feat/drafter-cotraining

cd /root/verl                           # parent
git add recipe                          # stage gitlink bump
git commit -m "[drafter] bump recipe submodule: <topic>"
git push                                # parent stays on feat/draft-model-train
```

### Key Documents

| Doc | Role | What |
|---|---|---|
| `claude_docs/migration-status.md` | **Current state** | What's done, what's deferred, verification checklist |
| `claude_docs/drafter-design.md` | **As-built design** | Architecture, data lifecycle, training-step shape, engine-choice rationale |
| `claude_docs/weight-sync-flows.md` | **Weight sync** | Parameter inventory + 4 sync flows (Flows 2-4 deferred) |

### Testing

**Verification command** (Qwen3-8B pretrain):

```bash
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16
```

Recipe-side test/diag scripts (run directly):

```bash
# Eagle3 loss kernel unit tests
pytest recipe/drafter_cotraining/tests/test_eagle3_loss.py

# Vocab mapping unit tests
pytest recipe/drafter_cotraining/tests/test_vocab_mapping.py

# Pretrain loss mask pipeline test
pytest recipe/drafter_cotraining/tests/test_pretrain_loss_mask_pipeline.py
```

---

## Core Architecture (Reference)

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
|-- FSDPEngine           (engine/fsdp/transformer_impl.py)
|-- FSDPDrafterEngine    (recipe/drafter_cotraining/workers/drafter_engine.py)   <- submodule
|-- MegatronEngine       (engine/megatron/transformer_impl.py)
|-- VeOmniEngine         (engine/veomni/transformer_impl.py)
`-- MindspeedEngine      (engine/mindspeed/transformer_impl.py)
```

Selected via `EngineRegistry` with key `(model_type, backend, device)`.

Key interface: `train_batch()`, `infer_batch()`, `get_per_tensor_param()`, `train_mode()`/`eval_mode()`, `to(device)`

### Rollout Engine (BaseRollout — `verl/workers/rollout/base.py`)
Handles autoregressive token generation via inference servers.

```
BaseRollout
|-- ServerAdapter (SGLang)   — HTTP client
|-- ServerAdapter (vLLM)     — Ray actor client
|-- ServerAdapter (TRT-LLM)  — HTTP client
|-- HFRollout                — Sync HuggingFace
`-- NaiveRollout             — Single-GPU PyTorch
```

Selected via `_ROLLOUT_REGISTRY` with key `(engine_name, mode)`.

Key interface: `async resume()`, `async update_weights()`, `async release()`, `generate_sequences()`

### TrainingWorker (`verl/workers/engine_workers.py:68`)
Wraps one BaseEngine. Adds dispatch/collect decorators, micro-batching, loss injection, metrics aggregation.

### ActorRolloutRefWorker (`verl/workers/engine_workers.py:412`)
The "hybrid worker" — composes multiple TrainingWorkers + BaseRollout:

```python
self.actor:   TrainingWorker  -> BaseEngine   # policy gradient
self.ref:     TrainingWorker  -> BaseEngine   # frozen reference (optional)
self.rollout: BaseRollout                    # token generation
```

Roles: "actor" | "rollout" | "ref" | "actor_rollout" | "actor_rollout_ref"

### Engine <-> Rollout Connection (Weight Sync)
`update_weights()` bridges the two engine types:
1. `rollout.resume(tags=["weights"])` — wake rollout GPU memory
2. `actor.engine.get_per_tensor_param()` — extract trained weights
3. `rollout.update_weights(params)` — push to inference server
4. `actor.engine.to("cpu")` — offload training model
5. `rollout.resume(tags=["kv_cache"])` — allocate KV cache

GPU time-multiplexing: training engine and rollout engine share GPU but never run simultaneously.

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
| `verl/workers/fsdp_workers.py` | Legacy FSDP workers (ActorRolloutRefWorker + CriticWorker) |
| `verl/workers/engine/base.py` | BaseEngine + EngineRegistry |
| `verl/workers/rollout/base.py` | BaseRollout + rollout registry |
| `verl/single_controller/ray/base.py` | RayWorkerGroup (dispatch/collect) |
| `recipe/drafter_cotraining/workers/engine_workers.py` | DrafterPretrainWorker + ActorRolloutRefDrafterWorker (deferred) |
| `recipe/drafter_cotraining/workers/drafter_engine.py` | FSDPDrafterEngine (Eagle3 model) |
| `recipe/drafter_cotraining/trainer/pretrain_trainer.py` | DraftModelPretrainTrainer |
| `recipe/drafter_cotraining/hs_collector/` | HSCollectorManager |
| `recipe/drafter_cotraining/mooncake/` | Mooncake KV store + KV connector |

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
- **Zero verl core changes** — all drafter code lives in `recipe/drafter_cotraining/`. The verl/ submodule is used as-is from upstream.
- vLLM captures pre-norm last_hidden_states — `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` before target construction. The loss kernel's RMSNorm is for the draft model's own norm (separate concern).
- `DrafterPretrainWorker` has `self.rollout = None` — it never touches the vLLM rollout engine. Frozen weights are loaded from `target_model_path` on disk, not from a live actor.
