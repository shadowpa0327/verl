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
RayPPOTrainer (driver)
├── DrafterDataController              ← owns raw_prompts + sample_pool
│     drain_as_dataproto() → mesh dispatch per DP rank
│
└── actor_rollout_wg
    └── ActorRolloutRefDrafterWorker
            ├── actor        (TrainingWorker → FSDPEngine)       [inherited]
            ├── ref          (TrainingWorker → FSDPEngine)       [inherited]
            ├── rollout      (BaseRollout → vLLM)                [inherited]
            ├── hs_collector (VllmHSCollector → vLLM + KV connector) [NEW]
            └── drafter      (TrainingWorker → FSDPDrafterEngine)    [NEW]
```

### Key Design Decisions

- **vLLM for HS collection** — uses `extract_hidden_states` speculative config + custom `MooncakeHiddenStatesConnector` (KV connector). Zero patching of vLLM internals.
- **Mooncake** as data plane — heavy tensors (hidden states) flow through Mooncake KV store. The 3-level pipeline only routes lightweight metadata (keys, shapes).
- **DrafterDataController on driver** — mirrors verl's single-controller pattern. Levels 1 & 2 are global; Level 3 dispatch via `DataProto.chunk()` + drafter mesh.
- **No unanimity gate** — the centralized controller + mesh dispatch guarantees all ranks get data or none do. Consensus by construction.
- **Pre-norm handling** — vLLM captures `last_hidden_states` before the final RMSNorm (pre-norm). `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` (a frozen copy of the actor's `model.norm`) to normalize before target distribution computation. Note: the loss kernel's built-in RMSNorm is for the *draft* model's norm, not the verifier — these are separate.

### TorchSpec Reference (Source Knowledge)

The verl drafter pipeline is ported from TorchSpec. Reference material:

| Location | What |
|---|---|
| `reference/TorchSpec/` | Full TorchSpec source code |
| `reference/torch_spec_docs/Eagle3-Co-Trained/` | 9 design docs covering architecture, data flow, code maps |

**Key TorchSpec internals to understand:**
- **Eagle3 architecture**: 4 trainable components: `fc` (3*D→D) + `midlayer` (1 decoder layer) + `norm` (RMSNorm) + `lm_head` (D→V). ~140M params for 7B target. Only `embed_tokens` is frozen (from target, `requires_grad=False` at `base.py:191`). `lm_head` is the draft model's own trainable projection — NOT frozen, NOT shared from target. See `torchspec/models/draft/llama3_eagle.py:1737-1797`.
- **7-step TTT loop**: Each step predicts `i` positions ahead, KV cache accumulates, losses weighted 0.8^i. See `torchspec/models/eagle3.py:192-239`.
- **Forward KL loss**: NOT cross-entropy. Fused `torch.compile`: RMSNorm → lm_head → `-(target_p * log_softmax(logits)).sum(-1).mean()`. Two variants for vocab pruning vs lazy target. See `torchspec/models/ops/loss.py`.
- **Controller**: 4 FIFO stores (`_stored_dataset` → `prompt_buffer` → `sample_pool` → `train_queues[]`). Only routes metadata; tensors in Mooncake. See `torchspec/controller/training_controller.py`.
- **KV Connector**: Scheduler-side (pre-compute metadata) and Worker-side (KV cache extract → Mooncake) — **they do NOT share state**. See `torchspec/inference/engine/mooncake_hidden_states_connector.py`.
- **Module sharing**: Only `embed_tokens` is frozen from target (for draft input). `target_lm_head` + `verifier_norm` loaded separately from target (for computing target distribution). `fc` + `midlayer` + `norm` + `lm_head` are all trainable, part of the draft model. The draft's `lm_head` is distinct from `target_lm_head`.
- **verifier_norm**: vLLM captures pre-norm last_hs. In verl, `FSDPDrafterEngine._verifier_norm` (frozen copy of actor's `model.norm`) is applied in `prepare_model_inputs()` before target construction. TorchSpec equivalent: `eagle3_trainer.py:239-241`.
- **target_lm_head_weight**: Stored as `FSDPDrafterEngine._target_lm_head_weight` (frozen clone of actor's `lm_head.weight`). Used in `prepare_model_inputs()` → `compute_lazy_target_padded()` to build `LazyTarget`. Distinct from the draft model's `lm_head` which is used for draft logits in the loss kernel.

### Migration Files (branch `feat/drafter-cotraining`)

| Directory | What | Lines |
|---|---|---|
| `verl/utils/mooncake/` | Mooncake KV store (config, put/get/remove, buffers, KV connector) | ~2,600 |
| `verl/models/eagle3/` | Eagle3 model (draft arch, Forward KL loss, TTT loop) | ~2,700 |
| `verl/trainer/drafter/` | DrafterDataController + orchestration sketch | ~260 |
| `verl/workers/drafter_workers.py` | ActorRolloutRefDrafterWorker | ~230 |
| `verl/workers/engine/fsdp/drafter_impl.py` | FSDPDrafterEngine ("drafter_model") | ~130 |
| `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` | vLLM HS collector (prefill + KV connector) | ~475 |
| `scripts/test_*.py` | Test scripts (Mooncake store, vLLM HS pipeline) | ~570 |

### Key Documents

| Doc | Role | What |
|---|---|---|
| `claude_docs/migration-status.md` | **Current state** | What's done, what's TODO (ordered), frozen module sync, verification checklist |
| `claude_docs/rfc-drafter-trainer-integration.md` | **Design (locked)** | Why: 3-level pipeline, DrafterDataController, sleep/wake, dispatch mechanism |
| `claude_docs/torchspec-to-verl-migration-map.md` | **Reference** | TorchSpec internals + file-by-file connection map to verl |
| `reference/torch_spec_docs/Eagle3-Co-Trained/` | **Source docs** | 9 TorchSpec design docs (Eagle3 training, controller, data flow, code maps) |
| `reference/TorchSpec/torchspec/` | **Source code** | Full TorchSpec source — the upstream reference implementation |

### Testing

```bash
# Stage 1: Import check (no GPU needed)
python scripts/test_vllm_hs_collector.py --stage 1

# Stage 2: Mooncake store (needs mooncake_master)
python scripts/test_vllm_hs_collector.py --stage 2

# Stage 3: Full pipeline (needs GPU + mooncake_master)
python scripts/test_vllm_hs_collector.py --stage 3 --model-path Qwen/Qwen2.5-0.5B-Instruct
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
├── FSDPDrafterEngine    (engine/fsdp/drafter_impl.py)        ← NEW
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
3. **Self-improvement** — after any user correction, capture the lesson in `tasks/lessons.md`.
4. **Verify before done** — never mark complete without proving it works (tests, logs, diffs).
5. **Demand elegance** — pause on non-trivial changes to ask "is there a more elegant way?" Skip for simple fixes.
6. **Autonomous bug fixing** — given a bug report, just fix it. Zero hand-holding.

Task tracking: plan to `tasks/todo.md`, mark progress, document results, capture lessons.

Core principles: **Simplicity First** | **No Laziness** (root causes only) | **Minimal Impact**

## Development Notes

- Don't over-engineer — this is research code, can be experimental.
- Don't use too many try-catch unless necessary.
- Use `tasks` to maintain todos.
- No unanimity gate needed — controller pattern guarantees consensus by construction.
- vLLM captures pre-norm last_hidden_states — `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` before target construction. The loss kernel's RMSNorm is for the draft model's own norm (separate concern).
