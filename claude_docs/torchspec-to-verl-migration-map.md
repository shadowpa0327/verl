# TorchSpec → verl Migration Map

Maps TorchSpec source files to verl migration targets. Includes TorchSpec internals knowledge, file-by-file connections, and current wired/TODO status.

**Reference docs:** `reference/torch_spec_docs/Eagle3-Co-Trained/` (9 design docs), `reference/TorchSpec/` (full source)

---

## TorchSpec Internals — Quick Reference

### What TorchSpec Is
EAGLE3 speculative decoding drafter co-training framework. Trains a tiny drafter (~2% of target size) to predict the target model's output distribution via Forward KL distillation.

### EAGLE3 Architecture
4 trainable components: `fc` (Linear 3*D → D), `midlayer` (1 decoder layer), `norm` (RMSNorm), `lm_head` (Linear D → V). ~140M params for 7B target.

**Module inventory:**

| Module | Source | Status | Role | verl location |
|---|---|---|---|---|
| `embed_tokens` | Actor model | Frozen (weight copy, re-synced) | Token → embedding lookup for drafter input | `Eagle3Model.draft_model.embed_tokens` |
| `fc` | **New (random init)** | **Trainable** | Fuses 3 aux layer HS: `Linear(target_hidden_size * 3, draft_hidden_size)` | `Eagle3Model.draft_model.fc` |
| `midlayer` | **New (random init)** | **Trainable** | Single decoder layer (self-attention + MLP with KV cache) | `Eagle3Model.draft_model.midlayer` |
| `norm` | **New (in draft model)** | **Trainable** | RMSNorm before `lm_head` projection in loss kernel | `Eagle3Model.draft_model.norm` |
| `lm_head` | **Draft model's own** | **Trainable** | Draft hidden states → vocab logits (via `get_lm_head_params()` → `F.linear` in fused loss) | `Eagle3Model.draft_model.lm_head` |
| `target_lm_head` | Actor model | Frozen (weight clone, re-synced) | Compute target distribution from `last_hidden_states` (in `LazyTarget` path) | `FSDPDrafterEngine._target_lm_head_weight` |
| `verifier_norm` | Actor `model.norm` | Frozen (deepcopy, re-synced) | RMSNorm applied to pre-norm `last_hidden_states` from vLLM in `prepare_model_inputs()` | `FSDPDrafterEngine._verifier_norm` |

Note: There are **two separate lm_heads**. The draft model's `lm_head` (trainable) produces draft logits. The `target_lm_head` (frozen, from actor) produces the target distribution for Forward KL. These are distinct tensors — the draft model learns its own output projection. In TorchSpec, only `embed_tokens` has `requires_grad=False` (see `base.py:191`); `lm_head` is never frozen.

**Key difference from TorchSpec:** In TorchSpec the target model is fixed (pure offline distillation). In verl the target IS the actor, and the actor trains every RL step. So frozen modules (`embed_tokens`, `target_lm_head`, `verifier_norm`) must be **re-synced after each `update_actor()`**. We do NOT use live reference sharing (fragile under FSDP resharding). Instead:
1. At drafter init: load/copy from actor's initial weights
2. After each `update_actor()`: copy updated weights from actor → drafter's frozen modules

### 7-Step TTT Loop (`torchspec/models/eagle3.py:192-239`)
Each step simulates one speculative decoding draft step. Step `i` trains the drafter to predict `i` positions ahead. KV cache accumulates across steps. Losses weighted by `0.8^i` exponential decay.

### Forward KL Loss (`torchspec/models/ops/loss.py`)
**Not cross-entropy.** Fused `torch.compile` kernel: RMSNorm → lm_head → Forward KL (`-(target_p * log_softmax(logits)).sum(-1).mean()`). Two variants:
- `compiled_forward_kl_loss` — takes pre-computed target probs (vocab pruning path)
- `compiled_forward_kl_loss_from_hs` — lazy: computes target softmax inside compiled graph

### Controller Pipeline (`torchspec/controller/`)
`AsyncTrainingController` — CPU-only Ray actor, 4 FIFO stores: `_stored_dataset` → `prompt_buffer` → `sample_pool` → `train_queues[]`. Control plane routes metadata; heavy tensors flow through Mooncake. `AsyncInferenceManager` drives engines async. `training_loop()` is sync training with async inference.

### HS Collection — Two Paths

| | **vLLM (used in verl)** | **SGLang (reference only)** |
|---|---|---|
| File | `inference/engine/vllm_engine.py` + `mooncake_hidden_states_connector.py` | `inference/engine/sgl_engine.py` + patches |
| Method | `extract_hidden_states` speculative config + KVConnector | Monkey-patched model forward |
| Patching | **None** | `patches/sglang/sglang.patch` |
| Last HS | **Pre-norm** (trainer applies verifier_norm) | Post-norm |
| Skip decode | `max_tokens=1`, dummy token discarded | `max_new_tokens=0`, fake EOS |
| Connector | Scheduler-side (metadata) + Worker-side (KV cache extract → Mooncake) — **do NOT share state** | Patched scheduler calls `eagle_mooncake_store.put()` |

### Mooncake Store (`torchspec/transfer/mooncake/eagle_store.py`)
Key suffixes: `{key}_hs` (hidden_states), `{key}_ids` (input_ids), `{key}_lhs` (last_hidden_states), `{key}_tgt` (target). All bfloat16. Two paths: GPU-Direct RDMA or async host buffer. Deferred deletion respects lease TTL.

### Key Data Types (`torchspec/utils/types.py`)
- `InferenceInput`: data_id, prompt, input_ids, packed_loss_mask, metadata
- `InferenceOutput`: data_id, mooncake_key, tensor_shapes, tensor_dtypes, packed_loss_mask
- `TrainSample`: mooncake_key, tensor_shapes, tensor_dtypes, packed_loss_mask

### Trainer Hierarchy
`Trainer` (base: device mesh, data fetch loop, checkpointing) → `Eagle3Trainer` (model init with FSDP2, `_forward()` with target+draft, `_backward()` with 0.8^i weighting, `verifier_norm` for pre-norm, `TargetLMHead` broadcast). Batch sizes: micro_batch_size → per_dp_rank_batch_size (×sp_size) → dispatch_batch_size (×dp_size) → global_batch_size (×accumulation_steps).

---

## Architecture Shift: Async → Sync Single-Controller

| TorchSpec | verl | Key Change |
|---|---|---|
| `AsyncTrainingController` (Ray actor) | `DrafterDataController` (in-process on driver) | Ray actor → plain Python object; no locks needed |
| `AsyncInferenceManager` (Ray actor, event loop) | **Eliminated** — driver orchestrates directly | verl is synchronous; driver calls workers via RPC |
| `training_loop()` (driver-side coordinator) | `RayPPOTrainer.fit()` + `drafter_sub_pipeline_sketch()` | Folded into existing RL loop |
| 4 FIFO stores | 2 stores + mesh dispatch | Level 3 handled by verl's `DataProto.chunk()` |
| Standalone Ray actor engines | Colocated on same GPU, time-multiplexed | Sleep/wake instead of separate processes |

### Data Flow Comparison

```
TorchSpec (async):
  Dataset → Controller.prompt_buffer → InferenceManager → Engine pool (round-robin)
  → Mooncake.put() → Controller.sample_pool → train_queues[rank]
  → MooncakeDataFetcher → DataLoader → Eagle3Trainer._forward()

verl (sync, per RL step):
  generate_sequences() → DrafterDataController.push_raw_prompts()
  → pull_raw_prompts() → collect_hidden_states() [VllmHSCollector]
  → Mooncake.put() → DrafterDataController.push_samples()
  → drain_as_dataproto() → mesh dispatch → update_drafter()
  → Mooncake.get() → Eagle3Model.forward() → Mooncake.remove()
```

---

## File-by-File Connection Map

### 1. Controller Layer

| TorchSpec File | verl File | What Changed |
|---|---|---|
| `controller/training_controller.py` | `verl/trainer/drafter/controller.py` | 4 stores → 2 stores. `try_dispatch_batch()` → `drain_as_dataproto()`. Round-robin partition → `DataProto.chunk(dp_size)`. Ray actor → plain object. |
| `utils/types.py` (`InferenceInput`/`InferenceOutput`) | `verl/trainer/drafter/controller.py` (`SequenceMeta`/`SampleMeta`) | Simplified — no packed_loss_mask, no Ray Queue serialization. |
| `controller/loop.py` (`training_loop()`) | `verl/trainer/drafter/orchestration.py` | Async dispatch+retry loop → synchronous 3-phase block inserted into `RayPPOTrainer.fit()`. |
| `controller/inference_manager.py` | **Skipped** | Not needed — verl driver calls `collect_hidden_states()` directly. |
| `training/data_fetcher.py` (`MooncakeDataFetcher`) | Folded into `update_drafter()` | No DataLoader, no Ray Queue. Worker fetches from Mooncake inline. |

### 2. HS Collection (Data Producer)

| TorchSpec File | verl File | What Changed |
|---|---|---|
| `inference/engine/vllm_engine.py` (`VllmEngine`) | `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` (`VllmHSCollector`) | Nearly identical. `kv_connector_module_path` → `verl.utils.mooncake.hidden_states_connector`. Removed `RayActor` base, `InferenceEngine` base, `setup_file_logging`. |
| `inference/engine/mooncake_hidden_states_connector.py` | `verl/utils/mooncake/hidden_states_connector.py` | Copy with import paths: `torchspec.config` → `verl.utils.mooncake.config`, `torchspec.transfer` → `verl.utils.mooncake`. |
| Standalone Ray actor, engine pool | Colocated on same GPU, time-multiplexed via sleep/wake | Major architectural change. |

### 3. Mooncake Store (Data Plane)

| TorchSpec File | verl File | What Changed |
|---|---|---|
| `transfer/mooncake/eagle_store.py` | `verl/utils/mooncake/eagle_store.py` | Nearly identical. Same `put()`/`get()`/`remove_eagle3_tensors()` API, same key suffixes. |
| `transfer/mooncake/store.py` | `verl/utils/mooncake/store.py` | Copy of base `MooncakeHiddenStateStore`. |
| `transfer/mooncake/buffers.py` | `verl/utils/mooncake/buffers.py` | Copy — `HostBufferPool`, `GPUSendBuffer`, `GPUReceiveBuffer`, `AsyncPutManager`. |
| `transfer/mooncake/deferred_delete.py` | `verl/utils/mooncake/deferred_delete.py` | Copy — `DeferredDeleteManager`. |
| `config/mooncake_config.py` | `verl/utils/mooncake/config.py` | Copy — `MooncakeConfig`. |
| N/A | `verl/utils/mooncake/master.py` | **New** — verl manages its own Mooncake master process. |

### 4. Eagle3 Model & Loss (Training Core)

| TorchSpec File | verl File | What Changed |
|---|---|---|
| `models/eagle3.py` (`Eagle3Model`) | `verl/models/eagle3/eagle3_model.py` | Nearly identical TTT loop. Minor: `maybe_mark_dynamic` → `mark_dynamic`, `padding` utility inlined. |
| `models/ops/loss.py` | `verl/models/eagle3/ops/loss.py` | Identical — `compiled_forward_kl_loss`, `compiled_forward_kl_loss_from_hs`. |
| `models/ops/loss_mask.py` | `verl/models/eagle3/ops/loss_mask.py` | Copy. |
| `models/draft/llama3_eagle.py` | `verl/models/eagle3/draft/llama3_eagle.py` | Copy — fc + midlayer + shared embed/lm_head. |
| `models/draft/auto.py` | `verl/models/eagle3/draft/auto.py` | Copy — `AutoEagle3DraftModel` factory. |
| `models/draft/base.py` | `verl/models/eagle3/draft/base.py` | Copy — `Eagle3DraftModel` abstract base. |

### 5. Trainer & Engine (Training Orchestration)

| TorchSpec File | verl File | What Changed |
|---|---|---|
| `training/eagle3_trainer.py` → `init_model()` | `FSDPDrafterEngine.initialize()` + `sync_frozen_modules_from_actor()` | EngineRegistry. `initialize()` wraps `Eagle3Model(draft_model, length=7)`. Frozen modules copied from actor (not refs). |
| `training/eagle3_trainer.py` → `_forward()` (target construction) | `FSDPDrafterEngine.prepare_model_inputs()` | Applies `verifier_norm` to pre-norm last_hs, builds `LazyTarget` via `compute_lazy_target_padded()`. |
| `training/eagle3_trainer.py` → `_forward()` (model call) | `Eagle3Model.forward()` via `self.module(**inputs)` | 7-step TTT loop, unchanged from TorchSpec. |
| `training/eagle3_trainer.py` → `_backward()` | **TODO** — wire via `TrainingWorker.set_loss_fn()` or in `update_drafter()` | 0.8^i weighting not yet connected. |
| `training/trainer.py` (`Trainer` base) | `verl/workers/engine_workers.py` (`TrainingWorker`) | verl's generic worker replaces TorchSpec's base. Provides `train_batch()`, `set_loss_fn()`, micro-batching. |
| `training/trainer_actor.py` (`TrainerActor`) | `ActorRolloutRefDrafterWorker` | Ray actor → verl's `@register` decorator with mesh dispatch. |

### 6. Worker Integration (verl-only — no TorchSpec equivalent)

| verl File | Purpose | TorchSpec Equivalent |
|---|---|---|
| `verl/workers/drafter_workers.py` | `ActorRolloutRefDrafterWorker`: single worker owns actor, ref, rollout, hs_collector, drafter | TorchSpec uses separate Ray actors for each role |
| `verl/workers/engine/fsdp/drafter_impl.py` | `FSDPDrafterEngine`: wraps `Eagle3Model` with FSDP, copies frozen modules from actor, handles target construction in `prepare_model_inputs()` | Split across `eagle3_trainer.py` init + `fsdp.py` setup |

---

## Current Status

For what's done, what's TODO, and verification checklist, see **`claude_docs/migration-status.md`** (single source of truth).
