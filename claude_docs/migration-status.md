# Drafter Co-Training — Migration Status

Single source of truth for what's done, what's TODO, and how to verify.

**Design:** See `rfc-drafter-trainer-integration.md`
**TorchSpec reference:** See `torchspec-to-verl-migration-map.md`

---

## Architecture

```
RayPPOTrainer (driver)
├── DrafterDataController              ← owns Levels 1 & 2
│     raw_prompts → sample_pool → drain_as_dataproto()
│
└── actor_rollout_wg
    └── ActorRolloutRefDrafterWorker
            ├── actor        (inherited)
            ├── ref          (inherited)
            ├── rollout      (vLLM — generation)
            ├── hs_collector (vLLM with KV connector, sleep/wake)
            └── drafter      (FSDPDrafterEngine → Eagle3Model → draft_model)
```

**Data flow per RL step:**
```
generate_sequences()
  → driver: push_raw_prompts()
  → rollout sleeps, HS collector wakes
  → vLLM prefill → MooncakeHiddenStatesConnector → EagleMooncakeStore.put()
  → driver: push_samples(metadata) → drain_as_dataproto()
  → mesh dispatch → each rank gets its shard
  → Mooncake.get() → prepare_model_inputs() → Eagle3Model.forward() (7-step TTT)
  → 0.8^i backward → optimizer step → Mooncake.remove()
  → drafter offloads, actor loads
  → compute_log_prob() → update_actor() → update_weights() (re-sync frozen modules)
```

---

## Component Status

### Done

| Component | verl File(s) | What it does |
|---|---|---|
| **Mooncake store** | `verl/utils/mooncake/` (9 files, ~2,580 lines) | put/get/remove API, RDMA/TCP buffers, config, deferred_delete, master process |
| **KV connector** | `verl/utils/mooncake/hidden_states_connector.py` | vLLM `KVConnectorBase_V1` — writes HS to Mooncake during prefill |
| **Eagle3 model** | `verl/models/eagle3/eagle3_model.py` | `Eagle3Model`: 7-step TTT loop, `PrecomputedTarget`/`LazyTarget`, `_calculate_loss()` |
| **Forward KL loss** | `verl/models/eagle3/ops/loss.py` | `compiled_forward_kl_loss` / `compiled_forward_kl_loss_from_hs` (torch.compiled) |
| **Draft models** | `verl/models/eagle3/draft/` (4 files, ~2,170 lines) | `LlamaForCausalLMEagle3` (fc + midlayer), `AutoEagle3DraftModel` factory, base ABC |
| **HSCollectorManager** | `verl/experimental/hs_collector/` (~200 lines) | Clone of `TeacherModelManager` in colocated mode. Spawns vLLM replicas with `MooncakeHiddenStatesConnector` + `extract_hidden_states`. Called sync from trainer post-rollout. |
| **DrafterDataController** | `verl/trainer/drafter/controller.py` (~165 lines) | 2-level pipeline (raw_prompts, sample_pool) + `drain_as_dataproto()` |
| **Orchestration sketch** | `verl/trainer/drafter/orchestration.py` (~97 lines) | Documents insertion points in `RayPPOTrainer.fit()` |
| **FSDPDrafterEngine** | `verl/workers/engine/fsdp/drafter_impl.py` (~217 lines) | `initialize()`: wraps `Eagle3Model(draft_model, length=7)` with FSDP. `prepare_model_inputs()`: applies `verifier_norm`, builds `LazyTarget`. `sync_frozen_modules_from_actor()`: copies embed_tokens, lm_head, verifier_norm, target_lm_head_weight from actor. |
| **Worker skeleton** | `verl/workers/drafter_workers.py` (~266 lines) | `ActorRolloutRefDrafterWorker`: `collect_hidden_states()` + `update_drafter()` registered. `_sync_drafter_frozen_modules()` called at init + after `update_actor()`. |
| **Drafter CT Trainer** | `verl/trainer/drafter/drafter_ct_ray_trainer.py` (~561 lines) | `RayDrafterCTPPOTrainer`: subclass of `RayPPOTrainer` with full drafter sub-pipeline in `fit()`. |
| **Entry point** | `verl/trainer/drafter/main_drafter_ct_ppo.py` (~139 lines) | `DrafterCTTaskRunner`: entry point that wires `ActorRolloutRefDrafterWorker` + `RayDrafterCTPPOTrainer`. |

### TODO — In Implementation Order

#### TODO 1: `update_drafter()` body ← **Blocker**

**File:** `drafter_workers.py` (currently a stub with `# TODO`)

The engine is ready (`prepare_model_inputs()` handles verifier_norm + LazyTarget, `Eagle3Model.forward()` runs 7-step TTT). What's missing is the plumbing in `update_drafter()`:

1. **Mooncake fetch**: For each key → `mooncake_store.get(key, shapes, dtypes, device)` → unsqueeze batch dim
2. **Collation**: Pad to max seq length, create `attention_mask`, `loss_mask`
3. **Forward**: `self.drafter.engine.prepare_model_inputs(batch)` → `self.drafter.engine.module(**prepared)` → `plosses, _, acces`
4. **Backward**: `sum(0.8**i * plosses[i] for i in range(7)) / accum_steps` → `.backward()`
5. **Optimizer step**: After last micro-batch
6. **Cleanup**: `mooncake_store.remove_eagle3_tensors(key)`

**TorchSpec ref:** `training/data_fetcher.py:79-113` (Mooncake fetch), `training/eagle3_trainer.py:235-277` (forward/backward)

#### ~~TODO 2: Sleep/wake coordination~~ ← **Done**

Handled by `HSCollectorManager.compute_hidden_states()` (wake → infer → sleep) which wraps verl's `RolloutReplica` lifecycle. The manager replaces the previous hand-rolled `VllmHSCollector` + bespoke sleep/wake path.

#### ~~TODO 3: `RayPPOTrainer.fit()` integration~~ ← **Done**

Implemented as a subclass rather than modifying base `RayPPOTrainer`:
- `RayDrafterCTPPOTrainer` (`verl/trainer/drafter/drafter_ct_ray_trainer.py`, ~561 lines) — full drafter sub-pipeline in `fit()`
- `DrafterCTTaskRunner` (`verl/trainer/drafter/main_drafter_ct_ppo.py`, ~139 lines) — entry point wiring

#### TODO 4: Drafter → rollout weight sync ← Not a blocker

**File:** `drafter_workers.py` (has `pass`)

Push drafter weights to rollout vLLM for speculative decoding. Needs rollout to support drafter weight hot-reload. Training works without this.

#### TODO 5: Config schema (YAML) ← Not a blocker

Add `drafter:` section to verl config: `model_path`, `ttt_length`, `learning_rate`, `aux_hidden_states_layers`, `mooncake` sub-config. Can hardcode initially.

---

## Files Deleted (superseded by HSCollectorManager)

| File | Reason |
|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` | Hand-rolled Ray actor replaced by `HSCollectorManager` (clone of colocated `TeacherModelManager`). |
| `scripts/test_real_hs_collector.py` | Superseded by `scripts/test_hs_collector.py`. |
| `scripts/test_hs_collector_verl.py` | Superseded by `scripts/test_hs_collector.py`. |

---

## Frozen Module Sync (verl-specific)

In TorchSpec the target model is fixed. In verl the actor IS the target and trains every RL step. Frozen modules are **weight copies** (not live references — FSDP-safe), re-synced after each `update_actor()`:

| Module | Stored at | Source | Used by |
|---|---|---|---|
| `embed_tokens` | `Eagle3Model.draft_model.embed_tokens` | `actor.model.embed_tokens` | `draft_model.embed_input_ids()` in TTT loop |
| `verifier_norm` | `FSDPDrafterEngine._verifier_norm` | `actor.model.norm` | `prepare_model_inputs()` — normalizes pre-norm last_hs from vLLM |
| `target_lm_head_weight` | `FSDPDrafterEngine._target_lm_head_weight` | `actor.lm_head.weight` | `prepare_model_inputs()` → `compute_lazy_target_padded()` |

Note: `draft_model.lm_head` is **trainable** (the draft model's own output projection) and is NOT synced from actor. Only `target_lm_head_weight` (a separate frozen tensor) comes from the actor for computing the Forward KL target distribution.

Sync chain: `_init_drafter()` → `_sync_drafter_frozen_modules()` and `update_weights()` → `_sync_drafter_frozen_modules()` → `engine.sync_frozen_modules_from_actor(embed, lm_head, norm)`

---

## Verification Checklist

**Imports:**
- [ ] `from verl.utils.mooncake import EagleMooncakeStore, MooncakeConfig`
- [ ] `from verl.utils.mooncake.hidden_states_connector import MooncakeHiddenStatesConnector`
- [ ] `from verl.models.eagle3.ops.loss import compiled_forward_kl_loss`
- [ ] `from verl.models.eagle3.eagle3_model import Eagle3Model, LazyTarget`
- [ ] `from verl.models.eagle3.draft.auto import AutoEagle3DraftModel`
- [ ] `from verl.trainer.drafter.controller import DrafterDataController`
- [ ] `from verl.workers.rollout.vllm_rollout.vllm_hs_collector import VllmHSCollector`

**Unit tests:**
- [ ] `DrafterDataController`: push/pull/drain round-trip, empty pool returns None
- [ ] `DataProto.chunk()` splits `non_tensor_batch` (np.ndarray dtype=object) correctly
- [ ] Mooncake put/get/remove with running master
- [ ] `FSDPDrafterEngine.initialize()` creates `Eagle3Model` wrapping draft model
- [ ] `sync_frozen_modules_from_actor()` copies weights, sets requires_grad=False
- [ ] `prepare_model_inputs()` applies verifier_norm, returns LazyTarget in output

**Integration tests:**
- [ ] VllmHSCollector: vLLM forward → Mooncake write → `kv_transfer_params` has mooncake_key
- [ ] After `update_actor()`, `_sync_drafter_frozen_modules()` re-copies updated weights
- [ ] Full RL step with drafter sub-pipeline end-to-end

---

## Import Adaptation (TorchSpec → verl)

| TorchSpec import | verl replacement |
|---|---|
| `torchspec.config.mooncake_config.*` | `verl.utils.mooncake.config.*` |
| `torchspec.transfer.mooncake.*` | `verl.utils.mooncake.*` |
| `torchspec.models.ops.loss.*` | `verl.models.eagle3.ops.loss.*` |
| `torchspec.models.draft.*` | `verl.models.eagle3.draft.*` |
| `torchspec.models.eagle3.*` | `verl.models.eagle3.eagle3_model.*` |
| `torchspec.utils.logging.logger` | `logging.getLogger(__name__)` |
| `torchspec.models.target.eagle3_target_model.Eagle3TargetOutput` | Local dataclass in `eagle_store.py` |
