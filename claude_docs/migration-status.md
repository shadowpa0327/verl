# Drafter Co-Training Migration Status

## Summary

Ported TorchSpec's EAGLE drafter co-training pipeline into verl. 26 files, 6,820 lines across 6 modules. **HS collector now uses vLLM — no SGLang patching needed.**

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
            └── drafter      (FSDPDrafterEngine, "drafter_model")
```

**Data flow per RL step:**
```
generate_sequences()
  → driver: push_raw_prompts()
  → rollout sleeps, HS collector wakes
  → vLLM forward pass → save_kv_layer() → EagleMooncakeStore.put()
  → request_finished() → kv_transfer_params with mooncake_key
  → driver: push_samples() → drain_as_dataproto()
  → mesh dispatch (drafter mesh, pure DP) → each rank gets its shard
  → Mooncake.get(keys) → FSDPDrafterEngine.train_batch() → Mooncake.remove(keys)
  → drafter offloads, actor loads
  → compute_log_prob() → update_actor() → update_weights()
```

vLLM HS collection detail:
```
vLLM LLM(speculative_config=extract_hidden_states, kv_transfer_config=MooncakeHiddenStatesConnector)
→ forward pass → save_kv_layer() → EagleMooncakeStore.put()
→ request_finished() → kv_transfer_params with mooncake_key
```

---

## Files Created

### Plan A: Mooncake KV Store — `verl/utils/mooncake/` (2,195 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `config.py` | 280 | `torchspec/config/mooncake_config.py` | MooncakeConfig dataclass (protocol, RDMA, buffers, TTL) |
| `store.py` | 262 | `torchspec/transfer/mooncake/store.py` | MooncakeHiddenStateStore base (setup, register buffers, RDMA/TCP) |
| `eagle_store.py` | 586 | `torchspec/transfer/mooncake/eagle_store.py` | EagleMooncakeStore (put/get/remove for HS + input_ids + logits) |
| `buffers.py` | 319 | `torchspec/transfer/mooncake/buffers.py` | HostBuffer, HostBufferPool, AsyncPutManager, GPU send/receive buffers |
| `deferred_delete.py` | 307 | `torchspec/transfer/mooncake/deferred_delete.py` | DeferredDeleteManager (TTL-aware async deletion with retry) |
| `helpers.py` | 80 | `torchspec/transfer/mooncake/helpers.py` | calculate_eagle3_buffer_size, _format_bytes |
| `master.py` | 337 | `torchspec/transfer/mooncake/utils.py` | MooncakeMaster Ray actor (subprocess lifecycle) |
| `hidden_states_connector.py` | 387 | Ported from TorchSpec | vLLM KV connector — `MooncakeHiddenStatesConnector` wires vLLM's KV transfer API to EagleMooncakeStore |
| `__init__.py` | 24 | New | Module exports |

**Adaptations:** `torchspec.*` imports → `verl.utils.mooncake.*`, `RayActor` → `@ray.remote`, logger → `logging.getLogger(__name__)`, `Eagle3TargetOutput` defined locally.

### Plan B: Eagle3 Model & Loss — `verl/models/eagle3/` (2,681 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `ops/loss.py` | 108 | `torchspec/models/ops/loss.py` | `compiled_forward_kl_loss` — fused RMSNorm + lm_head + Forward KL (torch.compiled) |
| `ops/loss_mask.py` | 113 | `torchspec/models/ops/loss_mask.py` | Loss mask for assistant tokens (numba-compiled) |
| `draft/base.py` | 246 | `torchspec/models/draft/base.py` | Eagle3DraftModel ABC (embed, project_hs, backbone, get_lm_head_params) |
| `draft/auto.py` | 111 | `torchspec/models/draft/auto.py` | AutoEagle3DraftModel factory (Llama config → LlamaForCausalLMEagle3) |
| `draft/llama3_eagle.py` | 1,804 | `torchspec/models/draft/llama3_eagle.py` | Llama3 EAGLE: fc + 1 decoder layer + multi attention backends |
| `eagle3_model.py` | 296 | `torchspec/models/eagle3.py` | Eagle3Model TTT loop (PrecomputedTarget, LazyTarget, forward) |

**Adaptations:** flex_attention made optional (try/except), DeepSeek mapping deferred, `padding` utility inlined. Loss files copied as-is (no torchspec deps).

### Plan C: DrafterDataController — `verl/trainer/drafter/` (263 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `controller.py` | 165 | New (rewrite) | DrafterDataController: 2 stores, push/pull/drain_as_dataproto(). Lives on driver. |
| `orchestration.py` | 97 | New | Documents insertion points in RayPPOTrainer.fit() |

**Design:** Mirrors TorchSpec's AsyncTrainingController but synchronous, in-process. Levels 1 & 2 are global on the driver. Level 3 dispatch handled by verl's `make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")` — DataProto.chunk() splits non_tensor_batch (np.ndarray, dtype=object) per DP rank.

### Plan D: HS Collector — vLLM with KV Connector (475 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` | 475 | New (rewrite) | VllmHSCollector: BaseRollout subclass, runs vLLM with MooncakeHiddenStatesConnector to collect hidden states during prefill |

**Design:** Colocated on same GPU as rollout, time-multiplexed via sleep/wake. Instantiates `vLLM LLM(speculative_config=extract_hidden_states, kv_transfer_config=MooncakeHiddenStatesConnector)`. Hidden states are written to Mooncake during the forward pass via `save_kv_layer()` inside the connector; mooncake keys are returned via `kv_transfer_params` on request completion. No SGLang patching needed.

**Replaces:** `sglang_hs_collector.py` + `async_sglang_hs_server.py` (deleted)

### Plan E: Drafter Worker — `verl/workers/` (238 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `drafter_workers.py` | 238 | New (rewrite) | ActorRolloutRefDrafterWorker: extends ActorRolloutRefWorker, adds hs_collector + drafter, registers drafter mesh |

**Design:** Composes inherited actor/ref/rollout with new hs_collector (VllmHSCollector) + drafter. `update_drafter()` uses drafter mesh dispatch. `update_weights()` extended to sync actor → HS collector and drafter → rollout.

### FSDPDrafterEngine — `verl/workers/engine/fsdp/` (129 lines)

| File | Lines | Source | What it does |
|---|---|---|---|
| `drafter_impl.py` | 129 | New (rewrite) | FSDPDrafterEngine registered as "drafter_model". Loads EAGLE via AutoEagle3DraftModel, shares embed_tokens/lm_head from actor, FSDP wraps trainable params only. |

---

## Files Deleted (SGLang HS collector approach — superseded)

| File | Reason |
|---|---|
| `verl/utils/sglang/spec_training_patch.py` | SGLang monkey-patching no longer needed; vLLM KV connector handles HS collection natively |
| `verl/utils/sglang/spec_training_info.py` | SGLang tracking dataclass, no longer needed |
| `verl/workers/rollout/sglang_rollout/sglang_hs_collector.py` | Replaced by `vllm_hs_collector.py` |
| `verl/workers/rollout/sglang_rollout/async_sglang_hs_server.py` | Replaced by vLLM (no custom server launcher needed) |

---

## Test Scripts

| Script | Status |
|---|---|
| `test_sglang_prefill_hs.py` | Deleted (SGLang approach retired) |
| `test_vllm_hs_collector.py` | Replaces above; tests VllmHSCollector end-to-end |

---

## Import Adaptation Summary

| TorchSpec import | verl replacement |
|---|---|
| `torchspec.config.mooncake_config.*` | `verl.utils.mooncake.config.*` |
| `torchspec.transfer.mooncake.*` | `verl.utils.mooncake.*` |
| `torchspec.models.ops.loss.*` | `verl.models.eagle3.ops.loss.*` |
| `torchspec.models.draft.*` | `verl.models.eagle3.draft.*` |
| `torchspec.utils.logging.logger` | `logging.getLogger(__name__)` |
| `torchspec.ray.ray_actor.RayActor` | `@ray.remote` + `ray.util.get_node_ip_address()` |
| `torchspec.ray.ray_actor.node_affinity_for_ip` | `NodeAffinitySchedulingStrategy` |
| `torchspec.utils.env.get_torchspec_env_vars` | Removed |
| `torchspec.models.target.eagle3_target_model.Eagle3TargetOutput` | Local dataclass in `eagle_store.py` |
| `torchspec.models.ops.flex_attention.*` | Optional (try/except, fallback to SDPA) |
| SGLang `HSCollectorAdapter` | `VllmHSCollector` (vLLM KV connector) |

---

## What's New vs Copied

**Copied/adapted from TorchSpec (5,263 lines, ~77%):**
- `verl/utils/mooncake/` — 9 files (including `hidden_states_connector.py`)
- `verl/models/eagle3/` — 9 files

**New code for verl (1,557 lines, ~23%):**
- `verl/trainer/drafter/controller.py` — DrafterDataController
- `verl/trainer/drafter/orchestration.py` — Integration sketch
- `verl/workers/drafter_workers.py` — ActorRolloutRefDrafterWorker (imports VllmHSCollector)
- `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py` — VllmHSCollector (vLLM + KV connector)
- `verl/workers/engine/fsdp/drafter_impl.py` — FSDPDrafterEngine

---

## End-to-End Verification Checklist

To bring this to a working state, test these in order:

- [ ] `from verl.utils.mooncake import EagleMooncakeStore, MooncakeConfig` — imports clean
- [ ] `from verl.utils.mooncake.hidden_states_connector import MooncakeHiddenStatesConnector` — imports clean
- [ ] `from verl.models.eagle3.ops.loss import compiled_forward_kl_loss` — imports clean
- [ ] `from verl.models.eagle3.draft.auto import AutoEagle3DraftModel` — imports clean
- [ ] `from verl.trainer.drafter.controller import DrafterDataController` — imports clean
- [ ] `from verl.workers.rollout.vllm_rollout.vllm_hs_collector import VllmHSCollector` — imports clean
- [ ] `DrafterDataController(dp_size=8).drain_as_dataproto()` — returns None on empty pool
- [ ] `DrafterDataController.push_samples([...]).drain_as_dataproto()` — returns DataProto with non_tensor_batch
- [ ] `DataProto.chunk()` correctly splits non_tensor_batch (np.ndarray dtype=object)
- [ ] Mooncake put/get/remove with a running Mooncake master
- [ ] VllmHSCollector instantiates vLLM with `speculative_config=extract_hidden_states` and `kv_transfer_config=MooncakeHiddenStatesConnector`
- [ ] vLLM forward pass writes HS to Mooncake via `save_kv_layer()`; `kv_transfer_params` contains `mooncake_key`
- [ ] FSDPDrafterEngine loads EAGLE model, shares embed_tokens/lm_head
- [ ] Full RL step with drafter sub-pipeline (integration test): `test_vllm_hs_collector.py`

---

## Related Documents

- `claude_docs/rfc-drafter-trainer-integration.md` — Approved RFC (locked)
- `claude_docs/torchspec-to-verl-migration-map.md` — File-by-file mapping
- `docs/superpowers/plans/2026-04-06-mooncake-store.md` — Plan A
