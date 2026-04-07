# TorchSpec → verl Migration Map

Maps TorchSpec source files to verl migration targets based on the approved RFC.

## Overview

```
TorchSpec (async, Ray actors)          verl (sync, single-controller)
─────────────────────────              ───────────────────────────────

AsyncTrainingController ─────────────→ DrafterDataController (on driver)
AsyncInferenceManager ───────────────→ (not needed — driver orchestrates)
VllmEngine + KV Connector ───────────→ HS Collector vLLM worker (no patching)
EagleMooncakeStore ──────────────────→ Mooncake integration in verl
MooncakeDataFetcher ─────────────────→ Folded into update_drafter()
TrainerActor / Eagle3Trainer ────────→ FSDPDrafterEngine + TrainingWorker
training_loop() ─────────────────────→ RayPPOTrainer.fit() orchestration
```

---

## File-by-File Mapping

### 1. Controller: `torchspec/controller/training_controller.py`

**What it does:** Central hub — owns 4 FIFO stores (prompt_buffer, sample_pool, train_queues, eval_pool). Routes metadata between inference and training. CPU-only Ray actor.

**Key interfaces to port:**
- `push_inference_results(results)` — fills sample_pool with InferenceOutput (mooncake_key + metadata)
- `try_dispatch_batch()` — pops from sample_pool, partitions round-robin, pushes to train_queues
- `get_prompts(n)` — drains prompt_buffer for inference
- `_partition_results(results, dp_size)` — round-robin partition

**verl target:** `DrafterDataController` (in-process on RayPPOTrainer, not a Ray actor)

**What changes:**
- TorchSpec: Ray actor with threading locks, async polling
- verl: Simple Python object, synchronous, no locks needed
- TorchSpec: 4 stores (dataset, prompt_buffer, sample_pool, train_queues)
- verl: 2 stores (raw_prompts, sample_pool) — Level 3 handled by mesh dispatch
- TorchSpec: `try_dispatch_batch()` pushes to Ray Queues
- verl: `drain_as_dataproto()` packs into DataProto, mesh dispatch splits per rank

**What to reference/copy:**
- Partition logic (`_partition_results`) — though verl's DataProto.chunk() replaces this
- Pool size tracking for backpressure (if needed later)
- Status/monitoring interface (`get_status()`, `get_speeds()`)

---

### 2. Inference Manager: `torchspec/controller/inference_manager.py`

**What it does:** Async event loop — continuously pulls prompts from controller, dispatches to SGLang engines, collects results, pushes back to controller. Handles backpressure (pauses when sample_pool too large).

**verl target:** NOT NEEDED as a separate component

**Why:** verl is synchronous. The driver (RayPPOTrainer) directly:
1. Sends sequences to HS collector workers via mesh dispatch
2. Collects results via mesh collect
3. Pushes metadata into DrafterDataController

The driver IS the inference manager in verl's sync model.

**What to reference:**
- Backpressure logic (`_await_pool_capacity`) — useful if we add buffer limits later
- Metrics collection (`MetricsCollector`) — throughput tracking

---

### 3. VllmEngine + KV Connector: `torchspec/inference/engine/mooncake_hidden_states_connector.py`

**What it does:** `KVConnectorBase_V1` subclass that intercepts vLLM model forward and writes hidden states to Mooncake. No patching of vLLM internals — uses the official KV Connector plugin interface.

**Key code:**
```python
class MooncakeHiddenStatesConnector(KVConnectorBase_V1):
    def save_hidden_states(self, hidden_states, input_ids, ...):
        # Called from vLLM model runner after prefill forward
        # Async DtoH + Mooncake.put()
        # Returns: mooncake_key, tensor_shapes, tensor_dtypes
```

**verl target:** `self.hs_collector` in ActorRolloutRefDrafterWorker — a new vLLM instance (BaseRollout) configured with this KV connector for prefill-only mode

**What changes:**
- TorchSpec: Standalone Ray actor, manages own GPU lifecycle
- verl: Colocated with rollout on same GPU, sleep/wake time-multiplexed
- TorchSpec: Multi-engine pool with round-robin (`EnginePool`)
- verl: Single HS collector instance per worker (same DP topology as rollout)

**What to port:**
- `mooncake_hidden_states_connector.py` — copy mostly as-is, plug in via vLLM `--kv-connector` flag
- generate() call pattern (prefill-only, return mooncake_key + metadata)
- Mooncake store setup registered within the connector

**Key advantage:** No patching of vLLM internals. The KV Connector interface is stable and officially supported.

> **Warning — pre-norm last_hidden_states:** vLLM returns `last_hidden_states` *before* the final RMSNorm layer. The Eagle3 drafter training must account for this: either apply RMSNorm as a preprocessing step or adapt the loss accordingly.

---

### 4. SGLang Patches: `patches/sglang/v0.5.8.post1/sglang.patch` — **Skipped — using vLLM instead**

**Status:** This section is skipped. The migration pivoted to vLLM + KV Connector, which requires no SGLang patching.

**Original plan (for reference only):** Would have patched sglang internals to capture hidden states during prefill and write to Mooncake.

| File patched | What was planned |
|---|---|
| `engine.py` | `spec_training_data_id`, `packed_loss_mask` fields on generate() |
| `io_struct.py` | New fields in `GenerateReqInput` for spec_training mode |
| `scheduler_output_processor_mixin.py` | Hidden states extraction, target logits, Mooncake put |
| `model_runner.py` | Capture aux hidden states at specified layer IDs |
| `logits_processor.py` | Skip sampling in spec_training mode |
| `spec_training_info.py` | New data structure tracking spec_training through pipeline |

**Why skipped:** vLLM's `KVConnectorBase_V1` achieves the same without any patching, and verl already has vLLM integration in its rollout registry.

---

### 5. Mooncake Store: `torchspec/transfer/mooncake/`

**Files:**
- `store.py` — `MooncakeHiddenStateStore` (base: setup, register buffers, RDMA/TCP)
- `eagle_store.py` — `EagleMooncakeStore` (Eagle3: put/get/remove for HS, logits, input_ids)
- `buffers.py` — `HostBufferPool`, `GPUSendBuffer`, `GPUReceiveBuffer`
- `deferred_delete.py` — `DeferredDeleteManager` (async cleanup)

**Key interfaces:**
```python
class EagleMooncakeStore:
    def put(self, key, hidden_states, target_logits, input_ids, ...):
        # Async DtoH copy on _copy_stream
        # batch_put_from to Mooncake via RDMA/TCP
        # Returns: {shapes, dtypes} metadata dict
    
    def get(self, key, shapes, dtypes):
        # batch_get_into from Mooncake → GPU buffer
        # Returns: dict of tensors on GPU
    
    def remove_eagle3_tensors(self, key):
        # Deferred delete via DeferredDeleteManager
    
    def flush(self):
        # Block until all in-flight puts complete
```

**Storage format per sample:**
- `{key}_hs` — hidden_states (bfloat16)
- `{key}_tgt` — target logits (bfloat16)
- `{key}_ids` — input_ids
- `{key}_lhs` — last_hidden_states (optional)

**verl target:** Can be used mostly as-is, or adapted into a verl utility module

**What to port:**
- `EagleMooncakeStore` — the main put/get/remove interface
- `HostBufferPool` — RDMA-registered buffer management
- `DeferredDeleteManager` — async cleanup (or simplify to sync for V1)
- Configuration: `MooncakeConfig` (protocol, device, buffer sizes, TTL)

**What might change:**
- verl's colocated model may allow simpler buffer management (same GPU)
- GPU Direct may not be needed in V1 (same-node, CPU buffer sufficient)
- Could start with TCP protocol, upgrade to RDMA later

---

### 6. Data Fetcher: `torchspec/training/data_fetcher.py`

**What it does:** Bridges Ray Queue → Mooncake → DataLoader. `MooncakeDataset` is an IterableDataset that blocks on queue.get(), loads tensors from Mooncake, and yields training batches.

**Key code:**
```python
class MooncakeDataset(IterableDataset):
    def __iter__(self):
        while True:
            sample = self.ray_queue.get()  # blocks
            if sample is None: break       # sentinel
            tensors = self._load_from_mooncake(sample)
            self._cleanup_mooncake_data(sample.mooncake_key)
            yield tensors

class MooncakeDataFetcher:
    def __init__(self, queue, mooncake_store, collator, ...):
        dataset = MooncakeDataset(queue, mooncake_store)
        self.dataloader = DataLoader(dataset, collate_fn=collator, ...)
```

**verl target:** Folded into `update_drafter()` on workers — no separate DataLoader needed

**Why:** In verl's sync model, the worker receives its shard of SampleMeta via mesh dispatch, then directly calls Mooncake.get() + train_batch(). No blocking queue, no DataLoader.

**What to reference:**
- `_load_from_mooncake()` — the Mooncake.get() call pattern (shapes, dtypes handling)
- Collation logic — padding, batching of variable-length hidden states
- Loss mask handling — `packed_loss_mask` parsing

---

### 7. Eagle3 Model & Loss: `torchspec/models/`

**Files:**
- `eagle3.py` — `Eagle3Model` (nn.Module: PrecomputedTarget/LazyTarget forward, wraps draft model)
- `ops/loss.py` — `compiled_forward_kl_loss`, `compiled_forward_kl_loss_from_hs` (torch.compiled fused RMSNorm + lm_head + Forward KL loss kernels)
- `ops/loss_mask.py` — Numba-compiled loss mask computation
- `draft/base.py` — `Eagle3DraftModelBase` (base class for trainable draft params: fc + decoder layer)
- `draft/auto.py` — `AutoEagle3DraftModel` (auto-registry: model config → architecture-specific impl)
- `draft/llama3_eagle.py`, `draft/deepseek_eagle.py`, etc. — Architecture-specific draft models

**Key detail — loss is Forward KL, NOT cross-entropy:**
```python
# torchspec/models/ops/loss.py
# Forward KL: -(target_p * log_softmax(logits)).sum(-1).mean()
# Fused with RMSNorm + lm_head for efficiency
```

> **Warning — Forward KL loss:** The drafter loss is **forward KL divergence** (teacher → student), not cross-entropy. Do not substitute cross-entropy when porting `compiled_forward_kl_loss`. The fused kernel applies RMSNorm → lm_head → forward KL in one pass; if using vLLM's pre-norm hidden states as input, the RMSNorm step is mandatory before calling lm_head.

**verl target:** Split across:
- Draft model definitions → model for `FSDPDrafterEngine`
- Forward KL loss → `eagle_draft_loss` for `TrainingWorker.set_loss_fn()`
- Loss mask utilities → port or adapt

**What to port:**
- `Eagle3Model` forward pass logic
- `compiled_forward_kl_loss` — the actual loss (Forward KL, not cross-entropy)
- Architecture-specific draft models (at least the ones we need: e.g. Qwen, Llama)
- `AutoEagle3DraftModel` registry pattern

---

### 8. Trainer: `torchspec/training/`

**Files:**
- `trainer_actor.py` — `TrainerActor` (Ray actor, manages distributed training)
- `trainer.py` — `Trainer` (base class, training loop mechanics)
- `eagle3_trainer.py` — `Eagle3Trainer` (drives Eagle3Model, loss, optimizer)
- `fsdp.py` — `apply_fsdp2()` (FSDP2 wrapping utilities)

**verl target:** Split across:
- `FSDPDrafterEngine` — the EAGLE model forward/backward (Layer 1)
- `TrainingWorker` — micro-batching, loss injection (Layer 2)
- `update_drafter()` — orchestration on worker side

**What to port:**
- Eagle3Trainer._train_step() flow → adapted into verl's train_batch pattern
- Model initialization (shared embed_tokens/lm_head from actor)
- Gradient accumulation / micro-batching config
- FSDP2 wrapping approach (reference `fsdp.py`)

**What changes:**
- TorchSpec: Standalone trainer with own optimizer, scheduler, DDP
- verl: Uses FSDP via FSDPDrafterEngine, verl manages optimizer

---

### 8. Training Loop: `torchspec/controller/loop.py`

**What it does:** Main orchestration — per optimizer step: dispatch batches → fire training → collect metrics → checkpoint → epoch management.

**verl target:** Added to `RayPPOTrainer.fit()` as the drafter sub-pipeline

**What changes:**
- TorchSpec: Async loop with retry polling (sleep 10ms until data ready)
- verl: Sync — data is always ready after HS collection completes
- TorchSpec: Separate training_loop function
- verl: Integrated into existing RL step in RayPPOTrainer.fit()

---

### 10. Decode-Side: Weight Sync & Speculative Decoding

**Files:**
- `inference/engine/sgl_engine_decode.py` — `SglDecodeEngineMixin` (speculative decoding generation, `_sync_draft_weights()`)
- `patches/sglang/v0.5.8.post1/sglang_decode.patch` — Decode-mode SGLang patches (20 files: scheduler weight sync, CUDA graph, etc.)

**verl target:** Relevant for the weight sync phase (`drafter weights → rollout vLLM` in the RFC). The `_sync_draft_weights()` method shows how TorchSpec pushes updated drafter weights to the inference server for speculative decoding at rollout time.

**Priority:** P2 — needed after the training pipeline works, when integrating with speculative decoding rollout.

---

### 11. vLLM KV Connector (P0 — Primary HS Collection Path)

**File:** `inference/engine/mooncake_hidden_states_connector.py`

**What it does:** `KVConnectorBase_V1` subclass. Intercepts vLLM model forward during prefill and writes hidden states to Mooncake using the official KV Connector plugin interface. No patching of vLLM internals required.

**verl target:** Copy into verl as the primary HS collection mechanism. Register via `--kv-connector MooncakeHiddenStatesConnector` when launching the vLLM-backed HS collector worker.

> **Warning — pre-norm last_hidden_states:** vLLM's KV Connector receives `last_hidden_states` *before* the final RMSNorm. The Eagle3 forward KL loss fuses RMSNorm + lm_head, so this must be applied at training time or the connector must be configured to return post-norm states.

---

### 12. Data Types

**Files:**
- `utils/types.py` — `InferenceInput`, `InferenceOutput`
- `training/data_fetcher.py` — `TrainSample` (not in types.py)

**Key types:**
```python
# torchspec/utils/types.py
InferenceInput:  data_id, prompt, input_ids, packed_loss_mask, metadata
InferenceOutput: data_id, mooncake_key, tensor_shapes, tensor_dtypes, packed_loss_mask

# torchspec/training/data_fetcher.py
TrainSample:     mooncake_key, tensor_shapes, tensor_dtypes, packed_loss_mask
```

**verl target:** Maps to `SequenceMeta` / `SampleMeta` in the RFC, packed into `DataProto.non_tensor_batch`

---

### 13. Config: `torchspec/config/`

**Files:**
- `mooncake_config.py` — Mooncake connection params (protocol, RDMA device, buffer sizes)
- `inference_config.py` — Engine config (batch size, num engines, TP, aux layers)
- `train_config.py` — Training config (LR, warmup, accumulation, checkpoint)

**verl target:** Extend verl's existing config (yaml) with drafter/mooncake sections

---

## Priority Order for Migration

| Priority | Component | TorchSpec Source | verl Target | Complexity |
|---|---|---|---|---|
| **P0** | vLLM engine + KV Connector | `inference/engine/mooncake_hidden_states_connector.py` | Copy into verl, plug via `--kv-connector` | Low — no patching needed |
| **P0** | Mooncake store | `transfer/mooncake/` | New verl module | Medium — copy mostly as-is |
| **P1** | Eagle3 model & loss | `models/eagle3.py`, `models/ops/loss.py`, `models/draft/` | FSDPDrafterEngine + Forward KL loss | Medium |
| **P1** | HS Collector worker | `inference/engine/mooncake_hidden_states_connector.py` | New BaseRollout subclass (vLLM) | Medium |
| **P1** | DrafterDataController | `controller/training_controller.py` | New class on driver | Low |
| **P2** | Decode-side weight sync | `inference/engine/sgl_engine_decode.py` | Weight sync for SD rollout | Medium |
| **P2** | Orchestration | `controller/loop.py` | RayPPOTrainer.fit() additions | Low |
| **P2** | Data fetching | `training/data_fetcher.py` | Folded into update_drafter() | Low |
| **P3** | Config | `config/` | verl yaml extensions | Low |
| **Skip** | InferenceManager | `controller/inference_manager.py` | Not needed (sync model) | — |
| **Skip** | SGLang prefill patches | `patches/sglang/sglang.patch` | Using vLLM instead | — |
| **Skip** | SGLang decode patches | `patches/sglang/sglang_decode.patch` | Using vLLM instead | — |

## Files to Copy vs Rewrite

**Copy/adapt (minimal changes):**
- `transfer/mooncake/eagle_store.py` — core Mooncake interface
- `transfer/mooncake/buffers.py` — buffer pool management
- `transfer/mooncake/deferred_delete.py` — async cleanup
- `transfer/mooncake/helpers.py`, `utils.py` — dependencies of eagle_store
- `models/ops/loss.py` — Forward KL loss kernels (torch.compiled)
- `models/draft/` — draft model architectures
- `config/mooncake_config.py` — connection config
- `inference/engine/mooncake_hidden_states_connector.py` — vLLM KV Connector for HS capture (copy as-is, no patching required)

**Rewrite for verl patterns:**
- `controller/training_controller.py` → `DrafterDataController` (sync, in-process, DataProto)
- `inference/engine/sgl_engine.py` → HS Collector (BaseRollout subclass wrapping vLLM, sleep/wake)
- `models/eagle3.py` + `training/eagle3_trainer.py` → `FSDPDrafterEngine` + `eagle_draft_loss`
- `training/data_fetcher.py` → folded into `update_drafter()`

**Skip entirely:**
- `controller/inference_manager.py` — async polling loop, not needed
- `controller/loop.py` — verl has its own training loop
- `patches/sglang/sglang.patch` — SGLang prefill patches, using vLLM instead
- `patches/sglang/sglang_decode.patch` — SGLang decode patches, using vLLM instead
