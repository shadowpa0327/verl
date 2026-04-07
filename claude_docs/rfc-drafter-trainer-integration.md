# RFC: Drafter Trainer Integration in verl

## Goal

Add EAGLE drafter co-training to verl's RL pipeline. The drafter trains on hidden states extracted from the target model via **dedicated vLLM inference workers** (separate from rollout).

**Key design decisions**:
- Hidden states collected by **dedicated vLLM workers** using vLLM's `extract_hidden_states` speculative config + a custom `MooncakeHiddenStatesConnector` (KV connector). No SGLang patching needed.
- **3-level data pipeline** mirroring TorchSpec: raw_prompts → sample_pool → train_queues
- **Mooncake** KV store for heavy tensor transport (hidden states), connected via the `MooncakeHiddenStatesConnector`
- **No gating** — drafter trains every step on whatever data is in its queue
- Dedicated workers **sleep/wake** with GPU time-multiplexing on the same GPUs as rollout
- **DrafterDataController lives on the driver** (RayPPOTrainer), mirroring verl's single-controller pattern
- **Pre-norm trade-off**: vLLM captures `last_hidden_states` before the final RMSNorm (pre-norm). This is a known trade-off of using vLLM's built-in extraction path.

---

## 3-Level Data Pipeline

Mirrors TorchSpec's controller hub stores. The control plane routes lightweight metadata (keys, shapes); heavy tensors (hidden states) flow through Mooncake.

```
════════════════════════════════════════════════════════════════════
                      3-Level Data Pipeline
════════════════════════════════════════════════════════════════════

  Level 1: raw_prompts                           ← on driver (global)
  ┌─────────────────────────────────────────────────────────────┐
  │ Complete sequences from rollout (prompt + response)          │
  │                                                              │
  │ Filled by: generate_sequences() output                      │
  │ Drained by: driver sends to HS Collector workers            │
  │ Contents: input_ids, attention_mask, seq metadata            │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             │  HS Collector: prefill-only fwd pass
                             │    → hidden state tensors to Mooncake
                             │    → metadata + keys returned to driver
                             ▼
  Level 2: sample_pool (global)                  ← on driver (global)
  ┌─────────────────────────────────────────────────────────────┐
  │ Training-ready samples (metadata only — tensors in Mooncake) │
  │                                                              │
  │ Filled by: HS Collector results returned to driver           │
  │ Drained by: driver.dispatch()                                │
  │ Contents: mooncake_key, shapes, dtypes, seq_len, n_tokens   │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             │  Driver partitions across DP ranks
                             │  and dispatches to workers
                             ▼
  Level 3: train_queues[0..N-1] (per-DP-rank)   ← dispatched to workers
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │rank 0│ │rank 1│ │rank 2│ │...   │
  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
     │        │        │        │
     ▼        ▼        ▼        ▼
  Drafter training workers
  (fetch tensors from Mooncake by key → train_batch → delete key)
```

### What Lives Where

| Level | Location | Control plane (metadata) | Data plane (tensors) |
|---|---|---|---|
| raw_prompts | **Driver** (global) | input_ids, attention_mask, lengths | (tokens are lightweight, inline) |
| sample_pool | **Driver** (global) | mooncake_key, shapes, dtypes, seq_len | Hidden states in Mooncake |
| train_queues[i] | **Worker rank i** (local) | Same metadata, partitioned per rank | Same Mooncake keys |

**Levels 1 and 2 are global on the driver** — the driver has the full view of all data. Level 3 is per-rank, dispatched to workers the same way verl dispatches DataProto via `make_nd_compute_dataproto_dispatch_fn`.

### Comparison with TorchSpec

| TorchSpec | verl Drafter | Difference |
|---|---|---|
| `_stored_dataset` | (verl's own dataset handling) | N/A |
| `prompt_buffer` | **raw_prompts** | Contains complete sequences, not raw prompts |
| `sample_pool` | **sample_pool** | Same — metadata + Mooncake keys |
| `train_queues[i]` | **train_queues[i]** | Same — per-DP-rank partitioned |
| `AsyncTrainingController` (CPU-only Ray actor) | **DrafterDataController** (on driver) | Sync, in-process on driver |
| InferenceManager (async, continuous) | HS Collector (sync, wake after rollout) | Sync vs async |
| Mooncake KV store | Mooncake KV store | Same |

---

## DrafterDataController — Lives on the Driver

### Why on the Driver (Not Inside Workers)

In verl, `RayPPOTrainer` is already the **single controller** — it orchestrates all workers via RPC. The `DrafterDataController` extends this pattern:

```
verl's existing single-controller pattern:

  RayPPOTrainer (driver, CPU)
  │
  │── orchestrates actor_rollout_wg via RPC
  │── dispatches DataProto to workers (splits along DP dim)
  │── collects results from workers
  │── runs compute_advantage() on driver
  │
  └── NEW: DrafterDataController
      │── owns raw_prompts (Level 1, global)
      │── owns sample_pool (Level 2, global)
      └── dispatches train batches to workers (Level 3, per-rank)
```

**Why this is the right placement:**

1. **Levels 1 & 2 are global** — they hold ALL data across all ranks. Only the driver has this global view. Putting them inside workers would mean N redundant copies or awkward cross-worker coordination.

2. **Dispatch is the driver's job** — verl already splits data across DP ranks on the driver. `dispatch()` (Level 2 → Level 3) is the same pattern: partition and send to workers.

3. **Workers are pure compute** — HS collector runs prefill, drafter runs train_batch. They receive data and return results. No coordination logic.

4. **Mirrors TorchSpec** — `AsyncTrainingController` is a separate CPU-only actor, not inside GPU workers. Our controller is in-process on the driver (simpler, since we're sync).

5. **Mirrors verl convention** — `compute_advantage()` already runs on the driver, not on workers. The drafter data pipeline is analogous.

### Controller Diagram

```
╔══════════════ DrafterDataController (on RayPPOTrainer) ══════════════╗
║                                                                       ║
║  Level 1: raw_prompts (global)    ← push: driver after rollout        ║
║  ┌───────────────────────┐        ← pull: driver sends to HS workers  ║
║  │ list[SequenceMeta]    │                                            ║
║  └────────┬──────────────┘                                            ║
║           │                                                           ║
║  Level 2: sample_pool (global)    ← push: driver after HS collection  ║
║  ┌───────────────────────┐        ← drain: driver.dispatch()          ║
║  │ list[SampleMeta]      │                                            ║
║  └────────┬──────────────┘                                            ║
║           │                                                           ║
║           │  dispatch() partitions and returns per-rank batches        ║
║           │  → driver dispatches to workers via RayWorkerGroup         ║
║           ▼                                                           ║
║  (Level 3 lives on workers, not here)                                 ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Interface

```python
@dataclass
class SequenceMeta:
    input_ids: Tensor       # lightweight, kept inline
    attention_mask: Tensor
    prompt_len: int
    response_len: int

@dataclass
class SampleMeta:
    mooncake_key: str
    shapes: dict[str, tuple]
    dtypes: dict[str, torch.dtype]
    seq_len: int
    n_tokens: int

class DrafterDataController:
    """
    Lives on the driver (RayPPOTrainer). Owns Levels 1 & 2.
    Mirrors TorchSpec's AsyncTrainingController (sync, in-process).

    Level 3 (train_queues) does not exist here — verl's mesh-based dispatch
    splits the DataProto per rank automatically when update_drafter() is called.
    """

    def __init__(self, dp_size: int):
        self._raw_prompts: list[SequenceMeta] = []    # Level 1 (global)
        self._sample_pool: list[SampleMeta] = []      # Level 2 (global)
        self._dp_size = dp_size

    # ── Level 1: raw_prompts ──────────────────────────────────

    def push_raw_prompts(self, sequences: list[SequenceMeta]):
        """Called by: driver after generate_sequences(). Fills Level 1."""
        self._raw_prompts.extend(sequences)

    def pull_raw_prompts(self) -> list[SequenceMeta]:
        """Called by: driver, to send to HS collector workers. Drains Level 1."""
        batch = self._raw_prompts
        self._raw_prompts = []
        return batch

    # ── Level 2: sample_pool ──────────────────────────────────

    def push_samples(self, samples: list[SampleMeta]):
        """Called by: driver after HS collection returns. Fills Level 2."""
        self._sample_pool.extend(samples)

    # ── Drain Level 2 → DataProto for dispatch ────────────────

    def drain_as_dataproto(self) -> DataProto:
        """
        Pack all samples into DataProto.non_tensor_batch for mesh dispatch.
        The dispatch fn on update_drafter() splits this per DP rank automatically.
        No manual partitioning needed — verl handles it.

        non_tensor_batch values must be np.ndarray(dtype=object) —
        DataProto.chunk() uses np.array_split() on axis 0.
        """
        proto = DataProto(
            batch=TensorDict({}),
            non_tensor_batch={
                'mooncake_keys': np.array([m.mooncake_key for m in self._sample_pool], dtype=object),
                'shapes': np.array([m.shapes for m in self._sample_pool], dtype=object),
                'dtypes': np.array([m.dtypes for m in self._sample_pool], dtype=object),
                'seq_lens': np.array([m.seq_len for m in self._sample_pool], dtype=object),
            },
        )
        self._sample_pool = []
        return proto
```

### Data Flow: Driver ↔ Workers

```
RayPPOTrainer (driver)                           Workers (GPU)
─────────────────────                            ──────────────

  controller.push_raw_prompts(seqs)
       │
       │── raw_prompts = controller.pull_raw_prompts()
       │── send to HS collector workers ──────────→  HS Collector:
       │                                              prefill fwd pass
       │                                              Mooncake.put(key, hs)
       │◄── receive sample metadata ──────────────  return SampleMeta list
       │
       │── controller.push_samples(metadata)
       │── per_rank = controller.dispatch()
       │── send per_rank[i] to worker i ──────────→  Worker rank i:
       │   (see "Dispatch Mechanism" below)           receives SampleMeta list
       │                                              stores in local train_queue
       │
       │── tell workers: update_drafter() ────────→  Drafter:
       │                                              pop from train_queue
       │                                              Mooncake.get(keys)
       │                                              train_batch()
       │                                              Mooncake.remove(keys)
```

### Dispatch Mechanism: How per-rank Data Reaches Workers

Pack `SampleMeta` into `DataProto.non_tensor_batch`, register a "drafter" mesh (pure DP), use verl's standard `make_nd_compute_dataproto_dispatch_fn`. Same pattern as `compute_log_prob` — just a different mesh name.

Dispatch is folded into `update_drafter` directly: one driver call that dispatches + trains.

```python
# ── Worker: register drafter mesh at init ──
# Pure DP — every rank is a unique DP rank
self._register_dispatch_collect_info(
    mesh_name="drafter",
    dp_rank=dist.get_rank(),      # world_rank = dp_rank (pure DP)
    is_collect=True,
)

# ── Worker: update_drafter receives per-rank DataProto ──
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
def update_drafter(self, data: DataProto):
    """data is already this rank's shard — dispatch fn split it."""
    if self.drafter is None:
        return
    meta_batch = unpack_sample_meta(data)
    for step in range(self._drafter_max_steps):
        batch = meta_batch[step * bs : (step + 1) * bs]
        if not batch:
            break
        tensors = mooncake.get_batch(
            keys=[m.mooncake_key for m in batch],
            shapes=[m.shapes for m in batch],
            dtypes=[m.dtypes for m in batch],
        )
        self.drafter.train_batch(data=tensors)
        mooncake.remove_batch([m.mooncake_key for m in batch])
```

The dispatch function (`make_nd_compute_dataproto_dispatch_fn`) handles everything:
1. Queries each worker's `dp_rank` in the "drafter" mesh
2. Calls `proto.chunk(dp_size)` to split into per-rank shards
3. Routes `chunk[dp_rank]` to each worker

```python
# ── Driver: drain controller → DataProto → mesh dispatch ──
drafter_proto = self._drafter_ctrl.drain_as_dataproto()
# dispatch fn calls proto.chunk(dp_size), sends chunk[i] to rank i
# each worker: unpack non_tensor_batch → Mooncake.get → train_batch
actor_rollout_wg.update_drafter(drafter_proto)
```

---

## RL Step Timeline

The drafter sub-pipeline (HS collection → dispatch → training) runs as a **contiguous block** right after rollout, before actor training. Drafter training does not depend on actor training.

```
Time ──────────────────────────────────────────────────────────────►

├─ RL step ───────────────────────────────────────────────────────┤

┌───────────────────┐
│ Rollout vLLM     │  generate_sequences()
│ (on GPU)           │  → driver: controller.push_raw_prompts()
└──────┬────────────┘
       │ rollout sleeps, vLLM HS collector wakes
       ▼
┌──────────────────────────┐
│ HS Collector vLLM         │  driver sends raw_prompts → workers
│ (on GPU, dedicated)       │  → extract_hidden_states speculative config
│                           │  → MooncakeHiddenStatesConnector.put(key, hs)
│                           │  → return sample metadata to driver
└──────┬───────────────────┘
       │ HS collector sleeps
       │ driver: controller.push_samples() + dispatch()
       │ driver: send per-rank batches to workers
       ▼
┌──────────┐
│ Drafter  │  pop from local train_queue
│ (on GPU) │  Mooncake.get(keys) → train_batch()
│          │  Mooncake.remove(keys)
└──────┬───┘
       │ drafter offloads, actor loads
       ▼
┌──────────┐
│  Actor   │  compute_log_prob()
└──────────┘
         ┌──────────┐
         │  Actor   │  update_actor()
         └──────────┘
                  ┌─────────────┐
                  │ Weight Sync │  actor + drafter → rollout + HS collector
                  └─────────────┘
```

### Why Drafter Before Actor

- Drafter training only depends on HS collection + dispatch — **no dependency on actor training**
- Makes the drafter sub-pipeline contiguous: rollout → HS → dispatch → train → done
- Actor training is a separate contiguous block after
- Weight sync at the end pushes both actor + drafter weights to rollout

---

## Data Lifecycle: 3 Phases

### Phase 1: Rollout → raw_prompts (Level 1, on driver)

`generate_sequences()` produces complete sequences. Driver pushes them into the controller.

```
Driver                              DrafterDataController
  │                                       │
  │── sequences = wg.generate_sequences() │
  │── controller.push_raw_prompts(seqs) ─→│  Level 1 fills (global)
```

### Phase 2: raw_prompts → HS Collector → sample_pool (Level 1 → Level 2, on driver)

Driver pulls from Level 1, sends to HS collector workers. Workers run prefill, write to Mooncake, return metadata. Driver pushes metadata into Level 2.

```
Driver                  HS Collector (GPU)        Mooncake     DrafterDataController
  │                           │                      │                │
  │── pull_raw_prompts() ────────────────────────────────────────────│ Level 1 drains
  │── send to workers ──────→│                      │                │
  │                           │── prefill fwd pass   │                │
  │                           │── put(key, hs) ─────→│ store tensors  │
  │◄── sample metadata ──────│                      │                │
  │                                                                   │
  │── controller.push_samples(metadata) ─────────────────────────────→│ Level 2 fills
```

### Dispatch: sample_pool → train_queues (Level 2 → Level 3, driver → workers)

Driver partitions and dispatches. Same pattern as verl's existing DP data dispatch.

```
Driver                              DrafterDataController     Workers
  │                                       │                     │
  │── per_rank = controller.dispatch() ──│ Level 2 drains      │
  │                                                              │
  │── send per_rank[0] to worker 0 ─────────────────────────────→│ rank 0 train_queue
  │── send per_rank[1] to worker 1 ─────────────────────────────→│ rank 1 train_queue
  │── ...                                                        │
```

### Phase 3: train_queues → Drafter Training (Level 3 on workers → GPU)

Each drafter training worker pops from its local train_queue, fetches from Mooncake, trains, cleans up. No driver involvement during training.

```
Worker rank i                   local train_queue     Mooncake
     │                              │                    │
     │── pop meta batch ───────────│                    │
     │                                                   │
     │── Mooncake.get(keys, shapes, dtypes) ────────────→│
     │◄── tensors on GPU ──────────────────────────────  │
     │                                                   │
     │── drafter.train_batch(tensors)                    │
     │                                                   │
     │── Mooncake.remove(keys) ─────────────────────────→│  free memory
```

---

## Dedicated HS Collector Workers

### Why Dedicated vLLM HS Collector (Not Actor Engine)

- **No patching required**: vLLM's `extract_hidden_states` speculative config is a built-in extraction path; the `MooncakeHiddenStatesConnector` plugs in as a standard KV connector. Zero SGLang source modifications.
- **Clean separation**: rollout (vLLM) does generation; HS collector (vLLM) does prefill-only extraction. The rollout also uses vLLM — only the HS collector pivoted to vLLM.
- **Different configuration**: vLLM HS collector is configured with `extract_hidden_states` speculative config and the custom `MooncakeHiddenStatesConnector`; no sampling/decode logic needed.
- **Independent lifecycle: HS collector and rollout vLLM instances have separate configs.
- **Pre-norm trade-off**: vLLM's built-in path captures `last_hidden_states` before the final RMSNorm (pre-norm). This is acceptable for drafter training.

### Sleep/Wake Lifecycle

The vLLM HS collector workers share GPUs with the rollout vLLM server via verl's existing sleep/wake mechanism:

```
RL Step Timeline (GPU perspective):

  ┌──────────────────────────────┐
  │ Rollout vLLM: AWAKE        │  generate_sequences()
  │ HS Collector vLLM: SLEEPING  │
  └──────────────────────────────┘
           │ rollout.release() → vLLM HS collector.resume()
           ▼
  ┌──────────────────────────────┐
  │ Rollout vLLM: SLEEPING     │
  │ HS Collector vLLM: AWAKE     │  collect_hidden_states()
  └──────────────────────────────┘
           │ HS collector.release() → dispatch (CPU) → drafter loads
           ▼
  ┌──────────────────────────────┐
  │ Drafter Engine: AWAKE        │  update_drafter()
  └──────────────────────────────┘
           │ drafter offloads → actor engine loads
           ▼
  ┌──────────────────────────────┐
  │ Actor Engine:   AWAKE        │  compute_log_prob() + update_actor()
  └──────────────────────────────┘
```

### Dispatch: How HS Collector Receives Sequences

Reuses the "rollout" mesh — same DP topology, same data distribution:

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def collect_hidden_states(self, sequences: DataProto):
    """sequences is already this rank's shard — same split as generate_sequences."""
    for seq in sequences:
        # vLLM runs prefill with extract_hidden_states config;
        # MooncakeHiddenStatesConnector.put() writes hs to Mooncake internally
        key = self.hs_collector.extract(seq.input_ids, seq.attention_mask)
        sample_meta.append(SampleMeta(mooncake_key=key, shapes=..., ...))
    return pack_as_dataproto(sample_meta)  # collect fn gathers back to driver
```

Driver calls it exactly like `generate_sequences`: pass full DataProto, dispatch fn splits per rank, collect fn gathers results back.

### vLLM HS Extraction (No Patching)

vLLM's `extract_hidden_states` speculative config triggers prefill-only extraction:

```
vLLM Model Forward (extract_hidden_states speculative config):

  input_ids → embed_tokens → decoder_layers[0..N-1] → last_hidden_states (pre-norm)
                                                                │
                                                                ▼
                                                  MooncakeHiddenStatesConnector
                                                         │
                                                         ▼
                                                  Mooncake.put(key, hs)
```

- No sampling or generation; prefill-only pass
- `last_hidden_states` are captured **before** the final RMSNorm (pre-norm) — accepted trade-off
- `MooncakeHiddenStatesConnector` is a custom KV connector; no vLLM source patching required
- Returns only metadata (key, shapes) to the driver

---

## Worker Hierarchy

```
RayPPOTrainer (driver, CPU)
├── self._drafter_ctrl   (DrafterDataController)               [NEW — owns Levels 1 & 2]
│
└── actor_rollout_wg (RayWorkerGroup) ── single GPU pool
    └── ActorRolloutRefDrafterWorker (extends ActorRolloutRefWorker)
            ├── self.actor           (TrainingWorker → FSDPEngine)       [inherited]
            ├── self.ref             (TrainingWorker → FSDPEngine)       [inherited]
            ├── self.rollout         (BaseRollout → vLLM)              [inherited]
            ├── self.hs_collector    (VllmHSCollector)                   [NEW — vLLM with KV connector]
            └── self.drafter         (TrainingWorker → FSDPDrafterEngine) [NEW]
```

The **controller lives on the driver**, not inside workers. Workers have no local queue — Level 3 dispatch is handled by verl's mesh-based `make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")`, which splits the DataProto per rank when `update_drafter()` is called. `self.rollout` is a vLLM instance; `self.hs_collector` is a `VllmHSCollector` instance (vLLM with `extract_hidden_states` config + `MooncakeHiddenStatesConnector`). Both share the same GPU via sleep/wake time-multiplexing.

---

## Orchestration

```python
# RayPPOTrainer.fit(), per RL step:

# ── Phase 1: Rollout → raw_prompts (driver Level 1) ──
sequences = actor_rollout_wg.generate_sequences(prompts)
self._drafter_ctrl.push_raw_prompts(sequences)

# ── Phase 2: raw_prompts → HS collection → sample_pool (driver Level 2) ──
raw = self._drafter_ctrl.pull_raw_prompts()
# (rollout sleeps, HS collector wakes internally)
sample_metadata = actor_rollout_wg.collect_hidden_states(raw)  # → prefill → Mooncake
self._drafter_ctrl.push_samples(sample_metadata)               # results back to driver

# ── Dispatch + Drafter training (before actor!) ──
# Pack all samples into DataProto; mesh dispatch splits per rank automatically
drafter_proto = self._drafter_ctrl.drain_as_dataproto()
actor_rollout_wg.update_drafter(drafter_proto)    # dispatch fn → chunk[i] to rank i
                                                   # each worker: unpack → Mooncake → train

# ── Actor training ──
actor_rollout_wg.compute_log_prob(sequences)
compute_advantage(...)
actor_rollout_wg.update_actor(sequences)

# ── Weight sync ──
actor_rollout_wg.update_weights()  # actor + drafter → rollout + HS collector
```

---

## Mooncake Integration

### Role in the Pipeline

Mooncake is the **data plane** — it holds the heavy tensors (hidden states) and provides location-transparent access. The DrafterDataController (on the driver) and 3-level pipeline only route lightweight metadata (keys). This directly mirrors TorchSpec's architecture.

### Operations

| Operation | Who | When |
|---|---|---|
| `put(key, hidden_states)` | `MooncakeHiddenStatesConnector` (inside vLLM HS Collector) | After prefill fwd pass, via vLLM KV connector hook |
| `get(key, shapes, dtypes)` | Drafter training worker | Before train_batch() |
| `remove(key)` | Drafter training worker | After consuming tensors |

The driver never touches Mooncake. Tensors flow directly between GPU workers via Mooncake.

---

## Drafter Training (Consumption)

No gating, no local queue. The driver packs all samples into DataProto, mesh dispatch splits per rank, and `update_drafter` receives only this rank's shard directly.

```python
# On the worker (see "Dispatch Mechanism" section for full code):
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
def update_drafter(self, data: DataProto):
    """data is already this rank's shard — dispatch fn split it."""
    if self.drafter is None:
        return
    meta_batch = unpack_sample_meta(data)
    for step in range(self._drafter_max_steps):
        batch = meta_batch[step * bs : (step + 1) * bs]
        if not batch:
            break
        tensors = mooncake.get_batch(
            keys=[m.mooncake_key for m in batch],
            shapes=[m.shapes for m in batch],
            dtypes=[m.dtypes for m in batch],
        )
        self.drafter.train_batch(data=tensors)
        mooncake.remove_batch([m.mooncake_key for m in batch])
```

---

## Weight Sync

```
update_weights():
  actor weights   → rollout vLLM          (existing)
  actor weights   → HS collector vLLM       (same GPU, update on wake)
  drafter weights → rollout vLLM          (for speculative decoding)
```

The vLLM HS collector must have the actor's current weights to produce consistent hidden states.

---

## FSDPDrafterEngine

```python
@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPDrafterEngine(FSDPEngine):
    """
    Engine for drafter models (EAGLE).
    Tiny (~2% of target params): fc + 1 decoder layer.
    Borrows embed_tokens/lm_head from actor (frozen, shared refs).
    """

    def set_shared_modules(self, embed_tokens, lm_head):
        """Set shared frozen modules from target model."""
        ...

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Expects: input_ids, hidden_states (fetched from Mooncake)."""
        ...
```

---

## Open Questions

1. **Mooncake setup**: How to initialize Mooncake store in verl's existing infrastructure? Per-node or global?

2. **Drafter DP mesh**: The drafter trains as pure DP. How to register its dispatch mesh for the per-rank queue consumption?

3. **embed_tokens / lm_head sharing**: Drafter borrows these from actor. After `update_actor()`, verify FSDP doesn't invalidate the shared reference.

4. **MooncakeHiddenStatesConnector interface**: Pin down the exact vLLM KV connector API (`send_kv_caches_and_hidden_states` / `recv_kv_caches_and_hidden_states`) and confirm the `extract_hidden_states` speculative config is sufficient to bypass sampling entirely.

5. **Pre-norm impact**: Assess whether using pre-RMSNorm `last_hidden_states` (vLLM trade-off) degrades drafter acceptance rates vs post-norm hidden states.

6. **Buffer sizing**: How large should raw_prompts and sample_pool be? Bounded with eviction, or sized exactly for one RL step's worth of data?

---

## Appendix

### EAGLE Model Architecture

```
┌─────────────────┐                ┌─────────────────────────────────────┐
│  Target Model   │                │         EAGLE Drafter               │
│                 │                │                                     │
│  embed_tokens ◄═══ shared ══════►  embed_tokens   (frozen)            │
│                 │                │       │                             │
│  N decoder      │                │       ▼                             │
│  layers ─────────── hs ─────────►  fc([embed, hs])  *trainable        │
│                 │                │       │                             │
│                 │                │       ▼                             │
│                 │                │  1 decoder layer  *trainable        │
│                 │                │       │                             │
│  lm_head ◄══════ shared ════════►  lm_head   (frozen)                 │
│                 │                │       │                             │
│                 │                │       ▼                             │
│                 │                │  next token logits                  │
└─────────────────┘                └─────────────────────────────────────┘

◄══►  shared weights (frozen)
────►  data flow (hidden states)
```

### verl Workers Architecture (Baseline)

```
ActorRolloutRefWorker                          ← Layer 3: composes multiple workers
├── self.actor    = TrainingWorker(...)         ← Layer 2: micro-batching, loss injection
│                 └── .engine = FSDPEngineWithLMHead  ← Layer 1: fwd/bwd/opt
├── self.ref      = TrainingWorker(...)
│                 └── .engine = FSDPEngineWithLMHead
└── self.rollout  = vLLMRollout(...)         ← Layer 1b: token generation
```

### verl's Single Controller Pattern

verl uses a "single controller" architecture where `RayPPOTrainer` on the driver process orchestrates all workers:

```
RayPPOTrainer (driver, CPU-only)
│
├── Orchestrates via RPC:
│   actor_rollout_wg.generate_sequences()
│   actor_rollout_wg.compute_log_prob()
│   actor_rollout_wg.update_actor()
│
├── Runs on driver directly:
│   compute_advantage()          ← driver-side computation
│   DrafterDataController        ← driver-side coordination (NEW)
│
└── Dispatches data:
│   DataProto split along DP dim → workers    (existing)
│   SampleMeta split per rank → workers       (NEW, same pattern)
```

The `DrafterDataController` follows this convention: coordination and global state on the driver, pure computation on workers.
