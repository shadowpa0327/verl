# Drafter Co-Training — As-Built Design

Concise architecture reference for the drafter co-training pipeline as
shipped. For per-component details:

- `weight-sync-flows.md` — parameter inventory + 4 sync flows
- `torchspec-to-verl-migration-map.md` — TorchSpec ↔ recipe mapping
- `migration-status.md` — current state, latest verification, open TODOs
- `project-guide.md` — verl architecture + recipe-first workflow rule
- `workflow-orchestration.md` — task/plan rules

---

## Goal

Add EAGLE3 speculative decoding drafter co-training to verl's RL loop.
The drafter trains on hidden states extracted from the target model via
a colocated vLLM replica that uses vLLM's `extract_hidden_states`
speculative config + a custom `MooncakeHiddenStatesConnector`. No vLLM
or SGLang patching.

---

## Top-level architecture

```
RayPPOTrainer (driver, CPU)
├── DrafterDataController                ← on driver; owns Levels 1 & 2
│     raw_prompts → sample_pool → drain_as_dataproto()
│
├── HSCollectorManager                   ← on driver; colocated vLLM replicas
│     (clone of TeacherModelManager; sleep/wake-multiplexed with rollout)
│
└── actor_rollout_wg (RayWorkerGroup, GPU)
    └── ActorRolloutRefDrafterWorker     ← extends ActorRolloutRefWorker
            ├── actor       (FSDPEngine — RL training)            [inherited]
            ├── ref         (FSDPEngine — log_prob)               [inherited]
            ├── rollout     (vLLM async server — generation)      [inherited]
            └── drafter     (FSDPDrafterEngine → Eagle3Model)     [NEW]
```

**Key placements:**
- `DrafterDataController` and `HSCollectorManager` live **on the driver**
  (RayPPOTrainer). Workers are pure compute; controllers route metadata
  and orchestrate sleep/wake. Mirrors verl's existing single-controller
  convention (`compute_advantage` also runs on the driver).
- vLLM rollout and HS-collector replicas **share GPUs** with the trainer
  workers, time-multiplexed via verl's `RolloutReplica` sleep/wake
  (`HSCollectorManager` clones `TeacherModelManager` for this).
- All drafter library code lives in the recipe submodule:
  `recipe/drafter_cotraining/`. Parent `verl/` has only one carve-out
  (`vllm_async_server.py` 3-line `kv_transfer_params` propagation) plus
  smoke wrappers.

---

## Data lifecycle (2 stores + mesh dispatch)

The control plane routes lightweight metadata (Mooncake keys, shapes,
dtypes, lengths). Heavy tensors (hidden states) flow directly through
Mooncake.

```
Level 1: raw_prompts                              ← driver, global
   list[SequenceMeta(input_ids, attention_mask, prompt_len, response_len)]
        │  push: trainer after generate_sequences()
        │  pull: trainer feeds HS collector
        ▼
HSCollectorManager.compute_hidden_states(batch)   ← driver, sync
   - vLLM colocated replicas wake, run prefill-only
   - MooncakeHiddenStatesConnector.put(key, hidden_states + last_hidden_states)
   - returns DataProto with non_tensor_batch:
       hs_mooncake_keys, hs_shapes, hs_dtypes, hs_seq_lens,
       hs_prompt_lens, hs_response_lens
        │
        ▼
Level 2: sample_pool                              ← driver, global
   list[SampleMeta(mooncake_key, shapes, dtypes, seq_len,
                   n_tokens, prompt_len, response_len)]
        │  push: trainer (from compute_hidden_states output)
        │  drain: drain_as_dataproto() → DataProto.non_tensor_batch
        ▼
Drafter mesh dispatch                             ← verl handles, per-rank
   make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")
   chunks DataProto along sample axis; routes chunk[i] to rank i
        │
        ▼
update_drafter(data) per worker                   ← GPU, per rank
   1. Mooncake.get(key) for each sample → ids, hidden_states, last_hidden_states
   2. Build response-only loss_mask = [0]*plen + [1]*(rlen-1) + [0]
   3. Eagle3Collator pads to [B, T_pad] / [B, T_pad, D]
   4. prepare_model_inputs (left-shift input_ids + last_hidden_states by 1)
   5. Eagle3Model.forward (7-step TTT)
   6. 0.8^i weighted backward
   7. optimizer_step + lr_scheduler_step
   8. all-reduce metrics over DP group
   9. Mooncake.remove_eagle3_tensors(key) per sample
```

`SampleMeta` and `DrafterDataController` definitions:
`recipe/drafter_cotraining/controller.py`.

---

## RL step timeline

Drafter training runs as a contiguous block right after rollout, before
actor training. Drafter has no dependency on actor training.

```
Time ──────────────────────────────────────────────────────────────────►

generate_sequences()                                rollout vLLM AWAKE
       │                                            HS-collector vLLM SLEEPING
       │ rollout.release() → hs_collector.resume()
       ▼
HSCollectorManager.compute_hidden_states()          HS-collector AWAKE
       │   Mooncake.put(key, hs)
       │ hs_collector.release()
       ▼
DrafterDataController.push_samples + drain          driver-only
       │
       ▼
actor_rollout_wg.update_drafter(drafter_proto)      drafter AWAKE
       │   per rank: Mooncake.get → forward → backward → optimizer.step → Mooncake.remove
       │
       ▼
compute_log_prob → compute_advantage                actor AWAKE
       │
       ▼
update_actor                                        actor weights change
       │
       ▼
update_weights()                                    sync flows fire (see weight-sync-flows.md)
```

Why drafter before actor:
- Drafter only depends on HS collection + dispatch — independent of `update_actor`.
- Keeps the drafter sub-pipeline as a contiguous block (rollout → HS → train → done).
- Weight sync at step end pushes both actor + drafter to rollout/HS.

---

## HS collection — one mechanism, no patching

vLLM's `extract_hidden_states` speculative config triggers a prefill-
only forward pass. Our custom `MooncakeHiddenStatesConnector` (a
`KVConnectorBase_V1` implementation) hooks into that pass and writes
the captured layer outputs to Mooncake. No vLLM source modifications,
no SGLang patches.

Captured tensors per sample:
- `hidden_states` — concatenation of N-1 aux layer outputs
  (shape `[T, (N-1)·D]`). For Qwen3-4B with default
  `aux_hidden_state_layer_ids=[2, 18, 33, 35]` that's 3 aux layers.
- `last_hidden_states` — final layer output, **pre-norm**
  (shape `[T, D]`). Trainer applies `verifier_norm` before computing
  target logits — see `weight-sync-flows.md` "Why verifier_norm matters".
- `input_ids` — token IDs corresponding to those positions.
- `prompt_len`, `response_len` — boundaries for the response-only
  loss mask.

**Aux layer ID convention.** TorchSpec uses *post-layer N* semantics
(`[1,17,32,35]` = "after layers 1, 17, 32, 35"). vLLM's hook fires at
the *input* of each listed layer (= output of the previous layer), so
TorchSpec shifts non-final IDs by +1 and appends the final layer
separately. For Qwen3-4B (36 layers): `[1,17,32,35]` →
`[2,18,33,35]`. The wrapper script's `AUX_LAYER_IDS` default reflects
this.

---

## Drafter training step

`update_drafter` in `recipe/drafter_cotraining/engine_workers.py` runs
a paged-Mooncake-fetch + micro-batch accumulation loop (mirrors verl's
canonical `forward_backward_batch` divisor pattern):

1. **Empty-mask filter (metadata)** — drop samples with `rlen-1 <= 0`
   before any tensor fetch; eagerly free their Mooncake keys. Mirrors
   TorchSpec `data_fetcher.py:177`.
2. **Preflight `total_valid_global`** — sum `max(0, rlen-1)` across
   surviving samples and all-reduce SUM across the DP group. This is
   the divisor for the exact-mean per-micro-batch loss scaling.
3. **`T_pad_macro` precompute** — `max(prompt_lens + response_lens)`
   across all surviving samples on this rank; passed into the collator
   per micro-batch so all micro-batches share identical `T_pad`. This
   prevents `torch.compile` recompilation across micro-batches.
4. **`engine.train_mode` (one context for the whole macro-step)** —
   for each micro-batch:
   1. **Grad-sync suppression** — `set_requires_gradient_sync(is_last)`
      on the FSDP2 root suppresses inter-rank reduce-scatter on
      all-but-last micro-batch (mirrors TorchSpec's pattern; on FSDP1
      this attribute is missing → no-op via `getattr` guard).
   2. **Paged Mooncake fetch** — `_fetch_drafter_batch_from_mooncake`
      pulls just this micro-batch's keys, builds the response-only
      `loss_mask = [0]*plen + [1]*(rlen-1) + [0]` per sample, calls
      `Eagle3Collator(features, bucket_size_override=T_pad_macro)`,
      then `EagleMooncakeStore.remove_eagle3_tensors(key)` per key
      (eager cleanup).
   3. **Prepare** — `FSDPDrafterEngine.prepare_model_inputs`
      left-shifts `input_ids` + `last_hidden_states`, applies
      `verifier_norm`, builds a `PrecomputedTarget` (bf16-stored
      `target_p`; supports `t2d=None` no-pruning case via
      `compute_target_p_padded`).
   4. **Forward** — `Eagle3Model.forward(...)` runs the 7-step TTT
      loop and returns `(plosses[L], _, acces[L])`.
   5. **Weighted backward** — local `mb_valid` = `loss_mask.sum()`
      (or `position_mask.sum()` if pruning); scale =
      `mb_valid / total_valid_global`; backward
      `Σ_i 0.8^i · plosses[i] · scale`. Skip backward for
      `mb_valid == 0` (in-kernel zero-grad fallback already touches
      every param so reduce-scatter on the LAST mb still works).
5. **Optimizer step** — re-enable grad sync defensively, then
   `engine.optimizer_step()` (clips grad to `clip_grad`, skips on
   non-finite); `engine.lr_scheduler_step()`. Both inside the same
   `train_mode` context so `train_mode.__exit__` zeros grads.
6. **Aggregate metrics** — `_aggregate_micro_metrics` combines per-mb
   `plosses`/`acces` (weighted by `mb_valid`) and reuses
   `_aggregate_drafter_metrics` for the DP all-reduce + final dict.
   Surfaces `train/accum_steps`, `train/macro_valid_global`,
   `train/t_pad_macro` for verification.

`micro_batch_size_per_gpu` is configurable in
`drafter.engine_config.micro_batch_size_per_gpu` (default 1). Setting
it equal to `data.train_batch_size / world_size` recovers single-shot
behavior (one optimizer step per macro-batch).

---

## Engine choice — FSDP2 with selective wrap (TorchSpec-style)

The drafter engine uses FSDP2 (`torch.distributed.fsdp.fully_shard`)
with a selective wrap that mirrors TorchSpec's pattern: shard **only**
`LlamaDecoderLayer` sub-units; let `lm_head`, `model.norm`, `fc`, and
`embed_tokens` fall under the root `fully_shard` call. The override
lives in `FSDPDrafterEngine._build_fsdp_module` (drafter_engine.py).

**Why selective wrap (not verl's default `apply_fsdp2`):** verl's
`_select_fsdp2_wrap_targets` (`fsdp_utils.py:510-531`) wraps
`embed_tokens` and `lm_head` as their own FSDP units when
`tie_word_embeddings=False` (Qwen3-8B). The Eagle3 loss kernel reads
`lm_head.weight` as an extracted tensor (`F.linear(input, lm_head_w)`),
not via `lm_head.forward()` — so the sub-unit's pre-forward hook never
fires and the kernel sees a non-gathered DTensor.

**Why root keeps params gathered through backward:** PyTorch's
`fully_shard` auto-detects the root unit and forces its effective
`reshard_after_forward=False` regardless of what the caller passes
(see `set_reshard_after_forward` docstring in
`verl/utils/fsdp_utils.py:734-766`). So root-resident params
(`lm_head`, `norm`, `fc`, `embed_tokens`) are gathered once at root
forward entry and stay gathered through all 7 TTT steps + backward —
exactly what the compiled kernel needs. Sub-units (`LlamaDecoderLayer`)
default to `reshard_after_forward=True` for the actual memory saving.

**Why we dropped FSDP1 + `use_orig_params=True`:** FSDP1 worked but
required `use_orig_params=True` to handle the mixed-grad
(frozen embed + trainable rest) wrap. FSDP2 always behaves
orig-params-like, so the flag is moot. The compile-graph stability
issue with `lm_head.weight` (the original deterrent for FSDP2)
disappears once we shard selectively per the override above.

**Mirror of TorchSpec.** TorchSpec's `apply_fsdp2`
(`ref/TorchSpec/torchspec/training/fsdp.py:137-200`) shards
the individual `Linear` modules inside `midlayer`; we shard
`LlamaDecoderLayer` (one unit per decoder block). For a one-block
drafter the choice is just FSDP-unit granularity — equivalent
behavior, simpler code.

---

## Frozen weight loading — from `target_model_path`, not from actor

`FSDPDrafterEngine` loads three frozen weights from `target_model_path`
on every rank at init:

| Weight | Read from key | Lives at | Source order |
|---|---|---|---|
| `embed_tokens.weight` | `model.embed_tokens.weight` | `Eagle3Model.draft_model.embed_tokens` | `_build_module` (pre-FSDP wrap, on every rank). The draft *architecture* itself is auto-derived from `target_model_path` via `generate_draft_model_config` (`eagle3/draft/auto.py`) — `local_path` is an optional template overlay only needed for vocab pruning or non-Llama drafts. |
| `lm_head.weight` (`target_lm_head_weight`) | `lm_head.weight`, fallback `model.embed_tokens.weight` if `tie_word_embeddings=True` | `FSDPDrafterEngine._target_lm_head_weight` | `initialize` → `_load_target_frozen_weights` |
| `model.norm.weight` (`verifier_norm`) | `model.norm.weight` | `FSDPDrafterEngine._verifier_norm` (a `LlamaRMSNorm` module) | `initialize` → `_load_target_frozen_weights` |

This sidesteps the FSDP-sharded actor params at the cost of needing
`target_model_path` set (defaults to `${actor_rollout_ref.model.path}`
in the recipe YAML).

`sync_frozen_modules_from_actor` is currently a documented no-op stub
— properly implementing it requires FSDP-aware gather of the actor's
sharded params and isn't needed for the smoke (target frozen). See
`weight-sync-flows.md` Flow 3 for the eventual re-sync wiring.

---

## Verification

`scripts/run_drafter_training.sh` smokes the full pipeline. Expected
signal on a 16-step Qwen3-4B / 2× H100 run:
- `train/loss_weighted` decreases (~12.1 → ~7-8 over 16 steps).
- `train/simulated_acc_len` increases (0 → ~0.2).
- `train/grad_norm` finite throughout.
- No NaN/Inf, no Mooncake key leak.

Latest baseline numbers in `migration-status.md`. Detailed step-by-step
metrics live in commit messages on `feat/drafter-cotraining`.

---

## What's not built yet

- **Drafter → rollout weight sync** (TODO 4) — needs an
  `update_drafter_weights` API on the rollout side. TorchSpec's analog
  is `_maybe_sync_draft_weights` in `controller/loop.py:42-73`,
  activated only for the `train_with_decode` recipe.
- **Actor → drafter frozen-module re-sync after `update_actor`** —
  `sync_frozen_modules_from_actor` is a no-op stub; needs FSDP-aware
  actor-param gather. Frozen-target smoke doesn't exercise this.
- **Vocab pruning end-to-end** — `compute_target_p_padded` already
  supports both the pruning (`t2d` set) and no-pruning (`t2d=None`)
  paths post-Phase-A refactor. To activate: provide a `local_path` JSON
  template setting `draft_vocab_size < vocab_size`, populate `t2d` on
  the engine (`FSDPDrafterEngine._t2d_index`), and pass it through
  `prepare_model_inputs`. The loss-kernel + position-mask paths stay
  unchanged.
