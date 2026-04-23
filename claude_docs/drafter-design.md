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

`update_drafter` in `recipe/drafter_cotraining/fsdp_workers.py`:

1. **Fetch** — `EagleMooncakeStore.get(key, shapes, dtypes, device)`
   per sample.
2. **Mask** — `loss_mask = [0]*prompt_len + [1]*(response_len-1) + [0]`
   per sample. Drops the final response position (no valid next-token
   target — matches TorchSpec `sgl_engine_decode.py:249`
   `completion_tokens-1`).
3. **Collate** — `Eagle3Collator` pads to `[B, T_pad]` / `[B, T_pad, D]`,
   `T_pad` = ceil to next multiple of 256.
4. **Prepare model inputs** — `FSDPDrafterEngine.prepare_model_inputs`
   left-shifts `input_ids` and `last_hidden_states` by 1
   (`padding(..., left=False)`), applies `verifier_norm` to
   `last_hidden_states`, builds a `LazyTarget` from
   `target_lm_head_weight`. Left-shift matches Eagle3 inference
   semantics: at TTT step 0 / position t, draft sees
   `(aux[t], token[t+1])` and predicts `token[t+2]`.
5. **Forward** — `Eagle3Model.forward(input_ids, attention_mask, target,
   loss_mask, hidden_states)` — 7-step TTT loop, returns
   `(plosses[L], _, acces[L])`.
6. **Backward** — `loss = sum(0.8^i * plosses[i] for i in range(L)) /
   accumulation_steps`; `loss.backward()`.
7. **Optimizer step** — `engine.optimizer_step()` (clips grad to
   `clip_grad`, skips on non-finite); `engine.lr_scheduler_step()`.
8. **Aggregate metrics** — `_aggregate_drafter_metrics` all-reduces
   per-TTT-step plosses + acces over the DP group, computes
   `simulated_acc_len = acc_0 + acc_0·acc_1 + …`, returns dict in
   `meta_info["train_metrics"]`.
9. **Cleanup** — `EagleMooncakeStore.remove_eagle3_tensors(key)` per
   sample (frees the buffer for the next prefill).

`accumulation_steps` is fixed at 1 today; multi-micro-batch
accumulation is a small follow-up.

---

## Engine choice — FSDP1 with `use_orig_params=True`

The drafter engine uses FSDP1 (`torch.distributed.fsdp.FullyShardedDataParallel`)
with `use_orig_params=True` for two reasons:

1. **Mixed-grad compatibility.** Draft has frozen `embed_tokens` +
   trainable `fc`/`midlayer`/`norm`/`lm_head`. FSDP1's default
   `use_orig_params=False` rejects mixed `requires_grad` in a wrap
   group. `True` lifts that restriction.

2. **Plain Tensor params.** FSDP2 (`fully_shard`) wraps params as
   `DTensor`s. The Eagle3 loss kernel does
   `F.linear(plain_input, draft_lm_head_weight)`; with FSDP2, that
   becomes a mixed `Tensor × DTensor` matmul which PyTorch doesn't
   auto-promote. FSDP1 with `use_orig_params=True` keeps params as
   plain `nn.Parameter`, so the kernel runs eagerly without
   `distribute_tensor` plumbing.

TorchSpec's analog is `torch.distributed._composable.replicate` (DDP
with FSDP-style API, default for Eagle3 training in TorchSpec). Verl
doesn't expose a `replicate`-style strategy today; FSDP1 is the
closest. Sharding adds an all-gather per forward but the draft is tiny
(<1 GB bf16) so the cost is negligible at our scale.

---

## Frozen weight loading — from `target_model_path`, not from actor

`FSDPDrafterEngine` loads three frozen weights from `target_model_path`
on every rank at init:

| Weight | Read from key | Lives at | Source order |
|---|---|---|---|
| `embed_tokens.weight` | `model.embed_tokens.weight` | `Eagle3Model.draft_model.embed_tokens` | `_build_module` (pre-FSDP wrap, on every rank) |
| `lm_head.weight` (`target_lm_head_weight`) | `lm_head.weight`, fallback `model.embed_tokens.weight` if `tie_word_embeddings=True` | `FSDPDrafterEngine._target_lm_head_weight` | `initialize` → `_load_target_frozen_weights` |
| `model.norm.weight` (`verifier_norm`) | `model.norm.weight` | `FSDPDrafterEngine._verifier_norm` (a `LlamaRMSNorm` module) | `initialize` → `_load_target_frozen_weights` |

This sidesteps the FSDP1-sharded actor params at the cost of needing
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
- **Gradient accumulation > 1** — single-step path proven, multi-step
  is a small extension to `_drafter_train_step`.
- **Vocab pruning** (`draft_vocab_size < vocab_size` +
  `set_vocab_buffers`) — `prepare_model_inputs` would gain a branch on
  `eagle3.vocab_pruning` to use `compute_target_p_padded` instead of
  `compute_lazy_target_padded`.
