# Drafter Co-Training — Migration Status

Compact status snapshot. Detailed RCA, change log, and per-fix
rationale all live in **`git log`** on `feat/drafter-cotraining`
(parent) and `feat/drafter-cotraining` (recipe submodule).

**Design:** `rfc-drafter-trainer-integration.md`
**TorchSpec mapping:** `torchspec-to-verl-migration-map.md`
**Workflow:** `project-guide.md` → "Where to make changes — recipe submodule first"

---

## Where to make changes

**Recipe submodule first.** All drafter library code lives in
`recipe/drafter_cotraining/`. Touch parent `verl/...` only for verl-
side base-class hooks, vLLM rollout integration, top-level wrapper
scripts, or docs. Full guidance + workflow snippet in `project-guide.md`.

---

## Current state

The drafter co-training pipeline is wired end-to-end and verified:

```
raw prompt → rollout → hidden states (Mooncake) → mesh dispatch
                                                            │
                                                            ▼
                                  forward → backward → optimizer.step
                                                            │
                                                            └── repeat
```

**All canonical code under `recipe/drafter_cotraining/`:**

| Component | File |
|---|---|
| Drafter worker | `fsdp_workers.py` (`ActorRolloutRefDrafterWorker`) |
| Drafter engine | `drafter_engine.py` (`FSDPDrafterEngine`, `DrafterModelConfig`) |
| Trainer + entry point | `ray_trainer.py`, `main_drafter_ct.py` |
| Data controller | `controller.py` (`DrafterDataController`, `SampleMeta`) |
| Eagle3 model + loss | `eagle3/{eagle3_model,draft/,ops/}` |
| Mooncake transport | `mooncake/` (KV connector, store, master) |
| HS collector | `hs_collector/` (`HSCollectorManager`) |
| Eagle3 collator | `eagle3_collator.py` |
| Smoke harness | `scripts/test_drafter_{rollout_hs,training,training_offline}.py` |

**Parent verl carve-out** (one file): the 3-line `kv_transfer_params`
propagation hook in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` — generic
improvement, no-op without an active KV connector. Queued as a
separate small upstream PR.

**Top-level smoke wrappers** (parent, shell out to recipe scripts):
- `scripts/run_drafter_rollout_hs.sh` — rollout + HS dispatch (no training)
- `scripts/run_drafter_training.sh` — full close-loop training

---

## Latest end-to-end verification

Qwen3-4B / 2× H100 / `MAX_STEPS=16` post-cleanup smoke
(`scripts/run_drafter_training.sh`):

| metric | step 0 | step 15 | Δ |
|---|---:|---:|---:|
| `train/loss_weighted` | 12.0811 | 7.6583 | **−4.42** ✓ |
| `train/simulated_acc_len` | 0.00 | 0.21 | **+0.21** ✓ |
| `train/acc_0` | 0.00 | 0.18 | +0.18 |
| `train/grad_norm` | 30.25 | 14.06 | finite throughout |
| `train/lr` | 9.90e-05 | 0.00e+00 | cosine decay full course |

Per-step wall-clock ~2.6s (gen 1.0s + HS 1.0s + drafter train 0.65s).

Defaults match TorchSpec's Eagle3 training: `lr=1e-4`,
`weight_decay=0`, `lr_warmup_steps_ratio=0.015`, `lr_scheduler_type=cosine`,
`clip_grad=0.5`, `accumulation_steps=1`, FSDP1 + `use_orig_params=True`.
vLLM aux layer IDs default to `[2,18,33,35]` (TorchSpec post-layer +1
shift + final-layer append for Qwen3-4B 36 layers). Loss mask is
response-only with the final response position dropped.

---

## Open TODOs

### TODO 4 — Drafter → rollout weight sync (not a blocker)

`recipe/drafter_cotraining/fsdp_workers.py::update_weights` currently
has a `pass` placeholder. `engine.get_per_tensor_param()` already
returns drafter weights; the rollout side needs an
`update_drafter_weights()` API to receive them. Required only for
speculative-decoding rollout speedup; training works without it.

TorchSpec equivalent: `_maybe_sync_draft_weights` in
`controller/loop.py:42-73` — saves draft to disk, calls
`engine.update_weights_from_disk(update_draft_model=True)`. Activated
only for the `train_with_decode` recipe (every `decode_weight_sync_interval`
steps, default 500).

### Smaller follow-ups

- **Gradient accumulation** (single-step path is proven; trivial extension).
- **Vocab pruning** (`draft_vocab_size < vocab_size` + `set_vocab_buffers`
  preflight) — see `tasks/drafter-training-milestone.md` §4.5 for the
  branch in `prepare_model_inputs` that would activate it.
- **Actor → drafter frozen-module re-sync** after `update_actor()`. The
  current `sync_frozen_modules_from_actor` is a no-op stub (target-
  path init replaces it for the smoke). Properly implementing this
  needs FSDP-aware gathering of actor params (the actor is FSDP1-
  sharded, naive copy fails — caused one of the original RCA'd bugs).

---

## History

This branch added EAGLE drafter co-training as a new recipe submodule
(`shadowpa0327/verl-recipe`, branch `feat/drafter-cotraining`) plus a
minimal parent-side carve-out. The full timeline — initial TorchSpec
port, in-tree → recipe migration, close-loop training activation,
nine-fix RCA pass against TorchSpec, in-tree mirror cleanup — lives
in commit messages on both branches:

```bash
git log --oneline 52bf6abb..HEAD                      # parent
cd recipe/drafter_cotraining && git log --oneline ba24641..HEAD   # submodule
```
