# Drafter Pretrain — Migration Status

Compact status snapshot for the **pretrain-only** scope. Co-training (RL +
drafter) is deferred — see §"Deferred: Co-Training" below.

**Design:** `drafter-design.md`
**Workflow:** `project-guide.md` → "Where to make changes — recipe submodule first"

---

## Where to make changes

**Recipe submodule only.** All drafter code lives in
`recipe/drafter_cotraining/`. The parent `verl/` directory is used as-is
from upstream — **zero verl core changes** in the pretrain-only scope.

---

## Current scope: pretrain-only

Standalone EAGLE drafter pretraining pipeline — no RL loop, no actor,
no rollout, no verl core modifications.

```
parquet data → tokenize → HS collector (vLLM prefill, Mooncake KV)
                                                    │
                                                    ▼
                              DrafterPretrainWorker: Mooncake.get → forward → backward → opt.step
                                                    │
                                                    └── repeat
```

**Active code under `recipe/drafter_cotraining/`:**

| Component | File |
|---|---|
| Pretrain launcher | `main_drafter_pretrain.py` |
| Pretrain trainer | `trainer/pretrain_trainer.py` (`DraftModelPretrainTrainer`) |
| Pretrain worker | `workers/engine_workers.py` (`DrafterPretrainWorker`) |
| Drafter engine | `workers/drafter_engine.py` (`FSDPDrafterEngine`, `DrafterModelConfig`) |
| Eagle3 model + loss | `eagle3/{eagle3_model,draft/,ops/}` |
| Mooncake transport | `mooncake/` (KV connector, store, master) |
| HS collector | `hs_collector/` (`HSCollectorManager`) |
| Data collator | `data/collator.py` (`DataCollatorWithPadding`) |
| Chat template + tokenize | `utils/chat_template_tokenize.py` |
| Vocab mapping | `utils/vocab_mapping.py` |
| Pretrain config | `config/draft_model_pretrain_trainer.yaml` |

**Key architectural point:** `DrafterPretrainWorker` extends bare `Worker`
(not `ActorRolloutRefWorker`), sets `self.rollout = None`, and reuses
`ActorRolloutRefDrafterWorker` methods via class-level attribute aliasing.
Frozen weights (embed_tokens, target_lm_head_weight, verifier_norm) are
loaded from `target_model_path` on disk — no live actor to sync from.

---

## Latest end-to-end verification

Qwen3-4B / 2× H100 / `MAX_STEPS=16` post-cleanup smoke:

| metric | step 0 | step 15 | Δ |
|---|---:|---:|---:|
| `train/loss_weighted` | 12.0811 | 7.6583 | **−4.42** ✓ |
| `train/simulated_acc_len` | 0.00 | 0.21 | **+0.21** ✓ |
| `train/acc_0` | 0.00 | 0.18 | +0.18 |
| `train/grad_norm` | 30.25 | 14.06 | finite throughout |
| `train/lr` | 9.90e-05 | 0.00e+00 | cosine decay full course |

**Verification command** (Qwen3-8B pretrain — actively developed path):

```bash
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16
```

---

## FSDP2 + micro-batching refactor (2026-04-26)

Three-phase refactor; design + rationale in
`claude_docs/research/Eagle3-Co-Trained/Drafter Micro-Batching {Refactor,Concrete} Plan.md`.

| Phase | Change | Files |
|---|---|---|
| **A — Kernel fix** | Drop `LazyTarget` + `compiled_forward_kl_loss_from_hs`; generalize `compute_target_p_padded` to support `t2d=None` (no-pruning); store `target_p` as bf16. | `eagle3/eagle3_model.py`, `eagle3/ops/loss.py`, `engine/drafter_engine.py::prepare_model_inputs`, `tests/test_eagle3_loss.py` |
| **B — FSDP2 wrap** | Override `FSDPDrafterEngine._build_fsdp_module` for selective wrap (only `LlamaDecoderLayer`). Yaml: `strategy: fsdp → fsdp2`, drop `use_orig_params`. | `engine/drafter_engine.py`, both yaml configs |
| **C — Micro-batching** | Paged Mooncake fetch + per-mb weighted backward; metadata-time empty-mask filter; `total_valid_global` preflight; `T_pad_macro` precompute; `set_requires_gradient_sync(is_last)` on FSDP2 root. New helpers: `_allreduce_sum_int`, `_select_data_indices`, `_iter_micro_batch_keys`, `_drafter_train_step_micro`, `_drafter_micro_step`, `_aggregate_micro_metrics`. New `DataCollatorWithPadding(features, bucket_size_override=...)`. New yaml `drafter.engine_config.micro_batch_size_per_gpu` (default 1). | `engine/workers.py`, `data/collator.py`, both yaml configs |
| **E — Rename** | `recipe/drafter_cotraining/fsdp_workers.py` → `engine_workers.py` (engine-agnostic pattern; FSDP-naming was misleading). | recipe + claude_docs |

---

## Package layout refactor (2026-04-29)

Cleanup pass to align `recipe/drafter_cotraining/` with verl's recipe
convention (`main_<recipe>.py` launcher + `<name>_trainer.py` class)
and group library modules by role.

```
recipe/drafter_cotraining/
├── __init__.py
├── main_drafter_ct.py            ← RL launcher (DEFERRED)
├── main_drafter_pretrain.py      ← pretrain launcher (ACTIVE)
├── trainer/   {ray_trainer,pretrain_trainer}.py
├── data/      {controller,collator}.py
├── engine/    {drafter_engine,workers}.py  (note: in workers/ not engine/ on disk)
├── workers/   {engine_workers,drafter_engine}.py
├── utils/     {chat_template_tokenize,vocab_mapping}.py
└── eagle3/  hs_collector/  mooncake/  config/  scripts/  tests/
```

---

## Scope narrowing: pretrain-only (2026-05-01)

**Decision:** Ship pretrain-only first. Co-training (RL + drafter) is
deferred to a future milestone. This eliminates all verl core changes.

### What stays (pretrain-only scope)

| Component | Why |
|---|---|
| `DrafterPretrainWorker` | Core pretrain worker — no actor/rollout/ref |
| `DraftModelPretrainTrainer` | Standalone trainer — no PPO loop |
| `main_drafter_pretrain.py` | Pretrain entry point |
| `draft_model_pretrain_trainer.yaml` | Pretrain config |
| `FSDPDrafterEngine` / `Eagle3Model` | Shared model engine |
| HS collector + Mooncake | Shared infrastructure |
| `DataCollatorWithPadding` / `chat_template_tokenize` | Shared data utils |
| Vocab pruning | Shared feature |

### What's deferred (co-training only, not needed for pretrain)

| Component | File | Why deferred |
|---|---|---|
| `ActorRolloutRefDrafterWorker` | `workers/engine_workers.py` | Requires actor + rollout + ref + drafter co-location |
| `RayDrafterCTPPOTrainer` | `trainer/ray_trainer.py` | PPO trainer with drafter sub-pipeline |
| `main_drafter_ct.py` + `drafter_ct_trainer.yaml` | Entry + config | Co-training launcher |
| `DrafterDataController` / `SampleMeta` | `data/controller.py` | Driver-side sample routing for RL loop |
| vllm_rollout drafter APIs | `verl/workers/rollout/vllm_rollout/*` | `update_drafter_weights`, `inspect_drafter_sharing`, `get_drafter_weights`, `probe_target_param_norms`, `get_spec_decode_counters`, `_collect_shared_drafter_param_names` |
| Actor → drafter frozen-module re-sync | `engine_workers.py::_sync_drafter_frozen_modules` | No live actor in pretrain |
| Drafter → rollout weight sync (TODO 4) | `vllm_rollout/vllm_rollout.py` | No rollout in pretrain |

### Verl core diff analysis (52bf6abb..HEAD)

Only 3 files changed in verl/ core, all in `vllm_rollout/` — **all
drafter-related and all eliminable for pretrain-only**:

| File | +Lines | Change |
|---|---|---|
| `vllm_rollout/utils.py` | +289 | `_collect_shared_drafter_param_names`, `target_model="drafter"` in `update_weights_from_ipc`/`_update_weights`, `inspect_drafter_sharing`, `probe_target_param_norms`, `get_drafter_weights` (duplicated for Omni) |
| `vllm_rollout/vllm_async_server.py` | +19 | `get_spec_decode_counters()`, `kv_transfer_params` propagation, missing `return` on `collective_rpc` |
| `vllm_rollout/vllm_rollout.py` | +46 | `update_drafter_weights`, `get_drafter_weights`, `inspect_drafter_sharing`, `probe_target_param_norms`, `VERL_VLLM_FORCE_WEIGHT_SHM` env var |

**Minor non-drafter changes** (could be upstreamed separately later):
1. Bug fix: missing `return` on `collective_rpc` (line 186)
2. `VERL_VLLM_FORCE_WEIGHT_SHM` env var for SHM weight transfer
3. `kv_transfer_params` propagation for disaggregated serving

---

## Deferred: Co-Training (RL + Drafter)

### Re-enable plan (future milestone)

1. **Revert verl/ to upstream** — start from a clean `52bf6abb` (or newer upstream) base with zero drafter changes.

2. **Re-apply vllm_rollout drafter APIs as a small upstream PR** — the 3 files in `vllm_rollout/` are self-contained additions that could be upstreamed as optional drafter support. Key APIs:
   - `update_drafter_weights` — receives drafter weights via IPC
   - `get_drafter_weights` — exports drafter weights for snapshot/restore
   - `inspect_drafter_sharing` — diagnostics for shared tensors
   - `_collect_shared_drafter_param_names` — filters shared params during weight update

3. **Re-enable `ActorRolloutRefDrafterWorker`** — extends `ActorRolloutRefWorker` with drafter training. The method bodies already exist (aliased into `DrafterPretrainWorker` today).

4. **Re-enable `RayDrafterCTPPOTrainer`** — inserts drafter sub-pipeline after rollout in the PPO loop.

5. **Implement remaining weight sync flows** (from `weight-sync-flows.md`):
   - Flow 2: Actor → HS Collector (currently stub)
   - Flow 3: Actor → Drafter frozen-module re-sync (currently no-op)
   - Flow 4: Drafter → Rollout weight sync (currently TODO 4)

### Co-training files to preserve (don't delete)

These files exist in the recipe submodule and are referenced by both
paths. Do NOT delete them — just don't ship/activate them:

- `workers/engine_workers.py` — contains both `ActorRolloutRefDrafterWorker` (deferred) and `DrafterPretrainWorker` (active)
- `trainer/ray_trainer.py` — co-training PPO trainer
- `main_drafter_ct.py` — co-training launcher
- `config/drafter_ct_trainer.yaml` — co-training config
- `data/controller.py` — driver-side sample routing

---

## Open TODOs (pretrain-only scope)

### Active

- **Vocab pruning end-to-end** — `compute_target_p_padded` supports both
  pruning and no-pruning paths. To activate: provide a `local_path` JSON
  template setting `draft_vocab_size < vocab_size`, populate `t2d` on
  the engine, and pass it through `prepare_model_inputs`.
- **Gradient accumulation** — single-step path is proven; trivial extension.

### Deferred (co-training scope)

- **Drafter → rollout weight sync** (TODO 4) — needs `update_drafter_weights` API on rollout side.
- **Actor → drafter frozen-module re-sync** — needs FSDP-aware gathering of actor params.
- **Actor → HS Collector weight sync** (Flow 2) — currently stub.

---

## History

This branch added EAGLE drafter co-training as a new recipe submodule
plus minimal parent-side carve-outs. Scope narrowed to pretrain-only on
2026-05-01 to ship without verl core changes. The full timeline lives
in commit messages:

```bash
git log --oneline 52bf6abb..HEAD                      # parent
cd recipe/drafter_cotraining && git log --oneline ba24641..HEAD   # submodule
```
