# Drafter Co-Training — Migration Status

Single source of truth for what's done, what's TODO, and how to verify.

**Design:** See `rfc-drafter-trainer-integration.md`
**TorchSpec reference:** See `torchspec-to-verl-migration-map.md`

---

## Where to make changes (post-cleanup)

**Recipe submodule first.** All drafter library code lives in
`recipe/drafter_cotraining/`. Touch parent `verl/...` only for
verl-side base-class hooks, vLLM rollout integration, top-level
wrapper scripts, or docs. Full guidance + workflow snippet in
`project-guide.md` → "Where to make changes — recipe submodule first".

---

## 2026-04-23 (cleanup) — In-tree drafter mirrors deleted

The recipe submodule (`recipe/drafter_cotraining/`) is now the **only**
drafter co-training path. All in-tree mirrors under `verl/` have been
removed across six small commits on `deat/drafter-cotraining`:

| commit | scope | files removed |
|---|---|---:|
| `5d6ef16c` | top-level scripts/tests + retarget rollout-hs wrapper | 7 |
| `9f336664` | `verl/experimental/hs_collector/` | 3 |
| `8c30e557` | `verl/trainer/drafter/` (incl. config/) | 6 |
| `4912b7a0` | `verl/workers/drafter_workers.py`, `verl/workers/engine/fsdp/drafter_impl.py` | 2 |
| `fcde42c3` | `verl/models/eagle3/` (model + draft + ops) | 9 |
| `7a7ad708` | `verl/utils/mooncake/`, `verl/utils/eagle3_collator.py` | 9 |

Total: **36 files / ~7,500 LOC removed.**

**Carve-out preserved**:
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` keeps its
3-line `kv_transfer_params` propagation hook — generic improvement,
no-op when no KV connector is active. Queued as a separate small
upstream PR.

**Verification**: `grep -rn 'from verl.{workers.drafter_workers,workers.engine.fsdp.drafter_impl,experimental.hs_collector,trainer.drafter,models.eagle3,utils.mooncake,utils.eagle3_collator}' --include='*.py' --include='*.yaml' /root/verl` returns zero hits outside `recipe/`. End-to-end smoke retest queued (see "Verification" below).

**Older entries below reference now-deleted `verl/...` paths** — those
historical references describe state at the time of the listed entry.
The recipe-side equivalents are the live code today. The full mapping
is in `torchspec-to-verl-migration-map.md`.

---

## 2026-04-23 (training-smoke) — Close-loop drafter training wired

**Plan reference:** `tasks/drafter-training-milestone.md`.

**Summary.** `update_drafter` now runs a real Eagle3 training step (fetch →
collate → prepare_model_inputs → 7-step TTT forward → 0.8^i-weighted backward
→ optimizer.step + lr_scheduler.step) with all-reduced metrics. Target model
frozen, `accumulation_steps=1`, LazyTarget (no vocab pruning). Shape-print
fallback retained when `drafter.model_config.local_path` is unset so the
existing rollout+HS smoke (`test_drafter_rollout_hs.sh`) still runs without a
draft model.

**Engine fix.** `FSDPDrafterEngine.initialize()` was broken: it set
`self.module = Eagle3Model(…)`, then called `super().initialize()`, which
re-invoked `_build_module()` (HF AutoModel path) and overwrote `self.module`.
Replaced with a `_build_module()` override that returns the Eagle3-wrapped
draft model, letting the parent's standard
`_build_model_optimizer` → `_build_fsdp_module` → optimizer → LR scheduler
flow run end-to-end. Also added a lightweight `DrafterModelConfig` dataclass
because `HFModelConfig` would try to load target-model tokenizer/HF config for
the draft.

**New:**

| File | What |
|---|---|
| `recipe/drafter_cotraining/scripts/test_drafter_training.py` | Subclass of `MicroRolloutHSOnlyTrainer` — fit() runs N steps of rollout → HS → `update_drafter`, reads `meta_info['train_metrics']`, prints loss/acc/acc_len/grad/lr per step and a start→end delta summary. `itertools.cycle(train_dataloader)` = verl analogue of TorchSpec's `controller.reload_dataset()`. |
| `scripts/run_drafter_training.sh` | Wrapper (clone of `run_drafter_rollout_hs.sh`) with `drafter.enable=True`, `drafter.model_config.local_path=$DRAFT_CONFIG`, `drafter.optimizer_config.total_training_steps=$MAX_STEPS` (so cosine LR decays over the run instead of plateauing — §4.7 of milestone plan). Default `MAX_STEPS=32`. |
| `recipe/drafter_cotraining/config/draft_models/qwen3_4b_eagle3.json` | Qwen3-4B draft config for the smoke. Adapted from TorchSpec's `qwen3_8b_eagle3.json` with 4B dimensions (`hidden_size=2560`, `intermediate_size=9728`). `draft_vocab_size` omitted → `AutoDraftModelConfig.from_dict` auto-fills to `vocab_size` → `vocab_pruning=False` → LazyTarget (matches TorchSpec default). |

**Changed:**

| File | What |
|---|---|
| `recipe/drafter_cotraining/fsdp_workers.py` (canonical) + `verl/workers/drafter_workers.py` (in-tree duplicate) | `update_drafter` rewritten: fetch+collate path preserved, then either shape-print (no drafter engine) or the full forward/backward/optimizer step via new `_drafter_train_step` + `_aggregate_drafter_metrics` helpers. Metrics all-reduced over the DP group and returned in `DataProto.meta_info["train_metrics"]` (DataProto.concat merges non-metric meta_info keys under equality, which holds since all ranks see the same reduced values). `_init_drafter` now converts sub-configs to dataclasses via `omega_conf_to_dataclass`. |
| `recipe/drafter_cotraining/drafter_engine.py` + `verl/workers/engine/fsdp/drafter_impl.py` | Added `DrafterModelConfig` dataclass (minimal fields: `local_path`, `dtype`, `ttt_length`, plus stubs for `lora_rank`, `use_remove_padding`, etc. that `FSDPEngine.__init__` reads). Replaced broken `initialize()` override with a `_build_module()` override returning `Eagle3Model(draft_model)`. |
| `recipe/drafter_cotraining/config/drafter_ct_trainer.yaml` | Expanded `actor_rollout_ref.drafter` with defaults for `model_config` (local_path=null so the existing smoke stays backwards-compatible, dtype=bf16, ttt_length=7), `engine_config` (fsdp strategy, no offload — tiny model), `optimizer_config` (lr=3e-4, warmup_ratio=0.1, cosine, clip=1.0), and `checkpoint_config`. |

**Verification — post-fix (2026-04-23, Qwen3-4B, N_GPUS=2, MAX_STEPS=16)** —
with the input-shift fix (§ [Eagle3 objective correction](#eagle3-objective-correction)
below) and TorchSpec-matched optimizer defaults (lr=1e-4, wd=0, warmup=0.015,
clip_grad=0.5):

| metric | step 0 | step 15 | Δ |
|---|---:|---:|---:|
| `train/loss_weighted` | 12.1003 | 7.8563 | **−4.24** ✓ |
| `train/simulated_acc_len` | 0.00 | 0.19 | **+0.19** ✓ |
| `train/acc_0` | 0.00 | 0.17 | +0.17 |
| `train/grad_norm` | 30.25 | 12.44 | finite throughout |
| `train/lr` | 9.90e-05 | 0.00e+00 | cosine decay ran full course |

Absolute numbers are smaller than the pre-fix run (loss 12.07→6.52,
acc_len 0→0.33 at lr=3e-4) — expected: (a) the pre-fix task was degenerate
(predict current given current + context that already encodes it), (b) the
peak LR is lower. Trend signs are the correct ones and now reflect a genuine
Eagle3 draft-quality signal.

Per-step wall-clock ~2.6s (gen 1.0s + HS 1.0s + drafter train 0.65s).

### Eagle3 objective correction

`prepare_model_inputs` in both `recipe/drafter_cotraining/drafter_engine.py`
and `verl/workers/engine/fsdp/drafter_impl.py` now left-shifts `input_ids`
and `last_hidden_states` by 1 (via `padding(..., left=False)` from
`eagle3_model.py`) before the verifier_norm + target build. Matches TorchSpec
`torchspec/training/eagle3_trainer.py::_forward`.

Why: at TTT step 0 / position t, Eagle3 inference gives the draft
`(aux[t], token[t+1])` and asks it to predict `token[t+2]` — i.e. "verifier
just emitted `token[t+1]`; drafter speculatively continues". Without the
shift, we were feeding `(aux[t], token[t])` and training it to reproduce
`token[t+1]`'s distribution, which is information already encoded in
`aux[t]`. Loss still descends, but the drafter learns a pass-through, not
the Eagle3 objective.

Detection: loss baseline at step 0 is ~ln(vocab_size) ≈ 11.93 in both cases,
but post-fix absolute numbers are higher at the same step count because the
true task is harder. Directional pass criteria (§4.8) still satisfied.

Also aligned with TorchSpec in this pass:
- `clip_grad`: 1.0 → **0.5** (YAML default + wrapper env var default)
- `weight_decay`: 0.01 → **0.0**
- `lr_warmup_steps_ratio`: 0.1 → **0.015**
- `lr`: 3e-4 → **1e-4**

### Three additional fixes (2026-04-23 late)

After the shift fix landed, an external review caught three more TorchSpec
mismatches. All addressed:

**1. vLLM aux layer IDs were off by one.** TorchSpec uses *post-layer N*
semantics (`[1,17,32,35]` = "after layers 1, 17, 32, 35"). vLLM's hook fires
at the *input* of each listed layer (= output of the previous layer), so
TorchSpec shifts non-final IDs by +1 and appends the final layer separately
(`vllm_engine.py:155-181`). For Qwen3-4B (36 layers): TorchSpec
`[1,17,32,35]` → vLLM `[2,18,33,35]`. Our wrapper had been passing the
TorchSpec semantic directly → captured one layer too early. Fixed default in
`scripts/run_drafter_training.sh`.

**2. Last response position has no valid next-token target.** TorchSpec masks
`[prompt_tokens, completion_tokens-1]` (`sgl_engine_decode.py:249-254`) and
explicitly zeroes the final loss position (`preprocessing.py:329-331`).
Updated `update_drafter` to use `loss_mask[..., plen:plen+rlen-1] = 1`
(rlen-1 ones) in both canonical (`recipe/.../fsdp_workers.py`) and in-tree
(`verl/workers/drafter_workers.py`) paths.

**3. In-tree drafter path was silently falling back to all-ones mask.** The
recipe path was wired (controller + ray_trainer + hs_collector + worker), but
the in-tree duplicates under `verl/trainer/drafter/` and
`verl/experimental/hs_collector/` had not been updated. Mirrored
`prompt_lens` / `response_lens` through:
`verl/experimental/hs_collector/hs_collector_manager.py`,
`verl/trainer/drafter/controller.py` (SampleMeta + drain),
`verl/trainer/drafter/drafter_ct_ray_trainer.py` (`_sample_metas_from_hs_batch`).

**Verification — final smoke (post all three fixes, MAX_STEPS=16):**

| metric | step 0 | step 15 | Δ |
|---|---:|---:|---:|
| `train/loss_weighted` | 12.0811 | 7.6583 | **−4.42** ✓ |
| `train/simulated_acc_len` | 0.00 | 0.21 | **+0.21** ✓ |
| `train/acc_0` | 0.00 | 0.18 | +0.18 |

acc_len modestly improved over the previous post-shift+mask run (0.21 vs
0.19 / 0.18) — the corrected aux layers + drop-last mask give a slightly
cleaner training signal even at this short horizon. Mask + aux-ID effects
sharpen in longer runs.

**Nine RCA'd fixes to get here** (worth keeping — same class of issue tends to
recur when adding new engines / model types):

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ValidationError: Incompatible value 'None' for field of type 'int'` on `FSDPEngineConfig.max_token_len_per_gpu` | `omega_conf_to_dataclass` routes through `OmegaConf.structured(cls)` which rejects `int = None` defaults | Added `build_drafter_subconfig` — bypasses `OmegaConf.structured`, uses the dataclass constructor directly with field-filtered kwargs |
| 2 | `NameError: DeepseekV3Config not defined` at `AutoDraftModelConfig._config_mapping` class-body eval | Module references the symbol in a dict literal without importing it | Added `from transformers import DeepseekV3Config` |
| 3 | `Must flatten tensors with uniform requires_grad when use_orig_params=False` | FSDP1 default rejects mixed-grad param groups; draft has frozen `embed_tokens` + trainable rest | `use_orig_params: True` in drafter `engine_config` |
| 4 | `size of tensor a (0) must match the size of tensor b (2560)` during sync | Actor→draft frozen-module sync copied full actor tensor into a draft buffer that FSDP had sharded to 0 on the other rank | Replaced actor sync with TorchSpec pattern: load `embed_tokens` in `_build_module` and `lm_head`/`model.norm` in `initialize()` from `target_model_path` (new `DrafterModelConfig` field) |
| 5 | `apply_fsdp2` asserts `len(transformer_layer_cls_to_wrap) > 0` and it's `None` | `Eagle3Model` is a custom `nn.Module` with no `_no_split_modules` attribute | Set `module._no_split_modules = ["LlamaDecoderLayer"]` in `_build_module` |
| 6 | `AttributeError: 'Eagle3Model' object has no attribute 'config'` | `_select_fsdp2_wrap_targets` reads `model.config.tie_word_embeddings` | Set `module.config = draft_model.config` after Eagle3 wrap |
| 7 | `KeyError: 'lm_head.weight'` in target safetensors weight map | Qwen3-4B has `tie_word_embeddings=True` → `lm_head.weight` isn't stored; it's tied to `model.embed_tokens.weight` | Fallback in `_load_target_frozen_weights`: try `lm_head.weight` first, on `KeyError` use `model.embed_tokens.weight` |
| 8 | `torch._dynamo.exc.TorchRuntimeError: Cannot call numel() on tensor with symbolic sizes/strides` | `torch.compile`'d Eagle3 loss kernel + FSDP2 `DTensor` weights + symbolic input shape confuses fake-tensor tracing | Added `TORCH_COMPILE_DISABLE=1` to run script (kept after FSDP2→FSDP1 switch for safety) |
| 9 | `RuntimeError: aten.mm.default got mixed torch.Tensor and DTensor` in the loss kernel | FSDP2 makes params `DTensor` everywhere; `F.linear(plain_tensor, dtensor_weight)` isn't auto-promoted | Switched `strategy: fsdp` (FSDP1) — params stay as plain `nn.Parameter`, `F.linear` works eagerly. Load frozen weights on every rank (FSDP1 lacks the FSDP2 rank-0 broadcast dance) |

**Mitigating fixes kept in the repo** (even where the immediate cause was
sidestepped, the fix is still load-bearing for other paths):

- `build_drafter_subconfig` (fix #1) is the only correct way to convert partial
  CLI overrides into `FSDPEngineConfig` / `FSDPOptimizerConfig` given the
  existing type-annotation bugs; will be needed by any future drafter-style
  engine.
- `module._no_split_modules` + `module.config` injection (fixes #5, #6) are
  still required by FSDP1's `get_fsdp_wrap_policy` / checkpoint manager.
- Target-path loading (fix #4) replaces the actor-side sync end-to-end; the
  old `sync_frozen_modules_from_actor` is now a documented no-op, kept as a
  stub for the eventual actor-updates path.

**Open follow-ups:** (a) response-only `loss_mask` (currently `ones_like` so
acc/acc_len are inflated by prompt positions — directional signal still
valid); (b) gradient accumulation once single-step path is proven;
(c) drafter → rollout weight sync (TODO 4 below); (d) vocab-pruning path
(set `draft_vocab_size < vocab_size` and wire `set_vocab_buffers`); (e) delete
the in-tree duplicates under `verl/` once the recipe path is the only consumer.

---

## 2026-04-23 (later) — Padding building block + Mooncake delete sync

**New:**

| File | What |
|---|---|
| `verl/utils/eagle3_collator.py` (new, ~70 LOC) | `Eagle3Collator` — pads variable-seq-len Mooncake samples to rectangular `[B, T_pad]` / `[B, T_pad, D]` with `attention_mask`. Right-pad zeros, ceil to next multiple of 256 (bucketing), drops `last_hidden_states` if absent. Ported 1:1 from `/root/TorchSpec/torchspec/data/utils.py:DataCollatorWithPadding`. |
| `scripts/diag_mooncake_init.py` (new) | Standalone diagnostic — spins up `mooncake_master` subprocess + N stores, times put success/failure across delays. Used to confirm zombie-worker hypothesis. |

**Changed:**

| File | What |
|---|---|
| `verl/workers/drafter_workers.py` | Replaced TODO stub: lazy-init cached `EagleMooncakeStore` → fetch each key → build per-sample dicts (`loss_mask = ones_like(input_ids)`) → `Eagle3Collator()` → print padded shapes per rank → `remove_eagle3_tensors(key)` immediately after each fetch. Deleted `_skeleton_fetch_and_report`, `_drafter_skeleton` flag, skeleton branch. Gated `_init_drafter()` on `drafter.model_config.local_path` so `drafter.enable=True` smoke-tests the dispatch + collator without needing a real Eagle3 checkpoint. |
| `verl/utils/mooncake/eagle_store.py` | **Synced delete behavior with TorchSpec commit `619ad48`.** Removed `DeferredDeleteManager` instantiation + `atexit` hook; `remove_eagle3_tensors()` now does immediate `batch_remove(force=True)` with up to 3 retries (accepting codes `0` and `-704` as success). |
| `verl/utils/mooncake/deferred_delete.py` | **Deleted** — unused after sync. |
| `scripts/test_mooncake_store.py` | Added `--launch-master` flag (subprocess, atexit cleanup); dropped `DeferredDeleteManager` import smoke and "deferred delete after TTL" message. |
| `scripts/test_drafter_rollout_hs.py` | Dropped the `drafter.skeleton` gate around `update_drafter` dispatch; updated docstring. |
| `scripts/run_drafter_rollout_hs.sh` | Extended pre-run cleanup to `pkill -x VLLM::EngineCore` and `VLLM::Worker` (zombie-worker fix). |
| `verl/trainer/drafter/config/drafter_ct_trainer.yaml` | Removed dead `drafter.skeleton: False`. |
| `claude_docs/torchspec-to-verl-migration-map.md` | Updated row 126: `deferred_delete.py` removed, references commit `619ad48`. |

**Verified end-to-end** (Qwen3-4B, N_GPUS=2, MAX_STEPS=1, `drafter.enable=True`):
- Mesh dispatch shards 8 samples into 4 per rank.
- `T_pad` = 512 (rank with max-seq in (256, 512]) — multiple of 256 ✓
- `hidden_states.shape = (4, 512, 7680)` = 3 aux × 2560 hidden ✓
- `last_hs.shape = (4, 512, 2560)` ✓
- `Force-deleted <key>` per fetched sample (no leak).
- `test_mooncake_store.py` end-to-end green (PUT, GET, integrity, force-REMOVE).

**Open follow-ups (non-blocking):** (a) replace shape-print with `self.drafter.train_batch(batch)` once a draft-model checkpoint is configured; (b) thread `response_len` through `SampleMeta` so `loss_mask` excludes prompt tokens (currently all-ones).

---

## 2026-04-23 — Extracted into `recipe/drafter_cotraining/` (submodule)

All drafter-private code now lives in the **`recipe/` git submodule** at
`recipe/drafter_cotraining/`. The originals under `verl/` are **still present
as duplicates** pending removal. Parent gitlink bumped in commit
`2a223c1b [drafter] bump recipe submodule: add drafter_cotraining`
(submodule HEAD: `574ce08e` on branch `dev/eagle-co-train`).

**New canonical paths** (all under `recipe/drafter_cotraining/`):

| Component | Old path (still present) | New path (canonical) |
|---|---|---|
| Mooncake store | `verl/utils/mooncake/` | `mooncake/` |
| HS connector | `verl/utils/mooncake/hidden_states_connector.py` | `mooncake/hidden_states_connector.py` |
| Eagle3 model | `verl/models/eagle3/` | `eagle3/` |
| HSCollectorManager | `verl/experimental/hs_collector/` | `hs_collector/` (renamed `hs_collector_model.py` → `model.py`, `hs_collector_manager.py` → `manager.py`) |
| FSDPDrafterEngine | `verl/workers/engine/fsdp/drafter_impl.py` | `drafter_engine.py` |
| Worker | `verl/workers/drafter_workers.py` | `fsdp_workers.py` |
| Trainer | `verl/trainer/drafter/drafter_ct_ray_trainer.py` | `ray_trainer.py` |
| Entry point | `verl/trainer/drafter/main_drafter_ct_ppo.py` | `main_drafter_ct.py` |
| Controller / orch | `verl/trainer/drafter/{controller,orchestration}.py` | same names |
| Config | `verl/trainer/drafter/config/drafter_ct_trainer.yaml` | `config/drafter_ct_trainer.yaml` |
| Tests / scripts | `tests/test_eagle3_loss.py`, `scripts/{run,test}_*` | `tests/`, `scripts/` |

**Imports:** all internal references rewritten to `recipe.drafter_cotraining.X`
(absolute), including the YAML `_target_` and the vLLM
`kv_connector_module_path`. Zero relative imports remain.

**Runnability:** `recipe/drafter_cotraining/scripts/run_drafter_rollout_hs.sh`
runs end-to-end from clean state through `trainer.fit()`, the first HS
push, and now the drafter mesh dispatch + Mooncake fetch + collation +
force-delete (verified 2026-04-23 on Qwen3-4B, N_GPUS=2: per-rank line
`[drafter rank 1] step 1/1: B=4 T_pad=512  input_ids=(4,512) hidden_states=(4,512,7680) last_hs=(4,512,2560) attn_mask=(4,512) loss_mask=(4,512)`).

The earlier `code=-800 / unregistered segment localhost:13807` failure was
diagnosed (`scripts/diag_mooncake_init.py`) as **zombie vLLM workers from
prior crashed runs** holding sockets and re-advertising phantom Mooncake
segments to fresh masters. Wrapper now does
`pkill -x VLLM::EngineCore VLLM::Worker mooncake_master` before each run;
not a Mooncake bug.

**Pending cleanup:** delete the in-tree duplicates under `verl/` once the
recipe path is the only consumer. Audit confirmed no non-drafter consumers
of `verl/utils/mooncake/*` or `verl/models/eagle3/*`.

**One generic carve-out kept in base verl:** 3-line `kv_transfer_params`
propagation in `verl/workers/rollout/vllm_rollout/vllm_async_server.py` —
to be submitted as a separate small upstream PR (no-op without an active
KV connector).

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

#### ~~TODO 1: `update_drafter()` body — real training step~~ ← **Done (2026-04-23 training-smoke)**

See the top "2026-04-23 (training-smoke)" entry. All six steps are now wired
(fetch → collate → prepare_model_inputs → 7-step TTT forward → 0.8^i-weighted
backward → optimizer.step → mooncake cleanup). `accumulation_steps=1` for the
smoke; multi-micro-batch accumulation deferred to follow-up.

Real `loss_mask` (response-only) is still on the follow-up list — current mask
is `ones_like(input_ids)`, which inflates `acc_i` and `simulated_acc_len` by
including prompt positions. Direction of change across training remains valid.

**TorchSpec ref:** `training/data_fetcher.py:79-113` (Mooncake fetch), `training/eagle3_trainer.py:235-277` (forward/backward)

#### ~~TODO 2: Sleep/wake coordination~~ ← **Done**

Handled by `HSCollectorManager.compute_hidden_states()` (wake → infer → sleep) which wraps verl's `RolloutReplica` lifecycle. The manager replaces the previous hand-rolled `VllmHSCollector` + bespoke sleep/wake path.

#### ~~TODO 3: `RayPPOTrainer.fit()` integration~~ ← **Done**

Implemented as a subclass rather than modifying base `RayPPOTrainer`:
- `RayDrafterCTPPOTrainer` (`verl/trainer/drafter/drafter_ct_ray_trainer.py`, ~561 lines) — full drafter sub-pipeline in `fit()`
- `DrafterCTTaskRunner` (`verl/trainer/drafter/main_drafter_ct_ppo.py`, ~139 lines) — entry point wiring

#### TODO 4: Drafter → rollout weight sync ← Not a blocker

**File:** `recipe/drafter_cotraining/fsdp_workers.py:261` (`update_weights()`)

Actor → drafter frozen-module re-sync after `update_actor()` is wired
(`_sync_drafter_frozen_modules()` is called inside `update_weights()`).
Drafter → rollout sync still a `pass` — `get_per_tensor_param()` returns
the params but `rollout.update_drafter_weights()` doesn't exist yet on
the rollout side. Training works without this.

#### ~~TODO 5: Config schema (YAML)~~ ← **Done**

`recipe/drafter_cotraining/config/drafter_ct_trainer.yaml` defines the full
schema (`hs_collector:`, `mooncake:`, `drafter:` sections) and inherits the
base `ppo_trainer` config via Hydra `defaults`. The micro-test script runs
against this config.

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
