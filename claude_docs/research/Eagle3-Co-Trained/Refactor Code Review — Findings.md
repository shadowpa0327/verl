---
title: Refactor Code Review — Findings
date: 2026-04-26
tags:
  - eagle3
  - drafter
  - verl
  - code-review
  - bugs
aliases:
  - Drafter Refactor Review
  - FSDP2 + Micro-Batch Review Findings
---

# Refactor Code Review — Findings

> [!summary] TL;DR
> Three parallel skeptical review agents covered the three refactor areas (Phase A kernel, Phase B FSDP2 wrap, Phase C micro-batching). Total raised: ~25 concerns. After triage: **2 critical bugs** (rank-divergence on `accum_steps`; missing try/finally around `set_requires_gradient_sync`), **3 real-but-lower-severity** items, **2 doc-only notes**, and ~10 false alarms (incorrect understanding of design intent or PyTorch internals). The two critical bugs need fixes before any production training; the others are worth fixing but lower priority.

> [!info] Reviewers
> Three Explore agents reviewed in parallel, each scoped to one refactor area:
> 1. Phase A: kernel + Eagle3Model + tests
> 2. Phase B: FSDP2 wrap + engine init
> 3. Phase C: micro-batching `update_drafter` + helpers
>
> Agents were instructed to be skeptical (no praise lists), find real issues, cite file:line, and rank by severity.

---

## 1. Critical bugs (act before production)

### 🔴 #1 — DP rank divergence on `accum_steps` after the empty-mask filter

**File:** `recipe/drafter_cotraining/engine_workers.py:265` (and the metadata-time empty-mask filter ~line 208-236)

**The bug:**
The empty-mask filter runs **per-rank AFTER** `make_nd_compute_dataproto_dispatch_fn` has already dispatched. So if rank 0 happens to have a sample with `rlen-1 ≤ 0` and rank 1 doesn't, the filter drops on rank 0 only. Result: ranks end up with different `len(mooncake_keys)` → different `accum_steps`.

The micro-batch loop keys `is_last = mb_idx == accum_steps - 1` off this per-rank value:

```python
for mb_idx, mb_data in self._iter_micro_batch_keys(data, micro_size):
    is_last = mb_idx == accum_steps - 1
    if set_grad_sync is not None:
        set_grad_sync(is_last)
    ...
```

If `accum_steps_rank0 = 2` and `accum_steps_rank1 = 1`:
- Rank 0 mb_idx=0: sync=False (no reduce-scatter)
- Rank 0 mb_idx=1: sync=True (reduce-scatter at end of backward)
- Rank 1 mb_idx=0: sync=True (reduce-scatter at end of backward)

Both ranks issue exactly 1 reduce-scatter call, but **at different points in their respective loops**. NCCL matches collectives in order across ranks → rank 0's reduce-scatter (its 2nd backward) matches rank 1's reduce-scatter (its 1st backward) → **shape mismatch / hang / undefined behavior**.

**Smoke caveat:** the 16-step Qwen3-8B run worked because the test data didn't trigger the empty-mask filter on any rank (response_lens ≥ 2 throughout). The bug is latent.

**Fixes (pick one):**
- **Option A (preferred):** Move the empty-mask filter PRE-dispatch (in the trainer driver, before calling `update_drafter`). All ranks see the same global filter result.
- **Option B:** All-reduce MAX on `accum_steps` after computing it locally. On ranks with fewer keys, run the loop `accum_steps_global` times anyway, with the extra iterations being no-ops (skip Mooncake fetch + skip backward) but participate in `set_grad_sync(is_last)` so the comm pattern stays uniform.
- **Option C:** Detect divergence and raise: `assert self._allreduce_max_int(accum_steps) == accum_steps` — fails loudly instead of hanging.

Option A is the architecturally cleanest. Option B is the most defensive. Option C is a one-line band-aid.

---

### 🔴 #2 — `set_requires_gradient_sync(False)` state persists on exception

**File:** `recipe/drafter_cotraining/engine_workers.py:_drafter_train_step_micro` (~line 484-525)

**The bug:**
The micro-batch loop calls `set_grad_sync(False)` then `set_grad_sync(True)` on the last iteration. If `_drafter_micro_step` raises an exception while sync is suppressed (e.g., OOM, NaN, kernel error), the FSDP root **stays in sync-suppressed state** for the next macro-step. Subsequent backwards on the same engine instance won't reduce-scatter → grads diverge silently across ranks.

Current code (paraphrased):
```python
with engine.train_mode():
    for mb_idx, mb_data in ...:
        if set_grad_sync is not None:
            set_grad_sync(is_last)
        mb_metrics = self._drafter_micro_step(...)   # ← can raise
        accum_metrics.append(mb_metrics)

    if set_grad_sync is not None:
        set_grad_sync(True)                          # ← skipped on exception
    grad_norm = engine.optimizer_step()
```

**Fix:**
```python
with engine.train_mode():
    try:
        for mb_idx, mb_data in ...:
            if set_grad_sync is not None:
                set_grad_sync(is_last)
            mb_metrics = self._drafter_micro_step(...)
            accum_metrics.append(mb_metrics)
        grad_norm = engine.optimizer_step()
        lr = engine.lr_scheduler_step()
    finally:
        # Restore default sync behavior even if anything in the loop raised.
        if set_grad_sync is not None:
            set_grad_sync(True)
```

Trivial surgical fix. Should be applied unconditionally.

---

## 2. Real but lower severity

### 🟡 #3 — Empty-mask in-kernel fallback uses `p.reshape(-1)[0]` on DTensors under FSDP2

**File:** `recipe/drafter_cotraining/eagle3/eagle3_model.py:107-113`

```python
if valid_idx.numel() == 0:
    total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
    zero = total * 0.0
    return zero, zero.detach()
```

Under FSDP2, midlayer params are DTensors (the root unit's params are DTensors too, just gathered). `dtensor.reshape(-1)[0]` semantics:
- It returns a DTensor scalar (value depends on which rank's local shard contains element 0).
- `total = sum(...)` adds across params → tensor sum across DTensors of mixed placements.
- Backward through this should still touch every param → reduce-scatter still fires correctly.

In practice this should work, but it's not exercised by the smoke (response_lens ≥ 2 means TTT-shifted mask never goes empty for any sample). The path is a **safety net** for the case where, mid-TTT-loop, the shifted mask becomes all-zeros on one rank but not another (then FSDP needs every param to participate in the reduce-scatter or it deadlocks).

**Fix:**
- Add a unit test that triggers `valid_idx.numel() == 0` under FSDP2 (using `test_fsdp2_drafter_wrap.py` fixture).
- If the test fails, replace with an explicit dummy-grad pattern that's known safe under DTensor (e.g., loop over `model.parameters()` and call `p.grad = torch.zeros_like(p)` — but that may also hit DTensor concerns; needs experimentation).

Low priority — unlikely to fire in production with response-only masks but worth nailing down.

---

### 🟡 #4 — Frozen target weights loaded on every rank without explicit broadcast

**File:** `recipe/drafter_cotraining/drafter_engine.py:_load_target_frozen_weights` (~line 270-314)
**Also:** `recipe/drafter_cotraining/drafter_engine.py:_build_module:225` (`draft_model.load_embedding(...)` per-rank)

Both `lm_head_w` / `norm_w` (in `_load_target_frozen_weights`) and `embed_tokens.weight` (in `_build_module`) are loaded from disk on every rank independently. For local checkpoints (our smoke case) this is fine — `safetensors` reads are deterministic.

For production with **HF Hub `snapshot_download`**, two ranks could see different bytes:
- Partial download on one rank.
- Concurrent file open with cache races.
- Hub API returning slightly different files due to mirror differences (rare but possible).

If ranks have different frozen weights → silent training divergence (loss curves diverge across ranks; FSDP reduce-scatter averages them but the macro-batch is corrupt).

**Fix (defensive):**
```python
if dist.get_rank() == 0:
    weights = _load_tensors_from_model_path(target_model_path, [...])
    lm_head_w = weights["lm_head.weight"].to(device, dtype)
    norm_w = weights["model.norm.weight"].to(device, dtype)
else:
    lm_head_w = torch.empty((vocab_size, hidden_size), dtype=dtype, device=device)
    norm_w = torch.empty((hidden_size,), dtype=dtype, device=device)
dist.broadcast(lm_head_w, src=0)
dist.broadcast(norm_w, src=0)
```

Same pattern for `embed_tokens.weight` in `_build_module` (or rely on the post-wrap `fsdp2_load_full_state_dict` broadcast which already handles it for that one).

---

### 🟡 #5 — `micro_size=0` is a runtime footgun

**File:** `recipe/drafter_cotraining/engine_workers.py:update_drafter:257-260`

```python
micro_size = int(
    self.config.drafter.engine_config.get("micro_batch_size_per_gpu", 1)
)
accum_steps = max(1, math.ceil(len(mooncake_keys) / micro_size))
```

If `micro_size == 0` (config typo), `math.ceil(N / 0)` raises `ZeroDivisionError` at runtime. If somehow we reach `_iter_micro_batch_keys`, `range(0, n, 0)` raises `ValueError`.

**Fix:** one-line assert at the top of `update_drafter`:
```python
assert micro_size > 0, (
    f"drafter.engine_config.micro_batch_size_per_gpu must be > 0, got {micro_size}"
)
```

Trivial.

---

## 3. Doc-only notes (not bug fixes)

### 📝 #6 — TTT-shifted mask vs `mb_valid` divisor

**File:** `recipe/drafter_cotraining/engine_workers.py:_drafter_micro_step`

`mb_valid` is computed from the **initial** `loss_mask` (or `position_mask`). The forward kernel divides by `valid_idx.numel()` per TTT step. If the per-step `valid_idx.numel()` differs from `mb_valid` (because the mask shifted left and 1s dropped off the left edge), the `scale = mb_valid / total_valid_global` is mathematically slightly wrong.

For our **response-only mask** `[0,…,0, 1,…,1, 0,…,0]`, shifting drops 0s from the left edge → 1-count is preserved → `mb_valid == valid_idx.numel()` for every TTT step. **Bug doesn't manifest.**

For **SFT-style masks** (1s starting at position 0), shifting could drop 1s → `valid_idx.numel() < mb_valid` → kernel's `.mean()` divides by less → loss term is slightly larger than intended. Effect is at most `length / mb_valid` fractional skew per step.

**Fix:** add a comment in `_drafter_micro_step` explicitly noting the assumption: "mb_valid assumes the response-only mask shape (1s in the middle of the sequence). For SFT masks with 1s at position 0, this divisor is approximate."

Or fix properly: pass `valid_idx.numel()` back from the kernel via the metrics dict and use that as the per-step divisor. Adds a layer of plumbing.

---

### 📝 #7 — `PrecomputedTarget.position_mask ⊆ loss_mask` is implicit

**File:** `recipe/drafter_cotraining/eagle3/eagle3_model.py:48-65`

The dataclass docstring says position_mask is "subset of loss_mask, only positions whose verifier-argmax token falls in V_draft." The code in `compute_target_p_padded` enforces this by construction (position_mask is built from loss_mask AND the t2d argmax check). But there's no `assert` and a future refactor could break the invariant.

**Fix:** add to the docstring: "Invariant: when set, every position with `position_mask[b, t] == 1` also has `loss_mask[b, t] == 1`." Optionally add a debug-mode assert in `_calculate_loss`.

---

## 4. Confirmed NOT bugs (defending the design)

The following were flagged but are correct by design. Documented here so future reviews don't re-flag them.

### Phase A (kernel + Eagle3Model)

| Flagged | Status |
|---|---|
| "`loss_mask` unused in no-pruning branch of `compute_target_p_padded`" | **Correct.** target_p is built for ALL positions outside the kernel; `valid_idx` (derived from loss_mask in `_calculate_loss`) filters inside the kernel. Loose coupling is intentional — keeps `compute_target_p_padded` simple |
| "Accuracy meaningless under bf16 target_p" | **No.** argmax is invariant to monotonic transforms + bf16 rounding. Argmax of bf16 softmax matches argmax of fp32 softmax in nearly all cases (rounding can flip when two logits are within ~bf16 ULP, which is rare in a non-degenerate distribution) |
| "bf16 softmax overflow" | **No.** Softmax output is bounded in [0, 1]; cannot overflow |
| "verifier_norm applied AFTER padding shifts semantics" | **No.** Padding(left=False) appends a zero at the right edge. RMSNorm of a zero vector → zero output (`0 / sqrt(0 + eps) * weight = 0`). The appended-zero position remains zero after norm, then is filtered out by `valid_idx` downstream. No semantic issue |

### Phase B (FSDP2 wrap)

| Flagged | Status |
|---|---|
| "Missing `reshard_after_forward` in fsdp_kwargs" | **Correct by design.** We deliberately don't pass it so PyTorch's auto-root-detection takes over. Sub-units default to `True`, root auto-`False`. This is the entire point of the override; see `[[FSDP2 Wrap — TorchSpec Cross-Reference]]` and `verl/utils/fsdp_utils.py:734-766` |
| "State dict capture on all ranks corrupts broadcast" | **Probably fine.** `set_model_state_dict(broadcast_from_rank0=True)` uses rank 0's state authoritatively; other ranks' copies are wasteful but not corrupting. Empirically smoke ran clean. Could be tightened to rank-0-only for memory cleanliness |
| "`cast_forward_inputs=True` conflicts with kernel's `.float()` cast" | **No.** `cast_forward_inputs` casts FSDP-unit-boundary inputs to bf16 (`param_dtype`); the kernel's `.float()` upcasts to fp32 for numeric stability inside RMSNorm/softmax. Different concerns, no double-cast |
| "Module unwrap pattern `self.module.module if hasattr ... else self.module`" | **Vestigial but harmless.** Was needed for FSDP1 `FullyShardedDataParallel(module).module` access. Under FSDP2 (in-place wrap), `self.module` IS Eagle3Model and the hasattr returns False. Could be simplified to `self.module` post-FSDP1 deprecation |
| "`sync_frozen_modules_from_actor` is a no-op" | **Known/documented.** Not a bug — explicit deferred follow-up for the multi-actor production case |
| "Assert at `_build_fsdp_module` is too late" | **Fine.** The `@EngineRegistry.register(backend=["fsdp2"])` already routes away from this engine if strategy=`fsdp`. The assert is a defensive belt-and-suspenders check |

### Phase C (micro-batching)

| Flagged | Status |
|---|---|
| "Double-weighting in `_aggregate_micro_metrics`" | **False alarm.** Backward uses `scale = mb_valid / total_valid` applied once. Metric aggregation uses RAW `p.detach()` per-mb means reweighted by mb_valid for **reporting only** — not double-applied. The two are separate computation paths |
| "Loss-mask vs position-mask divisor mismatch" | **False.** `mb_valid` is computed from `position_mask` if set, matching what the kernel's `valid_idx` counts. Code is consistent (subject to the TTT-shift caveat in #6 above) |
| "`bucket_size_override` semantics backwards" | **No.** We pass the global all-reduced MAX. `max(local_max, override)` correctly degenerates to `override` because local_max ≤ global_max by construction. The collator's behavior is right |
| "`del prepared, plosses, ...` is cargo cult" | **True but harmless.** Backward already releases autograd refs; vars go out of scope at next iteration. The `del` is cosmetic. Could be removed for cleanliness; not a correctness issue |
| "Shallow copy of `meta_info`" | **Low risk.** `meta_info` contains immutable config-like values (`mooncake_cfg` dict, etc.); we don't mutate them downstream. Could deepcopy for safety; not load-bearing |
| "`len(accum_metrics) > 0` not validated" | **False alarm.** `if not accum_metrics: return ...` early-returns at the top of `_aggregate_micro_metrics` |

---

## 5. Recommended next actions

1. **Apply fixes #1 and #2 immediately.** Both are surgical edits to `engine_workers.py`. Preferred:
   - #1 → Option A (move filter pre-dispatch) if the trainer driver allows; else Option B (pad to global accum_steps with no-op iterations).
   - #2 → Add try/finally around the `set_grad_sync` lifecycle.

2. **Add coverage for #3 (empty-mask under FSDP2).** Extend `test_fsdp2_drafter_wrap.py` with a T6 test that triggers `valid_idx.numel() == 0` and verifies `loss.backward()` doesn't deadlock or error.

3. **Defer #4, #5 to a separate hardening PR.** Both are low-impact in our current setup but worth fixing before any production deployment that involves HF Hub downloads or user-overridable batch sizes.

4. **Add the doc-only notes (#6, #7) inline as comments** so future readers don't re-flag them.

5. **No action on the "confirmed NOT bugs" list.** Document in this file as the authoritative reference if the questions come up again.

---

## 6. Process notes — what worked / what didn't

**What worked:**
- Three parallel agents covering distinct scopes was efficient.
- Asking explicitly for "no praise lists, real issues only" cut down on noise.
- Triaging agent output against actual design intent (cross-referencing the design docs) caught most of the false alarms.

**What didn't:**
- Agents tend to flag patterns they don't recognize as "potentially buggy" even when they're deliberate. ~40% of flagged items were false alarms based on incorrect understanding of FSDP2 semantics, torch.compile's auto-promotion rules, or the design intent of loose coupling between target builder and loss kernel.
- Agents had no access to the `[[FSDP2 Wrap — TorchSpec Cross-Reference]]` and `[[Loss Kernel Choice — Lazy vs Precomputed]]` design docs that would have answered most of the false-alarm questions. Future reviews should pass these as context.

---

## Related

- [[Drafter Micro-Batching Concrete Plan]] — the refactor itself (Phases A-E)
- [[FSDP2 Wrap — TorchSpec Cross-Reference]] — answers most of the "Phase B" false alarms
- [[Loss Kernel Choice — Lazy vs Precomputed]] — answers most of the "Phase A" false alarms
- [[LazyTarget vs Precomputed Memory Analysis]] — original memory-side rationale
