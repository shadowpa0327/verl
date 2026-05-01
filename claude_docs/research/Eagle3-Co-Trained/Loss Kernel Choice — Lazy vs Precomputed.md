---
title: Loss Kernel Choice — Lazy vs Precomputed
date: 2026-04-26
tags:
  - eagle3
  - drafter
  - verl
  - torch-compile
  - kernels
aliases:
  - compiled_forward_kl_loss vs from_hs
  - LazyTarget vs PrecomputedTarget
  - Drafter loss kernel decision
---

# Loss Kernel Choice — Lazy vs Precomputed

> [!summary] TL;DR
> The pre-refactor code shipped **two compiled forward-KL loss kernels** (`compiled_forward_kl_loss` for the precomputed/pruning path; `compiled_forward_kl_loss_from_hs` for the lazy/no-pruning path) and **two target dataclasses** (`PrecomputedTarget` / `LazyTarget`) wired through `Eagle3Model._calculate_loss` via an `isinstance` dispatch. The lazy path materializes target probs *inside* the compile graph to avoid a (B, T, V_full) resident tensor; the precomputed path materializes them *outside*, sized by V_draft. The lazy path turns out to have **three structural compile-graph problems under micro-batching** that make it unusable for accumulation. The refactor deletes the lazy kernel + dataclass, generalizes the precomputed path for the no-pruning case (`t2d=None`), and switches `target_p` storage to bf16 to halve resident memory at any ρ. **One kernel, one factory, one path.**

> [!info] Related
> - [[LazyTarget vs Precomputed Memory Analysis]] — memory crossover analysis with numbers
> - [[Drafter Micro-Batching Refactor Plan]] §3 — the three compile-graph hazards in context
> - [[Drafter Micro-Batching Concrete Plan]] §3 — Phase A code-level edits

---

## 1. The two kernels — what each computed

Both kernels live(d) in `recipe/drafter_cotraining/eagle3/ops/loss.py` and were `@torch.compile(dynamic=None)`-decorated. They share the same RMSNorm + draft-`lm_head` + log-softmax + forward-KL math; the difference is **whether the target distribution `tp` is an input or computed inside the graph**.

### `compiled_forward_kl_loss` (precomputed path) — KEPT

```python
@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,   # (B*T, H)        bf16   — draft pre-norm HS
    target_p_flat,                # (B*T, V_out)    bf16   — pre-built target probs
    valid_idx,                    # (N,)            int64  — non-masked positions
    norm_weight,                  # (H,)            bf16   — draft RMSNorm weight
    lm_head_weight,               # (V_out, H)      bf16   — DRAFT lm_head weight
    norm_eps,                     # ()              float
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)   # (N, H)
    tp = target_p_flat.index_select(0, valid_idx)                # (N, V_out)

    # RMSNorm in fp32
    hs_f32   = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd     = torch.rsqrt(variance + norm_eps)
    norm_hs  = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)                   # (N, V_out)

    # Forward KL: −E_{tp}[log p_draft]
    log_p = F.log_softmax(logits.float(), dim=-1)                # fp32
    loss  = -(tp * log_p).sum(-1).mean()                         # auto-upcasts tp
    acc   = (logits.argmax(-1) == tp.argmax(-1)).float().mean()
    return loss, acc
```

Six tensor inputs, all with **constant shape** modulo `valid_idx.shape[0] = N` (which is `mark_dynamic`'d outside the kernel). The target side is a single `(B*T, V_out)` tensor that the caller built earlier.

### `compiled_forward_kl_loss_from_hs` (lazy path) — DELETED

Preserved here for historical reference (file deleted from `loss.py`):

```python
@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,   # (B*T, H)         bf16  — draft pre-norm HS
    target_hidden_states_flat,    # (B*T, D)         bf16  — TARGET hidden states
    valid_idx,                    # (N,)             int64
    norm_weight,                  # (H,)             bf16  — draft RMSNorm weight
    lm_head_weight,               # (V_draft, H)     bf16  — DRAFT lm_head weight
    target_lm_head_weight,        # (V_full,  D)     bf16  — TARGET lm_head weight
    norm_eps,                     # ()               float
):
    hs  = prenorm_hidden_states_flat.index_select(0, valid_idx)  # (N, H)
    ths = target_hidden_states_flat.index_select(0, valid_idx)   # (N, D)

    # Target probs computed INSIDE the compile graph
    tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)   # (N, V_full)

    # RMSNorm in fp32 (same as above)
    hs_f32   = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd     = torch.rsqrt(variance + norm_eps)
    norm_hs  = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)                              # (N, V_draft)

    log_p = F.log_softmax(logits.float(), dim=-1)
    loss  = -(tp * log_p).sum(-1).mean()
    acc   = (logits.argmax(-1) == tp.argmax(-1)).float().mean()
    return loss, acc
```

**Seven** tensor inputs (one extra: `target_lm_head_weight`). The target side is **two** tensors, and the kernel computes target_p (`F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)`) inside the compiled region.

### Body diff in one paragraph

The lazy kernel adds `tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)` and adds `target_lm_head_weight` to the input signature. Everything below that line is identical to the precomputed kernel. So the structural difference is "**target_p is a computed intermediate of two graph inputs**" (lazy) vs "**target_p is itself a graph input**" (precomputed).

---

## 2. The two paths — orchestration around the kernels

The kernel choice was driven by which target dataclass the caller built:

### Dataclass definitions (`eagle3_model.py`, before refactor)

```python
@dataclass
class PrecomputedTarget:
    """Pre-computed target probabilities (used with vocab pruning)."""
    target_p_padded: torch.Tensor               # (B, T + length, V_draft)
    position_mask: Optional[torch.Tensor] = None  # (B, T) — argmax-in-V_draft subset

@dataclass
class LazyTarget:
    """Deferred target computation to avoid materializing (B, T, V_full)."""
    hidden_states_padded: torch.Tensor          # (B, T + length, D)
    lm_head_weight: torch.Tensor                # (V_full, D)
```

Both contained tensors padded along the time dim by `length` (TTT steps) so that `_calculate_loss` could slice `[:, idx : idx + seq_length, :]` per TTT step without going out of bounds.

### Factory functions (`eagle3_model.py`)

```python
@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states, target_lm_head_weight, t2d, loss_mask, length, ...
) -> PrecomputedTarget:
    pruned_weight = target_lm_head_weight[t2d]        # (V_draft, D)
    # ... build position_mask via chunked argmax-in-V_draft check ...
    target_logits_pruned = F.linear(target_hidden_states, pruned_weight)
    target_p = F.softmax(target_logits_pruned.float(), dim=-1)        # fp32
    target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)
    return PrecomputedTarget(target_p_padded, position_mask)

def compute_lazy_target_padded(
    target_hidden_states, target_lm_head_weight, length,
) -> LazyTarget:
    return LazyTarget(
        hidden_states_padded=F.pad(target_hidden_states, (0, 0, 0, length), value=0.0),
        lm_head_weight=target_lm_head_weight.detach(),
    )
```

The precomputed factory does the *expensive* up-front work (full-vocab projection, fp32 softmax, padding). The lazy factory does almost nothing — just pads HS and detaches the weight reference.

### Dispatch (`Eagle3Model._calculate_loss`, before refactor)

```python
def _calculate_loss(self, hidden_states, target, mask, idx, seq_length, ...):
    valid_idx = mask.flatten().nonzero().squeeze(-1)
    if valid_idx.numel() == 0:
        # FSDP grad-sync fallback (touched-but-zero grad on every trainable param)
        ...

    torch._dynamo.mark_dynamic(valid_idx, 0)                         # ← N is symbolic
    hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    if isinstance(target, PrecomputedTarget):
        target_p_step = target.target_p_padded[:, idx : idx + seq_length, :]
        tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])
        return compiled_forward_kl_loss(
            hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, norm_eps,
        )
    else:                                                            # LazyTarget
        ths_flat = target.hidden_states_padded[
            :, idx : idx + seq_length, :
        ].reshape(-1, target.lm_head_weight.shape[-1])
        return compiled_forward_kl_loss_from_hs(
            hs_flat, ths_flat, valid_idx,
            norm_weight, lm_head_weight, target.lm_head_weight, norm_eps,
        )
```

Two compile traces, two dispatch arms, two factories. The driver (`prepare_model_inputs` in `drafter_engine.py`) chose between them based on whether vocab pruning was enabled.

---

## 3. The original rationale for two paths

The original split was a **memory tradeoff** keyed off vocab-pruning:

| Setting | V_draft / V_full | Right kernel |
|---|---|---|
| **Pruned** (`draft_vocab_size < vocab_size`) | e.g. 32k / 128k = 1/4 | Precomputed: store `(B, T, V_draft)` target_p resident — small enough |
| **No pruning** (`draft_vocab_size == vocab_size`) | e.g. 152k / 152k = 1 | Lazy: avoid storing `(B, T, V_full)` resident; compute target_p per TTT step |

When the draft and target share the same vocab (no pruning), the precomputed path's resident cost is `(B, T+length, V_full) × 4 B` (fp32). For Qwen3-8B (V=151,936, T_pad=4352, length=7) that's ~2.65 GB **per sample** — uncomfortable at micro_batch=1, prohibitive at larger batches. So the lazy path was designed to cap memory at `(N_valid, V_full) × 4 B` per TTT step (transient, not resident), which scales with the loss-mask density `ρ` rather than the full B·T budget.

Crossover point (from [[LazyTarget vs Precomputed Memory Analysis]]):

```
N_valid · V_full  ≈  (B · T) · V_draft
       ⇒  ρ · V_full  ≈  V_draft
       ⇒  ρ_cross  ≈  V_draft / V_full
```

For pruning with 4× ratio, ρ_cross ≈ 0.25 — below that, lazy wins on memory; above, precomputed wins. For our actual config (V_draft = V_full → ρ_cross = 1.0), the lazy path "wins on memory" at any ρ < 1.

---

## 4. The three compile-graph hazards in the lazy path

Memory aside, the lazy kernel has **three structural problems under `torch.compile` + FSDP + micro-batch accumulation**. Any one of them is enough to cause `Tensor × DTensor` mismatches, recompilation storms, or grad-graph corruption between micro-batches.

### Hazard 1 — `target_lm_head_weight` enters the compile graph as an input

```python
def compiled_forward_kl_loss_from_hs(..., target_lm_head_weight, ...):
    tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)
```

The kernel takes `target_lm_head_weight` as the 6th argument. Because it's a graph input, dynamo's guards observe its dtype, shape, and crucially its **stride / device / sharding state**. Under FSDP:

- **FSDP1 with `use_orig_params=True`** (the original config): `target_lm_head_weight` is a plain `torch.Tensor`. The kernel works.
- **FSDP2 (where we want to go)**: even though `target_lm_head_weight` is engine-owned (`FSDPDrafterEngine._target_lm_head_weight`), if any `lm_head` reference in the graph is a DTensor, the kernel hits `Tensor × DTensor` — `F.linear` doesn't auto-promote.

More subtly: even on FSDP1, if the target weight ever picks up `requires_grad=True` lineage (e.g., accidentally tied to a trainable param), the compiled function would treat it as needing grad tracking. After the first backward, FSDP's reduce-scatter has changed the effective shape; the second micro-batch's backward sees a mismatch and errors with "**RuntimeError: shape mismatch in grad accumulation**".

### Hazard 2 — Two `valid_idx`-dependent matmuls; only one dim marked dynamic

```python
hs  = prenorm_hidden_states_flat.index_select(0, valid_idx)    # (N, H)
ths = target_hidden_states_flat.index_select(0, valid_idx)     # (N, D)
tp  = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)  # (N, V_full)
...
logits = F.linear(norm_hs, lm_head_weight)                     # (N, V_draft)
```

`mark_dynamic(valid_idx, 0)` marks `N` as symbolic. Downstream:
- `tp.shape = (N, V_full)` — depends on `N` (dynamic)
- `logits.shape = (N, V_draft)` — depends on `N` (dynamic)

Both are intermediate shapes that depend on the dynamic dim. With ONE `mark_dynamic`, dynamo can sometimes propagate symbolicity through both. But with the lazy path's extra `F.linear(ths, ...)`, dynamo has more inference work to do, and our smoke debugging surfaced cases where the V dim was incorrectly inferred as a small value (`hint=2`) on recompilation across micro-batches with different shapes:

```
torch._dynamo.exc.TorchRuntimeError: ...
mul(FakeTensor(s75, s87), FakeTensor(s75, 256)):
  RuntimeError('The size of tensor a (s87: hint = 2) must match the size of tensor b (256)
                at non-singleton dimension 1)')
```

The precomputed path has only **one** `valid_idx`-dependent matmul (the `F.linear(norm_hs, lm_head_weight)`), and its V dim is the same `target_p`'s V dim — so dynamo's guards stay consistent.

### Hazard 3 — Target softmax inside the autograd graph

```python
tp = F.softmax(F.linear(ths, target_lm_head_weight).float(), dim=-1)
```

`F.linear(ths, target_lm_head_weight)` produces a tensor that participates in the autograd graph (because at least one of the two inputs may require grad — `ths` comes from `target.hidden_states_padded` which is detached at factory time, but `target_lm_head_weight` may not always be cleanly detached if the engine's reference is reseated mid-training).

`torch.compile` doesn't always honor `requires_grad=False` without an **explicit `.detach()` inside the compiled region** — and the lazy kernel doesn't do that. This means grad can leak through the target path, producing spurious grads on `target_lm_head_weight` that should be zero. Under FSDP this corrupts the reduce-scatter pattern.

The precomputed path sidesteps this: `target_p_flat` is built under `@torch.no_grad()` (factory function decorator) and arrives in the kernel as a leaf tensor with no grad lineage by construction.

---

## 5. Why these matter specifically for micro-batching

The original single-shot path (one forward + one backward per macro-batch) ran the kernel exactly once per TTT step per macro-batch. Recompilation only mattered between *macro*-batches, where the user might tolerate a few warm-up steps.

Under micro-batching with N micro-batches per macro-step, the kernel runs N times per TTT step per macro-step → 7·N times per macro-step. Any compile-cache invalidation or graph-input-shape drift between micro-batches **cascades**: one micro-batch's bad inference becomes the next micro-batch's recompilation, which becomes the third micro-batch's `Tensor × DTensor` error.

The precomputed kernel sidesteps all three hazards:
- Only inputs are `(prenorm_hs_flat, target_p_flat, valid_idx, norm_weight, lm_head_weight, norm_eps)`. No `target_lm_head_weight` in the graph.
- `target_p_flat` is built outside the compile graph, fully detached, with no `valid_idx`-dependent V dim (V is just `target_p_flat.shape[-1]`, statically inferable).
- `tp.requires_grad` is False by construction.

That's the entire reason to delete the lazy path.

---

## 6. The decision

### Delete lazy entirely; generalize precomputed for the no-pruning case

Specifically (Phase A of the refactor):

1. **Delete `compiled_forward_kl_loss_from_hs`** from `ops/loss.py`.
2. **Delete `LazyTarget` dataclass + `compute_lazy_target_padded` factory** from `eagle3_model.py`.
3. **Drop the `isinstance` dispatch arm** in `_calculate_loss` — only the `PrecomputedTarget` branch survives. Type signature becomes `target: PrecomputedTarget` instead of `Union[PrecomputedTarget, LazyTarget]`.
4. **Generalize `compute_target_p_padded`** so `t2d` is `Optional[torch.Tensor] = None`:
   - `t2d not None` → pruning path, projects to V_draft, builds `position_mask`.
   - `t2d is None` → no-pruning path, projects to V_full, `position_mask = None`.
5. **Switch `target_p` storage to bf16**:
   ```python
   target_p = F.softmax(target_logits.float(), dim=-1).to(torch.bfloat16)
   ```
6. **`prepare_model_inputs`** in `drafter_engine.py` calls `compute_target_p_padded(..., t2d=getattr(self, '_t2d_index', None))`. Today `_t2d_index` is unset → no-pruning branch.

The pruning machinery stays in the source as the `else` branch of `compute_target_p_padded` — cheap to keep, future-proof if pruning is enabled later.

### Why bf16 storage closes the memory gap

The single remaining concern was: doesn't deleting the lazy path mean we always pay `(B, T+length, V_full) × 4 B` resident? It would — except we also halve the dtype:

```python
# Before: fp32 target_p
target_p = F.softmax(target_logits_pruned.float(), dim=-1)             # 4 B/elem

# After: bf16 target_p
target_p = F.softmax(target_logits.float(), dim=-1).to(torch.bfloat16)  # 2 B/elem
```

Why this is safe: the kernel does `loss = -(tp * log_p).sum(-1).mean()` where `log_p = F.log_softmax(logits.float(), dim=-1)` is fp32. PyTorch auto-upcasts the bf16 `tp` to fp32 in the multiply, so the loss accumulation stays fp32. The only precision loss is in the *stored* probabilities themselves — bf16's 7-bit mantissa is plenty for values in [0, 1].

For Qwen3-8B (V_full=151,936, T_pad=4352, length=7), per sample:
- fp32 `target_p_padded`: ~1.32 GB
- **bf16 `target_p_padded`: ~0.66 GB**

At `micro_batch_size_per_gpu=1` (the shipped default), only one sample's worth is resident at a time. Comfortable on 80 GB H100.

### Why this is the strict winner for our config

[[LazyTarget vs Precomputed Memory Analysis]] §"Re-doing the memory comparison for our actual config" works through the numbers:

| ρ_loss | Precomputed bf16 | Lazy transient | Winner |
|---|---|---|---|
| 1.0 (SFT-like) | **1.32 GB** | 2.65 GB | Precomputed by 2× |
| 0.5 | 1.32 GB | **1.32 GB** (tied) | tie |
| 0.25 | 1.32 GB | **0.66 GB** | Lazy by 2× |
| 0.10 | 1.32 GB | **0.26 GB** | Lazy by 5× |

For our drafter co-training the loss mask is `mask[plen : plen + rlen - 1] = 1` → ρ = `(rlen-1)/(plen+rlen)`. Typical RL traces have prompt 0.5-1k and response 1-4k → ρ in the 0.5-0.8 band → precomputed bf16 wins on memory.

But even at low ρ where lazy "wins" memory: lazy is broken under micro-batching. The whole point of the refactor is to enable micro-batching → the regime where lazy could be the right call no longer exists for us.

> [!important] Strict-dominance argument
> 1. In our typical operating ρ band (0.5-0.8), precomputed bf16 uses less memory than lazy AND is compile-graph stable.
> 2. In the low-ρ band (< 0.25), lazy uses less memory than precomputed bf16 BUT is unusable under micro-batching.
> 3. We require micro-batching, so condition (2) means lazy is unusable in the only regime where it has an edge.
> 4. ⇒ Precomputed bf16 strictly dominates lazy across every regime we care about. Delete lazy.

---

## 7. The kernel after the refactor

`recipe/drafter_cotraining/eagle3/ops/loss.py` — single function, ~40 lines:

```python
@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    hs_f32   = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd     = torch.rsqrt(variance + norm_eps)
    norm_hs  = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    log_p  = F.log_softmax(logits.float(), dim=-1)
    loss   = -(tp * log_p).sum(-1).mean()
    acc    = (logits.argmax(-1) == tp.argmax(-1)).float().mean()
    return loss, acc
```

Five things to notice:

1. **`index_select` happens INSIDE the compile graph.** Lets the fuser combine gather + RMSNorm + matmul + KL into fewer GPU kernels; no round-trip of `(B*T, V)` through HBM.
2. **`mark_dynamic(valid_idx, 0)` is set OUTSIDE the kernel** (`_calculate_loss:115`) so `N` can vary without recompilation.
3. **Constant-shape inputs only.** `(B*T_pad, H)`, `(B*T_pad, V)`, `(N,)`, `(H,)`, `(V, H)`, `()`. With `T_pad_macro` all-reduced MAX across DP (Phase C), no shape jitter.
4. **No DTensor in kernel inputs.** `lm_head_weight` is the **draft's own** `lm_head` — under our FSDP2 selective wrap (see [[FSDP2 Wrap — TorchSpec Cross-Reference]]) it's a root-unit param that's auto-kept-gathered through forward and backward. The kernel just sees a plain `(V, H)` tensor.
5. **Empty-mask guard outside the kernel.** `_calculate_loss:107-113` returns `sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad) * 0.0` when `valid_idx.numel() == 0` — a zero loss whose backward touches every trainable param. FSDP requires every param to participate in reduce-scatter; this is the standard idiom and is preserved verbatim from TorchSpec.

---

## 8. Verification

### Unit (`recipe/drafter_cotraining/tests/test_eagle3_loss.py`, 14/14 pass)

- `TestCompiledForwardKLLoss.test_matches_reference` — kernel output vs naive python reference within 1e-3.
- `TestCompiledForwardKLLoss.test_perfect_prediction_equals_entropy` — when draft == target, loss → entropy of target.
- `TestComputeTargetPPadded.test_pruning_shapes_and_position_mask` — pruning path output shapes + probability sums.
- `TestComputeTargetPPadded.test_no_pruning_full_vocab` — **new test** for the no-pruning branch (t2d=None), asserts target_p shape (B, T+length, V_full), position_mask is None, dtype is bf16.
- `TestComputeTargetPPadded.test_pruning_returns_bf16` — **new test** that pruning path also stores bf16.
- `TestValidIdxSubsetting.test_forward_kl_*` — six masking patterns (first_half / second_half / strided / random_sparse / single / all); kernel with `valid_idx` filter must equal kernel on pre-filtered tensor.

The `TestLazyVsPrecomputedTarget` class and `_check_forward_kl_from_hs` test methods were deleted (no LazyTarget to compare against).

### End-to-end (16-step Qwen3-8B pretrain smoke)

Loss curve: **12.076 → 8.425 over 16 steps**, simulated_acc_len 0 → 0.10, grad finite throughout, val pass loss_weighted=8.389, checkpoint saved cleanly.

The kernel handles the production V_full=151,936, T_pad=4352, length=7 shape with bf16 target_p storage without any of the `Tensor × DTensor` errors the lazy path's three hazards would have produced.

---

## 9. What got deleted, what got added — file-by-file

### Deleted
- `recipe/drafter_cotraining/eagle3/ops/loss.py` — `compiled_forward_kl_loss_from_hs` function (~40 lines)
- `recipe/drafter_cotraining/eagle3/eagle3_model.py` — `LazyTarget` dataclass (~10 lines)
- `recipe/drafter_cotraining/eagle3/eagle3_model.py` — `compute_lazy_target_padded` factory (~14 lines)
- `recipe/drafter_cotraining/eagle3/eagle3_model.py` — `else: # lazy` branch in `_calculate_loss` (~22 lines)
- `recipe/drafter_cotraining/tests/test_eagle3_loss.py` — `TestLazyVsPrecomputedTarget` class + lazy-half of `_check_forward_kl_from_hs` (~150 lines)

### Generalized / changed
- `recipe/drafter_cotraining/eagle3/eagle3_model.py:compute_target_p_padded` — `t2d` becomes `Optional[Tensor] = None`, branches on `t2d is None` for the no-pruning path; `target_p` stored as bf16.
- `recipe/drafter_cotraining/eagle3/eagle3_model.py:_calculate_loss` — type signature `target: Union[PrecomputedTarget, LazyTarget]` → `target: PrecomputedTarget`; dispatch arm collapsed.
- `recipe/drafter_cotraining/eagle3/eagle3_model.py:Eagle3Model.forward` — same type signature collapse.
- `recipe/drafter_cotraining/drafter_engine.py:prepare_model_inputs` — calls `compute_target_p_padded(..., t2d=getattr(self, '_t2d_index', None))` instead of `compute_lazy_target_padded(...)`.

### Added
- `recipe/drafter_cotraining/tests/test_eagle3_loss.py` — `test_no_pruning_full_vocab`, `test_pruning_returns_bf16`.
- `recipe/drafter_cotraining/eagle3/eagle3_model.py:PrecomputedTarget` docstring rewritten to document both branches (pruning and no-pruning).

---

## 10. Open question — should we ever bring back lazy?

If you ever:
- Train at very-low ρ (say, prompt-only loss with 5% mask density), AND
- Don't need micro-batching (full-batch fits on one rank), AND
- Want to save the ~1.3 GB resident `target_p_padded` per sample,

then lazy could be reintroduced as an opt-in path. The cost would be:
- Re-adding `compiled_forward_kl_loss_from_hs` and `LazyTarget`.
- Re-adding the dispatch arm in `_calculate_loss`.
- A config flag `drafter.eagle3.use_lazy_target` defaulting to False.

But this is speculative — no current workload needs it, and the three compile-graph hazards still apply in their full glory. If we ever bring it back, the right move is probably a **non-compiled** lazy kernel (eager mode for the target softmax) so the compile-graph hazards don't apply.

For now: deleted, simpler, faster, more memory-efficient. No regrets.

---

## Related

- [[LazyTarget vs Precomputed Memory Analysis]] — original memory-side analysis with crossover formula
- [[FSDP2 Wrap — TorchSpec Cross-Reference]] — why FSDP2 selective wrap matters for the kernel's `lm_head_weight` argument
- [[Drafter Micro-Batching Refactor Plan]] §3 — full RCA of why naive micro-batching tripped over the lazy kernel
- [[Drafter Micro-Batching Concrete Plan]] §3 — Phase A code-level edits (this refactor)
