# Eagle3 AutoKernel — Lessons (session 2026-04-29)

## Round 3 (full-pipeline fusion + custom Triton scale)

bench.py now also reports TTT-7 (test-time-training, 7 step) memory which is
the production-faithful metric.

### Final state vs both baselines (round 3)

| Metric | Round 2 final | Round 3 final |
|---|---|---|
| `step_speedup_vs_compiled` (large) | 1.019x | **1.041x** |
| `step_speedup_vs_compiled` (prod V=152K) | 1.015x | **1.041x** |
| `step_speedup_vs_pytorch` (large) | 1.363x | 1.388x |
| transient_kernel_step_vram (large) | 141 MB | 141 MB |
| transient_kernel_step_vram (prod) | 313 MB | 313 MB |
| **`mem_ratio_ttt7_kernel_over_compiled` (large)** | n/a | **1.004** (≈ tied) |
| **`mem_ratio_ttt7_kernel_over_compiled` (prod)** | n/a | **0.965** (we win 3.5%) |

### What worked in round 3

- **exp 29: Custom Triton scale-in-place for `grad_lm_head`.**
  aten's `vectorized_elementwise_kernel` for `grad_lm_head.mul_(d_loss)` runs
  at ~65% of peak BW on bf16. A bespoke Triton kernel with `BLOCK_SIZE=16384`,
  `num_warps=8` lifts that to ~86% peak. **+1.6% step at prod.**

- **exp 30: Single autograd Function owning the full pipeline.**
  Gather + RMSNorm + matmul + Triton kernel + grad precompute now live in
  one `_Eagle3FullFn.forward`. Backward uses a custom Triton kernel that
  fuses RMSNorm-bwd + scatter-to-`grad_prenorm_hs_flat` + `d_loss` scale on
  `grad_norm_hs` + `atomic_add` reduction for `grad_norm_weight`.
  Eliminates `IndexSelectBackward` and `FusedRmsNormBackward` from the
  autograd graph. Speed roughly tied with prior version, but consolidates
  the architecture and unblocks further memory wins.

- **exp 31: Scale-in-place kernel BLOCK=16K, warps=8** (was 4K/4).
  Larger block + more warps → fewer launches and better BW utilization.
  **+1.3% step at large.**

### What didn't work in round 3 (reverted)

- **exp 28: `@triton.autotune`** — at small V (e.g. tiny V=256) some configs
  produced numerical errors on Pass 1 (probably an interaction with
  `BLOCK_SIZE >> n_cols`, `tl.argmax` over -inf-padded tile and the
  reduction across V-blocks). Smoke fails. Triton 3.6 quirk worth a deeper
  look later.
- **exp 32 / 32b: `BLOCK_SIZE=8K`** with `num_warps=4` and `=8` — slight
  regression vs (16K, 16).
- **exp 33: scale-in-place BLOCK=32K, warps=16** — too large; per-program
  register pressure tanks scheduling.
- **exp 33b: explicit `num_stages=2` on inner kernel** — wash.
- **exp 34: rms_norm bwd `num_warps=16`** — slight regression at large.
- **exp 35: re-gather `hs` from `prenorm_hs_flat` in bwd** (avoid saving
  (N, H) hs buffer). Saves 16 MB step VRAM and 6% TTT-7 at large, BUT bwd
  re-gather introduces ~330µs overhead → step slowed 2.1% at large.
  Violates the "no slowdown" rule. The trade is real but not
  unconditionally a win — the `hs` save is cheaper than re-gather at our
  shapes. Worth revisiting if memory pressure is the hard constraint.

### Architectural decision: matmul partition

The matmul stays in `torch.matmul` (cuBLAS). I considered fusing the
forward matmul + Triton kernel + backward matmul-1 into a single
`tl.dot`-based kernel. Analysis: each program would need to read
`lm_head_w` blocks twice per row (online softmax + recompute pass).
With BLOCK_N=1, that's `2 × N × V × H` reads vs cuBLAS's `2 × V × H`
(cuBLAS amortizes via tensor-core tiling) — at prod that's ~128× more
lm_head_w bandwidth. With BLOCK_N=16, still ~8× more. Triton `tl.dot`
also lands lower than cuBLAS on Ampere for our shapes. The Liger
partition (cuBLAS matmul + Triton loss kernel) is correct for this
kernel.

### Bench harness note

`reference.py` now returns a tuple `(loss, acc)` (matching production's
`compiled_forward_kl_loss`), but `bench.py::_compare` accesses `.shape`
on the expected output and crashes with `AttributeError: 'tuple' object
has no attribute 'shape'`. Smoke test reports `correctness: FAIL` for
all kernels (including pre-round-3 commits). Correctness was verified
manually:
- forward losses match reference within bf16 tolerance at all sizes
- `prenorm_hs_flat.grad`, `norm_weight.grad`, `lm_head_weight.grad`
  max-abs-error vs reference: 3e-8, 2e-6, 2e-6 (within bf16 noise floor)
The harness needs `_ref_eagle3_loss` to wrap the tuple in `torch.stack`
before passing to `_compare`. Out of scope per `program.md` ("no" for
modifying bench.py / reference.py).

---

## Round 2 (after `@torch.compile` baseline added)

bench.py now compares against `@torch.compile(reference)` as the
production-shipping baseline. Bar to ship: **`step_speedup_vs_compiled ≥ 1.0`**.

### Final state vs both baselines

| Metric | Round 1 final | Round 2 final |
|---|---|---|
| `step_speedup_vs_pytorch` (large) | 1.337x | **1.363x** |
| `step_speedup_vs_compiled` (large) | ~1.005x | **1.019x** ✓ ships |
| `step_speedup_vs_compiled` (prod V=152K) | ~0.996x | **1.015x** ✓ ships |
| transient_kernel_step_vram (large) | 141 MB | 141 MB |
| transient_kernel_step_vram (prod) | 313 MB | 313 MB (vs compile 593 MB → **47% less**) |

### What worked in round 2

- **exp 18: fuse argmax(target_p) into Pass 2** of the inner kernel.
  target_p ∈ [0, 1] so masked-with-0 slots can never beat valid slots; the
  argmax over the whole tile is correct without a separate -inf load. Drops
  Pass 1b entirely (a full V-walk over target_p). **+1.2% step at large.**

- **exp 24: BLOCK_SIZE=16384, num_warps=16** (was 32768, 32). With Pass 1b
  removed and the kernel now 2-pass instead of 3-pass, the smaller block /
  fewer warps fit better — less register pressure, better scheduling.
  **+1.5% step at large**, +0.5% at prod.

### What didn't work in round 2 (reverted)

- **`tl.exp2` instead of `tl.exp`** — Triton already lowers efficiently. Wash.
- **`num_stages=4` and `=3`** — slight regression at prod, flat at large.
- **`MAX_FUSED_SIZE=65536`** — register pressure tanks prod by 20%.
- **Absorb `inv_N` into kernel** — wash (one fewer host op).
- **Single-chunk at prod (budget=1024MB)** — speed flat, but doubles prod
  step VRAM (313 → 609 MB). The chunked version retains a 47% memory edge
  over `@torch.compile` at prod scale.
- **`BLOCK_SIZE=8192` / `num_warps=8` / `num_warps=4`** — all marginal regressions vs (16384, 16).
- **`@torch.compile(_gather_rmsnorm)` prefix fusion** — slight regression at
  large; inductor's fused backward path adds overhead it doesn't recoup.

### Why we win/lose vs `@torch.compile`

| | our kernel | `@torch.compile` |
|---|---|---|
| Forward partition | 3 matmuls + 1 fused Triton kernel (precompute grads) | 1 matmul + 1 inductor kernel |
| Backward partition | 2 in-place `mul_` (just scale saved grads) | 2 matmuls + 1 inductor kernel (recompute) |
| **Total matmul work** | **3** | **3** |

Net step time is fundamentally similar — same 3 matmuls, similar fused
kernels. The advantage flips with size:

- **small/medium (V≤8K)**: we win 1.04–4.9× because chunking + custom
  Function dodges compile's per-call dispatcher overhead.
- **large (V=32K)**: tied (1.019×) — matmul throughput dominates and cuBLAS
  is the same on both sides.
- **prod (V=152K)**: tied speed (1.015×) but **47% less step VRAM** because
  our chunked in-place buffer avoids materializing the full (N, V) logits
  that compile saves for backward.

### Things to try next (untried this round)

- Custom RMSNorm + gather fused Triton kernel (replaces eager `F.rms_norm`
  + `index_select` with one custom autograd.Function). Profile shows
  `IndexSelectBackward` + `FusedRmsNormBackward` together cost ~430us at
  large (~2.7% of step) — moderate ceiling.
- `@triton.autotune` over a small (BLOCK_SIZE, num_warps, num_stages) grid.
- Persistent kernel pattern (Stage 7 of program.md playbook) — launch
  `SM_count` programs and loop over rows internally; may reduce launch
  overhead at prod where we have 1024–2048 row programs.

## Final state (relative to baseline exp 0)

| Metric                              | Baseline | Final (exp14) | Δ            |
|-------------------------------------|----------|---------------|--------------|
| step_speedup_vs_pytorch (large)     | 0.214x   | 1.337x        | +6.2x        |
| step_speedup_vs_pytorch (prod)      | 0.096x   | ~1.30x        | +13.5x       |
| transient_kernel_step_vram (large)  | 1005 MB  | 141 MB        | −7.1×        |
| transient_kernel_step_vram (prod)   | 4294 MB  | 313 MB        | −13.7×       |
| mem_ratio_step (large)              | 1.835    | 0.257         | use 26% mem  |
| mem_ratio_step (prod)               | ~1.77    | ~0.13         | use 13% mem  |

## What worked (kept)

1. **Drop per-chunk `.float()` casts on lm_head_w and norm_hs** (exp 1).
   Each chunk allocated/copied 512 MB-2 GB of fp32 weights. Native bf16 matmul
   already uses tensor-core fp32 accumulation. **+5x fwd, +4.5x step at large**.

2. **Memory-budget chunking instead of Liger's `N*H/V` formula** (exp 3).
   At V=152K the formula gives chunk_size=64 — too thin for tensor cores. A
   512 MB bf16 byte budget yields ≥1024-row chunks at prod. **+24% fwd at large,
   +3x at prod.**

3. **Match Liger's `num_warps=32`** — kernel is "quite sensitive to num_warps"
   (cross_entropy.py:410). MAX_FUSED_SIZE=32768 (their default).

4. **`torch.matmul(out=...)` + skip `zeros_like` on grad_lm_head** (exp 5).
   Saves a 1.2 GB cudaMemset at prod plus a per-chunk allocation on grad_norm_hs.
   **+2.8% step.**

5. **`F.rms_norm` instead of manual RMSNorm** (exp 7).
   Single fused op replaces cast/pow/mean/rsqrt sequence. **+7.6% step, −15%
   step VRAM at large.**

6. **In-kernel target_p gather via `valid_idx`** (exp 8).
   Pass `target_p_flat + valid_idx` to the kernel; gather inside. Eliminates
   the (N, V) gathered tp intermediate (~622 MB at prod). **−23% fwd VRAM.**

7. **In-place `mul_(d_loss)` in backward** (exp 9).
   The saved (V, H) grad_lm_head is the dominant tensor; multiplying it in-place
   avoids a temp allocation. **Step VRAM 266 → 141 MB at large (−47%).**

8. **Pre-allocated logits buffer reused across chunks** (exp 14).
   `torch.empty((chunk_size, V))` outside the loop, `out=logits_buf[:cn]` inside.
   **Prod step VRAM 609 → 313 MB (−49%).**

## What didn't work (reverted)

- **exp 2 / exp 10: move argmax(target_p) or argmax(logits) outside Triton
  kernel.** Eager argmax adds kernel-launch overhead that exceeds the inner
  loop savings — the inline argmax tracking is essentially free since we're
  already iterating V.

- **exp 11: bump chunk budget to 1024 MB** (single-chunk at prod). Marginal —
  cuBLAS doesn't get much from the bigger matmul once chunk_size ≥ 256.

- **exp 12: fuse argmax(target_p) into Pass 2.** Adds register pressure
  in the hot loop; net regression at prod.

- **exp 13: num_warps=16.** Slight regression. 32 is the sweet spot per Liger.

- **exp 15: MAX_FUSED_SIZE=16384.** Smaller V-blocks → more loop iterations
  without compensating compute density gains.

## Architecture invariants (what should NOT change)

- The 3-matmul forward (forward + 2 backward matmuls precomputed) is the
  optimal trade vs eager: it's compute-equivalent to step but enables saving
  *only* (V, H) and (N, H) grads instead of (N, V) activations.

- Triton kernel writes `d_logits` IN-PLACE into the materialized logits
  buffer. Anything else doubles peak chunk memory.

- `save_for_backward` saves only pre-computed grads — never the (N, V)
  logits or log_p.

- `backward` returns must include `None` slots for non-trainable inputs
  (target_p_flat, valid_idx).

## Why forward looks slow (step is what matters)

- fwd_speedup_vs_pytorch ≈ 0.59x at large because we do 3 matmuls in forward
  vs eager's 1.
- step_speedup_vs_pytorch is 1.337x because backward is essentially free
  (one in-place mul vs eager's log_softmax_bwd + 2 matmuls).
- Total matmul work is identical; we just shifted work from backward to
  forward to enable the memory savings.
