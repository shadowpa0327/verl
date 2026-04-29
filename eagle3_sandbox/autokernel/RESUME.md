# How to resume optimization on this kernel

This file is the in-tree resume guide. It mirrors what's in
`~/.claude/projects/<this repo>/memory/project_eagle3_autokernel.md` so the
context travels with the code and any future contributor (human or AI) can
pick up without re-derivation.

## Where state lives

| File | Role |
|---|---|
| `kernel.py` | Current best version (always at `git HEAD`). |
| `program.md` | Rules, ship bar, Liger inspiration map. **Do not modify.** |
| `lessons.md` | Ledger of every kept *and* reverted experiment, with the why. |
| `workspace/results.tsv` | Per-experiment scoreboard (step_us, fwd_us, vram, speedup, note). |
| `workspace/run_exp*.log` | Raw bench output for each numbered experiment. |
| Git history (`exp 0`..`exp NN`) | Every kept change is a commit; reverted ones live only in `lessons.md`. |

To inspect at a glance:
```bash
git log --oneline | grep "exp "
cat workspace/results.tsv
```

## How to resume in a fresh Claude session

Say something like:
> "Continue optimizing `eagle3_sandbox/autokernel/kernel.py`. Read `RESUME.md`, `program.md`, `lessons.md`, and `workspace/results.tsv` first; then re-baseline and propose the next experiment."

Claude will:
1. Read the four files above.
2. `git log --oneline | head -20` for the experiment trail.
3. `python bench.py` once to re-baseline (kernel.py at HEAD is best-known good).
4. Propose experiments grounded in the *unkept* ideas at the bottom of
   `lessons.md` ("Things to try next") rather than re-trying things already
   in the reverted list.

## Current ship status (2026-04-29, end of round 3, exp 31 + bench `prod_b2`)

| Size | step_x_compiled | TTT-7 mem ratio (kernel/compile) |
|---|---|---|
| tiny | 1.36x | 1.42 |
| small | 1.43x | 1.02 |
| medium | 1.06x | 0.60 |
| large | 1.04x | 1.004 |
| **prod (B=1, T=4096)** | **1.04x** ✓ | 0.965 (we win 3.5%) |
| **prod_b2 (B=2, T=4096)** | **1.03x** ✓ | **~0.5** (we win ~50% on TTT-7 mem) |

Bar to ship is `step_speedup_vs_compiled >= 1.0` at all sizes. Currently met everywhere.

## Architectural invariants — DO NOT UNDO

1. **Matmul stays in `torch.matmul` / cuBLAS.** Fusing matmul into Triton via
   `tl.dot` blew up `lm_head_w` bandwidth ~128× in our analysis (cuBLAS
   amortizes lm_head_w across rows in a tile; Triton row-per-program does
   not). Liger's `fused_linear_jsd` makes the same partition.
2. **3 matmuls in forward (precompute grads), 0 in backward.** Backward is
   just custom Triton scale-in-place + RMSNorm-bwd-with-scatter. Total
   matmul work is identical to `@torch.compile`'s recompute pattern, but
   step time wins because we skip eager's softmax_bwd kernel + (N, V) buffer.
3. **Kernel writes `d_logits` IN-PLACE into the materialized logits chunk
   buffer.** Never `save_for_backward(logits)` or `save_for_backward(log_p)` —
   that's exactly what we're trying to avoid.
4. **`_Eagle3FullFn` (one autograd Function) owns the whole pipeline since
   exp 30** — gather + RMSNorm + matmul + KL + grad precompute. Backward is
   mul + RMSNorm-bwd + scatter via custom Triton kernels.
5. **`kernel_fn` returns `torch.stack([loss, acc])`** (tensor `(2,)` fp32).
   Do NOT change to a tuple even though `reference.eagle3_loss_ref` returns
   a tuple — the contract in `program.md` is for a tensor.

## Known bench bug

`bench.py::_compare` calls `expected.shape`, but `reference.py` returns a
`(loss, acc)` tuple, so smoke prints `correctness: FAIL` for *every* kernel
(including pre-round-3 commits). Correctness should be verified manually:

```python
import kernel as K, reference as R, bench as B, torch
inputs = B.gen_eagle3_loss_inputs({"N_full": 4096, "H": 4096, "V": 32000, "rho": 0.5}, torch.bfloat16, "cuda", seed=42)
gi = B._make_grad_inputs(inputs)
out = K.kernel_fn(**gi); out[0].backward()
gi_ref = B._make_grad_inputs(inputs)
ref_loss, _ = R.eagle3_loss_ref(**gi_ref); ref_loss.backward()
for k in ["prenorm_hs_flat", "norm_weight", "lm_head_weight"]:
    err = (gi[k].grad.float() - gi_ref[k].grad.float()).abs().max().item()
    print(f"  grad[{k}]: max_abs_err={err:.4e}")
```

Expected (bf16 noise floor): prenorm 3e-8, norm_w 2e-6, lm_head 2e-6.

The fix is for `_ref_eagle3_loss` in `bench.py` to wrap the tuple in
`torch.stack` before passing to `_compare`. `program.md` says don't modify
`bench.py`, but the `prod_b2` test-size addition (commit `26d9a88c`) was
authorized by the user. Don't fix `_compare` without similar authorization.

## Production scale reminder

`bench.py` `prod` row uses `N_full = 4096` = B=1, T=4096 (one micro-batch
per GPU per Qwen3-8B pretrain default). The realistic case the user runs
is `prod_b2` (`N_full = 8192` = B=2, T=4096). Tune for `prod_b2` when
trade-offs differ between sizes — that's the row that matters.
