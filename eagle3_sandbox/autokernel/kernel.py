"""
The file the agent modifies -- Eagle3 forward-KL loss kernel skeleton.

Architecture (mirrors Liger-Kernel's LigerFusedLinearJSDFunction with beta=0,
adapted for our target-distribution forward-KL + RMSNorm + index_select pipeline):

    eager (autograd-natural)        custom autograd.Function (Triton + chunked matmul)
    ────────────────────────        ──────────────────────────────────────────────────
    1. index_select (gather)
    2. RMSNorm (eager fp32)         3. Chunked: matmul → loss kernel (Triton)
                                    → in-place d_logits → backprop to {norm_hs, lm_head}
                                    → save pre-computed grads for backward

Memory pattern -- this is the Liger trick:
    - Materialize logits ONLY per chunk (size ≤ chunk_size × V).
    - The Triton inner kernel writes d_logits IN-PLACE into the logits buffer.
    - The matmul backprop (grad_norm_hs += d_logits @ lm_head_w;
                           grad_lm_head += d_logits.T @ norm_hs_chunk)
      runs per chunk, then the chunk is discarded.
    - ctx.save_for_backward stores ONLY the pre-computed grad tensors -- never
      the (N, V) logits or log_p. Backward is just: scale saved grads by d_loss.

Chunk-size formula (from fused_linear_cross_entropy.py / fused_linear_jsd.py):
    inc_factor = ceil(V / H)                     # vocab/hidden inflation
    chunk_size = next_pow2(ceil(N / inc_factor)) # ≤ N rows per chunk
    num_chunks = ceil(N / chunk_size)
    → peak chunk memory = chunk_size × V × dtype ≈ N × H × dtype  (input-bounded)

Optimization extension points (from program.md playbook):
    - The inner Triton kernel `_eagle3_kl_loss_kernel` -- block sizes, num_warps,
      online-softmax fusion, argmax fusion. <-- main agent target.
    - Replace eager RMSNorm with a Liger-style row-wise Triton kernel.
    - Replace eager index_select with a custom gather kernel.
    - Tune MAX_FUSED_SIZE / chunk size heuristics for the target GPU.
"""

KERNEL_TYPE = "eagle3_loss"

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ─── Tunables (agent edits these) ──────────────────────────────────────────
# Cap on the inner kernel's V-block. Liger ships with 32768 on non-HIP and
# notes the kernel is "quite sensitive to num_warps" (cross_entropy.py:410).
MAX_FUSED_SIZE = 32768
KERNEL_NUM_WARPS = 32

# Chunk-memory budget for the in-place (cn, V) bf16 logits buffer.
# Liger's formula chunk_size = next_pow2(N*H/V) keeps chunk memory ≈ N*H, but
# at V=152K that gives chunk_size=64 -- too thin for tensor-core matmul. We
# bound the chunk by an absolute byte budget instead, which yields larger
# chunks at prod (chunk_size≈512 at V=152K) and full N at large (V≤32K).
CHUNK_LOGITS_BYTES_BUDGET = 512 * 1024 * 1024


# ─── Triton kernel: per-row online softmax + KL loss + in-place d_logits ──
# Pattern lifted from liger_kernel/ops/cross_entropy.py (online softmax) and
# fused_linear_jsd.py (target-distribution KL + d_logits = softmax - target).
@triton.jit
def _eagle3_kl_loss_kernel(
    X_ptr,            # (CN, V)  logits chunk; OVERWRITTEN with d_logits
    TP_ptr,           # (B*T, V) full target_p, read-only (gathered via TP_idx)
    TP_idx_ptr,       # (CN,)    int64, row indices into TP_ptr
    loss_ptr,         # (CN,)    fp32, per-row -sum(tp * log_p) (positive)
    correct_ptr,      # (CN,)    fp32, 1.0 if argmax(logits) == argmax(tp)
    n_cols,           # V
    inv_N,            # 1.0 / total valid rows (mean-reduction scaling)
    X_row_stride,
    TP_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row of the chunk. Two passes over V.

    Pass 1: online softmax (running max m, running sum_exp d) over logits AND
            target_p, with running argmax tracking for both. log_d = log(d).
    Pass 2: stream V again, compute log_p = (x - m) - log_d, accumulate
            row_loss += sum(tp * log_p), write d_logits = (softmax - tp) * inv_N
            in-place into X_ptr.

    At end: store per-row -row_loss into loss_ptr, store argmax-agreement
    indicator into correct_ptr.
    """
    row = tl.program_id(0)
    X_row = X_ptr + row * X_row_stride
    # Gather target_p row via valid_idx -- avoids materializing the
    # (N, V) gathered target_p tensor.
    real_row = tl.load(TP_idx_ptr + row)
    TP_row = TP_ptr + real_row * TP_row_stride

    # ── Pass 1a: online softmax + argmax over logits ────────────────────
    m = float("-inf")
    d = 0.0
    arg_x = 0
    val_x = float("-inf")
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(X_row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
        # Argmax tracking (across V-blocks)
        block_arg = tl.argmax(x, axis=0) + i
        block_val = block_max
        update = block_val > val_x
        arg_x = tl.where(update, block_arg, arg_x)
        val_x = tl.where(update, block_val, val_x)
    log_d = tl.log(d)

    # ── Pass 1b: argmax over target_p ───────────────────────────────────
    arg_t = 0
    val_t = float("-inf")
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        tp = tl.load(TP_row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        block_arg = tl.argmax(tp, axis=0) + i
        block_val = tl.max(tp, axis=0)
        update = block_val > val_t
        arg_t = tl.where(update, block_arg, arg_t)
        val_t = tl.where(update, block_val, val_t)
    correct = (arg_x == arg_t).to(tl.float32)
    tl.store(correct_ptr + row, correct)

    # ── Pass 2: loss + in-place d_logits ────────────────────────────────
    row_loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(X_row + offs, mask=mask, other=0.0).to(tl.float32)
        tp = tl.load(TP_row + offs, mask=mask, other=0.0).to(tl.float32)
        log_p = (x - m) - log_d
        row_loss += tl.sum(tp * log_p, axis=0)
        softmax = tl.exp(log_p)
        d_logits = (softmax - tp) * inv_N
        # Mask off out-of-bounds writes (don't smear past V)
        tl.store(X_row + offs, d_logits, mask=mask)

    tl.store(loss_ptr + row, -row_loss)


# ─── Custom autograd.Function: chunked matmul + Triton loss + saved grads ─
class _Eagle3LossFn(torch.autograd.Function):
    """Mirrors LigerFusedLinearJSDFunction (fused_linear_jsd.py) but with:
        - target_p in probability space (not log-space)  → d_logits = softmax - tp
        - inline argmax-equality accuracy
        - returns (loss, acc) tuple instead of single loss
        - no temperature scaling
    """

    @staticmethod
    def forward(
        ctx,
        norm_hs: torch.Tensor,
        target_p_flat: torch.Tensor,
        lm_head_w: torch.Tensor,
        valid_idx: torch.Tensor,
    ):
        """
        Args:
            norm_hs        (N, H)     bf16/fp16 -- post-RMSNorm draft hs
            target_p_flat  (B*T, V)   bf16/fp16 -- ungathered target probs
            lm_head_w      (V, H)     bf16/fp16 -- draft lm_head weight
            valid_idx      (N,)       int64     -- row indices into target_p_flat
        Returns:
            loss (scalar fp32), acc (scalar fp32)
        """
        device = norm_hs.device
        dtype = norm_hs.dtype
        N, H = norm_hs.shape
        V, _ = lm_head_w.shape

        # ── Chunking schedule (memory-budget pattern) ───────────────────
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        # Cap chunk_size by an absolute bf16 byte budget so prod (V=152K)
        # gets ≥256-row chunks instead of the 64 rows that Liger's V-relative
        # formula produces. At V≤32K chunk_size lands at N (single matmul).
        elem_bytes = max(1, dtype.itemsize)
        budget_rows = max(1, CHUNK_LOGITS_BYTES_BUDGET // (V * elem_bytes))
        chunk_size = min(N, max(1, triton.next_power_of_2(budget_rows)))
        # next_power_of_2 may overshoot; if budget_rows is not pow2 the rounded
        # chunk_size could exceed budget_rows × 2 -- cap back to keep budget.
        if chunk_size > budget_rows:
            chunk_size = max(1, chunk_size // 2)
        num_chunks = triton.cdiv(N, chunk_size)

        # ── Output buffers ──────────────────────────────────────────────
        # No zero-init: kernel writes every entry of loss_per_row and
        # correct_per_row across all chunks (s:e ranges cover [0, N)).
        loss_per_row = torch.empty(N, dtype=torch.float32, device=device)
        correct_per_row = torch.empty(N, dtype=torch.float32, device=device)

        # ── Pre-computed gradients (saved for backward; NOT logits/log_p) ──
        # Skip the zero-init: grad_norm_hs is overwritten chunk-by-chunk via
        # out=, and grad_lm_head uses mm (overwrite) on the first chunk and
        # addmm_ (accumulate) thereafter. This saves a 1.2 GB cudaMemset at
        # prod scale (V=152K, H=4096, bf16).
        grad_norm_hs = torch.empty_like(norm_hs)
        grad_lm_head = torch.empty_like(lm_head_w)
        # Reused (chunk_size, V) buffer for the in-place logits/d_logits.
        # Avoids per-chunk alloc-free of a 311 MB (prod) bf16 tensor.
        logits_buf = torch.empty((chunk_size, V), dtype=dtype, device=device)

        inv_N = 1.0 / max(1, N)

        for chunk_id in range(num_chunks):
            s = chunk_id * chunk_size
            e = min(s + chunk_size, N)
            cn = e - s
            if cn <= 0:
                break

            norm_hs_chunk = norm_hs[s:e]                      # (cn, H)
            valid_idx_chunk = valid_idx[s:e]                  # (cn,) int64

            # 1. Matmul in native dtype (bf16/fp16) with tensor-core fp32
            #    accumulation. Output stays in dtype -- no fp32 cast of the
            #    (V, H) lm_head_w (~512 MB at large size, ~2.5 GB at prod).
            logits_chunk = logits_buf[:cn]                    # (cn, V) view
            torch.matmul(norm_hs_chunk, lm_head_w.T, out=logits_chunk)

            # 2. Triton kernel: writes d_logits in-place into logits_chunk;
            #    fills loss_per_row[s:e] and correct_per_row[s:e]. Internal
            #    accumulators are fp32; load/store dtype matches X_ptr.
            _eagle3_kl_loss_kernel[(cn,)](
                logits_chunk,
                target_p_flat,
                valid_idx_chunk,
                loss_per_row[s:e],
                correct_per_row[s:e],
                V,
                inv_N,
                logits_chunk.stride(0),
                target_p_flat.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=KERNEL_NUM_WARPS,
            )

            # 3. Backprop matmul using d_logits (now in logits_chunk):
            #    d_norm_hs[chunk] = d_logits @ lm_head_w   (overwrite slice)
            #    d_lm_head        = d_logits.T @ norm_hs   (chunk 0: overwrite)
            #    d_lm_head       += d_logits.T @ norm_hs   (chunk ≥1: accumulate)
            torch.matmul(logits_chunk, lm_head_w, out=grad_norm_hs[s:e])
            if chunk_id == 0:
                torch.matmul(logits_chunk.T, norm_hs_chunk, out=grad_lm_head)
            else:
                grad_lm_head.addmm_(logits_chunk.T, norm_hs_chunk)

            # logits_chunk is a view into logits_buf -- reused next iter.

        # ── Reductions ──────────────────────────────────────────────────
        loss = loss_per_row.sum() / max(1, N)
        acc = correct_per_row.mean()

        # ── Save ONLY pre-computed grads (Liger pattern) ───────────────
        ctx.save_for_backward(grad_norm_hs, grad_lm_head)
        return loss, acc

    @staticmethod
    def backward(ctx, d_loss: torch.Tensor, d_acc: torch.Tensor):
        """Backward = scale saved grads by d_loss in-place. d_acc is unused
        (argmax is non-differentiable). target_p_flat and valid_idx have no
        grads. The in-place mul avoids allocating (N, H) and (V, H) temporaries
        -- the latter is 1.2 GB at prod."""
        grad_norm_hs, grad_lm_head = ctx.saved_tensors
        grad_norm_hs.mul_(d_loss)
        grad_lm_head.mul_(d_loss)
        return grad_norm_hs, None, grad_lm_head, None


# ─── Public entry point (must match reference.eagle3_loss_ref signature) ─
def kernel_fn(
    prenorm_hs_flat: torch.Tensor,    # (B*T, H)  bf16/fp16
    target_p_flat: torch.Tensor,      # (B*T, V)  bf16/fp16  (no grad)
    valid_idx: torch.Tensor,          # (N,)      int64
    norm_weight: torch.Tensor,        # (H,)      bf16/fp16
    lm_head_weight: torch.Tensor,     # (V, H)    bf16/fp16
    norm_eps: float,
) -> torch.Tensor:
    """
    Forward + backward Eagle3 forward-KL loss with argmax accuracy.

    The gather + RMSNorm stays in eager PyTorch (autograd-natural) -- this
    is the same partition Liger uses (fused_linear_jsd doesn't fuse layernorm).
    The matmul + log_softmax + KL + accuracy stage is wrapped in a custom
    autograd.Function that chunks over rows and saves only pre-computed grads.

    Returns:
        torch.stack([loss, acc]) -- shape (2,) fp32. out[0] is the autograd-
        tracked loss; out[1] is the (no-grad) accuracy.
    """
    assert prenorm_hs_flat.is_cuda and target_p_flat.is_cuda
    assert valid_idx.is_cuda and norm_weight.is_cuda and lm_head_weight.is_cuda
    assert valid_idx.dtype == torch.int64

    # 1. Gather valid hidden states (eager, autograd-natural). target_p is NOT
    #    gathered eagerly -- the Triton kernel reads target_p_flat[valid_idx]
    #    directly, saving a (N, V) intermediate (~622 MB at prod scale).
    hs = prenorm_hs_flat.index_select(0, valid_idx)                 # (N, H)

    # 2. RMSNorm via the fused PyTorch op (autograd-aware, fp32-internal).
    norm_hs = F.rms_norm(hs, (hs.shape[-1],), weight=norm_weight, eps=norm_eps)  # (N, H)

    # 3. Custom autograd-aware fused matmul + KL + accuracy. Pass
    #    target_p_flat + valid_idx so the kernel gathers per-row.
    loss, acc = _Eagle3LossFn.apply(norm_hs, target_p_flat, lm_head_weight, valid_idx)

    return torch.stack([loss.float(), acc.float()])
