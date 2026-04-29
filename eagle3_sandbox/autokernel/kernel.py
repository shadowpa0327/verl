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
MAX_FUSED_SIZE = 16384
KERNEL_NUM_WARPS = 16

# Chunk-memory budget for the in-place (cn, V) bf16 logits buffer.
# Liger's formula chunk_size = next_pow2(N*H/V) keeps chunk memory ≈ N*H, but
# at V=152K that gives chunk_size=64 -- too thin for tensor-core matmul. We
# bound the chunk by an absolute byte budget instead, which yields larger
# chunks at prod (chunk_size≈512 at V=152K) and full N at large (V≤32K).
CHUNK_LOGITS_BYTES_BUDGET = 512 * 1024 * 1024


# ─── Triton kernel: fused gather + RMSNorm forward ────────────────────────
# Reads `prenorm_hs_flat[valid_idx[row], :]`, computes RMSNorm in fp32,
# writes both `norm_hs` and `rstd` in one pass. Avoids the (N, H) fp32
# intermediate that an eager RMSNorm + duplicate rstd path produces.
@triton.jit
def _gather_rmsnorm_fwd_kernel(
    PRENORM_HS_ptr,    # (B*T, H) bf16/fp16
    VALID_IDX_ptr,     # (N,)     int64
    NORM_W_ptr,        # (H,)     bf16/fp16
    HS_ptr,            # (N, H)   bf16/fp16  output (gathered hs, saved for bwd)
    NORM_HS_ptr,       # (N, H)   bf16/fp16  output (RMSNormed)
    RSTD_ptr,          # (N,)     fp32       output
    eps,
    H_const: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H_const

    real_row = tl.load(VALID_IDX_ptr + row)
    hs = tl.load(PRENORM_HS_ptr + real_row * H_const + offs, mask=mask, other=0.0)
    hs_f32 = hs.to(tl.float32)

    # rstd = 1 / sqrt(mean(hs^2) + eps)
    sum_sq = tl.sum(hs_f32 * hs_f32, axis=0)
    mean_sq = sum_sq / H_const
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    norm_w = tl.load(NORM_W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # F.rms_norm computes: (hs * rstd) * norm_w in fp32 then casts.
    norm_hs = hs_f32 * rstd * norm_w

    # Save the gathered hs (so bwd doesn't re-gather), the normed output,
    # and the per-row rstd.
    tl.store(HS_ptr + row * H_const + offs, hs, mask=mask)
    tl.store(NORM_HS_ptr + row * H_const + offs, norm_hs, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


# ─── Triton kernel: in-place scalar scaling for backward grad ────────────
@triton.jit
def _scale_inplace_kernel(X_ptr, scalar_ptr, n_elems, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(scalar_ptr).to(tl.float32)
    tl.store(X_ptr + offs, x * s, mask=mask)


# ─── Triton kernel: RMSNorm backward + scatter to grad_prenorm_hs_flat ────
# One program per gathered row. Computes:
#   - g  = grad_norm_hs * d_loss        (rescaled in-place into GRAD_NORM_HS_ptr)
#   - z  = hs * rstd                    (re-derived; no fp32 intermediate saved)
#   - h  = g * norm_w
#   - A  = mean_j(h_j * z_j)
#   - grad_hs = rstd * (h - z * A)      (RMSNorm bwd formula)
#   - scatter grad_hs into grad_prenorm_hs_flat[valid_idx[row]]
#   - atomic_add per-row contribution (g * z) into grad_norm_weight
# Single kernel replaces IndexSelectBackward + FusedRmsNormBackward + the
# separate d_loss mul on grad_norm_hs.
@triton.jit
def _rms_norm_bwd_scatter_kernel(
    GRAD_NORM_HS_ptr,         # (N, H)   bf16/fp16, written-back scaled by d_loss
    HS_ptr,                   # (N, H)   bf16/fp16
    RSTD_ptr,                 # (N,)     fp32
    NORM_W_ptr,               # (H,)     bf16/fp16
    VALID_IDX_ptr,            # (N,)     int64
    GRAD_PRENORM_HS_ptr,      # (B*T, H) bf16/fp16, must be pre-zeroed
    GRAD_NORM_W_ptr,          # (H,)     fp32, must be pre-zeroed (atomic target)
    D_LOSS_ptr,               # ()       fp32 scalar
    H_const: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H_const

    g_raw = tl.load(GRAD_NORM_HS_ptr + row * H_const + offs, mask=mask).to(tl.float32)
    hs = tl.load(HS_ptr + row * H_const + offs, mask=mask).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)
    norm_w = tl.load(NORM_W_ptr + offs, mask=mask).to(tl.float32)
    d_loss = tl.load(D_LOSS_ptr).to(tl.float32)

    g = g_raw * d_loss
    z = hs * rstd
    h = g * norm_w
    A = tl.sum(h * z, axis=0) / H_const

    grad_hs = rstd * (h - z * A)

    # Scatter grad_hs into grad_prenorm_hs_flat at the valid_idx row.
    real_row = tl.load(VALID_IDX_ptr + row)
    tl.store(GRAD_PRENORM_HS_ptr + real_row * H_const + offs, grad_hs, mask=mask)

    # atomic_add per-row contribution into grad_norm_weight (fp32). Per-slot
    # contention is bounded by N rows but distributes across H slots, and
    # atomic_add on fp32 is hardware-accelerated on Ampere.
    tl.atomic_add(GRAD_NORM_W_ptr + offs, g * z, mask=mask)


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

    # ── Pass 2: loss + in-place d_logits + argmax(target_p) ────────────
    # target_p is in [0, 1]; masked slots loaded as 0.0 cannot beat any valid
    # slot, so argmax over the loaded tile is correct without an explicit -inf
    # fill. This eliminates the separate Pass 1b V-walk over target_p.
    row_loss = 0.0
    arg_t = 0
    val_t = -1.0  # any valid tp ≥ 0 will beat this
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
        # Inline argmax(target_p) tracking, reusing the tp load above.
        block_arg = tl.argmax(tp, axis=0) + i
        block_val = tl.max(tp, axis=0)
        update = block_val > val_t
        arg_t = tl.where(update, block_arg, arg_t)
        val_t = tl.where(update, block_val, val_t)

    correct = (arg_x == arg_t).to(tl.float32)
    tl.store(correct_ptr + row, correct)
    tl.store(loss_ptr + row, -row_loss)


# ─── Custom autograd.Function: full pipeline ──────────────────────────────
# Owns gather + RMSNorm + chunked matmul + Triton loss + grad precompute.
# Backward uses a custom RMSNorm-bwd-with-scatter kernel that writes
# directly into grad_prenorm_hs_flat at valid_idx rows -- eliminating both
# IndexSelectBackward (~250us at large) and FusedRmsNormBackward (~180us
# at large) from the autograd graph.
class _Eagle3FullFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prenorm_hs_flat: torch.Tensor,    # (B*T, H)  bf16/fp16
        target_p_flat: torch.Tensor,      # (B*T, V)  bf16/fp16  (no grad)
        lm_head_w: torch.Tensor,          # (V, H)    bf16/fp16
        valid_idx: torch.Tensor,          # (N,)      int64
        norm_weight: torch.Tensor,        # (H,)      bf16/fp16
        norm_eps: float,
    ):
        device = prenorm_hs_flat.device
        dtype = prenorm_hs_flat.dtype
        BT, H = prenorm_hs_flat.shape
        N = valid_idx.shape[0]
        V, _ = lm_head_w.shape

        # ── 1+2. Fused gather + RMSNorm in one Triton kernel ────────────
        # Reads prenorm_hs at valid_idx, computes rstd, writes hs/norm_hs/rstd
        # in a single pass. Avoids the (N, H) fp32 intermediate that a
        # non-fused gather + RMSNorm path produces, and saves the duplicate
        # rstd reduction the eager F.rms_norm call would force.
        hs = torch.empty((N, H), dtype=dtype, device=device)
        norm_hs = torch.empty((N, H), dtype=dtype, device=device)
        rstd = torch.empty(N, dtype=torch.float32, device=device)
        BLOCK_H_FWD = triton.next_power_of_2(H)
        _gather_rmsnorm_fwd_kernel[(N,)](
            prenorm_hs_flat, valid_idx, norm_weight,
            hs, norm_hs, rstd, norm_eps,
            H_const=H, BLOCK_H=BLOCK_H_FWD,
            num_warps=8,
        )

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

        # ── Save for backward ──────────────────────────────────────────
        ctx.save_for_backward(hs, rstd, norm_weight, valid_idx, grad_norm_hs, grad_lm_head)
        ctx.prenorm_shape = prenorm_hs_flat.shape
        ctx.dtype = dtype
        return loss, acc

    @staticmethod
    def backward(ctx, d_loss: torch.Tensor, d_acc: torch.Tensor):
        hs, rstd, norm_weight, valid_idx, grad_norm_hs, grad_lm_head = ctx.saved_tensors
        BT_H_shape = ctx.prenorm_shape
        dtype = ctx.dtype
        N, H = grad_norm_hs.shape

        # ── Scale grad_lm_head by d_loss in-place via Triton ────────────
        n_elems = grad_lm_head.numel()
        BLOCK = 16384
        grid = (triton.cdiv(n_elems, BLOCK),)
        _scale_inplace_kernel[grid](grad_lm_head, d_loss, n_elems, BLOCK_SIZE=BLOCK, num_warps=8)

        # ── RMSNorm bwd + scatter to grad_prenorm_hs in one custom kernel.
        #    Also folds the d_loss scaling on grad_norm_hs and the
        #    grad_norm_weight reduction (atomic_add over H slots).
        BLOCK_H = triton.next_power_of_2(H)
        grad_prenorm_hs = torch.zeros(BT_H_shape, dtype=dtype, device=grad_norm_hs.device)
        grad_norm_weight_acc = torch.zeros(H, dtype=torch.float32, device=grad_norm_hs.device)

        d_loss_f32 = d_loss.detach().to(torch.float32).contiguous()

        _rms_norm_bwd_scatter_kernel[(N,)](
            grad_norm_hs, hs, rstd, norm_weight, valid_idx,
            grad_prenorm_hs, grad_norm_weight_acc, d_loss_f32,
            H_const=H, BLOCK_H=BLOCK_H,
            num_warps=8,
        )

        grad_norm_weight = grad_norm_weight_acc.to(norm_weight.dtype)

        # Return order matches forward inputs:
        # (prenorm_hs, target_p, lm_head_w, valid_idx, norm_weight, norm_eps)
        return grad_prenorm_hs, None, grad_lm_head, None, grad_norm_weight, None


# ─── Public entry point (must match reference.eagle3_loss_ref signature) ─
def kernel_fn(
    prenorm_hs_flat: torch.Tensor,    # (B*T, H)  bf16/fp16
    target_p_flat: torch.Tensor,      # (B*T, V)  bf16/fp16  (no grad)
    valid_idx: torch.Tensor,          # (N,)      int64
    norm_weight: torch.Tensor,        # (H,)      bf16/fp16
    lm_head_weight: torch.Tensor,     # (V, H)    bf16/fp16
    norm_eps: float,
) -> torch.Tensor:
    """Forward + backward Eagle3 forward-KL loss with argmax accuracy.

    Owns the entire pipeline (gather + RMSNorm + matmul + KL + accuracy +
    grad precompute) in one autograd Function. Backward uses a custom
    Triton kernel for RMSNorm-bwd + scatter + d_loss scaling + grad_norm_w
    atomic_add reduction -- eliminating IndexSelectBackward and
    FusedRmsNormBackward from the autograd graph.

    Returns:
        torch.stack([loss, acc]) -- shape (2,) fp32. out[0] is the autograd-
        tracked loss; out[1] is the (no-grad) accuracy.
    """
    assert prenorm_hs_flat.is_cuda and target_p_flat.is_cuda
    assert valid_idx.is_cuda and norm_weight.is_cuda and lm_head_weight.is_cuda
    assert valid_idx.dtype == torch.int64

    loss, acc = _Eagle3FullFn.apply(
        prenorm_hs_flat, target_p_flat, lm_head_weight,
        valid_idx, norm_weight, norm_eps,
    )

    return torch.stack([loss.float(), acc.float()])
