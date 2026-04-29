#!/usr/bin/env python3
"""
bench.py -- Eagle3 loss-kernel benchmark harness (FIXED -- the agent NEVER modifies this file).

Adapted from autokernel/bench.py for the Eagle3 forward-KL loss pipeline. Runs:
  1. GPU hardware detection + roofline
  2. 5-stage correctness vs reference.eagle3_loss_ref
  3. Performance benchmark (latency, TFLOPS, GB/s)
  4. Per-call peak VRAM measurement (the headline memory metric)
  5. Structured greppable output for an autokernel-style edit/run/keep loop

Usage:
  python bench.py                    # full run
  python bench.py --quick            # skip stages 3-5; bench primary size only
  python bench.py --sizes large      # bench just one size
  python bench.py --profile          # emit torch profiler trace
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Timeout helper (Unix only -- this sandbox is Linux/CUDA only)
# ---------------------------------------------------------------------------
class BenchTimeoutError(Exception):
    pass


class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise BenchTimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


# =========================================================================
# 1. GPU HARDWARE DETECTION (subset of autokernel/bench.py)
# =========================================================================
@dataclass
class GPUSpec:
    name: str = "Unknown"
    sm_count: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)


_KNOWN_GPUS: Dict[str, Tuple[float, float, float]] = {
    "H100 SXM":  (989.5,  3352.0, 50.0),
    "H100 PCIe": (756.0,  2039.0, 50.0),
    "H100":      (756.0,  2039.0, 50.0),
    "A100-SXM":  (312.0,  2039.0, 40.0),
    "A100-PCIE": (312.0,  1935.0, 40.0),
    "A100":      (312.0,  2039.0, 40.0),
    "L40S":      (362.05, 864.0,  48.0),
    "L4":        (121.0,  300.0,  48.0),
    "A10":       (125.0,  600.0,  6.0),
    "4090":      (330.0,  1008.0, 72.0),
    "4080":      (305.0,  716.8,  64.0),
    "3090":      (142.0,  936.2,  6.0),
}


def detect_gpu() -> GPUSpec:
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected, using dummy spec")
        return GPUSpec()

    props = torch.cuda.get_device_properties(0)
    name = props.name
    sm_count = props.multi_processor_count
    memory_gb = round(props.total_memory / (1024 ** 3), 1)
    cc = (props.major, props.minor)

    matched = next((s for frag, s in _KNOWN_GPUS.items() if frag in name), None)
    if matched is not None:
        peak_fp16, peak_bw, l2 = matched
    else:
        peak_fp16 = 200.0
        peak_bw = 1500.0
        l2 = props.L2_cache_size / (1024 * 1024) if hasattr(props, "L2_cache_size") else 0.0

    return GPUSpec(
        name=name,
        sm_count=sm_count,
        memory_gb=memory_gb,
        peak_tflops_fp16=peak_fp16,
        peak_tflops_bf16=peak_fp16,
        peak_tflops_fp32=peak_fp16 / 2.0,
        peak_bandwidth_gb_s=peak_bw,
        l2_cache_mb=l2,
        compute_capability=cc,
    )


# =========================================================================
# 2. INPUT GENERATOR -- Eagle3 forward-KL loss
# =========================================================================
def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def gen_eagle3_loss_inputs(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    """
    Build a synthetic Eagle3 loss-kernel input set.

    Size dict keys:
        N_full        : flattened B*T (number of rows in prenorm_hs / target_p)
        H             : hidden dim
        V             : vocab dim
        rho           : fraction of rows that are valid (random mask, default 0.5)
        mask_pattern  : optional override -- {"first_half", "second_half",
                        "strided"}. When set, `rho` is ignored. Used by the
                        edge-case stages to exercise non-random valid_idx
                        layouts (matches torchspec's _make_mask_patterns).

    All float tensors are in `dtype`. The target probabilities are produced via
    softmax to match the production lifecycle (compute_target_p_padded), so the
    loss has the right magnitude and a meaningful argmax.
    """
    torch.manual_seed(seed)
    N_full = size["N_full"]
    H = size["H"]
    V = size["V"]

    prenorm_hs_flat = torch.randn(N_full, H, device=device, dtype=dtype)

    # Build a normalized target distribution so KL is finite and acc is meaningful.
    target_logits = torch.randn(N_full, V, device=device, dtype=torch.float32) * 2.0
    target_p_flat = torch.softmax(target_logits, dim=-1).to(dtype)
    del target_logits

    # Loss-mask construction. Default is sparse-random; explicit patterns
    # exercise contiguous / strided layouts that catch indexing bugs.
    mask_pattern = size.get("mask_pattern", None)
    if mask_pattern == "first_half":
        valid_idx = torch.arange(max(1, N_full // 2), device=device, dtype=torch.int64)
    elif mask_pattern == "second_half":
        start = N_full // 2
        valid_idx = torch.arange(start, N_full, device=device, dtype=torch.int64)
    elif mask_pattern == "strided":
        valid_idx = torch.arange(0, N_full, 2, device=device, dtype=torch.int64)
    elif mask_pattern is not None:
        raise ValueError(f"unknown mask_pattern {mask_pattern!r}")
    else:
        rho = size.get("rho", 0.5)
        n_valid = max(1, int(N_full * rho))
        perm = torch.randperm(N_full, device=device)
        valid_idx = perm[:n_valid].to(torch.int64).contiguous().sort().values

    norm_weight = torch.ones(H, device=device, dtype=dtype) + 0.05 * torch.randn(
        H, device=device, dtype=dtype
    )
    lm_head_weight = torch.randn(V, H, device=device, dtype=dtype) * (1.0 / (H ** 0.5))
    norm_eps = 1e-6

    return {
        "prenorm_hs_flat": prenorm_hs_flat,
        "target_p_flat": target_p_flat,
        "valid_idx": valid_idx,
        "norm_weight": norm_weight,
        "lm_head_weight": lm_head_weight,
        "norm_eps": norm_eps,
    }


def _ref_eagle3_loss(inputs: dict) -> torch.Tensor:
    import reference
    return reference.eagle3_loss_ref(
        inputs["prenorm_hs_flat"],
        inputs["target_p_flat"],
        inputs["valid_idx"],
        inputs["norm_weight"],
        inputs["lm_head_weight"],
        inputs["norm_eps"],
    )


# ── Compiled reference -- THE production baseline ──────────────────
# Production wraps the same math in @torch.compile(dynamic=None) at
# recipe/drafter_cotraining/eagle3/ops/loss.py:25 (compiled_forward_kl_loss).
# The decorator args and the math are byte-for-byte identical to
# reference.eagle3_loss_ref, so wrapping the reference with the same
# torch.compile call reproduces the production baseline -- no need to
# couple this sandbox to the recipe layout.
_COMPILED_REF_FN = None


def _get_compiled_ref() -> Callable:
    """Lazy-build a torch.compile'd reference, cached for the run.

    dynamic=None matches production -- lets TorchInductor auto-detect dynamic
    shapes across the size sweep instead of recompiling per shape.
    """
    global _COMPILED_REF_FN
    if _COMPILED_REF_FN is None:
        import reference
        _COMPILED_REF_FN = torch.compile(reference.eagle3_loss_ref, dynamic=None)
    return _COMPILED_REF_FN


def _ref_compiled_eagle3_loss(inputs: dict) -> torch.Tensor:
    fn = _get_compiled_ref()
    return fn(
        inputs["prenorm_hs_flat"],
        inputs["target_p_flat"],
        inputs["valid_idx"],
        inputs["norm_weight"],
        inputs["lm_head_weight"],
        inputs["norm_eps"],
    )


# =========================================================================
# 3. KERNEL CONFIG
# =========================================================================
# FLOPs (dominant terms; index_select is free, RMSNorm and argmax are tiny):
#   lm_head:    2 * N * H * V
#   log_softmax: ~5 * N * V    (max, sub, exp, sum, log)
#   KL sum:      2 * N * V     (mul, sum)
#   ≈ 2 * N * H * V + 8 * N * V       (matmul-dominated for H >= a few)
def _flops_eagle3(s: dict) -> float:
    N = max(1, int(s["N_full"] * s.get("rho", 0.5)))
    H = s["H"]
    V = s["V"]
    return 2.0 * N * H * V + 8.0 * N * V + 6.0 * N * H


# Bytes touched (assume the kernel must at least read each input once and
# stream the (N, V) intermediates twice -- once produced, once consumed).
# For a perfectly fused kernel the (N, V) traffic disappears; we still credit
# the kernel based on the *workload* not the implementation.
def _bytes_eagle3(s: dict, dt: torch.dtype) -> float:
    N = max(1, int(s["N_full"] * s.get("rho", 0.5)))
    H = s["H"]
    V = s["V"]
    eb = _dtype_bytes(dt)
    # Reads: prenorm_hs (full B*T*H, gather is from this), target_p (full B*T*V),
    #        norm_weight (H), lm_head (V*H). Writes: 2 fp32 scalars (negligible).
    return float((s["N_full"] * H + s["N_full"] * V + H + V * H) * eb)


KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "eagle3_loss": {
        # NOTE: vocab is the dominant memory cost. Sizes scale up to Qwen3-8B (V=151936).
        "test_sizes": [
            ("tiny",   {"N_full": 64,    "H": 128,  "V": 256,    "rho": 0.5}),
            ("small",  {"N_full": 512,   "H": 512,  "V": 2048,   "rho": 0.5}),
            ("medium", {"N_full": 2048,  "H": 1024, "V": 8192,   "rho": 0.5}),
            ("large",  {"N_full": 4096,  "H": 4096, "V": 32000,  "rho": 0.5}),
            ("prod",   {"N_full": 4096,  "H": 4096, "V": 151936, "rho": 0.5}),  # Qwen3-8B B=1,T=4096
            # B=2, T=4096 production case: micro_batch_size_per_gpu=2 with
            # max_seq_length=4096 -> B*T = 8192 (= 2x the `prod` row).
            ("prod_b2", {"N_full": 8192, "H": 4096, "V": 151936, "rho": 0.5}),
        ],
        "test_dtypes": [torch.bfloat16, torch.float16],
        # KL sums over V; with V≈151936, even ~1e-3 per-element error compounds.
        # Allow ~1% relative + 5e-3 absolute -- matches the bf16 noise floor in the
        # bench_lazy_vs_precomputed memory study.
        "tolerances": {
            torch.float16:  {"atol": 5e-3, "rtol": 1e-2},
            torch.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
        },
        "flops_fn": _flops_eagle3,
        "bytes_fn": _bytes_eagle3,
        "input_generator": gen_eagle3_loss_inputs,
        "reference_fn": _ref_eagle3_loss,
        "edge_sizes": [
            # Non-power-of-2 N and V; sparsity extremes.
            ("edge_777",       {"N_full": 777,  "H": 513,  "V": 2049, "rho": 0.5}),
            ("edge_sparse",    {"N_full": 1024, "H": 1024, "V": 4096, "rho": 0.05}),
            ("edge_dense",     {"N_full": 1024, "H": 1024, "V": 4096, "rho": 1.0}),
            ("edge_one_valid", {"N_full": 1024, "H": 1024, "V": 4096, "rho": 0.001}),
            # Deterministic mask layouts (mirrors torchspec _make_mask_patterns).
            # `strided` is the trickiest -- exposes any kernel that assumes
            # contiguous valid rows. `first_half` / `second_half` exercise
            # contiguous offsets at both ends.
            ("edge_first_half",  {"N_full": 1024, "H": 1024, "V": 4096, "mask_pattern": "first_half"}),
            ("edge_second_half", {"N_full": 1024, "H": 1024, "V": 4096, "mask_pattern": "second_half"}),
            ("edge_strided",     {"N_full": 1024, "H": 1024, "V": 4096, "mask_pattern": "strided"}),
        ],
    },
}


# =========================================================================
# 4. CORRECTNESS (5 stages -- adapted from autokernel/bench.py)
# =========================================================================
def _compare(output: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    if output.shape != expected.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {output.shape} vs {expected.shape}",
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "pct_within_tol": 0.0,
        }
    out_f = output.float()
    exp_f = expected.float()
    abs_diff = (out_f - exp_f).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    within = (abs_diff <= atol + rtol * exp_f.abs()).float().mean().item() * 100.0
    match = torch.allclose(out_f, exp_f, atol=atol, rtol=rtol)
    return {
        "match": match,
        "reason": "" if match else f"max_abs_error={max_abs:.6e} exceeds tol(atol={atol}, rtol={rtol})",
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "pct_within_tol": within,
    }


def _has_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


# Trainable input keys (those the harness will require_grad and check grads on).
TRAINABLE_KEYS = ("prenorm_hs_flat", "norm_weight", "lm_head_weight")

# Production TTT (Test-Time Training) loop length. Eagle3Model.forward at
# eagle3_sandbox/eagle3_model.py:237 calls compiled_forward_kl_loss this many
# times before running ONE backward over the weighted sum of per-step losses.
# Saved-for-backward tensors from all TTT_STEPS calls coexist in VRAM until
# the single .backward() -- that is the production training-step memory profile.
TTT_STEPS = 7
TTT_WEIGHTS = tuple(0.8**i for i in range(TTT_STEPS))


def _make_grad_inputs(inputs: dict) -> dict:
    """Clone the input dict, marking the trainable tensors as leaves with grad.

    The non-trainable tensors are detached clones (so the autograd graph cannot
    leak through them between iterations). Returns a fresh dict that callers can
    reuse across timing iterations -- ``_zero_grads`` clears stale grads.
    """
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            cloned = v.detach().clone()
            if k in TRAINABLE_KEYS:
                cloned.requires_grad_(True)
            out[k] = cloned
        else:
            out[k] = v
    return out


def _zero_grads(grad_inputs: dict) -> None:
    for k in TRAINABLE_KEYS:
        t = grad_inputs.get(k)
        if isinstance(t, torch.Tensor):
            t.grad = None


def _step_loss_backward(call_fn, grad_inputs: dict) -> None:
    """Run forward then backward on out[0] (loss). Used for fwd+bwd timing."""
    _zero_grads(grad_inputs)
    out = call_fn(**grad_inputs) if callable(call_fn) and call_fn.__code__.co_argcount > 1 \
        else call_fn(grad_inputs)
    # call_fn may be a kernel_fn (takes **kwargs) or a wrapped ref_fn (takes dict).
    out[0].backward()


def _kernel_step_factory(kernel_fn: Callable, grad_inputs: dict):
    """Closure: zero grads, run kernel forward, run backward on loss.

    Returns a 0-arg callable suitable for triton.testing.do_bench.
    """
    def step():
        _zero_grads(grad_inputs)
        out = kernel_fn(**grad_inputs)
        out[0].backward()
    return step


def _ref_step_factory(ref_fn: Callable, grad_inputs: dict):
    """Same but for the reference -- ref_fn takes the dict (not **kwargs)."""
    def step():
        _zero_grads(grad_inputs)
        out = ref_fn(grad_inputs)
        out[0].backward()
    return step


def _kernel_ttt_step_factory(
    kernel_fn: Callable, grad_inputs: dict,
    n_steps: int = TTT_STEPS, weights: tuple = TTT_WEIGHTS,
):
    """Production-faithful TTT-N training step closure for the kernel.

    Mirrors Eagle3Model.forward (eagle3_model.py:237-279):
        for idx in range(self.length):
            ...
            loss, acc = self._calculate_loss(...)   # → kernel_fn
            plosses.append(loss); acces.append(acc)
        ploss = sum(0.8**i * plosses[i] for i in range(7))
        ploss.backward()

    All n_steps forward calls' saved-for-backward tensors are held in the
    autograd graph until the single .backward() -- exactly the memory pattern
    production sees during training. Inputs are reused across the n_steps
    calls (production's per-step input variation -- mask shifts, hidden
    states advance through the backbone -- changes values, not memory shape;
    the peak VRAM profile is identical).
    """
    def step():
        _zero_grads(grad_inputs)
        plosses = []
        for _ in range(n_steps):
            out = kernel_fn(**grad_inputs)
            plosses.append(out[0])  # works for tuple OR (2,)-stacked tensor
        ploss = sum(weights[i] * plosses[i] for i in range(n_steps))
        ploss.backward()
    return step


def _ref_ttt_step_factory(
    ref_fn: Callable, grad_inputs: dict,
    n_steps: int = TTT_STEPS, weights: tuple = TTT_WEIGHTS,
):
    """TTT-N closure for a reference function (eager or compiled). Same loop
    structure as _kernel_ttt_step_factory but ref_fn takes a dict, not **kwargs.
    """
    def step():
        _zero_grads(grad_inputs)
        plosses = []
        for _ in range(n_steps):
            out = ref_fn(grad_inputs)
            plosses.append(out[0])
        ploss = sum(weights[i] * plosses[i] for i in range(n_steps))
        ploss.backward()
    return step


def _capture_grads(grad_inputs: dict) -> dict:
    """Snapshot current .grad tensors for the trainable keys (cloned)."""
    return {
        k: grad_inputs[k].grad.detach().clone() if grad_inputs[k].grad is not None else None
        for k in TRAINABLE_KEYS
    }


def _compare_grad_dict(out_grads: dict, ref_grads: dict, atol: float, rtol: float) -> dict:
    """Per-key tensor compare. Returns {match, reason, per_key: {key: stats}}."""
    per_key = {}
    all_match = True
    worst_reason = ""
    for k in TRAINABLE_KEYS:
        og = out_grads.get(k)
        rg = ref_grads.get(k)
        if og is None and rg is None:
            per_key[k] = {"match": True, "max_abs_error": 0.0, "mean_abs_error": 0.0}
            continue
        if og is None or rg is None:
            all_match = False
            worst_reason = worst_reason or f"{k}: one grad is None (kernel={og is not None}, ref={rg is not None})"
            per_key[k] = {"match": False, "reason": worst_reason}
            continue
        cmp = _compare(og, rg, atol=atol, rtol=rtol)
        per_key[k] = cmp
        if not cmp["match"]:
            all_match = False
            worst_reason = worst_reason or f"{k}: {cmp['reason']}"
    return {"match": all_match, "reason": worst_reason, "per_key": per_key}


def _safe_target_transform(transform_fn, t: torch.Tensor, key: str) -> torch.Tensor:
    """Apply adversarial transform but re-normalize target_p so it stays a distribution."""
    out = transform_fn(t)
    if key == "target_p_flat":
        # Re-softmax to keep the distribution well-formed; otherwise KL is ill-defined.
        out = torch.softmax(out.float() + 1e-30, dim=-1).to(t.dtype)
    return out


def run_correctness(kernel_fn: Callable, config: dict, quick: bool = False) -> dict:
    device = "cuda"
    results = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "numerical_stability": "SKIP",
        "determinism": "SKIP",
        "edge_cases": "SKIP",
        "backward_grad": "SKIP",
        "subset_invariance": "SKIP",
        "correctness": "FAIL",
    }
    details: List[str] = []
    all_pass = True

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    tols = config["tolerances"]

    # ── Stage 1: smoke ────────────────────────────────────────────────
    print("\n--- Stage 1: Smoke Test ---")
    try:
        tiny_label, tiny_size = sizes[0]
        dtype0 = dtypes[0]
        inputs = gen_fn(tiny_size, dtype0, device, seed=42)
        expected = ref_fn(inputs)
        with _Timeout(30):
            output = kernel_fn(**inputs)
        if _has_nan_inf(output):
            results["smoke_test"] = "FAIL"
            details.append("  smoke: NaN/Inf in output")
            all_pass = False
            print("  FAIL: NaN/Inf in output")
        else:
            tol = tols.get(dtype0, {"atol": 5e-3, "rtol": 1e-2})
            cmp = _compare(output, expected, **tol)
            if cmp["match"]:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={cmp['max_abs_error']:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                details.append(f"  smoke: {cmp['reason']}")
                all_pass = False
                print(f"  FAIL: {cmp['reason']}")
    except BenchTimeoutError:
        results["smoke_test"] = "FAIL"; all_pass = False
        details.append("  smoke: TIMEOUT"); print("  FAIL: TIMEOUT")
    except torch.cuda.OutOfMemoryError:
        results["smoke_test"] = "FAIL"; all_pass = False
        details.append("  smoke: OOM"); print("  FAIL: OOM")
    except Exception as e:
        results["smoke_test"] = "FAIL"; all_pass = False
        details.append(f"  smoke: CRASH ({type(e).__name__}: {e})")
        print(f"  FAIL: CRASH ({type(e).__name__}: {e})")

    if results["smoke_test"] == "FAIL":
        results["correctness"] = "FAIL"
        results["details"] = details
        print("\ncorrectness: FAIL (smoke test failed, aborting remaining stages)")
        return results

    # ── Stage 2: shape sweep ──────────────────────────────────────────
    print("\n--- Stage 2: Shape Sweep ---")
    sweep_pass = True
    sweep_count = 0
    sweep_fail_count = 0
    worst_error = 0.0
    worst_case = ""
    for label, sz in sizes:
        for dtype in dtypes:
            sweep_count += 1
            try:
                inputs = gen_fn(sz, dtype, device, seed=42)
                expected = ref_fn(inputs)
                with _Timeout(60):
                    output = kernel_fn(**inputs)
                if _has_nan_inf(output):
                    sweep_pass = False; sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: NaN/Inf")
                    print(f"  FAIL: {label} {dtype} -> NaN/Inf")
                    continue
                tol = tols.get(dtype, {"atol": 5e-3, "rtol": 1e-2})
                cmp = _compare(output, expected, **tol)
                if cmp["max_abs_error"] > worst_error:
                    worst_error = cmp["max_abs_error"]; worst_case = f"{label}/{dtype}"
                if not cmp["match"]:
                    sweep_pass = False; sweep_fail_count += 1
                    details.append(f"  sweep {label}/{dtype}: {cmp['reason']}")
                    print(f"  FAIL: {label} {dtype} -> {cmp['reason']}")
                else:
                    print(f"  PASS: {label} {dtype} (max_err={cmp['max_abs_error']:.2e}, "
                          f"within_tol={cmp['pct_within_tol']:.1f}%)")
            except torch.cuda.OutOfMemoryError:
                print(f"  SKIP: {label} {dtype} -> OOM"); torch.cuda.empty_cache()
            except BenchTimeoutError:
                sweep_pass = False; sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: TIMEOUT")
                print(f"  FAIL: {label} {dtype} -> TIMEOUT")
            except Exception as e:
                sweep_pass = False; sweep_fail_count += 1
                details.append(f"  sweep {label}/{dtype}: {type(e).__name__}: {e}")
                print(f"  FAIL: {label} {dtype} -> {type(e).__name__}: {e}")
            finally:
                torch.cuda.empty_cache()
    if sweep_pass:
        results["shape_sweep"] = (
            f"PASS ({sweep_count} configs, worst_err={worst_error:.2e} at {worst_case})"
        )
        print(f"  shape_sweep: PASS ({sweep_count} configs, worst_err={worst_error:.2e})")
    else:
        results["shape_sweep"] = f"FAIL ({sweep_fail_count}/{sweep_count} failed)"
        all_pass = False
        print(f"  shape_sweep: FAIL ({sweep_fail_count}/{sweep_count} failed)")

    if quick:
        results["numerical_stability"] = "SKIP (quick mode)"
        results["determinism"] = "SKIP (quick mode)"
        results["edge_cases"] = "SKIP (quick mode)"
        results["backward_grad"] = "SKIP (quick mode)"
        results["subset_invariance"] = "SKIP (quick mode)"
        results["correctness"] = "PASS" if all_pass else "FAIL"
        results["details"] = details
        print(f"\ncorrectness: {results['correctness']} (quick mode: stages 3-7 skipped)")
        return results

    # ── Stage 3: numerical stability ──────────────────────────────────
    print("\n--- Stage 3: Numerical Stability ---")
    stability_pass = True
    stab_size = next((sz for lbl, sz in sizes if lbl == "small"), sizes[min(1, len(sizes) - 1)][1])
    stab_dtype = dtypes[0]

    cases = [
        ("near_zero",   lambda t: t * 1e-3),
        ("large_scale", lambda t: t * 50.0),
        ("all_zeros",   lambda t: torch.zeros_like(t)),
        ("all_same",    lambda t: torch.ones_like(t) * 0.5),
    ]
    for case_name, transform in cases:
        try:
            inputs = gen_fn(stab_size, stab_dtype, device, seed=42)
            transformed = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    transformed[k] = _safe_target_transform(transform, v, k)
                else:
                    transformed[k] = v
            expected = ref_fn(transformed)
            with _Timeout(30):
                output = kernel_fn(**transformed)
            if _has_nan_inf(output) and not _has_nan_inf(expected):
                stability_pass = False
                details.append(f"  stability {case_name}: NaN/Inf (reference is clean)")
                print(f"  FAIL: {case_name} -> NaN/Inf (reference is clean)")
            elif _has_nan_inf(output) and _has_nan_inf(expected):
                print(f"  PASS: {case_name} -> both have NaN/Inf (expected)")
            else:
                tol = tols.get(stab_dtype, {"atol": 5e-3, "rtol": 1e-2})
                cmp = _compare(output, expected, atol=tol["atol"] * 10, rtol=tol["rtol"] * 10)
                if cmp["match"]:
                    print(f"  PASS: {case_name} (max_err={cmp['max_abs_error']:.2e})")
                else:
                    stability_pass = False
                    details.append(f"  stability {case_name}: {cmp['reason']}")
                    print(f"  FAIL: {case_name} -> {cmp['reason']}")
        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: {case_name} -> OOM"); torch.cuda.empty_cache()
        except BenchTimeoutError:
            stability_pass = False
            details.append(f"  stability {case_name}: TIMEOUT")
            print(f"  FAIL: {case_name} -> TIMEOUT")
        except Exception as e:
            stability_pass = False
            details.append(f"  stability {case_name}: {type(e).__name__}: {e}")
            print(f"  FAIL: {case_name} -> {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()
    results["numerical_stability"] = "PASS" if stability_pass else "FAIL"
    if not stability_pass:
        all_pass = False
    print(f"  numerical_stability: {results['numerical_stability']}")

    # ── Stage 4: determinism ──────────────────────────────────────────
    print("\n--- Stage 4: Determinism ---")
    determinism_pass = True
    try:
        det_size = stab_size
        det_dtype = dtypes[0]
        outputs = []
        for _ in range(3):
            inputs_i = gen_fn(det_size, det_dtype, device, seed=42)
            with _Timeout(30):
                out_i = kernel_fn(**inputs_i)
            outputs.append(out_i)
        for i in range(1, 3):
            if not torch.equal(outputs[0], outputs[i]):
                determinism_pass = False
                diff = (outputs[0].float() - outputs[i].float()).abs()
                details.append(f"  determinism: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")
                print(f"  FAIL: run 0 vs run {i} differ (max_diff={diff.max().item():.6e})")
        if determinism_pass:
            print("  PASS: 3 runs are bitwise identical")
        results["determinism"] = "PASS" if determinism_pass else "FAIL"
    except Exception as e:
        results["determinism"] = f"FAIL ({type(e).__name__})"
        all_pass = False
        details.append(f"  determinism: {type(e).__name__}: {e}")
        print(f"  FAIL: {type(e).__name__}: {e}")
    finally:
        torch.cuda.empty_cache()
    if not determinism_pass:
        all_pass = False

    # ── Stage 5: edge cases ───────────────────────────────────────────
    print("\n--- Stage 5: Edge Cases ---")
    edge_pass = True
    edge_sizes = config.get("edge_sizes", [])
    if not edge_sizes:
        results["edge_cases"] = "SKIP (no edge sizes defined)"
        print("  SKIP: no edge sizes defined")
    else:
        for label, sz in edge_sizes:
            for dtype in dtypes[:1]:
                try:
                    inputs = gen_fn(sz, dtype, device, seed=42)
                    expected = ref_fn(inputs)
                    with _Timeout(30):
                        output = kernel_fn(**inputs)
                    if _has_nan_inf(output) and not _has_nan_inf(expected):
                        edge_pass = False
                        details.append(f"  edge {label}: NaN/Inf"); print(f"  FAIL: {label} -> NaN/Inf")
                    else:
                        tol = tols.get(dtype, {"atol": 5e-3, "rtol": 1e-2})
                        cmp = _compare(output, expected, **tol)
                        if cmp["match"]:
                            print(f"  PASS: {label} (max_err={cmp['max_abs_error']:.2e})")
                        else:
                            edge_pass = False
                            details.append(f"  edge {label}: {cmp['reason']}")
                            print(f"  FAIL: {label} -> {cmp['reason']}")
                except torch.cuda.OutOfMemoryError:
                    print(f"  SKIP: {label} -> OOM"); torch.cuda.empty_cache()
                except BenchTimeoutError:
                    edge_pass = False
                    details.append(f"  edge {label}: TIMEOUT"); print(f"  FAIL: {label} -> TIMEOUT")
                except Exception as e:
                    edge_pass = False
                    details.append(f"  edge {label}: {type(e).__name__}: {e}")
                    print(f"  FAIL: {label} -> {type(e).__name__}: {e}")
                finally:
                    torch.cuda.empty_cache()
        results["edge_cases"] = "PASS" if edge_pass else "FAIL"
        if not edge_pass:
            all_pass = False
        print(f"  edge_cases: {results['edge_cases']}")

    # ── Stage 6: backward gradient check ──────────────────────────────
    # Compare grads of {prenorm_hs_flat, norm_weight, lm_head_weight}
    # between kernel and reference. Backward grads accumulate error from
    # the entire fwd+bwd graph -- looser tolerance than forward.
    print("\n--- Stage 6: Backward Gradient Check ---")
    bwd_pass = True
    bwd_size = next((sz for lbl, sz in sizes if lbl == "small"), sizes[min(1, len(sizes) - 1)][1])
    for dtype in dtypes:
        try:
            base_inputs = gen_fn(bwd_size, dtype, device, seed=42)

            # Reference grads
            g_ref = _make_grad_inputs(base_inputs)
            with _Timeout(60):
                ref_out = ref_fn(g_ref)
            ref_loss_val = ref_out[0].item()
            ref_out[0].backward()
            ref_grads = _capture_grads(g_ref)

            # Kernel grads (fresh leaves to keep graphs disjoint)
            g_ker = _make_grad_inputs(base_inputs)
            with _Timeout(60):
                ker_out = kernel_fn(**g_ker)
            ker_loss_val = ker_out[0].item()
            ker_out[0].backward()
            ker_grads = _capture_grads(g_ker)

            # NaN/Inf check
            any_nan = any(g is not None and _has_nan_inf(g) for g in ker_grads.values())
            if any_nan:
                bwd_pass = False
                details.append(f"  backward {dtype}: NaN/Inf in kernel grads")
                print(f"  FAIL: {dtype} -> NaN/Inf in kernel grads")
                continue

            # Tolerance: 2x forward atol + rtol (grads compound rounding).
            tol = tols.get(dtype, {"atol": 5e-3, "rtol": 1e-2})
            grad_tol = {"atol": tol["atol"] * 2, "rtol": tol["rtol"] * 2}
            cmp = _compare_grad_dict(ker_grads, ref_grads, **grad_tol)
            if cmp["match"]:
                stats = []
                for k in TRAINABLE_KEYS:
                    pk = cmp["per_key"].get(k, {})
                    err = pk.get("max_abs_error", 0.0)
                    stats.append(f"{k.split('_')[0]}={err:.2e}")
                print(f"  PASS: {dtype} loss(ref={ref_loss_val:.4f}, ker={ker_loss_val:.4f}) "
                      f"grad_max_err: {', '.join(stats)}")
            else:
                bwd_pass = False
                details.append(f"  backward {dtype}: {cmp['reason']}")
                # Per-key breakdown for debugging
                for k in TRAINABLE_KEYS:
                    pk = cmp["per_key"].get(k, {})
                    if not pk.get("match", False):
                        print(f"  FAIL ({dtype}/{k}): "
                              f"max_abs={pk.get('max_abs_error', float('inf')):.4e} "
                              f"mean_abs={pk.get('mean_abs_error', float('inf')):.4e}")
        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: backward {dtype} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            bwd_pass = False
            details.append(f"  backward {dtype}: TIMEOUT")
            print(f"  FAIL: backward {dtype} -> TIMEOUT")
        except Exception as e:
            bwd_pass = False
            details.append(f"  backward {dtype}: {type(e).__name__}: {e}")
            print(f"  FAIL: backward {dtype} -> {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    results["backward_grad"] = "PASS" if bwd_pass else "FAIL"
    if not bwd_pass:
        all_pass = False
    print(f"  backward_grad: {results['backward_grad']}")

    # ── Stage 7: subset-invariance ────────────────────────────────────
    # The kernel's output must depend ONLY on the rows pointed to by
    # valid_idx. So:
    #     kernel(hs_full, tp_full, valid_idx)  ==
    #     kernel(hs_full[valid_idx], tp_full[valid_idx], arange(N_valid))
    # Any kernel that has an indexing bug in its gather will fail this even
    # if Stages 1-2 (against the eager reference) pass. Mirrors torchspec's
    # TestValidIdxSubsetting -- the strided pattern is the load-bearing one.
    print("\n--- Stage 7: Subset-Invariance ---")
    subset_pass = True
    sub_size = next((sz for lbl, sz in sizes if lbl == "small"), sizes[min(1, len(sizes) - 1)][1])
    sub_dtype = dtypes[0]

    def _strided(BT):
        return torch.arange(0, BT, 2, device=device, dtype=torch.int64)

    def _first_half(BT):
        return torch.arange(max(1, BT // 2), device=device, dtype=torch.int64)

    def _second_half(BT):
        return torch.arange(BT // 2, BT, device=device, dtype=torch.int64)

    def _random_sparse(BT):
        g = torch.Generator(device=device).manual_seed(99)
        n = max(1, BT // 4)
        return torch.randperm(BT, generator=g, device=device)[:n].sort().values.to(torch.int64)

    def _single(BT):
        return torch.tensor([BT // 3], device=device, dtype=torch.int64)

    def _all(BT):
        return torch.arange(BT, device=device, dtype=torch.int64)

    mask_patterns = [
        ("first_half", _first_half),
        ("second_half", _second_half),
        ("strided", _strided),
        ("random_sparse", _random_sparse),
        ("single", _single),
        ("all", _all),
    ]

    # Subset-invariance compares the kernel against itself, so the only
    # source of disagreement is float-order-of-operations on the valid
    # rows -- bf16 noise floor is the right tolerance.
    sub_tol = {"atol": 5e-3, "rtol": 1e-2}

    for pat_name, pat_fn in mask_patterns:
        try:
            base_inputs = gen_fn(sub_size, sub_dtype, device, seed=42)
            BT = base_inputs["prenorm_hs_flat"].shape[0]
            valid_idx = pat_fn(BT)
            n_valid = valid_idx.numel()
            if n_valid == 0:
                print(f"  SKIP: {pat_name} (empty mask)")
                continue

            # Path A: pass full inputs + the pattern's valid_idx.
            inputs_a = dict(base_inputs)
            inputs_a["valid_idx"] = valid_idx
            with _Timeout(30):
                out_a = kernel_fn(**inputs_a)

            # Path B: pre-filter inputs with the pattern, pass arange(n_valid).
            inputs_b = dict(base_inputs)
            inputs_b["prenorm_hs_flat"] = base_inputs["prenorm_hs_flat"][valid_idx].contiguous()
            inputs_b["target_p_flat"] = base_inputs["target_p_flat"][valid_idx].contiguous()
            inputs_b["valid_idx"] = torch.arange(n_valid, device=device, dtype=torch.int64)
            with _Timeout(30):
                out_b = kernel_fn(**inputs_b)

            cmp = _compare(out_a, out_b, **sub_tol)
            if cmp["match"]:
                print(f"  PASS: {pat_name} (n_valid={n_valid}, "
                      f"max_err={cmp['max_abs_error']:.2e})")
            else:
                subset_pass = False
                details.append(f"  subset-invariance {pat_name}: {cmp['reason']}")
                print(f"  FAIL: {pat_name} (n_valid={n_valid}) -> {cmp['reason']}")
        except torch.cuda.OutOfMemoryError:
            print(f"  SKIP: {pat_name} -> OOM")
            torch.cuda.empty_cache()
        except BenchTimeoutError:
            subset_pass = False
            details.append(f"  subset-invariance {pat_name}: TIMEOUT")
            print(f"  FAIL: {pat_name} -> TIMEOUT")
        except Exception as e:
            subset_pass = False
            details.append(f"  subset-invariance {pat_name}: {type(e).__name__}: {e}")
            print(f"  FAIL: {pat_name} -> {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()

    results["subset_invariance"] = "PASS" if subset_pass else "FAIL"
    if not subset_pass:
        all_pass = False
    print(f"  subset_invariance: {results['subset_invariance']}")

    results["correctness"] = "PASS" if all_pass else "FAIL"
    results["details"] = details
    print(f"\ncorrectness: {results['correctness']}")
    return results


# =========================================================================
# 5. PERFORMANCE + MEMORY
# =========================================================================
def _do_bench(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    try:
        from triton.testing import do_bench
        return do_bench(fn, warmup=warmup, rep=rep)
    except ImportError:
        pass
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _measure_peak_vram(fn: Callable, repeat: int = 5) -> Tuple[float, float]:
    """Return (peak_total_mb, peak_transient_mb).

    peak_total: torch.cuda.max_memory_allocated across the call (includes inputs)
    peak_transient: peak_total minus the resident allocation BEFORE the call,
                    i.e. the kernel-induced delta (closer to "what the kernel costs")
    """
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024 / 1024
    peak_total = 0.0
    for _ in range(repeat):
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        torch.cuda.synchronize()
        peak_total = max(peak_total, torch.cuda.max_memory_allocated() / 1024 / 1024)
        del out
    return peak_total, max(0.0, peak_total - baseline)


def run_performance(kernel_fn: Callable, config: dict, gpu: GPUSpec, sizes_filter: str = "all") -> dict:
    device = "cuda"
    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    bytes_fn = config["bytes_fn"]
    dtypes = config["test_dtypes"]
    sizes = config["test_sizes"]

    if sizes_filter == "all":
        bench_sizes = sizes
    else:
        bench_sizes = [(lbl, sz) for lbl, sz in sizes if lbl == sizes_filter]
        if not bench_sizes:
            bench_sizes = [next(((lbl, sz) for lbl, sz in sizes if lbl == "large"), sizes[-1])]

    primary = next(((lbl, sz) for lbl, sz in sizes if lbl == "large"), sizes[-1])
    primary_label, primary_size = primary
    dtype = dtypes[0]

    all_results = []
    primary_result = None

    for label, sz in bench_sizes:
        print(f"\n  Benchmarking: {label} ...")
        try:
            flops = flops_fn(sz)
            nbytes = bytes_fn(sz, dtype)
            inputs = gen_fn(sz, dtype, device, seed=42)

            # ── Forward-only timing ──────────────────────────────────
            with _Timeout(60):
                kernel_fwd_ms = _do_bench(lambda: kernel_fn(**inputs), warmup=10, rep=50)
            with _Timeout(60):
                ref_fwd_ms = _do_bench(lambda: ref_fn(inputs), warmup=10, rep=50)

            # ── Forward+Backward (training step) timing -- primary ───
            grad_inputs = _make_grad_inputs(inputs)
            kernel_step = _kernel_step_factory(kernel_fn, grad_inputs)
            ref_step = _ref_step_factory(ref_fn, grad_inputs)
            with _Timeout(120):
                kernel_step_ms = _do_bench(kernel_step, warmup=5, rep=30)
            with _Timeout(120):
                ref_step_ms = _do_bench(ref_step, warmup=5, rep=30)

            # ── Memory measurement (forward only) ────────────────────
            peak_kernel_fwd_mb, transient_kernel_fwd_mb = _measure_peak_vram(
                lambda: kernel_fn(**inputs), repeat=3
            )
            peak_ref_fwd_mb, transient_ref_fwd_mb = _measure_peak_vram(
                lambda: ref_fn(inputs), repeat=3
            )

            # ── Memory measurement (forward+backward) -- primary ─────
            peak_kernel_step_mb, transient_kernel_step_mb = _measure_peak_vram(
                kernel_step, repeat=3
            )
            peak_ref_step_mb, transient_ref_step_mb = _measure_peak_vram(ref_step, repeat=3)

            # ── Compiled (torch.compile) reference -- production baseline ───
            # First call triggers TorchInductor compile (slow); subsequent
            # calls are fast. _do_bench's wall-clock warmup absorbs the
            # one-shot compile cost and median-filters it out.
            print(f"    (compiling reference for {label}...)")
            _ref_compiled_eagle3_loss(inputs)  # force compile
            torch.cuda.synchronize()

            with _Timeout(180):
                compiled_fwd_ms = _do_bench(
                    lambda: _ref_compiled_eagle3_loss(inputs), warmup=10, rep=50
                )

            grad_inputs_c = _make_grad_inputs(inputs)
            compiled_step = _ref_step_factory(_ref_compiled_eagle3_loss, grad_inputs_c)
            compiled_step()  # force compile of the bwd graph too
            torch.cuda.synchronize()
            with _Timeout(240):
                compiled_step_ms = _do_bench(compiled_step, warmup=5, rep=30)

            peak_compiled_fwd_mb, transient_compiled_fwd_mb = _measure_peak_vram(
                lambda: _ref_compiled_eagle3_loss(inputs), repeat=3
            )
            peak_compiled_step_mb, transient_compiled_step_mb = _measure_peak_vram(
                compiled_step, repeat=3
            )

            # ── TTT-7 memory (PRODUCTION-FAITHFUL) ─────────────────────
            # Production calls the loss kernel TTT_STEPS=7 times and runs ONE
            # backward over the weighted sum of per-step losses. Memory peak
            # is dominated by 7× saved-for-backward tensors coexisting until
            # the final .backward() -- this is the metric that maps to
            # production training-step VRAM pressure, not the 1-call number.
            kernel_ttt7 = _kernel_ttt_step_factory(kernel_fn, grad_inputs)
            ref_ttt7 = _ref_ttt_step_factory(ref_fn, grad_inputs)
            compiled_ttt7 = _ref_ttt_step_factory(_ref_compiled_eagle3_loss, grad_inputs_c)

            # Force compile of the 7-step graph for the compiled ref. Note:
            # `dynamic=None` should reuse the per-call cached graph -- the
            # 7 separate calls don't fuse, just stack in autograd.
            compiled_ttt7()
            torch.cuda.synchronize()

            # repeat=2 keeps overhead bounded; the 7-call graph is itself
            # already 7× more work per measurement than the 1-call profile.
            peak_kernel_ttt7_mb, transient_kernel_ttt7_mb = _measure_peak_vram(
                kernel_ttt7, repeat=2
            )
            peak_ref_ttt7_mb, transient_ref_ttt7_mb = _measure_peak_vram(
                ref_ttt7, repeat=2
            )
            peak_compiled_ttt7_mb, transient_compiled_ttt7_mb = _measure_peak_vram(
                compiled_ttt7, repeat=2
            )

            # ── Derived metrics (forward) ────────────────────────────
            kernel_fwd_us = kernel_fwd_ms * 1000.0
            ref_fwd_us = ref_fwd_ms * 1000.0
            throughput_tflops = flops / (kernel_fwd_ms / 1000.0) / 1e12 if kernel_fwd_ms > 0 else 0.0
            bandwidth_gb_s = nbytes / (kernel_fwd_ms / 1000.0) / 1e9 if kernel_fwd_ms > 0 else 0.0
            ref_throughput_tflops = flops / (ref_fwd_ms / 1000.0) / 1e12 if ref_fwd_ms > 0 else 0.0
            fwd_speedup = ref_fwd_ms / kernel_fwd_ms if kernel_fwd_ms > 0 else 0.0

            # ── Derived metrics (step = fwd+bwd) ─────────────────────
            kernel_step_us = kernel_step_ms * 1000.0
            ref_step_us = ref_step_ms * 1000.0
            # FLOPs for step: forward + ~2x backward (matmul + softmax bwd) ≈ 3x.
            step_flops = 3.0 * flops
            step_throughput_tflops = (
                step_flops / (kernel_step_ms / 1000.0) / 1e12 if kernel_step_ms > 0 else 0.0
            )
            step_speedup = ref_step_ms / kernel_step_ms if kernel_step_ms > 0 else 0.0

            arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
            ridge_point = (
                (gpu.peak_tflops_fp16 * 1e12) / (gpu.peak_bandwidth_gb_s * 1e9)
                if gpu.peak_bandwidth_gb_s > 0 else 0.0
            )
            bottleneck = "memory_bound" if arithmetic_intensity < ridge_point else "compute_bound"
            pct_peak_compute = (
                throughput_tflops / gpu.peak_tflops_fp16 * 100.0 if gpu.peak_tflops_fp16 > 0 else 0.0
            )
            pct_peak_bandwidth = (
                bandwidth_gb_s / gpu.peak_bandwidth_gb_s * 100.0
                if gpu.peak_bandwidth_gb_s > 0 else 0.0
            )

            mem_savings_fwd_mb = max(0.0, transient_ref_fwd_mb - transient_kernel_fwd_mb)
            mem_ratio_fwd = (
                transient_kernel_fwd_mb / transient_ref_fwd_mb if transient_ref_fwd_mb > 0 else 1.0
            )
            mem_savings_step_mb = max(0.0, transient_ref_step_mb - transient_kernel_step_mb)
            mem_ratio_step = (
                transient_kernel_step_mb / transient_ref_step_mb if transient_ref_step_mb > 0 else 1.0
            )

            # ── Derived metrics (vs compiled) ────────────────────────
            compiled_fwd_us = compiled_fwd_ms * 1000.0
            compiled_step_us = compiled_step_ms * 1000.0
            fwd_speedup_compiled = (
                compiled_fwd_ms / kernel_fwd_ms if kernel_fwd_ms > 0 else 0.0
            )
            step_speedup_compiled = (
                compiled_step_ms / kernel_step_ms if kernel_step_ms > 0 else 0.0
            )
            mem_savings_step_vs_compiled_mb = max(
                0.0, transient_compiled_step_mb - transient_kernel_step_mb
            )
            mem_ratio_step_vs_compiled = (
                transient_kernel_step_mb / transient_compiled_step_mb
                if transient_compiled_step_mb > 0 else 1.0
            )

            # ── Derived TTT-7 metrics (production-faithful headline) ───
            mem_savings_ttt7_vs_pytorch_mb = max(
                0.0, transient_ref_ttt7_mb - transient_kernel_ttt7_mb
            )
            mem_ratio_ttt7_vs_pytorch = (
                transient_kernel_ttt7_mb / transient_ref_ttt7_mb
                if transient_ref_ttt7_mb > 0 else 1.0
            )
            mem_savings_ttt7_vs_compiled_mb = max(
                0.0, transient_compiled_ttt7_mb - transient_kernel_ttt7_mb
            )
            mem_ratio_ttt7_vs_compiled = (
                transient_kernel_ttt7_mb / transient_compiled_ttt7_mb
                if transient_compiled_ttt7_mb > 0 else 1.0
            )

            entry = {
                "label": label,
                "size": sz,
                "dtype": str(dtype),
                "flops": flops,
                "bytes": nbytes,
                # Forward-only
                "kernel_fwd_latency_us": kernel_fwd_us,
                "pytorch_fwd_latency_us": ref_fwd_us,
                "fwd_speedup_vs_pytorch": fwd_speedup,
                "throughput_tflops": throughput_tflops,
                "bandwidth_gb_s": bandwidth_gb_s,
                "ref_throughput_tflops": ref_throughput_tflops,
                "pct_peak_compute": pct_peak_compute,
                "pct_peak_bandwidth": pct_peak_bandwidth,
                "arithmetic_intensity": arithmetic_intensity,
                "ridge_point": ridge_point,
                "bottleneck": bottleneck,
                "peak_kernel_fwd_vram_mb": peak_kernel_fwd_mb,
                "transient_kernel_fwd_vram_mb": transient_kernel_fwd_mb,
                "peak_pytorch_fwd_vram_mb": peak_ref_fwd_mb,
                "transient_pytorch_fwd_vram_mb": transient_ref_fwd_mb,
                "mem_savings_fwd_mb": mem_savings_fwd_mb,
                "mem_ratio_fwd": mem_ratio_fwd,
                # Step (fwd+bwd) -- primary metrics
                "kernel_step_latency_us": kernel_step_us,
                "pytorch_step_latency_us": ref_step_us,
                "step_speedup_vs_pytorch": step_speedup,
                "step_throughput_tflops": step_throughput_tflops,
                "peak_kernel_step_vram_mb": peak_kernel_step_mb,
                "transient_kernel_step_vram_mb": transient_kernel_step_mb,
                "peak_pytorch_step_vram_mb": peak_ref_step_mb,
                "transient_pytorch_step_vram_mb": transient_ref_step_mb,
                "mem_savings_step_mb": mem_savings_step_mb,
                "mem_ratio_step_kernel_over_pytorch": mem_ratio_step,
                # Compiled (torch.compile) -- THE production baseline
                "compiled_fwd_latency_us": compiled_fwd_us,
                "compiled_step_latency_us": compiled_step_us,
                "fwd_speedup_vs_compiled": fwd_speedup_compiled,
                "step_speedup_vs_compiled": step_speedup_compiled,
                "peak_compiled_fwd_vram_mb": peak_compiled_fwd_mb,
                "transient_compiled_fwd_vram_mb": transient_compiled_fwd_mb,
                "peak_compiled_step_vram_mb": peak_compiled_step_mb,
                "transient_compiled_step_vram_mb": transient_compiled_step_mb,
                "mem_savings_step_vs_compiled_mb": mem_savings_step_vs_compiled_mb,
                "mem_ratio_step_kernel_over_compiled": mem_ratio_step_vs_compiled,
                # TTT-7 (production-faithful: 7 fwds, 1 bwd over weighted sum)
                "ttt_steps": TTT_STEPS,
                "peak_kernel_ttt7_vram_mb": peak_kernel_ttt7_mb,
                "transient_kernel_ttt7_vram_mb": transient_kernel_ttt7_mb,
                "peak_pytorch_ttt7_vram_mb": peak_ref_ttt7_mb,
                "transient_pytorch_ttt7_vram_mb": transient_ref_ttt7_mb,
                "peak_compiled_ttt7_vram_mb": peak_compiled_ttt7_mb,
                "transient_compiled_ttt7_vram_mb": transient_compiled_ttt7_mb,
                "mem_savings_ttt7_vs_pytorch_mb": mem_savings_ttt7_vs_pytorch_mb,
                "mem_ratio_ttt7_kernel_over_pytorch": mem_ratio_ttt7_vs_pytorch,
                "mem_savings_ttt7_vs_compiled_mb": mem_savings_ttt7_vs_compiled_mb,
                "mem_ratio_ttt7_kernel_over_compiled": mem_ratio_ttt7_vs_compiled,
                # Back-compat aliases (primary == step)
                "kernel_latency_us": kernel_step_us,
                "pytorch_latency_us": ref_step_us,
                "speedup_vs_pytorch": step_speedup,
                "peak_kernel_vram_mb": peak_kernel_step_mb,
                "transient_kernel_vram_mb": transient_kernel_step_mb,
                "transient_pytorch_vram_mb": transient_ref_step_mb,
            }
            all_results.append(entry)
            if label == primary_label:
                primary_result = entry

            print(f"    fwd:      {kernel_fwd_us:.1f} us  eager={ref_fwd_us:.1f} us  "
                  f"compiled={compiled_fwd_us:.1f} us  "
                  f"x_e={fwd_speedup:.3f}x  x_c={fwd_speedup_compiled:.3f}x")
            print(f"    step:     {kernel_step_us:.1f} us  eager={ref_step_us:.1f} us  "
                  f"compiled={compiled_step_us:.1f} us  "
                  f"x_e={step_speedup:.3f}x  x_c={step_speedup_compiled:.3f}x")
            print(f"    step-VRAM: kernel={transient_kernel_step_mb:.1f} MB  "
                  f"eager={transient_ref_step_mb:.1f} MB  "
                  f"compiled={transient_compiled_step_mb:.1f} MB  "
                  f"saved_vs_e={mem_savings_step_mb:.1f} ({(1.0 - mem_ratio_step) * 100:.0f}%)  "
                  f"saved_vs_c={mem_savings_step_vs_compiled_mb:.1f} "
                  f"({(1.0 - mem_ratio_step_vs_compiled) * 100:.0f}%)")
            print(f"    ttt7-VRAM: kernel={transient_kernel_ttt7_mb:.1f} MB  "
                  f"eager={transient_ref_ttt7_mb:.1f} MB  "
                  f"compiled={transient_compiled_ttt7_mb:.1f} MB  "
                  f"saved_vs_e={mem_savings_ttt7_vs_pytorch_mb:.1f} "
                  f"({(1.0 - mem_ratio_ttt7_vs_pytorch) * 100:.0f}%)  "
                  f"saved_vs_c={mem_savings_ttt7_vs_compiled_mb:.1f} "
                  f"({(1.0 - mem_ratio_ttt7_vs_compiled) * 100:.0f}%)  "
                  f"<-- production-faithful")

            del inputs, grad_inputs, kernel_step, ref_step
            del grad_inputs_c, compiled_step
            del kernel_ttt7, ref_ttt7, compiled_ttt7
        except torch.cuda.OutOfMemoryError:
            print(f"    SKIP: {label} -> OOM"); torch.cuda.empty_cache()
        except BenchTimeoutError:
            print(f"    SKIP: {label} -> TIMEOUT")
        except Exception as e:
            print(f"    ERROR: {label} -> {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    if primary_result is None and all_results:
        primary_result = all_results[-1]
    return {"primary": primary_result, "all": all_results}


# =========================================================================
# 6. PROFILER
# =========================================================================
def run_profile(kernel_fn: Callable, config: dict):
    device = "cuda"
    gen_fn = config["input_generator"]
    sizes = config["test_sizes"]
    prof_size = next((sz for lbl, sz in sizes if lbl == "medium"), sizes[0][1])
    dtype = config["test_dtypes"][0]
    inputs = gen_fn(prof_size, dtype, device, seed=42)
    trace_dir = "./traces"
    os.makedirs(trace_dir, exist_ok=True)
    print("\n=== PROFILING ===")
    print(f"size: {prof_size}  dtype: {dtype}")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            kernel_fn(**inputs)
        torch.cuda.synchronize()
        for _ in range(10):
            kernel_fn(**inputs)
        torch.cuda.synchronize()
    trace_path = os.path.join(trace_dir, "kernel_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"profile_trace: {trace_path}")
    try:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# =========================================================================
# 7. MAIN
# =========================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Eagle3 loss-kernel benchmark harness")
    parser.add_argument("--kernel", type=str, default=None)
    parser.add_argument("--sizes", type=str, default="all")
    parser.add_argument("--quick", action="store_true",
                        help="Skip stages 3-5 and bench only the primary size")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Eagle3 AutoKernel Benchmark Harness")
    print("=" * 60)

    # Resolve kernel.py from the current working dir or this script's dir.
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.getcwd(), here):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn
        kernel_type = args.kernel or getattr(kernel_module, "KERNEL_TYPE", None)
        if kernel_type is None:
            print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
            sys.exit(1)
        print(f"kernel_type: {kernel_type}")
    except SyntaxError as e:
        print(f"\nERROR: kernel.py has a syntax error: {e}")
        traceback.print_exc()
        print("\ncorrectness: FAIL"); print("throughput_tflops: 0.000")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to import kernel.py: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\ncorrectness: FAIL"); print("throughput_tflops: 0.000")
        sys.exit(1)

    if kernel_type not in KERNEL_CONFIGS:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'  "
              f"(available: {', '.join(KERNEL_CONFIGS.keys())})")
        sys.exit(1)
    config = KERNEL_CONFIGS[kernel_type]

    gpu = detect_gpu()
    print("\n=== GPU INFO ===")
    print(f"gpu_name: {gpu.name}")
    print(f"gpu_sm_count: {gpu.sm_count}")
    print(f"gpu_memory_gb: {gpu.memory_gb}")
    print(f"gpu_peak_tflops_bf16: {gpu.peak_tflops_bf16}")
    print(f"gpu_peak_bandwidth_gb_s: {gpu.peak_bandwidth_gb_s}")
    print(f"gpu_compute_capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")

    print("\n=== CORRECTNESS ===")
    try:
        correctness_results = run_correctness(kernel_fn, config, quick=args.quick)
    except Exception as e:
        print(f"\nFATAL: correctness crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        correctness_results = {
            "correctness": "FAIL",
            "smoke_test": "CRASH",
            "shape_sweep": "CRASH",
            "numerical_stability": "CRASH",
            "determinism": "CRASH",
            "edge_cases": "CRASH",
        }
    print("\n--- Correctness Summary ---")
    for k in ("smoke_test", "shape_sweep", "numerical_stability",
              "determinism", "edge_cases", "backward_grad", "subset_invariance"):
        print(f"{k}: {correctness_results.get(k, 'N/A')}")
    print(f"correctness: {correctness_results['correctness']}")

    sizes_filter = "large" if args.quick else args.sizes
    print(f"\n=== PERFORMANCE (sizes={sizes_filter}) ===")
    perf_results = {"primary": None, "all": []}
    overall_peak_vram_mb = 0.0
    try:
        torch.cuda.reset_peak_memory_stats()
        perf_results = run_performance(kernel_fn, config, gpu, sizes_filter=sizes_filter)
        overall_peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception as e:
        print(f"\nFATAL: performance crashed: {type(e).__name__}: {e}")
        traceback.print_exc()

    primary = perf_results.get("primary")
    if primary is not None:
        print(f"\n--- Performance Summary (primary: {primary['label']}) ---")
        # Forward-only metrics
        print(f"kernel_fwd_latency_us: {primary['kernel_fwd_latency_us']:.2f}")
        print(f"pytorch_fwd_latency_us: {primary['pytorch_fwd_latency_us']:.2f}")
        print(f"fwd_speedup_vs_pytorch: {primary['fwd_speedup_vs_pytorch']:.3f}x")
        print(f"throughput_tflops: {primary['throughput_tflops']:.3f}")
        print(f"bandwidth_gb_s: {primary['bandwidth_gb_s']:.1f}")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}%")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        print(f"arithmetic_intensity: {primary['arithmetic_intensity']:.2f}")
        print(f"ridge_point: {primary['ridge_point']:.2f}")
        print(f"bottleneck: {primary['bottleneck']}")
        print(f"flops: {primary['flops']:.0f}")
        print(f"bytes: {primary['bytes']:.0f}")
        # Forward-only memory
        print(f"transient_kernel_fwd_vram_mb: {primary['transient_kernel_fwd_vram_mb']:.1f}")
        print(f"transient_pytorch_fwd_vram_mb: {primary['transient_pytorch_fwd_vram_mb']:.1f}")
        print(f"mem_savings_fwd_mb: {primary['mem_savings_fwd_mb']:.1f}")
        print(f"mem_ratio_fwd: {primary['mem_ratio_fwd']:.3f}")
        # Step (fwd+bwd) -- PRIMARY metrics (kernel vs eager)
        print(f"kernel_step_latency_us: {primary['kernel_step_latency_us']:.2f}")
        print(f"pytorch_step_latency_us: {primary['pytorch_step_latency_us']:.2f}")
        print(f"step_speedup_vs_pytorch: {primary['step_speedup_vs_pytorch']:.3f}x")
        print(f"step_throughput_tflops: {primary['step_throughput_tflops']:.3f}")
        print(f"peak_kernel_step_vram_mb: {primary['peak_kernel_step_vram_mb']:.1f}")
        print(f"transient_kernel_step_vram_mb: {primary['transient_kernel_step_vram_mb']:.1f}")
        print(f"transient_pytorch_step_vram_mb: {primary['transient_pytorch_step_vram_mb']:.1f}")
        print(f"mem_savings_step_mb: {primary['mem_savings_step_mb']:.1f}")
        print(f"mem_ratio_step_kernel_over_pytorch: "
              f"{primary['mem_ratio_step_kernel_over_pytorch']:.3f}")
        print(f"peak_vram_mb: {overall_peak_vram_mb:.1f}")
        # Compiled (torch.compile) -- THE production baseline
        print(f"compiled_fwd_latency_us: {primary['compiled_fwd_latency_us']:.2f}")
        print(f"compiled_step_latency_us: {primary['compiled_step_latency_us']:.2f}")
        print(f"fwd_speedup_vs_compiled: {primary['fwd_speedup_vs_compiled']:.3f}x")
        print(f"step_speedup_vs_compiled: {primary['step_speedup_vs_compiled']:.3f}x")
        print(f"transient_compiled_step_vram_mb: "
              f"{primary['transient_compiled_step_vram_mb']:.1f}")
        print(f"mem_savings_step_vs_compiled_mb: "
              f"{primary['mem_savings_step_vs_compiled_mb']:.1f}")
        print(f"mem_ratio_step_kernel_over_compiled: "
              f"{primary['mem_ratio_step_kernel_over_compiled']:.3f}")
        # TTT-7 -- production training step memory profile
        print(f"ttt_steps: {primary['ttt_steps']}")
        print(f"peak_kernel_ttt7_vram_mb: {primary['peak_kernel_ttt7_vram_mb']:.1f}")
        print(f"transient_kernel_ttt7_vram_mb: "
              f"{primary['transient_kernel_ttt7_vram_mb']:.1f}")
        print(f"transient_pytorch_ttt7_vram_mb: "
              f"{primary['transient_pytorch_ttt7_vram_mb']:.1f}")
        print(f"transient_compiled_ttt7_vram_mb: "
              f"{primary['transient_compiled_ttt7_vram_mb']:.1f}")
        print(f"mem_savings_ttt7_vs_pytorch_mb: "
              f"{primary['mem_savings_ttt7_vs_pytorch_mb']:.1f}")
        print(f"mem_savings_ttt7_vs_compiled_mb: "
              f"{primary['mem_savings_ttt7_vs_compiled_mb']:.1f}")
        print(f"mem_ratio_ttt7_kernel_over_pytorch: "
              f"{primary['mem_ratio_ttt7_kernel_over_pytorch']:.3f}")
        print(f"mem_ratio_ttt7_kernel_over_compiled: "
              f"{primary['mem_ratio_ttt7_kernel_over_compiled']:.3f}")
        # Back-compat aliases for greppers
        print(f"latency_us: {primary['kernel_latency_us']:.2f}  (alias: step)")
        print(f"speedup_vs_pytorch: {primary['speedup_vs_pytorch']:.3f}x  (alias: step)")
        print(f"transient_kernel_vram_mb: {primary['transient_kernel_vram_mb']:.1f}  (alias: step)")
    else:
        for k in (
            "kernel_fwd_latency_us", "pytorch_fwd_latency_us", "fwd_speedup_vs_pytorch",
            "throughput_tflops", "bandwidth_gb_s", "pct_peak_compute", "pct_peak_bandwidth",
            "transient_kernel_fwd_vram_mb", "transient_pytorch_fwd_vram_mb",
            "kernel_step_latency_us", "pytorch_step_latency_us", "step_speedup_vs_pytorch",
            "transient_kernel_step_vram_mb", "transient_pytorch_step_vram_mb",
            "compiled_fwd_latency_us", "compiled_step_latency_us",
            "fwd_speedup_vs_compiled", "step_speedup_vs_compiled",
            "transient_compiled_step_vram_mb", "mem_ratio_step_kernel_over_compiled",
            "transient_kernel_ttt7_vram_mb", "transient_pytorch_ttt7_vram_mb",
            "transient_compiled_ttt7_vram_mb",
            "mem_ratio_ttt7_kernel_over_pytorch", "mem_ratio_ttt7_kernel_over_compiled",
            "latency_us", "speedup_vs_pytorch", "transient_kernel_vram_mb",
        ):
            print(f"{k}: 0.0")
        print(f"peak_vram_mb: {overall_peak_vram_mb:.1f}")

    all_perf = perf_results.get("all", [])
    if len(all_perf) > 1:
        print("\n=== SIZE SWEEP vs EAGER (step = fwd+bwd) ===")
        print(f"{'size':<8} {'fwd_us(k|e)':>20} {'step_us(k|e)':>20} "
              f"{'step_x':>7} {'fwd_vram(k|e)':>16} {'step_vram(k|e)':>16}")
        print("-" * 92)
        for e in all_perf:
            fwd = f"{e['kernel_fwd_latency_us']:>9.1f}|{e['pytorch_fwd_latency_us']:<8.1f}"
            step = f"{e['kernel_step_latency_us']:>9.1f}|{e['pytorch_step_latency_us']:<8.1f}"
            fmem = f"{e['transient_kernel_fwd_vram_mb']:>7.1f}|{e['transient_pytorch_fwd_vram_mb']:<7.1f}"
            smem = f"{e['transient_kernel_step_vram_mb']:>7.1f}|{e['transient_pytorch_step_vram_mb']:<7.1f}"
            print(f"{e['label']:<8} {fwd:>20} {step:>20} "
                  f"{e['step_speedup_vs_pytorch']:>6.3f}x {fmem:>16} {smem:>16}")

        print("\n=== SIZE SWEEP vs COMPILED (production @torch.compile baseline) ===")
        print(f"{'size':<8} {'fwd_us(k|c)':>20} {'step_us(k|c)':>20} "
              f"{'step_x_c':>9} {'fwd_vram(k|c)':>16} {'step_vram(k|c)':>16}")
        print("-" * 94)
        for e in all_perf:
            fwd = f"{e['kernel_fwd_latency_us']:>9.1f}|{e['compiled_fwd_latency_us']:<8.1f}"
            step = f"{e['kernel_step_latency_us']:>9.1f}|{e['compiled_step_latency_us']:<8.1f}"
            fmem = f"{e['transient_kernel_fwd_vram_mb']:>7.1f}|{e['transient_compiled_fwd_vram_mb']:<7.1f}"
            smem = f"{e['transient_kernel_step_vram_mb']:>7.1f}|{e['transient_compiled_step_vram_mb']:<7.1f}"
            print(f"{e['label']:<8} {fwd:>20} {step:>20} "
                  f"{e['step_speedup_vs_compiled']:>8.3f}x {fmem:>16} {smem:>16}")

        print(f"\n=== SIZE SWEEP TTT-{TTT_STEPS} MEMORY "
              f"(production training-step VRAM profile) ===")
        print(f"{'size':<8} {'ttt_vram_kernel':>17} {'ttt_vram_eager':>17} "
              f"{'ttt_vram_compiled':>19} {'ratio_v_e':>10} {'ratio_v_c':>10}")
        print("-" * 90)
        for e in all_perf:
            print(f"{e['label']:<8} "
                  f"{e['transient_kernel_ttt7_vram_mb']:>14.1f} MB "
                  f"{e['transient_pytorch_ttt7_vram_mb']:>14.1f} MB "
                  f"{e['transient_compiled_ttt7_vram_mb']:>16.1f} MB "
                  f"{e['mem_ratio_ttt7_kernel_over_pytorch']:>9.3f}x "
                  f"{e['mem_ratio_ttt7_kernel_over_compiled']:>9.3f}x")

    if args.profile:
        try:
            run_profile(kernel_fn, config)
        except Exception as e:
            print(f"\nWARNING: profiling failed: {type(e).__name__}: {e}")

    t_elapsed = time.time() - t_start
    throughput = primary["throughput_tflops"] if primary else 0.0
    print("\n=== FINAL ===")
    print(f"kernel_type: {kernel_type}")
    print(f"correctness: {correctness_results['correctness']}")
    print(f"backward_grad: {correctness_results.get('backward_grad', 'N/A')}")
    print(f"subset_invariance: {correctness_results.get('subset_invariance', 'N/A')}")
    print(f"throughput_tflops: {throughput:.3f}")
    if primary:
        # ── Speed headlines (1-call per-step) ─────────────────────
        # Per-call timing is right: training-step time is approx 7× this.
        print(f"step_speedup_vs_pytorch: {primary['step_speedup_vs_pytorch']:.3f}x")
        print(f"step_speedup_vs_compiled: {primary['step_speedup_vs_compiled']:.3f}x  "
              f"<-- THE production-relevant SPEED headline (per-call)")
        print(f"fwd_speedup_vs_pytorch: {primary['fwd_speedup_vs_pytorch']:.3f}x")
        print(f"fwd_speedup_vs_compiled: {primary['fwd_speedup_vs_compiled']:.3f}x")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}%")
        # ── Memory headlines (TTT-7 production training step) ────
        # The 1-call memory numbers underrepresent production -- prod
        # holds saved tensors from all 7 calls until backward.
        print(f"transient_kernel_ttt7_vram_mb: "
              f"{primary['transient_kernel_ttt7_vram_mb']:.1f}  "
              f"<-- THE production-faithful MEMORY headline (TTT-{primary['ttt_steps']})")
        print(f"mem_ratio_ttt7_kernel_over_compiled: "
              f"{primary['mem_ratio_ttt7_kernel_over_compiled']:.3f}  "
              f"<-- ship bar: <= 1.0 means kernel beats production memory")
        print(f"mem_ratio_ttt7_kernel_over_pytorch: "
              f"{primary['mem_ratio_ttt7_kernel_over_pytorch']:.3f}")
        # 1-call memory (per-step diagnostic)
        print(f"transient_kernel_step_vram_mb: {primary['transient_kernel_step_vram_mb']:.1f}  "
              f"(per-call diagnostic; multiply by ~{primary['ttt_steps']} for prod)")
        print(f"mem_ratio_step_kernel_over_pytorch: "
              f"{primary['mem_ratio_step_kernel_over_pytorch']:.3f}")
        print(f"mem_ratio_step_kernel_over_compiled: "
              f"{primary['mem_ratio_step_kernel_over_compiled']:.3f}")
        # Aliases for back-compat
        print(f"speedup_vs_pytorch: {primary['step_speedup_vs_pytorch']:.3f}x  (alias: step)")
        print(f"transient_kernel_vram_mb: {primary['transient_kernel_step_vram_mb']:.1f}  (alias: step)")
    else:
        print("step_speedup_vs_pytorch: 0.000x"); print("fwd_speedup_vs_pytorch: 0.000x")
        print("step_speedup_vs_compiled: 0.000x"); print("fwd_speedup_vs_compiled: 0.000x")
        print("transient_kernel_ttt7_vram_mb: 0.0")
        print("mem_ratio_ttt7_kernel_over_compiled: 0.000")
        print("pct_peak_bandwidth: 0.0%")
    print(f"bench_time_seconds: {t_elapsed:.1f}")
    if t_elapsed > 180:
        print(f"WARNING: bench.py took {t_elapsed:.1f}s (budget: 180s with backward at prod vocab)")


if __name__ == "__main__":
    main()
