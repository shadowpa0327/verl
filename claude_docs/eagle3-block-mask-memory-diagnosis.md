# EAGLE3 `create_block_mask` peak-memory diagnosis

> **Status:** investigation complete, fix proposed (1-line change in `recipe/drafter_cotraining/eagle3/ops/flex_attention.py`). Pending senior review.
>
> **Audience:** anyone touching the drafter co-training attention path or hitting OOMs that can be traced back to `flex_attention`.

---

## TL;DR

Every call to `create_block_mask` from the EAGLE3 drafter forward (`recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1434`) currently allocates **~476 MB** of peak GPU memory at the realistic training shape (`B=1, Q_LEN=2048, KV_LEN=14336`). The final `BlockMask` is **<0.05 MB** — the 476 MB is pure construction overhead.

Adding `_compile=True` to that call shrinks the peak to **~0.07 MB (a ~6800× reduction)** and cuts wall time from 8.2 ms to 1.4 ms. Validates the [PyTorch FlexAttention blog post](https://pytorch.org/blog/flexattention/#q-how-can-we-compute-blockmask-quicker).

The cost multiplies by `num_drafter_layers × num_microbatches × num_TTT_steps`, so the savings at full configuration are non-trivial (rough order-of-magnitude: hundreds of MB to several GB across a step).

---

## What happened

A reviewer asked us to diagnose the memory cost of `compile_friendly_create_block_mask()` defined in `recipe/drafter_cotraining/eagle3/ops/flex_attention.py:68`. The wrapper is one line over PyTorch's `torch.nn.attention.flex_attention.create_block_mask`, used by the drafter attention to build a sparse causal+suffix mask once per attention forward.

We split the work three ways:

1. **Map the call sites.** Find every place the wrapper (and `generate_eagle3_mask`) is invoked, with the exact `B/H/Q_LEN/KV_LEN` expressions.
2. **Pin the runtime shapes.** Trace EAGLE3 drafter training for `prompt_len=2048, response_len=2048` and resolve concrete numbers for those four dimensions.
3. **Measure.** Build a bench script and sweep the dimensions on an H100, reporting peak GPU memory and wall time.

### What we found at the call site

```python
# recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1425-1446 (LlamaFlexAttention.forward)
# TODO: Remove the usage of uncompiled create_block_mask after
# https://github.com/pytorch/pytorch/issues/160018
if q_len <= 128:
    create_block_mask_func = create_block_mask
    flex_attention_func = flex_attention
else:
    create_block_mask_func = compile_friendly_create_block_mask
    flex_attention_func = compile_friendly_flex_attention

block_mask = create_block_mask_func(
    mask_mod=generate_eagle3_mask(
        seq_lengths=seq_lengths,
        Q_LEN=q_len,
        KV_LEN=key_cache.shape[-2],
        lck=lck,
    ),
    B=bsz,
    H=1,                       # rely on broadcast across heads
    Q_LEN=q_len,
    KV_LEN=key_cache.shape[-2],
    device=query_states.device,
)
```

### Concrete shapes during training (prompt_len=2048, response_len=2048)

| dim | value | source |
|---|---|---|
| `B` | `1` (microbatch) | `recipe/drafter_cotraining/config/drafter_ct_trainer.yaml:200` (`micro_batch_size_per_gpu: 1`) |
| `H` | `1` (broadcast over heads) | `llama3_eagle.py:1442` |
| `Q_LEN` | `2048` (response length) | inferred from rollout payload; matches `claude_docs/drafter-design.md` |
| `lck` | `0..6` (TTT step index, `length-1=6` from `eagle3_model.py:101`) | TTT loop |
| `KV_LEN` | `Q_LEN * (1 + lck)` → up to `14336` | `key_cache.shape[-2]` after concat across TTT iterations |

The `H=1` is **not** the model's true head count (32). It's a deliberate broadcast: the EAGLE3 mask doesn't depend on `h`, so a one-head mask broadcasts to all heads inside `flex_attention(enable_gqa=True)`. We confirmed this is identical to passing `H=None`, and that passing `H=32` (no broadcast) blows peak to 8 GB.

The mask **does** depend on `b` (`seq_lengths[b]`), so `B` cannot be broadcast — `B=bsz` is correct.

There are three call sites of `create_block_mask` in the active recipe:

1. `recipe/drafter_cotraining/eagle3/ops/flex_attention.py:76` — the wrapper.
2. `recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1434` — main attention forward (uses the wrapper for `q_len > 128`).
3. `recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1017` — `_get_block_sparse()` cold path for the FA4 cache (`B=1, H=1`). Cache hits are free; cache misses pay the same construction cost.

### Benchmark results

Hardware: NVIDIA H100 80GB. PyTorch 2.10. Script: `recipe/drafter_cotraining/scripts/bench_eagle3_block_mask.py`.

**Sweep 1 — KV_LEN (TTT step), B=1, H=1, Q_LEN=2048:**

| `lck` | `KV_LEN` | peak MB | time ms | sparsity |
|--:|--:|--:|--:|--:|
| 0 | 2048 | 68.0 | 6.3 | 0.74 |
| 1 | 4096 | 136.1 | 6.3 | 0.85 |
| 2 | 6144 | 204.1 | 6.3 | 0.89 |
| 3 | 8192 | 272.1 | 7.9 | 0.90 |
| 4 | 10240 | 340.1 | 6.1 | 0.91 |
| 5 | 12288 | 408.2 | 6.2 | 0.92 |
| **6** | **14336** | **476.2** | 8.3 | 0.93 |

Linear in `KV_LEN`. Final BlockMask metadata is ~0.01 MB across all rows — every byte of the 476 MB is transient.

**Sweep 2 — batch size, lck=6, H=1, Q_LEN=2048:**

| `B` | peak MB |
|--:|--:|
| 1 | 476.2 |
| 2 | 560.1 |
| 4 | 1120.2 |
| 8 | **2240.3** |

Roughly linear past `B≥2`. At microbatch=8 a single `create_block_mask` call peaks at 2.24 GB.

**Sweep 3 — H broadcast, B=1, lck=6, Q_LEN=2048:**

| `H` arg | peak MB |
|--:|--:|
| `None` (broadcast) | 476.2 |
| `1` (current code) | 476.2 |
| `32` (no broadcast) | **8093.3** |

Confirms the `H=1` choice in the codebase. No action needed there.

**Sweep 4 — `BLOCK_SIZE`, B=1, lck=6, Q_LEN=2048, H=1:**

| `BLOCK_SIZE` | peak MB | sparsity |
|--:|--:|--:|
| 64 | 476.3 | 0.95 |
| 128 (default) | 476.2 | 0.93 |
| 256 | 476.2 | 0.87 |
| 512 | 476.2 | 0.79 |

`BLOCK_SIZE` has essentially no effect on construction peak (it only changes the size of the final metadata, which is already negligible). Bigger blocks lower sparsity, which would slow the eventual `flex_attention` kernel — orthogonal concern.

**Sweep 5 — `_compile=True`, B=1, lck=6, Q_LEN=2048, H=1, BLOCK_SIZE=128:**

| `_compile` | peak MB | time ms |
|---|--:|--:|
| `False` (current) | 476.18 | 8.2 |
| `True` | **0.07** | **1.4** |

Same final BlockMask, ~6800× less peak memory, ~6× faster.

---

## The insight

`create_block_mask` produces a tiny artifact (a list of block indices) but its **default Python path** materializes a dense `(B, H, Q_LEN, KV_LEN)` boolean tensor first by calling `mask_mod(b, h, q_idx, kv_idx)` element-wise, then reduces it to the sparse representation. With our shapes:

```
B · H · Q_LEN · KV_LEN  =  1 · 1 · 2048 · 14336  ≈  29.4M booleans
```

Boolean tensors on CUDA are stored as `int8` (1 B/elem), but the broadcasting machinery and intermediate index tensors push that to ~476 MB peak. Increase `B` to 8 and you scale linearly to ~2.2 GB. The mask is **92% sparse**, so 92% of the work going into the dense intermediate is thrown away the moment it's reduced.

The output BlockMask metadata is `(B, H, Q_LEN/BLOCK_SIZE, KV_LEN/BLOCK_SIZE)` ints, which at default block size 128 is `(1, 1, 16, 112) = 1792` ints — under 8 KB. The blow-up is entirely in construction.

This is exactly the failure mode the FlexAttention blog calls out:

> "Compile `create_block_mask`. […] you can set `_compile=True`, which will significantly reduce the peak memory and runtime (often an order of magnitude in our testing)."

We measured ~3 orders of magnitude on this mask, not one. The reason this mask benefits more than the average is its high sparsity (92%) at large `KV_LEN`: the dense intermediate is huge, but the sparse output is essentially free, so removing the intermediate is a near-100% win.

---

## What `_compile=True` actually does

When you pass `_compile=True`, PyTorch:

1. Wraps `mask_mod` in a Triton kernel that's evaluated **block-wise**, not element-wise. For each `(q_block, kv_block)` pair, the kernel computes whether *any* element in the block is masked-in. If yes, it emits the block index; if no, it skips.
2. Caches the compiled kernel keyed on shape signature + mask_mod identity. Subsequent calls with the same `(B, H, Q_LEN, KV_LEN, BLOCK_SIZE)` reuse the kernel.
3. Never instantiates the dense `(B, H, Q_LEN, KV_LEN)` boolean tensor.

The result is identical to the Python path — same `kv_indices`, same `kv_num_blocks` — only the build path differs.

### What you pay in return

- **First-call compile time.** Adds ~seconds the first time a new shape signature is seen. Amortized across thousands of training steps, this is rounding error.
- **Recompile risk if `mask_mod` identity churns.** Each `generate_eagle3_mask(...)` call returns a fresh `or_masks(causal_mask, suffix_mask)` closure, but the kernel is keyed on the structural mask body, not the Python object identity, so this is fine in practice. Worth keeping `dynamo.config.recompile_limit = 128` (already set in `flex_attention.py:28`) as a safety net.
- **Recompile if shape signature changes.** Different `(Q_LEN, KV_LEN, B)` tuples each get their own kernel. EAGLE3 training has 7 TTT steps × a small number of microbatch sizes, so we expect a single-digit number of compiled kernels per run.

### `_compile=True` vs `torch.compile(create_block_mask)`

PyTorch 2.10 emits:
```
DeprecationWarning: _compile flag on create_block_mask was originally added to work
around a torch.compile limitation. That limitation has since been addressed.
So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask).
```

We tested both head-to-head (`--mode compile_alt`, fresh-process verification):

| approach | cold compile (first 2 calls) | steady-state ms | peak MB | closure churn safe? |
|---|--:|--:|--:|---|
| `_compile=True` | ~3 s total | 2.1 | 0.04 | yes |
| `torch.compile(create_block_mask)` | ~3 s total | **0.7** | 0.04 | yes |

`torch.compile()` is **~3× faster steady-state** with identical memory characteristics and identical recompile behavior. The earlier hedge about closure-keyed recompiles was wrong — confirmed empirically that fresh `mask_mod` closures do not trigger recompiles.

**Final implementation:** module-level `torch.compile(create_block_mask)` cached once at import. No singleton class needed — the existing `WrappedFlexAttention` lazy-singleton pattern at `flex_attention.py:36-53` adds no value here because (a) `torch.compile()` is itself lazy (kernel only built on first call), and (b) the `is_torchdynamo_compiling()` guard already prevents re-entry from a compiled context. Equivalent behavior, ~20 lines smaller.

```python
_compiled_create_block_mask = torch.compile(create_block_mask)


def compile_friendly_create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, **kwargs):
    fn = _compiled_create_block_mask if not is_torchdynamo_compiling() else create_block_mask
    return fn(mask_mod, B, H, Q_LEN, KV_LEN, device, **kwargs)
```

No deprecation warning. ~9× total speedup vs baseline (6.3 ms → 0.7 ms).

---

## When `_compile=True` helps (and when it doesn't)

| scenario | does `_compile=True` help? |
|---|---|
| Large `Q_LEN · KV_LEN` (≥ 1M elements per batch-head) | **Yes — big win.** The dense intermediate is the bottleneck. |
| Small `Q_LEN · KV_LEN` (≤ 64K) | Marginal. Construction is cheap either way; compile setup may dominate the first call. |
| Mask is highly sparse (≥ 50%) | **Yes — biggest win.** Block-wise evaluation skips entire empty blocks. |
| Mask is dense (e.g. plain causal) | Modest. Still saves the dense intermediate but the sparse output is also large. |
| Tight loop with frequent shape changes | Be careful. Many distinct shapes ⇒ many compiled kernels ⇒ higher startup. Bound `dynamo.config.recompile_limit`. |
| Stable shapes amortized over many steps | **Yes.** The amortized per-call cost approaches the kernel's runtime alone. |
| You're constructing `BlockMask` manually with known indices | N/A — the blog's third recommendation (custom constructor) is faster still, but only viable for masks with closed-form index patterns (plain causal, sliding window). EAGLE3's mask depends on per-batch `seq_lengths`, so closed-form is impractical. |

**EAGLE3 fits the "biggest win" cell on every axis:** large `KV_LEN`, very high sparsity, stable shapes within a run, called many times per step. This is why we measured ~6800× rather than the blog's quoted ~10×.

---

## Recommendation (the change)

**Primary change** — `recipe/drafter_cotraining/eagle3/ops/flex_attention.py:76`:

```python
def compile_friendly_create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device):
    return create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, _compile=True)
```

**Secondary** — `recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1017`: same `_compile=True` on the FA4 block-sparse cache cold path. Cache hits are unaffected, but cold misses currently pay the same 476 MB.

**Optional** — collapse the `q_len <= 128` branch at `llama3_eagle.py:1425-1432` so both paths use the compile-friendly wrapper. Requires verifying [pytorch/pytorch#160018](https://github.com/pytorch/pytorch/issues/160018) is a `flex_attention` issue, not a `create_block_mask` issue (our reading: it's the former, but worth confirming).

The `H=1` broadcast at `llama3_eagle.py:1442` is correct as-is — leave it.

---

## How to reproduce

```bash
# Single sweep (KV_LEN ladder, the most informative one):
python recipe/drafter_cotraining/scripts/bench_eagle3_block_mask.py --mode lck

# All five sweeps:
python recipe/drafter_cotraining/scripts/bench_eagle3_block_mask.py --mode all

# Validate the fix (after applying the patch):
python recipe/drafter_cotraining/scripts/bench_eagle3_block_mask.py --mode compile
# Expect peak < 1 MB on the True row.
```

The script prints peak GPU memory, median wall time, sparsity, and the size of the final `kv_indices` tensor for each row.

---

## Extended `_compile=True` effectiveness study

After the initial finding, we ran four more sweeps to characterize where the win holds, where it doesn't, and what it costs upfront. All on H100, PyTorch 2.10. Logs in `/tmp/bench_compile_*.log`.

### 1. Savings curve across the full TTT loop (`--mode compile_grid`)

The earlier result was a single point at lck=6. Here is every TTT step:

| `lck` | `KV_LEN` | peak `_cmp=F` MB | peak `_cmp=T` MB | mem ratio | time `F` ms | time `T` ms | time ratio |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 2048 | 68.04 | 0.01 | **4805×** | 6.20 | 1.37 | 4.52× |
| 1 | 4096 | 136.07 | 0.03 | 5258× | 6.34 | 1.57 | 4.05× |
| 2 | 6144 | 204.09 | 0.04 | 5573× | 6.50 | 1.56 | 4.16× |
| 3 | 8192 | 272.11 | 0.05 | 5745× | 6.41 | 1.52 | 4.21× |
| 4 | 10240 | 340.14 | 0.06 | 5805× | 6.36 | 1.56 | 4.07× |
| 5 | 12288 | 408.16 | 0.07 | 5887× | 6.45 | 1.54 | 4.18× |
| 6 | 14336 | 476.18 | 0.08 | **5947×** | 6.38 | 1.52 | 4.19× |

The win is universal across the loop. Memory ratio grows mildly with `KV_LEN` (because the `_compile=False` cost grows linearly while `_compile=True` stays near-constant). Time speedup is steady ~4×.

**Cumulative savings per drafter-layer per microbatch (one full TTT loop, lck=0..6):**
- `_compile=False`: 68 + 136 + 204 + 272 + 340 + 408 + 476 = **1.9 GB** of allocate/free traffic
- `_compile=True`:  ≈ **0.34 MB** total

### 2. Q_LEN sweep — does the `q_len <= 128` branch still earn its keep? (`--mode compile_qlen`, `lck=6`)

| `Q_LEN` | `KV_LEN` | peak `_cmp=F` MB | peak `_cmp=T` MB | mem ratio | time `F` ms | time `T` ms |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 448 | 0.59 | 0.01 | 59× | 6.19 | 1.43 |
| **128** | 896 | 1.87 | 0.01 | 187× | 6.50 | 1.62 |
| 256 | 1792 | 7.46 | 0.01 | 746× | 6.51 | 1.72 |
| 512 | 3584 | 29.79 | 0.01 | 2979× | 6.78 | 1.69 |
| 1024 | 7168 | 119.09 | 0.03 | 3970× | 6.65 | 1.68 |
| 2048 | 14336 | 476.18 | 0.08 | 5947× | 6.50 | 1.66 |
| 4096 | 28672 | **1904.42** | 0.31 | 6143× | 7.77 | 2.21 |

**Even at `Q_LEN=64` (below the `q_len <= 128` short-circuit threshold), `_compile=True` is 60× cheaper on memory and ~4× faster.** The branch in `llama3_eagle.py:1427-1432` was added to dodge the dense-intermediate cost; once we fix the wrapper, the branch is obsolete.

Note also the `Q_LEN=4096` row: peak hits **1.9 GB for a single call** without the fix. This is the failure mode that would explain OOMs at higher `response_len`.

### 3. First-call amortization (`--mode compile_amortize`, 10 consecutive calls at the lck=6 shape)

| call # | `_cmp=F` ms | `_cmp=T` ms | `_cmp=F` peak MB | `_cmp=T` peak MB |
|--:|--:|--:|--:|--:|
| 0 | 810.2 | **1978.8** | 476.2 | **50.0** |
| 1 | 6.5 | 2.7 | 476.2 | 0.04 |
| 2..9 | 6.0–6.5 | 2.5–2.8 | 476.2 | 0.04 |

**Steady-state (calls 3..9):** `_cmp=False`=6.27 ms, `_cmp=True`=2.57 ms → **2.44× speedup.**

**First-call compile tax:** ~1.98 s extra wall time on call #0 to JIT the kernel. Time-only break-even after ~534 same-shape calls; in a real training run with 7 TTT steps × ~30 drafter layers × thousands of training steps, that breakeven is reached in <1 step.

**The memory win is immediate even on call #0**: 50 MB compile workspace vs 476 MB dense intermediate (~10× better) — and 0.04 MB on every call after. The 1.98 s wait is the only thing the compile flag costs you.

The `_compile=False` first call also pays 810 ms, presumably PyTorch's own one-time setup (lazy import of dynamo machinery for `mask_mod`). So the "true" extra cost of `_compile=True` on the first call is closer to 1.2 s, not 2 s.

### 4. Recompile behavior on shape change (`--mode compile_recompile`)

We sweep all 7 TTT shapes back-to-back, twice. If `_compile=True` recompiled per shape we'd see 7 expensive calls each pass; if it has a kernel cache we should see only the first few.

| pass | `lck` | `KV_LEN` | time ms | peak MB |
|--:|--:|--:|--:|--:|
| 1 | 0 | 2048 | **2676** | 0.01 |
| 1 | 1 | 4096 | **1067** | 50.01 |
| 1 | 2 | 6144 | 2.13 | 0.02 |
| 1 | 3 | 8192 | 1.98 | 0.03 |
| 1 | 4 | 10240 | 2.10 | 0.04 |
| 1 | 5 | 12288 | 2.13 | 0.04 |
| 1 | 6 | 14336 | 2.14 | 0.05 |
| 2 | 0..6 | (all) | **2.1–2.3** | 0.01–0.05 |

**Only the first two unique shapes pay the compile tax** (~3.7 s combined). From `lck=2` onward the kernel handles all `KV_LEN` values via dynamic-shape inference — no recompile. Pass 2 confirms steady-state: every shape hits ~2.1 ms.

This is much better than expected. The earlier worry about `mask_mod` closures triggering `dynamo.config.recompile_limit` is unfounded for this code: PyTorch generalizes the kernel across `KV_LEN` after a couple of warmups.

### 5. Stress test: dynamic shape variation (`--mode compile_chaos`)

The earlier recompile test was a "friendly" monotonic sweep (lck=0,1,...,6). To see what `_compile=True` does under genuinely dynamic shape changes we ran three sub-tests:

#### A. Shuffled `lck` order, 3 passes

Same 7 shapes as before but in a different random order each pass.

| pass | first 7 calls (ms) | total ms | notes |
|--:|---|--:|---|
| 1 | **18954, 1036**, 2.4, 2.2, 2.2, 2.1, 2.1 | 20001 | first 2 unique shapes pay compile |
| 2 | all 2.1–2.2 | 15.0 | every shape cached |
| 3 | all 2.1–2.2 | 14.9 | every shape cached |

Pass 1 is 1339× slower than pass 3 — entirely due to the first two cold compiles (~20 s combined). After that, the kernel generalizes across the remaining 5 shapes for free.

#### B. 30 random `(B, Q_LEN, KV_LEN)` tuples (B in 1..8, Q in 256..4096, multiplier in 1..7)

This is the worst case for shape stability — all three dimensions vary independently per call.

| metric | value |
|---|---|
| median per-call time | **2.34 ms** |
| p90 per-call time | 4.64 ms |
| max per-call time | 4730 ms (idx 1, second cold compile) |
| slow calls (> 50 ms) | **2 of 30** |

Only the first 2 distinct shapes paid a compile tax (1.2 s and 4.7 s). The remaining 28 calls — all with novel `(B, Q_LEN, KV_LEN)` combinations — hit 2–5 ms each. **`_compile=True` generalizes the kernel across `B`, `Q_LEN`, and `KV_LEN` simultaneously.**

#### C. Cycling `BLOCK_SIZE` on a fixed shape

| call | BLOCK_SIZE | time ms | notes |
|--:|--:|--:|---|
| 0 | 64 | 4255.8 | cold compile for BLOCK_SIZE=64 |
| 1 | 128 | 2.5 | (was already cached from earlier sub-tests) |
| 2 | 256 | 2.3 | cold but kernel reused |
| 3..8 | 64/128/256 cycled | 2.2–2.3 | all cached |

Only one new compile event for the entirely new `BLOCK_SIZE=64` configuration. After three values are seen, cycling among them is free.

#### Bottom line on dynamic shapes

Across all three sub-tests we ran roughly **50 distinct shape signatures** in one process. Total cold-compile time: ~30 s. Number of slow calls: **3**. Steady-state per-call time: ~2.3 ms regardless of shape.

`dynamo.config.recompile_limit=128` (set at the top of `flex_attention.py`) was never approached — the compiled kernel handles dynamic dims natively, so we don't generate a separate kernel per shape.

**Implication for production training:** even if the rollout produces variable-length sequences and microbatch composition causes `(B, Q_LEN, KV_LEN)` to shift call-to-call, you only pay for compile twice at the start of training. There is no runaway recompile risk.

### 6. Holistic memory picture: B × lck × `_compile` (`--mode holistic`)

The earlier sweeps varied one dimension at a time. Here we cross the full grid and compute aggregate cost across an entire TTT loop and a representative training step.

#### Per-call peak (MB)

`_compile=False`:
| B \ lck | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 68 | 136 | 204 | 272 | 340 | 408 | **476** |
| 2 | 80 | 160 | 240 | 320 | 400 | 480 | 560 |
| 4 | 160 | 320 | 480 | 640 | 800 | 960 | 1120 |
| 8 | 320 | 640 | 960 | 1280 | 1600 | 1920 | **2240** |

`_compile=True`:
| B \ lck | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.01 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
| 2 | 0.03 | 0.05 | 0.07 | 0.09 | 0.11 | 0.14 | 0.16 |
| 4 | 0.05 | 0.09 | 0.13 | 0.18 | 0.22 | 0.27 | 0.31 |
| 8 | 0.09 | 0.18 | 0.27 | 0.35 | 0.44 | 0.53 | 0.62 |

Savings ratios stay in the **3500–5900×** band across every cell of the 4×7 grid. The win does not degrade with batch size — if anything, the absolute MB freed scales with B, so larger microbatch configs benefit more.

#### Full TTT loop peak (one microbatch, all 7 calls)

| B | `_cmp=F` peak MB | `_cmp=T` peak MB | sum traffic F | sum traffic T | **headroom freed MB** |
|--:|--:|--:|--:|--:|--:|
| 1 | 476.2 | 0.05 | 1905 | 0.33 | **476** |
| 2 | 560.0 | 0.10 | 2240 | 0.64 | **560** |
| 4 | 1120.1 | 0.20 | 4481 | 1.25 | **1120** |
| 8 | 2240.1 | 0.39 | 8961 | 2.47 | **2240** |

**Headroom freed** is the peak GPU memory the fix returns to the rest of the training step (activations, KV cache, optimizer state, gradient buffers). At microbatch B=8 we recover **2.24 GB per drafter layer per microbatch**.

**Sum traffic** is the cumulative allocate/free volume across the TTT loop. At B=8 we eliminate ~9 GB of memory churn per microbatch — easier on the allocator and reduces fragmentation pressure.

#### Per-training-step extrapolation

Multiplying by representative drafter-layer counts:

| layers | B | `_cmp=F` per-step peak MB | `_cmp=T` per-step peak MB | savings |
|--:|--:|--:|--:|--:|
| 1 | 1 | 476 | 0.08 | 476 MB |
| 1 | 8 | 2240 | 0.62 | 2.24 GB |
| 4 | 1 | 1905 | 0.32 | 1.9 GB |
| 4 | 8 | **8961** | 2.46 | **8.96 GB** |

At the high end (4 drafter layers, microbatch 8), the fix returns ~9 GB of GPU headroom per step. On 80 GB H100s this can be the difference between OOM and a stable run, especially with the FSDP2 sharded actor + verifier hidden states already on-device.

#### Two takeaways

1. **The savings scale with B linearly.** No batch size makes the fix less attractive. The raw MB freed at B=8 (2.24 GB per layer) is large enough to materially shift what configurations fit on H100.
2. **The compiled path is essentially free in steady state.** 0.39 MB peak at B=8 lck=6 is below typical PyTorch allocator block granularity — effectively zero cost.

### Summary table — when to enable `_compile`

| dimension | result |
|---|---|
| Q_LEN range tested | 64 → 4096; **win at every size** (60× → 6143× memory) |
| KV_LEN range tested | 448 → 28672; win grows mildly with KV_LEN |
| Time speedup (steady state) | ~2.4–4× across the board |
| First-call cost | ~2 s wall time, 50 MB peak (vs 476 MB without) |
| Break-even (time only) | ~534 same-shape calls — reached in well under one training step |
| Memory break-even | call #0 already wins (10×), every subsequent call wins ~6000× |
| Recompile churn risk | observed only for the first 2 unique shapes; subsequent shapes free |

**Decision:** the `_compile=True` flag is unambiguously better for the EAGLE3 use case. There is no shape range or call pattern in this codebase where it loses. The earlier hedged recommendation can be strengthened: apply it everywhere `create_block_mask` is called from EAGLE3 paths.

---

## References

- [PyTorch FlexAttention blog — "How can we compute BlockMask quicker?"](https://pytorch.org/blog/flexattention/#q-how-can-we-compute-blockmask-quicker)
- [pytorch/pytorch#160018](https://github.com/pytorch/pytorch/issues/160018) — referenced in the existing `q_len <= 128` workaround.
- `recipe/drafter_cotraining/eagle3/ops/flex_attention.py` — wrapper definitions.
- `recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py` — call sites at lines 1017 and 1434.
- `recipe/drafter_cotraining/scripts/bench_eagle3_block_mask.py` — bench harness.
- `claude_docs/drafter-design.md` — TTT loop structure (`length=7`, KV cache concat).
