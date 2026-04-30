---
title: FSDP2 Wrap — TorchSpec Cross-Reference
date: 2026-04-26
tags:
  - eagle3
  - drafter
  - verl
  - fsdp2
  - sharding
aliases:
  - FSDP2 wrap analysis
  - TorchSpec FSDP wrap comparison
  - Drafter FSDP2 sharding decision
---

# FSDP2 Wrap — TorchSpec Cross-Reference

> [!summary] TL;DR
> **TorchSpec does NOT separately wrap `lm_head`** (or `fc` / `norm` / `embed_tokens`). Their `apply_fsdp2(modules_to_shard=...)` filter is `isinstance(m, nn.Linear) and "midlayer" in name` — only the **7 Linear modules inside `midlayer`** are passed as `modules_to_shard`. Everything else is covered by the trailing `fully_shard(model)` root wrap. Our `FSDPDrafterEngine._build_fsdp_module` override has the same intent, but at coarser granularity: we wrap the **whole `LlamaDecoderLayer`** as one FSDP unit (1 sub-unit) instead of its 7 internal Linears (7 sub-units). Functionally equivalent for our 1-block draft; the trade is comm pattern (1 large all-gather vs 7 smaller).

---

## 1. The question

When migrating the drafter from FSDP1 → FSDP2, the load-bearing concern was: **does the compiled loss kernel see `lm_head.weight` as a plain Tensor or a non-gathered DTensor?**

If `lm_head` is its OWN FSDP sub-unit, then under FSDP2's default behavior:
- `lm_head.weight` becomes a sharded DTensor.
- The DTensor is only auto-gathered inside `lm_head.forward()` via the sub-unit's pre-forward hook.
- Our compiled kernel reads `lm_head.weight` as an **extracted tensor argument** (`F.linear(input, lm_head_weight)`), bypassing the module's forward → the hook never fires → the kernel sees a non-gathered DTensor → `RuntimeError: Tensor × DTensor not supported`.

If `lm_head` is in the **root** FSDP unit, PyTorch's `fully_shard` auto-detects "this is the root" and forces effective `reshard_after_forward=False` for it (see `verl/utils/fsdp_utils.py:734-766` `set_reshard_after_forward` docstring) — root params stay gathered through forward → backward, and the kernel just sees a plain Tensor.

So the question boils down to: **how does TorchSpec choose which modules become their own FSDP sub-units?**

---

## 2. TorchSpec's actual code (verbatim)

### `apply_fsdp2` definition

> [!example] `ref/TorchSpec/torchspec/training/fsdp.py:137-201`
> ```python
> def apply_fsdp2(
>     model,
>     mesh=None,
>     cpu_offload=False,
>     args=None,
>     modules_to_shard: Optional[List[nn.Module]] = None,
> ):
>     """Apply FSDP v2 or DDP to a model.
>
>     Args:
>         ...
>         modules_to_shard: Explicit list of sub-modules to individually shard
>             before sharding the root model.  When *None* the root model is
>             sharded as a single unit.
>     """
>     from torch.distributed._composable.replicate import replicate
>     from torch.distributed.fsdp import (
>         CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard,
>     )
>
>     strategy = getattr(args, "fsdp_strategy", "REPLICATE") if args else "REPLICATE"
>     strategy = strategy.upper()
>
>     if strategy == "REPLICATE":
>         logger.info("Using REPLICATE strategy (DDP-like, gradient all-reduce only)")
>         replicate(model, device_mesh=mesh)
>         return model
>     elif strategy != "FULL_SHARD":
>         raise ValueError(f"Unknown fsdp_strategy: {strategy}. Use 'FULL_SHARD' or 'REPLICATE'")
>
>     # ... mp_policy / offload_policy setup ...
>
>     fsdp_kwargs = {
>         "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
>         "offload_policy": offload_policy,
>         "mesh": mesh,
>     }
>
>     for module in modules_to_shard or []:
>         fully_shard(module, **fsdp_kwargs)
>
>     fully_shard(model, **fsdp_kwargs)
>
>     return model
> ```

Two important details from this:

1. **TorchSpec's default strategy is `REPLICATE`** (line 165) — DDP-style replication, no FSDP sharding at all. FSDP2 sharding only kicks in when `args.fsdp_strategy=FULL_SHARD` is explicitly set. So in the common configuration, `lm_head` (and every other param) is fully replicated across DP ranks.

2. **No `reshard_after_forward` argument.** TorchSpec passes identical `fsdp_kwargs` to both the sub-unit `fully_shard(module, ...)` calls AND the root `fully_shard(model, ...)`. PyTorch's auto-root-detection forces the root's effective `reshard_after_forward=False` regardless. (Sub-units default to `True`.)

### How TorchSpec calls `apply_fsdp2` for Eagle3

> [!example] `ref/TorchSpec/torchspec/training/eagle3_trainer.py:99-119`
> ```python
> full_state = eagle3_model.state_dict() if dist.get_rank() == 0 else {}
>
> midlayer_modules = [
>     m
>     for name, m in eagle3_model.named_modules()
>     if isinstance(m, torch.nn.Linear) and "midlayer" in name
> ]
> eagle3_model = apply_fsdp2(
>     eagle3_model,
>     mesh=self.dp_mesh,
>     cpu_offload=self.fsdp_cpu_offload,
>     args=self.args,
>     modules_to_shard=midlayer_modules,
> )
>
> eagle3_model = fsdp2_load_full_state_dict(
>     eagle3_model,
>     full_state,
>     self.dp_mesh,
>     cpu_offload=True if self.fsdp_cpu_offload else None,
> )
> ```

The filter `isinstance(m, torch.nn.Linear) and "midlayer" in name` is what determines membership in `modules_to_shard`.

### What `midlayer_modules` actually contains

For our 1-block draft, the Eagle3Model's structure is:

```
Eagle3Model
└── draft_model: LlamaForCausalLMEagle3
    ├── embed_tokens: nn.Embedding
    ├── fc:           nn.Linear        ← NOT in modules_to_shard ("midlayer" not in name)
    ├── midlayer:     LlamaDecoderLayer
    │   ├── self_attn: LlamaAttention
    │   │   ├── q_proj: nn.Linear      ← in modules_to_shard
    │   │   ├── k_proj: nn.Linear      ← in modules_to_shard
    │   │   ├── v_proj: nn.Linear      ← in modules_to_shard
    │   │   └── o_proj: nn.Linear      ← in modules_to_shard
    │   ├── mlp: LlamaMLP
    │   │   ├── gate_proj: nn.Linear   ← in modules_to_shard
    │   │   ├── up_proj:   nn.Linear   ← in modules_to_shard
    │   │   └── down_proj: nn.Linear   ← in modules_to_shard
    │   ├── input_layernorm:        LlamaRMSNorm  (NOT Linear → skipped)
    │   └── post_attention_layernorm: LlamaRMSNorm (NOT Linear → skipped)
    ├── norm:    LlamaRMSNorm        ← NOT in modules_to_shard (not Linear)
    └── lm_head: nn.Linear           ← NOT in modules_to_shard ("midlayer" not in name)
```

Verified with a live introspection (`recipe/drafter_cotraining/scripts/test_fsdp2_drafter_wrap.py` fixture):

```
--- TorchSpec selector: Linear AND "midlayer" in name ---
  draft_model.midlayer.self_attn.q_proj
  draft_model.midlayer.self_attn.k_proj
  draft_model.midlayer.self_attn.v_proj
  draft_model.midlayer.self_attn.o_proj
  draft_model.midlayer.mlp.gate_proj
  draft_model.midlayer.mlp.up_proj
  draft_model.midlayer.mlp.down_proj
Total: 7
--- Sibling Linears outside midlayer (root unit candidates) ---
  draft_model.fc
  draft_model.lm_head
```

**The model definition is identical** to TorchSpec's (we ported from there) — confirmed at `ref/TorchSpec/torchspec/models/draft/llama3_eagle.py:2117-2126`:
```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
...
self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)
...
self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
```

### Direct answer

> [!important] TorchSpec does NOT wrap `lm_head` as its own FSDP sub-unit.
> Their `modules_to_shard` is exactly the 7 Linears INSIDE `midlayer`. `lm_head`, `fc`, `norm`, and `embed_tokens` all fall under the trailing `fully_shard(model)` root wrap. This means in the FSDP unit topology, the root unit owns:
> - `lm_head` (the param the compiled loss kernel reads)
> - `norm` (the param `Eagle3Model.draft_model.get_lm_head_params()` returns alongside `lm_head_weight`)
> - `fc` (the projection that consumes 3 aux hidden states)
> - `embed_tokens` (frozen, copied from target)
>
> Plus PyTorch's auto-root-detection forces the root's effective `reshard_after_forward=False` → these four stay gathered across forward → backward, exactly what the compiled kernel needs.

---

## 3. What our override does (`drafter_engine.py:_build_fsdp_module`)

```python
def _build_fsdp_module(self, module):
    """Selective FSDP2 wrap: shard ONLY LlamaDecoderLayer; let the root unit
    (which holds lm_head, norm, fc, embed_tokens) stay gathered.
    """
    if self.engine_config.strategy != "fsdp2":
        return super()._build_fsdp_module(module)

    from torch.distributed.fsdp import (
        CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard,
    )
    from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

    param_dtype = getattr(torch, self.model_config.dtype, torch.bfloat16)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    offload_policy = (
        CPUOffloadPolicy(pin_memory=True)
        if self.engine_config.offload_policy
        else None
    )
    fsdp_kwargs = {"mesh": self.device_mesh, "mp_policy": mp_policy, "offload_policy": offload_policy}

    full_state = module.state_dict()

    sharded_count = 0
    for _name, sub in module.named_modules():
        if sub.__class__.__name__ == "LlamaDecoderLayer":
            fully_shard(sub, **fsdp_kwargs)              # ← coarser than TorchSpec
            sharded_count += 1

    fully_shard(module, **fsdp_kwargs)                   # ← root

    fsdp2_load_full_state_dict(module, full_state, self.device_mesh, offload_policy)
    return module
```

### Comparison vs TorchSpec

| Aspect | TorchSpec | Our override |
|---|---|---|
| **Strategy default** | `REPLICATE` (DDP-style, no sharding) — must opt into `FULL_SHARD` | Always FSDP2 sharding |
| **`modules_to_shard` selector** | `isinstance(m, nn.Linear) and "midlayer" in name` | `sub.__class__.__name__ == "LlamaDecoderLayer"` |
| **# of FSDP sub-units (1-block draft)** | 7 (one per Linear) | 1 (the whole decoder layer) |
| **Root wrap** | `fully_shard(model)` | `fully_shard(module)` |
| **`lm_head` in root unit** | ✅ | ✅ |
| **`norm` / `fc` / `embed_tokens` in root unit** | ✅ | ✅ |
| **`mp_policy.cast_forward_inputs`** | not set (defaults False) | `True` (matches verl's default) |
| **`reshard_after_forward` for sub-units** | default `True` | default `True` |
| **Root `reshard_after_forward`** | auto-`False` (PyTorch root-detect) | auto-`False` (same) |
| **Mixed precision** | `param_dtype=bf16, reduce_dtype=fp32` (configurable) | same |
| **State dict broadcast** | `fsdp2_load_full_state_dict(..., full_state, ...)` after wrap | same |

The intent is **identical** — both keep `lm_head` in the root unit so the compiled kernel works. The difference is purely the granularity inside the decoder layer.

---

## 4. The granularity trade

### Per-Linear sharding (TorchSpec)

- **7 FSDP sub-units** for our 1-block draft, each owning a single Linear's weight.
- Each sub-unit fires its own pre-forward all-gather → post-forward reshard pair.
- **More all-gather calls per forward** (7 vs 1).
- **Smaller per-call payload** (one Linear's params).
- **Finer overlap** of comm with compute is theoretically possible — while one Linear is computing, another's all-gather can start.

### Per-DecoderLayer sharding (ours)

- **1 FSDP sub-unit** (the whole `LlamaDecoderLayer`).
- One pre-forward all-gather brings in all 7 Linears + 2 RMSNorms.
- **Fewer all-gather calls** → less FSDP scheduler overhead per step.
- **Larger per-call payload** (entire layer).
- **Less overlap potential**, but for a 1-block draft there's nothing to overlap with anyway (root forward → midlayer forward → root forward; only one decoder layer in flight at a time).

### Verdict for our 1-block drafter

For ~140M trainable params on 2× H100 with TCP-only Mooncake, the comm cost is dominated by FSDP scheduler overhead, not bandwidth. Fewer all-gathers (mine) wins on a per-step basis. If the drafter ever scales to multi-block, per-Linear may pay off (overlap across blocks), but for now per-DecoderLayer is fine and simpler.

> [!note] Speculators uses yet another granularity
> `ref/speculators/.../utils.py` does `for layer in model.layers: fully_shard(layer); fully_shard(model)` — per-decoder-block (one FSDP unit per `LlamaDecoderLayer`). For a single-block draft, speculators' pattern collapses to ours. So our choice is also speculators-aligned.

---

## 5. Why we don't use verl's default `apply_fsdp2`

`verl/utils/fsdp_utils.py:534-561` provides verl's own `apply_fsdp2`. Its sub-unit selector is `_select_fsdp2_wrap_targets` (`fsdp_utils.py:510-531`):

```python
def _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, tuple(fsdp_transformer_layer_cls_to_wrap)):
            targets.append(module)
        elif _wrap_by_name and name.split(".")[-1] in _wrap_by_name:
            targets.append(module)
    return targets

# where:
_tie = getattr(model.config, "tie_word_embeddings", False)
_wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}
```

**The kicker:** verl wraps `embed_tokens` and `lm_head` as **their own FSDP units** when `tie_word_embeddings=False`. Qwen3-8B has `tie_word_embeddings=False` (verified earlier), so verl's default would put `lm_head` in its own sub-unit → DTensor → kernel error.

This is why we override. Verl's default is right for an actor-scale LM where `lm_head` is hundreds of MB and worth its own unit; for our tiny draft it's the wrong call.

---

## 6. Empirical verification

The standalone `recipe/drafter_cotraining/scripts/test_fsdp2_drafter_wrap.py` exercises the wrap on a tiny synthetic Eagle3Model (H=128, V=256, length=3) on 1 and 2 ranks. It asserts:

- Exactly **1 LlamaDecoderLayer** is sharded as a sub-unit (for our 1-block draft).
- `set_requires_gradient_sync(True/False)` is callable on the FSDP2 root.
- `lm_head.weight` and `midlayer.q_proj.weight` are reachable post-wrap (both are DTensors under FSDP2; the root-unit DTensor is auto-gathered before the kernel reads it).
- Forward + backward run **without `Tensor × DTensor` errors** through `compiled_forward_kl_loss`.
- 2-step backward (sync_off → sync_on) correctly populates `.grad`.

Result: **5/5 pass on world_size=1 and world_size=2** with PyTorch 2.10.

The 16-step Qwen3-8B pretrain smoke (loss 12.076 → 8.425 over 16 steps) confirms the wrap also works at production scale with the real model.

---

## 7. Open question — should we mirror TorchSpec exactly?

If you want to flip to TorchSpec's per-Linear granularity, the change is one block in `_build_fsdp_module`:

```python
# Replace this:
for _name, sub in module.named_modules():
    if sub.__class__.__name__ == "LlamaDecoderLayer":
        fully_shard(sub, **fsdp_kwargs)
        sharded_count += 1

# With this (mirrors TorchSpec eagle3_trainer.py:101-105):
for name, sub in module.named_modules():
    if isinstance(sub, torch.nn.Linear) and "midlayer" in name:
        fully_shard(sub, **fsdp_kwargs)
        sharded_count += 1
```

Identical end result on `lm_head`/`fc`/`norm`/`embed_tokens` placement (still in root). Only the FSDP topology inside `midlayer` differs.

**My recommendation: keep `LlamaDecoderLayer` granularity.** Reasons:
1. Fewer FSDP units → less scheduler overhead per step.
2. For 1-block draft, the per-Linear advantage (comm/compute overlap) doesn't apply.
3. Aligns with speculators (also wraps per-decoder-block).
4. Aligns with verl's own FSDP1 convention (which wraps at decoder-layer granularity).
5. If the draft ever scales to multi-block, per-block FSDP units still get the cross-block overlap; we can revisit per-Linear at that point.

If you'd prefer the TorchSpec mirror, it's a one-line edit and risk-free.

---

## Related

- [[Drafter Micro-Batching Concrete Plan]] §4 — Phase B FSDP2 selective wrap
- [[Drafter Micro-Batching Refactor Plan]] §1b — original analysis of why FSDP2 doesn't break the kernel
- [[FSDP Sharding in Speculators]] — speculators' per-block + root pattern
- [[Eagle3 Comparison — speculators vs TorchSpec]] — broader Eagle3 comparison
