---
title: FSDP Sharding in Speculators
date: 2026-04-26
tags:
  - speculators
  - fsdp
  - distributed-training
  - pytorch
aliases:
  - Speculators FSDP
  - apply_fully_sharded
---

# FSDP Sharding in Speculators

How [[Eagle3 Implementation (speculators repo)|the speculators repo]] applies PyTorch FSDP2 (`fully_shard`) to draft models during training.

> [!info] TL;DR
> Speculators uses **FSDP2** (the per-parameter `fully_shard` API). Each decoder layer is wrapped as its own FSDP unit, plus a root wrap on the model. Mixed precision: bf16 compute, fp32 gradient reduce. Pretrained weights are captured on rank 0 *before* sharding, then broadcast to all ranks *after* sharding.

## File Map

| File | Role |
|---|---|
| `src/speculators/train/utils.py:107-124` | `apply_fully_sharded` — the wrapping function |
| `src/speculators/train/trainer.py:103-141` | `Trainer.setup_model` — orchestrates init/load/shard/broadcast |
| `src/speculators/train/trainer.py:143-149` | `Trainer.setup_optimizer` — AdamW over sharded `DTensor`s |
| `src/speculators/train/checkpointer.py` | `DistributedCheckpointer` — DCP-based sharded I/O |

## The Wrapping Recipe

```python
# utils.py:107-124
def apply_fully_sharded(model: torch.nn.Module):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,   # compute in bf16
        reduce_dtype=torch.float32,   # reduce grads in fp32
    )

    for layer in model.layers:        # per-block FSDP unit
        fully_shard(layer, mp_policy=mp_policy)

    fully_shard(model)                # root unit
```

> [!tip] Why per-layer wrap?
> Each decoder block becomes its own FSDP unit, so only **one block's worth of full params** is materialized in memory at any moment during forward/backward. Wrapping only the root would defeat the memory benefit — the AllGather would rebuild the entire model before the first matmul.

## Mixed Precision Policy

```
                MixedPrecisionPolicy
                ┌─────────────────────────┐
   stored       │ params: bf16            │  ← see "gotcha" below
                │       ↓                 │
   compute      │ params used: bf16       │  ← matmuls/attn run here
                │       ↓                 │
   grads        │ grads: bf16             │
                │       ↓ cast on reduce  │
   communicate  │ ReduceScatter: fp32     │  ← prevents grad-noise loss
                └─────────────────────────┘
```

`reduce_dtype=fp32` decouples **compute dtype** from **communication dtype**. Averaging gradients in bf16 across many ranks loses precision; FSDP2 lets you upcast just for the collective.

> [!warning] The dtype gotcha
> `trainer.py:107` calls `self.model.to(self.config.hidden_states_dtype)` (bf16) **before** `apply_fully_sharded`. So the master parameters under FSDP are already bf16, *not* fp32. This is bf16-master / bf16-compute / fp32-reduce — not the classic fp32-master setup. Check this against any precision-sensitive ablations.

## The Init/Load/Broadcast Dance

The trickiest part. Pretrained weights live on rank 0 (loaded from HF or a base model). Each rank needs to end up with the **correct shard** of those weights — without ever materializing N full copies.

```mermaid
sequenceDiagram
    participant R0 as rank 0
    participant Rk as rank ≠ 0

    Note over R0,Rk: Step 1 (trainer.py:121-123)
    R0->>R0: full_state_dict = model.state_dict()
    Rk->>Rk: full_state_dict = {}

    Note over R0,Rk: Step 2 (trainer.py:125)
    R0->>R0: apply_fully_sharded(model)
    Rk->>Rk: apply_fully_sharded(model)

    Note over R0,Rk: Step 3 (trainer.py:131-139)
    R0->>Rk: broadcast full_state_dict
    Rk->>Rk: slice out local shard, write into DTensor
    R0->>R0: del full_state_dict; barrier()
```

**Why the dance?** If every rank loaded the full pretrained weights, peak memory would briefly hit `N × full_model_size`. By keeping the snapshot on rank 0 only and broadcasting **after** sharding, peak stays at `1 × full_model + (N-1) × shard`.

```python
# trainer.py:131-139
set_model_state_dict(
    self.model,
    full_state_dict,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,   # rank 0 sends, others receive
        strict=False,                # tolerate missing keys
    ),
)
```

### Resume Path

If `load_checkpoint` is true (`trainer.py:108-110, 127-128`), the broadcast dance is **skipped**. Instead, `DistributedCheckpointer.load_model_state_dict` uses `torch.distributed.checkpoint` (DCP) to read **already-sharded** tensors directly from disk — each rank reads only its own shard. Much faster than the broadcast path.

## Optimizer

```python
# trainer.py:145
self.opt = torch.optim.AdamW(self.model.named_parameters(), lr=...)
```

Looks identical to non-distributed code — that's the win of FSDP2. `model.parameters()` returns `DTensor`s (sharded views). AdamW's `m` and `v` momentum buffers are allocated to match the **shard's local shape**, so optimizer state is sharded automatically. No `ZeroRedundancyOptimizer`, no manual fiddling.

## Train Loop is Plain PyTorch

```python
# trainer.py:201-208
_draft_tokens, loss, metrics = self.model(**gpu_batch, ...)
self.opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
self.opt.step()
```

FSDP hooks installed by `fully_shard` automatically:
- AllGather a layer's full params right before its forward
- Free them after forward
- AllGather again before backward
- ReduceScatter gradients (in fp32) after backward

`clip_grad_norm_` Just Works on `DTensor`s — global norm is computed across shards via collective.

## Memory Picture

```
DDP (every GPU = full model):
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ [ALL θ] │  │ [ALL θ] │  │ [ALL θ] │  │ [ALL θ] │
└─────────┘  └─────────┘  └─────────┘  └─────────┘

FSDP2 in speculators:
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ [θ₀]    │  │ [θ₁]    │  │ [θ₂]    │  │ [θ₃]    │  ← persistent shards
└─────────┘  └─────────┘  └─────────┘  └─────────┘
       ↕ AllGather one block at a time, then free
```

Per rank, persistent memory ≈ `(params + grads + optimizer_state) / N + one_full_block`.

## Gotchas Specific to This Codebase

> [!danger] `model.layers` must exist
> `apply_fully_sharded` iterates `model.layers` directly (`utils.py:119`). The docstring notes this assumption and points to `SpeculatorModel.verify_training_compatible` (`trainer.py:105`). If a new speculator architecture uses `blocks` or `decoder.layers`, this loop silently shards nothing per-layer — only the root wrap fires, collapsing FSDP to one giant unit and **killing the memory benefit**.

> [!warning] `strict=False` on load
> `trainer.py:137` tolerates state-dict key mismatches. Intentional for speculator training (verifier vs draft layers live in different places), but a typo'd parameter name will silently skip loading. You won't notice until the loss curve looks wrong.

> [!warning] Order of `to(dtype)` and shard
> `trainer.py:107` casts to bf16 **before** `apply_fully_sharded` (`:125`). The DTensor shards end up bf16. For fp32 master weights, skip the cast and rely solely on `MixedPrecisionPolicy`.

> [!info] All ranks call `set_model_state_dict`
> `full_state_dict` is only populated on rank 0 (`trainer.py:122`), but every rank calls `set_model_state_dict` with their (possibly empty) dict. `broadcast_from_rank0=True` makes it a collective — rank 0 is source, others contribute nothing. Without distributed init, the dict would be empty everywhere and the model would silently keep its random init.

## FSDP1 vs FSDP2 Quick Refresher

| Aspect | FSDP1 (`FullyShardedDataParallel`) | FSDP2 (`fully_shard`) — used here |
|---|---|---|
| Wrapping | `model = FSDP(model)` returns wrapper | `fully_shard(model)` mutates in place |
| Param type | flat 1-D `FlatParameter` | per-param `DTensor` |
| Access | `model.module.layer.weight` | `model.layer.weight` (DTensor) |
| State dict | needs `FSDP.state_dict_type(...)` | works with `torch.distributed.checkpoint` natively |
| Mixed precision | `MixedPrecision(...)` (older API) | `MixedPrecisionPolicy(...)` |

## Related Notes

- [[Eagle3 Implementation (speculators repo)]]
- [[Eagle3 Training Explained]]
- [[Eagle3 Comparison — speculators vs TorchSpec]]
- [[Drafter Trainer Integration Plan (verl)]]
