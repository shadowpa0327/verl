---
title: Drafter Micro-Batching Concrete Plan
date: 2026-04-26
tags:
  - eagle3
  - drafter
  - verl
  - fsdp2
  - micro-batching
  - implementation
aliases:
  - Drafter Refactor Concrete Edits
  - Eagle3 Micro-Batch Implementation
---

# Drafter Micro-Batching Concrete Plan

> Companion to [[Drafter Micro-Batching Refactor Plan]]. That doc has the rationale + research; this doc is the actionable diff with verbatim code snippets, file:line anchors, and a phase-by-phase execution checklist.
>
> **Verification script:** `recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh` (Qwen3-8B pretrain — the actively developed path). Override:
> ```bash
> DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
>   ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
>   trainer.total_training_steps=16
> ```
> `DrafterPretrainWorker.update_drafter = ActorRolloutRefDrafterWorker.update_drafter` (fsdp_workers.py:590) — refactoring `update_drafter` covers both pretrain and cotraining paths.

---

## 0. Findings from code exploration that update the original plan

These are **deltas** vs the original `Drafter Micro-Batching Refactor Plan.md`. Apply on top of that doc's recommendations.

> [!info] Δ1 — FSDP2 *does* have a no-sync equivalent, and TorchSpec uses it
> Original plan §6 said: *"FSDP1's `model.no_sync()` ctx mgr doesn't exist on FSDP2, and there is no equivalent public method."* **This is wrong.** TorchSpec uses `model.set_requires_gradient_sync(is_last)` (`ref/TorchSpec/torchspec/training/trainer.py:321-346`) to suppress allreduce on all-but-last micro-batch. We **should** mirror this in Phase C — it's a free reduction in inter-rank traffic across N-1 of N micro-batches.

> [!info] Δ2 — TorchSpec's divisor pattern is `/ accumulation_steps`, not `mb_valid / total_valid_global`
> The original plan recommended the more accurate `mb_valid / total_valid_global` divisor (recovers exact single-batch mean semantics). TorchSpec uses simple `/ accumulation_steps` (`ref/TorchSpec/torchspec/training/eagle3_trainer.py:311-315`), which is approximate when per-micro-batch `N_valid` varies. **Keep the plan's recommendation** (exact mean) — verl's canonical `forward_backward_batch` (`transformer_impl.py:591-621`) does this.

> [!info] Δ3 — TorchSpec shards individual `Linear` modules inside `midlayer`, not the whole `LlamaDecoderLayer`
> `ref/TorchSpec/torchspec/training/eagle3_trainer.py:101-112`:
> ```python
> midlayer_modules = [m for name, m in eagle3_model.named_modules()
>                     if isinstance(m, torch.nn.Linear) and "midlayer" in name]
> ```
> They shard 4-5 individual Linears per FSDP unit. Original plan said shard `LlamaDecoderLayer` (one unit per decoder block). For our drafter (one decoder block), the difference is just FSDP-unit granularity — plan's `LlamaDecoderLayer` choice is simpler and works fine. **Keep plan's choice.**

> [!info] Δ4 — `train_mode` zeros grads on EXIT (not entry)
> `verl/workers/engine/fsdp/transformer_impl.py:864-868`:
> ```python
> def __exit__(self, exc_type, exc_value, traceback):
>     set_ulysses_sequence_parallel_group(self.prev_sp_group)
>     self.engine.optimizer_zero_grad()  # ← on exit
>     super().__exit__(...)
> ```
> Means: gradients implicitly persist between `.backward()` calls within one `train_mode` context. Phase C's `for mb in micro_batches: ... .backward()` works without manual zero between micro-batches. Just keep `optimizer_step` + `lr_scheduler_step` inside the same context.

> [!info] Δ5 — verl's `apply_fsdp2` already handles `tie_word_embeddings=True` correctly
> `_select_fsdp2_wrap_targets` (`fsdp_utils.py:510-531`) skips `embed_tokens`/`lm_head` wrap when `tie_word_embeddings=True`. Qwen3-4B is tied; Qwen3-8B (our active target) is **untied** → verl would wrap `lm_head` separately → DTensor failure when the compiled kernel reads `lm_head.weight`. **Override is needed for Qwen3-8B path.** Confirmed.

> [!info] Δ6 — `_drafter_train_step` already has the `train_mode` + forward + backward + optimizer_step shape
> Phase C's restructure is straightforward: extend the current `_drafter_train_step` (`fsdp_workers.py:329-368`) to loop over micro-batches inside the existing `train_mode` context. We don't need to invent a new context-management strategy.

> [!info] Δ7 — `Eagle3Collator` hardcodes `_BUCKET = 256` with no override
> `eagle3_collator.py:26`. To get uniform `T_pad` across micro-batches in one macro-step (avoiding `torch.compile` recompilation), we'll add an optional `bucket_size_override` param to `__call__`. Cleaner than wrapping the collator.

> [!info] Δ8 — verl's FSDP1 path with size-based wrap policy → one root FSDP unit for our tiny drafter
> Confirmed: `engine_config.wrap_policy` is unset; `get_fsdp_wrap_policy` defaults to size-based; our <1GB drafter doesn't trigger any wrap → single root unit. This is why FSDP1 + `use_orig_params=True` works today: `lm_head.weight` is a plain Tensor (orig-params semantics) inside the one flat unit.

---

## 1. Decision matrix — locked-in choices

| Decision | Choice | Rationale |
|---|---|---|
| Lazy vs precomputed kernel | **Delete LazyTarget; bf16 PrecomputedTarget** | Compile-graph stability under micro-batching (memory neutral in our V_full=V_draft regime — see [[LazyTarget vs Precomputed Memory Analysis]] §"Re-doing for our actual config") |
| Vocab pruning support | **Generalize, don't delete** | `compute_target_p_padded` accepts `t2d=None` for no-pruning. Both branches kept (pruning path is dead code today but cheap to keep — small switch in one function) |
| FSDP2 wrap | **Override `_build_fsdp_module` in `FSDPDrafterEngine`, shard only `LlamaDecoderLayer`** | Qwen3-8B has `tie_word_embeddings=False` → verl's default `apply_fsdp2` wraps `lm_head` separately → DTensor failure |
| Divisor in micro-batch loop | **`mb_valid / total_valid_global`** (exact mean) | Matches verl's canonical `forward_backward_batch` divisor pattern; exact reproduction of single-batch loss curve |
| Grad-sync suppression | **`model.set_requires_gradient_sync(is_last)`** | TorchSpec uses this; free traffic reduction. Apply on the FSDP2-wrapped root |
| Per-micro-batch fetch | **Paged Mooncake.get + immediate `remove_eagle3_tensors`** | Already half-implemented in `_fetch_drafter_batch_from_mooncake`; just split the outer key loop into micro-batches |
| `T_pad` jitter mitigation | **Pre-compute `T_pad_macro` once per macro-batch, pass to collator** | Avoids `torch.compile` recompilation across micro-batches in one accumulation step |
| Empty-mask handling | **Two layers (fetch-time metadata filter + in-kernel fallback)** | Mirrors TorchSpec exactly. In-kernel fallback (`eagle3_model.py:107-113`) stays as-is |
| Phase E rename | **First, separate commit** | Mechanical; gets file paths right for subsequent phases |
| Per-phase commits | **Yes** | Bisectable history if regression appears |

---

## 2. Phase E — File rename (mechanical, do first)

> [!todo] One commit, no behavior change.

`recipe/drafter_cotraining/fsdp_workers.py` extends `verl.workers.engine_workers.ActorRolloutRefWorker` (engine-agnostic pattern), not the legacy direct-FSDP pattern. Filename is misleading.

**Steps:**

```bash
# 1. Rename
git mv recipe/drafter_cotraining/fsdp_workers.py recipe/drafter_cotraining/engine_workers.py

# 2. Find all importers
git grep -l "drafter_cotraining\.fsdp_workers"
git grep -l "from recipe.drafter_cotraining.fsdp_workers"
git grep -l "from recipe.drafter_cotraining import fsdp_workers"
```

**Expected importer files** (verify with grep above):
- `recipe/drafter_cotraining/main_drafter_ct.py`
- `recipe/drafter_cotraining/draft_model_pretrain_trainer.py:55`
- `recipe/drafter_cotraining/ray_trainer.py`
- Any test files under `recipe/drafter_cotraining/scripts/`

**Edit each:** `from recipe.drafter_cotraining.fsdp_workers import X` → `from recipe.drafter_cotraining.engine_workers import X`.

**Doc updates** (string replace in each):
- `claude_docs/project-guide.md` — "Drafter worker / engine logic" row in the file table
- `claude_docs/drafter-design.md` — any references
- `claude_docs/migration-status.md` — any references

**Verify:**
```bash
git grep -l "fsdp_workers" recipe/drafter_cotraining/  # should be empty
PYTHONPATH=. python -c "from recipe.drafter_cotraining.engine_workers import ActorRolloutRefDrafterWorker, DrafterPretrainWorker"
```

**Commit:** `[drafter] rename recipe/drafter_cotraining/fsdp_workers.py → engine_workers.py`

---

## 3. Phase A — Kernel fix (drop LazyTarget, bf16 PrecomputedTarget, generalize for no-pruning)

### A.1 — `recipe/drafter_cotraining/eagle3/eagle3_model.py`

#### A.1.1 — Drop the `LazyTarget` import + dataclass

**Edit 1 — line 31** (delete the `compiled_forward_kl_loss_from_hs` import; keep `compiled_forward_kl_loss`):

```python
# OLD (line 30-32, approximately):
from recipe.drafter_cotraining.eagle3.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,  # ← delete this import line
)
# NEW:
from recipe.drafter_cotraining.eagle3.ops.loss import compiled_forward_kl_loss
```

**Edit 2 — lines 56-62** (delete `LazyTarget` dataclass entirely):

```python
# DELETE these lines verbatim:
@dataclass
class LazyTarget:
    """Deferred target computation to avoid materializing (B, T, V_full)."""

    hidden_states_padded: torch.Tensor  # (B, T + length, D)
    lm_head_weight: torch.Tensor  # (V_full, D)
```

#### A.1.2 — Generalize `PrecomputedTarget` semantics (no code change, doc only)

**Edit 3 — line 50 docstring** (clarify it now handles both pruned and non-pruned):

```python
@dataclass
class PrecomputedTarget:
    """Pre-computed target probabilities.

    target_p_padded: (B, T + length, V) where V is V_draft (with t2d pruning) or V_full (no pruning).
    position_mask: (B, T) — set only when vocab pruning is in effect (subset of loss_mask
        further filtered by 'verifier argmax ∈ V_draft'). None when no pruning.
    """
    target_p_padded: torch.Tensor
    position_mask: Optional[torch.Tensor] = None
```

#### A.1.3 — Drop the LazyTarget branch in `_calculate_loss`

**Edit 4 — lines 85-149** (collapse to single PrecomputedTarget path; keep empty-mask fallback):

```python
def _calculate_loss(
    self,
    hidden_states: torch.Tensor,
    target: PrecomputedTarget,                        # ← was: Union[PrecomputedTarget, LazyTarget]
    mask: torch.Tensor,
    idx: int,
    seq_length: int,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute forward-KL loss and accuracy for one TTT step.

    Pre-computed target probs (target.target_p_padded) sized either over V_draft
    (vocab pruning enabled) or V_full (no pruning) — kernel doesn't care.
    """
    valid_idx = mask.flatten().nonzero().squeeze(-1)
    if valid_idx.numel() == 0:
        # FSDP requires every trainable param to participate in gradient
        # all-reduce/reduce-scatter. Synthesize a zero-grad touching all params.
        total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
        zero = total * 0.0
        return zero, zero.detach()
    torch._dynamo.mark_dynamic(valid_idx, 0)
    hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    target_p_step = target.target_p_padded[:, idx : idx + seq_length, :]
    tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])
    args = (hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, norm_eps)
    if self.gradient_checkpointing and self.training:
        return torch_checkpoint(
            compiled_forward_kl_loss,
            *args,
            use_reentrant=False,
        )
    return compiled_forward_kl_loss(*args)
```

#### A.1.4 — Update `forward()` signature: `target: PrecomputedTarget`

**Edit 5 — line 155** (drop `Union[..., LazyTarget]`):

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target: PrecomputedTarget,            # ← was: Union[PrecomputedTarget, LazyTarget]
    loss_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
):
    ...
```

#### A.1.5 — Generalize `compute_target_p_padded` to accept `t2d=None`

**Edit 6 — lines 260-290** (signature change + skip pruning machinery when t2d is None + bf16 storage):

```python
@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    length: int,
    t2d: Optional[torch.Tensor] = None,       # ← now optional; None = no pruning
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    """Build target probabilities for the EAGLE forward-KL loss kernel.

    With pruning (t2d not None):
        - lm_head projects to V_draft
        - position_mask filters loss positions whose verifier argmax ∈ V_draft

    Without pruning (t2d is None):
        - lm_head projects to V_full
        - position_mask is None; loss_mask used directly downstream

    Returns target_p stored in bf16 (kernel auto-upcasts in `tp * log_p`).
    """
    target_lm_head_weight = target_lm_head_weight.detach()

    if t2d is not None:
        # Pruning path (matches existing behavior)
        pruned_weight = target_lm_head_weight[t2d]  # (V_draft, D)
        B, T, _D = target_hidden_states.shape
        loss_mask_bool = loss_mask.bool()
        valid_flat_idx = loss_mask_bool.reshape(-1).nonzero(as_tuple=True)[0]
        valid_hs = target_hidden_states.reshape(-1, _D)[valid_flat_idx]

        position_mask_flat = torch.zeros(B * T, device=target_hidden_states.device, dtype=torch.float)
        for i in range(0, valid_hs.shape[0], chunk_size):
            chunk_hs = valid_hs[i : i + chunk_size]
            chunk_argmax = F.linear(chunk_hs, target_lm_head_weight).argmax(-1)
            in_draft = t2d[chunk_argmax]
            position_mask_flat[valid_flat_idx[i : i + chunk_size]] = in_draft.float()
        position_mask = position_mask_flat.reshape(B, T)

        target_logits = F.linear(target_hidden_states, pruned_weight)
    else:
        # No-pruning path: project to full vocab, no position_mask
        target_logits = F.linear(target_hidden_states, target_lm_head_weight)
        position_mask = None

    # bf16 storage halves resident memory; kernel's `tp * log_p` (log_p in fp32)
    # auto-upcasts tp, so loss arithmetic stays fp32.
    target_p = F.softmax(target_logits.float(), dim=-1).to(torch.bfloat16)
    target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

    return PrecomputedTarget(target_p_padded, position_mask)
```

**Note on argument order:** moved `t2d` after `loss_mask`/`length` to make it optional. **All callers must update** (Edit 8 below).

#### A.1.6 — Delete `compute_lazy_target_padded` entirely

**Edit 7 — lines 293-306** (delete the whole function; nothing references it after Edit 8):

```python
# DELETE lines 293-306:
def compute_lazy_target_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
) -> LazyTarget:
    """Build a LazyTarget that defers softmax to the forward loop."""
    return LazyTarget(
        hidden_states_padded=F.pad(target_hidden_states, (0, 0, 0, length), value=0.0),
        lm_head_weight=target_lm_head_weight.detach(),
    )
```

### A.2 — `recipe/drafter_cotraining/eagle3/ops/loss.py`

**Edit 8 — lines 68-108** (delete `compiled_forward_kl_loss_from_hs` entirely):

```python
# DELETE the entire `compiled_forward_kl_loss_from_hs` function (lines 68-108).
# Keep only `compiled_forward_kl_loss` (lines 25-65).
```

### A.3 — `recipe/drafter_cotraining/drafter_engine.py`

**Edit 9 — `prepare_model_inputs` (lines 422-475)**: switch from `compute_lazy_target_padded` to generalized `compute_target_p_padded`:

```python
def prepare_model_inputs(self, micro_batch: TensorDict):
    """Prepare inputs for Eagle3Model.forward().

    Builds PrecomputedTarget (bf16-stored target_p) outside the compile graph.
    Handles both pruning and no-pruning configs uniformly.
    """
    from recipe.drafter_cotraining.eagle3.eagle3_model import compute_target_p_padded, padding

    input_ids = micro_batch["input_ids"]
    hidden_states = micro_batch["hidden_states"]
    attention_mask = micro_batch["attention_mask"]
    loss_mask = micro_batch["loss_mask"]
    last_hidden_states = micro_batch["last_hidden_states"]

    # Left-shift to align verifier emission positions (unchanged from current).
    input_ids = padding(input_ids, left=False)
    last_hidden_states = padding(last_hidden_states, left=False)

    # Apply verifier_norm to pre-norm last_hidden_states from vLLM
    if self._verifier_norm is not None:
        with torch.no_grad():
            last_hidden_states = self._verifier_norm(last_hidden_states)

    eagle3 = self.module.module if hasattr(self.module, "module") else self.module

    # No vocab pruning today — t2d=None falls into the no-pruning branch.
    # If pruning is added later, surface t2d via _t2d_index attribute or similar.
    t2d = getattr(self, "_t2d_index", None)

    target = compute_target_p_padded(
        target_hidden_states=last_hidden_states,
        target_lm_head_weight=self._target_lm_head_weight,
        loss_mask=loss_mask,
        length=eagle3.length,
        t2d=t2d,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": target,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
    }
```

**Note:** `_target_lm_head_weight` plumbing stays — the no-pruning path still needs the full lm_head weight for `target_logits = F.linear(...)`. The original plan suggested removing it; this was wrong (it's needed for the no-pruning precomputed path).

### A.4 — Tests

**Edit 10 — `recipe/drafter_cotraining/tests/test_eagle3_loss.py`**:

- **Delete** `TestLazyVsPrecomputedTarget` class (lines 214-296) — no LazyTarget to compare against.
- **Delete** `TestValidIdxSubsetting`'s `_check_forward_kl_from_hs` half (lines ~372-462, dynamically generated `*_lazy` test methods).
- **Add** a no-pruning test in `TestComputeTargetPPadded`:
  ```python
  def test_no_pruning_full_vocab(self):
      """When t2d=None, target_p shape is (B, T+length, V_full) and position_mask is None."""
      target_hs = torch.randn(2, 16, 64)
      lm_head = torch.randn(1000, 64)
      loss_mask = torch.ones(2, 16, dtype=torch.long)
      length = 7

      out = compute_target_p_padded(
          target_hs, lm_head, loss_mask, length, t2d=None,
      )
      assert out.target_p_padded.shape == (2, 16 + length, 1000)
      assert out.position_mask is None
      assert out.target_p_padded.dtype == torch.bfloat16
  ```
- **Update** `test_pruning_shapes_and_position_mask` to pass `t2d=t2d_tensor` as a kwarg (post-signature-change).

### A.5 — Verify Phase A

```bash
# Unit tests
PYTHONPATH=. pytest recipe/drafter_cotraining/tests/test_eagle3_loss.py -v

# Smoke (16 steps; capture loss curve to compare against post-Phase-B/C)
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16 \
  2>&1 | tee /tmp/phase_a_smoke.log

# Extract loss curve
grep "loss_weighted" /tmp/phase_a_smoke.log
```

**Commit:** `[drafter] phase A: drop LazyTarget; bf16 PrecomputedTarget supports no-pruning`

---

## 4. Phase B — FSDP2 migration

### B.1 — Add `_build_fsdp_module` override on `FSDPDrafterEngine`

**Edit 11 — `recipe/drafter_cotraining/drafter_engine.py`** (add new method after `_build_module`, around line 254):

```python
def _build_fsdp_module(self, module):
    """Selective FSDP2 wrap: shard ONLY LlamaDecoderLayer; let the root unit
    (which holds lm_head, norm, fc, embed_tokens) stay gathered.

    Why we override the parent's apply_fsdp2:
        verl's _select_fsdp2_wrap_targets wraps embed_tokens + lm_head as
        their own FSDP units when tie_word_embeddings=False (Qwen3-8B). The
        Eagle3 loss kernel reads `lm_head.weight` as an extracted tensor,
        not via the lm_head module's forward — so its sub-unit's pre-forward
        hook never fires and the kernel sees a non-gathered DTensor.
        Mirroring TorchSpec's selective wrap (only midlayer Linears) keeps
        lm_head.weight in the root unit, which auto-stays-gathered through
        backward via PyTorch's root-detection logic.
    """
    if self.engine_config.strategy != "fsdp2":
        # Defer to parent for FSDP1 (legacy path stays available).
        return super()._build_fsdp_module(module)

    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
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
    fsdp_kwargs = {
        "mesh": self.device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
    }

    # Capture full state PRE-WRAP — fsdp2_load_full_state_dict broadcasts
    # from rank 0 onto all ranks' sharded DTensors after wrap.
    full_state = module.state_dict()

    # Shard only the LlamaDecoderLayer (midlayer), per TorchSpec.
    # Sub-units default to reshard_after_forward=True.
    sharded_count = 0
    for _name, sub in module.named_modules():
        if sub.__class__.__name__ == "LlamaDecoderLayer":
            fully_shard(sub, **fsdp_kwargs)
            sharded_count += 1
    logger.info("FSDPDrafterEngine: sharded %d LlamaDecoderLayer sub-units", sharded_count)

    # Wrap root. Auto-detected as root → effective reshard_after_forward=False
    # → root params (lm_head, norm, fc, embed_tokens) stay gathered through bwd.
    fully_shard(module, **fsdp_kwargs)

    # Broadcast rank-0 full state to all ranks' sharded DTensors.
    fsdp2_load_full_state_dict(module, full_state, self.device_mesh, offload_policy)
    return module
```

**Edit 12 — `_build_module`** (drafter_engine.py:242, 245): keep `_no_split_modules` + `module.config` lines as-is. They were FSDP2-default-wrap hints; our override doesn't read them but they're harmless and document intent. The plan said "delete" but doing so prevents fallback to verl's default `apply_fsdp2` if we ever swap strategies back.

### B.2 — Config flips

**Edit 13 — `recipe/drafter_cotraining/config/draft_model_pretrain_trainer.yaml` (lines ~256-275)**:

```yaml
engine_config:
  strategy: fsdp2                   # was: fsdp
  # use_orig_params is FSDP1-only — drop it. FSDP2 always behaves orig-params-like.
  fsdp_size: -1
  param_offload: False
  optimizer_offload: False
  use_remove_padding: False
```

**Edit 14 — `recipe/drafter_cotraining/config/drafter_ct_trainer.yaml` (lines ~170-186)**: same change.

### B.3 — Verify Phase B

> [!warning] Checkpoint break
> Existing FSDP1 sharded checkpoints (FlatParameter layout) are not loadable by FSDP2 (DTensor layout). Smoke runs do not require resume — start fresh.

```bash
# Same smoke command — single-batch path, no micro-batching yet.
# Loss curve MUST match Phase A output within ~1e-3.
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16 \
  2>&1 | tee /tmp/phase_b_smoke.log

# Compare
diff <(grep "loss_weighted" /tmp/phase_a_smoke.log) <(grep "loss_weighted" /tmp/phase_b_smoke.log)
```

If divergence: bisect by reverting Phase B alone (revert engine override + yaml flip; kernel changes from Phase A stay).

> [!info] Defensive fallback (only apply if a real failure surfaces)
> If `compiled_forward_kl_loss` errors with "Tensor × DTensor" under our override, add to `Eagle3Model._calculate_loss` (just before the `args = (...)` line):
> ```python
> from torch.distributed.tensor import DTensor
> if isinstance(lm_head_weight, DTensor):
>     lm_head_weight = lm_head_weight.full_tensor()
>     norm_weight = norm_weight.full_tensor() if isinstance(norm_weight, DTensor) else norm_weight
> ```
> Don't add preemptively — TorchSpec runs this exact kernel on FSDP2 successfully.

**Commit:** `[drafter] phase B: FSDP2 selective wrap (LlamaDecoderLayer only)`

---

## 5. Phase C — Micro-batching

### C.1 — `recipe/drafter_cotraining/eagle3_collator.py` — add bucket override

**Edit 15** (drafter_cotraining/eagle3_collator.py):

```python
class Eagle3Collator:
    """Pad variable-seq-len Eagle3 samples into a rectangular batch."""

    def __call__(
        self,
        features: List[Dict[str, Any]],
        bucket_size_override: Optional[int] = None,    # ← NEW
    ) -> Dict[str, torch.Tensor]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        if bucket_size_override is not None:
            # Caller passes T_pad_macro precomputed across all micro-batches in
            # a macro-step → identical T_pad across micro-batches → no compile recompile.
            max_length = max(max_length, bucket_size_override)
        max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET

        # ... rest unchanged
```

### C.2 — `recipe/drafter_cotraining/engine_workers.py` — restructure `update_drafter`

> [!info] After Phase E, this file is `engine_workers.py`. If skipping Phase E, paths read `fsdp_workers.py` instead.

**Edit 16 — replace the current `update_drafter` body** (lines 169-215). New shape:

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
def update_drafter(self, data: DataProto):
    """Per-rank drafter update with paged Mooncake fetch + micro-batch accumulation.

    The macro-step:
      1. Drop samples whose loss-mask would be empty (metadata-only filter).
      2. Pre-compute total_valid_global across DP for an exact mean divisor.
      3. Pre-compute T_pad_macro = ceil(max(seq_len), 256) across all micro-batches
         so torch.compile doesn't recompile per micro-batch.
      4. Engine train_mode: for each micro-batch, fetch Mooncake → forward+TTT
         → weighted backward (with set_requires_gradient_sync(is_last) on FSDP2).
      5. optimizer_step + lr_scheduler_step.
      6. Aggregate metrics across micro-batches + all-reduce.
    """
    if len(data) == 0:
        return DataProto(non_tensor_batch={})

    mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
    if len(mooncake_keys) == 0:
        return DataProto(non_tensor_batch={})

    import math
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Smoke-only fallback (no drafter engine): stays as today
    if self.drafter is None:
        batch = self._fetch_drafter_batch_from_mooncake(data, rank)
        if batch is None:
            return DataProto(non_tensor_batch=data.non_tensor_batch)
        self._log_drafter_batch_shapes(batch, rank)
        return DataProto(non_tensor_batch=data.non_tensor_batch)

    # Step 1: metadata-only empty-mask filter (mirrors TorchSpec data_fetcher.py:177)
    response_lens = data.non_tensor_batch.get("response_lens", [])
    keep_indices = [i for i, r in enumerate(response_lens) if int(r) - 1 > 0]
    if len(keep_indices) < len(response_lens):
        n_dropped = len(response_lens) - len(keep_indices)
        logger.warning(
            "[drafter rank=%d] dropping %d samples with zero loss-mask positions",
            rank, n_dropped,
        )
        # Eagerly remove dropped Mooncake keys so producer can reuse buffers.
        store = self._get_mooncake_store(data.meta_info.get("mooncake_cfg", {}), rank)
        if store is not None:
            for i in range(len(response_lens)):
                if i not in keep_indices:
                    try:
                        store.remove_eagle3_tensors(
                            key=str(mooncake_keys[i]), has_last_hidden_states=True
                        )
                    except Exception as exc:
                        logger.debug("Mooncake remove for dropped sample %d failed: %s", i, exc)
        data = self._select_data_indices(data, keep_indices)
        mooncake_keys = data.non_tensor_batch["mooncake_keys"]
        response_lens = data.non_tensor_batch["response_lens"]

    # Step 2: preflight total_valid_global from metadata (no fetch needed)
    local_total_valid = sum(max(0, int(r) - 1) for r in response_lens)
    total_valid_global = self._allreduce_sum_int(local_total_valid)
    if total_valid_global == 0:
        logger.warning("[drafter rank=%d] total_valid_global=0; skipping macro-step", rank)
        return DataProto(non_tensor_batch=data.non_tensor_batch)

    # Step 3: T_pad_macro across micro-batches
    seq_lens = data.non_tensor_batch.get("seq_lens", None)
    if seq_lens is None:
        # Fallback: prompt_lens + response_lens
        plens = data.non_tensor_batch.get("prompt_lens", [0] * len(mooncake_keys))
        seq_lens = [int(p) + int(r) for p, r in zip(plens, response_lens)]
    t_pad_macro = max(int(s) for s in seq_lens)

    # Step 4: micro-batch loop
    micro_size = int(self.config.drafter.engine_config.get("micro_batch_size_per_gpu", 1))
    accum_steps = math.ceil(len(mooncake_keys) / micro_size)
    metrics = self._drafter_train_step_micro(
        data,
        rank=rank,
        micro_size=micro_size,
        accum_steps=accum_steps,
        total_valid_global=total_valid_global,
        t_pad_macro=t_pad_macro,
    )

    if not getattr(self, "_drafter_shapes_logged", False):
        # One-time shape print on first macro-step (smoke continuity).
        self._drafter_shapes_logged = True

    return DataProto(
        non_tensor_batch=data.non_tensor_batch,
        meta_info={"train_metrics": metrics},
    )
```

### C.3 — New helpers on `ActorRolloutRefDrafterWorker`

**Edit 17 — add the following methods** (in the same class as `update_drafter`):

```python
def _allreduce_sum_int(self, value: int) -> int:
    """All-reduce SUM of a Python int across the drafter DP group."""
    import torch.distributed as dist
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=torch.cuda.current_device(), dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _select_data_indices(self, data: DataProto, indices: List[int]) -> DataProto:
    """Filter a DataProto's non_tensor_batch (and tensor_batch if present)
    to the given indices. Used for the metadata-time empty-mask filter."""
    new_nt = {}
    for k, v in data.non_tensor_batch.items():
        if isinstance(v, (list, np.ndarray)):
            new_nt[k] = type(v)([v[i] for i in indices]) if isinstance(v, list) else v[indices]
        else:
            new_nt[k] = v
    new_tb = None
    if data.batch is not None:
        new_tb = data.batch[indices]
    return DataProto(batch=new_tb, non_tensor_batch=new_nt, meta_info=dict(data.meta_info))


def _iter_micro_batch_keys(self, data: DataProto, micro_size: int):
    """Yield (mb_idx, sub_data) DataProto slices over Mooncake keys."""
    n = len(data.non_tensor_batch["mooncake_keys"])
    for mb_idx, start in enumerate(range(0, n, micro_size)):
        end = min(start + micro_size, n)
        sub = self._select_data_indices(data, list(range(start, end)))
        yield mb_idx, sub


def _drafter_train_step_micro(
    self,
    data: DataProto,
    rank: int,
    micro_size: int,
    accum_steps: int,
    total_valid_global: int,
    t_pad_macro: int,
) -> dict:
    """Outer loop: train_mode + per-micro-batch fetch+forward+backward + optimizer step."""
    engine = self.drafter.engine
    device = torch.device("cuda", torch.cuda.current_device())

    # FSDP2 grad-sync suppression on all-but-last micro-batch (mirrors TorchSpec
    # trainer.py:321-346). On FSDP1 (legacy) this no-op's gracefully.
    fsdp_root = engine.module
    set_grad_sync = getattr(fsdp_root, "set_requires_gradient_sync", None)

    accum_metrics = []
    with engine.train_mode():
        for mb_idx, mb_data in self._iter_micro_batch_keys(data, micro_size):
            is_last = mb_idx == accum_steps - 1
            if set_grad_sync is not None:
                set_grad_sync(is_last)

            # Page-fetch this micro-batch's tensors from Mooncake.
            mb_batch = self._fetch_drafter_batch_from_mooncake(
                mb_data, rank, t_pad_override=t_pad_macro,
            )
            if mb_batch is None:
                continue

            mb_metrics = self._drafter_micro_step(
                mb_batch,
                rank=rank,
                total_valid_global=total_valid_global,
                device=device,
            )
            accum_metrics.append(mb_metrics)

        # Re-enable sync before optimizer step (defensive; optimizer_step doesn't
        # trigger comms but make the engine state predictable).
        if set_grad_sync is not None:
            set_grad_sync(True)

        grad_norm = engine.optimizer_step()
        lr = engine.lr_scheduler_step()

    return self._aggregate_micro_metrics(accum_metrics, grad_norm, lr, rank)


def _drafter_micro_step(
    self, mb_batch, rank: int, total_valid_global: int, device,
) -> dict:
    """Per-micro-batch: prepare → forward → weighted backward → free."""
    engine = self.drafter.engine
    batch_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mb_batch.items()
    }
    prepared = engine.prepare_model_inputs(batch_dev)
    plosses, _, acces = engine.module(**prepared)

    num_ttt = len(plosses)
    loss_weights = [0.8 ** i for i in range(num_ttt)]

    # Local valid count for this micro-batch — drives the per-mb scale factor.
    # Use position_mask if present (vocab pruning), else loss_mask.
    target_obj = prepared["target"]
    if getattr(target_obj, "position_mask", None) is not None:
        mb_valid = int(target_obj.position_mask.sum().item())
    else:
        mb_valid = int(prepared["loss_mask"].sum().item())

    if mb_valid == 0:
        # Empty after TTT shifting too — skip backward to avoid NaN.
        # The in-kernel fallback already produced zero-grad-touching-all-params
        # from forward, so FSDP's reduce-scatter on the LAST mb still works.
        # We just don't accumulate this mb's loss.
        return {
            "plosses": [p.detach() for p in plosses],
            "acces": [a.detach() for a in acces],
            "loss_weights": loss_weights,
            "mb_valid": 0,
        }

    # Exact mean divisor: Σ_k (mb_valid_k / total_valid_global) · per_pos_mean_k
    # = (1 / total_valid_global) · Σ_k Σ_pos per_pos_loss = mean over total valid.
    scale = mb_valid / total_valid_global
    weighted = sum(w * p * scale for w, p in zip(loss_weights, plosses))
    weighted.backward()

    out = {
        "plosses": [p.detach() for p in plosses],
        "acces": [a.detach() for a in acces],
        "loss_weights": loss_weights,
        "mb_valid": mb_valid,
    }
    # Eagerly free heavy tensors before next fetch.
    del prepared, plosses, acces, weighted, batch_dev, target_obj
    return out


def _aggregate_micro_metrics(self, accum_metrics, grad_norm, lr, rank) -> dict:
    """Sum per-micro-batch losses (already pre-weighted by mb_valid/total) and
    accumulate accuracies weighted by mb_valid. Reuses the existing
    `_aggregate_drafter_metrics` shape so smoke output is unchanged."""
    if not accum_metrics:
        return {}
    num_ttt = len(accum_metrics[0]["plosses"])
    loss_weights = accum_metrics[0]["loss_weights"]
    total_valid = sum(m["mb_valid"] for m in accum_metrics)

    # Combine per-TTT-step losses across micro-batches: weighted by mb_valid.
    combined_plosses = []
    combined_acces = []
    for ttt_i in range(num_ttt):
        if total_valid == 0:
            combined_plosses.append(torch.tensor(0.0))
            combined_acces.append(torch.tensor(0.0))
            continue
        ploss_sum = sum(m["plosses"][ttt_i] * m["mb_valid"] for m in accum_metrics) / total_valid
        acc_sum = sum(m["acces"][ttt_i] * m["mb_valid"] for m in accum_metrics) / total_valid
        combined_plosses.append(ploss_sum)
        combined_acces.append(acc_sum)

    return self._aggregate_drafter_metrics(
        plosses=combined_plosses,
        acces=combined_acces,
        loss_weights=loss_weights,
        grad_norm=grad_norm,
        lr=lr,
        rank=rank,
        prefix="train",
    )
```

### C.4 — `_fetch_drafter_batch_from_mooncake` — accept `t_pad_override`

**Edit 18 — engine_workers.py (around line 248)**: thread `t_pad_override` through to the collator call.

```python
def _fetch_drafter_batch_from_mooncake(
    self, data: DataProto, rank: int, t_pad_override: Optional[int] = None,    # ← NEW
):
    # ... existing fetch logic for the keys in `data` ...

    return collator(features, bucket_size_override=t_pad_override)              # ← UPDATED CALL
```

The existing per-key `store.get` + `store.remove_eagle3_tensors` cleanup pattern is already paged-friendly — works unchanged when called per micro-batch.

### C.5 — Config additions

**Edit 19 — `draft_model_pretrain_trainer.yaml`** (under `engine_config:`):

```yaml
engine_config:
  strategy: fsdp2
  fsdp_size: -1
  param_offload: False
  optimizer_offload: False
  use_remove_padding: False
  # NEW:
  # Number of samples per micro-batch in update_drafter. Total samples per
  # macro-step is data.train_batch_size; accumulation_steps = ceil(B_macro / micro_size).
  # micro_size=1 minimizes peak resident memory for target_p_padded; raise for fewer
  # accumulation steps if memory headroom permits.
  micro_batch_size_per_gpu: 1
```

**Edit 20** — same addition in `drafter_ct_trainer.yaml`.

### C.6 — Verify Phase C

```bash
# Test 1: micro_size=B (single-shot equivalence) — must match Phase B baseline
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16 \
  +actor_rollout_ref.drafter.engine_config.micro_batch_size_per_gpu=4 \
  2>&1 | tee /tmp/phase_c_micro4.log

# Test 2: micro_size=1 (4-way accumulation if data.train_batch_size=4) — must match within float noise
DATA_DIR=/root/verl/data/qwen3_8b_eagle3_ultrachat \
  ./recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  trainer.total_training_steps=16 \
  +actor_rollout_ref.drafter.engine_config.micro_batch_size_per_gpu=1 \
  2>&1 | tee /tmp/phase_c_micro1.log

# Compare both against Phase B
for t in micro4 micro1; do
  echo "=== $t vs Phase B ==="
  diff <(grep "loss_weighted" /tmp/phase_b_smoke.log) <(grep "loss_weighted" /tmp/phase_c_$t.log)
done
```

**Acceptance criteria:**
- Phase C `micro_size=4` curve = Phase B curve **bitwise** (single-shot equivalence — same code path effectively).
- Phase C `micro_size=1` curve = Phase B curve within `~1e-3` (float-noise from accumulation order).

**Commit:** `[drafter] phase C: paged Mooncake fetch + micro-batch accumulation`

---

## 6. Phase D — Documentation

**Edit 21 — `claude_docs/drafter-design.md`**:
- §"Engine choice — FSDP1 with `use_orig_params=True`": rewrite to "FSDP2 selective wrap of `LlamaDecoderLayer` (TorchSpec-style); root unit holds `lm_head`/`norm`/`fc`/`embed_tokens` and stays gathered through backward via auto-root-detection."
- §"What's not built yet": strike "Gradient accumulation > 1" bullet.

**Edit 22 — `claude_docs/migration-status.md`**: add an "FSDP2 + micro-batching" entry summarizing the four phases.

**Edit 23 — `claude_docs/project-guide.md`**: update the file table row for "Drafter worker / engine logic" if Phase E was applied (`engine_workers.py` instead of `fsdp_workers.py`).

**Commit:** `[docs] update drafter docs for FSDP2 + micro-batching`

---

## 7. Verification protocol — full

Run after Phase C is committed:

| Test | Command | Acceptance |
|---|---|---|
| **Unit tests** | `pytest recipe/drafter_cotraining/tests/test_eagle3_loss.py -v` | All pass |
| **Phase A baseline** | `... total_training_steps=16` (default micro=1, but Phase A still uses single-shot path) | Loss curve recorded as `/tmp/phase_a_smoke.log` |
| **Phase B no-regression** | Same | Curve matches Phase A within `~1e-3` |
| **Phase C single-shot equivalence** | `... +micro_batch_size_per_gpu=B_macro` | Bitwise match Phase B |
| **Phase C accumulation parity** | `... +micro_batch_size_per_gpu=1` | Match Phase B within `~1e-3` |
| **Memory** | `nvidia-smi dmon` during Phase C micro=1 | Peak `target_p_padded` per-mb ≈ `T_pad × V_full × 2 B`, not `B_macro ×` that |
| **DP determinism** | Add a one-time `dist.all_reduce` on `total_valid_global` + assert across ranks | Bitwise equal |

---

## 8. Risk register (deltas vs original plan)

| Risk | Likelihood | Detection | Mitigation |
|---|---|---|---|
| Qwen3-8B's `tie_word_embeddings=False` triggers verl's default lm_head wrap (Δ5) | **Confirmed-need** for override | Phase B FSDP2 smoke errors with "Tensor × DTensor" if override missing | Override `_build_fsdp_module` per Edit 11 — already in plan |
| `set_requires_gradient_sync` API not present on root unit | Low (PyTorch 2.4+) | `getattr(fsdp_root, "set_requires_gradient_sync", None)` returns None → no-op | Code uses `getattr` guard (Edit 17). If FSDP1 mode somehow re-engaged, no-op is correct |
| `_select_data_indices` mishandles `np.ndarray` non_tensor_batch entries | Medium (verl's DataProto has both list and ndarray) | Phase C smoke errors on `len()` mismatch | Edit 17's helper handles both branches; check via `isinstance(v, (list, np.ndarray))` |
| `t_pad_macro` differs across DP ranks → reduce-scatter shape mismatch | Low (`make_nd_compute_dataproto_dispatch_fn` chunks evenly) | Backward fails with "shape mismatch" | Optionally `all_reduce(MAX, t_pad_macro)` before passing to collator (cheap insurance) |
| First micro-batch exhausts GPU before any free → OOM | Low (target_p_padded for one Qwen3-8B sample ≈ 1.3 GB bf16 at T=4096) | OOM on first macro-step | `micro_batch_size_per_gpu=1` (the default we're shipping); `torch.cuda.empty_cache()` once per macro-step in outer loop if needed |

---

## 9. Open questions to confirm before/during execution

1. **Baseline.** Auto mode default: capture fresh by running a 16-step smoke on the current branch *before* Phase A (so Phase A vs Phase B vs Phase C diffs are bisectable). If this is too slow, skip and just compare phase-to-phase.
2. **Vocab pruning** — keeping the t2d branch in `compute_target_p_padded` (vs deleting). Decision: **keep both branches** (cheap, future-proof). If you prefer ripping out the pruning machinery entirely, say so.
3. **Defensive DTensor `.full_tensor()` fallback in `_calculate_loss`** — apply preemptively or only on failure? Plan default: only on failure.
4. **Rename Phase E first?** Plan default: yes, separate commit.

---

## Related

- [[Drafter Micro-Batching Refactor Plan]] — rationale + research (read this for "why")
- [[LazyTarget vs Precomputed Memory Analysis]] — kernel choice deep-dive
- [[FSDP Sharding in Speculators]] — alternative FSDP2 wrap pattern (per-block + root)
- [[Eagle3 Comparison — speculators vs TorchSpec]]
