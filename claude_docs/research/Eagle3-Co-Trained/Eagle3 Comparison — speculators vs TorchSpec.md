---
title: Eagle3 Comparison — speculators vs TorchSpec
date: 2026-04-25
tags:
  - eagle
  - eagle3
  - speculative-decoding
  - speculators
  - torchspec
  - comparison
aliases:
  - speculators vs TorchSpec Eagle3
  - Eagle3 implementation diff
---

# Eagle3 Comparison — speculators vs TorchSpec

Side-by-side comparison of the Eagle3 draft-model training implementation in the **`speculators`** repo (`Eagle3DraftModel.forward` in `src/speculators/models/eagle3/core.py`) against the **TorchSpec** reference implementation summarized in [[Eagle3 Training Explained]].

> [!info] Source notes
> - speculators: [[Eagle3 Implementation (speculators repo)]]
> - TorchSpec: [[Eagle3 Training Explained]]

---

## TL;DR

> [!summary]
> Both implementations follow the same EAGLE-3 recipe: project 3-layer concatenated verifier hidden states → run an N-step unrolled draft loop sharing one growing KV cache → KL-divergence loss against the verifier's distribution at each step → single backward. They differ in **rollout policy** (on-policy argmax in speculators vs teacher-forced shift in TorchSpec), **configurability** (speculators makes step count / decay / vocab-pruning runtime args; TorchSpec hardcodes them), **packing strategy** (speculators uses FlexAttention block masks for true sequence packing; TorchSpec uses standard padded batches), **loss kernel** (TorchSpec fuses RMSNorm+lm_head+KL via `torch.compile`; speculators uses `torch.nn.functional.kl_div`), and **data pipeline** (TorchSpec pulls precomputed hidden states from a Mooncake KV store; speculators is data-format-agnostic and drives both online and offline training from the same `forward`).

---

## At-a-glance table

| Aspect | speculators | TorchSpec |
|---|---|---|
| Entry point | `Eagle3DraftModel.forward` (`src/speculators/models/eagle3/core.py:265-406`) | `Eagle3Model.forward` (`torchspec/models/eagle3.py:192-239`) |
| Number of TTT steps | **Configurable**, `--ttt-steps` (default `3`) | **Hardcoded** `length = 7` |
| Per-step loss decay | **Configurable**, `ttt_step_loss_decay**k` (default `1.0` = no decay) | **Hardcoded** `0.8**k` |
| Decoder layers in drafter | `tl_config.num_hidden_layers` (multi-layer capable; first layer is special `first_layer_class`) — `core.py:192-205` | Single `midlayer` (`torchspec/models/draft/llama3_eagle.py:1737-1797`) |
| Draft-token rollout (default) | **On-policy**: feeds back `argmax(logits)` (with `d2t` remap) — `core.py:374-381` | **Teacher-forced**: shifts ground-truth `input_ids` right with padding |
| Off-policy / teacher-forced | Optional via `use_off_policy_tokens=True` (`core.py:383-396`) | Default and only mode |
| Loss form | `F.kl_div(log_softmax(logits), softmax(targets), log_target=False)` — `core.py:91-95` | Hand-rolled forward-KL fused with RMSNorm+lm_head in `@torch.compile` kernels — `torchspec/models/ops/loss.py:25-108` |
| Target construction | Eager: `verifier_lm_head(verifier_norm(verifier_last_hidden_states))` cached once before loop (`core.py:312-316`) | `LazyTarget`: target softmax computed inside compiled loss kernel from raw last-hidden-states + frozen `lm_head` (`compiled_forward_kl_loss_from_hs`) |
| Vocabulary | Reduced draft vocab via `d2t`/`t2d` (`DraftVocabMixin`); draft logits live in `draft_vocab_size` space and are remapped to verifier vocab before re-embedding | Full target vocabulary; optional vocab pruning |
| Sequence packing | True packing: `lengths` + FlexAttention `BlockMask` via `create_combined_mask_mod` (`src/speculators/models/eagle3/attention.py`) — respects per-document boundaries inside one sequence | Standard padded batches `(B, T)` with attention mask |
| Attention mask growth across steps | `extend_mask_for_draft_tokens(attention_mask)` widens BlockMask each step (`core.py:398`) | KV cache grows; no explicit mask extension (causal mask handles it) |
| KV cache | HF `DynamicCache(...)` keyed by `cache_position` chunks of size `total_seq_len` (`core.py:291`, `core.py:335-340`) | Manual `cache_keys`/`cache_values` tensors threaded through `backbone()` returns |
| Per-step alignment | `align_for_step` slices `logits[:, :-k]`, `targets[:, k:]` after the fact (`core.py:23-55`) | Targets pre-padded to `T+length`, sliced at `target_p[i:T+i]` vs `draft_p[0:T]`; inputs pre-shifted with `pad_right` |
| Compile strategy | `@conditional_torch_compile` wraps the entire `forward` (whole graph compiled when CUDA + `torch.compile` available) — `core.py:149-153, 265` | Per-kernel `@torch.compile(dynamic=None)` on the loss only (`compiled_forward_kl_loss`) |
| Accuracy metrics | `full_acc_k` (under loss mask) **and** `cond_acc_k` (chain accuracy via in-place `prev_correct` AND-reduction) (`core.py:58-82, 319-327`) | Per-step argmax-match accuracy only |
| Data pipeline | Format-agnostic; `forward` consumes tensors regardless of source. Online or offline training driven from the same script (`scripts/train.py`) | Hidden states fetched from **Mooncake KV store** via `EagleMooncakeStore.get()` (`torchspec/transfer/mooncake/eagle_store.py`) |
| Gradient accumulation | Handled by training script (no explicit micro-batch loop in `forward`) | Explicit micro-batch loop in `_train_core_from_queue()` (`torchspec/training/trainer.py:279-342`) |
| Conversion path | HF EAGLE checkpoints → speculators format via `src/speculators/convert/eagle/eagle_converter.py` (e.g. `nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT`) | n/a in TorchSpec note |

---

## 1. Drafter architecture

> [!example] speculators
> - `fc`: `Linear(3*hidden_size → hidden_size)` — `core.py:189`
> - `layers`: `ModuleList` of `num_hidden_layers` decoder layers; first uses `first_layer_class`, rest use `decoder_layer_class` from `model_classes[model_type]` (`core.py:192-205`)
> - `embed_tokens`: token embedding (typically shared from verifier; `requires_grad` controlled by `embed_requires_grad`)
> - `norm` + `lm_head`: produce draft logits in **draft-vocab space**
> - `verifier_norm` + `verifier_lm_head`: frozen, used to compute targets at training time only (`core.py:312-316`); ignored on save (`_keys_to_ignore_on_save` at `core.py:167-170`)
> - `input_norm` (optional, for `gpt-oss`): `RMSNorm(3*hidden_size)` applied before `fc` (`core.py:222-228`)
> - `rotary_emb`: rotary embedding sized for `2*hidden_size` (because the decoder input is `cat(input_embeds, hidden_states)`) — `core.py:208-211`

> [!example] TorchSpec
> - `fc`: same `3*D → D` projection (`project_hidden_states`)
> - `midlayer`: a **single** decoder layer (~140M params total with `fc`)
> - `embed_tokens` and `lm_head`: **shared frozen references** to the target model's tensors (zero-copy)
> - No separate verifier-target head; targets are derived from `last_hidden_states` and the frozen target `lm_head` lazily

> [!tip] Key differences
> - **Multi-layer drafter** is supported in speculators (depending on `tl_config.num_hidden_layers`); TorchSpec is a single-layer recipe.
> - speculators carries a **dedicated `verifier_lm_head` / `verifier_norm` module** in the draft model itself (loaded from verifier weights), making the draft model self-contained at training time.
> - speculators reduces the draft vocabulary explicitly via `d2t`/`t2d`; TorchSpec keeps the full target vocabulary.

---

## 2. The TTT loop

> [!example] speculators (`core.py:331-399`)
> ```python
> for ttt_step in range(ttt_steps):
>     input_embeds  = embed_tokens(input_ids)                  # no grad
>     cache_position = arange(ttt_step*L, (ttt_step+1)*L)
>     hidden_states  = cat([input_embeds, hidden_states], -1)  # [1, L, 2*H]
>     pos_emb        = rotary_emb(hidden_states, position_ids)
>     for layer in self.layers:
>         hidden_states = layer(..., attention_mask=BlockMask, ...)
>     logits = lm_head(norm(hidden_states))
>     if return_loss:
>         loss += compute_metrics(logits, targets, loss_mask,
>                                 prev_correct, ttt_step, ttt_step_loss_decay)
>     input_ids = argmax(logits)                # default: on-policy
>     if d2t is not None: input_ids = input_ids + d2t[input_ids]
>     if use_off_policy_tokens: input_ids = shift_left(orig_ids, 1+ttt_step, pad=0)
>     attention_mask = extend_mask_for_draft_tokens(attention_mask)
>     position_ids   = position_ids + 1
> ```

> [!example] TorchSpec (`torchspec/models/eagle3.py:192-239`)
> ```python
> for idx in range(7):
>     embeds = embed_input_ids(input_ids)
>     draft_hs, cache_k, cache_v = backbone(
>         input_embeds=embeds,
>         hidden_states=hidden_states,    # projected target HS — residual
>         attention_mask=mask,
>         cache_keys=cache_k, cache_values=cache_v,
>     )
>     loss_i, acc_i = calculate_loss(draft_hs, target, mask, idx)
>     plosses.append(loss_i)
>     if not last_step:
>         input_ids = pad_right(input_ids)     # teacher-forced shift
>         mask      = pad_right(mask)
> ```

### Key per-step contrasts

| Step mechanic | speculators | TorchSpec |
|---|---|---|
| What feeds `input_ids` next step | `argmax(logits) + d2t[...]` (on-policy) — defines the trained model's *own* prefix | Right-shift of original `input_ids` with padding (teacher forcing) |
| Hidden-state path | `hidden_states` is the projected verifier HS only at step 0; subsequent steps use the **decoder output** of the previous step (re-concatenated with `embed_tokens(input_ids)`) | `hidden_states` (projected target HS) is the residual at every step; only `input_ids` shifts |
| Cache addressing | `cache_position` chunks of size `L` (each step writes a fresh chunk to a single `DynamicCache`) | `cache_k`/`cache_v` returned and re-passed; cache extends by 1 effective slot per step (per the diagram) |
| Mask extension | Explicit: `extend_mask_for_draft_tokens(BlockMask)` per step | Implicit causal mask + cache growth |
| Position IDs | Increment by 1 per step | Implicit via cache offset |

> [!tip] On-policy vs teacher-forced
> The default rollout policy is the **most consequential behavioral difference**. On-policy training (speculators default) closes the train/inference gap because the model is graded on prefixes it actually produces, not on prefixes it would never see at inference. Teacher forcing (TorchSpec) trains faster and is simpler but can leave the model exposed when its own argmax diverges from the ground truth.

---

## 3. Loss and metrics

### Loss form

> [!example] speculators (`core.py:85-104`)
> ```python
> logits  = log_softmax(logits, dim=-1)
> target_p = softmax(targets, dim=-1)
> loss = F.kl_div(logits, target_p, reduction="none", log_target=False)
> # loss-mask × per-position; denominator is mask sum (or seq len)
> ```
> Native PyTorch KL-div; whole `forward` is wrapped in `@conditional_torch_compile` so this is part of the larger compiled graph.

> [!example] TorchSpec (`torchspec/models/ops/loss.py:25-108`)
> Hand-rolled forward-KL fused with RMSNorm and the frozen `lm_head` projection in a `@torch.compile(dynamic=None)` kernel:
> ```python
> hs_normed   = (hs * rsqrt(var+eps)) * norm_weight   # manual RMSNorm
> draft_logits = hs_normed @ lm_head_weight.T          # frozen lm_head
> loss = -(target_p * log_softmax(draft_logits, -1)).sum(-1).mean()
> ```
> A `LazyTarget` variant (`compiled_forward_kl_loss_from_hs`) folds even the *target* softmax into the kernel, avoiding materializing the full `(B, T, V)` target distribution.

### Per-step weighting & aggregation

| | speculators | TorchSpec |
|---|---|---|
| Decay | `ttt_step_loss_decay**ttt_step` (default `1.0`) | `0.8**i` |
| Aggregation | `loss += s_loss` inside the forward pass | `ploss = sum(w_i * loss_i) / accum_steps`, then `.backward()` outside |
| Backward through cache | Single `.backward()` from training script | Single `.backward()` in `_backward()` |

### Metrics

| Metric | speculators | TorchSpec |
|---|---|---|
| Per-step loss | `loss_{k}` | `plosses[i]` |
| Argmax accuracy | `full_acc_{k}` (under loss mask) | per-step `acc_i` |
| **Chain accuracy** | `cond_acc_{k}` — token correct at step `k` *and* at every previous step (in-place `prev_correct = logical_and(prev_correct, correct, out=prev_correct)`, `core.py:75`) | not present |

> [!note] `cond_acc` is a more honest acceptance-rate proxy
> The conditional/chain accuracy in speculators directly mirrors the **acceptance rate** of speculative decoding: a draft chunk is only useful up to the first wrong token, so `cond_acc_k` measures "fraction of tokens still on the accepted prefix at depth `k`".

---

## 4. Sequence packing & attention masks

> [!example] speculators
> Inputs come as a single packed batch of shape `[1, total_seq_len, ...]` with a `lengths: [batch_size]` tensor giving per-sample lengths. The mask is constructed via `create_combined_mask_mod(lengths, total_seq_len)` and `torch.nn.attention.flex_attention.create_block_mask` in `src/speculators/models/eagle3/attention.py`. The mask combines:
> - `causal_mask_mod` — standard causal,
> - `document_mask_mod` — blocks attention across document boundaries inside the packed sequence,
> - `diagonal_draft_mask_mod` — used by `extend_mask_for_draft_tokens` to allow new draft positions to attend to their corresponding earlier-step keys.
> The whole forward uses `_attn_implementation = "simple_flex_attention"` (forced in `__init__` at `core.py:177-178`).

> [!example] TorchSpec
> Standard `(B, T)` padded batches with a per-token attention mask. Padding is handled by `pad_right` and per-step expansion is implicit via the autoregressive cache.

> [!tip] Why this matters
> True sequence packing (speculators) significantly improves throughput for variable-length training data, because no GPU time is wasted on padding tokens — but it requires FlexAttention and a more elaborate mask-construction path. TorchSpec's padded form is simpler and portable but pays the padding tax.

---

## 5. Targets and vocabulary

> [!example] speculators
> - **Eager precompute**: `targets = verifier_lm_head(verifier_norm(verifier_last_hidden_states))` is computed once before the loop (`core.py:312-316`), under `no_grad`. This stores `[1, L, draft_vocab_size]` in memory for the whole TTT loop.
> - **Reduced draft vocabulary**: `verifier_lm_head` projects to `draft_vocab_size` (a subset of the full verifier vocabulary). `d2t` (draft-to-target) maps draft token IDs back to verifier IDs before re-embedding inside the loop. `DraftVocabMixin` supplies the plumbing.

> [!example] TorchSpec
> - **`LazyTarget`**: target probabilities are *not* materialized. The compiled loss kernel takes `target_hs_flat` and `target_lm_head_weight` and computes `softmax(target_hs @ lm_head.T)` inside the kernel, only at the indices flagged as valid by `valid_idx`. This trades a re-projection per step for not allocating `(B, T, V)`.
> - **Full vocabulary** (with optional pruning).

> [!tip] Memory tradeoff
> speculators' eager `(1, L, draft_vocab_size)` target is much smaller than `(1, L, V_full)` *because the draft vocab is reduced* — that's the design point that lets it precompute. TorchSpec keeps the full vocab, so it has to defer the softmax to keep peak memory in check.

---

## 6. Per-step alignment

The two implementations align step `k`'s logits and targets in the same conceptual way ("logits at depth `k` are supervised against tokens shifted left by `k`"), but the mechanics differ.

| | speculators | TorchSpec |
|---|---|---|
| Pre-padding | None | Targets pre-padded to `T + length` |
| Per-step slicing | `align_for_step` (`core.py:23-55`): `logits[:, :-k]`, `targets[:, k:]`, `loss_mask[:, k:]`, `prev_correct[:, :-k]` | At step `i`: `target_p[i:T+i]` vs `draft_p[0:T]`; `input_ids` is right-shifted with padding before the next step |
| `loss_mask` semantics | Boolean per-position; both used as a multiplier in loss and as a filter in accuracy | per-step mask aligned with `target_p` slice |

---

## 7. Data pipeline & training driver

> [!example] speculators
> `scripts/train.py` is **format-agnostic**: it consumes whatever batch dict the dataloader provides (`hidden_states`, `input_ids`, `lengths`, `loss_mask`, `position_ids`, `verifier_last_hidden_states`) and forwards them straight into `Eagle3DraftModel.forward`. Both **online training** (running the verifier in-process to produce hidden states on the fly) and **offline training** (consuming pre-extracted hidden states from disk) are driven from the same script and same forward path. CLI knobs include `--ttt-steps` (default `3`), `--ttt-step-loss-decay` (default `1.0`).

> [!example] TorchSpec
> Hidden states are produced by an offline collection stage that runs the target model on complete sequences (vLLM via `extract_hidden_states`, or SGLang via patched prefill) and writes `_hs`, `_ids`, `_lhs` tensors to the **Mooncake KV store**. `FSDPDrafterEngine` + `Eagle3Model` then read tensors back through `EagleMooncakeStore.get()`. Targets are constructed as `LazyTarget` (or `PrecomputedTarget`) and the 7-step loop runs on the cached representations.

> [!tip] Architectural style
> - **TorchSpec** is a tightly-coupled offline pipeline: HS collection (Mooncake) → trainer queue → drafter engine. The model code expects to be fed via that pipeline.
> - **speculators** decouples the model from the pipeline: `forward` takes plain tensors, and the conversion subsystem (`src/speculators/convert/eagle/eagle_converter.py`) lets external EAGLE checkpoints (e.g. HF's `nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT`) be loaded and continued.

---

## 8. Conditional accuracy / acceptance proxy

A subtle but useful difference: speculators emits `cond_acc_{k}` ("fraction of tokens still on the correct chain at depth `k`") which decays monotonically with `k` and approximates the speculative decoding acceptance distribution. TorchSpec exposes only per-step argmax accuracy, which is unconditional and overestimates the useful accept rate at deeper steps.

```
Token positions ──►
         step 0:  ✓ ✓ ✓ ✗ ✓ ✓ ✗ ✓        full_acc_0 = 6/8
                                            cond_acc_0 = 6/8

         step 1:  ✓ ✓ ✓ ✗ ✓ ✗ ✗ ✓        full_acc_1 = 5/8
                                            cond_acc_1 = 4/8   (positions where step 0 AND step 1 both ✓)

         step 2:  ✓ ✓ ✗ ✗ ✓ ✓ ✗ ✓        full_acc_2 = 5/8
                                            cond_acc_2 = 3/8   (chain-survivors)
```

> [!note]
> `prev_correct` is updated **in place** inside `compute_accuracy` (`core.py:75`), so the chain mask threads through the whole TTT loop in O(L) memory.

---

## 9. Compile / kernel strategy

| | speculators | TorchSpec |
|---|---|---|
| Granularity | Whole `forward` is `@torch.compile`'d via `@conditional_torch_compile` (`core.py:149-153`) | Only the loss is `@torch.compile`'d (`@torch.compile(dynamic=None)`) |
| Backbone compile | Implicit (covered by the outer wrapper) | None — eager backbone, custom KV cache plumbing |
| Loss kernel fusion | None beyond what the outer compile does | Manual: RMSNorm + `lm_head` matmul + KL fused into one Triton kernel; `LazyTarget` variant fuses target softmax too |
| Practical effect | Less peak-memory pressure if the compile graph fits; relies on PyTorch's compiler to fuse | Predictable kernel count; explicit memory tradeoffs (`LazyTarget`) |

---

## 10. Conversion / interoperability

`speculators` ships an explicit conversion path from HF EAGLE checkpoints to its own format (`src/speculators/convert/eagle/eagle_converter.py`; integration tests in `tests/integration/convert/test_eagle.py:178` reference `nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT`). TorchSpec, in the linked note, is internally consistent but doesn't document an interop conversion — its checkpoints are coupled to the FSDP+Mooncake training stack.

---

## 11. Where to look — quick file index

| Concern | speculators | TorchSpec |
|---|---|---|
| Forward / TTT loop | `src/speculators/models/eagle3/core.py:265-406` | `torchspec/models/eagle3.py:192-239` |
| Per-step alignment | `src/speculators/models/eagle3/core.py:23-55` (`align_for_step`) | inline in `Eagle3Model.forward` + `pad_right` |
| Loss | `src/speculators/models/eagle3/core.py:85-104` (`loss_function`) | `torchspec/models/ops/loss.py:25-108` |
| Metrics / accuracy | `src/speculators/models/eagle3/core.py:58-82, 107-146` | `torchspec/models/eagle3.py` calculate_loss |
| Backbone / FC | `src/speculators/models/eagle3/core.py:188-205` | `torchspec/models/draft/llama3_eagle.py:1737-1797` |
| Attention / packing | `src/speculators/models/eagle3/attention.py` | standard padded mask |
| Verifier weight loading | `src/speculators/models/eagle3/core.py:232-263` (`load_verifier_weights`) | shared frozen tensor refs in TorchSpec |
| Data path | `scripts/train.py:564-565` (CLI) | `torchspec/transfer/mooncake/eagle_store.py` (`EagleMooncakeStore`) |
| Conversion | `src/speculators/convert/eagle/eagle_converter.py` | n/a |

---

## 12. Tradeoffs cheat sheet

> [!quote] speculators leans toward **flexibility and online training**
> Configurable step count and decay, on-policy default, packed sequences with FlexAttention, format-agnostic data path, explicit draft-vocab reduction, and a conversion bridge to external checkpoints. Cost: more moving parts, multi-layer support means a larger per-step compute budget if you scale it up, and the eager target tensor only fits in memory because the draft vocab is small.

> [!quote] TorchSpec leans toward **a tightly engineered offline pipeline**
> Hardcoded EAGLE-3 hyperparameters that match the original recipe (length 7, decay 0.8), single-layer drafter, fused compiled loss kernel with `LazyTarget` to keep peak memory low on full-vocab targets, teacher-forced rollout, and a Mooncake-based hidden-state cache. Cost: rigid step count, no on-policy mode in the loop, and a pipeline strongly coupled to FSDP + Mooncake.

---

## Related

- [[Eagle3 Implementation (speculators repo)]]
- [[Eagle3 Training Explained]]
- [[TorchSpec to verl Migration Map]]
- [[Drafter Trainer Integration Plan (verl)]]
- [[EAGLE-co-train Verl Integration V1]]