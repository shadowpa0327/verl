---
title: Eagle3 Implementation (speculators repo)
date: 2026-04-25
tags:
  - eagle
  - eagle3
  - speculative-decoding
  - drafter
  - ttt
  - speculators
aliases:
  - Eagle3 in speculators
  - Eagle3 TTT (speculators)
---

# Eagle3 Implementation (speculators repo)

How the Eagle3 draft model is implemented in the `speculators` repository (`/Users/brian1009/Desktop/Cornell_Documents/Research/Projects/speculators`). The core entry point is `Eagle3DraftModel.forward` in `src/speculators/models/eagle3/core.py`, which runs an unrolled multi-step draft loop ("test-time training", TTT) per forward pass.

> [!info] Repo layout
> - Core model: `src/speculators/models/eagle3/core.py`
> - Attention / mask helpers: `src/speculators/models/eagle3/attention.py`
> - Model definitions registry: `src/speculators/models/eagle3/model_definitions.py`
> - Config: `src/speculators/models/eagle3/__init__.py` (exports `Eagle3SpeculatorConfig`)
> - Converter (HF EAGLE → speculators): `src/speculators/convert/eagle/eagle_converter.py`
> - Training script: `scripts/train.py`
> - Docs: `docs/cli/train.md`, `docs/developer/add_algorithm.md`

---

## Architecture (what gets trained)

`Eagle3DraftModel` (registered as `"eagle3"` via `SpeculatorModel.register("eagle3")` at `src/speculators/models/eagle3/core.py:156`) is a draft model with:

- **`fc`** — projects concatenated `[input_embed | verifier_hidden]` (size `2*hidden_size` after concat at runtime, but `3*hidden_size` upstream when verifier hidden states from 3 layers are concatenated) down to `hidden_size`.
- **`layers`** — Eagle3 decoder layers (one or a few transformer blocks, configured by `transformer_layer_config`).
- **`norm`** + **`lm_head`** — produce draft logits.
- **`embed_tokens`** — token embedding (typically shared from verifier).
- **`verifier_norm`** + **`verifier_lm_head`** — frozen, used during training to produce supervision targets from the verifier's last hidden states (`core.py:312-316`).
- **`d2t`** / **`t2d`** — draft↔verifier vocab remapping when the draft vocabulary is a reduced subset.

Mixin: `DraftVocabMixin` from `src/speculators/model.py` supplies the reduced-vocab logic; base class is `SpeculatorModel`.

---

## Forward pass overview

`Eagle3DraftModel.forward` at `src/speculators/models/eagle3/core.py:265-406` accepts:

```python
hidden_states                  # [1, total_seq_len, 3 * hidden_size]
input_ids                      # [1, total_seq_len]
lengths                        # [batch_size] — per-sample lengths in the packed batch
loss_mask                      # [1, total_seq_len] (optional)
position_ids                   # [1, total_seq_len] (optional, defaults to 1..L)
verifier_last_hidden_states    # [1, total_seq_len, hidden_size] — gates training mode
ttt_steps: int = 3
ttt_step_loss_decay: float = 1.0
use_off_policy_tokens: bool = False
```

Pre-loop setup (`core.py:280-307`):

1. Build a packed-sequence block mask via `create_combined_mask_mod(lengths, total_seq_len)` and `torch.nn.attention.flex_attention.create_block_mask` — the mask respects per-sample document boundaries inside the packed batch.
2. Optional `input_norm` then `fc` project `hidden_states` from `3 * hidden_size` down to `hidden_size`.
3. If `verifier_last_hidden_states` is provided (training), compute frozen targets `targets = verifier_lm_head(verifier_norm(verifier_last_hidden_states))`.
4. Initialize `prev_correct` (boolean accumulator across TTT steps, used for conditional/chain accuracy).

---

## The TTT loop (`ttt_steps` unrolled)

Implemented at `src/speculators/models/eagle3/core.py:331-399`:

```python
for ttt_step in range(ttt_steps):
    input_embeds  = embed_tokens(input_ids)                  # no grad
    cache_position = arange(ttt_step*L, (ttt_step+1)*L)      # KV cache slot
    hidden_states  = cat([input_embeds, hidden_states], -1)  # [1, L, 2*H]
    pos_emb        = rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=pos_emb,
        )

    logits = lm_head(norm(hidden_states))                    # [1, L, V_draft]

    if return_loss:
        s_loss, s_metrics = compute_metrics(
            logits, targets, loss_mask, prev_correct,
            ttt_step, ttt_step_loss_decay,
        )
        loss += s_loss
        metrics.update(s_metrics)

    input_ids = argmax(logits, dim=-1)
    if d2t is not None:                  # remap draft vocab → verifier vocab
        input_ids = input_ids + d2t[input_ids]
    if use_off_policy_tokens:            # teacher-forced ground-truth feed
        input_ids = shift_left(original_input_ids, 1 + ttt_step, pad=0)

    attention_mask = extend_mask_for_draft_tokens(attention_mask)
    position_ids   = position_ids + 1
```

Returns `(draft_tokens, loss, metrics)` in training, `draft_tokens` otherwise (`core.py:402-406`).

### Per-step mechanics

1. **Single growing KV cache** — `past_key_values = DynamicCache(...)` (`core.py:291`); each step writes a new chunk at `cache_position = [k*L .. (k+1)*L]`, so step `k` attends back to all previous chunks.
2. **Mask growth** — `extend_mask_for_draft_tokens(attention_mask)` (`core.py:398`, defined in `src/speculators/models/eagle3/attention.py`) widens the BlockMask each step so newly drafted positions can attend to keys produced in earlier steps. The behavior is covered by `tests/unit/train/test_eagle3_attention.py` (see comment at line 106).
3. **Position advance** — `position_ids += 1` per step (`core.py:399`); step `k` predicts the token `k` positions further to the right.
4. **Concat trick** — the input to the decoder block at each step is `cat(input_embeds, hidden_states)` of size `2*hidden_size`; this is the Eagle "embedding + previous-layer hidden" pattern (the initial `hidden_states` enters the loop at `hidden_size` after `fc`).
5. **Self-feeding draft tokens** — by default `input_ids` for step `k+1` is `argmax(logits)` from step `k` (on-policy rollout), with `d2t` remapping back to verifier vocab space before re-embedding (`core.py:380-381`).
6. **Off-policy mode** — `use_off_policy_tokens=True` overrides with the verifier's ground-truth shifted by `1 + ttt_step` and zero-padded (`core.py:383-396`); the padded tail is later sliced out by `align_for_step`.

---

## Loss and metrics

### `align_for_step` (`core.py:23-55`)

Aligns logits/targets for step `k`:

```
logits[:, :-k]      # drop last k positions (no targets exist that far ahead)
targets[:, k:]      # drop first k positions (no logits at depth k for them)
loss_mask[:, k:]    # follow targets
prev_correct[:, :-k]  # follow logits (draft side)
```

This implements "logits at step `k` supervised against tokens `k+1` ahead".

### `loss_function` (`core.py:85-104`)

Forward-KL between draft log-softmax and verifier softmax targets, optionally masked, denominator is the masked count (or sequence length if no mask).

### `compute_metrics` (`core.py:107-146`)

For each step:

- `s_loss = (ttt_step_loss_decay ** ttt_step) * loss_function(...)`
- `s_full_acc` — argmax accuracy under the loss mask.
- `s_cond_acc` — chain accuracy: argmax matches **and** all previous steps were also correct (`prev_correct` is `logical_and`-updated in place inside `compute_accuracy` at `core.py:75`).

Per-step metrics are emitted as `loss_{k}`, `full_acc_{k}`, `cond_acc_{k}`; the total `loss` is the sum over steps.

---

## Construction / wiring

- **`Eagle3DraftModel.from_training_args`** (`core.py:408`-) builds the model from a verifier config, vocab maps, and CLI hyperparameters; it sets up a `GreedyTokenProposalConfig` with `speculative_tokens=ttt_steps` (`core.py:447`) and stores `ttt_steps` / `ttt_step_loss_decay` in both the `SpeculatorsConfig` and the per-model config (`core.py:473-479`).
- **CLI** (`scripts/train.py:564-565`) — `--ttt-steps` (default `3`) and `--ttt-step-loss-decay` (default `1.0`).
- **Docs** — `docs/cli/train.md:117-119` documents both flags as user-facing knobs.
- **Add-algorithm guide** — `docs/developer/add_algorithm.md:100` references `--ttt-steps` as the canonical example of an algorithm-specific hyperparameter passed through `from_training_args`.

---

## Conversion path (HF EAGLE checkpoints → speculators)

`src/speculators/convert/eagle/eagle_converter.py` converts external EAGLE checkpoints (e.g. `nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT`, see docstring at `eagle_converter.py:94` and integration test at `tests/integration/convert/test_eagle.py:178`) into the `Eagle3SpeculatorConfig` format consumed by `Eagle3DraftModel`.

---

## Tests

- `tests/unit/train/test_eagle3_attention.py` — mask extension across TTT steps (line 106 comment explains the extension semantics).
- `tests/unit/train/test_setup_model.py` — Eagle3 model construction.
- `tests/unit/convert/test_eagle3_converter.py` / `tests/integration/convert/test_eagle3.py` — converter coverage.
- `tests/e2e/regression/test_eagle3_conversion_acceptance.py`, `tests/e2e/regression/test_eagle3_online_acceptance.py` — end-to-end regression.
- `tests/e2e/smoke/test_online_training.py`, `tests/e2e/smoke/test_offline_training.py` — smoke runs of the training loop.

---

## Quick mental model

> [!summary]
> One forward pass = `ttt_steps` unrolled draft predictions, sharing one growing KV cache. Step `k` is supervised against the verifier's distribution `k+1` tokens ahead, weighted by `ttt_step_loss_decay**k`, summed into a single backward. Default rollout is on-policy (feed own argmax + `d2t` remap); `use_off_policy_tokens=True` switches to teacher forcing.

## Related

- [[Eagle3 Training Explained]] — TorchSpec-flavored explanation of the same algorithm.
- [[TorchSpec to verl Migration Map]]