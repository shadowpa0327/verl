# Drafter–Target Parameter & Data Sharing

Reference: TorchSpec (`reference/TorchSpec/torchspec/`)

---

## Parameter Sharing Summary

| Module | Location | Source | Trainable? | Purpose |
|---|---|---|---|---|
| `embed_tokens` | `draft_model.embed_tokens` | Loaded from target checkpoint, frozen | No | Token embeddings for draft model input |
| `lm_head` | `draft_model.lm_head` | Draft model's own (from draft checkpoint) | **Yes** | Produces draft logits in loss kernel |
| `norm` | `draft_model.norm` | Draft model's own | **Yes** | Draft model's RMSNorm (used in loss kernel) |
| `fc` | `draft_model.fc` | Draft model's own (new projection layer) | **Yes** | Fuses [hidden_state, token_embed, pos_info] (3*D -> D) |
| `midlayer` | `draft_model.midlayer` | Draft model's own | **Yes** | 1 transformer decoder layer |
| `target_lm_head_weight` | `eagle3_trainer.target_lm_head_weight` | Loaded from target via `TargetLMHead` | No | Computes target distribution (Forward KL "labels") |
| `verifier_norm` | `eagle3_trainer.verifier_norm` | Loaded from target via `TargetLMHead` | No | Normalizes pre-norm hidden states before target logit computation |

### Key distinction: two lm_heads

- **`draft_model.lm_head`** (trainable) — produces draft logits. Used inside the fused loss kernel via `get_lm_head_params()` -> `F.linear(norm_hs, lm_head_weight)`.
- **`target_lm_head_weight`** (frozen) — produces target distribution. Used in `compute_lazy_target_padded()` / `compute_target_p_padded()` to build the Forward KL target.

These are **not** the same tensor. The draft model learns its own output projection.

---

## Frozen modules: how they're loaded in TorchSpec

### embed_tokens

- `Eagle3Trainer.init_model()` calls `draft_model.load_embedding(target_model_path)` (loads from target)
- Then `draft_model.freeze_embedding()` sets `embed_tokens.weight.requires_grad = False`
- See: `torchspec/models/draft/base.py:187-246`

### target_lm_head_weight + verifier_norm

- `Eagle3Trainer._init_target_lm_head(target_model_path)` loads a `TargetLMHead` object
- `TargetLMHead` contains `lm_head.weight` and optionally `norm` (when `last_hidden_states_prenorm=True`)
- Loaded from target checkpoint, broadcast across ranks, kept frozen
- See: `torchspec/training/eagle3_trainer.py:188-229`

---

## Data shared at runtime

| Data | Direction | Transport | When |
|---|---|---|---|
| Hidden states (pre-norm) | Target (vLLM HS Collector) -> Drafter | Mooncake KV store | After rollout, prefill-only forward pass |
| Input sequences (input_ids) | RL pipeline -> HS Collector -> Drafter | 3-level data pipeline (metadata on driver) | After rollout |
| Actor weights (for frozen module sync) | Actor -> Drafter | `sync_frozen_modules_from_actor()` | After actor weight update |

---

## verl implementation — aligned with TorchSpec

`sync_frozen_modules_from_actor()` syncs:
- `embed_tokens` — frozen, copied from actor
- `target_lm_head_weight` — frozen, from actor's `lm_head` (for target distribution only)
- `verifier_norm` — frozen, from actor's `model.norm` (for pre-norm correction)

`draft_model.lm_head` is the draft model's own trainable parameter and is **not** overwritten or frozen.
