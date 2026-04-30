# Drafter ↔ Target — Parameters & Weight-Sync Flows

Documents:
1. **What's shared** between draft and target (parameter inventory).
2. **How it stays in sync** as the actor trains every RL step.

**Key principle.** In verl, the actor IS the target model and trains
every RL step (unlike TorchSpec's offline distillation where the
target is fixed). Frozen module copies in the drafter need re-syncing
after each `update_actor()`.

---

## Parameter inventory

| Module | Lives at | Source | Trainable? | Used for |
|---|---|---|---|---|
| `embed_tokens` | `Eagle3Model.draft_model.embed_tokens` | Loaded from target checkpoint at init, frozen | No | Token embeddings for draft input in TTT loop |
| `fc` | `Eagle3Model.draft_model.fc` | Random init (new) | **Yes** | Project 3*D aux HS → D for the backbone |
| `midlayer` | `Eagle3Model.draft_model.midlayer` | Random init (new) | **Yes** | One transformer decoder layer |
| `norm` | `Eagle3Model.draft_model.norm` | Random init (new) | **Yes** | Draft's own RMSNorm before its lm_head (in the loss kernel) |
| `lm_head` | `Eagle3Model.draft_model.lm_head` | Random init (new) | **Yes** | Draft's own output projection — produces draft logits in the loss kernel |
| `target_lm_head_weight` | `FSDPDrafterEngine._target_lm_head_weight` | Loaded from target checkpoint at init | No | Compute target distribution for Forward KL (separate from draft's lm_head) |
| `verifier_norm` | `FSDPDrafterEngine._verifier_norm` | Built from target's `model.norm.weight` at init | No | Apply to pre-norm `last_hidden_states` from vLLM before target logit computation |

**Two lm_heads, one trainable.** `draft_model.lm_head` (trainable)
produces draft logits; `target_lm_head_weight` (frozen) produces
target distribution. Distinct tensors — the draft learns its own
output projection.

**Tied-embedding fallback.** When the target uses
`tie_word_embeddings=True` (e.g. Qwen3-4B) and `lm_head.weight` isn't
stored separately in the safetensors, `_load_target_frozen_weights`
falls back to `model.embed_tokens.weight`. Same convention as TorchSpec.

---

## Sync flows

```
                    ┌─────────────┐
                    │   Actor     │  (trains every RL step)
                    │   (FSDP)    │
                    └──┬──┬──┬───┘
          Flow 1       │  │  │       Flow 3
       (full wts)      │  │  │    (frozen only)
           ┌───────────┘  │  └───────────┐
           ▼              ▼ Flow 2       ▼
     ┌──────────┐  ┌────────────┐  ┌──────────────┐
     │ Rollout  │  │HS Collector│  │   Drafter    │
     │  (vLLM)  │  │   (vLLM)   │  │   Trainer    │
     └──────────┘  └────────────┘  └──────┬───────┘
           ▲                              │
           └──────────────────────────────┘
                      Flow 4
                (drafter trainable wts)
```

### Flow 1 — Actor → Rollout (full weights)
- Inherited from `ActorRolloutRefWorker.update_weights()` in `verl/workers/engine_workers.py`.
- Fires every RL step's `update_weights()`.
- Status: **done** (upstream verl).

### Flow 2 — Actor → HS Collector (full weights)
- `RayDrafterCTPPOTrainer` calls `self.hs_collector_manager.update_weights(actor_params)`; manager fans out to its `RolloutReplica` instances.
- Files: `recipe/drafter_cotraining/hs_collector/model.py` (`HSCollectorManager.update_weights`), `recipe/drafter_cotraining/trainer/ray_trainer.py` (call site).
- Status: **stub** — wire to verl's checkpoint_engine, same mechanism as actor → rollout.

### Flow 3 — Actor → Drafter (frozen modules)

| Frozen module | From actor | To drafter |
|---|---|---|
| `embed_tokens` | `actor.model.embed_tokens` | `Eagle3Model.draft_model.embed_tokens` |
| `verifier_norm` | `actor.model.norm` (final RMSNorm) | `FSDPDrafterEngine._verifier_norm` |
| `target_lm_head_weight` | `actor.lm_head.weight` | `FSDPDrafterEngine._target_lm_head_weight` |

- **NOT synced**: the drafter's own `lm_head` — trainable, part of the draft model.
- **Current implementation**: at drafter init, all three are loaded directly from `target_model_path` (not from the actor module) — see `FSDPDrafterEngine._load_target_frozen_weights` and `_build_module`. This sidesteps FSDP1-sharded actor param access at the cost of needing `target_model_path` set in config.
- **Re-sync after `update_actor()`**: `sync_frozen_modules_from_actor` is currently a **no-op stub**. Properly implementing this needs FSDP-aware gathering of the actor's sharded params; deferred. For the smoke (target frozen) it's not needed.
- File: `recipe/drafter_cotraining/engine/drafter_engine.py` + `recipe/drafter_cotraining/engine/workers.py::_sync_drafter_frozen_modules`.

### Flow 4 — Drafter → Rollout (trainable weights)
- `self.drafter.engine.get_per_tensor_param()` returns the drafter's trainable params.
- `self.rollout.update_drafter_weights(...)` — **not yet implemented** on the rollout side.
- File: `recipe/drafter_cotraining/engine/workers.py::update_weights` (`pass` placeholder).
- Status: **TODO 4**. Required only for speculative-decoding rollout speedup, not for training correctness. TorchSpec analog: `_maybe_sync_draft_weights` in `controller/loop.py:42-73` (every 500 steps in the `train_with_decode` recipe).

---

## Sync timing within an RL step

```
generate_sequences()                              ← rollout uses prev step's actor wts (Flow 1)
  ↓
hs_collector_manager.compute_hidden_states()      ← HS collector wakes, uses prev step's wts (Flow 2), sleeps
  ↓
update_drafter()                                  ← drafter uses frozen copies (Flow 3)
  ↓
update_actor()                                    ← actor weights change here
  ↓
update_weights()                                  ← Flows 1–4 all fire here
  ├─ Flow 1: actor → rollout                      (inherited via checkpoint_manager)
  ├─ Flow 2: actor → HS collector                 (TODO stub)
  ├─ Flow 3: actor → drafter frozen               (no-op stub today; target-path init covers it for the smoke)
  └─ Flow 4: drafter → rollout                    (TODO 4)
```

Flows 1–3 prepare the rollout / HS collector / drafter for the **next**
RL step. The drafter trains on hidden states collected from the
**current** step's actor weights.

---

## Why `verifier_norm` matters (Flow 3 detail)

vLLM captures `last_hidden_states` **before** the final RMSNorm — this
is a vLLM implementation detail (the KV connector pulls layer N's
output, which is pre-norm). The drafter's `prepare_model_inputs`
applies `_verifier_norm` explicitly before computing target logits:

```
vLLM output:    last_hidden_states (pre-norm)
                       ↓
Drafter:        verifier_norm(last_hidden_states)        ← Flow 3 keeps in sync
                       ↓
                target_lm_head_weight @ normed_hs        ← Flow 3 keeps in sync
                       ↓
                target_logits → Forward KL loss
```

If `verifier_norm` drifts from the actor's `model.norm`, target logits
become invalid. That's why the actor → drafter re-sync is necessary
when the actor trains. Until the FSDP-aware sync is implemented, target
init from disk + frozen-actor smoke is the working configuration.
