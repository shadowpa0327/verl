# Weight Sync Flows — Drafter Co-Training

Documents all weight synchronization paths between modules in the
EAGLE drafter co-training pipeline.

**Key principle:** In verl the actor IS the target model and trains every
RL step. Unlike TorchSpec (where the target is fixed/offline), frozen
module copies in the drafter must be re-synced after each `update_actor()`.

---

## Module Inventory

| Module | Engine | Weights | Lives on |
|---|---|---|---|
| **Actor** | FSDPEngine | Full target model (all layers) | GPU (training) |
| **Rollout** | vLLM | Mirror of actor (for generation) | GPU (inference) |
| **HS Collector** | vLLM + KV connector | Mirror of actor (for prefill → HS extraction) | GPU (inference), time-multiplexed with rollout |
| **Drafter Trainer** | FSDPDrafterEngine → Eagle3Model | Trainable: `fc`, `midlayer`, `norm`, `lm_head`. Frozen copies: `embed_tokens`, `verifier_norm`, `target_lm_head_weight` | GPU (training), time-multiplexed |

---

## Sync Flows

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

### Flow 1: Actor → Rollout (full weights)

- **What:** All actor parameters
- **When:** `update_weights()` at end of each RL step
- **How:** Inherited from `ActorRolloutRefWorker.update_weights()`
- **Status:** Done (upstream verl)
- **File:** `verl/workers/engine_workers.py`

### Flow 2: Actor → HS Collector (full weights)

- **What:** All actor parameters (HS collector runs the same model for prefill)
- **When:** `update_weights()` at end of each RL step
- **How:** `self.actor.engine.get_per_tensor_param()` → `self.hs_collector.update_weights()`
- **Status:** Wired (code exists, not yet tested end-to-end)
- **File:** `verl/workers/drafter_workers.py:256-259`

### Flow 3: Actor → Drafter Trainer (frozen modules only)

- **What:** Three frozen weight sets copied from actor into drafter:

  | Frozen Module | Source (Actor) | Destination (Drafter) | Purpose |
  |---|---|---|---|
  | `embed_tokens` | `actor.model.embed_tokens` | `Eagle3Model.draft_model.embed_tokens` | Token embedding for drafter input in TTT loop |
  | `verifier_norm` | `actor.model.norm` (final RMSNorm) | `FSDPDrafterEngine._verifier_norm` | Normalizes pre-norm `last_hidden_states` from vLLM before computing target distribution |
  | `target_lm_head_weight` | `actor.lm_head.weight` | `FSDPDrafterEngine._target_lm_head_weight` | Computes target logits for Forward KL loss (via `LazyTarget`) |

- **NOT synced:** The drafter's own `lm_head` — this is trainable, part of the draft model
- **When:** At init (`_init_drafter()`) and after each `update_actor()` (in `update_weights()`)
- **How:** `_sync_drafter_frozen_modules()` → `engine.sync_frozen_modules_from_actor(embed, lm_head, norm)`
- **Status:** Wired (code exists, not yet tested end-to-end)
- **File:** `verl/workers/drafter_workers.py:123-153`

### Flow 4: Drafter Trainer → Rollout (trainable weights)

- **What:** Drafter's trainable parameters (`fc`, `midlayer`, `norm`) pushed to rollout vLLM for speculative decoding
- **When:** `update_weights()` at end of each RL step (after drafter training)
- **How:** `self.drafter.engine.get_per_tensor_param()` → `self.rollout.update_drafter_weights()` (not yet implemented)
- **Status:** TODO (stub with `pass`) — not a blocker for training, only for spec-decode speedup during rollout
- **File:** `verl/workers/drafter_workers.py:261-266`

---

## Sync Timing in RL Step

```
generate_sequences()          ← rollout uses actor weights (Flow 1, prev step)
  ↓
collect_hidden_states()       ← HS collector uses actor weights (Flow 2, prev step)
  ↓
update_drafter()              ← drafter uses frozen copies (Flow 3, prev step)
  ↓
update_actor()                ← actor weights change here
  ↓
update_weights()              ← Flows 1-4 all fire here
  ├─ Flow 1: actor → rollout         (inherited)
  ├─ Flow 2: actor → HS collector    (wired)
  ├─ Flow 3: actor → drafter frozen  (wired)
  └─ Flow 4: drafter → rollout       (TODO)
```

**Important:** Flows 1-3 ensure that rollout, HS collector, and drafter
frozen modules all reflect the latest actor weights for the **next** RL
step. The drafter trains on hidden states collected from the **current**
step's actor weights.

---

## Why Pre-Norm Matters (Flow 3, verifier_norm)

vLLM captures `last_hidden_states` **before** the final RMSNorm layer
(this is a vLLM implementation detail — the KV connector extracts from
the last layer's output, which is pre-norm). The trainer must apply
`verifier_norm` explicitly before computing target logits:

```
vLLM output:  last_hidden_states (pre-norm)
                    ↓
Drafter:      verifier_norm(last_hidden_states)   ← Flow 3 keeps this in sync
                    ↓
              target_lm_head_weight @ normed_hs   ← Flow 3 keeps this in sync
                    ↓
              target_logits → Forward KL loss
```

This is handled by `FSDPDrafterEngine.prepare_model_inputs()` which
applies `self._verifier_norm` and builds a `LazyTarget` containing
`self._target_lm_head_weight`. Both must match the current actor.
