# TorchSpec → verl Migration Map

Reference for porting more TorchSpec features into the verl recipe.
Most drafter code now lives in `recipe/drafter_cotraining/` and the
file paths there mirror TorchSpec's layout intentionally.

**Reference source tree:** `/root/verl/ref/TorchSpec/` (untracked
local checkout).

---

## EAGLE3 — what it is

EAGLE3 speculative decoding drafter co-training. A tiny drafter
(~2% of target params) is trained to predict the target's output
distribution via Forward KL distillation. At inference, the drafter
speculatively continues from the verifier's just-emitted token; the
verifier then accepts/rejects the speculated tokens.

**Trainable components in the draft:** `fc` (Linear 3·D → D),
`midlayer` (one decoder layer), `norm` (RMSNorm), `lm_head` (Linear D
→ V). For a 7B target this is ~140M params.

**Two lm_heads, one trainable:** see `weight-sync-flows.md` →
"Parameter inventory".

---

## Conceptual mapping

| Concept | TorchSpec | recipe/drafter_cotraining |
|---|---|---|
| Async controller | `controller/training_controller.py` (Ray actor, 4 FIFOs) | `controller.py` (in-process driver, 2 stores + mesh dispatch) |
| Outer loop | `controller/loop.py::training_loop` (async dispatch + retry) | `ray_trainer.py::RayDrafterCTPPOTrainer.fit()` (synchronous, folded into RL loop) |
| Trainer entry | `train_entry.py` | `main_drafter_ct.py::DrafterCTTaskRunner` |
| Eagle3 model + TTT loop | `models/eagle3.py` (`Eagle3Model`) | `eagle3/eagle3_model.py` — same 7-step TTT loop |
| Forward KL loss kernel | `models/ops/loss.py` (`compiled_forward_kl_loss[_from_hs]`) | `eagle3/ops/loss.py` — identical |
| Loss-mask helpers | `models/ops/loss_mask.py` (assistant-header Numba scan) | `eagle3/ops/loss_mask.py` — copy; not used in our RL flow (we have explicit prompt/response_len) |
| Draft model arch | `models/draft/{auto,base,llama3_eagle}.py` | `eagle3/draft/` — same files |
| Trainer init | `training/eagle3_trainer.py::init_model` | `drafter_engine.py::FSDPDrafterEngine._build_module` + `initialize` + `_load_target_frozen_weights` |
| Forward step | `training/eagle3_trainer.py::_forward` | `drafter_engine.py::prepare_model_inputs` (with the `padding(..., left=False)` shift) + `Eagle3Model.forward` |
| 0.8^i backward + accumulation | `training/eagle3_trainer.py::_backward` | `fsdp_workers.py::_drafter_train_step` |
| Metric aggregation | `training/eagle3_trainer.py::_aggregate_metrics` | `fsdp_workers.py::_aggregate_drafter_metrics` |
| Mooncake KV store | `transfer/mooncake/{eagle_store,store,buffers,helpers}.py` | `mooncake/{eagle_store,store,buffers,helpers}.py` — copies |
| vLLM HS connector | `inference/engine/vllm_engine.py` + `mooncake_hidden_states_connector.py` | `mooncake/hidden_states_connector.py` (KVConnectorBase_V1 implementation) + `hs_collector/` (replica manager) |
| HS collection topology | Standalone Ray engine pool | Colocated vLLM replicas, time-multiplexed via sleep/wake |
| Training-with-decode loop | `controller/loop.py::_maybe_sync_draft_weights` | **TODO 4** (parent verl rollout-side support not yet wired) |
| Optimizer | Custom `BF16Optimizer` (fp32 master weights) | Standard PyTorch AdamW. Numerical drift small for typical run lengths. |
| Sharding | `_composable.replicate` (DDP, default for Eagle3) or FSDP2 `fully_shard` (`FULL_SHARD`, only `dflash` recipes) | FSDP1 `FULL_SHARD` + `use_orig_params=True` — params stay plain `Tensor` so the loss kernel doesn't hit `mixed Tensor and DTensor` |

---

## Architecture shifts vs TorchSpec

1. **Async → sync.** TorchSpec's `AsyncInferenceManager` and
   per-engine event loop are eliminated; verl's RayPPOTrainer drives
   workers via RPC inside one synchronous outer loop. Levels 1 & 2 of
   the data pipeline are in-process on the driver; Level 3 dispatch
   uses verl's mesh-based `DataProto.chunk()`.

2. **Colocation instead of separate processes.** HS collector vLLM
   replicas share GPUs with the actor's rollout vLLM and time-
   multiplex via `sleep`/`wake_up` (`HSCollectorManager` clones
   `TeacherModelManager`).

3. **Frozen weight sourcing.** TorchSpec syncs frozen modules
   directly from the (fixed) target. verl loads them from
   `target_model_path` at init (with a `tie_word_embeddings`
   fallback) — see `weight-sync-flows.md` Flow 3. Re-sync after
   `update_actor()` is the missing piece pending FSDP-aware actor
   gather.

4. **Input shift baked into `prepare_model_inputs`.** TorchSpec's
   `_forward` applies `padding(input_ids, left=False)` and
   `padding(last_hidden_states, left=False)`. We do the same to
   align with Eagle3 inference semantics
   `(aux[t], token[t+1]) → predict token[t+2]`.

5. **vLLM aux layer ID convention.** TorchSpec's "post-layer N" IDs
   need a +1 shift before vLLM (whose hook fires at the *input* of
   each listed layer). Final layer is appended separately for
   `last_hidden_states`. Default for Qwen3-4B (36 layers):
   `[1,17,32,35]` (TorchSpec semantic) → `[2,18,33,35]` (vLLM).

---

## When porting more TorchSpec features

- Live source: `/root/verl/ref/TorchSpec/`.
- Map a TorchSpec file to its recipe equivalent via the table above.
- For new files, follow the recipe layout (don't add new
  `verl/.../mooncake/` etc. — that's the deleted in-tree mirror).
- See `weight-sync-flows.md` for parameter / data flows.
- See `migration-status.md` for current state + open TODOs.
- See `drafter-design.md` for the as-built architecture and rationale.
