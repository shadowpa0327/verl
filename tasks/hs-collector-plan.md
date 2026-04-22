# HS Collector via Colocated Manager — Plan

**Date:** 2026-04-22
**Branch:** `feat/drafter-cotraining`

## What

Replace the hand-rolled `VllmHSCollector` with an `HSCollectorManager` that clones verl's colocated `TeacherModelManager` pattern. Same GPUs as rollout, time-multiplexed via sleep/wake, called sync from the trainer after rollout.

**Naming:** `hs_collector`. Model inside = actor model (not a separate teacher).

## Why

Colocated `TeacherModelManager` already does everything our `VllmHSCollector` was trying to do: spawn replicas, sleep/wake, batch requests through a server manager. Proven in production. Clone it.

No AgentLoop changes. No extra GPUs.

## Changes

### 1. Patch response path

`verl/workers/rollout/vllm_rollout/vllm_async_server.py:520` — after `extract_prompt_logprobs(...)`:

```python
kv_params = getattr(final_res, "kv_transfer_params", None)
if kv_params:
    extra_fields["kv_transfer_params"] = kv_params
```

5 lines. Additive. Teacher path unaffected.

### 2. HSCollectorManager

Clone `verl/experimental/teacher_loop/` → `verl/experimental/hs_collector/` with:

**`hs_collector_model.py`** — clone of `TeacherModelManager`:
- Same init / replica spawn / sleep-on-init pattern
- Replace `compute_logprobs(data)` with `compute_hidden_states(data)`: wake → run `server_manager.compute_hidden_states_batch(data)` → sleep → return `DataProto` (same shape as teacher's return, with non-tensor fields for mooncake metadata)
- Add `update_weights(params)` — fan-out `replica.update_weights(params)` for each replica (teacher doesn't need; we do because HS collector mirrors the live actor)

**`hs_collector_manager.py`** — minimal `AsyncHSCollectorServerManager`:
- `compute_hidden_states_batch(data)`: sends `max_tokens=1` prefill requests, parses `kv_transfer_params` from `TokenOutput.extra_fields`, returns `DataProto` with `non_tensor_batch["hs_mooncake_key" | "hs_shapes" | "hs_dtypes"]`
- No padding, no logprob parsing

### 3. Config

Add minimal `hs_collector` section with two fields:
```yaml
hs_collector:
  model_path: <same as actor>
  inference:         # RolloutConfig
    name: vllm
    engine_kwargs:
      vllm:
        kv_transfer_config: {...}
        speculative_config: {method: extract_hidden_states, ...}
```

### 4. Trainer integration

`verl/trainer/drafter/drafter_ct_ray_trainer.py` — mirror the teacher colocate API exactly:

**`init_workers()`** — instantiate `HSCollectorManager(config=..., resource_pool=...)` (mirror `ray_trainer.py:848`).

**Add guards + colocate method** (parallel to `_should_compute_teacher_colocate` / `_compute_teacher_colocate` at `ray_trainer.py:513-522`):
```python
def _should_compute_hidden_states_colocate(self, batch: DataProto) -> bool:
    return self.use_hs_collector  # colocate-only; no streaming flag needed

def _compute_hidden_states_colocate(self, batch: DataProto) -> DataProto:
    """Collect hidden states after rollout when HS collector and actor are colocated."""
    assert self.hs_collector_manager is not None, "HSCollectorManager is None"
    return self.hs_collector_manager.compute_hidden_states(batch)
```

**`fit()`** — call after `generate_sequences()` (mirror invocation at `ray_trainer.py:1414-1417`):
```python
if self._should_compute_hidden_states_colocate(batch):
    hs_batch = self._compute_hidden_states_colocate(batch)
    self.drafter_data_controller.push_samples_from_dataproto(hs_batch)
```

**`update_weights()`** — pass actor params to HS collector after actor update:
```python
self.hs_collector_manager.update_weights(actor_params)
```

### 5. Standalone test

`scripts/test_hs_collector.py` — launches `HSCollectorManager` standalone, sends prompts, verifies Mooncake contents. Replaces the two scripts below.

### 6. Delete

- `verl/workers/rollout/vllm_rollout/vllm_hs_collector.py`
- `_init_hs_collector()`, `collect_hidden_states()`, `self.hs_collector` block in `update_weights()`, `self.hs_collector` / `self._mooncake_config` attrs in `verl/workers/drafter_workers.py`
- `scripts/test_real_hs_collector.py`
- `scripts/test_hs_collector_verl.py`

Keep `scripts/test_mooncake_store.py` and `scripts/test_vllm_hs_collector.py`.

## Doc updates

| File | Change |
|---|---|
| `claude_docs/weight-sync-flows.md` | Flow 2 rewrite: HS collector owned by `HSCollectorManager` (colocated replicas), synced via `.update_weights(actor_params)` from trainer, not worker. Timing diagram stays the same (sequential post-rollout). |
| `claude_docs/migration-status.md` | HS collector row: `HSCollectorManager`. Close TODO 2 (sleep/wake handled by manager). List deleted files. |
| `claude_docs/project-guide.md` | Architecture + file table: remove `vllm_hs_collector.py`, add `verl/experimental/hs_collector/`. |
| `claude_docs/rfc-drafter-trainer-integration.md` | One-line revision note: "HS collector mechanism changed to `HSCollectorManager` (colocated, post-rollout, sync). Data lifecycle unchanged." |
| `claude_docs/torchspec-to-verl-migration-map.md` | HS Collection row: point to new files. |
| `CLAUDE.md` | Doc table only if file names changed. |

`drafter-target-sharing.md` — no change.

## Open items (handle when hit)

- **`RolloutReplica.update_weights()` API.** Confirm replicas accept weight updates from the actor's `get_per_tensor_param()` format. If not, wire manually (same pattern as existing rollout).
- **Resource pool split.** `TeacherModelManager` uses `split_resource_pool()` to divide the actor's pool among teacher replicas. Confirm this works when teacher == actor model (same size). For our HS collector (identical to actor), TP=1 replicas likely match actor TP.

## Cut points

- **MVP:** 1 + 2 + 5 (patch + manager + test). Standalone proves it works.
- **Full:** 1–6 + docs.

## Success

Standalone test passes → trainer integrates → old files gone → docs coherent → teacher path regression-free.
