# Agent Instructions for verl

> See [AGENTS.md](./AGENTS.md) for upstream contribution policy (duplicate checks, commit messages, accountability).

---

## Development Style
See **[`claude_docs/workflow-orchestration.md`](./claude_docs/workflow-orchestration.md)**.

## Project Guide — EAGLE Drafter Pretrain

See **[`claude_docs/project-guide.md`](./claude_docs/project-guide.md)** for the full project context:

- Active branch: `feat/draft-model-train`
- Scope: **pretrain-only** — standalone drafter pretraining (no RL co-training, no verl core changes)
- Architecture: DrafterPretrainWorker + vLLM HS collector + Mooncake + Eagle3
- Key design decisions, file table, testing commands

**Start here** if you're working on the drafter pretrain pipeline.

**Co-training (RL + drafter) is deferred** — see `claude_docs/migration-status.md` §"Deferred: Co-Training" for the plan to re-enable it later.

## Key Documents

| Doc | What |
|---|---|
| `claude_docs/project-guide.md` | Full project context — architecture, files, testing |
| `claude_docs/migration-status.md` | **Current state** — what's done, what's deferred, verification checklist |
| `claude_docs/drafter-design.md` | **As-built design** — data lifecycle, dispatch, worker hierarchy, training-step shape |
| `claude_docs/weight-sync-flows.md` | Weight sync between actor, rollout, HS collector, drafter |
| `claude_docs/workflow-orchestration.md` | Workflow rules for this project |

## Development Notes

- Don't over-engineer — this is research code, can be experimental.
- Don't use too many try-catch unless necessary.
- Use `tasks` to maintain todos.
- **Minimal verl core changes** — all drafter code lives in `recipe/drafter_cotraining/`. Only 2 generic fixes in `vllm_async_server.py`: (1) `kv_transfer_params` propagation for KV connectors, (2) missing `return` on `collective_rpc`. Both are upstreamable as small PRs.
- vLLM captures pre-norm last_hidden_states — `FSDPDrafterEngine.prepare_model_inputs()` applies `verifier_norm` before target construction. The loss kernel's RMSNorm is for the draft model's own norm (separate).
- vLLM 0.18+ required for KV connector API (`KVConnectorBase_V1`).
- `DrafterPretrainWorker` has `self.rollout = None` — it never touches the vLLM rollout engine. Frozen weights are loaded from `target_model_path` on disk, not from a live actor.
