# Agent Instructions for verl

> See [AGENTS.md](./AGENTS.md) for upstream contribution policy (duplicate checks, commit messages, accountability).

---

## Project Guide — EAGLE Drafter Co-Training

See **[`claude_docs/project-guide.md`](./claude_docs/project-guide.md)** for the full project context:

- Active branch: `feat/drafter-cotraining`
- Architecture: ActorRolloutRefDrafterWorker with vLLM HS collector + Mooncake + Eagle3
- Key design decisions, file table, testing commands
- RFC, migration status, and migration map

**Start here** if you're working on the drafter co-training pipeline.

## Key Documents

| Doc | What |
|---|---|
| `claude_docs/project-guide.md` | Full project context — architecture, files, testing |
| `claude_docs/rfc-drafter-trainer-integration.md` | Approved RFC — data lifecycle, dispatch, worker hierarchy |
| `claude_docs/migration-status.md` | File-by-file status, import mappings, verification checklist |
| `claude_docs/torchspec-to-verl-migration-map.md` | TorchSpec → verl mapping, what to copy vs rewrite |
| `claude_docs/workflow-orchestration.md` | Workflow rules for this project |

## Development Notes

- Don't over-engineer — this is research code, can be experimental.
- Don't use too many try-catch unless necessary.
- Use `tasks` to maintain todos.
- No unanimity gate needed — controller pattern guarantees consensus by construction.
- vLLM captures pre-norm last_hidden_states — `compiled_forward_kl_loss` handles this with explicit RMSNorm.
- vLLM 0.18+ required for KV connector API (`KVConnectorBase_V1`).
