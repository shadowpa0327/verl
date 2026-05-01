
  # claude_docs/research/

  Reference notes carried over from the team Obsidian vault.

  | File | What |
  |---|---|
  | `drafter-micro-batching-refactor-plan.md` | **The active implementation plan** for FSDP2 + micro-batching. Phases A-E with verification commands. |
  | `lazytarget-vs-precomputed-memory-analysis.md` | Justifies dropping the LazyTarget loss path. Memory crossover analysis at varying mask densities. |
  | `eagle3-comparison-speculators-vs-torchspec.md` | Side-by-side of the two reference Eagle3 trainers. |
  | `fsdp-sharding-in-speculators.md` | Speculators' FSDP2 wrap pattern (per-block + root). |