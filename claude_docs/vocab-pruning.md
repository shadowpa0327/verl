# Draft-Vocabulary Pruning

End-to-end guide for activating vocab pruning in the EAGLE drafter
co-training pipeline. The draft LM head is shrunk to the top-K most
frequent tokens (typically 32K out of 152K), giving a ~5× smaller
output projection at <2% loss of training signal.

## Overview

Three things must agree at training time:

1. **Mapping file** (`.pt`) — `{t2d: bool[V_target], d2t: int64[V_draft]}`.
   Built once from a training corpus by counting supervised tokens.
2. **Draft model template** (JSON) — declares `draft_vocab_size`. Without
   this, the model has no `t2d`/`d2t` buffers and the engine load fails.
3. **Recipe config** (YAML) — points the engine at both the mapping file
   and the draft template.

When all three are set, the engine:
- Writes the mapping into the draft model's persistent buffers (so it
  ships with checkpoints and downstream inference engines see it).
- Surfaces `t2d` to `prepare_model_inputs`, which slices target logits
  from `[B,T,V_target]` to `[B,T,V_draft]` and builds a `position_mask`
  zeroing positions where the verifier-argmax falls outside V_draft.

When **any** of the three is missing, training falls back to full V_target
(no behavioral change vs. a non-pruning run).

## Step 1 — Build the mapping file

The CLI reads canonical pretrain parquet (schema: `id`, `conversations`),
applies the same chat template + assistant-loss-mask logic the trainer
uses, counts supervised tokens, and writes a `.pt`.

```bash
python -m recipe.drafter_cotraining.scripts.data_preprocess.build_vocab_mapping \
  --data_files ~/data/qwen3_8b_eagle3_10k/train.parquet \
  --tokenizer Qwen/Qwen3-8B \
  --chat_template qwen \
  --max_length 4096 \
  --target_vocab_size 151936 \
  --draft_vocab_size 32000 \
  --output ~/cache/vocab_mapping/qwen3_8b_32k.pt
```

Output ends with the **top-K frequency coverage** — the headline metric:

```
top 32000 token frequency ratio: 98.75%
Saved vocab mapping to: ~/cache/vocab_mapping/qwen3_8b_32k.pt
```

Notes:
- `--target_vocab_size` should match the *model's* `vocab_size`, not
  `tokenizer.vocab_size` — Qwen3-8B uses 151936 (padded), not 151643.
- `--max_length` should match `data.max_seq_length` in your trainer
  config so the mask covers the same positions.
- `--max_samples N` caps the scan if you want a quick draft mapping.
- One pass over 10K rows takes ~5 min on a single CPU (tokenizer-bound).

## Step 2 — Create the draft template JSON

Minimal template — only `draft_vocab_size` is mandatory; every other
architecture field is auto-derived from `target_model_path` by
`generate_draft_model_config()` (`eagle3/draft/auto.py`).

```jsonc
// ~/cfg/qwen3_8b_draft_32k.json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "draft_vocab_size": 32000
}
```

Without this template, `auto.py` defaults `draft_vocab_size` to the full
`vocab_size` and `LlamaForCausalLMEagle3` skips the `register_buffer`
calls — the engine then has no `t2d`/`d2t` to copy into and the load
fails fast (`drafter.model_config.vocab_mapping_path is set but the
draft model has no t2d/d2t buffers`).

`draft_vocab_size` here MUST equal `--draft_vocab_size` from step 1. The
engine sanity-checks the buffer shapes against the file and rejects a
mismatch.

## Step 3 — Wire into the recipe config

Two new YAML fields, both under
`actor_rollout_ref.drafter.model_config`:

| Field | Purpose |
|---|---|
| `local_path` | Path to the JSON template from step 2 |
| `vocab_mapping_path` | Path to the `.pt` from step 1 |

Either set them in `config/drafter_ct_trainer.yaml` or override via Hydra
on the command line:

```bash
recipe/drafter_cotraining/scripts/run_qwen3_8b_eagle3_pretrain.sh \
  actor_rollout_ref.drafter.model_config.local_path=$HOME/cfg/qwen3_8b_draft_32k.json \
  actor_rollout_ref.drafter.model_config.vocab_mapping_path=$HOME/cache/vocab_mapping/qwen3_8b_32k.pt
```

At init, look for these log lines from rank 0 to confirm activation:

```
Loaded target-frozen weights from .../Qwen3-8B: lm_head=(151936, 4096), verifier_norm=(4096,)
Loaded vocab mapping from .../qwen3_8b_32k.pt: V_target=151936, V_draft=32000
```

## What changes at training time

| Before | After |
|---|---|
| `lm_head: Linear(4096, 151936)` — 622M params | `lm_head: Linear(4096, 32000)` — 131M params |
| `target_p: bf16[B, T+L, 151936]` resident | `target_p: bf16[B, T+L, 32000]` resident |
| `position_mask = None` | `position_mask = (verifier_argmax in V_draft) ∧ loss_mask` |

The loss kernel automatically picks up the smaller `V_draft` axis on
both sides (draft logits, target probs) — no manual changes needed.

`enable_lazy_target` is a no-op once pruning is on; the precomputed
path is forced because the lazy kernel can't slice by t2d.

## Resume / checkpointing

The `t2d`/`d2t` buffers are persistent on the draft model, so they're
saved into both:
- The FSDP-sharded checkpoint (`model_world_size_*_rank_*.pt`) — picked
  up automatically by `load_checkpoint`.
- The HF export (`huggingface/model.safetensors`) — readable by SGLang
  for spec decoding inference.

When resuming, the `vocab_mapping_path` from the YAML still loads first
(in `initialize`), then the FSDP shards overwrite the buffers. They
should be identical, so the load is idempotent. If you change the
mapping mid-run, re-do step 1 and clear the resume checkpoint —
mismatched buffers between a saved drafter and a new mapping would
silently misroute logits.

## Troubleshooting

**"draft model has no t2d/d2t buffers"** — the template JSON wasn't
loaded, or it's missing `draft_vocab_size`. Verify
`drafter.model_config.local_path` resolves to a file with
`"draft_vocab_size": <K>` < `vocab_size`.

**"Vocab mapping shape mismatch"** — `--draft_vocab_size` from step 1
disagrees with `draft_vocab_size` in the JSON template, or
`--target_vocab_size` doesn't match the target model's `vocab_size`.
Rebuild step 1 with matching values.

**Coverage ratio < 95%** — your `--draft_vocab_size` is too small for
the corpus, or the corpus is OOD vs. the deployment distribution.
Either bump `--draft_vocab_size` or rebuild the mapping on a corpus
closer to your inference workload.
