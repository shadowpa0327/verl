#!/usr/bin/env bash
# Drafter training smoke — close-loop rollout -> HS -> update_drafter (real
# forward/backward/optimizer.step). Target model frozen. Success criteria:
# train/loss_weighted decreases, train/simulated_acc_len increases over N steps.
#
# Override any knob with KEY=VALUE env vars, e.g.:
#   MAX_STEPS=32 BATCH_SIZE=8 ./scripts/run_drafter_training.sh
# Extra Hydra overrides flow through "$@":
#   ./scripts/run_drafter_training.sh actor_rollout_ref.drafter.optimizer_config.lr=1e-4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$SCRIPT_DIR")"
RECIPE_DIR="$VERL_ROOT/recipe/drafter_cotraining"

# ── Defaults (override with env vars) ─────────────────────────────────
VENV_DIR="${VENV_DIR:-$VERL_ROOT/.venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/gsm8k/test.parquet}"

# Draft model config — defaults to the Qwen3-4B Eagle3 config shipped in
# recipe/drafter_cotraining/config/draft_models/ (draft_vocab_size omitted so
# vocab_pruning stays off and we run the LazyTarget path).
DRAFT_CONFIG="${DRAFT_CONFIG:-$RECIPE_DIR/config/draft_models/qwen3_4b_eagle3.json}"

# Qwen3-4B has 36 layers. TorchSpec's "post-layer N" defaults are
# [1, n/2-1, n-4, n-1] = [1, 17, 32, 35]; vLLM's hook fires at the *input* of
# each listed layer (= output of layer N-1), so TorchSpec shifts non-final ids
# by +1 and appends the final layer separately for last_hidden_states. For 36
# layers: [1,17,32,35] → [2,18,33] (35+1=36 dropped) + [35] = [2,18,33,35].
# We pass the already-shifted list directly here. See
# ref/TorchSpec/torchspec/inference/engine/vllm_engine.py:155-181.
AUX_LAYER_IDS="${AUX_LAYER_IDS:-[2,18,33,35]}"

BATCH_SIZE="${BATCH_SIZE:-8}"          # must be multiple of rollout.agent.num_workers
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-512}"
HS_MAX_MODEL_LEN="${HS_MAX_MODEL_LEN:-$ROLLOUT_MAX_MODEL_LEN}"
GPU_MEM="${GPU_MEM:-0.5}"
N_GPUS="${N_GPUS:-2}"

# MAX_STEPS also drives the cosine LR schedule — see §4.7 of the milestone
# plan. Keep >= 16 for a visible loss/acc_len trend.
MAX_STEPS="${MAX_STEPS:-32}"

DRAFTER_LR="${DRAFTER_LR:-1.0e-4}"
DRAFTER_WARMUP_RATIO="${DRAFTER_WARMUP_RATIO:-0.015}"
DRAFTER_CLIP_GRAD="${DRAFTER_CLIP_GRAD:-0.5}"

# ── Sanity ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"; exit 1
fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: data file missing: $f"; exit 1
    fi
done
if [ ! -f "$DRAFT_CONFIG" ]; then
    echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG"; exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "WARN: MODEL_PATH not a local dir — HF download will trigger: $MODEL_PATH"
fi

# Kill stale processes from prior crashed runs.
# - mooncake_master: needs to free :50051 / :8090.
# - VLLM::EngineCore / VLLM::Worker: zombie workers re-advertise phantom
#   Mooncake segments to fresh masters and cause batch_put_from to fail
#   intermittently with code=-800 (TRANSFER_FAIL).
pkill -x mooncake_master  2>/dev/null || true
pkill -x VLLM::EngineCore 2>/dev/null || true
pkill -x VLLM::Worker     2>/dev/null || true
sleep 1

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export HYDRA_FULL_ERROR=1
# Disable torch.compile on the Eagle3 loss kernels — torch.compile + FSDP2
# DTensor-sharded weights + symbolic input shapes blows up during tracing
# ("Cannot call numel() on tensor with symbolic sizes/strides"). Safe to
# disable for the smoke; re-enable once we size up runs.
#export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

python "$RECIPE_DIR/scripts/test_drafter_training.py" \
    data.train_files="['$TRAIN_FILE']" \
    data.val_files="['$VAL_FILE']" \
    data.train_batch_size="$BATCH_SIZE" \
    data.val_batch_size="$BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.drafter.enable=True \
    actor_rollout_ref.drafter.model_config.local_path="$DRAFT_CONFIG" \
    actor_rollout_ref.drafter.optimizer_config.lr="$DRAFTER_LR" \
    actor_rollout_ref.drafter.optimizer_config.lr_warmup_steps_ratio="$DRAFTER_WARMUP_RATIO" \
    actor_rollout_ref.drafter.optimizer_config.clip_grad="$DRAFTER_CLIP_GRAD" \
    actor_rollout_ref.drafter.optimizer_config.total_training_steps="$MAX_STEPS" \
    hs_collector.inference.max_model_len="$HS_MAX_MODEL_LEN" \
    hs_collector.inference.engine_kwargs.vllm.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids="$AUX_LAYER_IDS" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name=drafter_training_smoke \
    trainer.experiment_name=training_smoke \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$MAX_STEPS" \
    trainer.val_before_train=False \
    +micro.max_steps="$MAX_STEPS" \
    "$@"
