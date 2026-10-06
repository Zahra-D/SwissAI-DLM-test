#!/usr/bin/env bash
set -euo pipefail

# Matched fp32-weight traces of the base (L12H768) and 0.5B (L18H1536) models
# to measure the L (and mu) ratio between them with correct step sizes.
#
# Same settings as the bf16 constants traces they are compared with
# (GBS 512, beta 8e-5, alpha 0.06, no warmup/warmdown, no validation, full
# packed cache, seed 4), but:
#   * fp32 weights / optimizer state with bf16 compute
#     (model.torch_dtype=fp32, training.forward_autocast_dtype=bf16);
#   * 3000 steps: in the bf16 traces the L ratio is within ~1% of its
#     4000-step value from step ~2500 on;
#   * L every 50 steps (consecutive-step differences), mu inputs every step,
#     rho/sigma off.
#
# MODELS: space-separated name:n_blocks:hidden:micro_batch:accumulation:act_ckpt

DRY_RUN="${DRY_RUN:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
GBS="${GBS:-512}"
BETA="${BETA:-8e-5}"
STEPS="${STEPS:-3000}"
GROUP="scion_bst_fp32_L_ratio_traces_gbs${GBS}"

for SPEC in ${MODELS:-base:12:768:4:2:false 0p5b:18:1536:8:1:true}; do
  IFS=: read -r NAME NB HS MB ACC CKPT <<< "${SPEC}"
  echo "=== ${NAME}: L${NB} H${HS}, micro-batch ${MB} x accumulation ${ACC}"
  DRY_RUN="${DRY_RUN}" \
  PARTITION="${PARTITION:-normal}" \
  TIME_LIMIT="${TIME_LIMIT:-04:00:00}" \
  N_BLOCKS="${NB}" \
  HIDDEN_SIZE="${HS}" \
  INTERMEDIATE_SIZE="$((4 * HS))" \
  N_HEADS="$((HS / 64))" \
  ACTIVATION_CHECKPOINTING="${CKPT}" \
  MODEL_DTYPE=fp32 \
  FORWARD_AUTOCAST=bf16 \
  GBS="${GBS}" \
  LOCAL_BATCH="${MB}" \
  ACCUMULATION_STEPS="${ACC}" \
  AUTO_RESUME=1 \
  WANDB_RUN_ID="fp32Lratio-L${NB}H${HS}-gbs${GBS}-beta${BETA//./p}-${STEPS}steps-s4" \
  WANDB_RESUME=allow \
  DATA_CACHE_DIR="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok" \
  REQUIRE_SUBSET_MANIFEST=0 \
  MAX_STEPS_OVERRIDE="${STEPS}" \
  BUDGET_LABEL="${STEPS}steps_fp32w_L50" \
  WARMDOWN_ITERS=0 \
  TRAIN_LOG_EVERY_N_STEPS=1 \
  VAL_CHECK_INTERVAL=1000000 \
  LIMIT_VAL_BATCHES=0 \
  VALIDATE_AFTER_TRAINING=false \
  CHECKPOINT_EVERY=500 \
  TRACE_ENABLED=true \
  TRACE_EVERY=50 \
  TRACE_COLLECT_NOISE_STATS=false \
  PAIR_SPECS_OVERRIDE="${BETA}:0.06" \
  EXPERIMENT_NAME="${GROUP}_${NAME}" \
  WANDB_PROJECT=SwissAI_Scion_B3_HP_Tuning \
  WANDB_GROUP="${GROUP}" \
  bash "${REPO_ROOT}/scripts/training/submit_scion_b3_fixed_init_no_rho_gbs64.sh"
done
