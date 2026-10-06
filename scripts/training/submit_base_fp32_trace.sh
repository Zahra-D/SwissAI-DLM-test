#!/usr/bin/env bash
set -euo pipefail

# L / mu trace of the base model (L12H768) with fp32 master weights, to compare
# with the bf16 trace bstbase-L12H768-gbs256-5Ksteps-m32e250-beta6e-5-s4.
#
# Same setup as that trace: GBS 256, beta 6e-5, alpha 0.06, 5000 steps, no
# warmup/warmdown, no validation, full packed cache, seed 4.  Differences:
#   * model.torch_dtype=fp32 + training.forward_autocast_dtype=bf16 (fp32
#     weights / optimizer state, bf16 compute); micro-batch 4 x accumulation 2
#     on 8 nodes (fp32 weights do not fit micro-batch 8);
#   * L only every TRACE_EVERY=50 steps (still consecutive-step differences),
#     mu inputs (dual gradient norm, loss) every step;
#   * rho/sigma collector off (NOISE=1 turns it on: m=32 every 250 steps).

DRY_RUN="${DRY_RUN:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
BETA="${BETA:-6e-5}"
TRACE_EVERY="${TRACE_EVERY:-50}"
NOISE="${NOISE:-0}"
GROUP="scion_bst_base_constants_L12H768_S2048_5k_fp32weights"
TAG="fp32w-L${TRACE_EVERY}"
[[ "${NOISE}" == "1" ]] && TAG+="-m32e250"

DRY_RUN="${DRY_RUN}" \
PARTITION="${PARTITION:-normal}" \
TIME_LIMIT="${TIME_LIMIT:-04:00:00}" \
MODEL_DTYPE=fp32 \
FORWARD_AUTOCAST=bf16 \
GBS=256 \
LOCAL_BATCH=4 \
ACCUMULATION_STEPS=2 \
AUTO_RESUME=1 \
WANDB_RUN_ID="bstbase-L12H768-gbs256-5Ksteps-${TAG}-beta${BETA//./p}-s4" \
WANDB_RESUME=allow \
DATA_CACHE_DIR="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok" \
REQUIRE_SUBSET_MANIFEST=0 \
MAX_STEPS_OVERRIDE=5000 \
BUDGET_LABEL="5Ksteps_${TAG//-/_}" \
WARMDOWN_ITERS=0 \
TRAIN_LOG_EVERY_N_STEPS=1 \
VAL_CHECK_INTERVAL=1000000 \
LIMIT_VAL_BATCHES=0 \
VALIDATE_AFTER_TRAINING=false \
CHECKPOINT_EVERY=500 \
TRACE_ENABLED=true \
TRACE_EVERY="${TRACE_EVERY}" \
TRACE_COLLECT_NOISE_STATS="$([[ "${NOISE}" == "1" ]] && echo true || echo false)" \
TRACE_NOISE_STATS_EVERY=250 \
TRACE_NOISE_MIN_SAMPLES=3 \
TRACE_M=32 \
TRACE_PROGRESS_EVERY=8 \
PAIR_SPECS_OVERRIDE="${BETA}:0.06" \
EXPERIMENT_NAME="${GROUP}_gbs256" \
WANDB_PROJECT=SwissAI_Scion_B3_HP_Tuning \
WANDB_GROUP="${GROUP}" \
bash "${REPO_ROOT}/scripts/training/submit_scion_b3_fixed_init_no_rho_gbs64.sh"
