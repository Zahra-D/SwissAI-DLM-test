#!/usr/bin/env bash
set -euo pipefail

# FW-stepsize sweep of the base model (L12H768) with fp32 master weights.
#
# Identical to the tuned base run (GBS 256, 2B tokens on the seed-4 subset,
# 3814 steps, alpha 0.06, 28% warmdown, validation every 250 steps on half the
# validation set + after training, seed 4) except for the precision:
# model.torch_dtype=fp32 with training.forward_autocast_dtype=bf16, i.e. fp32
# weights / optimizer state and bf16 compute.  fp32 weights do not fit
# micro-batch 8, so micro-batch 4 is used: 8 nodes x accumulation 2 (the first
# fp32 run, beta 6e-5, used 16 nodes x accumulation 1; same global batch).
#
# The beta 6e-5 point already exists (base-L12H768-2B-gbs256-beta6e-5-fp32weights-s4);
# the default list adds a ~2x grid from 1.5e-5 to 4.8e-4 around it.

DRY_RUN="${DRY_RUN:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for BETA in ${BETAS:-1.5e-5 3e-5 9e-5 1.2e-4 2.4e-4 4.8e-4}; do
  echo "=== beta ${BETA}"
  DRY_RUN="${DRY_RUN}" \
  PARTITION="${PARTITION:-normal}" \
  TIME_LIMIT="${TIME_LIMIT:-02:30:00}" \
  MODEL_DTYPE=fp32 \
  FORWARD_AUTOCAST=bf16 \
  GBS=256 \
  LOCAL_BATCH=4 \
  ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}" \
  REQUIRE_SUBSET_MANIFEST=0 \
  PAIR_SPECS_OVERRIDE="${BETA}:0.06" \
  BUDGET_LABEL=2B_fp32w \
  AUTO_RESUME=1 \
  WANDB_RUN_ID="base-L12H768-2B-gbs256-beta${BETA//./p}-fp32weights-s4" \
  WANDB_RESUME=allow \
  EXPERIMENT_NAME=scion_b3_base_fp32_master_weights_ab_gbs256 \
  WANDB_PROJECT=SwissAI_Scion_B3_HP_Tuning \
  WANDB_GROUP=scion_b3_base_fp32_master_weights_ab \
  bash "${REPO_ROOT}/scripts/training/submit_scion_b3_fixed_init_no_rho_gbs64.sh"
done
