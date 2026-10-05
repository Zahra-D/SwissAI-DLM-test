#!/usr/bin/env bash
set -euo pipefail

# Full-horizon 2x token-budget LR grid with gradient clipping disabled.
# Names and storage paths are isolated from the historical clipped sweep.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET_MULTIPLIER=2
export GBS=416
export LR_GRID_CSV="0.01152,0.0144,0.018,0.0225,0.028125"
export GRADIENT_CLIP_VAL=0.0
export RUN_SUFFIX="_no_gradient_clipping"
export EXPERIMENT_NAME="scion_set_c_2x_gbs416_lr_grid_no_gradient_clipping"
export WANDB_GROUP="scion_base_model_2x_lr_grid_no_gradient_clipping_L12H768_S2048"
export TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
exec "${SCRIPT_DIR}/submit_scion_budget_lr_grid_common.sh" "$@"
