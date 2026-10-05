#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/training/run_scion_constants_pipeline.sh <WANDB_ENTITY> <WANDB_PROJECT> [WANDB_GROUP]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <WANDB_ENTITY> <WANDB_PROJECT> [WANDB_GROUP]"
  exit 1
fi

ENTITY="$1"
PROJECT="$2"
GROUP="${3:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv-docker/bin/python"
OUT_DIR="${ROOT_DIR}/outputs/analysis/constant_fits"
CSV_PATH="${OUT_DIR}/scion_step_metrics.csv"

mkdir -p "${OUT_DIR}"

EXPORT_CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/export_scion_trace_wandb.py"
  --entity "${ENTITY}"
  --project "${PROJECT}"
  --output_csv "${CSV_PATH}"
)

if [[ -n "${GROUP}" ]]; then
  EXPORT_CMD+=(--group "${GROUP}")
fi

"${EXPORT_CMD[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/analysis/fit_scion_constants.py" \
  --input_csv "${CSV_PATH}" \
  --output_dir "${OUT_DIR}" \
  --run_col run_id \
  --step_col step \
  --n_layer_col n_layer \
  --n_embd_col n_embd \
  --batch_col batch_size \
  --train_loss_col run/train_loss \
  --dual_grad_col stats/grad_norm_nuc_power_1 \
  --l_proxy_col stats/local_smooth_spec \
  --rho_proxy_col rho/rho_over_averaged_norms \
  --tail 100 \
  --mu_loss_max 5.0 \
  --min_points_mu 20


echo "Done. Outputs in: ${OUT_DIR}"
