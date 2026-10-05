#!/usr/bin/env bash
set -euo pipefail

# Run the common-beta/momentum child grid at GBS 64, 128, 256, and 512. The
# child derives steps, ranks, and nodes independently. DRY_RUN defaults to 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHILD_SCRIPT="${SCRIPT_DIR}/submit_scion_b3_equal_lr_base_hparam_grid.sh"
GLOBAL_BATCH_SIZES=(${GLOBAL_BATCH_SIZES:-64 128 256 512})
DRY_RUN="${DRY_RUN:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_b3_equalLR_fixedSetC_betaMom_grid_L12H768_S2048_2B_GBS64-512}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

if [[ ! -f "${CHILD_SCRIPT}" ]]; then
  echo "Missing child launcher: ${CHILD_SCRIPT}" >&2
  exit 1
fi

echo "Batch-size sweep: ${GLOBAL_BATCH_SIZES[*]}"
echo "Dry run: ${DRY_RUN}"
echo "Shared experiment/group: ${EXPERIMENT_NAME}"

for GBS_VALUE in "${GLOBAL_BATCH_SIZES[@]}"; do
  if ! [[ "${GBS_VALUE}" =~ ^[0-9]+$ ]] || (( GBS_VALUE <= 0 )); then
    echo "Invalid global batch size: ${GBS_VALUE}" >&2
    exit 1
  fi
  echo "Launching child grid for GBS=${GBS_VALUE}"
  GBS="${GBS_VALUE}" \
  DRY_RUN="${DRY_RUN}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  bash "${CHILD_SCRIPT}"
done

echo "Processed ${#GLOBAL_BATCH_SIZES[@]} batch sizes."
