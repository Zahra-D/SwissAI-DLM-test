#!/usr/bin/env bash
set -euo pipefail

# Run the existing B.3 grid against the deterministic physical 2B-token
# subset. The original validation cache is linked read-only into this cache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
SUBSET_CACHE="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok-2b-seed4"
TRAIN_NAME="nemotron-cc-pretok-train_train_bs2048_unwrapped_eosFalse_specialFalse.dat"
MANIFEST="${SUBSET_CACHE}/subset_manifest.json"

if [[ ! -f "${MANIFEST}" || ! -d "${SUBSET_CACHE}/${TRAIN_NAME}" ]]; then
  echo "The verified 2B subset is not ready: ${SUBSET_CACHE}" >&2
  echo "First submit: sbatch scripts/build_nemotron_2b_reproducible_subset.sbatch" >&2
  exit 2
fi

export DATA_CACHE_DIR="${SUBSET_CACHE}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export LAZY_SPAWN_DATASET="true"
export NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_b3_equalLR_fixedSetC_betaMom_L12H768_S2048_2BsubsetSeed4_GBS${GBS:-256}}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

exec bash "${SCRIPT_DIR}/submit_scion_b3_equal_lr_base_hparam_grid.sh"
