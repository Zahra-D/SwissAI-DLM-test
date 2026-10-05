#!/bin/bash
#SBATCH --job-name=nemotron_2b_subset
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/nemotron_2b_subset_%j.out
#SBATCH --error=logs/nemotron_2b_subset_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
SOURCE_CACHE="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok"
OUTPUT_CACHE="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok-2b-seed4"
TRAIN_NAME="nemotron-cc-pretok-train_train_bs2048_unwrapped_eosFalse_specialFalse.dat"
VALID_NAME="nemotron-cc-pretok-valid_validation_bs2048_unwrapped_eosFalse_specialFalse.dat"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export JOB_TMPDIR="${SLURM_TMPDIR:-/tmp/SwissAI-DLM-${USER:-unknown}-${SLURM_JOB_ID:-manual}}"
export TMPDIR="${JOB_TMPDIR}"
mkdir -p "${TMPDIR}" "${OUTPUT_CACHE}"

echo "Source: ${SOURCE_CACHE}/${TRAIN_NAME}"
echo "Output: ${OUTPUT_CACHE}/${TRAIN_NAME}"
echo "Rows: 976512; sequence length: 2048; tokens: 1999896576"
echo "Selection seed: 4"

"${VENV_PYTHON}" -u scripts/data/build_reproducible_packed_subset.py \
  --source "${SOURCE_CACHE}/${TRAIN_NAME}" \
  --output-cache-dir "${OUTPUT_CACHE}" \
  --output-train-name "${TRAIN_NAME}" \
  --validation-source "${SOURCE_CACHE}/${VALID_NAME}" \
  --validation-name "${VALID_NAME}" \
  --num-sequences 976512 \
  --sequence-length 2048 \
  --seed 4 \
  --max-shard-size 1GB
