#!/bin/bash
#SBATCH --job-name=smoke_subset_builder
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/smoke_subset_builder_%j.out
#SBATCH --error=logs/smoke_subset_builder_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

DATA_ROOT="${SCRATCH}/SwissAI-DLM-data"
VALID_NAME="nemotron-cc-pretok-valid_validation_bs2048_unwrapped_eosFalse_specialFalse.dat"
VALID_SOURCE="${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok/${VALID_NAME}"
JOB_TMPDIR="${SLURM_TMPDIR:-/tmp/SwissAI-DLM-${USER:-unknown}-${SLURM_JOB_ID}}"
SMOKE_ROOT="${JOB_TMPDIR}/subset-builder-smoke"
mkdir -p "${JOB_TMPDIR}"

srun --environment=uni-d2 --ntasks=1 \
  "${SLURM_SUBMIT_DIR}/venv-docker/bin/python" -u \
  scripts/data/build_reproducible_packed_subset.py \
  --source "${VALID_SOURCE}" \
  --output-cache-dir "${SMOKE_ROOT}" \
  --output-train-name smoke-train.dat \
  --validation-source "${VALID_SOURCE}" \
  --validation-name smoke-validation.dat \
  --num-sequences 128 \
  --sequence-length 2048 \
  --seed 4 \
  --max-shard-size 16MB

echo "SUBSET BUILDER SMOKE PASSED (the builder reloads and verifies its output)"
