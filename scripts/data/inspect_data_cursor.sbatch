#!/usr/bin/env bash
#SBATCH --job-name=inspect_data_cursor
#SBATCH --account=ab035
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/inspect_data_cursor_%j.out
#SBATCH --error=logs/inspect_data_cursor_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
srun --environment=uni-d2 \
  "${REPO_ROOT}/venv-docker/bin/python" \
  scripts/data/inspect_data_cursor.py \
  "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the checkpoint to inspect}"
