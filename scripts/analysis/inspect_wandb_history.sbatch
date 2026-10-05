#!/bin/bash
#SBATCH --job-name=inspect_wandb_history
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/inspect_wandb_history_%j.out
#SBATCH --error=logs/inspect_wandb_history_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
WANDB_RUN_PATH="${WANDB_RUN_PATH:-SwissAI_DLM/SwissAI_Scion_B3_HP_Tuning/u03efu3k}"

srun --environment=uni-d2 --ntasks=1 \
  "${SLURM_SUBMIT_DIR}/venv-docker/bin/python" -u \
  scripts/analysis/inspect_wandb_history.py \
  "api:${WANDB_RUN_PATH}"
