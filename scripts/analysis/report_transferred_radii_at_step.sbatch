#!/bin/bash
#SBATCH --job-name=wandb_transferred_at_1k
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/wandb_transferred_at_1k_%j.out
#SBATCH --error=logs/wandb_transferred_at_1k_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
REPORT_STEP="${REPORT_STEP:-1000}"
srun --environment=uni-d2 --ntasks=1 \
  "${SLURM_SUBMIT_DIR}/venv-docker/bin/python" -u \
  scripts/analysis/report_transferred_radii_at_step.py --step "${REPORT_STEP}"
