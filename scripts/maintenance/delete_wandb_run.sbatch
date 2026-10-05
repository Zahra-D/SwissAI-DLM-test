#!/bin/bash
#SBATCH --job-name=wandb_baseline_cleanup
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/wandb_baseline_cleanup_%j.out
#SBATCH --error=logs/wandb_baseline_cleanup_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
srun --environment=uni-d2 --ntasks=1 \
  "${SLURM_SUBMIT_DIR}/venv-docker/bin/python" -u \
  scripts/maintenance/delete_wandb_run.py \
  --run SwissAI_DLM/SwissAI_Scion_B3_HP_Tuning/dt5v9lgi \
  --expected-source SwissAI_DLM/SwissAI_Scaling_Law/ak4wwm6p \
  --expected-rows 16541
