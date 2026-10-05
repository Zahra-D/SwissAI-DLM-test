#!/bin/bash
#SBATCH --job-name=wandb_baseline_copy
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --output=logs/wandb_baseline_copy_%j.out
#SBATCH --error=logs/wandb_baseline_copy_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

SOURCE_RUN="${SOURCE_RUN:-SwissAI_DLM/SwissAI_Scaling_Law/ak4wwm6p}"
DEST_PROJECT="${DEST_PROJECT:-SwissAI_Scion_B3_HP_Tuning}"
DEST_GROUP="${DEST_GROUP:-scion_b3_equalLR_transferredSetC_subset2Bseed4_betaMom_grid_L12H768_S2048_2B_GBS64}"
DEST_NAME="${DEST_NAME:-baseline__SwissAI_Scaling_Law__ak4wwm6p__complete_history}"

srun --environment=uni-d2 --ntasks=1 \
  "${SLURM_SUBMIT_DIR}/venv-docker/bin/python" -u \
  scripts/analysis/copy_wandb_run_as_baseline.py \
  --source "${SOURCE_RUN}" \
  --project "${DEST_PROJECT}" \
  --group "${DEST_GROUP}" \
  --name "${DEST_NAME}"
