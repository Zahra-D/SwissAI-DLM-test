#!/usr/bin/env bash
#SBATCH --job-name=plot_scion_budget_lr_fig5
#SBATCH --partition=debug
#SBATCH --account=ab035
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/scion_budget_lr_fig5_plot_%j.out
#SBATCH --error=logs/scion_budget_lr_fig5_plot_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${SCRATCH}/SwissAI-DLM-data"
export WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
export WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Extending_Budget}"
export WANDB_GROUP="${WANDB_GROUP:-scion_base_model_token_budget_extension_lr_grid_L12H768_S2048}"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
export PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${DATA_ROOT}/outputs/extending_budget/figures/scion_budget_lr_fig5}"
export JOB_TMPDIR="${SLURM_TMPDIR:-/tmp/SwissAI-DLM-${USER:-unknown}-${SLURM_JOB_ID:-manual}}"
export TMPDIR="${JOB_TMPDIR}"
export MPLCONFIGDIR="${JOB_TMPDIR}/matplotlib"

mkdir -p "${REPO_ROOT}/logs" "${WANDB_DIR}" "${PLOT_OUTPUT_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

srun --environment=uni-d2 \
  "${REPO_ROOT}/venv-docker/bin/python" -u \
  scripts/analysis/plot_scion_extending_budget_lr.py \
  --entity "${WANDB_ENTITY}" \
  --project "${WANDB_PROJECT}" \
  --group "${WANDB_GROUP}" \
  --seed "${SEED:-4}" \
  --require-seeds "${REQUIRE_SEEDS:-1}" \
  --window-steps 500 \
  --output-dir "${PLOT_OUTPUT_DIR}"
