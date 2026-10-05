#!/usr/bin/env bash
#SBATCH --job-name=delete_old_rho_remeasure_wandb
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ab035
#SBATCH --output=logs/debug/delete_old_rho_remeasure_wandb_%j.out
#SBATCH --error=logs/debug/delete_old_rho_remeasure_wandb_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ "${DELETE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing to delete W&B runs without DELETE_CONFIRMED=1." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p logs/debug

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"

srun --environment=uni-d2 --ntasks=1 "${REPO_ROOT}/venv-docker/bin/python" - <<'PY'
import wandb

entity = "SwissAI_DLM"
project = "SwissAI_Extending_Budget"
expected_group = "scion_rho_sigma_global_time_remeasure_compact_L12H768_S2048"
run_ids = [
    "rho_global_t_compact_remeasure-L12H768-B64-S4",
    "rho_global_t_compact_remeasure-L12H768-B128-S4",
    "rho_global_t_compact_remeasure-L12H768-B256-S4",
    "rho_global_t_compact_remeasure-L12H768-B512-S4",
    "rho_global_t_compact_remeasure-L12H768-B1024-S4",
    "rho_global_t_compact_remeasure-L12H768-B2048-S4",
    "rho_global_t_compact_remeasure-L12H768-B4096-S4",
]

api = wandb.Api(timeout=60)
runs = []
for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    if run.group != expected_group:
        raise RuntimeError(
            f"Safety check failed for {run_id}: group={run.group!r}, "
            f"expected {expected_group!r}")
    runs.append(run)

for run in runs:
    print(f"Deleting {run.path} name={run.name!r} group={run.group!r}", flush=True)
    run.delete()

print(f"Deleted {len(runs)} obsolete checkpoint-remeasurement runs.", flush=True)
PY
