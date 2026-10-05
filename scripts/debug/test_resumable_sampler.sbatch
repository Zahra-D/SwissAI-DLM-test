#!/usr/bin/env bash
#SBATCH --job-name=test_resumable_sampler
#SBATCH --account=ab035
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/test_resumable_sampler_%j.out
#SBATCH --error=logs/test_resumable_sampler_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
srun --environment=uni-d2 \
  "${REPO_ROOT}/venv-docker/bin/python" -m py_compile \
  src/discrete_diffusion/data/resumable.py \
  src/discrete_diffusion/data/lazy_dataset.py \
  src/discrete_diffusion/data/loaders.py \
  src/discrete_diffusion/train.py \
  src/discrete_diffusion/algorithms/base.py
srun --environment=uni-d2 \
  "${REPO_ROOT}/venv-docker/bin/python" \
  scripts/debug/test_resumable_sampler.py
srun --environment=uni-d2 \
  "${REPO_ROOT}/venv-docker/bin/python" \
  scripts/debug/test_lazy_spawn_dataset.py
