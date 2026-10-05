#!/usr/bin/env bash
set -euo pipefail

# Validates last.ckpt from every run under a given outputs directory, one
# run after another, on a single GPU. Runs scripts/training/validate_checkpoints_single_gpu.py,
# which is a standalone script (does not touch discrete_diffusion/train.py)
# that reloads each run's own recorded .hydra/config.yaml, forces
# devices=1/num_nodes=1/SingleDeviceStrategy, and calls trainer.validate()
# on that run's checkpoint with dataloaders/model built exactly as training
# built them -- no checkpoint saving, no logger, no wandb.
#
# Usage:
#   ./scripts/training/validate_checkpoints_single_gpu.sh <OUTPUTS_DIR> [CKPT_NAME] [EXTRA_PY_ARGS...]
#
# Example (normal run):
#   ./scripts/training/validate_checkpoints_single_gpu.sh \
#     \$SCRATCH/SwissAI-DLM-data/outputs/scion_new_runs_2026-07-27
#
# Example (memory-leak diagnostic on one run, capped at 400 batches):
#   ./scripts/training/validate_checkpoints_single_gpu.sh \
#     \$SCRATCH/SwissAI-DLM-data/outputs/scion_new_runs_2026-07-27 last.ckpt \
#     --debug_memory --max_val_batches 400 --only_run gbs256_lr0.02_mom0.08

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <OUTPUTS_DIR> [CKPT_NAME=last.ckpt] [EXTRA_PY_ARGS...]"
  exit 1
fi

OUTPUTS_DIR="$1"
CKPT_NAME="${2:-last.ckpt}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi
EXTRA_PY_ARGS=("$@")

ACCOUNT="ab035"
TIME_LIMIT="01:30:00"  # debug partition caps at 1:30:00
CPUS_PER_TASK=72

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
JOB_NAME="validate_ckpts_single_gpu_$(basename "${OUTPUTS_DIR}")"

mkdir -p "${REPO_ROOT}/logs"

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=1
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${REPO_ROOT}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${REPO_ROOT}/logs/${JOB_NAME}_%j.err
#SBATCH --partition=debug

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
cd "\${REPO_ROOT}" || exit 1

export PYTHONPATH="\${REPO_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export WANDB_MODE="disabled"
export HYDRA_FULL_ERROR=1
# Validation batches have variable numbers of masked tokens (random t per
# batch), which produces variably-shaped intermediate tensors in the loss
# computation. The default CUDA caching allocator can't reuse differently
# sized freed blocks, so its reserved pool ratchets up monotonically across
# a long uninterrupted validation pass until it fills the GPU (confirmed via
# a memory probe: 'allocated' stayed flat/noisy, 'reserved' climbed in ~8-9
# GiB steps to 96.88 GiB by batch 400). expandable_segments lets the
# allocator grow existing segments instead of requiring exact-size blocks.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER}-\${SLURM_JOB_ID}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export HF_HOME="\${SCRATCH}/SwissAI-DLM-data/cache/hf"
mkdir -p "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${HF_HOME}"

srun --environment=uni-d2 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=${CPUS_PER_TASK} \
     "${VENV_PYTHON}" -u "${REPO_ROOT}/scripts/training/validate_checkpoints_single_gpu.py" \
  --outputs_dir "${OUTPUTS_DIR}" \
  --ckpt_name "${CKPT_NAME}" \
  ${EXTRA_PY_ARGS[@]+"${EXTRA_PY_ARGS[@]}"}
SBATCH_EOF

echo "Submitted validation job: ${JOB_NAME}"
