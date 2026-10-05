#!/usr/bin/env bash
set -euo pipefail

# Minimal single-run launcher focused on SCION trace constant collection.
# Adjust Hydra overrides below to your dataset/model/cluster.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv-docker/bin/python"
WANDB_DIR="${ROOT_DIR}/wandb_logs"
mkdir -p "${WANDB_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR
export HYDRA_FULL_ERROR=1

"${PYTHON_BIN}" -u -m discrete_diffusion \
  optim=scion \
  optim.trace_enabled=true \
  optim.trace_collect_noise_stats=true \
  optim.trace_noise_stats_every=200 \
  optim.trace_noise_min_samples=3 \
  trainer.log_every_n_steps=25 \
  trainer.val_check_interval=500 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  "$@"
