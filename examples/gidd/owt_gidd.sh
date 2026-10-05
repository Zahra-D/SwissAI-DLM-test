#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="4,5,6,7"

export WANDB_MODE="online"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
mkdir -p "${WANDB_DIR}"

echo "Starting training with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"

uv run python -u -m discrete_diffusion \
  data=tinystories \
  model=tiny \
  algo=gidd \
  algo.loss_type=gidd \
  algo.p_uniform=0.0 \
  algo.loss_weighting=raw \
  lr_scheduler=cosine_decay_warmup \
  trainer.deterministic=true \
  trainer.num_nodes=1 trainer.devices=4 \
  trainer.max_epochs=1 \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  trainer.log_every_n_steps=1 \
  trainer.val_check_interval=1000 \
  trainer.limit_train_batches=0.15 \
  trainer.limit_val_batches=0.15 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  model.length=128 \
  model.dropout=0.0 \
  optim.lr=5e-4 \
  optim.weight_decay=0.02 \
  optim.beta1=0.9 \
  optim.beta2=0.99 \
  optim.eps=1e-9 \
  lr_scheduler.warmup_t=100 \
  wandb.save_dir="${WANDB_DIR}" \
  wandb.name='tiny-gidd-p-0-raw-hardcode-is-loss-0' \
  wandb.project=UNI-D2
