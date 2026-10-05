#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export P_UNIFORM=1.0

export WANDB_MODE="online"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
mkdir -p "${WANDB_DIR}"

echo "Starting training with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"

python -u -m discrete_diffusion \
  data=tinystories \
  model=tiny \
  algo=gidd \
  algo.loss_type=gidd_constant_pi \
  algo.p_uniform="${P_UNIFORM}" \
  algo.loss_weighting=dynamic \
  lr_scheduler=cosine_decay_warmup \
  strategy=ddp \
  trainer.deterministic=false \
  trainer.num_nodes=1 trainer.devices=4 \
  trainer.max_epochs=20 \
  loader.global_batch_size=1024 \
  loader.batch_size=256 \
  loader.eval_batch_size=4 \
  trainer.log_every_n_steps=25 \
  trainer.val_check_interval=1000 \
  trainer.limit_train_batches=0.5 \
  trainer.limit_val_batches=0.3 \
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
  wandb.name="tiny-gidd-constant-pi-${P_UNIFORM}" \
  wandb.project=UNI-D2
