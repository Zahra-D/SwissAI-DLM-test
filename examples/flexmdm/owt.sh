#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
  algo=flexmdm-anyorder \
  data=openwebtext \
  data.wrap=false \
  data.chunking=double_newline \
  data.insert_train_special=false data.insert_valid_special=false \
  data.insert_train_eos=false data.insert_valid_eos=false \
  data.train_min_length=0 data.valid_min_length=0 \
  model=flexmdm_anyorder \
  lr_scheduler=cosine_decay_warmup \
  noise=linear \
  trainer.num_nodes=8 trainer.devices=4 \
  trainer.max_steps=1000000 \
  trainer.accumulate_grad_batches=1 \
  loader.global_batch_size=1024 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  trainer.log_every_n_steps=100 \
  trainer.val_check_interval=10_000 \
  trainer.limit_val_batches=0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  trainer.precision=32 \
  training.torch_compile=true \
  model.length=1024 \
  model.dropout=0.05 \
  optim.lr=3e-4 \
  optim.weight_decay=0.03 \
  optim.beta1=0.9 \
  optim.beta2=0.999 \
  optim.eps=1e-8 \
  lr_scheduler.warmup_t=2000 \
  lr_scheduler.warmup_lr_init=3e-10 \
  lr_scheduler.lr_min=0.0 \
  seed=42 \
  training.ema=0.9999 \
  wandb.project=UNI-D2 \
  wandb.name='flexmdm-anyorder-owt'

