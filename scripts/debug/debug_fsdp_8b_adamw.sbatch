#!/bin/bash
# One-node, one-step 8B FSDP diagnostic. This validates model memory and FSDP
# execution only; it deliberately uses AdamW because SCION is not compatible
# with the current root-wrapped/sharded FSDP parameter representation.
# Submit with: sbatch scripts/debug/debug_fsdp_8b_adamw.sbatch
#SBATCH --job-name=fsdp_8b_adamw_debug
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --account=a137
#SBATCH --output=logs/debug/%x_%j.out
#SBATCH --error=logs/debug/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DEBUG_OUTPUT="${SCRATCH}/SwissAI-DLM-data/outputs/fsdp_debug"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/debug" "${DEBUG_OUTPUT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SCRATCH}/SwissAI-DLM-data/training-data"
export JOB_TMPDIR="${SLURM_TMPDIR:-/tmp/SwissAI-DLM-${USER:-unknown}-${SLURM_JOB_ID:-manual}}"
export TMPDIR="${JOB_TMPDIR}"
export XDG_CACHE_HOME="${JOB_TMPDIR}/xdg-cache"
export HF_HOME="${SCRATCH}/SwissAI-DLM-data/cache/hf"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}"

srun --environment=uni-d2 \
  --ntasks=4 \
  --ntasks-per-node=4 \
  --cpus-per-task=72 \
  "${VENV_PYTHON}" -u -m discrete_diffusion \
  data=synthetic-gidd \
  model=gidd_hf_8b \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  strategy=fsdp \
  parallel.pipeline.enabled=false \
  trainer.num_nodes=1 \
  trainer.devices=4 \
  trainer.max_steps=1 \
  trainer.accumulate_grad_batches=1 \
  trainer.precision=bf16-mixed \
  trainer.gradient_clip_val=0.0 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=0 \
  trainer.val_check_interval=1000000 \
  loader.global_batch_size=32 \
  loader.eval_global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  model.length=2048 \
  model.max_tokens=2048 \
  model.fsdp_flatten_compatible=true \
  model.activation_checkpointing=true \
  training.ema=0.0 \
  training.torch_compile=false \
  training.fault_tolerant=false \
  training.log_train_aux_metrics=false \
  training.sync_train_loss=false \
  optim=adamw \
  optim.lr=3e-4 \
  optim.weight_decay=0.0 \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last=false \
  callbacks.checkpoint_monitor.save_top_k=0 \
  callbacks.pytorch_profiler.enabled=false \
  checkpointing.resume_from_ckpt=false \
  checkpointing.save_dir="${DEBUG_OUTPUT}/8b_adamw_${SLURM_JOB_ID}/no_checkpoints" \
  hydra.run.dir="${DEBUG_OUTPUT}/8b_adamw_${SLURM_JOB_ID}" \
  wandb=null
