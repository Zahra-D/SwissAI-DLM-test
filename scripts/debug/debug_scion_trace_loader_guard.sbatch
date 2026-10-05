#!/usr/bin/env bash
# Verify that periodic rho/sigma collection gets its dedicated trace loader
# while legacy per-step SCION tracing remains disabled.
#
# Success criteria in the .out log:
#   1. "[SCION_TRACE] independent trace loader" appears once per rank.
#   2. "[SCION_TRACE] begin ... trace_m=3" appears at step 1.
#   3. "[SCION_TRACE] report complete" appears.
#
# This is deliberately one node / four ranks and one optimizer step. It is a
# correctness test for the loader guard, not a throughput benchmark.
#SBATCH --job-name=scion_trace_loader_guard
#SBATCH --partition=debug
#SBATCH --account=ab035
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/debug/%x_%j.out
#SBATCH --error=logs/debug/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT=${SCRATCH}/SwissAI-DLM-data
DATA_CACHE_DIR=${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok
OUTPUT_DIR=${DATA_ROOT}/outputs/debug/scion_trace_loader_guard_${SLURM_JOB_ID}
TIME_SAMPLING_SCOPE="${TIME_SAMPLING_SCOPE:-local}"

mkdir -p "${REPO_ROOT}/logs/debug" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/training-data"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

echo "TRACE_LOADER_GUARD_START $(date -Is) time_sampling_scope=${TIME_SAMPLING_SCOPE}"
srun --environment=uni-d2 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=18 \
  "${REPO_ROOT}/venv-docker/bin/python" -u -m discrete_diffusion \
  seed=4 \
  data=nemotron-cc-pretok \
  data.pretok_local_dir="${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model=gidd_hf model.length=2048 model.max_tokens=1024 \
  model.n_blocks=12 model.hidden_size=768 model.intermediate_size=3072 model.n_heads=12 \
  model.dropout=0.0 model.activation_scale=2.0 model.attn_soft_cap=30.0 \
  algo=gidd algo.loss_type=gidd_easydel_lowmem algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic algo.low_discrepancy_sampling=true algo.time_sampling_scope=${TIME_SAMPLING_SCOPE} \
  strategy=ddp trainer.num_nodes=1 trainer.devices=4 trainer.accumulate_grad_batches=1 \
  trainer.max_steps=1 trainer.log_every_n_steps=1 trainer.gradient_clip_val=1.0 \
  trainer.num_sanity_val_steps=0 trainer.val_check_interval=1000000 trainer.limit_val_batches=0 \
  trainer.precision=bf16-mixed \
  loader.global_batch_size=32 loader.eval_global_batch_size=32 loader.batch_size=8 loader.eval_batch_size=8 \
  +loader.exact_resume=false loader.multiprocessing_context=spawn loader.num_workers=1 loader.pin_memory=true \
  training.ema=0.0 training.antithetic_sampling=true training.loss_precision=bf16 \
  training.fault_tolerant=false training.torch_compile=false training.log_train_aux_metrics=false training.sync_train_loss=true \
  eval.generate_samples=false callbacks.pytorch_profiler.enabled=false \
  checkpointing.save_dir="${OUTPUT_DIR}" checkpointing.resume_from_ckpt=false \
  optim=scion optim.lr=0.0225 optim.weight_decay=0.0 optim.momentum=0.0658 \
  optim.aux_lr_factor=0.02 optim.scale_embed=3500 optim.scale_bias=90 optim.scale_layer_norm=6.8 \
  optim.scale_matrix=336 optim.bias_norm=RowNorm optim.norm_layer_norm=BiasRMS \
  optim.spectral_norm_steps=5 optim.unconstrained=false optim.warmup_iters=0 optim.warmdown_iters=0 \
  optim.trace_enabled=false optim.trace_collect_noise_stats=true \
  optim.trace_noise_stats_every=1 optim.trace_noise_min_samples=3 optim.trace_m=3 \
  +optim.trace_progress_every=1 \
  hydra.run.dir="${OUTPUT_DIR}/hydra" \
  wandb=null

echo "TRACE_LOADER_GUARD_FINISHED $(date -Is)"
