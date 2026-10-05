#!/usr/bin/env bash
# Two-node startup benchmark for the packed Nemotron cache.
#
# This intentionally tests the former high-worker configuration:
# 2 nodes x 4 DDP ranks/node x 12 DataLoader workers/rank = 96 spawned workers.
#
# Baseline (loaded HF Dataset sent through spawn):
#   sbatch scripts/debug/debug_spawn_loader_2node_12workers.sbatch
# Path-only wrapper comparison (submit only after the baseline completes):
#   sbatch --job-name=spawn2n_lazy12 --export=ALL,LAZY_SPAWN_DATASET=true \
#     scripts/debug/debug_spawn_loader_2node_12workers.sbatch
#
# Both jobs stop after two optimizer steps by default.  Override MAX_STEPS=1
# for an actual one-step trace_m smoke. They do not checkpoint or resume.
# They must not be used for exact-order-resumable training.
#SBATCH --job-name=spawn2n_base12
#SBATCH --partition=debug
#SBATCH --account=ab035
# The debug QoS limits two-node jobs to 60 node-minutes, so 30 minutes is the
# largest admissible limit.  This smoke finishes after two steps.
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --output=logs/debug/%x_%j.out
#SBATCH --error=logs/debug/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT=${SCRATCH}/SwissAI-DLM-data
DATA_CACHE_DIR=${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok
OUTPUT_DIR=${DATA_ROOT}/outputs/debug/spawn2n_12workers_${SLURM_JOB_ID}
LAZY_SPAWN_DATASET="${LAZY_SPAWN_DATASET:-false}"
# Keep 12 as the historical comparison point, but allow later throughput
# sweeps without editing the script, e.g. WORKERS_PER_RANK=2 sbatch ...
WORKERS_PER_RANK="${WORKERS_PER_RANK:-12}"
MAX_STEPS="${MAX_STEPS:-2}"
TRACE_SMOKE="${TRACE_SMOKE:-false}"
TRACE_M="${TRACE_M:-3}"
TRACE_EVERY="${TRACE_EVERY:-1}"
TRACE_PROGRESS_EVERY="${TRACE_PROGRESS_EVERY:-0}"

mkdir -p "${REPO_ROOT}/logs/debug" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/training-data"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

echo "STARTUP_2NODE $(date -Is): nodes=2 ranks=8 workers/rank=${WORKERS_PER_RANK} total_workers=96 spawn=true pin_memory=true lazy_wrapper=${LAZY_SPAWN_DATASET} max_steps=${MAX_STEPS} trace_smoke=${TRACE_SMOKE} trace_m=${TRACE_M}"

srun --environment=uni-d2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=18 \
  "${REPO_ROOT}/venv-docker/bin/python" -u -m discrete_diffusion \
  seed=4 \
  data=nemotron-cc-pretok \
  data.pretok_local_dir="${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model=gidd_hf model.length=2048 model.max_tokens=1024 \
  model.n_blocks=12 model.hidden_size=768 model.intermediate_size=3072 model.n_heads=12 \
  model.dropout=0.0 model.activation_scale=2.0 model.attn_soft_cap=30.0 \
  algo=gidd algo.loss_type=gidd_easydel_lowmem algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic algo.low_discrepancy_sampling=true \
  strategy=ddp trainer.num_nodes=2 trainer.devices=4 trainer.accumulate_grad_batches=1 \
  trainer.max_steps=${MAX_STEPS} trainer.log_every_n_steps=1 trainer.gradient_clip_val=1.0 \
  trainer.num_sanity_val_steps=0 trainer.val_check_interval=1000000 trainer.limit_val_batches=0 \
  trainer.precision=bf16-mixed \
  loader.global_batch_size=64 loader.eval_global_batch_size=64 loader.batch_size=8 loader.eval_batch_size=8 \
  +loader.exact_resume=false loader.multiprocessing_context=spawn loader.num_workers=${WORKERS_PER_RANK} loader.pin_memory=true \
  +loader.lazy_spawn_dataset="${LAZY_SPAWN_DATASET}" \
  training.ema=0.0 training.antithetic_sampling=true training.loss_precision=bf16 \
  training.fault_tolerant=false training.torch_compile=false training.log_train_aux_metrics=false training.sync_train_loss=true \
  eval.generate_samples=false callbacks.pytorch_profiler.enabled=false \
  checkpointing.save_dir="${OUTPUT_DIR}" checkpointing.resume_from_ckpt=false \
  optim=scion optim.lr=0.0225 optim.weight_decay=0.0 optim.momentum=0.0658 \
  optim.aux_lr_factor=0.02 optim.scale_embed=3500 optim.scale_bias=90 optim.scale_layer_norm=6.8 \
  optim.scale_matrix=336 optim.bias_norm=RowNorm optim.norm_layer_norm=BiasRMS \
  optim.spectral_norm_steps=5 optim.unconstrained=false optim.warmup_iters=0 optim.warmdown_iters=0 \
  optim.trace_enabled="${TRACE_SMOKE}" optim.trace_collect_noise_stats="${TRACE_SMOKE}" \
  optim.trace_noise_stats_every=${TRACE_EVERY} optim.trace_noise_min_samples=3 optim.trace_m=${TRACE_M} \
  +optim.trace_progress_every=${TRACE_PROGRESS_EVERY} \
  hydra.run.dir="${OUTPUT_DIR}/hydra" \
  wandb=null

echo "STARTUP_2NODE_SMOKE_FINISHED $(date -Is)"
