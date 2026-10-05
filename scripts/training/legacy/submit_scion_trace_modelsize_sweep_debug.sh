#!/usr/bin/env bash
set -euo pipefail

# Sweep script for SCION trace runs where we vary:
# 1) number of layers
# 2) hidden dimension
#
# Keeps the existing mu/L/rho tracing setup:
# - optim.trace_enabled=true
# - optim.trace_collect_noise_stats=true

TOKEN_BUDGET=5000000000
MAX_LENGTH=2048
TOTAL_NODES=1
MICRO_BATCH_SIZE_PER_GPU=8
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="ab035"
TIME_LIMIT="01:30:00"
SEED=4


EXPERIMENT_NAME="scion_new_runs_tracing_2026-07-27_debug"

GLOBAL_BATCH_SIZES=(256)
LRS=(0.0225)
MOMENTA=(0.0658)
OSCALE_EMBEDS=(3500)
OSCALE_BIASES=(90)
OSCALE_LNS=(6.8)
OSCALE_MATRICES=(336)

# GLOBAL_BATCH_SIZES=(256)
# LRS=(0.02)
# MOMENTA=(0.08)
# OSCALE_EMBEDS=(2000)
# OSCALE_BIASES=(100)
# OSCALE_LNS=(10)
# OSCALE_MATRICES=(400)

# GLOBAL_BATCH_SIZES=(256)
# LRS=(0.018)
# MOMENTA=(0.0885)
# OSCALE_EMBEDS=(3905)
# OSCALE_BIASES=(44)
# OSCALE_LNS=(10)
# OSCALE_MATRICES=(430)

# Rho/noise trace controls
TRACE_NOISE_EVERY=500
TRACE_NOISE_MIN_SAMPLES=3
TRACE_M=64

# Model-size knobs you asked for
LAYERS=(9)
# LAYERS=(12 9 6 3)
HIDDEN_DIMS=(768)
N_HEADS=12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs"

submit_count=0

for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do
  NODES=${TOTAL_NODES}
  NTASKS=$((NODES * NTASKS_PER_NODE))
  TOKENS_PER_MICRO_STEP=$((NTASKS * MICRO_BATCH_SIZE_PER_GPU))

  if (( NTASKS % NTASKS_PER_NODE != 0 )); then
    echo "Skipping global batch size ${GBS}: ntasks=${NTASKS} is not divisible by ${NTASKS_PER_NODE}"
    continue
  fi
  if (( GBS % TOKENS_PER_MICRO_STEP != 0 )); then
    echo "Skipping global batch size ${GBS}: not divisible by ntasks=${NTASKS} * microbatch=${MICRO_BATCH_SIZE_PER_GPU}"
    continue
  fi
  ACCUMULATE_GRAD_BATCHES=$((GBS / TOKENS_PER_MICRO_STEP))

  MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))


  for N_LAYER in "${LAYERS[@]}"; do
    for HIDDEN_SIZE in "${HIDDEN_DIMS[@]}"; do
      # Safety: attention heads must divide hidden size.
      if (( HIDDEN_SIZE % N_HEADS != 0 )); then
        echo "Skipping n_layer=${N_LAYER}, hidden_size=${HIDDEN_SIZE}: hidden_size % n_heads != 0"
        continue
      fi

      INTERMEDIATE_SIZE=$((HIDDEN_SIZE * 4))

      for MOM in "${MOMENTA[@]}"; do
        for OSE in "${OSCALE_EMBEDS[@]}"; do
          for OSB in "${OSCALE_BIASES[@]}"; do
            for OSLN in "${OSCALE_LNS[@]}"; do
              for OSM in "${OSCALE_MATRICES[@]}"; do
                for LR in "${LRS[@]}"; do
                  JOB_NAME="scion_gbs${GBS}_lr${LR}_mom${MOM}_ose${OSE}_osb${OSB}_osln${OSLN}_osm${OSM}_L${N_LAYER}_H${HIDDEN_SIZE}_constant_masked_fast_debug_2_cliping_att_"

                  sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${REPO_ROOT}/logs/debug_pipeline/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/debug_pipeline/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --partition=debug

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
cd "\${REPO_ROOT}" || exit 1

export PYTHONPATH="\${REPO_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
export HYDRA_FULL_ERROR=1

export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER}-\${SLURM_JOB_ID}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export HF_HOME="\${SCRATCH}/SwissAI-DLM-data/cache/hf"
mkdir -p "\${WANDB_DIR}" "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${HF_HOME}"

srun --environment=uni-d2 \
     --ntasks=${NTASKS} \
     --ntasks-per-node=${NTASKS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     "${VENV_PYTHON}" -u -m discrete_diffusion \
  seed=${SEED} \
  data=nemotron-cc-pretok \
  model=gidd_hf \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  lr_scheduler=constant_warmup \
  strategy=ddp \
  model.activation_scale=2.0 \
  training.ema=0.0 \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
  trainer.log_every_n_steps=1 \
  loader.global_batch_size=${GBS} \
  loader.batch_size=${MICRO_BATCH_SIZE_PER_GPU} \
  eval.generate_samples=false \
  perf.enabled=true \
  perf.theoretical_peak_tflops_per_gpu=989 \
  loader.eval_batch_size=${MICRO_BATCH_SIZE_PER_GPU} \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=0.5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  model.length=${MAX_LENGTH} \
  model.dropout=0.0 \
  model.n_blocks=${N_LAYER} \
  model.hidden_size=${HIDDEN_SIZE} \
  model.intermediate_size=${INTERMEDIATE_SIZE} \
  model.n_heads=${N_HEADS} \
  optim=scion \
  optim.lr=${LR} \
  optim.weight_decay=0.00 \
  optim.momentum=${MOM} \
  optim.scale_embed=${OSE} \
  optim.scale_bias=${OSB} \
  optim.scale_layer_norm=${OSLN} \
  optim.scale_matrix=${OSM} \
  optim.unconstrained=false \
  optim.trace_enabled=true \
  optim.trace_collect_noise_stats=true \
  optim.trace_noise_stats_every=${TRACE_NOISE_EVERY} \
  optim.trace_noise_min_samples=${TRACE_NOISE_MIN_SAMPLES} \
  optim.trace_m=${TRACE_M} \
  optim.warmup_iters=0 \
  optim.warmdown_iters=0 \
  optim.min_lr=1e-8 \
  loader.multiprocessing_context=spawn \
  loader.num_workers=4 \
  loader.pin_memory=true \
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/debug/${JOB_NAME} \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="woEMA_${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group="scion_debug" \
  wandb.job_type="train" \
  +wandb.tags='[constants,scion,gidd_hf,modelsize,mu_l_rho,gbs_${GBS},lr_${LR},mom_${MOM},layers_${N_LAYER},hidden_${HIDDEN_SIZE}]'
SBATCH_EOF

                  submit_count=$((submit_count + 1))
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Submitted ${submit_count} jobs."
