#!/usr/bin/env bash
set -euo pipefail

# L12/H768 base-model sweep for the Section-B.3 SCION parameterization.
# Only the common Frank-Wolfe stepsize and SCION momentum are swept. The Set-C
# radii are fixed. DRY_RUN=1 by default; this child grid contains 20 jobs.

TOKEN_BUDGET="${TOKEN_BUDGET:-2000000000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
GBS="${GBS:-256}"
LOCAL_BATCH="${LOCAL_BATCH:-8}"
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE=4
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEED="${SEED:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAZY_SPAWN_DATASET="${LAZY_SPAWN_DATASET:-false}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
DEFER_STORAGE_SETUP="${DEFER_STORAGE_SETUP:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

# Requested first-pass grid. Supply LRS explicitly for a narrower follow-up
# around whichever edge/interior value performs best.
LRS=(${LRS:-1e-4 3e-4 5e-4 7e-4 9e-4})
# Momentum is the new-gradient weight alpha. Preserve the requested order.
MOMENTA=(${MOMENTA:-0.2 0.1 0.05 0.02})

# Hold the previously selected Set-C radii fixed for this two-parameter sweep.
SCALE_EMBED="${SCALE_EMBED:-3500}"
SCALE_BIAS="${SCALE_BIAS:-90}"
SCALE_LN="${SCALE_LN:-6.8}"
SCALE_MATRIX="${SCALE_MATRIX:-336}"
# Distinguish alternative radius parameterizations without changing the
# historical/default run names.
RADIUS_VARIANT="${RADIUS_VARIANT:-fixedSetC}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_b3_equalLR_fixedSetC_betaMom_grid_L12H768_S2048_2B_GBS${GBS}}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Scion_B3_HP_Tuning}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/hparam_tuning/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"

if (( TOKEN_BUDGET <= 0 || SEQUENCE_LENGTH <= 0 || GBS <= 0 || LOCAL_BATCH <= 0 )); then
  echo "TOKEN_BUDGET, SEQUENCE_LENGTH, GBS, and LOCAL_BATCH must be positive." >&2
  exit 1
fi
if (( GBS % (LOCAL_BATCH * NTASKS_PER_NODE) != 0 )); then
  echo "GBS=${GBS} must be divisible by $((LOCAL_BATCH * NTASKS_PER_NODE))." >&2
  exit 1
fi
if [[ ! -d "${PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing pretokenized dataset: ${PRETOK_LOCAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_CACHE_DIR}" ]]; then
  echo "Missing dataset cache: ${DATA_CACHE_DIR}" >&2
  exit 1
fi

NTASKS=$((GBS / LOCAL_BATCH))
NODES=$((NTASKS / NTASKS_PER_NODE))
MAX_STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * GBS)))
ACTUAL_TOKENS=$((MAX_STEPS * SEQUENCE_LENGTH * GBS))
WARMDOWN_ITERS=$((MAX_STEPS * 28 / 100))
PARTITION_DIRECTIVE=""
if [[ -n "${PARTITION}" ]]; then
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

echo "SCION Section-B.3 equal-LR beta/momentum sweep"
echo "Model: L12 H768 I3072 heads12 | S=${SEQUENCE_LENGTH}"
echo "Budget: requested=${TOKEN_BUDGET}, steps=${MAX_STEPS}, actual=${ACTUAL_TOKENS}"
echo "Batch/topology: GBS=${GBS}, local=${LOCAL_BATCH}, nodes=${NODES}, ranks=${NTASKS}, accumulation=1"
echo "Common beta grid: ${LRS[*]}"
echo "Momentum-alpha grid: ${MOMENTA[*]}"
echo "Fixed radii: embed=${SCALE_EMBED}, bias=${SCALE_BIAS}, norm=${SCALE_LN}, matrix=${SCALE_MATRIX}"
echo "Initialization: Section B.3 | residual scale: 4/D | clipping: off"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"
echo "Jobs in full grid: $((${#LRS[@]} * ${#MOMENTA[@]}))"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%-12s %-12s %-8s %-8s %-8s %-8s\n' beta momentum embed bias norm matrix
  for LR in "${LRS[@]}"; do
    for MOMENTUM in "${MOMENTA[@]}"; do
      printf '%-12s %-12s %-8s %-8s %-8s %-8s\n' \
        "${LR}" "${MOMENTUM}" "${SCALE_EMBED}" "${SCALE_BIAS}" "${SCALE_LN}" "${SCALE_MATRIX}"
    done
  done
  echo "Dry run only. Use DRY_RUN=0 to submit the grid."
  exit 0
fi

mkdir -p "${LOG_DIR}"
if [[ "${DEFER_STORAGE_SETUP}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}" "${WANDB_DIR}"
fi
SUBMITTED=0
for LR in "${LRS[@]}"; do
  for MOMENTUM in "${MOMENTA[@]}"; do
    LR_ID="${LR//./p}"
    LR_ID="${LR_ID//-/m}"
    MOM_ID="${MOMENTUM//./p}"
    RUN_NAME="scion_b3_equalLR_${RADIUS_VARIANT}_L12H768_S2048_2B_gbs${GBS}_beta${LR_ID}_alpha${MOM_ID}_seed${SEED}"
    RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

    if [[ "${DEFER_STORAGE_SETUP}" != "1" && "${ALLOW_EXISTING_OUTPUT}" != "1" && -d "${RUN_STORAGE_DIR}" && -n "$(ls -A "${RUN_STORAGE_DIR}")" ]]; then
      echo "Refusing to reuse non-empty output: ${RUN_STORAGE_DIR}" >&2
      exit 1
    fi

    "${SBATCH_BIN}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
${PARTITION_DIRECTIVE}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export TMP="\${JOB_TMPDIR}"
export TEMP="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
mkdir -p "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${WANDB_DIR}" "${RUN_STORAGE_DIR}"

srun --environment=uni-d2 \\
  --ntasks=${NTASKS} \\
  --ntasks-per-node=${NTASKS_PER_NODE} \\
  --cpus-per-task=${CPUS_PER_TASK} \\
  "${VENV_PYTHON}" -u -m discrete_diffusion \\
  seed=${SEED} \\
  data=nemotron-cc-pretok \\
  data.pretok_local_dir="${PRETOK_LOCAL_DIR}" \\
  data.cache_dir="${DATA_CACHE_DIR}" \\
  model=gidd_hf \\
  model.length=${SEQUENCE_LENGTH} \\
  model.max_tokens=1024 \\
  model.n_blocks=12 \\
  model.hidden_size=768 \\
  model.intermediate_size=3072 \\
  model.n_heads=12 \\
  model.dropout=0.0 \\
  model.activation_scale=2.0 \\
  model.attn_soft_cap=30.0 \\
  model.resid_scale=4.0 \\
  model.head_output_scaling_enabled=false \\
  algo=gidd \\
  algo.loss_type=gidd_easydel_lowmem \\
  algo.hybrid_mixing_shift=-1000 \\
  algo.loss_weighting=dynamic \\
  algo.low_discrepancy_sampling=true \\
  algo.time_sampling_scope=global_batch \\
  lr_scheduler=constant_warmup \\
  strategy=ddp \\
  trainer.deterministic=false \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=1 \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=25 \\
  trainer.gradient_clip_val=0.0 \\
  trainer.num_sanity_val_steps=${NUM_SANITY_VAL_STEPS} \\
  trainer.val_check_interval=250 \\
  trainer.limit_val_batches=0.5 \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
  +loader.lazy_spawn_dataset=${LAZY_SPAWN_DATASET} \\
  loader.num_workers=${NUM_WORKERS} \\
  loader.pin_memory=true \\
  training.ema=0.0 \\
  training.antithetic_sampling=true \\
  training.loss_precision=bf16 \\
  training.fault_tolerant=true \\
  training.torch_compile=false \\
  training.log_train_aux_metrics=true \\
  training.sync_train_loss=true \\
  training.validate_after_training=true \\
  eval.generate_samples=false \\
  callbacks.pytorch_profiler.enabled=false \\
  perf.enabled=true \\
  perf.theoretical_peak_tflops_per_gpu=989 \\
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \\
  callbacks.checkpoint_every_n_steps.save_top_k=1 \\
  callbacks.checkpoint_every_n_steps.save_last=true \\
  callbacks.checkpoint_monitor.save_top_k=0 \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=false \\
  optim=scion \\
  optim.lr=${LR} \\
  optim.equal_group_lr=true \\
  optim.boundary_init=true \\
  optim.weight_decay=0.0 \\
  optim.momentum=${MOMENTUM} \\
  optim.scale_embed=${SCALE_EMBED} \\
  optim.scale_bias=${SCALE_BIAS} \\
  optim.scale_layer_norm=${SCALE_LN} \\
  optim.scale_matrix=${SCALE_MATRIX} \\
  optim.bias_norm=RowNorm \\
  optim.norm_layer_norm=BiasRMS \\
  optim.spectral_norm_steps=5 \\
  optim.unconstrained=false \\
  optim.warmup_iters=0 \\
  optim.warmdown_iters=${WARMDOWN_ITERS} \\
  optim.min_lr=1e-8 \\
  optim.trace_enabled=false \\
  optim.trace_collect_noise_stats=false \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  +wandb.save_code=true \\
  wandb.job_type=train \\
  +wandb.tags="[scion,b3_boundary_init,equal_group_lr,beta_momentum_sweep,radius_${RADIUS_VARIANT},L12,H768,gbs_${GBS},seed_${SEED},no_gradient_clipping,no_head_output_scaling,global_time_sampling,residual_4_over_D]"
SBATCH_EOF
    SUBMITTED=$((SUBMITTED + 1))
  done
done

echo "Submitted ${SUBMITTED} jobs."
