#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for the 2x/4x/6x/8x full-horizon SCION LR grids.
# Invoke one of the submit_scion_budget_{2,4,6,8}x_lr_grid.sh wrappers rather
# than running this file directly.

: "${BUDGET_MULTIPLIER:?Set BUDGET_MULTIPLIER via a budget-grid wrapper.}"
: "${GBS:?Set GBS via a budget-grid wrapper.}"
: "${LR_GRID_CSV:?Set LR_GRID_CSV via a budget-grid wrapper.}"

BASE_TOKEN_BUDGET="${BASE_TOKEN_BUDGET:-2000000000}"
TOKEN_BUDGET="${TOKEN_BUDGET:-$((BASE_TOKEN_BUDGET * BUDGET_MULTIPLIER))}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
LOCAL_BATCH="${LOCAL_BATCH:-8}"
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE=4
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-3:00:00}"
SEED="${SEED:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
DRY_RUN="${DRY_RUN:-1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
ONLY_GRID_INDEX="${ONLY_GRID_INDEX:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"

PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_set_c_${BUDGET_MULTIPLIER}x_gbs${GBS}_lr_grid}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/extending_budget/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Extending_Budget}"
WANDB_GROUP="${WANDB_GROUP:-scion_base_model_token_budget_extension_lr_grid_L12H768_S2048}"

IFS=',' read -r -a LRS <<< "${LR_GRID_CSV}"

if (( BASE_TOKEN_BUDGET <= 0 || TOKEN_BUDGET <= 0 || SEQUENCE_LENGTH <= 0 || GBS <= 0 || LOCAL_BATCH <= 0 )); then
  echo "Token budgets, sequence length, GBS, and local batch must be positive." >&2
  exit 1
fi
if (( TOKEN_BUDGET != BASE_TOKEN_BUDGET * BUDGET_MULTIPLIER )); then
  echo "Warning: TOKEN_BUDGET=${TOKEN_BUDGET} overrides the default ${BUDGET_MULTIPLIER}x budget." >&2
fi
if (( GBS % (LOCAL_BATCH * NTASKS_PER_NODE) != 0 )); then
  echo "GBS=${GBS} must be divisible by one-node batch $((LOCAL_BATCH * NTASKS_PER_NODE))." >&2
  exit 1
fi
if (( NUM_WORKERS < 1 )); then
  echo "NUM_WORKERS must be at least 1 for the spawn-wrapper data path." >&2
  exit 1
fi
if (( ${#LRS[@]} != 5 )); then
  echo "Expected exactly five learning rates, got ${#LRS[@]} from LR_GRID_CSV=${LR_GRID_CSV}." >&2
  exit 1
fi
if [[ -n "${ONLY_GRID_INDEX}" && ! "${ONLY_GRID_INDEX}" =~ ^[1-5]$ ]]; then
  echo "ONLY_GRID_INDEX must be an integer from 1 through 5." >&2
  exit 1
fi
if [[ ! "${RUN_SUFFIX}" =~ ^[A-Za-z0-9_-]*$ ]]; then
  echo "RUN_SUFFIX may contain only letters, digits, underscores, and hyphens." >&2
  exit 1
fi
if [[ ! -d "${PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing pretokenized dataset: ${PRETOK_LOCAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_CACHE_DIR}" ]]; then
  echo "Missing historical data cache: ${DATA_CACHE_DIR}" >&2
  exit 1
fi

NTASKS=$((GBS / LOCAL_BATCH))
NODES=$((NTASKS / NTASKS_PER_NODE))
MAX_STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * GBS)))
WARM_DOWN_STEPS=$((MAX_STEPS * 28 / 100))
ACTUAL_TOKENS=$((MAX_STEPS * SEQUENCE_LENGTH * GBS))
PARTITION_DIRECTIVE=""
if [[ -n "${PARTITION}" ]]; then
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

# Tuned set_c configuration at T0; only optim.lr changes across this grid.
MOMENTUM="0.0658"
SCALE_EMBED="3500"
SCALE_BIAS="90"
SCALE_LN="6.8"
SCALE_MATRIX="336"

echo "SCION set_c ${BUDGET_MULTIPLIER}x full-horizon LR grid"
echo "S=${SEQUENCE_LENGTH}, GBS=${GBS}, local batch=${LOCAL_BATCH}"
echo "Topology: ${NODES} nodes, ${NTASKS} ranks, accumulation=1"
echo "Requested budget: ${TOKEN_BUDGET}; steps=${MAX_STEPS}; actual tokens=${ACTUAL_TOKENS}"
echo "Warmdown: ${WARM_DOWN_STEPS} steps (28% of the full horizon)"
echo "Gradient clipping value: ${GRADIENT_CLIP_VAL} (0.0 means disabled)"
echo "LR grid: ${LRS[*]}"
echo "Dataset/cache: ${PRETOK_LOCAL_DIR} | ${DATA_CACHE_DIR}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%-4s %-12s %-8s %-8s %-8s %-8s %-8s\n' \
    "idx" "lr" "mom" "embed" "bias" "one_d" "matrix"
  for i in "${!LRS[@]}"; do
    if [[ -n "${ONLY_GRID_INDEX}" && "$((i + 1))" != "${ONLY_GRID_INDEX}" ]]; then
      continue
    fi
    printf '%-4s %-12s %-8s %-8s %-8s %-8s %-8s\n' \
      "$((i + 1))" "${LRS[$i]}" "${MOMENTUM}" "${SCALE_EMBED}" \
      "${SCALE_BIAS}" "${SCALE_LN}" "${SCALE_MATRIX}"
  done
  echo "Dry run only. Use DRY_RUN=0 to submit the selected run(s)."
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${WANDB_DIR}"
SUBMITTED=0
for i in "${!LRS[@]}"; do
  LR="${LRS[$i]}"
  GRID_INDEX=$((i + 1))
  if [[ -n "${ONLY_GRID_INDEX}" && "${GRID_INDEX}" != "${ONLY_GRID_INDEX}" ]]; then
    continue
  fi
  RUN_NAME="scion_set_c_${BUDGET_MULTIPLIER}x_gbs${GBS}_lr${LR}_grid${GRID_INDEX}${RUN_SUFFIX}"
  RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" && -d "${RUN_STORAGE_DIR}" && -n "$(ls -A "${RUN_STORAGE_DIR}")" ]]; then
    echo "Refusing to reuse non-empty output directory: ${RUN_STORAGE_DIR}" >&2
    echo "Set ALLOW_EXISTING_OUTPUT=1 only if this is intentional." >&2
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
export WANDB_RESUME=allow
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
  trainer.gradient_clip_val=${GRADIENT_CLIP_VAL} \\
  trainer.num_sanity_val_steps=2 \\
  trainer.val_check_interval=250 \\
  trainer.limit_val_batches=0.5 \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
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
  optim.weight_decay=0.0 \\
  optim.momentum=${MOMENTUM} \\
  optim.aux_lr_factor=0.02 \\
  optim.scale_embed=${SCALE_EMBED} \\
  optim.scale_bias=${SCALE_BIAS} \\
  optim.scale_layer_norm=${SCALE_LN} \\
  optim.scale_matrix=${SCALE_MATRIX} \\
  optim.bias_norm=RowNorm \\
  optim.norm_layer_norm=BiasRMS \\
  optim.spectral_norm_steps=5 \\
  optim.unconstrained=false \\
  optim.warmup_iters=0 \\
  optim.warmdown_iters=${WARM_DOWN_STEPS} \\
  optim.min_lr=1e-8 \\
  optim.trace_enabled=false \\
  optim.trace_collect_noise_stats=false \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  wandb.notes=full_horizon_${BUDGET_MULTIPLIER}x_set_c_lr_grid_gradient_clip_${GRADIENT_CLIP_VAL} \\
  +wandb.save_code=true \\
  wandb.job_type=train \\
  +wandb.tags="[extending_budget,full_horizon,lr_grid,scion,gidd_hf,global_time_sampling,set_c,gradient_clip_${GRADIENT_CLIP_VAL},${BUDGET_MULTIPLIER}x,gbs_${GBS},local_batch_${LOCAL_BATCH},nodes_${NODES},tokens_${ACTUAL_TOKENS}]"
SBATCH_EOF
  SUBMITTED=$((SUBMITTED + 1))
done

echo "Submitted ${SUBMITTED} run(s) for the ${BUDGET_MULTIPLIER}x budget grid."
