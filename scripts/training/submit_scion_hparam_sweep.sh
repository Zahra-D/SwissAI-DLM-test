#!/bin/bash
set -euo pipefail

# Hyperparameter sweep for gidd_hf optimizer/perf comparison
# Rules from user request:
# - Defaults to a 128-file data subset so preprocessing stays fast.
# - Total token budget per run defaults to 40M tokens.
# - iterations = TOKEN_BUDGET / (max_length * global_batch_size)
# - loader.batch_size is per GPU
# - target_micro_global_batch = target_nodes * ntasks_per_node * loader.batch_size
# - grad_accum_steps = global_batch_size / target_micro_global_batch

TOKEN_BUDGET="${TOKEN_BUDGET:-40000000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MODEL_MAX_TOKENS="${MODEL_MAX_TOKENS:-${MAX_LENGTH}}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
TARGET_NODES="${TARGET_NODES:-1}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ACCOUNT="${ACCOUNT:-a137}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
PERF_ENABLED="${PERF_ENABLED:-true}"
PERF_PEAK_TFLOPS_PER_GPU="${PERF_PEAK_TFLOPS_PER_GPU:-989}"
PERF_SYNC_CUDA_FOR_TIMING="${PERF_SYNC_CUDA_FOR_TIMING:-true}"
MODEL_CONFIG="${MODEL_CONFIG:-gidd_hf}"
MODEL_TAG="${MODEL_CONFIG//\//_}"
GIDD_LOSS_TYPE="${GIDD_LOSS_TYPE:-gidd_easydel_lowmem}"
GIDD_LOSS_TAG="${GIDD_LOSS_TYPE//\//_}"
GIDD_LOSS_JOB_TAG=""
if [[ "${GIDD_LOSS_TYPE}" != "gidd_easydel" ]]; then
  GIDD_LOSS_JOB_TAG="_${GIDD_LOSS_TAG#gidd_}"
fi
MODEL_ACTIVATION_CHECKPOINTING="${MODEL_ACTIVATION_CHECKPOINTING:-}"
MODEL_ACTIVATION_CHECKPOINT_PRESERVE_RNG_STATE="${MODEL_ACTIVATION_CHECKPOINT_PRESERVE_RNG_STATE:-}"
STRATEGY_CONFIG="${STRATEGY_CONFIG:-ddp}"
STRATEGY_TAG="${STRATEGY_CONFIG//\//_}"
OPTIM_CONFIG="${OPTIM_CONFIG:-scion}"
OPTIM_TAG="${OPTIM_CONFIG//\//_}"
OUTPUT_SUBDIR="hparam_sweep_${OPTIM_TAG}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-false}"
PROFILE_START_STEP="${PROFILE_START_STEP:-2}"
PROFILE_WARMUP_BATCHES="${PROFILE_WARMUP_BATCHES:-2}"
# Active batches are Lightning train batches, i.e. microbatches under gradient
# accumulation. Keep the default above common acc=16 runs so the trace includes
# at least one optimizer step/reduction boundary.
PROFILE_ACTIVE_BATCHES="${PROFILE_ACTIVE_BATCHES:-20}"
PROFILE_ROW_LIMIT="${PROFILE_ROW_LIMIT:-50}"
# Capture every rank by default so DDP waits can be attributed to the slow rank.
# Override with PROFILE_ALL_RANKS=false when you only need a smaller rank-0 trace.
PROFILE_ALL_RANKS="${PROFILE_ALL_RANKS:-true}"
TRAINING_TORCH_COMPILE="${TRAINING_TORCH_COMPILE:-true}"
TRAINING_EMA="${TRAINING_EMA:-0}"
TRAINING_ANTITHETIC_SAMPLING="${TRAINING_ANTITHETIC_SAMPLING:-false}"
TRAINING_FAULT_TOLERANT="${TRAINING_FAULT_TOLERANT:-false}"
TRAINING_LOG_TRAIN_AUX_METRICS="${TRAINING_LOG_TRAIN_AUX_METRICS:-false}"
TRAINING_SYNC_TRAIN_LOSS="${TRAINING_SYNC_TRAIN_LOSS:-false}"
TRAINER_GRADIENT_CLIP_VAL="${TRAINER_GRADIENT_CLIP_VAL:-0.0}"
TRAINER_NUM_SANITY_VAL_STEPS="${TRAINER_NUM_SANITY_VAL_STEPS:-0}"
TRAINER_VAL_CHECK_INTERVAL="${TRAINER_VAL_CHECK_INTERVAL:-1000000}"
TRAINER_LIMIT_VAL_BATCHES="${TRAINER_LIMIT_VAL_BATCHES:-0}"
EVAL_GENERATE_SAMPLES="${EVAL_GENERATE_SAMPLES:-false}"
CHECKPOINT_RESUME_FROM_CKPT="${CHECKPOINT_RESUME_FROM_CKPT:-false}"
CHECKPOINT_EVERY_SAVE_TOP_K="${CHECKPOINT_EVERY_SAVE_TOP_K:-0}"
CHECKPOINT_EVERY_SAVE_LAST="${CHECKPOINT_EVERY_SAVE_LAST:-false}"
CHECKPOINT_MONITOR_SAVE_TOP_K="${CHECKPOINT_MONITOR_SAVE_TOP_K:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Scaling_Law-perf-synthetic}"
WANDB_GROUP="${WANDB_GROUP:-${OPTIM_TAG}_nemotron_hparam_sweep}"

# Sweep knobs. Override any list from the shell, e.g.
#   GLOBAL_BATCH_SIZES="256 512" LRS="3e-5 1e-4" SCION_SCALE_MATRICES="25 50" ./scripts/training/submit_scion_hparam_sweep.sh
#
# LRS is the common SCION Frank-Wolfe stepsize. With equal_group_lr=true it is
# used unchanged by embeddings, biases, 1D/norm vectors, and matrices.
GLOBAL_BATCH_SIZES=(${GLOBAL_BATCH_SIZES:-64 128 256 512})
LRS=(${LRS:-1e-4})

# SCION momentum is "new-gradient weight": 1.0 means no momentum buffer,
# 0.1 is roughly traditional momentum=0.9.
SCION_MOMENTA=(${SCION_MOMENTA:-${MOMENTA:-1.0}})

# Retained only for reproducing legacy runs with equal_group_lr=false.
SCION_AUX_LR_FACTORS=(${SCION_AUX_LR_FACTORS:-0.02})

# SCION update scales/radii per parameter group. These multiply the normalized
# update direction before the group LR is applied.
SCION_SCALE_EMBEDS=(${SCION_SCALE_EMBEDS:-${OSCALE_EMBEDS:-3000}})
SCION_SCALE_BIASES=(${SCION_SCALE_BIASES:-${OSCALE_BIASES:-10}})
SCION_SCALE_LAYER_NORMS=(${SCION_SCALE_LAYER_NORMS:-${OSCALE_LNS:-10}})
SCION_SCALE_MATRICES=(${SCION_SCALE_MATRICES:-${OSCALE_MATRICES:-50}})

# Norm backends for non-matrix SCION groups. Matrix tensors always use Spectral.
SCION_BIAS_NORMS=(${SCION_BIAS_NORMS:-RowNorm})
SCION_LAYER_NORM_NORMS=(${SCION_LAYER_NORM_NORMS:-BiasRMS})
SCION_SPECTRAL_NORM_STEPS=(${SCION_SPECTRAL_NORM_STEPS:-5})

# false keeps SCION's constrained update shrink p <- (1-lr)*p before the update.
SCION_UNCONSTRAINED_VALUES=(${SCION_UNCONSTRAINED_VALUES:-false})

# SCION LR schedule knobs. Warmdown is an integer percentage of MAX_STEPS.
SCION_WARMUP_ITERS=(${SCION_WARMUP_ITERS:-0})
SCION_WARMDOWN_PCTS=(${SCION_WARMDOWN_PCTS:-28})
SCION_MIN_LRS=(${SCION_MIN_LRS:-1e-8})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE:-}"
SOURCE_PRETOK_LOCAL_DIR="${SOURCE_PRETOK_LOCAL_DIR:-${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok}"
USE_SYNTHETIC_DATA="${USE_SYNTHETIC_DATA:-1}"
USE_PRETOK_SUBSET="${USE_PRETOK_SUBSET:-1}"
PRETOK_SUBSET_FILES="${PRETOK_SUBSET_FILES:-128}"
LOADER_NUM_WORKERS="${LOADER_NUM_WORKERS:-}"
SWEEP_ROOT="${SWEEP_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
SUBSET_ROOT="${SUBSET_ROOT:-${SCRATCH}/SwissAI-DLM-data/smoke}"
PRETOK_SUBSET_DIR="${PRETOK_SUBSET_DIR:-${SUBSET_ROOT}/gidd-nemotron-cc-pretok-${PRETOK_SUBSET_FILES}files}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SWEEP_ROOT}/outputs}"
CE_IMAGE="${CE_IMAGE:?Set CE_IMAGE to the shared .sqsh path}"
CONTAINER_ENV_FILE="${CONTAINER_ENV_FILE:-${SWEEP_ROOT}/uni-d2-ce.toml}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-/usr/bin/python}"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs"

if [[ ! -f "${CE_IMAGE}" ]]; then
  echo "Missing CE image: ${CE_IMAGE}" >&2
  exit 1
fi

if [[ "${USE_SYNTHETIC_DATA}" == "1" ]]; then
  DATA_CONFIG="synthetic-gidd"
  PRETOK_LOCAL_DIR=""
  DEFAULT_DATA_CACHE_DIR="${SUBSET_ROOT}/training-data/synthetic-gidd"
  DEFAULT_CACHE_ROOT="${SUBSET_ROOT}/cache"
  DATA_TAG="synthetic"
  if [[ -z "${LOADER_NUM_WORKERS}" ]]; then
    LOADER_NUM_WORKERS=0
  fi
else
  DATA_CONFIG="nemotron-cc-pretok"
  if [[ -z "${LOADER_NUM_WORKERS}" ]]; then
    LOADER_NUM_WORKERS=4
  fi

  if [[ ! -d "${SOURCE_PRETOK_LOCAL_DIR}" ]]; then
    echo "Missing source pretokenized Nemotron data directory: ${SOURCE_PRETOK_LOCAL_DIR}" >&2
    echo "Set SOURCE_PRETOK_LOCAL_DIR to the shared parquet source directory." >&2
    exit 1
  fi

  if [[ "${USE_PRETOK_SUBSET}" == "1" ]]; then
    mkdir -p "${PRETOK_SUBSET_DIR}"
    existing_subset_files=$(find "${PRETOK_SUBSET_DIR}" -type l -name '*.parquet' | wc -l)
    if (( existing_subset_files < PRETOK_SUBSET_FILES )); then
      echo "Creating ${PRETOK_SUBSET_FILES}-file Nemotron sweep subset at ${PRETOK_SUBSET_DIR}"
      while IFS= read -r src; do
        rel="${src#${SOURCE_PRETOK_LOCAL_DIR}/}"
        mkdir -p "${PRETOK_SUBSET_DIR}/$(dirname "${rel}")"
        ln -sfn "${src}" "${PRETOK_SUBSET_DIR}/${rel}"
      done < <(find "${SOURCE_PRETOK_LOCAL_DIR}" -type f -name '*.parquet' | sort | head -n "${PRETOK_SUBSET_FILES}")
    fi

    PRETOK_LOCAL_DIR="${PRETOK_SUBSET_DIR}"
    DEFAULT_DATA_CACHE_DIR="${SUBSET_ROOT}/training-data/nemotron-cc-pretok-${PRETOK_SUBSET_FILES}files"
    DEFAULT_CACHE_ROOT="${SUBSET_ROOT}/cache"
    DATA_TAG="${PRETOK_SUBSET_FILES}files"
  else
    PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${SOURCE_PRETOK_LOCAL_DIR}}"
    DEFAULT_DATA_CACHE_DIR="${SWEEP_ROOT}/training-data/nemotron-cc-pretok"
    DEFAULT_CACHE_ROOT="${SWEEP_ROOT}/cache"
    DATA_TAG="full"
  fi
fi

DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DEFAULT_DATA_CACHE_DIR}}"
CACHE_ROOT="${CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
mkdir -p "${DATA_CACHE_DIR}" "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"

mkdir -p "$(dirname "${CONTAINER_ENV_FILE}")"
cat > "${CONTAINER_ENV_FILE}" <<EDF_EOF
image = "${CE_IMAGE}"

mounts = [
  "/capstor",
  "/iopsstor",
  "/users"
]

writable = true
workdir = "/workspace"

[env]
NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"

[annotations]
"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "cuda12"
EDF_EOF

submit_count=0

if ! [[ "${TARGET_NODES}" =~ ^[0-9]+$ ]]; then
  echo "TARGET_NODES=${TARGET_NODES} must be a positive integer" >&2
  exit 1
fi

if (( TARGET_NODES <= 0 )); then
  echo "TARGET_NODES=${TARGET_NODES} must be positive" >&2
  exit 1
fi

for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do
  NODES=${TARGET_NODES}
  NTASKS=$((NODES * NTASKS_PER_NODE))
  MICRO_GBS=$((NTASKS * BATCH_SIZE_PER_GPU))

  if (( GBS % MICRO_GBS != 0 )); then
    echo "Skipping global batch size ${GBS}: cannot fit exactly on TARGET_NODES=${TARGET_NODES} with BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} (micro_gbs=${MICRO_GBS})"
    continue
  fi

  GRAD_ACCUM_STEPS=$((GBS / MICRO_GBS))

  if (( NTASKS <= 0 )); then
    echo "Skipping global batch size ${GBS}: ntasks=${NTASKS} must be positive"
    continue
  fi

  if (( NTASKS % NTASKS_PER_NODE != 0 )); then
    echo "Skipping global batch size ${GBS}: ntasks=${NTASKS} is not divisible by ${NTASKS_PER_NODE}"
    continue
  fi

  NODE_TAG="_n${NODES}"
  ACC_TAG=""
  if (( GRAD_ACCUM_STEPS > 1 )); then
    ACC_TAG="_acc${GRAD_ACCUM_STEPS}"
  fi
  STRATEGY_JOB_TAG=""
  if [[ "${STRATEGY_CONFIG}" != "ddp" ]]; then
    STRATEGY_JOB_TAG="_${STRATEGY_TAG}"
  fi

  # Integer floor as requested formula implies discrete iterations.
  MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))

  for LR in "${LRS[@]}"; do
    for MOM in "${SCION_MOMENTA[@]}"; do
      for AUX_LR_FACTOR in "${SCION_AUX_LR_FACTORS[@]}"; do
        for SCALE_EMBED in "${SCION_SCALE_EMBEDS[@]}"; do
          for SCALE_BIAS in "${SCION_SCALE_BIASES[@]}"; do
            for SCALE_LAYER_NORM in "${SCION_SCALE_LAYER_NORMS[@]}"; do
              for SCALE_MATRIX in "${SCION_SCALE_MATRICES[@]}"; do
                for BIAS_NORM in "${SCION_BIAS_NORMS[@]}"; do
                  for LAYER_NORM_NORM in "${SCION_LAYER_NORM_NORMS[@]}"; do
                    for SPECTRAL_NORM_STEPS in "${SCION_SPECTRAL_NORM_STEPS[@]}"; do
                      for UNCONSTRAINED in "${SCION_UNCONSTRAINED_VALUES[@]}"; do
                        for WARMUP_ITERS in "${SCION_WARMUP_ITERS[@]}"; do
                          for WARMDOWN_PCT in "${SCION_WARMDOWN_PCTS[@]}"; do
                            for MIN_LR in "${SCION_MIN_LRS[@]}"; do
                              WARMDOWN_ITERS=$((MAX_STEPS * WARMDOWN_PCT / 100))
                              JOB_NAME="${OPTIM_TAG}_${MODEL_TAG}${GIDD_LOSS_JOB_TAG}${STRATEGY_JOB_TAG}_${DATA_TAG}_gbs${GBS}_lr${LR}${NODE_TAG}${ACC_TAG}"
                              WANDB_TAGS="[${OPTIM_TAG},${MODEL_TAG},${GIDD_LOSS_TAG},${STRATEGY_TAG},sweep,${DATA_TAG},gbs_${GBS},nodes_${NODES},acc_${GRAD_ACCUM_STEPS},lr_${LR}]"
                              if [[ "${OPTIM_CONFIG}" == "scion" ]]; then
                                JOB_NAME="${OPTIM_TAG}_${MODEL_TAG}${GIDD_LOSS_JOB_TAG}${STRATEGY_JOB_TAG}_${DATA_TAG}_gbs${GBS}_lr${LR}_aux${AUX_LR_FACTOR}_mom${MOM}_se${SCALE_EMBED}_sb${SCALE_BIAS}_sln${SCALE_LAYER_NORM}_sm${SCALE_MATRIX}${NODE_TAG}${ACC_TAG}"
                                WANDB_TAGS="[${OPTIM_TAG},${MODEL_TAG},${GIDD_LOSS_TAG},${STRATEGY_TAG},sweep,${DATA_TAG},gbs_${GBS},nodes_${NODES},acc_${GRAD_ACCUM_STEPS},lr_${LR},scion_aux_lr_factor_${AUX_LR_FACTOR},scion_momentum_${MOM},scion_scale_embed_${SCALE_EMBED},scion_scale_bias_${SCALE_BIAS},scion_scale_layer_norm_${SCALE_LAYER_NORM},scion_scale_matrix_${SCALE_MATRIX},scion_bias_norm_${BIAS_NORM},scion_layer_norm_norm_${LAYER_NORM_NORM},scion_spectral_norm_steps_${SPECTRAL_NORM_STEPS},scion_unconstrained_${UNCONSTRAINED},scion_warmup_iters_${WARMUP_ITERS},scion_warmdown_pct_${WARMDOWN_PCT},scion_min_lr_${MIN_LR}]"
                              fi

                              sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${REPO_ROOT}/logs/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
cd "\${REPO_ROOT}" || exit 1

RUN_ID="\${SLURM_JOB_ID:-manual_\$(date +%Y%m%d_%H%M%S)_\$\$}"
RUN_NAME="${JOB_NAME}_\${RUN_ID}"

export PYTHONPATH="\${REPO_ROOT}:\${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SWEEP_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export TMP="\${JOB_TMPDIR}"
export TEMP="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export MPLCONFIGDIR="\${JOB_TMPDIR}/matplotlib"
export HF_HOME="${CACHE_ROOT}/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
WANDB_EXTRA_ARGS=()
JOB_WANDB_ENTITY="${WANDB_ENTITY}"
JOB_WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE}"
if [[ -n "\${JOB_WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY="\${JOB_WANDB_ENTITY}"
  WANDB_EXTRA_ARGS+=(+wandb.entity="\${JOB_WANDB_ENTITY}")
fi
if [[ -n "\${JOB_WANDB_TOKEN_FILE}" ]]; then
  if [[ ! -r "\${JOB_WANDB_TOKEN_FILE}" ]]; then
    echo "Requested W&B token file is not readable: \${JOB_WANDB_TOKEN_FILE}" >&2
    exit 1
  fi
  export WANDB_API_KEY="\$(tr -d '[:space:]' < "\${JOB_WANDB_TOKEN_FILE}")"
  if [[ -z "\${WANDB_API_KEY}" ]]; then
    echo "Empty W&B API key in \${JOB_WANDB_TOKEN_FILE}" >&2
    exit 1
  fi
fi
export TORCHDYNAMO_DISABLE=0
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
export HYDRA_FULL_ERROR=1

mkdir -p \
  "\${TMPDIR}" \
  "\${XDG_CACHE_HOME}" \
  "\${MPLCONFIGDIR}" \
  "\${HF_DATASETS_CACHE}" \
  "\${HUGGINGFACE_HUB_CACHE}" \
  "\${TRANSFORMERS_CACHE}" \
  "\${WANDB_DIR}" \
  "\${DISCRETE_DIFFUSION_SCRATCH_DIR}" \
  "${DATA_CACHE_DIR}" \
  "${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"

echo "Runtime temp dir: \${TMPDIR}"
echo "Runtime XDG cache: \${XDG_CACHE_HOME}"
echo "Runtime HF cache: \${HF_HOME}"
echo "Run name: \${RUN_NAME}"

OPTIM_HYDRA_ARGS=(
  optim="${OPTIM_CONFIG}"
  optim.lr=${LR}
  optim.weight_decay=0.00
)
MODEL_HYDRA_ARGS=()
if [[ -n "${MODEL_ACTIVATION_CHECKPOINTING}" ]]; then
  MODEL_HYDRA_ARGS+=(model.activation_checkpointing=${MODEL_ACTIVATION_CHECKPOINTING})
fi
if [[ -n "${MODEL_ACTIVATION_CHECKPOINT_PRESERVE_RNG_STATE}" ]]; then
  MODEL_HYDRA_ARGS+=(model.activation_checkpoint_preserve_rng_state=${MODEL_ACTIVATION_CHECKPOINT_PRESERVE_RNG_STATE})
fi
if [[ "${OPTIM_CONFIG}" == "scion" ]]; then
  OPTIM_HYDRA_ARGS+=(
    optim.momentum=${MOM}
    optim.equal_group_lr=true
    optim.boundary_init=true
    optim.aux_lr_factor=${AUX_LR_FACTOR}
    optim.scale_embed=${SCALE_EMBED}
    optim.scale_bias=${SCALE_BIAS}
    optim.scale_layer_norm=${SCALE_LAYER_NORM}
    optim.scale_matrix=${SCALE_MATRIX}
    optim.bias_norm=${BIAS_NORM}
    optim.norm_layer_norm=${LAYER_NORM_NORM}
    optim.spectral_norm_steps=${SPECTRAL_NORM_STEPS}
    optim.unconstrained=${UNCONSTRAINED}
    optim.warmup_iters=${WARMUP_ITERS}
    optim.warmdown_iters=${WARMDOWN_ITERS}
    optim.min_lr=${MIN_LR}
  )
fi

srun --environment="${CONTAINER_ENV_FILE}" \
     --ntasks=${NTASKS} \
     --ntasks-per-node=${NTASKS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     "${CONTAINER_PYTHON}" -u -m discrete_diffusion \
  data="${DATA_CONFIG}" \
  data.pretok_local_dir="${PRETOK_LOCAL_DIR}" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model="${MODEL_CONFIG}" \
  algo=gidd \
  algo.loss_type="${GIDD_LOSS_TYPE}" \
  sampling.predictor=gidd \
  sampling.sampler._target_=discrete_diffusion.sampling.gidd.GIDDSampler \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  algo.low_discrepancy_sampling=true \
  lr_scheduler=constant_warmup \
  strategy="${STRATEGY_CONFIG}" \
  seed=1 \
  trainer.deterministic=false \
  trainer.gradient_clip_val=${TRAINER_GRADIENT_CLIP_VAL} \
  trainer.num_sanity_val_steps=${TRAINER_NUM_SANITY_VAL_STEPS} \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  loader.global_batch_size=${GBS} \
  loader.eval_global_batch_size=${MICRO_GBS} \
  loader.batch_size=${BATCH_SIZE_PER_GPU} \
  loader.eval_batch_size=${BATCH_SIZE_PER_GPU} \
  trainer.log_every_n_steps=5 \
  trainer.val_check_interval=${TRAINER_VAL_CHECK_INTERVAL} \
  trainer.limit_val_batches=${TRAINER_LIMIT_VAL_BATCHES} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  callbacks.checkpoint_every_n_steps.save_top_k=${CHECKPOINT_EVERY_SAVE_TOP_K} \
  callbacks.checkpoint_every_n_steps.save_last=${CHECKPOINT_EVERY_SAVE_LAST} \
  callbacks.checkpoint_monitor.save_top_k=${CHECKPOINT_MONITOR_SAVE_TOP_K} \
  trainer.precision=bf16-mixed \
  training.ema=${TRAINING_EMA} \
  training.antithetic_sampling=${TRAINING_ANTITHETIC_SAMPLING} \
  training.loss_precision=bf16 \
  training.fault_tolerant=${TRAINING_FAULT_TOLERANT} \
  training.torch_compile=${TRAINING_TORCH_COMPILE} \
  training.log_train_aux_metrics=${TRAINING_LOG_TRAIN_AUX_METRICS} \
  training.sync_train_loss=${TRAINING_SYNC_TRAIN_LOSS} \
  perf.enabled=${PERF_ENABLED} \
  perf.log_every_n_steps=1 \
  perf.sync_cuda_for_timing=${PERF_SYNC_CUDA_FOR_TIMING} \
  perf.theoretical_peak_tflops_per_gpu=${PERF_PEAK_TFLOPS_PER_GPU} \
  callbacks.pytorch_profiler.enabled=${ENABLE_PYTORCH_PROFILER} \
  callbacks.pytorch_profiler.start_step=${PROFILE_START_STEP} \
  callbacks.pytorch_profiler.warmup_batches=${PROFILE_WARMUP_BATCHES} \
  callbacks.pytorch_profiler.active_batches=${PROFILE_ACTIVE_BATCHES} \
  callbacks.pytorch_profiler.row_limit=${PROFILE_ROW_LIMIT} \
  callbacks.pytorch_profiler.profile_all_ranks=${PROFILE_ALL_RANKS} \
  callbacks.pytorch_profiler.dirpath="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/\${RUN_NAME}/pytorch_profiler" \
  eval.generate_samples=${EVAL_GENERATE_SAMPLES} \
  checkpointing.resume_from_ckpt=${CHECKPOINT_RESUME_FROM_CKPT} \
  model.length=${MAX_LENGTH} \
  model.max_tokens=${MODEL_MAX_TOKENS} \
  model.dropout=0.0 \
  "\${MODEL_HYDRA_ARGS[@]}" \
  "\${OPTIM_HYDRA_ARGS[@]}" \
  loader.multiprocessing_context=spawn \
  loader.num_workers=${LOADER_NUM_WORKERS} \
  loader.pin_memory=true \
  hydra.run.dir="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/\${RUN_NAME}" \
  checkpointing.save_dir="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/\${RUN_NAME}/dummy_checkpoints" \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="\${RUN_NAME}" \
  wandb.project=${WANDB_PROJECT} \
  wandb.group="${WANDB_GROUP}" \
  wandb.job_type="train" \
  "\${WANDB_EXTRA_ARGS[@]}" \
  +wandb.tags='${WANDB_TAGS}'
SBATCH_EOF

                              submit_count=$((submit_count + 1))
                              echo "Submitted ${JOB_NAME}: nodes=${NODES}, ntasks=${NTASKS}, micro_gbs=${MICRO_GBS}, grad_accum=${GRAD_ACCUM_STEPS}, max_steps=${MAX_STEPS}"
                            done
                          done
                        done
                      done
                    done
                  done
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
echo "Container image: ${CE_IMAGE}"
echo "Container EDF: ${CONTAINER_ENV_FILE}"
echo "Container Python: ${CONTAINER_PYTHON}"
echo "Target nodes per training: ${TARGET_NODES}"
echo "Model config: ${MODEL_CONFIG}"
echo "GIDD loss type: ${GIDD_LOSS_TYPE}"
echo "Strategy config: ${STRATEGY_CONFIG}"
echo "Optimizer config: ${OPTIM_CONFIG}"
echo "Global batch sweep: ${GLOBAL_BATCH_SIZES[*]}"
echo "Base LR sweep: ${LRS[*]}"
if [[ "${OPTIM_CONFIG}" == "scion" ]]; then
  echo "SCION aux LR factors: ${SCION_AUX_LR_FACTORS[*]}"
  echo "SCION momenta: ${SCION_MOMENTA[*]}"
  echo "SCION scales: embed=${SCION_SCALE_EMBEDS[*]}, bias=${SCION_SCALE_BIASES[*]}, layer_norm=${SCION_SCALE_LAYER_NORMS[*]}, matrix=${SCION_SCALE_MATRICES[*]}"
  echo "SCION norm knobs: bias=${SCION_BIAS_NORMS[*]}, layer_norm=${SCION_LAYER_NORM_NORMS[*]}, spectral_steps=${SCION_SPECTRAL_NORM_STEPS[*]}"
  echo "SCION schedule knobs: warmup_iters=${SCION_WARMUP_ITERS[*]}, warmdown_pct=${SCION_WARMDOWN_PCTS[*]}, min_lr=${SCION_MIN_LRS[*]}"
  echo "SCION unconstrained values: ${SCION_UNCONSTRAINED_VALUES[*]}"
fi
echo "Data config: ${DATA_CONFIG}"
echo "Synthetic data mode: ${USE_SYNTHETIC_DATA}"
echo "Loader workers: ${LOADER_NUM_WORKERS}"
echo "PyTorch profiler enabled: ${ENABLE_PYTORCH_PROFILER}"
echo "PyTorch profiler window: start_step=${PROFILE_START_STEP}, warmup_batches=${PROFILE_WARMUP_BATCHES}, active_batches=${PROFILE_ACTIVE_BATCHES}, all_ranks=${PROFILE_ALL_RANKS}"
echo "W&B entity override: ${WANDB_ENTITY:-none}"
echo "W&B token file override: ${WANDB_TOKEN_FILE:-none}"
echo "Data source: ${PRETOK_LOCAL_DIR}"
echo "Data cache: ${DATA_CACHE_DIR}"
echo "Runtime cache root: ${CACHE_ROOT}"
