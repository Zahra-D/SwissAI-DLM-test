#!/usr/bin/env bash
set -euo pipefail

# Single gidd_hf + scion smoke run.
#
# Differences from submit_scion_hparam_sweep.sh:
# - Submits one sbatch job instead of looping over the full hyperparameter grid.
# - Defaults to one node: GBS=16, BATCH_SIZE_PER_GPU=4, ntasks=4, nodes=1.
# - Defaults to a short smoke budget: TOKEN_BUDGET=3,276,800, which gives
#   max_steps=100 at MAX_LENGTH=2048 and GBS=16.
# - Uses one conservative Scion point from the sweep grid:
#   LR=1e-3, MOMENTUM=0.1, OSCALE_EMBED=3000, OSCALE_BIAS=100,
#   OSCALE_LN=100, OSCALE_MATRIX=50.
# - Uses SOURCE_PRETOK_LOCAL_DIR as the full Nemotron parquet source, but by
#   default creates a symlinked PRETOK_SUBSET_FILES=128 parquet smoke subset
#   under $SCRATCH/SwissAI-DLM-data/smoke.
# - Passes data.pretok_local_dir to that subset by default; set
#   USE_PRETOK_SUBSET=0 to run against the full parquet directory.
# - Overrides data.cache_dir to a separate smoke cache so the processed
#   fixed-length dataset does not collide with the full-run cache.
# - Sets DISCRETE_DIFFUSION_SCRATCH_DIR, hydra.run.dir, and outputs under
#   $SCRATCH/SwissAI-DLM-data/smoke.
# - Uses wandb.group=scion_nemotron_single and "single" W&B tags.
#
# Useful overrides:
#   PRETOK_SUBSET_FILES=32 bash scripts/training/submit_scion_single_run.sh
#   USE_PRETOK_SUBSET=0 GBS=256 TOKEN_BUDGET=2000000000 bash scripts/training/submit_scion_single_run.sh
#   LR=1e-2 MOMENTUM=0.02 OSCALE_EMBED=10000 OSCALE_MATRIX=100 bash scripts/training/submit_scion_single_run.sh

TOKEN_BUDGET="${TOKEN_BUDGET:-3276800}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
GBS="${GBS:-16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ACCOUNT="${ACCOUNT:-a137}"
TIME_LIMIT="${TIME_LIMIT:-04:30:00}"

LR="${LR:-1e-3}"
MOMENTUM="${MOMENTUM:-0.1}"
OSCALE_EMBED="${OSCALE_EMBED:-3000}"
OSCALE_BIAS="${OSCALE_BIAS:-100}"
OSCALE_LN="${OSCALE_LN:-100}"
OSCALE_MATRIX="${OSCALE_MATRIX:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TOKEN_FILE="${WANDB_TOKEN_FILE:-}"
SOURCE_PRETOK_LOCAL_DIR="${SOURCE_PRETOK_LOCAL_DIR:-${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok}"
USE_PRETOK_SUBSET="${USE_PRETOK_SUBSET:-1}"
PRETOK_SUBSET_FILES="${PRETOK_SUBSET_FILES:-128}"
SMOKE_ROOT="${SMOKE_ROOT:-${SCRATCH}/SwissAI-DLM-data/smoke}"
PRETOK_SUBSET_DIR="${PRETOK_SUBSET_DIR:-${SMOKE_ROOT}/gidd-nemotron-cc-pretok-${PRETOK_SUBSET_FILES}files}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${SMOKE_ROOT}/training-data/nemotron-cc-pretok-${PRETOK_SUBSET_FILES}files}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SMOKE_ROOT}/outputs}"
CE_IMAGE="${CE_IMAGE:?Set CE_IMAGE to the shared .sqsh path}"
CONTAINER_ENV_FILE="${CONTAINER_ENV_FILE:-${SMOKE_ROOT}/uni-d2-ce.toml}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-/usr/bin/python}"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs"

if [[ ! -f "${CE_IMAGE}" ]]; then
  echo "Missing CE image: ${CE_IMAGE}" >&2
  exit 1
fi

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

if [[ ! -d "${SOURCE_PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing source pretokenized Nemotron data directory: ${SOURCE_PRETOK_LOCAL_DIR}" >&2
  echo "Set SOURCE_PRETOK_LOCAL_DIR to the shared parquet source directory." >&2
  exit 1
fi

if [[ "${USE_PRETOK_SUBSET}" == "1" ]]; then
  mkdir -p "${PRETOK_SUBSET_DIR}"
  existing_subset_files=$(find "${PRETOK_SUBSET_DIR}" -type l -name '*.parquet' | wc -l)
  if (( existing_subset_files < PRETOK_SUBSET_FILES )); then
    echo "Creating ${PRETOK_SUBSET_FILES}-file Nemotron smoke subset at ${PRETOK_SUBSET_DIR}"
    while IFS= read -r src; do
      rel="${src#${SOURCE_PRETOK_LOCAL_DIR}/}"
      mkdir -p "${PRETOK_SUBSET_DIR}/$(dirname "${rel}")"
      ln -sfn "${src}" "${PRETOK_SUBSET_DIR}/${rel}"
    done < <(find "${SOURCE_PRETOK_LOCAL_DIR}" -type f -name '*.parquet' | sort | head -n "${PRETOK_SUBSET_FILES}")
  fi
  PRETOK_LOCAL_DIR="${PRETOK_SUBSET_DIR}"
else
  PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${SOURCE_PRETOK_LOCAL_DIR}}"
fi

mkdir -p "${DATA_CACHE_DIR}" "${OUTPUT_ROOT}"

if (( GBS % BATCH_SIZE_PER_GPU != 0 )); then
  echo "GBS=${GBS} must be divisible by BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}" >&2
  exit 1
fi

NTASKS=$((GBS / BATCH_SIZE_PER_GPU))
if (( NTASKS % NTASKS_PER_NODE != 0 )); then
  echo "ntasks=${NTASKS} must be divisible by NTASKS_PER_NODE=${NTASKS_PER_NODE}" >&2
  exit 1
fi

NODES=$((NTASKS / NTASKS_PER_NODE))
MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))
WARMDOWN_ITERS=$((MAX_STEPS * 28 / 100))

JOB_NAME="${JOB_NAME:-scion_single_gbs${GBS}_lr${LR}_mom${MOMENTUM}_ose${OSCALE_EMBED}_osb${OSCALE_BIAS}_osln${OSCALE_LN}_osm${OSCALE_MATRIX}}"

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

export PYTHONPATH="\${REPO_ROOT}:\${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SMOKE_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export TMP="\${JOB_TMPDIR}"
export TEMP="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="${SMOKE_ROOT}/cache/xdg"
export HF_HOME="${SMOKE_ROOT}/cache/hf"
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
  "\${HF_DATASETS_CACHE}" \
  "\${HUGGINGFACE_HUB_CACHE}" \
  "\${TRANSFORMERS_CACHE}" \
  "\${WANDB_DIR}" \
  "\${DISCRETE_DIFFUSION_SCRATCH_DIR}" \
  "${DATA_CACHE_DIR}" \
  "${OUTPUT_ROOT}"

echo "Runtime temp dir: \${TMPDIR}"
echo "Runtime XDG cache: \${XDG_CACHE_HOME}"
echo "Runtime HF cache: \${HF_HOME}"

srun --environment="${CONTAINER_ENV_FILE}" \
     --ntasks=${NTASKS} \
     --ntasks-per-node=${NTASKS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     "${CONTAINER_PYTHON}" -u -m discrete_diffusion \
  data=nemotron-cc-pretok \
  data.pretok_local_dir="${PRETOK_LOCAL_DIR}" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model=gidd_hf \
  algo=gidd \
  algo.loss_type=gidd_easydel \
  sampling.predictor=gidd \
  sampling.sampler._target_=discrete_diffusion.sampling.gidd.GIDDSampler \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  lr_scheduler=constant_warmup \
  strategy=ddp \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.max_steps=${MAX_STEPS} \
  loader.global_batch_size=${GBS} \
  loader.batch_size=${BATCH_SIZE_PER_GPU} \
  loader.eval_batch_size=${BATCH_SIZE_PER_GPU} \
  trainer.log_every_n_steps=25 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=0.5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  model.length=${MAX_LENGTH} \
  model.dropout=0.0 \
  optim=scion \
  optim.lr=${LR} \
  optim.weight_decay=0.00 \
  optim.momentum=${MOMENTUM} \
  optim.scale_embed=${OSCALE_EMBED} \
  optim.scale_bias=${OSCALE_BIAS} \
  optim.scale_layer_norm=${OSCALE_LN} \
  optim.scale_matrix=${OSCALE_MATRIX} \
  optim.unconstrained=false \
  optim.warmup_iters=0 \
  optim.warmdown_iters=${WARMDOWN_ITERS} \
  optim.min_lr=1e-8 \
  loader.multiprocessing_context=spawn \
  loader.num_workers=4 \
  loader.pin_memory=true \
  hydra.run.dir="${OUTPUT_ROOT}/scion_single/${JOB_NAME}" \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group="scion_nemotron_single" \
  wandb.job_type="train" \
  "\${WANDB_EXTRA_ARGS[@]}" \
  +wandb.tags='[scion,gidd_hf,single,gbs_${GBS},lr_${LR},mom_${MOMENTUM},ose_${OSCALE_EMBED},osb_${OSCALE_BIAS},osln_${OSCALE_LN},osm_${OSCALE_MATRIX}]'
SBATCH_EOF

echo "Submitted ${JOB_NAME}: nodes=${NODES}, ntasks=${NTASKS}, max_steps=${MAX_STEPS}, warmdown_iters=${WARMDOWN_ITERS}"
echo "Container image: ${CE_IMAGE}"
echo "Container EDF: ${CONTAINER_ENV_FILE}"
echo "Container Python: ${CONTAINER_PYTHON}"
echo "W&B entity override: ${WANDB_ENTITY:-none}"
echo "W&B token file override: ${WANDB_TOKEN_FILE:-none}"
echo "Data source: ${PRETOK_LOCAL_DIR}"
echo "Data cache: ${DATA_CACHE_DIR}"
echo "Outputs: ${OUTPUT_ROOT}/scion_single/${JOB_NAME}"
