#!/usr/bin/env bash
set -euo pipefail

# Fixed-model SCION rho/sigma sweep for the extending-budget project.
#
# Preview (default):
#   ./scripts/training/submit_scion_rho_sigma_batch_sweep.sh
# Submit all runs:
#   DRY_RUN=0 ./scripts/training/submit_scion_rho_sigma_batch_sweep.sh
# Continue existing runs beyond 7k from their periodic last.ckpt files:
#   DRY_RUN=0 MAX_STEPS=10000 ./scripts/training/submit_scion_rho_sigma_batch_sweep.sh
# Override the grid, for example:
#   DRY_RUN=0 BATCH_SIZES="128 256" ./scripts/training/submit_scion_rho_sigma_batch_sweep.sh

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZES=(${BATCH_SIZES:-64 128 256 512 1024 2048 4096})
MAX_STEPS="${MAX_STEPS:-7000}"
TRACE_REFERENCE_SEQUENCES="${TRACE_REFERENCE_SEQUENCES:-32768}"
TRACE_NOISE_EVERY="${TRACE_NOISE_EVERY:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
AUTO_RESUME="${AUTO_RESUME:-1}"
# Fast, CUDA-safe loading of the already packed Arrow dataset. Two workers per
# rank is the initial throughput choice; override only after a steady-state
# throughput test (e.g. RHO_LOADER_NUM_WORKERS=4). Twelve is unnecessary here
# and caused a very slow 96-worker startup in the two-node probe.
RHO_LOADER_NUM_WORKERS="${RHO_LOADER_NUM_WORKERS:-2}"

# Match the recent fast Set-C runs: use enough nodes to realize every global
# batch physically with no gradient accumulation. Each node contributes
# 4 GPUs * local batch 8 = 32 sequences per optimizer step.
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
BATCH_SIZE_PER_GPU=8
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE=4
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-6:00:00}"
SEED="${SEED:-4}"
DRY_RUN="${DRY_RUN:-1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

# Recent Set-C HPs from the 2B-token comparison report.
SCION_LR=0.0225
SCION_MOMENTUM=0.0658
SCION_AUX_LR_FACTOR=0.02
SCION_SCALE_EMBED=3500
SCION_SCALE_BIAS=90
SCION_SCALE_LAYER_NORM=6.8
SCION_SCALE_MATRIX=336
SCION_SPECTRAL_NORM_STEPS=5
SCION_BIAS_NORM=RowNorm
SCION_LAYER_NORM_NORM=BiasRMS

# Fixed 124M-class model used in the recent HP comparison.
N_BLOCKS=12
HIDDEN_SIZE=768
INTERMEDIATE_SIZE=3072
N_HEADS=12

WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Extending_Budget}"
WANDB_GROUP="${WANDB_GROUP:-scion_rho_sigma_fixed7000steps_L12_H768_S2048}"
WANDB_ID_PREFIX="${WANDB_ID_PREFIX:-rho7k}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
# Reuse the fully packed train/validation cache produced by precache job 3216116.
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_rho_sigma_fixed7000steps_L12_H768_S2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/extending_budget/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"

if [[ ! -d "${PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing pretokenized dataset: ${PRETOK_LOCAL_DIR}" >&2
  exit 1
fi
TRAIN_CACHE_PATH="${DATA_CACHE_DIR}/nemotron-cc-pretok-train_train_bs${SEQUENCE_LENGTH}_unwrapped_eosFalse_specialFalse.dat"
VALID_CACHE_PATH="${DATA_CACHE_DIR}/nemotron-cc-pretok-valid_validation_bs${SEQUENCE_LENGTH}_unwrapped_eosFalse_specialFalse.dat"
if [[ ! -d "${TRAIN_CACHE_PATH}" || ! -d "${VALID_CACHE_PATH}" ]]; then
  echo "Missing completed packed train/validation cache under: ${DATA_CACHE_DIR}" >&2
  echo "Expected: ${TRAIN_CACHE_PATH}" >&2
  echo "Expected: ${VALID_CACHE_PATH}" >&2
  exit 1
fi
if (( MAX_STEPS <= 0 || TRACE_NOISE_EVERY <= 0 || CHECKPOINT_EVERY <= 0 || TRACE_REFERENCE_SEQUENCES <= 0 )); then
  echo "MAX_STEPS, TRACE_NOISE_EVERY, CHECKPOINT_EVERY, and TRACE_REFERENCE_SEQUENCES must be positive." >&2
  exit 1
fi
if (( RHO_LOADER_NUM_WORKERS < 1 )); then
  echo "RHO_LOADER_NUM_WORKERS must be at least 1 for the spawn-wrapper run." >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${OUTPUT_ROOT}"
fi

echo "Fixed model: L${N_BLOCKS} H${HIDDEN_SIZE}, sequence length ${SEQUENCE_LENGTH}"
echo "Fixed HPs: lr=${SCION_LR}, momentum=${SCION_MOMENTUM}, radii=${SCION_SCALE_EMBED}/${SCION_SCALE_BIAS}/${SCION_SCALE_LAYER_NORM}/${SCION_SCALE_MATRIX}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"
echo "Packed dataset cache: ${DATA_CACHE_DIR}"
echo "Loader: spawn + lazy path-only dataset wrapper, workers/rank=${RHO_LOADER_NUM_WORKERS}, pin_memory=true"
echo "Resume: optimizer/model state resumes; exact sample-cursor resume is disabled because worker prefetch is enabled"
echo "Experiment output: ${OUTPUT_ROOT}"
printf '%-8s %-8s %-10s %-8s %-10s %-12s %-14s %-8s\n' "GBS" "nodes" "steps" "accum" "trace_m" "trace_every" "actual_tokens" "resume"

submit_count=0
for GBS in "${BATCH_SIZES[@]}"; do
  BATCH_PER_NODE=$((DEVICES_PER_NODE * BATCH_SIZE_PER_GPU))
  if (( GBS % BATCH_PER_NODE != 0 )); then
    echo "GBS=${GBS} is not divisible by per-node batch ${BATCH_PER_NODE}." >&2
    exit 1
  fi
  NODES_FOR_RUN=$((GBS / BATCH_PER_NODE))
  NTASKS=$((NODES_FOR_RUN * NTASKS_PER_NODE))
  MICRO_GLOBAL_BATCH=$((NODES_FOR_RUN * BATCH_PER_NODE))
  if (( TRACE_REFERENCE_SEQUENCES % GBS != 0 )); then
    echo "GBS=${GBS} does not divide TRACE_REFERENCE_SEQUENCES=${TRACE_REFERENCE_SEQUENCES}." >&2
    exit 1
  fi

  GRAD_ACCUM_STEPS=1
  TRACE_M=$((TRACE_REFERENCE_SEQUENCES / GBS))
  if (( TRACE_M < 3 )); then
    echo "Invalid trace plan for GBS=${GBS}: trace_m=${TRACE_M}." >&2
    exit 1
  fi
  ACTUAL_TOKENS=$((MAX_STEPS * SEQUENCE_LENGTH * GBS))
  RUN_BASENAME="scion_noise_L${N_BLOCKS}H${HIDDEN_SIZE}_gbs${GBS}_n${NODES_FOR_RUN}_seed${SEED}"
  RUN_STORAGE_DIR="${OUTPUT_ROOT}/gbs_${GBS}_seed_${SEED}"
  LAST_CKPT="${RUN_STORAGE_DIR}/checkpoints/last.ckpt"
  RESUME_FROM_CKPT=false
  if [[ "${AUTO_RESUME}" == "1" && -f "${LAST_CKPT}" ]]; then
    RESUME_FROM_CKPT=true
  fi
  WANDB_RUN_ID="${WANDB_ID_PREFIX}-L${N_BLOCKS}H${HIDDEN_SIZE}-B${GBS}-S${SEED}"
  printf '%-8s %-8s %-10s %-8s %-10s %-12s %-14s %-8s\n' \
    "${GBS}" "${NODES_FOR_RUN}" "${MAX_STEPS}" "${GRAD_ACCUM_STEPS}" "${TRACE_M}" "${TRACE_NOISE_EVERY}" "${ACTUAL_TOKENS}" "${RESUME_FROM_CKPT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  JOB_NAME="${RUN_BASENAME}"
  PARTITION_DIRECTIVE=""
  if [[ -n "${PARTITION}" ]]; then
    PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
  fi
  "${SBATCH_BIN}" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES_FOR_RUN}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
${PARTITION_DIRECTIVE}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"

RUN_ID="\${SLURM_JOB_ID:-manual}"
RUN_NAME="${RUN_BASENAME}"
export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${RUN_ID}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export MPLCONFIGDIR="\${JOB_TMPDIR}/matplotlib"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"
export WANDB_RESUME=allow
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
mkdir -p "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${MPLCONFIGDIR}" "\${WANDB_DIR}" "${RUN_STORAGE_DIR}"

# Fast, CUDA-safe packed-data path: workers spawn clean interpreters and open
# their own Arrow memory maps. The lightweight wrapper prevents HF shard
# metadata from being serialized to every worker. Multi-worker prefetch means
# checkpoints restore model/optimizer state but cannot restore the exact next
# sample cursor; the independent trace loader does not disturb training order.
srun --environment=uni-d2 \
  --ntasks=${NTASKS} \
  --ntasks-per-node=${NTASKS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  "${VENV_PYTHON}" -u -m discrete_diffusion \
  seed=${SEED} \
  data=nemotron-cc-pretok \
  data.pretok_local_dir="${PRETOK_LOCAL_DIR}" \
  data.cache_dir="${DATA_CACHE_DIR}" \
  model=gidd_hf \
  model.length=${SEQUENCE_LENGTH} \
  model.max_tokens=1024 \
  model.n_blocks=${N_BLOCKS} \
  model.hidden_size=${HIDDEN_SIZE} \
  model.intermediate_size=${INTERMEDIATE_SIZE} \
  model.n_heads=${N_HEADS} \
  model.dropout=0.0 \
  model.activation_scale=2.0 \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  algo.low_discrepancy_sampling=true \
  strategy=ddp \
  trainer.num_nodes=${NODES_FOR_RUN} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.log_every_n_steps=5 \
  trainer.gradient_clip_val=1.0 \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=1000000 \
  trainer.limit_val_batches=0 \
  trainer.precision=bf16-mixed \
  loader.global_batch_size=${GBS} \
  loader.eval_global_batch_size=${MICRO_GLOBAL_BATCH} \
  loader.batch_size=${BATCH_SIZE_PER_GPU} \
  loader.eval_batch_size=${BATCH_SIZE_PER_GPU} \
  +loader.exact_resume=false \
  loader.multiprocessing_context=spawn \
  loader.num_workers=${RHO_LOADER_NUM_WORKERS} \
  loader.pin_memory=true \
  +loader.lazy_spawn_dataset=true \
  training.ema=0.0 \
  training.antithetic_sampling=true \
  training.loss_precision=bf16 \
  training.fault_tolerant=false \
  training.torch_compile=false \
  training.log_train_aux_metrics=true \
  training.sync_train_loss=true \
  eval.generate_samples=false \
  callbacks.pytorch_profiler.enabled=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \
  callbacks.checkpoint_every_n_steps.save_top_k=1 \
  callbacks.checkpoint_every_n_steps.save_last=true \
  callbacks.checkpoint_monitor.save_top_k=0 \
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  checkpointing.resume_ckpt_path="${LAST_CKPT}" \
  optim=scion \
  optim.lr=${SCION_LR} \
  optim.weight_decay=0.0 \
  optim.momentum=${SCION_MOMENTUM} \
  optim.aux_lr_factor=${SCION_AUX_LR_FACTOR} \
  optim.scale_embed=${SCION_SCALE_EMBED} \
  optim.scale_bias=${SCION_SCALE_BIAS} \
  optim.scale_layer_norm=${SCION_SCALE_LAYER_NORM} \
  optim.scale_matrix=${SCION_SCALE_MATRIX} \
  optim.bias_norm=${SCION_BIAS_NORM} \
  optim.norm_layer_norm=${SCION_LAYER_NORM_NORM} \
  optim.spectral_norm_steps=${SCION_SPECTRAL_NORM_STEPS} \
  optim.unconstrained=false \
  optim.warmup_iters=0 \
  optim.warmdown_iters=0 \
  optim.min_lr=1e-8 \
  optim.trace_enabled=false \
  optim.trace_collect_noise_stats=true \
  optim.trace_noise_stats_every=${TRACE_NOISE_EVERY} \
  optim.trace_noise_min_samples=3 \
  optim.trace_m=${TRACE_M} \
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${RUN_ID}" \
  wandb.save_dir="${WANDB_DIR}" \
  +wandb.entity="${WANDB_ENTITY}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="\${RUN_NAME}" \
  wandb.id="${WANDB_RUN_ID}" \
  wandb.notes="Fixed-model rho/sigma estimation - 7000+ steps - trace every ${TRACE_NOISE_EVERY} - no warmup or warmdown" \
  +wandb.save_code=true \
  +wandb.log_model=${WANDB_LOG_MODEL} \
  wandb.job_type=train \
  +wandb.tags="[extending_budget,rho_sigma,batch_scaling,fixed_model,fixed_steps,set_c,no_warmup,no_warmdown,gbs_${GBS},nodes_${NODES_FOR_RUN},trace_m_${TRACE_M},trace_every_${TRACE_NOISE_EVERY},checkpoint_every_${CHECKPOINT_EVERY}]"
SBATCH_EOF

  submit_count=$((submit_count + 1))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only. Use DRY_RUN=0 to submit the sweep."
else
  echo "Submitted ${submit_count} rho/sigma jobs."
fi
