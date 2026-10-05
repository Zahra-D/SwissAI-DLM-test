#!/usr/bin/env bash
set -euo pipefail

# Remeasure rho/sigma on the completed fixed-model batch sweep checkpoints.
#
# Each job resumes a step-7000 checkpoint, runs exactly one additional trainer
# step, and collects the trace immediately before that update. Therefore the
# reported metrics are evaluated at the saved checkpoint parameters; no source
# checkpoint is modified. The sole measurement change is
# algo.time_sampling_scope=global_batch.
#
# This answers: how would the *existing trained models* measure under a global
# low-discrepancy t grid? It does not retroactively change how those checkpoints
# were trained, and it intentionally leaves token-corruption RNG unchanged.
#
# Preview (default):
#   bash scripts/training/submit_scion_rho_sigma_global_time_remeasure.sh
# Submit all checkpoint probes:
#   DRY_RUN=0 bash scripts/training/submit_scion_rho_sigma_global_time_remeasure.sh

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
BATCH_SIZES=(${BATCH_SIZES:-64 128 256 512 1024 2048 4096})
SOURCE_STEPS="${SOURCE_STEPS:-7000}"
TRACE_M="${TRACE_M:-128}"
SEED="${SEED:-4}"
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
BATCH_SIZE_PER_GPU=8
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE=4
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
DRY_RUN="${DRY_RUN:-1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

# The saved checkpoints used the fixed Set-C SCION configuration.
SCION_LR=0.0225
SCION_MOMENTUM=0.0658
SCION_AUX_LR_FACTOR=0.02
SCION_SCALE_EMBED=3500
SCION_SCALE_BIAS=90
SCION_SCALE_LAYER_NORM=6.8
SCION_SCALE_MATRIX=336

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
SOURCE_ROOT="${SOURCE_ROOT:-${DATA_ROOT}/outputs/extending_budget/scion_rho_sigma_fixed7000steps_L12_H768_S2048}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_rho_sigma_global_time_remeasure_fixed_m128_L12H768_S2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/extending_budget/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Extending_Budget}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
WANDB_ID_PREFIX="${WANDB_ID_PREFIX:-rho_global_t_fixed_m128_remeasure}"

if (( SOURCE_STEPS < 1 || TRACE_M < 3 )); then
  echo "SOURCE_STEPS must be positive and TRACE_M must be at least 3." >&2
  exit 1
fi
if [[ ! -d "${PRETOK_LOCAL_DIR}" || ! -d "${DATA_CACHE_DIR}" ]]; then
  echo "Missing packed dataset/cache: ${PRETOK_LOCAL_DIR} | ${DATA_CACHE_DIR}" >&2
  exit 1
fi

MAX_STEPS=$((SOURCE_STEPS + 1))
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${WANDB_DIR}"
fi

echo "Checkpoint remeasurement: global t only, L12 H768 S${SEQUENCE_LENGTH}"
echo "Source checkpoints: ${SOURCE_ROOT}; resume step=${SOURCE_STEPS}, max_steps=${MAX_STEPS}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"
printf '%-8s %-8s %-8s %-9s %-8s %-10s %-12s\n' \
  "GBS" "nodes" "accum" "trace_m" "resume" "t_scope" "source_ckpt"

submit_count=0
for GBS in "${BATCH_SIZES[@]}"; do
  BATCH_PER_NODE=$((DEVICES_PER_NODE * BATCH_SIZE_PER_GPU))
  if (( GBS % BATCH_PER_NODE != 0 )); then
    echo "GBS=${GBS} must be divisible by per-node batch ${BATCH_PER_NODE}." >&2
    exit 1
  fi
  # Scale data-parallel workers with GBS so every measurement has accumulation
  # one. With fixed M this keeps per-rank trace work equal across batch sizes.
  NODES=$((GBS / BATCH_PER_NODE))
  NTASKS=$((NODES * NTASKS_PER_NODE))
  ACCUMULATION_STEPS=1

  SOURCE_CKPT="${SOURCE_ROOT}/gbs_${GBS}_seed_${SEED}/checkpoints/last.ckpt"
  if [[ ! -f "${SOURCE_CKPT}" ]]; then
    echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
    exit 1
  fi
  RUN_BASENAME="scion_rho_global_t_L12H768_gbs${GBS}_n${NODES}_acc${ACCUMULATION_STEPS}_seed${SEED}"
  RUN_STORAGE_DIR="${OUTPUT_ROOT}/gbs_${GBS}_seed_${SEED}"
  WANDB_RUN_ID="${WANDB_ID_PREFIX}-L12H768-B${GBS}-S${SEED}"
  printf '%-8s %-8s %-8s %-9s %-8s %-10s %-12s\n' \
    "${GBS}" "${NODES}" "${ACCUMULATION_STEPS}" "${TRACE_M}" "true" "global" "$(basename "${SOURCE_CKPT}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  PARTITION_DIRECTIVE=""
  if [[ -n "${PARTITION}" ]]; then
    PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
  fi
  "${SBATCH_BIN}" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${RUN_BASENAME}
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
RUN_ID="\${SLURM_JOB_ID:-manual}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${RUN_ID}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"
export WANDB_RESUME=allow
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
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
  strategy=ddp \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=${ACCUMULATION_STEPS} \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=1 \\
  trainer.gradient_clip_val=1.0 \\
  trainer.num_sanity_val_steps=0 \\
  trainer.val_check_interval=1000000 \\
  trainer.limit_val_batches=0 \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${BATCH_SIZE_PER_GPU} \\
  loader.eval_batch_size=${BATCH_SIZE_PER_GPU} \\
  +loader.exact_resume=false \\
  loader.num_workers=0 \\
  loader.pin_memory=false \\
  training.ema=0.0 \\
  training.antithetic_sampling=true \\
  training.loss_precision=bf16 \\
  training.fault_tolerant=false \\
  training.torch_compile=false \\
  training.log_train_aux_metrics=true \\
  training.sync_train_loss=true \\
  eval.generate_samples=false \\
  callbacks.pytorch_profiler.enabled=false \\
  callbacks.checkpoint_every_n_steps.save_top_k=0 \\
  callbacks.checkpoint_every_n_steps.save_last=false \\
  callbacks.checkpoint_monitor.save_top_k=0 \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=true \\
  checkpointing.resume_ckpt_path="${SOURCE_CKPT}" \\
  optim=scion \\
  optim.lr=${SCION_LR} \\
  optim.weight_decay=0.0 \\
  optim.momentum=${SCION_MOMENTUM} \\
  optim.aux_lr_factor=${SCION_AUX_LR_FACTOR} \\
  optim.scale_embed=${SCION_SCALE_EMBED} \\
  optim.scale_bias=${SCION_SCALE_BIAS} \\
  optim.scale_layer_norm=${SCION_SCALE_LAYER_NORM} \\
  optim.scale_matrix=${SCION_SCALE_MATRIX} \\
  optim.bias_norm=RowNorm \\
  optim.norm_layer_norm=BiasRMS \\
  optim.spectral_norm_steps=5 \\
  optim.unconstrained=false \\
  optim.warmup_iters=0 \\
  optim.warmdown_iters=0 \\
  optim.min_lr=1e-8 \\
  optim.trace_enabled=false \\
  optim.trace_collect_noise_stats=true \\
  optim.trace_noise_stats_every=1 \\
  optim.trace_noise_min_samples=3 \\
  optim.trace_m=${TRACE_M} \\
  +optim.trace_progress_every=${TRACE_M} \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${RUN_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_BASENAME}" \\
  wandb.id="${WANDB_RUN_ID}" \\
  wandb.notes=global_t_checkpoint_remeasurement_step_${SOURCE_STEPS}_fixed_m_${TRACE_M} \\
  +wandb.save_code=true \\
  wandb.job_type=rho-sigma-remeasure \\
  +wandb.tags="[extending_budget,rho_sigma,remeasure,global_time_sampling,global_t_only,fixed_trace_m,checkpoint_step_${SOURCE_STEPS},gbs_${GBS},nodes_${NODES},accum_${ACCUMULATION_STEPS},trace_m_${TRACE_M}]"
SBATCH_EOF
  submit_count=$((submit_count + 1))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only. Use DRY_RUN=0 to submit all checkpoint probes."
else
  echo "Submitted ${submit_count} checkpoint remeasurement jobs."
fi
