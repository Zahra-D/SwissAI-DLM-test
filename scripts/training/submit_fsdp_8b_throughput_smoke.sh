#!/bin/bash
set -euo pipefail

# Short AdamW+FSDP throughput smoke test for the ~7.9B GIDD model.
#
# This deliberately does NOT enable pipeline parallelism.  It runs ordinary
# sharded data parallel training, so
#   global_batch = nodes * gpus_per_node * local_microbatch * grad_accum.
#
# Defaults reproduce the node and global-batch axes of the throughput plot.
# A rank uses at most LOCAL_BATCH_SIZE samples; the script lowers it only when
# necessary to make an exact global batch (256 on 16 nodes becomes local batch
# 4, accumulation 1).
#
# It runs only WARMUP_STEPS + MEASURE_STEPS optimizer updates, uses synthetic
# data, disables validation/checkpoint saving/resume, and logs perf metrics on
# every optimizer update. Ignore the first WARMUP_STEPS perf points and use
# the median of the remaining measurements in W&B.
#
# This is intentionally AdamW. Current SCION requires whole 2-D matrix
# parameters for its spectral LMO, while FSDP exposes flattened/sharded
# parameters; selecting SCION here would fail before the first training step.

NODE_COUNTS=(${NODE_COUNTS:-1 2 4 8 16})
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
ACCOUNT="${ACCOUNT:-ab035}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"

GLOBAL_BATCH_SIZES=(${GLOBAL_BATCH_SIZES:-256 512 1024 2048})
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
MEASURE_STEPS="${MEASURE_STEPS:-5}"
MAX_STEPS=$((WARMUP_STEPS + MEASURE_STEPS))

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/venv-docker/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/outputs/fsdp_8b_throughput_smoke}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Scaling_Law-perf}"
WANDB_GROUP="${WANDB_GROUP:-fsdp_8b_adamw_throughput_smoke}"
OPTIM_NAME="${OPTIM_NAME:-adamw}"

if (( GPUS_PER_NODE <= 0 || NTASKS_PER_NODE <= 0 || LOCAL_BATCH_SIZE <= 0 )); then
  echo "GPUS_PER_NODE, NTASKS_PER_NODE, and LOCAL_BATCH_SIZE must be positive." >&2
  exit 1
fi

if (( GPUS_PER_NODE != NTASKS_PER_NODE )); then
  echo "This script expects one training rank per GPU: GPUS_PER_NODE must equal NTASKS_PER_NODE." >&2
  exit 1
fi

if [[ "${OPTIM_NAME}" != "adamw" ]]; then
  echo "Only OPTIM_NAME=adamw is supported by this FSDP smoke script." >&2
  echo "Current SCION cannot operate on FSDP-flattened/sharded matrices." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${WANDB_DIR}" "${REPO_ROOT}/logs"

submitted=0
for NODES in "${NODE_COUNTS[@]}"; do
  if (( NODES <= 0 )); then
    echo "Skipping invalid node count: ${NODES}" >&2
    continue
  fi

  TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
  for GLOBAL_BATCH_SIZE in "${GLOBAL_BATCH_SIZES[@]}"; do
    EFFECTIVE_LOCAL_BATCH=${LOCAL_BATCH_SIZE}
    while (( EFFECTIVE_LOCAL_BATCH > 1 )) \
      && (( GLOBAL_BATCH_SIZE % (TOTAL_GPUS * EFFECTIVE_LOCAL_BATCH) != 0 )); do
      EFFECTIVE_LOCAL_BATCH=$((EFFECTIVE_LOCAL_BATCH - 1))
    done

    MICRO_GLOBAL_BATCH=$((TOTAL_GPUS * EFFECTIVE_LOCAL_BATCH))
    if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
      echo "Skipping global batch ${GLOBAL_BATCH_SIZE} on ${NODES} nodes: no exact local batch <= ${LOCAL_BATCH_SIZE}." >&2
      continue
    fi

    GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))
    JOB_NAME="fsdp8b_gbs${GLOBAL_BATCH_SIZE}_mb${EFFECTIVE_LOCAL_BATCH}_n${NODES}_acc${GRAD_ACCUM_STEPS}_smoke"

    echo "Submitting ${JOB_NAME}"
    echo "  global batch: ${GLOBAL_BATCH_SIZE} = ${NODES} nodes * ${GPUS_PER_NODE} GPUs * ${EFFECTIVE_LOCAL_BATCH} local batch * ${GRAD_ACCUM_STEPS} accumulation"

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

set -euo pipefail
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
export HYDRA_FULL_ERROR=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${SCRATCH_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export HF_HOME="${SCRATCH_ROOT}/cache/hf"
mkdir -p "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${HF_HOME}" "${OUTPUT_ROOT}"

RUN_NAME="${JOB_NAME}_\${SLURM_JOB_ID}"

srun --environment=uni-d2 \
  --ntasks=${TOTAL_GPUS} \
  --ntasks-per-node=${NTASKS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  "${VENV_PYTHON}" -u -m discrete_diffusion \
  data=synthetic-gidd \
  model=gidd_hf_8b \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  strategy=fsdp \
  parallel.pipeline.enabled=false \
  seed=1 \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${GPUS_PER_NODE} \
  trainer.accumulate_grad_batches=${GRAD_ACCUM_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.log_every_n_steps=1 \
  trainer.gradient_clip_val=0.0 \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=1000000 \
  trainer.limit_val_batches=0 \
  trainer.precision=bf16-mixed \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
  loader.eval_global_batch_size=${EFFECTIVE_LOCAL_BATCH} \
  loader.batch_size=${EFFECTIVE_LOCAL_BATCH} \
  loader.eval_batch_size=${EFFECTIVE_LOCAL_BATCH} \
  loader.num_workers=0 \
  loader.pin_memory=true \
  model.length=${SEQ_LEN} \
  model.max_tokens=${SEQ_LEN} \
  model.fsdp_flatten_compatible=true \
  model.activation_checkpointing=true \
  training.ema=0.0 \
  training.torch_compile=false \
  training.fault_tolerant=false \
  training.log_train_aux_metrics=false \
  training.sync_train_loss=false \
  optim=${OPTIM_NAME} \
  optim.lr=3e-4 \
  optim.weight_decay=0.0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000000 \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last=false \
  callbacks.checkpoint_monitor.save_top_k=0 \
  callbacks.pytorch_profiler.enabled=false \
  eval.generate_samples=false \
  checkpointing.resume_from_ckpt=false \
  checkpointing.save_dir="${OUTPUT_ROOT}/\${RUN_NAME}/no_checkpoints" \
  hydra.run.dir="${OUTPUT_ROOT}/\${RUN_NAME}" \
  perf.enabled=true \
  perf.log_every_n_steps=1 \
  perf.sync_cuda_for_timing=true \
  perf.theoretical_peak_tflops_per_gpu=989 \
  wandb.save_dir="${WANDB_DIR}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.job_type=adamw-fsdp-throughput-smoke \
  wandb.name="\${RUN_NAME}" \
  +wandb.tags='[fsdp,8b,adamw,no_pipeline,synthetic,throughput_smoke,gbs_${GLOBAL_BATCH_SIZE},local_batch_${EFFECTIVE_LOCAL_BATCH},nodes_${NODES},accum_${GRAD_ACCUM_STEPS}]'
SBATCH_EOF
    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} short FSDP throughput jobs."
echo "Node counts: ${NODE_COUNTS[*]}"
echo "Global batch sizes: ${GLOBAL_BATCH_SIZES[*]}"
echo "Each job has ${WARMUP_STEPS} warm-up and ${MEASURE_STEPS} measured optimizer steps."
