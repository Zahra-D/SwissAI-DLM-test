#!/usr/bin/env bash
set -euo pipefail

# Smoke/fit test for the dedicated 1F1B pipeline path.
#
# Current pipeline implementation is pure pipeline parallelism:
# - 1 node  => PP=4, DP=1
# - 2 nodes => PP=8, DP=1
#
# The goal is not throughput tuning yet. This checks that the 8B model can be
# constructed, split across ranks, run forward/backward with Schedule1F1B, and
# write per-rank checkpoints.

MAX_LENGTH=2048
MAX_STEPS=5000
SEED=4
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="ab035"
TIME_LIMIT="00:30:00"

# Keep microbatch size 1 for the first 8B fit test.
# In pipeline mode:
#   pipeline_microbatch = PIPELINE_BATCH_SIZE / NUM_MICROBATCHES
PIPELINE_MICROBATCH_SIZE=1
PIPELINE_BATCH_SIZES=(256)
NODE_COUNTS=(2)

LR=2e-2
MOMENTUM=0.08
OSCALE_EMBED=4000
OSCALE_BIAS=100
OSCALE_LN=10
OSCALE_MATRIX=400

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs/debug"

submit_count=0

for NODES in "${NODE_COUNTS[@]}"; do
  NTASKS=$((NODES * NTASKS_PER_NODE))
  PIPELINE_STAGES=${NTASKS}

  for PIPELINE_BATCH_SIZE in "${PIPELINE_BATCH_SIZES[@]}"; do
    if (( PIPELINE_BATCH_SIZE % PIPELINE_MICROBATCH_SIZE != 0 )); then
      echo "Skipping pipeline batch ${PIPELINE_BATCH_SIZE}: not divisible by microbatch ${PIPELINE_MICROBATCH_SIZE}"
      continue
    fi

    NUM_MICROBATCHES=$((PIPELINE_BATCH_SIZE / PIPELINE_MICROBATCH_SIZE))
    DDP_SAFE_GLOBAL_BATCH_SIZE=$((PIPELINE_BATCH_SIZE * NTASKS))
    if (( NUM_MICROBATCHES < PIPELINE_STAGES )); then
      echo "Warning: num_microbatches=${NUM_MICROBATCHES} < pipeline stages=${PIPELINE_STAGES}; bubble will be large."
    fi

    JOB_NAME="pipe1f1b_8b_pp${PIPELINE_STAGES}_nodes${NODES}_pbs${PIPELINE_BATCH_SIZE}_mb${PIPELINE_MICROBATCH_SIZE}"

    sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${REPO_ROOT}/logs/debug/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/debug/%x_%j.err
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
export NCCL_DEBUG=WARN
export MASTER_ADDR="\$(scontrol show hostnames "\${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT="\$((29500 + (SLURM_JOB_ID % 1000)))"
export WORLD_SIZE="\${SLURM_NTASKS}"

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
  data=synthetic-gidd \
  model=gidd_hf_8b \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  lr_scheduler=constant_warmup \
  ++parallel.pipeline.enabled=true \
  ++parallel.pipeline.schedule=1f1b \
  ++parallel.pipeline.batch_size=${PIPELINE_BATCH_SIZE} \
  ++parallel.pipeline.num_microbatches=${NUM_MICROBATCHES} \
  ++parallel.pipeline.checkpoint_every_n_steps=0 \
  training.ema=0.0 \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.accumulate_grad_batches=1 \
  trainer.log_every_n_steps=1 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  eval.generate_samples=false \
  perf.enabled=True \
  loader.global_batch_size=${DDP_SAFE_GLOBAL_BATCH_SIZE} \
  loader.eval_global_batch_size=${DDP_SAFE_GLOBAL_BATCH_SIZE} \
  loader.batch_size=${PIPELINE_BATCH_SIZE} \
  loader.eval_batch_size=${PIPELINE_BATCH_SIZE} \
  loader.num_workers=0 \
  loader.pin_memory=true \
  model.length=${MAX_LENGTH} \
  model.activation_checkpointing=true \
  model.attn_backend=flash_attn2 \
  optim=scion \
  optim.lr=${LR} \
  optim.weight_decay=0.00 \
  optim.momentum=${MOMENTUM} \
  optim.scale_embed=${OSCALE_EMBED} \
  optim.scale_bias=${OSCALE_BIAS} \
  optim.scale_layer_norm=${OSCALE_LN} \
  optim.scale_matrix=${OSCALE_MATRIX} \
  optim.unconstrained=false \
  optim.trace_enabled=false \
  optim.trace_collect_noise_stats=false \
  optim.warmup_iters=0 \
  optim.warmdown_iters=0 \
  optim.min_lr=1e-8 \
  checkpointing.save_dir=\$SCRATCH/SwissAI-DLM-data/outputs/debug/${JOB_NAME}/checkpoints \
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/debug/${JOB_NAME} \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group="pipeline_1f1b_8b_debug" \
  wandb.job_type="pipeline-smoke" \
  +wandb.tags='[pipeline,1f1b,8b,smoke,pp_${PIPELINE_STAGES},nodes_${NODES},pbs_${PIPELINE_BATCH_SIZE},mb_${PIPELINE_MICROBATCH_SIZE}]'
SBATCH_EOF

    submit_count=$((submit_count + 1))
  done
done

echo "Submitted ${submit_count} pipeline smoke jobs."
