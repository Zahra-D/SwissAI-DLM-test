#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for gidd_hf + scion
# Rules:
# - Total token budget per run: 2B tokens
# - iterations = 2B / (max_length * global_batch_size)
# - optim.warmdown_iters = 28% of iterations
# - loader.batch_size fixed at 4
# - ntasks = global_batch_size / 4
# - nodes = ntasks / 4 (4 tasks per node)

TOKEN_BUDGET=2000000000
MAX_LENGTH=2048
BATCH_SIZE_PER_GPU=8
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="ab035"
TIME_LIMIT="02:00:00"
SEED=4

# New optimizer parameterization: common FW stepsize for every group and
# Section B.3 boundary initialization. Keep it separate from historical runs.
EXPERIMENT_NAME="scion_b3_boundary_equal_group_lr_hparam_tuning_2026-08-31"

# GLOBAL_BATCH_SIZES=(512)
# LRS=(0.0225)
# MOMENTA=(0.0658)
# OSCALE_EMBEDS=(3500)
# OSCALE_BIASES=(90)
# OSCALE_LNS=(6.8)
# OSCALE_MATRICES=(336)

GLOBAL_BATCH_SIZES=(256)
LRS=(0.02)
MOMENTA=(0.08)
OSCALE_EMBEDS=(2000)
OSCALE_BIASES=(100)
OSCALE_LNS=(10)
OSCALE_MATRICES=(400)

# GLOBAL_BATCH_SIZES=(256)
# LRS=(0.018)
# MOMENTA=(0.0885)
# OSCALE_EMBEDS=(3905)
# OSCALE_BIASES=(44)
# OSCALE_LNS=(10)
# OSCALE_MATRICES=(430)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs/${EXPERIMENT_NAME}"

submit_count=0

for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do
  NTASKS=$((GBS / BATCH_SIZE_PER_GPU))
  NODES=$((NTASKS / NTASKS_PER_NODE))
  # NTASKS=8
  # NODES=2



  if (( NTASKS % NTASKS_PER_NODE != 0 )); then
    echo "Skipping global batch size ${GBS}: ntasks=${NTASKS} is not divisible by ${NTASKS_PER_NODE}"
    continue
  fi

  # Integer floor as requested formula implies discrete iterations.
  MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))
  WARMDOWN_ITERS=$((MAX_STEPS * 28 / 100))

  for MOM in "${MOMENTA[@]}"; do
    for OSE in "${OSCALE_EMBEDS[@]}"; do
        for OSB in "${OSCALE_BIASES[@]}"; do
          for OSLN in "${OSCALE_LNS[@]}"; do
            for OSM in "${OSCALE_MATRICES[@]}"; do
              for LR in "${LRS[@]}"; do
                JOB_NAME="scion_b3_equalLR_gbs${GBS}_lr${LR}_mom${MOM}_ose${OSE}_osb${OSB}_osln${OSLN}_osm${OSM}"

            sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${REPO_ROOT}/logs/${EXPERIMENT_NAME}/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/${EXPERIMENT_NAME}/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL



set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
cd "\${REPO_ROOT}" || exit 1

export PYTHONPATH="\${REPO_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
export HYDRA_FULL_ERROR=1

# Debugging the multi-node hang at distributed init (reproduced with and
# without checkpoint resume, so resume isn't the cause) -- get real NCCL
# diagnostics instead of just a generic barrier() warning and silence.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Ring log showed "GDR 0" (GPUDirect RDMA disabled) for cross-node
# connections -- CSCS docs recommend this for Alps' Slingshot topology.
# Testing whether it's what's slowing the post-setup DDP weight broadcast.
export NCCL_NET_GDR_LEVEL=PHB

# Safe Cache Isolation mapping to prevent /tmp saturation
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
  eval.generate_samples=false \
  algo.loss_weighting=dynamic \
  lr_scheduler=constant_warmup \
  strategy=ddp \
  training.ema=0.0 \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.max_steps=${MAX_STEPS} \
  loader.global_batch_size=${GBS} \
  loader.batch_size=${BATCH_SIZE_PER_GPU} \
  loader.eval_batch_size=${BATCH_SIZE_PER_GPU} \
  trainer.log_every_n_steps=25 \
  trainer.val_check_interval=250 \
  trainer.limit_val_batches=0.5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  trainer.precision=bf16-mixed \
  training.torch_compile=False \
  model.length=${MAX_LENGTH} \
  model.dropout=0.0 \
  model.activation_scale=2.0 \
  model.resid_scale=4.0 \
  optim=scion \
  optim.lr=${LR} \
  optim.equal_group_lr=true \
  optim.boundary_init=true \
  optim.weight_decay=0.00 \
  optim.momentum=${MOM} \
  optim.scale_embed=${OSE} \
  optim.scale_bias=${OSB} \
  optim.scale_layer_norm=${OSLN} \
  optim.scale_matrix=${OSM} \
  optim.unconstrained=false \
  optim.warmup_iters=0 \
  optim.warmdown_iters=${WARMDOWN_ITERS} \
  optim.min_lr=1e-8 \
  loader.multiprocessing_context=spawn \
  loader.num_workers=4 \
  loader.pin_memory=true \
  perf.enabled=true \
  perf.theoretical_peak_tflops_per_gpu=989 \
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/${EXPERIMENT_NAME}/${JOB_NAME} \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="woEMA_${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group=${EXPERIMENT_NAME} \
  wandb.job_type="train" \
  +wandb.tags='[woEMA,masked,scion,gidd_hf,sweep,b3_boundary_init,equal_group_lr,residual_4_over_D,gbs_${GBS},lr_${LR},mom_${MOM},ose_${OSE},osb_${OSB},osln_${OSLN},osm_${OSM}]'
SBATCH_EOF

                submit_count=$((submit_count + 1))
              done
            done
          done
        done
      done
    done
  done

echo "Submitted ${submit_count} jobs."
