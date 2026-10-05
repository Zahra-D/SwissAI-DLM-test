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
BATCH_SIZE_PER_GPU=4
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="a137"
TIME_LIMIT="4:00:00"
SEED=4

# GLOBAL_BATCH_SIZES=(64 128 256 512)
# LRS=(1e-3 1e-4 1e-5 )
# MOMENTA=(0.02 0.08 0.1 0.2)
# OSCALE_EMBEDS=(100 3000 10000)
# OSCALE_BIASES=(1 10 100)
# OSCALE_LNS=(1 10 100)
 
 #first round of sweeps 
# GLOBAL_BATCH_SIZES=(64 128 256 512)
# LRS=(1e-1 5e-2 1e-2 1e-3)
# MOMENTA=(0.02  0.1 0.2 0.8)
# OSCALE_EMBEDS=(3000)
# OSCALE_BIASES=(100)
# OSCALE_LNS=(100)


 
#  #second round of sweeps 
# GLOBAL_BATCH_SIZES=( 64 128 256 512)
# LRS=(5e-3 7e-3 1e-2 2e-2 5e-2)
# MOMENTA=(0.02 0.06 0.08 0.1 0.15 0.2)
# OSCALE_EMBEDS=(3000)
# OSCALE_BIASES=(100)
# OSCALE_LNS=(100)


# #  #fourth round of sweeps 
# GLOBAL_BATCH_SIZES=(128)
# LRS=(2e-2)
# MOMENTA=(0.08)
# OSCALE_EMBEDS=(400 4000 8000)
# OSCALE_BIASES=(50 100)
# OSCALE_LNS=(1 5 10)
# OSCALE_MATRICES=(20 200 400)

# test 
GLOBAL_BATCH_SIZES=(128)
LRS=(2e-2)
MOMENTA=(0.08)
OSCALE_EMBEDS=(2000)
OSCALE_BIASES=(100)
OSCALE_LNS=(10)
OSCALE_MATRICES=(400)




SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs"

submit_count=0

for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do
  NTASKS=$((GBS / BATCH_SIZE_PER_GPU))
  NODES=$((NTASKS / NTASKS_PER_NODE))

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
                JOB_NAME="scion_gbs${GBS}_lr${LR}_mom${MOM}_ose${OSE}_osb${OSB}_osln${OSLN}_osm${OSM}"

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

export PYTHONPATH="\${REPO_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
export HYDRA_FULL_ERROR=1

mkdir -p "\${WANDB_DIR}"

srun --environment=uni-d2 \
     --ntasks=${NTASKS} \
     --ntasks-per-node=${NTASKS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     "${VENV_PYTHON}" -u -m discrete_diffusion \
  seed=${SEED} \
  data=nemotron-cc-pretok \
  model=gidd_hf \
  algo=gidd \
  algo.loss_type=gidd_easydel \
  algo.hybrid_mixing_shift=-1000 \
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
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/sweeping_experiments/hparam_sweep_scion_woEMA_fourth_round/${JOB_NAME}_test_speedup \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="woEMA_${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group="scion_nemotron_hparam_sweep_woEMA_fourth_round" \
  wandb.job_type="train" \
  +wandb.tags='[woEMA,fourth_round,scion,gidd_hf,sweep,gbs_${GBS},lr_${LR},mom_${MOM},ose_${OSE},osb_${OSB},osln_${OSLN},osm_${OSM}]'
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
