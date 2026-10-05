#!/usr/bin/env bash
set -euo pipefail

# Validates the `last.ckpt` from each run in the current
# submit_scion_hparam_sweep_newImg.sh sweep grid, using trainer.validate()
# (training.validate_only=true -- no training step, no weight drift, exact
# checkpoint weights). Mirrors that script's grid and Hydra overrides
# exactly (only checkpoint-resume/validate-only/output-path settings
# differ) so validation runs under the identical config that produced each
# checkpoint.
#
# Keep GLOBAL_BATCH_SIZES/LRS/MOMENTA/... and SOURCE_EXPERIMENT_NAME in
# sync with submit_scion_hparam_sweep_newImg.sh's current grid.

TOKEN_BUDGET=2000000000
MAX_LENGTH=2048
BATCH_SIZE_PER_GPU=8
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="ab035"
TIME_LIMIT="00:30:00"
SEED=4

SOURCE_EXPERIMENT_NAME="scion_new_runs_2026-07-27"
EXPERIMENT_NAME="${SOURCE_EXPERIMENT_NAME}_validate_only"

GLOBAL_BATCH_SIZES=(256)
LRS=(0.02)
MOMENTA=(0.08)
OSCALE_EMBEDS=(2000)
OSCALE_BIASES=(100)
OSCALE_LNS=(10)
OSCALE_MATRICES=(400)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

  MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))
  WARMDOWN_ITERS=$((MAX_STEPS * 28 / 100))

  for MOM in "${MOMENTA[@]}"; do
    for OSE in "${OSCALE_EMBEDS[@]}"; do
        for OSB in "${OSCALE_BIASES[@]}"; do
          for OSLN in "${OSCALE_LNS[@]}"; do
            for OSM in "${OSCALE_MATRICES[@]}"; do
              for LR in "${LRS[@]}"; do
                SOURCE_JOB_NAME="scion_gbs${GBS}_lr${LR}_mom${MOM}_ose${OSE}_osb${OSB}_osln${OSLN}_osm${OSM}_compare_de"
                JOB_NAME="${SOURCE_JOB_NAME}_validate"
                CKPT_PATH="\$SCRATCH/SwissAI-DLM-data/outputs/${SOURCE_EXPERIMENT_NAME}/${SOURCE_JOB_NAME}/dummy_checkpoints/checkpoints/last.ckpt"

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
  training.validate_only=true \
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
  perf.enabled=true \
  perf.theoretical_peak_tflops_per_gpu=989 \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path=${CKPT_PATH} \
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/${EXPERIMENT_NAME}/${JOB_NAME} \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="validate_${SOURCE_JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group=${EXPERIMENT_NAME} \
  wandb.job_type="validate" \
  +wandb.tags='[validate_only,scion,gidd_hf,gbs_${GBS},lr_${LR},mom_${MOM},ose_${OSE},osb_${OSB},osln_${OSLN},osm_${OSM}]'
SBATCH_EOF

                submit_count=$((submit_count + 1))
              done
            done
          done
        done
      done
    done
  done

echo "Submitted ${submit_count} validation jobs."
