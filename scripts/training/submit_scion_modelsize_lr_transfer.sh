#!/usr/bin/env bash
set -euo pipefail

# Test whether the CompleteP/SCION base learning rate transfers across model
# sizes.  This follows submit_scion_hparam_sweep_newImg.sh exactly, except that
# width and depth are overridden for each model-size variant and the base LR is
# swept around the 100M-model candidate.
#
# Default model variants:
#   small:  H=384, L=6
#   medium: H=576, L=9
# The existing H=768, L=12 runs in EXPERIMENT_NAME provide the large-model
# reference.  Set INCLUDE_BASELINE=1 to submit that reference again with the
# same LR grid and make the comparison self-contained.

TOKEN_BUDGET=2000000000
MAX_LENGTH=2048
BATCH_SIZE_PER_GPU=16
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK=72
GPUS_PER_NODE=4
ACCOUNT="ab035"
TIME_LIMIT="00:45:00"
SEED=4

EXPERIMENT_NAME="scion_new_runs_2026-07-27"

# Keep the batch size and all SCION settings from the reference job fixed.
GLOBAL_BATCH_SIZES=(256)
LRS=(0.01 0.02 0.05 0.005)
MOMENTA=(0.08)
OSCALE_EMBEDS=(2000)
OSCALE_BIASES=(100)
OSCALE_LNS=(10)
OSCALE_MATRICES=(400)

# Width and depth are changed together.  Keep n_heads=12, matching the
# existing model-size jobs and the gidd_hf base configuration.
MODEL_TAGS=(small medium)
MODEL_LAYERS=(6 9)
MODEL_HIDDEN=(384 576)
MODEL_HEADS=(12 12)

if [[ "${INCLUDE_BASELINE:-0}" == "1" ]]; then
  MODEL_TAGS+=(base)
  MODEL_LAYERS+=(12)
  MODEL_HIDDEN+=(768)
  MODEL_HEADS+=(12)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
WANDB_DIR="${REPO_ROOT}/wandb_logs"
mkdir -p "${WANDB_DIR}" "${REPO_ROOT}/logs/${EXPERIMENT_NAME}"

if (( ${#MODEL_TAGS[@]} != ${#MODEL_LAYERS[@]} ||
      ${#MODEL_TAGS[@]} != ${#MODEL_HIDDEN[@]} ||
      ${#MODEL_TAGS[@]} != ${#MODEL_HEADS[@]} )); then
  echo "Model variant arrays must have equal lengths" >&2
  exit 1
fi

submit_count=0

for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do
  # The debug partition is limited to two nodes.  Derive the number of Slurm
  # tasks from the allocation, then use gradient accumulation to reach GBS.
  NODES=2
  NTASKS=$((NODES * NTASKS_PER_NODE))
  MICRO_GLOBAL_BATCH_SIZE=$((NODES * DEVICES_PER_NODE * BATCH_SIZE_PER_GPU))

  if (( GBS % BATCH_SIZE_PER_GPU != 0 )); then
    echo "Skipping global batch size ${GBS}: not divisible by per-GPU batch size ${BATCH_SIZE_PER_GPU}" >&2
    continue
  fi
  if (( GBS % MICRO_GLOBAL_BATCH_SIZE != 0 )); then
    echo "Skipping global batch size ${GBS}: not divisible by micro global batch size ${MICRO_GLOBAL_BATCH_SIZE}" >&2
    continue
  fi
  ACCUMULATION_STEPS=$((GBS / MICRO_GLOBAL_BATCH_SIZE))

  # Compare models at the same number of training tokens.  Integer floor is
  # intentional and matches the reference sweep.
  MAX_STEPS=$((TOKEN_BUDGET / (MAX_LENGTH * GBS)))
  WARMDOWN_ITERS=$((MAX_STEPS * 28 / 100))

  for MODEL_INDEX in "${!MODEL_TAGS[@]}"; do
    MODEL_TAG="${MODEL_TAGS[$MODEL_INDEX]}"
    N_LAYER="${MODEL_LAYERS[$MODEL_INDEX]}"
    HIDDEN_SIZE="${MODEL_HIDDEN[$MODEL_INDEX]}"
    N_HEADS="${MODEL_HEADS[$MODEL_INDEX]}"

    if (( HIDDEN_SIZE % N_HEADS != 0 )); then
      echo "Skipping ${MODEL_TAG}: hidden_size=${HIDDEN_SIZE} is not divisible by n_heads=${N_HEADS}" >&2
      continue
    fi
    INTERMEDIATE_SIZE=$((HIDDEN_SIZE * 4))

    for MOM in "${MOMENTA[@]}"; do
      for OSE in "${OSCALE_EMBEDS[@]}"; do
        for OSB in "${OSCALE_BIASES[@]}"; do
          for OSLN in "${OSCALE_LNS[@]}"; do
            for OSM in "${OSCALE_MATRICES[@]}"; do
              for LR in "${LRS[@]}"; do
                JOB_NAME="scion_${MODEL_TAG}_L${N_LAYER}_H${HIDDEN_SIZE}_gbs${GBS}_lr${LR}_mom${MOM}_ose${OSE}_osb${OSB}_osln${OSLN}_osm${OSM}_lr_transfer"

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
#SBATCH --partition=debug

set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
cd "\${REPO_ROOT}" || exit 1

export PYTHONPATH="\${REPO_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR}"
export HYDRA_FULL_ERROR=1

# Keep the same NCCL/cache setup as the reference job.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_NET_GDR_LEVEL=PHB
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER}-\${SLURM_JOB_ID}}"
export TMPDIR="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg-cache"
export HF_HOME="\${SCRATCH}/SwissAI-DLM-data/cache/hf"
export DISCRETE_DIFFUSION_SCRATCH_DIR="\${SCRATCH}/SwissAI-DLM-data/cache/discrete_diffusion"
mkdir -p "\${WANDB_DIR}" "\${TMPDIR}" "\${XDG_CACHE_HOME}" "\${HF_HOME}" "\${DISCRETE_DIFFUSION_SCRATCH_DIR}"

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
  trainer.accumulate_grad_batches=${ACCUMULATION_STEPS} \
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
  model.n_blocks=${N_LAYER} \
  model.hidden_size=${HIDDEN_SIZE} \
  model.intermediate_size=${INTERMEDIATE_SIZE} \
  model.n_heads=${N_HEADS} \
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
  hydra.run.dir=\$SCRATCH/SwissAI-DLM-data/outputs/${EXPERIMENT_NAME}/${JOB_NAME} \
  wandb.save_dir="\${WANDB_DIR}" \
  wandb.name="woEMA_${JOB_NAME}" \
  wandb.project=SwissAI_Scaling_Law \
  wandb.group=${EXPERIMENT_NAME} \
  wandb.job_type="train" \
  +wandb.tags='[woEMA,masked,scion,gidd_hf,lr_transfer,modelsize_${MODEL_TAG},gbs_${GBS},lr_${LR},mom_${MOM},layers_${N_LAYER},hidden_${HIDDEN_SIZE},heads_${N_HEADS},ose_${OSE},osb_${OSB},osln_${OSLN},osm_${OSM}]'
SBATCH_EOF

                submit_count=$((submit_count + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo "Submitted ${submit_count} jobs."
