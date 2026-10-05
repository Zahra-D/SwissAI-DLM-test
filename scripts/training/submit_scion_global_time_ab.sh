#!/usr/bin/env bash
set -euo pipefail

# Controlled three-HP comparison for globally sharded GIDD low-discrepancy
# time sampling, with gradient clipping disabled for the BST-paper test.
#
# This recreates three 2B-token, GBS-256 L12/H768 SCION HP sets from
# submit_scion_hparam_sweep_newImg.sh / scion_new_runs_2026-07-27. Relative to
# those historical runs, the controlled changes are:
#
#   algo.time_sampling_scope=global_batch
#   trainer.gradient_clip_val=0.0
#
# That creates one B_global-point randomized low-discrepancy t-grid and assigns
# each rank its own local slice.  It does NOT change token-corruption RNG yet;
# therefore this experiment isolates the effect of global t coverage.
#
# Preview (default):
#   bash scripts/training/submit_scion_global_time_ab.sh
# Submit all three runs:
#   DRY_RUN=0 bash scripts/training/submit_scion_global_time_ab.sh
# Short debug version, retaining the same topology/HPs:
#   DRY_RUN=0 TOKEN_BUDGET=1310720 TIME_LIMIT=00:45:00 bash scripts/training/submit_scion_global_time_ab.sh

TOKEN_BUDGET="${TOKEN_BUDGET:-2000000000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
GBS="${GBS:-256}"
LOCAL_BATCH="${LOCAL_BATCH:-8}"
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
GPUS_PER_NODE=4
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SEED="${SEED:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DRY_RUN="${DRY_RUN:-1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"

# Use the packed dataset and cache that were verified by the rho/sigma runs.
# The older home-directory cache path is not present on compute nodes.
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/extending_budget/scion_global_time_3hp_no_gradient_clipping_L12H768_GBS${GBS}_2B}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/scion_global_time_3hp_no_gradient_clipping_L12H768_GBS${GBS}_2B}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"
WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-test}"
WANDB_GROUP="${WANDB_GROUP:-scion_global_time_3hp_no_gradient_clipping_L12H768_GBS${GBS}_2B}"

if (( TOKEN_BUDGET <= 0 || SEQUENCE_LENGTH <= 0 || GBS <= 0 || LOCAL_BATCH <= 0 )); then
  echo "TOKEN_BUDGET, SEQUENCE_LENGTH, GBS, and LOCAL_BATCH must be positive." >&2
  exit 1
fi
if (( GBS % (LOCAL_BATCH * NTASKS_PER_NODE) != 0 )); then
  echo "GBS=${GBS} must be divisible by one-node batch $((LOCAL_BATCH * NTASKS_PER_NODE))." >&2
  exit 1
fi
if (( NUM_WORKERS < 1 )); then
  echo "NUM_WORKERS must be at least 1 for the spawn-wrapper data path." >&2
  exit 1
fi
if [[ ! -d "${PRETOK_LOCAL_DIR}" ]]; then
  echo "Missing pretokenized dataset: ${PRETOK_LOCAL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_CACHE_DIR}" ]]; then
  echo "Missing historical data cache: ${DATA_CACHE_DIR}" >&2
  exit 1
fi

NTASKS=$((GBS / LOCAL_BATCH))
NODES=$((NTASKS / NTASKS_PER_NODE))
MAX_STEPS=$((TOKEN_BUDGET / (SEQUENCE_LENGTH * GBS)))
WARM_DOWN_STEPS=$((MAX_STEPS * 28 / 100))
ACTUAL_TOKENS=$((MAX_STEPS * SEQUENCE_LENGTH * GBS))
PARTITION_DIRECTIVE=""
if [[ -n "${PARTITION}" ]]; then
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
fi

# All three configurations recorded in submit_scion_hparam_sweep_newImg.sh.
HP_NAMES=("baseline_lr002_mom008" "set_c" "compare_de")
HP_LRS=("0.02" "0.0225" "0.018")
HP_MOMENTA=("0.08" "0.0658" "0.0885")
HP_SCALE_EMBEDS=("2000" "3500" "3905")
HP_SCALE_BIASES=("100" "90" "44")
HP_SCALE_LNS=("10" "6.8" "10")
HP_SCALE_MATRICES=("400" "336" "430")

echo "Global-time three-HP comparison: L12 H768, S=${SEQUENCE_LENGTH}, GBS=${GBS}, local batch=${LOCAL_BATCH}"
echo "Topology: ${NODES} nodes, ${NTASKS} ranks, accumulation=1"
echo "Budget: ${MAX_STEPS} steps, ${ACTUAL_TOKENS} actual tokens, warmdown=${WARM_DOWN_STEPS} steps"
echo "Dataset/cache: ${PRETOK_LOCAL_DIR} | ${DATA_CACHE_DIR}"
echo "Changes versus historical runs: global-batch t sampling and no gradient clipping"
echo "Gradient clipping: disabled (trainer.gradient_clip_val=0.0)"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
    "HP" "lr" "mom" "embed" "bias" "one_d" "matrix"
  for i in "${!HP_NAMES[@]}"; do
    printf '%-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
      "${HP_NAMES[$i]}" "${HP_LRS[$i]}" "${HP_MOMENTA[$i]}" \
      "${HP_SCALE_EMBEDS[$i]}" "${HP_SCALE_BIASES[$i]}" \
      "${HP_SCALE_LNS[$i]}" "${HP_SCALE_MATRICES[$i]}"
  done
  echo "Dry run only. Use DRY_RUN=0 to submit all three runs."
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${WANDB_DIR}"
for i in "${!HP_NAMES[@]}"; do
  HP_NAME="${HP_NAMES[$i]}"
  LR="${HP_LRS[$i]}"
  MOMENTUM="${HP_MOMENTA[$i]}"
  SCALE_EMBED="${HP_SCALE_EMBEDS[$i]}"
  SCALE_BIAS="${HP_SCALE_BIASES[$i]}"
  SCALE_LN="${HP_SCALE_LNS[$i]}"
  SCALE_MATRIX="${HP_SCALE_MATRICES[$i]}"
  RUN_NAME="scion_global_time_no_gradient_clipping_${HP_NAME}_gbs${GBS}_lr${LR}_mom${MOMENTUM}_ose${SCALE_EMBED}_osb${SCALE_BIAS}_osln${SCALE_LN}_osm${SCALE_MATRIX}"
  RUN_STORAGE_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  "${SBATCH_BIN}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
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

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"
export WANDB_RESUME=allow
export WANDB_SAVE_CODE=true
export HYDRA_FULL_ERROR=1
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
export TMPDIR="\${JOB_TMPDIR}"
export TMP="\${JOB_TMPDIR}"
export TEMP="\${JOB_TMPDIR}"
export XDG_CACHE_HOME="\${JOB_TMPDIR}/xdg"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
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
  lr_scheduler=constant_warmup \\
  strategy=ddp \\
  trainer.deterministic=false \\
  trainer.num_nodes=${NODES} \\
  trainer.devices=${DEVICES_PER_NODE} \\
  trainer.accumulate_grad_batches=1 \\
  trainer.max_steps=${MAX_STEPS} \\
  trainer.log_every_n_steps=25 \\
  trainer.gradient_clip_val=0.0 \\
  trainer.num_sanity_val_steps=2 \\
  trainer.val_check_interval=250 \\
  trainer.limit_val_batches=0.5 \\
  trainer.precision=bf16-mixed \\
  loader.global_batch_size=${GBS} \\
  loader.eval_global_batch_size=${GBS} \\
  loader.batch_size=${LOCAL_BATCH} \\
  loader.eval_batch_size=${LOCAL_BATCH} \\
  loader.multiprocessing_context=spawn \\
  loader.num_workers=${NUM_WORKERS} \\
  loader.pin_memory=true \\
  training.ema=0.0 \\
  training.antithetic_sampling=true \\
  training.loss_precision=bf16 \\
  training.fault_tolerant=true \\
  training.torch_compile=false \\
  training.log_train_aux_metrics=true \\
  training.sync_train_loss=true \\
  eval.generate_samples=false \\
  callbacks.pytorch_profiler.enabled=false \\
  perf.enabled=true \\
  perf.theoretical_peak_tflops_per_gpu=989 \\
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \\
  callbacks.checkpoint_every_n_steps.save_top_k=1 \\
  callbacks.checkpoint_every_n_steps.save_last=true \\
  callbacks.checkpoint_monitor.save_top_k=0 \\
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \\
  checkpointing.resume_from_ckpt=false \\
  optim=scion \\
  optim.lr=${LR} \\
  optim.weight_decay=0.0 \\
  optim.momentum=${MOMENTUM} \\
  optim.aux_lr_factor=0.02 \\
  optim.scale_embed=${SCALE_EMBED} \\
  optim.scale_bias=${SCALE_BIAS} \\
  optim.scale_layer_norm=${SCALE_LN} \\
  optim.scale_matrix=${SCALE_MATRIX} \\
  optim.bias_norm=RowNorm \\
  optim.norm_layer_norm=BiasRMS \\
  optim.spectral_norm_steps=5 \\
  optim.unconstrained=false \\
  optim.warmup_iters=0 \\
  optim.warmdown_iters=${WARM_DOWN_STEPS} \\
  optim.min_lr=1e-8 \\
  optim.trace_enabled=false \\
  optim.trace_collect_noise_stats=false \\
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \\
  wandb.save_dir="${WANDB_DIR}" \\
  +wandb.entity="${WANDB_ENTITY}" \\
  wandb.project="${WANDB_PROJECT}" \\
  wandb.group="${WANDB_GROUP}" \\
  wandb.name="${RUN_NAME}" \\
  wandb.notes=global_t_three_hp_controlled_comparison_2B_no_gradient_clipping \\
  +wandb.save_code=true \\
  wandb.job_type=train \\
  +wandb.tags="[extending_budget,scion,gidd_hf,global_time_sampling,global_t_only,controlled_three_hp,no_gradient_clipping,${HP_NAME},gbs_${GBS},local_batch_${LOCAL_BATCH},nodes_${NODES},tokens_${ACTUAL_TOKENS}]"
SBATCH_EOF
done

echo "Submitted ${#HP_NAMES[@]} global-time no-clipping comparison runs."
