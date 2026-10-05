#!/usr/bin/env bash
set -euo pipefail

# Figure-5-style SCION Frank-Wolfe stepsize sweep for the extending-budget
# experiment. Every run starts from initialization (unless an interrupted run
# already has last.ckpt and AUTO_RESUME=1), uses one seed, and logs train loss
# at every optimizer step so that the final 500-step arithmetic mean can be
# computed exactly.
#
# Preview all jobs (default):
#   bash scripts/training/submit_scion_extending_budget_lr_sweep.sh
# Submit the complete sweep:
#   DRY_RUN=0 bash scripts/training/submit_scion_extending_budget_lr_sweep.sh
# Submit selected budgets, e.g. 2x and 4x:
#   DRY_RUN=0 BUDGET_MULTIPLIERS="2 4" bash scripts/training/submit_scion_extending_budget_lr_sweep.sh
# Submit only the predicted LR and unchanged baseline LR:
#   DRY_RUN=0 LR_FACTORS="1.0" bash scripts/training/submit_scion_extending_budget_lr_sweep.sh

BUDGET_MULTIPLIERS=(${BUDGET_MULTIPLIERS:-2 4 6 8 10})
LR_FACTORS=(${LR_FACTORS:-0.70 0.85 1.00 1.15 1.30})
INCLUDE_BASELINE_LR="${INCLUDE_BASELINE_LR:-1}"
DRY_RUN="${DRY_RUN:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"

BASE_TOKEN_BUDGET=2000000000
SEQUENCE_LENGTH=2048
LOCAL_BATCH=8
DEVICES_PER_NODE=4
NTASKS_PER_NODE=4
GPUS_PER_NODE=4
CPUS_PER_TASK="${CPUS_PER_TASK:-72}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
SEED="${SEED:-4}"
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

# Set-C baseline found at B0=256 and T0=2B (20 TPP).
BASE_LR=0.0225
SCION_MOMENTUM=0.0658
SCION_AUX_LR_FACTOR=0.02
SCION_SCALE_EMBED=3500
SCION_SCALE_BIAS=90
SCION_SCALE_LAYER_NORM=6.8
SCION_SCALE_MATRIX=336

# Rounded no-accumulation batches and sigma-star-corrected predicted LRs.
# beta/beta0 = [sqrt(T0/T) * rho(B)/rho(B0)
#               * sigma_star(B)/sigma_star(B0)]^(2/3)
# rho(B) propto B^-0.00291078; sigma_star(B) propto B^0.2005635.
declare -A GBS_BY_MULT=(
  [2]=448 [4]=736 [6]=1024 [8]=1248 [10]=1504
)
declare -A THEORY_LR_BY_MULT=(
  [2]=0.01922489340521193
  [4]=0.016290329418226557
  [6]=0.014863849451137948
  [8]=0.013861361578881005
  [10]=0.013188037793255399
)

N_BLOCKS=12
HIDDEN_SIZE=768
INTERMEDIATE_SIZE=3072
N_HEADS=12

WANDB_ENTITY="${WANDB_ENTITY:-SwissAI_DLM}"
WANDB_PROJECT="${WANDB_PROJECT:-SwissAI_Extending_Budget}"
WANDB_GROUP="${WANDB_GROUP:-scion_budget_lr_fig5_global_t_L12H768_S2048_seed${SEED}}"
WANDB_ID_PREFIX="${WANDB_ID_PREFIX:-budget-lr-v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_ROOT="${DATA_ROOT:-${SCRATCH}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-scion_budget_lr_fig5_global_t_L12H768_S2048}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/extending_budget/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${EXPERIMENT_NAME}}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb_logs}"

TRAIN_CACHE_PATH="${DATA_CACHE_DIR}/nemotron-cc-pretok-train_train_bs${SEQUENCE_LENGTH}_unwrapped_eosFalse_specialFalse.dat"
VALID_CACHE_PATH="${DATA_CACHE_DIR}/nemotron-cc-pretok-valid_validation_bs${SEQUENCE_LENGTH}_unwrapped_eosFalse_specialFalse.dat"
if [[ ! -d "${PRETOK_LOCAL_DIR}" || ! -d "${TRAIN_CACHE_PATH}" || ! -d "${VALID_CACHE_PATH}" ]]; then
  echo "The verified packed Nemotron train/validation dataset is missing." >&2
  echo "Pretokenized source: ${PRETOK_LOCAL_DIR}" >&2
  echo "Train cache: ${TRAIN_CACHE_PATH}" >&2
  echo "Validation cache: ${VALID_CACHE_PATH}" >&2
  exit 1
fi
if (( NUM_WORKERS < 1 || CHECKPOINT_EVERY < 1 )); then
  echo "NUM_WORKERS and CHECKPOINT_EVERY must be positive." >&2
  exit 1
fi

echo "Figure-5 LR sweep: Set-C L${N_BLOCKS}/H${HIDDEN_SIZE}, S=${SEQUENCE_LENGTH}, seed=${SEED}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}, group=${WANDB_GROUP}"
echo "Loss logging: every optimizer step; reported statistic: final 500-step arithmetic mean"
echo "Schedule: no warmup; linear warmdown over final 28% (baseline/paper-compatible)"
echo "Sampling: globally sharded low-discrepancy t; antithetic sampling enabled"
echo "Loader: verified packed cache, spawn + lazy path-only wrapper, ${NUM_WORKERS} workers/rank"
printf '%-5s %-7s %-6s %-8s %-12s %-13s %-12s %-12s %-8s\n' \
  "T/T0" "TPP" "GBS" "nodes" "steps" "actual_tokens" "lr_factor" "lr" "kind"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${WANDB_DIR}"
fi

submit_count=0
for MULT in "${BUDGET_MULTIPLIERS[@]}"; do
  if [[ -z "${GBS_BY_MULT[$MULT]+x}" ]]; then
    echo "Unsupported budget multiplier ${MULT}; choose from 2 4 6 8 10." >&2
    exit 1
  fi
  GBS="${GBS_BY_MULT[$MULT]}"
  THEORY_LR="${THEORY_LR_BY_MULT[$MULT]}"
  TOKEN_BUDGET=$((BASE_TOKEN_BUDGET * MULT))
  MAX_STEPS=$((TOKEN_BUDGET / (GBS * SEQUENCE_LENGTH)))
  ACTUAL_TOKENS=$((MAX_STEPS * GBS * SEQUENCE_LENGTH))
  WARM_DOWN_STEPS=$((MAX_STEPS * 28 / 100))
  TPP=$(awk -v t="${ACTUAL_TOKENS}" 'BEGIN {printf "%.3f", t/100000000}')
  NODES=$((GBS / (LOCAL_BATCH * DEVICES_PER_NODE)))
  NTASKS=$((NODES * NTASKS_PER_NODE))
  if (( NODES * LOCAL_BATCH * DEVICES_PER_NODE != GBS )); then
    echo "GBS=${GBS} cannot be realized without accumulation at local batch ${LOCAL_BATCH}." >&2
    exit 1
  fi

  RUN_FACTORS=("${LR_FACTORS[@]}")
  if [[ "${INCLUDE_BASELINE_LR}" == "1" ]]; then
    RUN_FACTORS+=("baseline")
  fi
  for FACTOR in "${RUN_FACTORS[@]}"; do
    if [[ "${FACTOR}" == "baseline" ]]; then
      LR="${BASE_LR}"
      KIND="baseline"
      FACTOR_LABEL="baseline"
    else
      LR=$(awk -v center="${THEORY_LR}" -v factor="${FACTOR}" 'BEGIN {printf "%.12g", center*factor}')
      KIND="sweep"
      FACTOR_LABEL="${FACTOR}"
      if awk -v factor="${FACTOR}" 'BEGIN {exit !(factor > 0.999999 && factor < 1.000001)}'; then
        KIND="theory"
      fi
    fi
    LR_ID="${LR//./p}"
    FACTOR_ID="${FACTOR_LABEL//./p}"
    RUN_NAME="scion_budget_m${MULT}_tpp${TPP}_gbs${GBS}_lr${LR_ID}_${KIND}_seed${SEED}"
    RUN_STORAGE_DIR="${OUTPUT_ROOT}/m${MULT}_gbs${GBS}/lr_${LR_ID}_seed_${SEED}"
    LAST_CKPT="${RUN_STORAGE_DIR}/checkpoints/last.ckpt"
    RESUME_FROM_CKPT=false
    if [[ "${AUTO_RESUME}" == "1" && -f "${LAST_CKPT}" ]]; then
      RESUME_FROM_CKPT=true
    fi
    printf '%-5s %-7s %-6s %-8s %-12s %-13s %-12s %-12s %-8s\n' \
      "${MULT}" "${TPP}" "${GBS}" "${NODES}" "${MAX_STEPS}" "${ACTUAL_TOKENS}" \
      "${FACTOR_LABEL}" "${LR}" "${KIND}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    PARTITION_DIRECTIVE=""
    if [[ -n "${PARTITION}" ]]; then
      PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
    fi
    WANDB_RUN_ID="${WANDB_ID_PREFIX}-m${MULT}-b${GBS}-f${FACTOR_ID}-s${SEED}"
    "${SBATCH_BIN}" <<SBATCH_EOF
#!/usr/bin/env bash
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

export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export DISCRETE_DIFFUSION_SCRATCH_DIR="${DATA_ROOT}/training-data"
export JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/SwissAI-DLM-\${USER:-unknown}-\${SLURM_JOB_ID:-manual}}"
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
  model.attn_soft_cap=30.0 \
  algo=gidd \
  algo.loss_type=gidd_easydel_lowmem \
  algo.hybrid_mixing_shift=-1000 \
  algo.loss_weighting=dynamic \
  algo.low_discrepancy_sampling=true \
  algo.time_sampling_scope=global_batch \
  lr_scheduler=constant_warmup \
  strategy=ddp \
  trainer.deterministic=false \
  trainer.num_nodes=${NODES} \
  trainer.devices=${DEVICES_PER_NODE} \
  trainer.accumulate_grad_batches=1 \
  trainer.max_steps=${MAX_STEPS} \
  trainer.log_every_n_steps=1 \
  trainer.gradient_clip_val=1.0 \
  trainer.num_sanity_val_steps=2 \
  trainer.val_check_interval=500 \
  trainer.limit_val_batches=0.5 \
  trainer.precision=bf16-mixed \
  loader.global_batch_size=${GBS} \
  loader.eval_global_batch_size=${GBS} \
  loader.batch_size=${LOCAL_BATCH} \
  loader.eval_batch_size=${LOCAL_BATCH} \
  +loader.exact_resume=false \
  loader.multiprocessing_context=spawn \
  loader.num_workers=${NUM_WORKERS} \
  loader.pin_memory=true \
  +loader.lazy_spawn_dataset=true \
  training.ema=0.0 \
  training.antithetic_sampling=true \
  training.loss_precision=bf16 \
  training.fault_tolerant=false \
  training.torch_compile=false \
  training.log_train_aux_metrics=false \
  training.sync_train_loss=true \
  eval.generate_samples=false \
  callbacks.pytorch_profiler.enabled=false \
  perf.enabled=true \
  perf.theoretical_peak_tflops_per_gpu=989 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY} \
  callbacks.checkpoint_every_n_steps.save_top_k=1 \
  callbacks.checkpoint_every_n_steps.save_last=true \
  callbacks.checkpoint_monitor.save_top_k=0 \
  checkpointing.save_dir="${RUN_STORAGE_DIR}" \
  checkpointing.resume_from_ckpt=${RESUME_FROM_CKPT} \
  checkpointing.resume_ckpt_path="${LAST_CKPT}" \
  optim=scion \
  optim.lr=${LR} \
  optim.weight_decay=0.0 \
  optim.momentum=${SCION_MOMENTUM} \
  optim.aux_lr_factor=${SCION_AUX_LR_FACTOR} \
  optim.scale_embed=${SCION_SCALE_EMBED} \
  optim.scale_bias=${SCION_SCALE_BIAS} \
  optim.scale_layer_norm=${SCION_SCALE_LAYER_NORM} \
  optim.scale_matrix=${SCION_SCALE_MATRIX} \
  optim.bias_norm=RowNorm \
  optim.norm_layer_norm=BiasRMS \
  optim.spectral_norm_steps=5 \
  optim.unconstrained=false \
  optim.warmup_iters=0 \
  optim.warmdown_iters=${WARM_DOWN_STEPS} \
  optim.min_lr=1e-8 \
  optim.trace_enabled=false \
  optim.trace_collect_noise_stats=false \
  hydra.run.dir="${RUN_STORAGE_DIR}/hydra/\${SLURM_JOB_ID}" \
  wandb.save_dir="${WANDB_DIR}" \
  +wandb.entity="${WANDB_ENTITY}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="${RUN_NAME}" \
  wandb.id="${WANDB_RUN_ID}" \
  wandb.notes="Figure-5 LR sweep; final train loss is arithmetic mean over last 500 optimizer steps" \
  +wandb.save_code=true \
  wandb.job_type=train \
  +wandb.tags="[extending_budget,fig5_lr_sweep,one_seed,set_c,global_time_sampling,budget_mult_${MULT},tpp_${TPP},gbs_${GBS},nodes_${NODES},lr_kind_${KIND},lr_factor_${FACTOR_LABEL},theory_lr_${THEORY_LR},actual_tokens_${ACTUAL_TOKENS},window_500]"
SBATCH_EOF
    submit_count=$((submit_count + 1))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only. Use DRY_RUN=0 to submit the ${#BUDGET_MULTIPLIERS[@]}-budget sweep."
else
  echo "Submitted ${submit_count} jobs."
fi
