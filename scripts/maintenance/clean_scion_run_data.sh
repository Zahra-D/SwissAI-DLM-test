#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/clean_scion_run_data.sh [options]

Interactively choose Scion run output directories to remove.

Options:
  --kind sweep|single|all   Which default run roots to scan. Default: all.
  --root PATH               Add an explicit run root to scan. May be repeated.
  --pattern GLOB            Only list run directory names matching GLOB.
  --older-than DAYS         Only list run directories older than DAYS.
  --select SELECTION        Non-interactive selection, e.g. "1 3-5" or "all".
  --logs                    Also delete matching files from repo logs/.
  --dry-run                 Print what would be deleted, then exit.
  -y, --yes                 Delete without the final DELETE prompt.
  -h, --help                Show this help.

Selection syntax:
  1 3 5-7                   Select numbered rows.
  all                       Select all listed rows.
  pattern:GLOB              Select listed rows whose basename matches GLOB.
  q                         Quit without deleting.

Environment defaults:
  OUTPUT_ROOT               Sweep output root. Default: $SCRATCH/SwissAI-DLM-data/outputs
  SMOKE_ROOT                Single-run smoke root. Default: $SCRATCH/SwissAI-DLM-data/smoke
  SINGLE_OUTPUT_ROOT        Single-run output root. Default: $SMOKE_ROOT/outputs
  LOG_DIR                   Slurm log dir. Default: <repo>/logs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${SCRATCH:-}" ]]; then
  DEFAULT_SWEEP_ROOT="${SCRATCH}/SwissAI-DLM-data"
  DEFAULT_SMOKE_ROOT="${SCRATCH}/SwissAI-DLM-data/smoke"
else
  DEFAULT_SWEEP_ROOT="${REPO_ROOT}"
  DEFAULT_SMOKE_ROOT="${REPO_ROOT}/smoke"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_SWEEP_ROOT}/outputs}"
SMOKE_ROOT="${SMOKE_ROOT:-${DEFAULT_SMOKE_ROOT}}"
SINGLE_OUTPUT_ROOT="${SINGLE_OUTPUT_ROOT:-${SMOKE_ROOT}/outputs}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

KIND="all"
PATTERN="*"
OLDER_THAN=""
SELECTION=""
INCLUDE_LOGS=0
DRY_RUN=0
ASSUME_YES=0
CUSTOM_ROOTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      KIND="${2:-}"
      shift 2
      ;;
    --root)
      CUSTOM_ROOTS+=("${2:-}")
      shift 2
      ;;
    --pattern)
      PATTERN="${2:-}"
      shift 2
      ;;
    --older-than)
      OLDER_THAN="${2:-}"
      shift 2
      ;;
    --select)
      SELECTION="${2:-}"
      shift 2
      ;;
    --logs)
      INCLUDE_LOGS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -y|--yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${KIND}" in
  sweep|single|all) ;;
  *)
    echo "--kind must be one of: sweep, single, all" >&2
    exit 1
    ;;
esac

if [[ -n "${OLDER_THAN}" && ! "${OLDER_THAN}" =~ ^[0-9]+$ ]]; then
  echo "--older-than must be an integer number of days" >&2
  exit 1
fi

ROOTS=()

add_root() {
  local root="$1"
  local abs_root

  [[ -n "${root}" && -d "${root}" ]] || return 0
  abs_root="$(cd "${root}" && pwd -P)"
  for existing in "${ROOTS[@]:-}"; do
    [[ "${existing}" == "${abs_root}" ]] && return 0
  done
  ROOTS+=("${abs_root}")
}

if [[ "${KIND}" == "sweep" || "${KIND}" == "all" ]]; then
  add_root "${OUTPUT_ROOT}/hparam_sweep_scion"
fi

if [[ "${KIND}" == "single" || "${KIND}" == "all" ]]; then
  add_root "${OUTPUT_ROOT}/scion_single"
  add_root "${SINGLE_OUTPUT_ROOT}/scion_single"
fi

for root in "${CUSTOM_ROOTS[@]}"; do
  add_root "${root}"
done

if (( ${#ROOTS[@]} == 0 )); then
  echo "No run roots found."
  echo "Checked OUTPUT_ROOT=${OUTPUT_ROOT} and SINGLE_OUTPUT_ROOT=${SINGLE_OUTPUT_ROOT}."
  exit 0
fi

RUN_DIRS=()

add_run_dir() {
  local dir="$1"
  local base

  base="$(basename "${dir}")"
  [[ "${base}" == ${PATTERN} ]] || return 0

  if [[ -n "${OLDER_THAN}" ]]; then
    find "${dir}" -maxdepth 0 -mtime +"${OLDER_THAN}" -print -quit | grep -q . || return 0
  fi

  RUN_DIRS+=("${dir}")
}

for root in "${ROOTS[@]}"; do
  while IFS= read -r line; do
    add_run_dir "${line#* }"
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr)
done

if (( ${#RUN_DIRS[@]} == 0 )); then
  echo "No run directories matched."
  echo "Pattern: ${PATTERN}"
  [[ -n "${OLDER_THAN}" ]] && echo "Older than: ${OLDER_THAN} days"
  exit 0
fi

printf 'Run roots:\n'
for root in "${ROOTS[@]}"; do
  printf '  %s\n' "${root}"
done
printf '\n'

printf '%4s  %-8s  %-16s  %s\n' "#" "size" "modified" "run"
for i in "${!RUN_DIRS[@]}"; do
  dir="${RUN_DIRS[$i]}"
  size="$(du -sh "${dir}" 2>/dev/null | awk '{print $1}')"
  modified="$(date -r "${dir}" '+%Y-%m-%d %H:%M' 2>/dev/null || printf 'unknown')"
  printf '%4d  %-8s  %-16s  %s\n' "$((i + 1))" "${size:-?}" "${modified}" "$(basename "${dir}")"
done
printf '\n'

if [[ -z "${SELECTION}" ]]; then
  read -r -p 'Select runs to clean (numbers/ranges, all, pattern:GLOB, q): ' SELECTION
fi

if [[ "${SELECTION}" == "q" || "${SELECTION}" == "quit" ]]; then
  echo "No runs selected."
  exit 0
fi

SELECTED=()
SELECTED_FLAGS=()
for _ in "${RUN_DIRS[@]}"; do
  SELECTED_FLAGS+=("0")
done

select_index() {
  local idx="$1"
  if (( idx < 1 || idx > ${#RUN_DIRS[@]} )); then
    echo "Selection index out of range: ${idx}" >&2
    exit 1
  fi
  SELECTED_FLAGS[$((idx - 1))]=1
}

for token in ${SELECTION}; do
  if [[ "${token}" == "all" ]]; then
    for i in "${!RUN_DIRS[@]}"; do
      SELECTED_FLAGS[$i]=1
    done
  elif [[ "${token}" =~ ^pattern:(.+)$ ]]; then
    glob="${BASH_REMATCH[1]}"
    for i in "${!RUN_DIRS[@]}"; do
      base="$(basename "${RUN_DIRS[$i]}")"
      [[ "${base}" == ${glob} ]] && SELECTED_FLAGS[$i]=1
    done
  elif [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    if (( start > end )); then
      echo "Invalid descending range: ${token}" >&2
      exit 1
    fi
    for ((idx = start; idx <= end; idx++)); do
      select_index "${idx}"
    done
  elif [[ "${token}" =~ ^[0-9]+$ ]]; then
    select_index "${token}"
  else
    echo "Invalid selection token: ${token}" >&2
    exit 1
  fi
done

for i in "${!RUN_DIRS[@]}"; do
  [[ "${SELECTED_FLAGS[$i]}" == "1" ]] && SELECTED+=("${RUN_DIRS[$i]}")
done

if (( ${#SELECTED[@]} == 0 )); then
  echo "Selection did not match any runs."
  exit 0
fi

LOG_FILES=()
if (( INCLUDE_LOGS )) && [[ -d "${LOG_DIR}" ]]; then
  shopt -s nullglob
  for dir in "${SELECTED[@]}"; do
    run="$(basename "${dir}")"
    for file in "${LOG_DIR}/${run}"*.out "${LOG_DIR}/${run}"*.err; do
      [[ -f "${file}" ]] && LOG_FILES+=("${file}")
    done
  done
  shopt -u nullglob
fi

printf '\nSelected run directories:\n'
for dir in "${SELECTED[@]}"; do
  printf '  %s\n' "${dir}"
done

if (( INCLUDE_LOGS )); then
  printf '\nMatching log files:\n'
  if (( ${#LOG_FILES[@]} == 0 )); then
    printf '  none\n'
  else
    for file in "${LOG_FILES[@]}"; do
      printf '  %s\n' "${file}"
    done
  fi
fi

printf '\n'
if (( DRY_RUN )); then
  echo "Dry run only. Nothing deleted."
  exit 0
fi

echo "This removes output/checkpoint/sample data for the selected runs."
echo "It does not cancel running Slurm jobs and it does not delete hosted W&B runs."

if (( ! ASSUME_YES )); then
  read -r -p 'Type DELETE to continue: ' CONFIRM
  if [[ "${CONFIRM}" != "DELETE" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

rm -rf -- "${SELECTED[@]}"

if (( INCLUDE_LOGS && ${#LOG_FILES[@]} > 0 )); then
  rm -f -- "${LOG_FILES[@]}"
fi

echo "Cleaned ${#SELECTED[@]} run director$( (( ${#SELECTED[@]} == 1 )) && printf 'y' || printf 'ies' )."
if (( INCLUDE_LOGS )); then
  echo "Cleaned ${#LOG_FILES[@]} log file$( (( ${#LOG_FILES[@]} == 1 )) && printf '' || printf 's' )."
fi
