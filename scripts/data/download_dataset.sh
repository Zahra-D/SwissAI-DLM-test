#!/usr/bin/env bash
#SBATCH --job-name=gidd_download
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=gidd_download_%j.log
#SBATCH --account=ab035
#SBATCH --environment=uni-d2
#SBATCH --partition=debug

set -euo pipefail

# --- 2. SET ENVIRONMENT VARIABLES ---
# Point the Hugging Face cache to scratch unless the caller selected a path.
export HF_HOME="${HF_HOME:-${SCRATCH}/SwissAI-DLM-data/cache/hf}"
mkdir -p "${HF_HOME}"

# Paste your Hugging Face read token between the quotes if you want to keep it
# in this local script.  Do not commit this file after adding the token.
HF_TOKEN_OVERRIDE=""

# Token priority: literal override, environment variable, then token file.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_HOME}/token}"
if [[ -n "${HF_TOKEN_OVERRIDE}" ]]; then
  export HF_TOKEN="${HF_TOKEN_OVERRIDE}"
elif [[ -z "${HF_TOKEN:-}" && -r "${HF_TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(<"${HF_TOKEN_FILE}")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Missing Hugging Face token. Store it in ${HF_TOKEN_FILE} or export HF_TOKEN before sbatch." >&2
  exit 1
fi

DATA_DIR="${NEMOTRON_PRETOK_DIR:-${SCRATCH}/SwissAI-DLM-data/training-data/gidd-nemotron-cc-pretok}"
mkdir -p "${DATA_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"

# --- 3. RUN THE CODE ---
echo "Starting download of dvruette/gidd-nemotron-cc-pretok..."

"${VENV_PYTHON}" -m huggingface_hub.commands.huggingface_cli download \
    dvruette/gidd-nemotron-cc-pretok \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --local-dir-use-symlinks False

PARQUET_COUNT=$(find "${DATA_DIR}" -type f -name '*.parquet' | wc -l)
if (( PARQUET_COUNT < 4608 )); then
  echo "Download is incomplete: found ${PARQUET_COUNT} parquet files; expected 4608." >&2
  exit 1
fi

echo "Verified ${PARQUET_COUNT} parquet files in ${DATA_DIR}."
echo "Download completed successfully!"
