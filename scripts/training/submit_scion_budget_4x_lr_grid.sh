#!/usr/bin/env bash
set -euo pipefail

# Full-horizon 4x token-budget grid: 8B requested tokens, GBS 672.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET_MULTIPLIER=4
export GBS=672
export LR_GRID_CSV="0.01152,0.0144,0.018,0.0225,0.028125"
exec "${SCRIPT_DIR}/submit_scion_budget_lr_grid_common.sh" "$@"
