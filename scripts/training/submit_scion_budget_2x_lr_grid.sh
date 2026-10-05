#!/usr/bin/env bash
set -euo pipefail

# Full-horizon 2x token-budget grid: 4B requested tokens, GBS 416.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUDGET_MULTIPLIER=2
export GBS=416
export LR_GRID_CSV="0.01152,0.0144,0.018,0.0225,0.028125"
exec "${SCRIPT_DIR}/submit_scion_budget_lr_grid_common.sh" "$@"
