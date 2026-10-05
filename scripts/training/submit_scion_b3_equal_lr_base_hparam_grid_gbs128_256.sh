#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias retained for commands shared before the sweep expanded.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/submit_scion_b3_equal_lr_base_hparam_grid_gbs64_128_256_512.sh" "$@"
