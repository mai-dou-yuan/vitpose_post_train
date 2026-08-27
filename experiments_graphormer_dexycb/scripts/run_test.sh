#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

python experiments_graphormer_dexycb/test.py \
  --config experiments_graphormer_dexycb/configs/dexycb_graphormer.yaml \
  "$@"
