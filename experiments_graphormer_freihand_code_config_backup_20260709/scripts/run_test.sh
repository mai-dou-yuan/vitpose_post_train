#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"

python -m experiments_graphormer_freihand.test \
  --config experiments_graphormer_freihand/configs/freihand_graphormer.yaml \
  "$@"
