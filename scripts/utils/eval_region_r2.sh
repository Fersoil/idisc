#!/bin/bash
# usage: eval_region_r2.sh <resolved_config.yaml> <checkpoint.pt> [out_dir]   (R2_LIMIT=0 full)
set -euo pipefail
REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"

CONFIG="${1:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
CKPT="${2:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
OUT="${3:-output/runs/region-r2}"
LIMIT="${R2_LIMIT:-0}"

. /etc/profile.d/modules.sh
module add cuda/12.8
[[ -f "$REPO/.venv/bin/activate" ]] && source "$REPO/.venv/bin/activate"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -u scripts/experiments/eval_region_r2.py \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --output-dir "$OUT" --limit "$LIMIT"
