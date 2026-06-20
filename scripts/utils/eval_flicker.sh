#!/bin/bash
# usage: eval_flicker.sh <resolved_config.yaml> <checkpoint.pt> [out_dir]   (FLICKER_LIMIT=0 full)
set -euo pipefail
REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"

CONFIG="${1:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
CKPT="${2:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
OUT="${3:-output/runs/flicker}"
LIMIT="${FLICKER_LIMIT:-0}"

. /etc/profile.d/modules.sh
module add cuda/12.8
[[ -f "$REPO/.venv/bin/activate" ]] && source "$REPO/.venv/bin/activate"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -u scripts/experiments/eval_flicker.py \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --output-dir "$OUT" --limit "$LIMIT"
