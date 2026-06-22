#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

CONFIG="${1:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
CKPT="${2:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_dir]}"
OUT="${3:-output/runs/region-r2}"
LIMIT="${R2_LIMIT:-0}"

python -u scripts/experiments/eval_region_r2.py \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --output-dir "$OUT" --limit "$LIMIT"
