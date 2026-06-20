#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

CONFIG="${1:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_prefix]}"
CKPT="${2:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_prefix]}"
PREFIX="${3:-eval-idr}"

for MODE in off zero swap; do
  echo "## IDR_ABLATE=$MODE ##"
  IDR_ABLATE="$MODE" python -u scripts/experiments/eval_depth.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output-dir "output/runs/${PREFIX}-${MODE}"
done
