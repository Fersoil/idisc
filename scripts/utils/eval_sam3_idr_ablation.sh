#!/bin/bash
# usage: eval_sam3_idr_ablation.sh <resolved_config.yaml> <checkpoint.pt> [out_prefix]
set -euo pipefail
REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"

CONFIG="${1:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_prefix]}"
CKPT="${2:?usage: $0 <resolved_config.yaml> <checkpoint.pt> [out_prefix]}"
PREFIX="${3:-eval-idr}"

. /etc/profile.d/modules.sh
module add cuda/12.8
[[ -f "$REPO/.venv/bin/activate" ]] && source "$REPO/.venv/bin/activate"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

for MODE in off zero swap; do
  echo "## IDR_ABLATE=$MODE ##"
  IDR_ABLATE="$MODE" python -u scripts/experiments/eval_depth.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output-dir "output/runs/${PREFIX}-${MODE}"
done
