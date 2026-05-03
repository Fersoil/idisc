#!/bin/bash
#SBATCH --job-name=iDisc-sam3-pure-smoke
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_sam3_pure_%j.out
#SBATCH --error=logs/smoke_sam3_pure_%j.err

set -euo pipefail

IDISC_REPO="$HOME/idisc"

# ── environment ──
. /etc/profile.d/modules.sh
module add cuda/12.8
# Prefer per-user venv; fall back to shared team venv.
if [[ -f "$IDISC_REPO/.venv/bin/activate" ]]; then
  source "$IDISC_REPO/.venv/bin/activate"
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PYTHONPATH="$IDISC_REPO:${PYTHONPATH:-}"

cd "$IDISC_REPO"

python3 scripts/experiments/smoke_test_sam3_pure.py \
  --config-file configs/kitti/kitti_sam3.json \
  --img-h 352 \
  --img-w 1216 \
  --batch-size 1
