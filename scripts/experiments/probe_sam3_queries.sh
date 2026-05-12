#!/bin/bash
#SBATCH --job-name=iDisc-probe-sam3
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/probe_sam3_%j.out
#SBATCH --error=logs/probe_sam3_%j.err

set -euo pipefail

IDISC_REPO="$HOME/idisc"

. /etc/profile.d/modules.sh
module add cuda/12.8
if [[ -f "$IDISC_REPO/.venv/bin/activate" ]]; then
  source "$IDISC_REPO/.venv/bin/activate"
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:${PYTHONPATH:-}"

cd "$IDISC_REPO"

python -u scripts/experiments/probe_sam3_queries.py \
  --config-file configs/kitti/kitti_sam3.json \
  --n-images 200
