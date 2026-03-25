#!/bin/bash

#SBATCH --job-name=iDisc
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# ── args
CFG=${1:?"Usage: sbatch submit.sh <config-file> [exp-name]"}
EXP_NAME=${2:-$(date +%Y%m%d_%H%M%S)}

# ── paths
IDISC_REPO="$HOME/idisc"
BASE_PATH="$HOME/store/idisc"
mkdir -p "$RUN_DIR"

# ── environment
. /etc/profile.d/modules.sh
module add cuda/12.8

source "$IDISC_REPO/.venv/bin/activate"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PYTHONPATH="$IDISC_REPO:$PYTHONPATH"

# ── info
echo "Job:      $SLURM_JOB_ID"
echo "User:     $USER"
echo "Config:   $CFG"
echo "Base Path: $BASE_PATH"
echo "Commit:   $(git -C $IDISC_REPO rev-parse --short HEAD)"

# ── run
cd "$IDISC_REPO"
python -u scripts/train.py \
  --config-file "$CFG" \
  --base-path "$BASE_PATH"
