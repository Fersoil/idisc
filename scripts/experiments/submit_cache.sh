#!/bin/bash

#SBATCH --job-name=iDisc-cache
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=48G

IDISC_REPO="$HOME/idisc"
KITTI_ROOT="/work/courses/3dv/team17/idisc/datasets/kitti"
SAM_CKPT="/work/courses/3dv/team17/sam3_checkpoints/sam3.pt"
CACHE_DIR="/work/courses/3dv/team17/sam3_cache"
MANIFEST="$IDISC_REPO/splits/kitti/sequence_manifest.json"

# ── environment
. /etc/profile.d/modules.sh
module add cuda/12.8

source "/work/courses/3dv/team17/idisc/.venv/bin/activate"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:$PYTHONPATH"

# ── info
echo "Job:       $SLURM_JOB_ID"
echo "User:      $USER"
echo "Node:      $SLURM_NODELIST"
echo "Manifest:  $MANIFEST"
echo "KITTI:     $KITTI_ROOT"
echo "Cache:     $CACHE_DIR"
echo "SAM ckpt:  $SAM_CKPT"
echo ""

mkdir -p "$CACHE_DIR"
mkdir -p "$IDISC_REPO/logs"

# ── run
cd "$IDISC_REPO"
python -u scripts/cache_sam3_video.py \
  --manifest "$MANIFEST" \
  --kitti-root "$KITTI_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --checkpoint "$SAM_CKPT" \
  --top-k 32
