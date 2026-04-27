#!/bin/bash

#SBATCH --job-name=iDisc-finetune
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=32G

IDISC_REPO="$HOME/idisc"
BASE_PATH="/work/courses/3dv/team17/idisc"
CFG="configs/kitti/kitti_r101.json"
MODEL="/work/courses/3dv/team17/models/kitti_resnet101.pt"
SAM3_CACHE="/work/courses/3dv/team17/sam3_cache"
OUTPUT_DIR="$IDISC_REPO/finetune_output"

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
echo "Config:    $CFG"
echo "Model:     $MODEL"
echo "SAM3 cache: $SAM3_CACHE"
echo "Output:    $OUTPUT_DIR"
echo "Branch:    $(git -C $IDISC_REPO branch --show-current)"
echo "Commit:    $(git -C $IDISC_REPO rev-parse --short HEAD)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$IDISC_REPO/logs"

# ── run
cd "$IDISC_REPO"
python -u scripts/finetune_sam.py \
  --config-file "$CFG" \
  --model-file "$MODEL" \
  --base-path "$BASE_PATH" \
  --sam3-cache-dir "$SAM3_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --n-iters 5000 \
  --lr 5e-5 \
  --val-interval 500 \
  --batch-size 2
