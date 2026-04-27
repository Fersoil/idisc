#!/bin/bash

#SBATCH --job-name=iDisc-eval
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=32G

IDISC_REPO="$HOME/idisc"
BASE_PATH="/work/courses/3dv/team17/idisc"
CFG="configs/kitti/kitti_r101.json"
MODEL="/work/courses/3dv/team17/models/kitti_resnet101.pt"
SAM_CKPT="/work/courses/3dv/team17/sam3_checkpoints/sam3.pt"
SAM3_CACHE="/work/courses/3dv/team17/sam3_cache"
OUTPUT_DIR="$IDISC_REPO/results"

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
echo "Base Path: $BASE_PATH"
echo "Output:    $OUTPUT_DIR"
echo "Branch:    $(git -C $IDISC_REPO branch --show-current)"
echo "Commit:    $(git -C $IDISC_REPO rev-parse --short HEAD)"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$IDISC_REPO/logs"

# ── run
cd "$IDISC_REPO"
python -u eval_comparison.py \
  --config-file "$CFG" \
  --model-file "$MODEL" \
  --base-path "$BASE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --sam-checkpoint "$SAM_CKPT" \
  --sam3-cache-dir "$SAM3_CACHE" \
  --num-vis 8
