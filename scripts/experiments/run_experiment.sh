#!/bin/bash
#
# Usage: sbatch scripts/experiments/run_experiment.sh <EXPERIMENT_ID>
#
# Every experiment in EXPERIMENTS.md maps to an ID here.
# Results: eval_results/<ID>/metrics.json
# Logs:    logs/iDisc-exp_<JOB>.out
#
# ┌─────────────────────────────┬──────────────────────────────────────────────┐
# │ SECTION 1: SAM3 DETECTION   │ SAM3-only on KITTI, no depth                 │
# │  D1-no-prompt               │ No text prompt                               │
# │  D2-singleclass             │ Per-class x8                                 │
# │  D3-multiclass              │ One multi-class prompt                       │
# │  D4-classonly               │ Class-logit only (no presence)               │
# ├─────────────────────────────┼──────────────────────────────────────────────┤
# │ SECTION 2: SAM3 + iDisc     │ Depth eval, no training                     │
# │  E1-baseline                │ iDisc AFP only                               │
# │  E2-branch-empty            │ s-seq original: avg_pool2d, empty prompt     │
# │  E3-branch-multiclass       │ s-seq + multi-class prompt                   │
# │  E4-branch-singleclass      │ s-seq + single-class prompt                  │
# │  E5-replace-multiclass      │ Linear proj replaces AFP, multi-class        │
# │  E6-replace-singleclass     │ Linear proj replaces AFP, single-class       │
# │  E7-concat-multiclass       │ Linear proj concat with AFP, multi-class     │
# │  E8-concat-singleclass      │ Linear proj concat with AFP, single-class    │
# │  E9-concat-classonly         │ Linear proj concat with AFP, class-logit     │
# │  E10-concat-video           │ Concat with AFP, cached video queries        │
# ├─────────────────────────────┼──────────────────────────────────────────────┤
# │ FINE-TUNE                   │                                              │
# │  F1-replace-multiclass      │ Replace AFP, multi-class, online SAM         │
# │  F2-replace-singleclass     │ Replace AFP, single-class, online SAM        │
# │  F3-concat-singleclass      │ Concat, single-class, online SAM             │
# │  F4-concat-video            │ Concat, cached video queries                 │
# ├─────────────────────────────┼──────────────────────────────────────────────┤
# │ EVAL FINE-TUNED             │                                              │
# │  E1-ft-baseline             │ F4 checkpoint, AFP only (regression check)   │
# │  E10-ft-video               │ F4 checkpoint + cached video queries         │
# ├─────────────────────────────┼──────────────────────────────────────────────┤
# │ CACHE                       │                                              │
# │  C1-cache-video             │ Pre-compute SAM3 video queries               │
# └─────────────────────────────┴──────────────────────────────────────────────┘
#
#SBATCH --job-name=iDisc-exp
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mem=48G

set -euo pipefail

# ── paths ──
IDISC_REPO="$HOME/idisc"
BASE_PATH="/work/courses/3dv/team17/idisc"
CFG="configs/kitti/kitti_r101.json"
PRETRAINED="/work/courses/3dv/team17/models/kitti_resnet101.pt"
SAM_CKPT="/work/courses/3dv/team17/sam3_checkpoints/sam3.pt"
SAM3_CACHE="/work/courses/3dv/team17/sam3_cache"
MANIFEST="$IDISC_REPO/splits/kitti/sequence_manifest.json"
KITTI_ROOT="/work/courses/3dv/team17/idisc/datasets/kitti"
FINETUNE_CKPT="$IDISC_REPO/finetune_output/kitti-best.pt"

EXP_ID="${1:?Usage: sbatch scripts/experiments/run_experiment.sh <EXPERIMENT_ID>}"
OUTPUT_DIR="$IDISC_REPO/eval_results/$EXP_ID"
EVAL_SAM="scripts/experiments/eval_sam.py"
EVAL_DEPTH="scripts/experiments/eval_depth.py"
FTUNE="scripts/experiments/finetune_sam.py"

# ── environment ──
. /etc/profile.d/modules.sh
module add cuda/12.8
source "/work/courses/3dv/team17/idisc/.venv/bin/activate"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:${PYTHONPATH:-}"

# ── logging ──
echo "═══════════════════════════════════════════"
echo " Experiment: $EXP_ID"
echo " Job:        $SLURM_JOB_ID"
echo " Node:       $SLURM_NODELIST"
echo " Branch:     $(git -C $IDISC_REPO branch --show-current)"
echo " Commit:     $(git -C $IDISC_REPO rev-parse --short HEAD)"
echo " Started:    $(date)"
echo "═══════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR" "$IDISC_REPO/logs"
cd "$IDISC_REPO"

# ── dispatch ──
case "$EXP_ID" in

  # ════════════════════════════════════════════
  # Section 1: SAM3 detection only (no depth)
  # ════════════════════════════════════════════

  D1-no-prompt)
    python -u "$EVAL_SAM" --prompt-mode none \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  D2-singleclass)
    python -u "$EVAL_SAM" --prompt-mode singleclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  D3-multiclass)
    python -u "$EVAL_SAM" --prompt-mode multiclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  D4-classonly)
    python -u "$EVAL_SAM" --prompt-mode classonly \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  # ════════════════════════════════════════════
  # Section 2: SAM3 + iDisc depth (no training)
  # ════════════════════════════════════════════

  E1-baseline)
    python -u "$EVAL_DEPTH" --variant baseline \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results
    ;;

  # -- Legacy s-seq (avg_pool2d, reproducing old branch bugs) --

  E2-branch-empty)
    python -u "$EVAL_DEPTH" --variant branch --prompt-mode empty \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E3-branch-multiclass)
    python -u "$EVAL_DEPTH" --variant branch --prompt-mode multiclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E4-branch-singleclass)
    python -u "$EVAL_DEPTH" --variant branch --prompt-mode singleclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  # -- Replace AFP with linear projection --

  E5-replace-multiclass)
    python -u "$EVAL_DEPTH" --variant sam-replace --prompt-mode multiclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E6-replace-singleclass)
    python -u "$EVAL_DEPTH" --variant sam-replace --prompt-mode singleclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  # -- Concat with AFP (linear proj + AFP preserved) --

  E7-concat-multiclass)
    python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode multiclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E8-concat-singleclass)
    python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode singleclass \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E9-concat-classonly)
    python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode classonly \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam-checkpoint "$SAM_CKPT"
    ;;

  E10-concat-video)
    python -u "$EVAL_DEPTH" --variant sam-cached-video \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam3-cache-dir "$SAM3_CACHE"
    ;;

  # ════════════════════════════════════════════
  # Fine-tuning
  # ════════════════════════════════════════════

  F1-replace-multiclass)
    python -u $FTUNE \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir "$IDISC_REPO/finetune_output/$EXP_ID" \
      --mode replace --prompt-mode multiclass \
      --sam-checkpoint "$SAM_CKPT" \
      --n-iters 5000 --lr 5e-5 --val-interval 500 --batch-size 2
    ;;

  F2-replace-singleclass)
    python -u $FTUNE \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir "$IDISC_REPO/finetune_output/$EXP_ID" \
      --mode replace --prompt-mode singleclass \
      --sam-checkpoint "$SAM_CKPT" \
      --n-iters 5000 --lr 5e-5 --val-interval 500 --batch-size 2
    ;;

  F3-concat-singleclass)
    python -u $FTUNE \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir "$IDISC_REPO/finetune_output/$EXP_ID" \
      --mode concat --prompt-mode singleclass \
      --sam-checkpoint "$SAM_CKPT" \
      --n-iters 5000 --lr 5e-5 --val-interval 500 --batch-size 2
    ;;

  F4-concat-video)
    python -u $FTUNE \
      --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
      --output-dir "$IDISC_REPO/finetune_output/$EXP_ID" \
      --mode concat \
      --sam3-cache-dir "$SAM3_CACHE" \
      --n-iters 5000 --lr 5e-5 --val-interval 500 --batch-size 2
    ;;

  # ════════════════════════════════════════════
  # Eval fine-tuned models
  # ════════════════════════════════════════════

  E1-ft-baseline)
    python -u "$EVAL_DEPTH" --variant baseline \
      --config-file "$CFG" --model-file "$FINETUNE_CKPT" --base-path "$BASE_PATH" \
      --output-dir eval_results
    ;;

  E10-ft-video)
    python -u "$EVAL_DEPTH" --variant sam-cached-video \
      --config-file "$CFG" --model-file "$FINETUNE_CKPT" --base-path "$BASE_PATH" \
      --output-dir eval_results --sam3-cache-dir "$SAM3_CACHE"
    ;;

  # ════════════════════════════════════════════
  # Cache
  # ════════════════════════════════════════════

  C1-cache-video)
    python -u scripts/data/cache_sam3_video.py \
      --manifest "$MANIFEST" --kitti-root "$KITTI_ROOT" \
      --cache-dir "$SAM3_CACHE" --checkpoint "$SAM_CKPT" --top-k 32
    ;;

  *)
    echo "ERROR: Unknown experiment ID '$EXP_ID'"
    echo ""
    echo "Valid IDs:"
    echo "  Detection:  D1-no-prompt  D2-singleclass  D3-multiclass  D4-classonly"
    echo "  Eval:       E1-baseline"
    echo "    Branch:   E2-branch-empty  E3-branch-multiclass  E4-branch-singleclass"
    echo "    Replace:  E5-replace-multiclass  E6-replace-singleclass"
    echo "    Concat:   E7-concat-multiclass  E8-concat-singleclass  E9-concat-classonly  E10-concat-video"
    echo "  Finetune:   F1-replace-multiclass  F2-replace-singleclass  F3-concat-singleclass  F4-concat-video"
    echo "  Eval FT:    E1-ft-baseline  E10-ft-video"
    echo "  Cache:      C1-cache-video"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo " Finished: $(date)"
echo "═══════════════════════════════════════════"
