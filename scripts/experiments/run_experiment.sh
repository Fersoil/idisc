#!/bin/bash
# Legacy transition entrypoint.
# Preferred Stage 1 path for new experiments:
#   python scripts/run_with_hydra.py experiment=baseline tracking=none
# Keep this script for backward-compatible job launches during migration.
#
# Usage:
#   sbatch scripts/experiments/run_experiment.sh <EXPERIMENT_ID>
#   sbatch scripts/experiments/run_experiment.sh --all
#
# Results: eval_results/<ID>/metrics.json
# Logs:    logs/iDisc-exp_<JOB>.out
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

EVAL_SAM="scripts/experiments/eval_sam.py"
EVAL_DEPTH="scripts/experiments/eval_depth.py"
FTUNE="scripts/experiments/finetune_sam.py"

# Ordered list for --all
ALL_EXPERIMENTS=(
  D1-no-prompt
  D2-singleclass
  D3-multiclass
  D4-classonly
  E1-baseline
  E2-branch-empty
  E3-branch-multiclass
  E4-branch-singleclass
  E5-replace-multiclass
  E6-replace-singleclass
  E7-concat-multiclass
  E8-concat-singleclass
  E9-concat-classonly
  E10-concat-video
)

usage() {
  cat <<EOF
Usage:
  sbatch scripts/experiments/run_experiment.sh <EXPERIMENT_ID>
  sbatch scripts/experiments/run_experiment.sh --all

Valid IDs:
  Detection:  D1-no-prompt  D2-singleclass  D3-multiclass  D4-classonly
  Eval:       E1-baseline
    Branch:   E2-branch-empty  E3-branch-multiclass  E4-branch-singleclass
    Replace:  E5-replace-multiclass  E6-replace-singleclass
    Concat:   E7-concat-multiclass  E8-concat-singleclass  E9-concat-classonly  E10-concat-video
  Finetune:   F1-replace-multiclass  F2-replace-singleclass  F3-concat-singleclass  F4-concat-video
  Eval FT:    E1-ft-baseline  E10-ft-video
  Cache:      C1-cache-video
EOF
}

# ── environment ──
. /etc/profile.d/modules.sh
module add cuda/12.8
source "/work/courses/3dv/team17/idisc/.venv/bin/activate"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:${PYTHONPATH:-}"

mkdir -p "$IDISC_REPO/logs"
cd "$IDISC_REPO"

run_one() {
  local EXP_ID="$1"
  local OUTPUT_DIR="$IDISC_REPO/eval_results/$EXP_ID"

  echo "═══════════════════════════════════════════"
  echo " Experiment: $EXP_ID"
  echo " Job:        ${SLURM_JOB_ID:-N/A}"
  echo " Node:       ${SLURM_NODELIST:-N/A}"
  echo " Branch:     $(git -C "$IDISC_REPO" branch --show-current)"
  echo " Commit:     $(git -C "$IDISC_REPO" rev-parse --short HEAD)"
  echo " Started:    $(date)"
  echo "═══════════════════════════════════════════"

  mkdir -p "$OUTPUT_DIR" "$IDISC_REPO/logs"

  case "$EXP_ID" in
    D1-no-prompt)
      python -u "$EVAL_SAM" --prompt-mode none \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    D2-singleclass)
      python -u "$EVAL_SAM" --prompt-mode singleclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    D3-multiclass)
      python -u "$EVAL_SAM" --prompt-mode multiclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    D4-classonly)
      python -u "$EVAL_SAM" --prompt-mode classonly \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E1-baseline)
      python -u "$EVAL_DEPTH" --variant baseline \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR"
      ;;

    E2-branch-empty)
      python -u "$EVAL_DEPTH" --variant branch --prompt-mode empty \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E3-branch-multiclass)
      python -u "$EVAL_DEPTH" --variant branch --prompt-mode multiclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E4-branch-singleclass)
      python -u "$EVAL_DEPTH" --variant branch --prompt-mode singleclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E5-replace-multiclass)
      python -u "$EVAL_DEPTH" --variant sam-replace --prompt-mode multiclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E6-replace-singleclass)
      python -u "$EVAL_DEPTH" --variant sam-replace --prompt-mode singleclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E7-concat-multiclass)
      python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode multiclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E8-concat-singleclass)
      python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode singleclass \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E9-concat-classonly)
      python -u "$EVAL_DEPTH" --variant sam-concat --prompt-mode classonly \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam-checkpoint "$SAM_CKPT"
      ;;

    E10-concat-video)
      python -u "$EVAL_DEPTH" --variant sam-cached-video \
        --config-file "$CFG" --model-file "$PRETRAINED" --base-path "$BASE_PATH" \
        --output-dir "$OUTPUT_DIR" --sam3-cache-dir "$SAM3_CACHE"
      ;;
      



    *)
      echo "ERROR: Unknown experiment ID '$EXP_ID'"
      usage
      exit 1
      ;;
  esac

  echo ""
  echo "═══════════════════════════════════════════"
  echo " Finished experiment: $EXP_ID"
  echo " Finished at:         $(date)"
  echo "═══════════════════════════════════════════"
  echo ""
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--all" ]]; then
  for exp in "${ALL_EXPERIMENTS[@]}"; do
    run_one "$exp"
  done
else
  run_one "$1"
fi
