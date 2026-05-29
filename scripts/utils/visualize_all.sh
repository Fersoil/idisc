#!/bin/bash
#SBATCH --job-name=iDisc-viz-all
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --constraint=5060ti
#SBATCH --output=logs/iDisc-viz-all_%j.out
#SBATCH --error=logs/iDisc-viz-all_%j.err
#
set -euo pipefail

REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"

. /etc/profile.d/modules.sh
module add cuda/12.8
if [[ -f "$REPO/.venv/bin/activate" ]]; then
  source "$REPO/.venv/bin/activate"
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BP=/work/courses/3dv/team17/idisc
MANIFEST="$REPO/splits/kitti/sequence_manifest.json"
VIZ="$REPO/output/runs/viz"

# resolved configs (one per model)
CFG_BASE_IMG="$REPO/output/runs/2026-05-28_18-22-11_finetune-idisc-image_7d21415/resolved_config.yaml"
CFG_BASE_VID="$REPO/output/runs/2026-05-28_16-03-16_finetune-idisc-video_7d21415/resolved_config.yaml"
CFG_SAM_IMG="$REPO/output/runs/2026-05-28_08-37-35_finetune-sam3-image_c608a2d/resolved_config.yaml"
CFG_SAM_VID="$REPO/output/runs/2026-05-28_10-37-58_finetune-sam3-video_c608a2d/resolved_config.yaml"

# checkpoints
CK_BASE_IMG="$REPO/output/models/finetune-idisc-image/best_sam_finetuned.pt"
CK_BASE_VID="$REPO/output/models/finetune-idisc-video/best_sam_finetuned.pt"
CK_SAM_IMG="$REPO/output/models/finetune-sam3-image/best_sam_finetuned.pt"
CK_SAM_VID="$REPO/output/models/finetune-sam3-video/best_sam_finetuned.pt"

COMMON=(--base-path "$BP" --manifest-path "$MANIFEST"
        --start-clip 50 --num-clips 3 --clip-length 4 --fps 1 --sam-mode replace)

echo "### viz_seq — image-encoder IDR maps ###"
python -u scripts/utils/visualize_sequence.py --config-file "$CFG_BASE_IMG" --model-file "$CK_BASE_IMG" --output-dir "$VIZ/seq_baseline_image" "${COMMON[@]}"
python -u scripts/utils/visualize_sequence.py --config-file "$CFG_SAM_IMG"  --model-file "$CK_SAM_IMG"  --output-dir "$VIZ/seq_sam3_image"     "${COMMON[@]}"

echo "### viz_seq_video — video-encoder IDR maps ###"
python -u scripts/utils/visualize_sequence.py --config-file "$CFG_BASE_VID" --model-file "$CK_BASE_VID" --output-dir "$VIZ/seq_baseline_video" "${COMMON[@]}"
python -u scripts/utils/visualize_sequence.py --config-file "$CFG_SAM_VID"  --model-file "$CK_SAM_VID"  --output-dir "$VIZ/seq_sam3_video"     "${COMMON[@]}"

echo "### viz_sam3 — SAM3 image-encoder masks ###"
python -u scripts/utils/visualize_sam3.py --config-file "$CFG_SAM_IMG" --model-file "$CK_SAM_IMG" --output-dir "$VIZ/sam3_masks" "${COMMON[@]}"

echo "### viz_masklets — SAM3 video tracker masklets ###"
# --det-thresh 0.0
python -u scripts/utils/visualize_sam3_masklets.py --config-file "$CFG_SAM_VID" --model-file "$CK_SAM_VID" --output-dir "$VIZ/masklets" --num-masklets 9 --det-thresh 0.0 "${COMMON[@]}"

echo "### attn — per-IDR attention grids (single frame, PNGs) ###"
# AFP grids appear for the baseline only 
ATTN=(--base-path "$BP" --manifest-path "$MANIFEST" --start-clip 50 --num-clips 1 --clip-length 4 --sam-mode replace)
python -u scripts/utils/visualize_experiments.py --config-file "$CFG_BASE_IMG" --model-file "$CK_BASE_IMG" --output-dir "$VIZ/attn_baseline_image" "${ATTN[@]}"
python -u scripts/utils/visualize_experiments.py --config-file "$CFG_SAM_IMG"  --model-file "$CK_SAM_IMG"  --output-dir "$VIZ/attn_sam3_image"     "${ATTN[@]}"

echo "### DONE — outputs under $VIZ ###"
