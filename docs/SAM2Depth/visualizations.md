# SAM2Depth Visualization Guide

Visualization scripts for IDR attention, SAM3 masks, and depth predictions over KITTI sequences.
All scripts are in `scripts/experiments/`. GIFs for clips 50–51 are in `gifs/` mirroring the
`outputs/runs/` directory structure.

---

## Cluster paths

```
BASE_PATH   = /work/courses/3dv/team17/idisc
MODEL_FILE  = /work/courses/3dv/team17/models/kitti_resnet101.pt
SAM3_CKPT   = /work/courses/3dv/team17/sam3_checkpoints/sam3.pt
E20_CKPT    = finetune_output/E20-sam3-pure-multiclass/best_sam_finetuned.pt
```

---

## 1. Sequence IDR visualization (`viz_seq`)

Shows per-frame depth prediction, GT depth, and ISD dominant-IDR maps across 3 FPN resolutions.
Supports `--sam-mode none|replace|concat|translate` and `--soft-assignment`.

```bash
# Baseline AFP (none)
.venv/bin/python scripts/experiments/visualize_sequence.py \
  --config-file configs/kitti/kitti_sam3_translate.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_seq \
  --sam-mode none --start-clip 50 --num-clips 2 --clip-length 8 --fps 3

# SAM3 replace mode
.venv/bin/python scripts/experiments/visualize_sequence.py \
  --config-file configs/kitti/kitti_sam3_translate.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_seq \
  --sam-mode replace --start-clip 50 --num-clips 2 --clip-length 8 --fps 3

# SAM3 translate mode
.venv/bin/python scripts/experiments/visualize_sequence.py \
  --config-file configs/kitti/kitti_sam3_translate.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_seq \
  --sam-mode translate --start-clip 50 --num-clips 2 --clip-length 8 --fps 3
```

**Layout:** `[RGB | Depth pred | GT depth] / [ISD res1 | ISD res2 | ISD res3]`

---

## 2. SAM3 image-encoder mask visualization (`viz_sam3`)

Shows per-frame SAM3 segmentation slot assignment and top-K mask overlays.
Requires a SAM3 image encoder (`kitti_sam3_translate.json`).

```bash
.venv/bin/python scripts/experiments/visualize_sam3.py \
  --config-file configs/kitti/kitti_sam3_translate.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_sam3 \
  --sam-mode translate --start-clip 50 --num-clips 2 --clip-length 8 --fps 3
```

**Layout:** `[RGB | SAM3 slot assignment | top-K overlay] / [slot 0 | slot 1 | ... ]`

---

## 3. Video masklet visualization (`viz_masklets`)

Uses `Sam3VideoPixelEncoder` for temporally consistent object tracking. Shows masklet ID
assignment (stable across frames via fixed-ID locking), top-K mask overlay, and ISD dominant-IDR
maps using the same display as `viz_seq`.

> **Note:** Checkpoint E20 was trained with the image encoder (`kitti_sam3.json`). Depth
> quality is degraded with the video config because `sam3_proj` weights were not trained for
> temporally-propagated video queries. Use for masklet/IDR analysis only.

```bash
# replace mode (bypasses AFP; safe given the AFP/ISD resolution mismatch in video config)
.venv/bin/python scripts/experiments/visualize_sam3_masklets.py \
  --config-file configs/kitti/kitti_sam3_video.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_masklets \
  --sam-mode replace --start-clip 50 --num-clips 2 --clip-length 8 \
  --num-masklets 6 --fps 3

# translate mode
.venv/bin/python scripts/experiments/visualize_sam3_masklets.py \
  --config-file configs/kitti/kitti_sam3_video.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_masklets \
  --sam-mode translate --start-clip 50 --num-clips 2 --clip-length 8 \
  --num-masklets 6 --fps 3
```

**Layout:** `[RGB | Masklet ID assignment | top-K overlay] / [ISD res1 | ISD res2 | ISD res3] / [Masklet 0..N]`

---

## 4. Video sequence IDR visualization (`viz_seq_video`)

Same as `viz_seq` but the full clip goes through the video encoder backbone once, and per-frame
FPN features are dispatched individually to iDisc. Shows how temporally-conditioned queries
affect the IDR maps.

```bash
.venv/bin/python scripts/experiments/visualize_sequence.py \
  --config-file configs/kitti/kitti_sam3_video.json \
  --model-file  $E20_CKPT \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_seq_video \
  --sam-mode replace --start-clip 50 --num-clips 2 --clip-length 8 --fps 3
```

---

## 5. Static single-frame IDR visualization (`viz_sequence`)

Baseline static eval on the standard KITTI validation split using ResNet-101.
No SAM3, AFP only.

```bash
.venv/bin/python scripts/experiments/visualize_experiments.py \
  --config-file configs/kitti/kitti_r101.json \
  --model-file  $MODEL_FILE \
  --base-path   $BASE_PATH \
  --output-dir  outputs/runs/viz_sequence \
  --num-samples 8
```

---

## Key flags

| Flag | Scripts | Effect |
|---|---|---|
| `--sam-mode` | `viz_seq`, `viz_masklets`, `viz_seq_video` | `none` AFP only · `replace` SAM3 proj · `translate` Sam3QueryToIDR |
| `--soft-assignment` | `viz_seq` | Weighted-average IDR index (continuous) instead of hard argmax |
| `--num-masklets` | `viz_masklets` | How many top-scoring SAM3 masklets to display per frame |
| `--clip-length` | `viz_seq`, `viz_masklets`, `viz_seq_video` | Frames per clip (persisted to config JSON) |
| `--start-clip` | all sequence scripts | Dataset clip index to start from |
| `--format gif\|mp4` | all sequence scripts | Output format (`mp4` needs ffmpeg or `pip install imageio-ffmpeg`) |
| `--fps` | all sequence scripts | Animation frame rate |
