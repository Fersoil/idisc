# SAM3 + iDisc Architecture

## Overview

This fork replaces iDisc's ResNet/Swin pixel encoder with a frozen SAM3 backbone (~840 M params). SAM3's internal feature pyramid (FPN) drives all depth prediction; a small trainable head (~12 M params) comprising `pixel_decoder`, `AFP/ISD`, and `sam3_proj` trains from random init on KITTI. Query embeddings extracted from SAM3's transformer decoder are routed through the head but have been found empirically to contribute negligible signal — the FPN is the only information source that matters.

---

## Key files

| File | Role |
|------|------|
| `idisc/models/sam3_encoder.py` | `Sam3PixelEncoder` — wraps `Sam3Processor` (image-only); extracts 3-level FPN + 200 decoder-slot queries per frame. Hooks `transformer.decoder` to capture hidden states. |
| `idisc/models/sam3_video_encoder.py` | `Sam3VideoPixelEncoder` — wraps `Sam3VideoInference` (video model); processes 4-frame clips with temporal tracker propagation. |
| `idisc/models/id_module.py` | `AFP` (Attention Feature Pyramid), `ISD` (Internal Slots Decoder), `Sam3QueryToIDR` (cross-attention query translator). |
| `idisc/models/idisc.py` | `IDisc` top-level model; builds encoder + pixel_decoder + AFP/ISD; `forward` dispatches by `sam_mode`. |
| `idisc/dataloaders/kitti_sequence.py` | `KITTISequenceDataset` — emits `(B, T, 3, H, W)` clips with stride-based non-overlapping sampling. |
| `scripts/experiments/finetune_sam.py` | Training loop invoked by `run_with_hydra.py`; handles freeze/unfreeze, per-param-group LR, sequence flattening, NaN guard. |
| `scripts/experiments/eval_depth.py` | Evaluation loop (abs_rel, d1, rmse, etc.); invoked by `run_with_hydra.py` for `task=eval`. |

---

## Running experiments

```bash
./scripts/launch.sh <key> [-- <hydra_overrides>]
```

| Key | Experiment | Notes |
|-----|-----------|-------|
| `baseline` | E1 iDisc-R101 pretrained | eval only, ~2 min |
| `e11` / `e20` | SAM3 pure, single-frame, replace | ~2h |
| `e12` | SAM3 translate (Sam3QueryToIDR) | ~2h |
| `e18` | SAM3 pure + 4-frame sequence | ~5h |
| `e17` | SAM3 translate + sequence | ~14h |
| `e19` | SAM3 video encoder + sequence | ~14h, requires 16 GB GPU |
| `cache` | Pre-compute SAM3 video queries | ~4h |

**Override examples:**
```bash
./scripts/launch.sh e11 -- finetune.n_iters=500       # quick shakedown
./scripts/launch.sh e18 -- finetune.lr=1e-4           # LR sweep
```

Output dirs: `outputs/runs/<timestamp>_<exp_id>_<git_sha>/` and `finetune_output/<exp_id>/`.

---

## Data flow

```
KITTI RGB frame  (B, 3, 352, 1216)
       │
       ▼
Sam3PixelEncoder.forward(image)
  ├── SAM3 backbone.forward_image()  ──► FPN [3 × (B, 256, H/s, W/s)]
  └── decoder hook ──────────────────► queries (B, 200, 256)
       │                                      │
       ▼                                      ▼
IDisc.forward()                        sam3_proj / Sam3QueryToIDR
  ├── MSDeformAttnPixelDecoder(FPN) ──► refined FPN + decoder_outputs
  ├── AFP(decoder_outputs) ──────────► IDRs  (B, 32, 128)  [replace: skipped]
  ├── sam_mode dispatch:
  │     replace ──► sam3_proj(queries) → IDRs
  │     translate ► Sam3QueryToIDR(queries) → IDRs
  │     none ────► AFP only
  └── ISD(fpn, IDRs) ─────────────────► depth map (B, 1, H, W)
```

---

## Config system

Each experiment uses two config files:

1. **Hydra YAML** (`conf/experiment/<name>.yaml`) — sets `run.exp_id`, `run.task`, `method.sam_mode`, dataset path, finetune hyperparams.
2. **Legacy JSON** (`configs/kitti/kitti_sam3*.json`) — sets model architecture (encoder name, prompt_mode, top_k_queries, clip_length, stride).

`run_with_hydra.py` merges them: legacy JSON is the base; YAML overrides take precedence.

**Adding a new experiment (3 steps):**
1. Copy the closest legacy JSON and edit (e.g., `configs/kitti/kitti_sam3_new.json`).
2. Create `conf/experiment/sam3_new.yaml` pointing at the new JSON.
3. Add a case entry in `scripts/launch.sh` with the new key.

---

## Current findings (summary)

See [SAM3_EXPERIMENTS.md](SAM3_EXPERIMENTS.md) for the full run log.

- **The FPN path drives all depth.** Query content (zeros, real tokens, 200-token multiclass) is invariant to val abs\_rel at convergence.
- **Best result: 0.0818 abs\_rel** (E11/E20, 5k iters, frozen SAM3, single-frame) vs 0.0600 for pretrained iDisc-R101 baseline.
- **Sequence training** (4-frame clips, stride=4) does not improve at comparable iter budgets; adds a training instability spike at step ~1000.
- **SAM3 video encoder** (temporal tracker memory) diverges during training due to train/val distribution mismatch (tracker memory populated during training, empty during val).
- **Next high-impact direction:** unfreeze SAM3 neck (`vision_backbone.convs`) with 0.1× backbone LR to improve the FPN features that actually drive depth.
