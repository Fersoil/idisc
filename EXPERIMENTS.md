# Experiments

This document describes the experiments in this fork of iDisc, which investigates whether SAM3 instance-level segmentation queries can improve monocular depth estimation.

## Background

**iDisc** is a monocular depth estimation model. Its key idea is to represent the scene as a small set of learned query vectors — *Internal Discrete Representations* (IDRs) — and use cross-attention between those queries and the pixel feature map to predict depth. The AFP (Attention Feature Pyramid) module learns to produce IDRs from the decoder's multi-scale outputs.

**SAM3** is a segmentation model. Given text prompts (e.g. class names) and optionally bounding boxes, it produces per-instance query embeddings: 256-dimensional vectors that encode what objects are present and where.

**The hypothesis:** object-instance queries from SAM3 carry semantic and spatial information that could help iDisc reason better about depth — especially at object boundaries, occluded regions, and thin structures that are common failure cases for depth networks. This fork tests whether injecting those queries into the IDR slot improves depth estimation on KITTI.

---

## Architecture

The iDisc forward pass:

```
image
  → pixel encoder  (Swin-L / ResNet-101)
  → pixel decoder  (deformable attention FPN)  →  FPN pixel features
  → AFP            (Attention Feature Pyramid)  →  IDRs  [3 resolutions × 128-dim]
  → ISD head       (cross-attention: pixels × IDRs)  →  depth map
```

**AFP** distills the decoder's multi-scale outputs into a compact set of query vectors through attention. These IDRs represent high-level scene concepts that the ISD head attends to when predicting per-pixel depth.

**SAM3 integration:** SAM3's per-instance queries (256-dim) are projected to 128-dim via `sam3_proj` — three learned linear layers (one per IDR resolution) — and inserted into the IDR slot in one of the modes below.

### Integration modes

| Mode | IDR source | What the ISD sees |
|------|-----------|-------------------|
| **baseline** | AFP only | Scene-level learned concepts, no instance information |
| **pooled** | Raw encoder hidden states, averaged over layers and spatially avg_pool2d'd to (32, 128) | Spatially pooled encoder features; SAM3 still runs (for detection metrics) but its queries are not used for depth |
| **replace** | SAM3 projected queries only | Instance-level object representations replace AFP entirely |
| **concat** | AFP IDRs + SAM3 projected queries concatenated | Both scene-level (AFP) and instance-level (SAM3) context |

**pooled** is an ablation. It bypasses AFP and uses a simple spatial pooling of raw encoder features as IDRs. Because SAM3's projected queries are discarded on this path, all `pooled_*` variants should produce identical depth metrics regardless of prompt — any divergence would indicate a bug. It answers: "how much does AFP matter compared to raw pooled features?"

### Prompt strategies

SAM3 is queried over 7 KITTI classes: `car, truck, person, bicycle, motorcycle, bus, train`.

| Suffix | Strategy |
|--------|---------|
| `_empty` | Empty string prompt — SAM3 segments without text guidance (similar to SAM2 automatic mode) |
| `_singleclass` | Each class queried separately; top-K instances by confidence score are selected and merged |
| `_multiclass` | All classes in a single prompt: `"car . truck . person . bicycle . motorcycle . bus . train"` |
| `_classonly` | Multiclass prompt but only the class logit score is used, not the presence score — ablation for scoring strategy |
| `_video` | Instance queries cached from full KITTI video sequences (see C1); loaded from disk at eval time instead of running SAM3 online |

---

## Experiment groups

### D — SAM3 detection quality

**Task:** `eval_sam`. Runs SAM3 on KITTI and measures detection quality (AP, IoU) independently of depth. These experiments exist to answer: *does SAM3 actually find the right objects with each prompt strategy?* Run these before drawing conclusions about D/depth experiments — if SAM3 isn't detecting well, it can't help depth.

| Experiment name | ID | Prompt |
|-----------------|-----|--------|
| `detect_none` | D1 | none (no SAM) |
| `detect_singleclass` | D2 | singleclass |
| `detect_multiclass` | D3 | multiclass |
| `detect_classonly` | D4 | classonly |

```bash
python scripts/run_with_hydra.py experiment=detect_none
python scripts/run_with_hydra.py experiment=detect_singleclass
python scripts/run_with_hydra.py experiment=detect_multiclass
python scripts/run_with_hydra.py experiment=detect_classonly
```

---

### E — Depth evaluation (zero-shot)

**Task:** `eval`. Uses pretrained iDisc weights with no fine-tuning. Tests whether SAM3 queries improve depth estimation out-of-the-box, without any gradient signal.

#### E1 — Baseline

Standard iDisc with AFP, no SAM3. The reference point for all other E experiments.

```bash
python scripts/run_with_hydra.py experiment=baseline
```

#### E2–E4 — Pooled (ablation: AFP vs. raw pooling)

IDRs come from avg_pool2d on raw encoder hidden states. SAM3 runs with various prompts for detection metrics, but its queries are not used for depth. The three variants should produce the same depth numbers — the prompt doesn't matter here.

```bash
python scripts/run_with_hydra.py experiment=pooled_empty
python scripts/run_with_hydra.py experiment=pooled_multiclass
python scripts/run_with_hydra.py experiment=pooled_singleclass
```

#### E5–E6 — Replace

SAM3 projected queries fully replace AFP IDRs. The ISD receives only instance-level information. Tests whether object-instance representations alone are sufficient to guide depth.

```bash
python scripts/run_with_hydra.py experiment=replace_multiclass
python scripts/run_with_hydra.py experiment=replace_singleclass
```

#### E7–E10 — Concat

SAM3 projected queries are concatenated with AFP IDRs. The ISD receives both scene-level and instance-level context. This is the most conservative integration — it adds information without removing the AFP signal.

```bash
python scripts/run_with_hydra.py experiment=concat_multiclass
python scripts/run_with_hydra.py experiment=concat_singleclass
python scripts/run_with_hydra.py experiment=concat_classonly
python scripts/run_with_hydra.py experiment=concat_video   # requires C1 cache first
```

---

### F — Fine-tuning

**Task:** `finetune`. Unfreezes `sam3_proj` and the ISD head while keeping the pixel encoder and AFP frozen. Tests whether the model can learn to use SAM3 queries better when trained end-to-end with gradient signal through the projection layers.

| Experiment name | ID | Mode | Prompt |
|-----------------|-----|------|--------|
| `finetune_replace_multiclass` | F1 | replace | multiclass |
| `finetune_replace_singleclass` | F2 | replace | singleclass |
| `finetune_concat_singleclass` | F3 | concat | singleclass |
| `finetune_concat_video` | F4 | concat | video |

```bash
python scripts/run_with_hydra.py experiment=finetune_replace_multiclass
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass
```

Use `finetune=fast` for a reduced schedule during iteration:

```bash
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass finetune=fast
```

---

### C1 — Cache video queries

**Task:** `cache`. Runs SAM3 on the full KITTI video sequences and writes per-frame instance queries to disk as `.pt` files. Required before running any `_video` experiment (`concat_video`, `finetune_concat_video`). Caching enables video-mode temporal consistency: SAM3 tracks objects across frames, producing more stable instance representations than per-frame online inference.

```bash
python scripts/run_with_hydra.py experiment=cache_video
```

---

## Paths and tracking

Switch path presets:

```bash
python scripts/run_with_hydra.py experiment=baseline paths=local    # local dev machine
python scripts/run_with_hydra.py experiment=baseline paths=cluster  # ETH cluster
```

Enable W&B tracking (disabled by default):

```bash
python scripts/run_with_hydra.py experiment=baseline tracking=wandb
```

---

## Output structure

Each run writes to:

```
outputs/runs/<timestamp>_<exp_id>_<gitsha>/
  config.yaml      resolved Hydra config
  manifest.json    run metadata (git branch, commit, timestamp)
  metrics.json     depth or detection metrics
  *.log            stdout
```

---

## Smoke test

To verify all experiments launch and complete without errors:

```bash
sbatch scripts/experiments/smoke_test.sh
```

Or run the representative subset manually:

```bash
# 1. Detection — check SAM3 is finding objects
python scripts/run_with_hydra.py experiment=detect_none
python scripts/run_with_hydra.py experiment=detect_multiclass

# 2. Baseline — plain iDisc, no SAM3
python scripts/run_with_hydra.py experiment=baseline

# 3. Pooled ablation — SAM3 queries not used for depth
python scripts/run_with_hydra.py experiment=pooled_empty

# 4. Replace — SAM3 queries replace AFP
python scripts/run_with_hydra.py experiment=replace_singleclass

# 5. Concat — SAM3 queries + AFP
python scripts/run_with_hydra.py experiment=concat_singleclass
python scripts/run_with_hydra.py experiment=concat_classonly
python scripts/run_with_hydra.py experiment=concat_multiclass

# 6. Fine-tuning (fast schedule)
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass finetune=fast

# 7. Cache video queries (prerequisite for video experiments)
python scripts/run_with_hydra.py experiment=cache_video

# 8. Video (after cache)
python scripts/run_with_hydra.py experiment=concat_video
```

---

## Key files

| Path | Role |
|------|------|
| `scripts/run_with_hydra.py` | Main entrypoint for all experiments |
| `scripts/experiments/run_experiment.sh` | Legacy SLURM dispatcher (backward compat, maps E/D/F/C IDs) |
| `scripts/experiments/eval_depth.py` | Depth evaluation logic (E variants) |
| `scripts/experiments/eval_sam.py` | Detection evaluation logic (D variants) |
| `scripts/experiments/finetune_sam.py` | Fine-tuning loop (F variants) |
| `scripts/data/cache_sam3_video.py` | Video query caching (C1) |
| `idisc/models/idisc.py` | Model `forward()`: `instance_queries`, `raw_idrs`, `sam_mode` |
| `idisc/utils/config_bridge.py` | Maps Hydra config (`idr_source`, `sam_mode`) → runtime variant string |
| `idisc/dataloders/kitti.py` | KITTI dataloader with `sam3_cache_dir` and `sam3_top_k` support |

---

## Changes from upstream iDisc

1. **`sam3_proj`**: 3 × Linear(256→128), one per IDR resolution, projects SAM3 instance queries into IDR space
2. **`IDisc.forward()`**: accepts `instance_queries` (projection path) and `raw_idrs` (pooled path), plus `sam_mode` (`concat` / `replace`)
3. **`KITTIDataset`**: added `sam3_cache_dir` and `sam3_top_k` to load pre-cached video queries per frame
4. **`Sam3Processor`**: exposed `instance_queries` and `topk_scores` for the linear projection path
5. **ImageNet normalisation fix**: SAM3 was receiving double-normalised images; fixed by denormalising before passing to SAM3
6. **Linear projection replaces avg_pool2d**: the original s-seq integration used `avg_pool2d` on hidden states; the current concat/replace modes use a learned linear projection (`sam3_proj`) instead
