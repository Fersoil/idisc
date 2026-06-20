# SAM3+iDisc Online Inference Experiments

All experiments in this log use **live SAM3 inference only** — queries are extracted from the running SAM3 encoder at every training step. No cache. This supersedes the cached-query results in `SAM3_EXPERIMENTS.md`.

Reference: pretrained iDisc-R101 baseline, KITTI Eigen val abs_rel = **0.0600** (~120k iters).

---

## How to run

```bash
./scripts/launch.sh <key> [-- <hydra_overrides>]
```

| Key | Experiment | Output dir | When to kill |
|-----|-----------|-----------|--------------|
| `e11` | E11 pure replace, single-frame | `finetune_output/E11-online-sam3-pure/` | step 5000 (~2h) |
| `e12` | E12 translate (Sam3QueryToIDR), single-frame | `finetune_output/E12-online-sam3-translate/` | step 5000 (~2h) |
| `e13` | E13 pure replace + 4-frame sequence | `finetune_output/E13-online-sam3-sequence/` | step 5000 (~5h) |
| `e14` | E14 video encoder + 4-frame sequence | `finetune_output/E14-online-sam3-video/` |  |

---

## Run log

### E11 — pure SAM3, replace mode, single-frame

| Setting | Value |
|---------|-------|
| Mode | `replace` (Linear `sam3_proj`) |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per frame |
| Tokens | 200 (all decoder slots) |
| Queries | **live** |
| Data | Single-frame KITTI Eigen, 23,158 train / 652 val |
| Trainable params | 11,878,211 / 852,387,961 (1.4%) |
| GPU | RTX 5060 Ti |

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.1321 | 0.835 | 3.680 |
| 1000 | 0.1222 | 0.861 | 3.660 |
| 1500 | 0.1204 | 0.860 | 3.750 |
| 2000 | 0.1093 | 0.881 | 3.314 |
| 2500 | 0.0974 | 0.892 | 2.999 |
| 3000 | 0.0972 | 0.919 | 2.835 |
| 3500 | 0.0877 | 0.923 | 2.861 |
| 4000 | **0.0839** | 0.926 | 2.780 |
| 4500 | 0.0848 | 0.925 | 2.754 |
| 5000 | 0.0848 | 0.926 | 2.723 |

#### E11 extended — 15,000 iters (job 60408)

Same config, n_iters extended to 15,000.

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.1471 | 0.803 | 3.979 |
| 1000 | 0.1240 | 0.857 | 3.626 |
| 2000 | 0.1111 | 0.861 | 3.341 |
| 3000 | 0.0976 | 0.905 | 3.176 |
| 5000 | 0.0869 | 0.929 | 2.737 |
| 6000 | 0.0859 | 0.925 | 2.739 |
| 7000 | 0.0836 | 0.924 | 2.656 |
| 8000 | 0.0819 | 0.924 | 2.653 |
| 9000 | 0.0801 | 0.932 | 2.566 |
| 11000 | 0.0766 | 0.940 | 2.545 |
| 12000 | 0.0759 | 0.936 | 2.615 |
| 14500 | **0.0753** | 0.941 | 2.487 |
| 15000 | 0.0792 | 0.942 | 2.495 |

**Best: 0.0753 at step 14500.** Continuous descent through 15k — the previously-established 0.082. Training on a single-frame with frozen SAM3 can reach 0.075 given enough budget.

---

### E12 — translate mode (Sam3QueryToIDR), single-frame

| Setting | Value |
|---------|-------|
| Mode | `translate` (Sam3QueryToIDR cross-attention) |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per frame |
| Tokens | 200 |
| Queries | **live** |
| Data | Single-frame KITTI Eigen, 23,158 train / 652 val |
| Trainable params | 12,584,771 (1.5%) |

Killed at step 4900 due to SLURM time limit; last checkpoint at step 4500.

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.1438 | 0.837 | 3.526 |
| 1000 | 0.1320 | 0.814 | 3.889 |
| 1500 | 0.1559 | 0.826 | 3.592 |
| 2000 | **0.0911** | 0.901 | 3.332 |
| 2500 | 0.1044 | 0.908 | 3.016 |
| 3000 | 0.0914 | 0.912 | 2.918 |
| 3500 | 0.0882 | 0.912 | 2.831 |
| 4000 | 0.0862 | 0.924 | 2.727 |
| 4500 | **0.0853** | 0.923 | 2.714 |

**Best: 0.0853 at step 4500** (still improving; killed before step 5000). On pace to reach ~0.083 by step 5000. 

---

### E13 — pure replace, 4-frame sequence, image encoder

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per frame |
| Tokens | 200 |
| Queries | **live** (FPN + IDR queries both from live encoder, no cache) |
| Data | `KITTISequenceDataset`, clip=4, stride=4, 5,779 clips |
| Trainable params | 11,878,211 (1.4%) |

Killed at step 4700; best checkpoint at step 4500. ~7.8 min/500 steps on RTX 5060 Ti.

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.1191 | 0.844 | 3.842 |
| 1000 | 0.1676 (spike) | 0.789 | 4.182 |
| 1500 | **0.1075** | 0.889 | 3.193 |
| 2000 | 0.1125 | 0.868 | 3.209 |
| 2500 | 0.1155 | 0.864 | 3.341 |
| 3000 | 0.0941 | 0.925 | 2.746 |
| 3500 | 0.0901 | 0.915 | 2.730 |
| 4000 | 0.0972 | 0.924 | 2.712 |
| 4500 | **0.0837** | 0.928 | 2.715 |

**Best: 0.0837 at step 4500; completed at step 5000 (0.0863).**. Sequence training converges to the single-frame floor at the same iter budget but at ~4.6× compute cost per step — no practical advantage.

---

### E14 — video encoder, 4-frame sequence

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per clip on frame 0 |
| Tokens | 200 per frame (via tracker propagation) |
| Queries | **live** (video encoder branch always was live; unchanged) |
| Data | `KITTISequenceDataset`, clip=4, stride=4, 5,779 clips |

#### Run 1 — loss divided by `B×T` 

Loss normalised by the full `B×T` frame count regardless of how many frames had LiDAR labels. Since KITTI only labels ~1 in 5 frames, most clips contribute ≤1 valid frame out of 8 — so gradients were deflated by ~`valid/B×T`. Killed early.

**Validation on sequential KITTI**

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.312 | 0.648 | 7.340 |
| 1000 | 0.233 | 0.709 | 6.638 |
| 1500 | 0.251 | 0.705 | 6.674 |
| 2000 | 0.231 | 0.699 | 6.709 |
| 2500 | 0.223 | 0.706 | 6.690 |
| 3000 | **0.222** | 0.682 | 6.852 |

Killed at step 3400; best checkpoint at step 3000. ~45 min/500 steps on RTX 5060 Ti.

**Validation on non-sequential KITTI**

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.270 | 0.664 | 6.988 |
| 1000 | 0.268 | 0.690 | 6.230 |
| 1500 | 0.202 | 0.747 | 5.584 |
| 2000 | 0.179 | 0.766 | 5.355 |
| 2500 | 0.171 | 0.777 | 5.082 |
| 3000 | **0.161** | 0.753 | 5.231 |

Killed at step 3000; best checkpoint at step 3000. ~45 min/500 steps on RTX 5060 Ti.

#### Run 2 — loss divided by `valid_frames`

Loss normalised by the actual number of frames with LiDAR labels (`any(dim=(2,3,4)).sum()`), computed upfront from masks before the forward pass. Gradient scale consistent regardless of label sparsity. Per-frame backward kept for VRAM efficiency. Completed 5000 steps.

**Validation on sequential KITTI**

| Step | abs_rel | d1 | rmse |
|-----:|--------:|---:|-----:|
| 500 | 0.283 | 0.670 | 7.136 |
| 1000 | 0.240 | 0.697 | 6.806 |
| 1500 | 0.245 | 0.701 | 6.651 |
| 2000 | 0.232 | 0.703 | 6.680 |
| 2500 | 0.222 | 0.705 | 6.692 |
| 3000 | 0.223 | 0.684 | 6.831 |
| 3500 | 0.224 | 0.695 | 6.759 |
| 4000 | 0.220 | 0.697 | 6.752 |
| 4500 | **0.220** | 0.697 | 6.715 |
| 5000 | 0.220 | 0.699 | 6.683 |

Best abs_rel: **0.2197 at step 4500**. ~47 min/500 steps on RTX 5060 Ti.

---

## Summary table

| Exp | Mode | Data | Best abs_rel | Steps | Status |
|-----|------|------|------------:|------:|--------|
| E1 baseline | Pretrained iDisc-R101 | — | **0.0600** | ~120k | reference |
| E11 | replace, live, 5k | single-frame | 0.0839 | 4000 | done |
| E11 ext | replace, live, 15k | single-frame | **0.0753** | 14500 | done |
| E12 | translate, live | single-frame | 0.0853 | 4500 | done† |
| E13 | replace, live | 4-frame sequence | 0.0837 | 4500 | done |
| E14 run1 | video encoder, live, `/B×T` | 4-frame sequence | 0.222 (seq) / 0.161 (non-seq) | 3000 / 3000 |
| E14 run2 | video encoder, live, `/valid_frames` | 4-frame sequence | **0.2197** (seq) | 4500 | done |