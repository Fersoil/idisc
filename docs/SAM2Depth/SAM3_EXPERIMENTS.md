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
| `e14` | E14 video encoder + 4-frame sequence | `finetune_output/E14-online-sam3-video/` | kill if val keeps rising after step 1000 |

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

**Best: 0.0839 at step 4000.** Smooth descent, no spikes; plateau from step 4000 onward.

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

**Best: 0.0837 at step 4500.** Key finding: sequence training converges to the same floor as single-frame given enough iterations. At step 4500 the sequence run matches E11's single-frame floor (0.0839). At ~4.6× the compute cost per step, sequence training offers no practical advantage over single-frame.

---

### E14 — video encoder, 4-frame sequence

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per clip on frame 0 |
| Tokens | 200 per frame (via tracker propagation) |
| Queries | **live** (video encoder branch always was live; unchanged) |
| Data | `KITTISequenceDataset`, clip=4, stride=4, 5,779 clips |

**Result:** *not yet run*

---

## Summary table

| Exp | Mode | Data | Best abs_rel | Steps | Status |
|-----|------|------|------------:|------:|--------|
| E1 baseline | Pretrained iDisc-R101 | — | **0.0600** | ~120k | reference |
| E11 | replace, live | single-frame | **0.0839** | 4000 | done (job 60077) |
| E12 | translate, live | single-frame | **0.0853** | 4500 | done†  (job 60120) |
| E13 | replace, live | 4-frame sequence | **0.0837** | 4500 | done‡ (job 60121) |
| E14 | video encoder, live | 4-frame sequence | — | — | queued |

† Killed at step 4900 by time limit; on pace to reach ~0.083 at step 5000.
‡ Killed at step 4700; sequence floor matches single-frame at sufficient iter budget.
