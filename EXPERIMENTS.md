# Experiments

```bash
sbatch scripts/experiments/run_experiment.sh <ID>
```

## Hydra Wrapper

For now in this ongoing refactor I just added a think hydra layer that exectues all the experiments as Mariia implemented, but instead of sbatch I use yamls.

```bash
python scripts/run_with_hydra.py experiment=baseline tracking=none
python scripts/run_with_hydra.py experiment=baseline tracking=wandb
python scripts/run_with_hydra.py dataset=kitti experiment=sam_branch_empty tracking=wandb
```

Notes:
- Legacy shell flow remains available through `scripts/experiments/run_experiment.sh`.
- Hydra owns experiment composition and tracking toggles.
- Legacy JSON under `configs/` remains the source for model/data/training internals.
- also added tracking with wandb

---

## 1 SAM3 Detection on KITTI

| Prompt | class_prob (mean) | presence (mean) | combined (mean) | detections > 0.5 |
|--------|-------------------|-----------------|-----------------|-----------------|
| Multi-class `"car . truck . person..."` | 0.56 | **0.064** | 0.04 | 0.0 per image |
| Single-class (run 8x, once per class) | 0.71 | **0.84** | 0.60 | 2.7 per image |


| ID | Setup | Command |
|----|-------|---------|
| `D1-no-prompt` | No text prompt, SAM inference only | `sbatch scripts/experiments/run_experiment.sh D1-no-prompt` |
| `D2-singleclass` | Per-class x8 | `sbatch scripts/experiments/run_experiment.sh D2-singleclass` |
| `D3-multiclass` | One multi-class prompt | `sbatch scripts/experiments/run_experiment.sh D3-multiclass` |
| `D4-classonly` | Class-logit only (no presence multiplication) | `sbatch scripts/experiments/run_experiment.sh D4-classonly` |


---

## 2 SAM3 + iDisc

### No training

| ID | What | abs_rel | rmse | d1 | Notes | Command |
|----|------|---------|------|----|-------|---------|
| `E1-baseline` | Baseline (AFP only, no SAM) | 0.0600 | 2.363 | 0.964 | Reference | `...run_experiment.sh E1-baseline` |
| `E2-branch-empty` | s-seq branch code | 0.0662 | 2.452 | 0.959 | avg_pool2d, empty prompt | `...run_experiment.sh E2-branch-empty` |
| `E3-branch-multiclass` | s-seq + multi-class prompt | 0.0665 | 2.449 | 0.960 | Only prompt changed | `...run_experiment.sh E3-branch-multiclass` |
| `E4-branch-singleclass` | s-seq + single-class prompt | 0.0685 | 2.447 | 0.960 | Only prompt changed | `...run_experiment.sh E4-branch-singleclass` |
| `E5-replace-multiclass` | Linear proj replaces AFP, multi-class | 0.0646 | - | 0.960 | Top-32 queries | `...run_experiment.sh E5-replace-multiclass` |
| `E6-replace-singleclass` | Linear proj replaces AFP, single-class | 0.0646 | - | 0.960 | Top-32 queries | `...run_experiment.sh E6-replace-singleclass` |
| `E7-concat-multiclass` | Linear proj concat with AFP, multi-class | 0.0605 | 2.363 | 0.964 | AFP preserved | `...run_experiment.sh E7-concat-multiclass` |
| `E8-concat-singleclass` | Linear proj concat with AFP, single-class | 0.0603 | 2.362 | 0.964 | Nearly baseline | `...run_experiment.sh E8-concat-singleclass` |
| `E9-concat-classonly` | Linear proj concat with AFP, class-logit only | 0.0605 | 2.363 | 0.964 | | `...run_experiment.sh E9-concat-classonly` |
| `E10-concat-video` | Concat with AFP, cached video queries | | | | Not yet run | `...run_experiment.sh E10-concat-video` |

### Fine-tuning (AdamW lr=5e-5, OneCycleLR, 5000 steps, batch 2)

| ID | What | Best abs_rel | Step | vs Baseline | Command |
|----|------|-------------|------|-------------|---------|
| `F1-replace-multiclass` | Replace AFP, multi-class, online SAM | 0.0600 | 2500 | Same | `...run_experiment.sh F1-replace-multiclass` |
| `F2-replace-singleclass` | Replace AFP, single-class, online SAM | 0.0609 | 1000 | +1.6% worse | `...run_experiment.sh F2-replace-singleclass` |
| `F3-concat-singleclass` | Concat with AFP, single-class, online SAM | 0.0591 | 2000 | -1.5% but unstable | `...run_experiment.sh F3-concat-singleclass` |
| `F4-concat-video` | Concat with AFP, cached video queries | | | Not yet run | `...run_experiment.sh F4-concat-video` |

### Eval fine-tuned model

| ID | What | Command |
|----|------|---------|
| `E1-ft-baseline` | F4 checkpoint, AFP only (regression check) | `...run_experiment.sh E1-ft-baseline` |
| `E10-ft-video` | F4 checkpoint + cached video queries | `...run_experiment.sh E10-ft-video` |

### Prerequisite

| ID | What | Command |
|----|------|---------|
| `C1-cache-video` | Pre-compute SAM3 video queries for all seqs (~2h) | `...run_experiment.sh C1-cache-video` |

---

## Code changes

What was chnaged:
- Replaced `adaptive_avg_pool2d` with `nn.Linear(256, 128)` per resolution
- Select top-32 queries by detection score (not average all 200)
- Single-class prompting
- Found that iDisc normalizes images before passing to SAM3 (double normalization, SAM does this too), added denormalization before SAM3

---

## Run order

```
D1 → D2 → D3 → D4                        SAM3 detection (no deps)
E1-baseline                                iDisc baseline (no deps)
E2 → E3 → E4                              Branch s-seq (no deps)
E5 → E6                                   Replace AFP (no deps)
E7 → E8 → E9                              Concat with AFP (no deps)
C1-cache-video                             Cache video queries (~2h, no deps)
E10-concat-video                           needs C1
F1 → F2 → F3                              Finetune online SAM (no deps)
F4-concat-video                            needs C1
E1-ft-baseline                             needs F4
E10-ft-video                               needs F4 + C1
```

---

## File structure

| Path | What |
|------|------|
| `scripts/experiments/run_experiment.sh` | Entry point for all experiments |
| `scripts/experiments/eval_comparison.py` | Eval script (needs `--variant`, `--prompt-mode`) |
| `scripts/experiments/finetune_sam.py` | Fine-tune (needs `--mode`, `--prompt-mode`) |
| `scripts/experiments/train.py` | Original iDisc training |
| `scripts/data/cache_sam3_video.py` | Cache SAM3 video queries to disk |
| `scripts/data/parse_sequences.py` | Parse KITTI sequences from Eigen splits |
| `scripts/utils/` | lint, shell helpers |
| `idisc/models/idisc.py` | Main model, `forward()` accepts `instance_queries` |
| `idisc/dataloders/kitti.py` | KITTI loader, supports `sam3_cache_dir` |
| `sam3/sam3/model/sam3_image_processor.py` | Modified to expose instance queries |

---

## Notes

`eval_comparison.py` needs these `--variant` values implemented:
- `baseline` -- iDisc AFP only
- `branch` -- old s-seq code path (avg_pool2d, same IDRs for all resolutions)
- `sam-replace` -- linear projection replaces AFP entirely
- `sam-multiclass`, `sam-singleclass`, `sam-classonly` -- concat with AFP
- `sam-cached-video` -- concat with AFP using cached video queries
- `sam-detection-only` -- SAM3 detection stats only, no depth

`finetune_sam.py` needs these args implemented:
- `--mode replace|concat` -- replace AFP or concat
- `--prompt-mode multiclass|singleclass` -- how to prompt SAM3 (for online inference)
- `--sam-checkpoint` -- for online SAM3 inference (F1/F2/F3)
- `--sam3-cache-dir` -- for cached video queries (F4)
