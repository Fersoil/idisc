# SAM3 + iDisc Depth Estimation

The project fork replaces iDisc's ResNet/Swin pixel encoder with a frozen SAM3 backbone
(~840M params) and trains a small iDisc head (~12M params) on top, on KITTI. The question:
is the depth signal in iDisc's internal discretization (the IDR bottleneck), so that a
grounded, object-aware partition from SAM3 would improve depth?

The answer is no: **depth accuracy is invariant to the partition's source.** Sourcing the
IDRs from SAM3 — queries via a linear projection or an attention adapter, or mask-pooled
object centers — leaves accuracy unchanged, and the partition's object-grounding does not
help. The partition is still *used* (ablating it multiplies error several-fold), but the head
treats it as a generic scene-specific container; depth supervision then erases the masks'
object structure (region/depth R² 0.75 → 0.26 under fine-tuning), which is why object-token
tracking cannot deliver temporal stability in this model.

## Results (KITTI Eigen test — AbsRel / δ1 / RMSE)

SAM3 rows use the fused-memory pixel source with the SAM3 post-trunk unfrozen.

| IDR source | steps | AbsRel ↓ | δ1 ↑ | RMSE ↓ |
|---|---|---|---|---|
| ResNet-101 (baseline) | 45k | **0.0600** | 0.9638 | 2.362 |
| SAM3 linear (queries) | 5k / 15k | 0.0691 / 0.0618 | 0.9546 / 0.9643 | 2.470 / 2.347 |
| SAM3 adapter (attention) | 5k / 15k | 0.0685 / 0.0607 | 0.9548 / 0.9652 | 2.459 / 2.310 |
| SAM3 mask-linear (masks) | 5k / 15k | 0.0689 / 0.0619 | 0.9549 / 0.9638 | 2.494 / 2.326 |

At a matched budget the three SAM3 sources agree to within ~0.001 AbsRel; with more training
they all reach the baseline (the adapter passing it on δ1 and RMSE at 15k). The longer budget,
not the choice of source, closes the gap. Full number → run provenance and re-run commands:
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Docs

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the model is wired together, the key files, the data flow, the Hydra config setup, and how to add an experiment |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | every reported number → the run/method and config that produced it, with re-run commands |

Older experiment notes, archived under [legacy/](legacy/):
`legacy_setup.md` (cluster environment setup),
`experiments.md` / `IDR_VISUALIZATION.md` (experiment log and IDR/mask viz),
`legacy_SAM3_EXPERIMENTS.md`, `legacy_SAM3_EXPERIMENTS_cached.md`, `legacy_EXPERIMENTS.md` (E1–E20 run logs).

## Quickstart

Everything runs on the cluster. `scripts/launch.sh` wraps
`run_with_hydra.py` in an sbatch job with the CUDA module, venv, and a GPU already
configured, so that setup does not need to be repeated for every run.

```bash
# train / eval — the full list of experiment keys is in ARCHITECTURE.md
./scripts/launch.sh experiment=finetune_sam3_kitti_linear_mem
./scripts/launch.sh experiment=eval_idisc_kitti_image
./scripts/launch.sh experiment=finetune_sam3_kitti_video finetune.n_iters=500   # quick shakedown

# regenerate the visualization GIFs (one GPU, ~5 min)
sbatch scripts/vis/visualize_all.sh        # writes to output/runs/viz/*
```

Every run writes its artifacts to `output/runs/<timestamp>_<exp_id>_<sha>/` (the
`resolved_config.yaml`, `metrics.json`, and logs), and the checkpoints go to
`output/models/<exp_id>/`. The visualization and evaluation scripts read a run's
`resolved_config.yaml` rather than the `conf/` tree directly.

## Repo layout (fork-specific)

```
conf/                     Hydra config tree (config.yaml + experiment/ model/ finetune/ paths/ tracking/)
idisc/models/             sam3_encoder.py, sam3_video_encoder.py, id_module.py, idisc.py
idisc/dataloders/         kitti_sequence.py (clip sampling), kitti.py, ...
scripts/launch.sh         SLURM launcher (train/eval)
scripts/train.py          training loop (run_train)
scripts/experiments/      eval_depth.py (run_eval)
scripts/utils/            visualize_sequence.py, visualize_sam3.py, visualize_sam3_masklets.py, visualize_all.sh
output/runs/, output/models/   per-run artifacts and checkpoints
docs/SAM2Depth/           this documentation (legacy/ holds archived experiment notes)
```
