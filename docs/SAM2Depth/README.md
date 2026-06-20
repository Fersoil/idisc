# SAM3 + iDisc Depth Estimation

The project fork replaces iDisc's ResNet/Swin pixel encoder with a frozen SAM3 backbone
(~840M params) and trains a small iDisc head (~12M params) on top, on KITTI. The goal is to
test whether SAM3's segmentation features and decoder queries improve monocular depth over
the plain iDisc model.

So far: The SAM3 FPN features are usable, but the frozen SAM3 queries do not
beat iDisc's own AFP latents. The results:

## Results (KITTI Eigen val, abs_rel)

| Model | Encoder | IDR source | abs_rel |
|---|---|---|---:|
| `finetune-idisc-image` | ResNet-101 | AFP (baseline) | **0.060** |
| `finetune-idisc-video` | ResNet-101 | AFP (baseline) | **0.059** |
| `finetune-sam3-image`  | SAM3 image (frozen) | SAM3 → `replace` | 0.084 |
| `finetune-sam3-video`  | SAM3 video (frozen) | SAM3 → `replace` | 0.103 |

All SAM3 runs use `replace` mode, where the decoder queries are projected directly into the
IDRs. In practice the FPNs are used for almost all of the depth signal.

## Docs

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the model is wired together, the key files, the data flow, the Hydra config setup, and how to add an experiment |

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
