# SAM3 + iDisc Architecture

## Overview

This fork replaces iDisc's ResNet/Swin pixel encoder with a frozen SAM3 backbone (~840M
params). SAM3's internal feature pyramid (FPN) provides the features that drive depth
prediction, and a small trainable head (~12M params, the size of `best_sam_finetuned.pt`)
made up of `pixel_decoder`, `AFP`/`ISD`, and `sam3_proj` is trained from random
initialization on KITTI. Query embeddings from SAM3's transformer decoder are also routed
through the head, but in practice they contribute very little.
The plain ResNet-101 iDisc, without SAM3, is kept as the baseline.

---

## Key files

| File | Role |
|------|------|
| `idisc/models/sam3_encoder.py` | `Sam3PixelEncoder`: wraps the SAM3 image model (image only). Extracts a 3-level FPN and 200 decoder-slot queries per frame, hooking the transformer decoder to capture hidden states. |
| `idisc/models/sam3_video_encoder.py` | `Sam3VideoPixelEncoder`: wraps `Sam3VideoInference`. Processes 4-frame clips with temporal tracker propagation and exposes per-frame FPN plus queries. |
| `idisc/models/id_module.py` | `AFP` (Attention Feature Pyramid), `ISD` and `ISDHead` (Internal Slots Decoder), `Sam3QueryToIDR` (the cross-attention query translator). |
| `idisc/models/idisc.py` | `IDisc`, the top-level model. Builds the encoder, `pixel_decoder`, and `AFP`/`ISD`; `forward` selects the IDR source by `sam_mode`. |
| `idisc/dataloders/kitti_sequence.py` | `KITTISequenceDataset`, which emits `(T, 3, H, W)` clips with stride-based non-overlapping sampling.  |
| `scripts/train.py` | `run_train`, the training loop called by `run_with_hydra.py` for `task=train`. Handles freeze/unfreeze, per-param-group LR, sequence flattening, and the NaN guard. |
| `scripts/experiments/eval_depth.py` | `run_eval`, the evaluation loop (abs_rel, d1, rmse, and so on), called for `task=eval`. |
| `scripts/run_with_hydra.py` | The Hydra entrypoint. Composes the config, writes `resolved_config.yaml` and `metrics.json` per run, and dispatches train or eval. |

---

## Running experiments

`scripts/launch.sh` wraps `run_with_hydra.py` in an `sbatch` call with the account, time,
GPU constraint, CUDA module, and venv already set. Run it directly, not through `sbatch`.

```bash
./scripts/launch.sh experiment=<name> [hydra.overrides...] [--name TAG]
```

Naming: `{task}_{backbone}_{dataset}_{mode}`.

| `experiment=` | What | Notes |
|---------------|------|-------|
| `eval_idisc_kitti_image` / `eval_idisc_kitti_video` | Released iDisc-R101 eval | no training |
| `finetune_idisc_{kitti,nyu}_image` | iDisc-R101 finetune, single-frame | AFP baseline |
| `finetune_idisc_kitti_video[_temporal]` | iDisc-R101 4-frame clips | AFP baseline / + temporal loss |
| `finetune_sam3_{kitti,nyu}_<idr>_<pixel>[_frozen]` | Frozen SAM3 image encoder + iDisc | idr∈{linear,adapter,mask_linear,mask_pool}, pixel∈{mem,msda} |
| `finetune_sam3_kitti_video` | Frozen SAM3 video encoder + iDisc | needs a 16 GB GPU (`--constraint=5060ti`) |

Examples (one config per variant — no axis overrides):
```bash
./scripts/launch.sh experiment=finetune_sam3_kitti_linear_mem finetune.n_iters=500  # quick shakedown
./scripts/launch.sh experiment=finetune_sam3_kitti_adapter_mem --name ablation1
./scripts/launch.sh experiment=finetune_sam3_kitti_mask_pool_mem_frozen
```

Each run produces `output/runs/<timestamp>_<exp_id>_<git_sha>/` (holding
`resolved_config.yaml`, `metrics.json`, `manifest.json`, and `stdout.log`) and checkpoints
under `output/models/<exp_id>/`.

---

## Data flow

```
KITTI RGB frame  (B, 3, 352, 1216)
       │
       ▼
pixel_encoder.forward(image)         (Sam3PixelEncoder / Sam3VideoPixelEncoder / ResNet)
  ├── SAM3 backbone.forward_image()  ──► FPN [3 × (B, 256, H/s, W/s)]
  └── decoder hook ──────────────────► queries (B, 200, 256)   [SAM3 encoders only]
       │                                      │
       ▼                                      ▼
IDisc.forward(image, instance_queries, sam_mode)
  ├── MSDeformAttnPixelDecoder(FPN) ──► refined FPN + decoder_outputs
  ├── IDR source dispatch:
  │     queries + sam_mode=replace   ──► sam3_proj(queries)        → IDRs
  │     queries + sam_mode=translate ──► Sam3QueryToIDR(queries)   → IDRs
  │     no queries (baseline)        ──► AFP(decoder_outputs)      → IDRs (B, 32, 128)
  └── ISD(fpn, IDRs) ─────────────────► depth map (B, 1, H, W)
```

`sam_mode` only matters when SAM3 queries are present. The plain iDisc baseline always
takes the AFP path (the `instance_queries is None` branch in `idisc/models/idisc.py`,
`forward`).

---

## Config system (pure Hydra)

Configuration is composed entirely from `conf/`; there are no legacy JSON configs in this
path anymore.

- `conf/config.yaml`: root defaults for `dataset`, `model`, `finetune`, `experiment`, `paths`, and `tracking`, plus the `run` and `method` blocks.
- `conf/experiment/<name>.yaml` (`# @package _global_`): one file per live experiment. It overrides `model`, `dataset`, and `finetune`, and sets `run.{exp_id,task,dataset_mode}`, `method.{idr_source,sam_mode,prompt}`, and `tags`.
- `conf/model/`: `idisc_r101`, `idisc_sam3_image`, `idisc_sam3_video` (encoder plus head architecture).
- `conf/finetune/`: `image`, `video` (iters, LR, batch, checkpoint dir).
- `conf/paths/`: `cluster`, `local`. `conf/tracking/`: `none`, `wandb`.

`run_with_hydra.py` composes the config, runs `build_runtime_config`
(`idisc/utils/config_bridge.py`) to flatten it into the runtime dict, and snapshots it to
`resolved_config.yaml`. The visualization and evaluation scripts read that
`resolved_config.yaml`, not the `conf/` tree.

To add a new experiment:
1. If it needs a new architecture, add a `conf/model/<name>.yaml`.
2. Add `conf/experiment/<name>.yaml`, overriding the right model/dataset/finetune and setting `run` and `method`.
3. Add a case in `scripts/launch.sh` (time and GPU constraint) so `experiment=<name>` launches.

---
