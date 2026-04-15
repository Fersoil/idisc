# Experiments

This repo extends [iDisc (CVPR 2023)](https://github.com/SysCV/idisc) with SAM3-guided depth estimation. Experiments run on the KITTI Eigen split using a pretrained iDisc ResNet-101 backbone. All results are evaluated zero-shot (no training) unless marked as fine-tuned (F experiments).

## Quick Start

The preferred way to launch experiments is through the Hydra wrapper, which handles config composition, run directory creation, and optional W&B logging in a single command. The legacy SLURM shell script remains available for backward-compatible cluster launches during the transition period.

**Hydra entrypoint (recommended):**
```bash
# Evaluate baseline without tracking
python scripts/run_with_hydra.py experiment=baseline tracking=none

# Evaluate baseline and log to W&B
python scripts/run_with_hydra.py experiment=baseline tracking=wandb

# Evaluate a SAM variant on KITTI with W&B
python scripts/run_with_hydra.py dataset=kitti experiment=sam_branch_empty tracking=wandb

# Override any config key inline (Hydra syntax)
python scripts/run_with_hydra.py experiment=baseline tracking=wandb run.exp_id=E1-rerun
```

**Legacy SLURM path (still supported during migration):**
```bash
sbatch scripts/experiments/run_experiment.sh <ID>
sbatch scripts/experiments/run_experiment.sh --all
```

---

## Repo Structure (Stage 1)

The Stage 1 refactor introduced a thin Hydra layer on top of the existing codebase without touching the original iDisc model or JSON configs. The key addition is `conf/` (Hydra config tree), two utility modules in `idisc/utils/`, and `scripts/run_with_hydra.py` as the new entrypoint. Everything else is unchanged.

```
idisc/
├── conf/                              # Hydra config tree — owns experiment composition
│   ├── config.yaml                    # Root config: defaults list, run identity, method flags
│   ├── dataset/
│   │   ├── kitti.yaml                 # Selects KITTI + points to legacy JSON path
│   │   └── nyu.yaml                   # Selects NYU + points to legacy JSON path
│   ├── experiment/
│   │   ├── baseline.yaml              # E1: iDisc AFP only, no SAM
│   │   └── sam_branch_empty.yaml      # E2: s-seq branch, empty prompt
│   ├── paths/
│   │   ├── cluster.yaml               # Absolute paths for the ETH cluster environment
│   │   └── local.yaml                 # Relative paths for local development
│   └── tracking/
│       ├── none.yaml                  # Disables all tracking (enabled: false)
│       └── wandb.yaml                 # Enables W&B logging to project idisc-sam3
│
├── configs/                           # Legacy JSON configs — owned by original iDisc, do not restructure
│   ├── kitti/kitti_r101.json          # Model architecture, data paths, training hyperparameters
│   └── nyu/nyu_r101.json
│
├── idisc/
│   └── utils/
│       ├── config_bridge.py           # Merges Hydra cfg + legacy JSON into a single runtime dict
│       └── tracking.py                # W&B abstraction: init_tracking / log_metrics / log_summary / finish_tracking
│
├── scripts/
│   ├── run_with_hydra.py              # New preferred entrypoint — composes config, creates run dir, calls run_eval
│   ├── experiments/
│   │   ├── run_experiment.sh          # Legacy SLURM dispatcher — kept for backward compat
│   │   ├── eval_depth.py              # Depth eval: exposes run_eval(cfg) callable + preserves legacy main()
│   │   ├── eval_sam.py                # SAM3 detection eval (D experiments)
│   │   ├── finetune_sam.py            # Fine-tunes sam3_proj + ISD (F experiments)
│   │   └── __init__.py
│   └── data/
│       └── cache_sam3_video.py        # Pre-computes SAM3 video queries and saves to disk (C1)
│
├── outputs/runs/                      # Auto-created per run, gitignored — do not commit
│   └── <exp_id>_<YYYY-MM-DD_HH-MM-SS>/
│       ├── run_manifest.json          # Run provenance: git branch, commit, paths, tracking backend
│       ├── resolved_config.yaml       # Full merged config snapshot — sufficient to reproduce the run
│       └── metrics.json              # Eval output: variant, prompt_mode, metrics dict, elapsed_s
│
├── EXPERIMENTS.md                     # This file
└── requirements.txt
```

---

## Config System

Two config layers coexist and are merged at runtime by `config_bridge.py`. The legacy JSON files own iDisc internals (model architecture, data loading, training hyperparameters) and must not be restructured. Hydra owns everything above that: which experiment to run, which dataset to select, where to find model weights, and whether to log to W&B.

| Layer | Owner | Format | Owns |
|-------|-------|--------|------|
| Model / data internals | `configs/*.json` (legacy) | JSON | Architecture, lr, crop sizes, val_dataset, etc. |
| Experiment identity | `conf/experiment/*.yaml` | YAML | exp_id, variant, use_sam, prompt_mode, tags |
| Dataset selection | `conf/dataset/*.yaml` | YAML | dataset_name, pointer to legacy JSON |
| Infrastructure paths | `conf/paths/*.yaml` | YAML | pretrained_model, sam_checkpoint, outputs_root |
| Tracking | `conf/tracking/*.yaml` | YAML | enabled, backend, project, entity, mode |

**Merge precedence:** legacy JSON loads first; Hydra values (including CLI overrides) win on conflict.

---

## Run Output Contract

Every `run_with_hydra.py` invocation creates a timestamped directory under `outputs/runs/`. The directory is self-contained: given `resolved_config.yaml` alone, the run can be reproduced exactly. When W&B is enabled, the full config dict is also uploaded at run init so every metric is traceable to its exact config.

| File | Contents |
|------|---------|
| `run_manifest.json` | exp_id, timestamp, git branch + commit SHA, dataset name, variant, tracking backend, output path |
| `resolved_config.yaml` | Complete merged config after all overrides — the single source of truth for reproducibility |
| `metrics.json` | `{"variant": ..., "prompt_mode": ..., "metrics": {...}, "elapsed_s": ...}` |

---

## Model Setup

The model is iDisc with a frozen ResNet-101 backbone and a SAM3 integration layer. The integration is zero-cost at the AFP level: SAM3 queries are projected into the same 128-dim IDR space as AFP outputs and injected into the ISD cross-attention. Fine-tuning only touches `sam3_proj` + ISD (4.6M / 59M params, 7.9%).

Four integration modes are supported, controlled by `method.variant` in the experiment config:

| Mode | What ISD attends to | AFP preserved? | sam3_proj used? |
|------|---------------------|---------------|----------------|
| `baseline` | AFP IDRs (32, 128) ×3 | ✓ | ✗ |
| `branch` | avg_pool2d of SAM hidden states ×3 | ✗ | ✗ |
| `replace` | sam3_proj(queries) → (N, 128) ×3 | ✗ | ✓ |
| `concat` | AFP + sam3_proj(queries) → (32+N, 128) ×3 | ✓ | ✓ |

> `branch` is the original s-seq implementation. `avg_pool2d` collapses spatial information and clones the same tensor across all three resolution heads — it is kept only as a diagnostic baseline.

---

## Adding a New Experiment

Adding an experiment requires one new YAML file and one row in the results table below. No Python changes are needed unless a new integration mode is introduced.

1. Create `conf/experiment/<your_exp>.yaml`:
   ```yaml
   run:
     exp_id: E11-your-exp

   method:
     variant: concat          # baseline | branch | replace | concat | sam-cached-video
     use_sam: true
     prompt_mode: singleclass # empty | multiclass | singleclass | classonly

   tags:
     - concat
     - your-tag
   ```

2. Launch and log:
   ```bash
   python scripts/run_with_hydra.py experiment=your_exp tracking=wandb
   ```

3. Add the result row to the relevant table in the **Results** section of this file.

4. W&B runs are grouped by `method.variant` — check `https://wandb.ai/<entity>/idisc-sam3`.

---

## Run Order (Dependencies)

Some experiments depend on outputs from others. The cache step (C1) takes ~4 hours and must complete before E10 and F4.

```
D1 → D2 → D3 → D4         detection baselines, no dependencies
E1                          iDisc baseline (no SAM)
E2 → E3 → E4               branch variants
E5 → E6                    replace variants
E7 → E8 → E9               concat variants
C1-cache-video              ~4h — pre-compute SAM3 video queries
E10                         needs C1
F1 → F2 → F3               fine-tune with online SAM
F4                          needs C1
E1-ft                       needs F4
E10-ft                      needs F4 + C1
```

---

## Files Reference

| Path | Purpose |
|------|---------|
| `scripts/run_with_hydra.py` | **Preferred entrypoint.** Composes Hydra config, merges with legacy JSON, creates run dir, calls `run_eval`, logs to W&B. |
| `scripts/experiments/run_experiment.sh` | Legacy SLURM dispatcher. Kept for backward compat; new experiments should use the Hydra path. |
| `scripts/experiments/eval_depth.py` | Depth evaluation logic. Exposes `run_eval(cfg: dict)` for programmatic use; `main()` preserved for direct CLI use. |
| `scripts/experiments/eval_sam.py` | SAM3 detection evaluation (D experiments). |
| `scripts/experiments/finetune_sam.py` | Fine-tunes `sam3_proj` + ISD (F experiments). Not yet wrapped by Hydra. |
| `scripts/data/cache_sam3_video.py` | Pre-computes SAM3 video queries and saves as `.pt` files (C1). |
| `idisc/utils/config_bridge.py` | Loads legacy JSON, merges with Hydra `DictConfig`, returns a plain `dict` for downstream use. |
| `idisc/utils/tracking.py` | W&B abstraction. Call `init_tracking(cfg, run_dir)` at the start, `log_metrics(...)` during eval, `finish_tracking()` at the end. |
| `conf/` | Hydra config tree. Add one file per new experiment, dataset, or path environment. |
| `configs/` | Legacy iDisc JSON configs. Do not restructure — these are loaded verbatim by `config_bridge.py`. |
| `idisc/models/idisc.py` | Main model. `forward()` accepts `instance_queries`, `raw_idrs`, `sam_mode`. |
| `idisc/models/id_module.py` | AFP and ISD modules. |
| `idisc/dataloders/kitti.py` | KITTI dataloader with `sam3_cache_dir` and `sam3_top_k` support. |

---

## Changes from Original iDisc

The following changes were made to the original iDisc codebase to support SAM3 integration:

1. Added `sam3_proj` — three `Linear(256→128)` layers projecting SAM3 queries into IDR space (trainable in F experiments)
2. Extended `forward()` to accept `instance_queries`, `raw_idrs`, and `sam_mode` for the replace/concat/branch paths
3. Extended `KITTIDataset` with `sam3_cache_dir` and `sam3_top_k`; loads `.pt` cache files and picks top-K queries by L2 norm
4. Exposed `instance_queries` and `topk_scores` from `Sam3Processor`
5. Fixed double-normalisation with ImageNet statistics before SAM3 (was normalising twice)
6. Replaced `avg_pool2d` from the original s-seq branch with the linear projection path
7. **Stage 1:** Added `conf/`, `idisc/utils/config_bridge.py`, `idisc/utils/tracking.py`, and `scripts/run_with_hydra.py`
