# Experiments (Stage 2 Hydra)

Hydra is now the primary experiment interface. Legacy JSON configs remain the model/data source of truth, while Hydra composes run metadata, task routing, prompts, paths, tracking, and finetune settings.

## Entrypoint

```bash
python scripts/run_with_hydra.py experiment=<name> [overrides...]
```

## Stage 2 Config Tree

```text
conf/
  config.yaml
  dataset/
    kitti.yaml
    nyu.yaml
  experiment/
    detect_none.yaml
    detect_singleclass.yaml
    detect_multiclass.yaml
    detect_classonly.yaml
    baseline.yaml
    branch_empty.yaml
    branch_multiclass.yaml
    branch_singleclass.yaml
    replace_multiclass.yaml
    replace_singleclass.yaml
    concat_multiclass.yaml
    concat_singleclass.yaml
    concat_classonly.yaml
    concat_video.yaml
    finetune_replace_multiclass.yaml
    finetune_replace_singleclass.yaml
    finetune_concat_singleclass.yaml
    finetune_concat_video.yaml
    cache_video.yaml
  finetune/
    default.yaml
    fast.yaml
  paths/
    local.yaml
    cluster.yaml
  tracking/
    none.yaml
    wandb.yaml
```

## Method Semantics

- `method.idr_source`: `afp` or `raw`
- `method.sam_mode`: `none`, `concat`, `replace`
- `method.query_source`: `online` or `cached`
- `method.prompt`: inline prompt configuration per experiment (`mode`, `classes`, `use_bbox`, `strategy`)

Bridge translation in `idisc/utils/config_bridge.py` maps these to legacy flat runtime keys used by scripts:

- `(raw, *) -> variant=branch`
- `(afp, none) -> variant=baseline`
- `(afp, replace) -> variant=sam-replace`
- `(afp, concat) -> variant=sam-concat`
- `query_source=cached -> variant=sam-cached-video, sam3_cache_dir=paths.sam3_cache_dir`

## Example Commands

```bash
# Depth evaluation
python scripts/run_with_hydra.py experiment=baseline
python scripts/run_with_hydra.py experiment=concat_singleclass tracking=none
python scripts/run_with_hydra.py experiment=concat_video

# Detection-only SAM eval (D experiments)
python scripts/run_with_hydra.py experiment=detect_singleclass

# Fine-tuning
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass finetune=fast

# Cache creation (required before cached-query runs)
python scripts/run_with_hydra.py experiment=cache_video

# Eval fine-tuned checkpoints (no duplicate experiment YAMLs needed)
python scripts/run_with_hydra.py \
  experiment=baseline \
  paths.pretrained_model=finetune_output/F4-concat-video/kitti-best.pt \
  run.exp_id=E1-ft \
  tracking=wandb

python scripts/run_with_hydra.py \
  experiment=concat_video \
  paths.pretrained_model=finetune_output/F4-concat-video/kitti-best.pt \
  run.exp_id=E10-ft \
  tracking=wandb
```

## Task Routing

`run.task` dispatches in `scripts/run_with_hydra.py`:

- `eval` -> `scripts/experiments/eval_depth.py:run_eval`
- `eval_sam` -> `scripts/experiments/eval_sam.py:run_eval_sam`
- `finetune` -> `scripts/experiments/finetune_sam.py:run_finetune`
- `cache` -> `scripts/data/cache_sam3_video.py:run_cache`

## Notes

- Legacy launcher remains available: `scripts/experiments/run_experiment.sh`.
- Outputs are written under `outputs/runs/<timestamp>_<exp_id>_<gitsha>/` with manifest, resolved config, and metrics.
