# Experiments

This document explains the experiment setup in this repository and how to run it.

The main entrypoint is:

```bash
python scripts/run_with_hydra.py experiment=<name> [overrides...]
```

By default, runs are not tracked remotely. Pass `tracking=wandb` to enable W&B logging. The resolved config and outputs are always saved per run.

## Experiment groups

### SAM3 detection

These runs check how SAM3 behaves on KITTI without depth prediction.

- `detect_none`: no prompt.
- `detect_singleclass`: one class at a time.
- `detect_multiclass`: multiple classes in one prompt.
- `detect_classonly`: class labels only.

Run example:

```bash
python scripts/run_with_hydra.py experiment=detect_multiclass
```

### Depth evaluation

These runs compare different ways of using SAM with iDisc.

- `baseline`: plain iDisc, no SAM.
- `branch_*`: the raw branch path. It uses `avg_pool2d` on features.
- `replace_*`: SAM replaces AFP.
- `concat_*`: SAM features are concatenated with AFP.

Run examples:

```bash
python scripts/run_with_hydra.py experiment=baseline
python scripts/run_with_hydra.py experiment=branch_empty
python scripts/run_with_hydra.py experiment=replace_singleclass
python scripts/run_with_hydra.py experiment=concat_video
```

The suffix describes the prompt setup:

- `_empty`: no prompt.
- `_singleclass`: each class is queried separately.
- `_multiclass`: all classes are queried together.
- `_classonly`: class labels only.
- `_video`: cached video queries.

### Finetuning

These runs finetune iDisc with SAM queries.

- `finetune_replace_multiclass`
- `finetune_replace_singleclass`
- `finetune_concat_singleclass`
- `finetune_concat_video`

Example:

```bash
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass
```

A faster preset is also available:

```bash
python scripts/run_with_hydra.py experiment=finetune_concat_singleclass finetune=fast
```

### Cache creation

Cached video runs need precomputed queries first.

```bash
python scripts/run_with_hydra.py experiment=cache_video
```

## Paths and tracking

You can switch path presets with:

- `paths=local`
- `paths=cluster`

Tracking is disabled by default. Enable W&B with `tracking=wandb`:

```bash
python scripts/run_with_hydra.py experiment=baseline tracking=wandb
```

To run without tracking (default):

```bash
python scripts/run_with_hydra.py experiment=baseline paths=local
```

## Output structure

Each run writes to:

```text
outputs/runs/<timestamp>_<exp_id>_<gitsha>/
```

The run directory includes the resolved config, manifest, logs, and metrics.

## Notes

The code still bridges to the legacy JSON configs for dataset/model setup, but Hydra is the main interface now. In the future, we may move fully to Hydra and remove the legacy JSONs entirely.
## Files

| Path | What |
|------|------|
| `scripts/experiments/run_experiment.sh` | SLURM dispatcher |
| `scripts/experiments/eval_depth.py` | depth eval (E exps) |
| `scripts/experiments/eval_sam.py` | detection eval (D exps) |
| `scripts/experiments/finetune_sam.py` | fine-tune sam3_proj+ISD (F exps) |
| `scripts/data/cache_sam3_video.py` | cache video queries (C1) |
| `idisc/models/idisc.py` | main model, forward() w/ instance_queries, sam_mode |
| `idisc/models/id_module.py` | AFP + ISD |
| `idisc/dataloders/kitti.py` | dataloader w/ sam3_cache_dir |

## Changes from original iDisc

1. sam3_proj = 3 x Linear(256->128) to project SAM3 queries into IDR space
2. forward() accepts instance_queries + raw_idrs + sam_mode for replace/concat/branch
3. KITTIDataset: sam3_cache_dir + sam3_top_k, loads .pt files, picks top-K by L2 norm
4. Sam3Processor: exposed instance_queries + topk_scores
5. Added denormalization before SAM3 (was double-normalizing w/ ImageNet stats)
6. Replaced avg_pool2d from s-seq branch w/ linear projection
