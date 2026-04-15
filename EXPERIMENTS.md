# Experiments

This document explains the experiment setup in this repository and how to run it.

The main entrypoint is:

```bash
python scripts/run_with_hydra.py experiment=<name> [overrides...]
```

By default, runs are logged to W&B, and the main metrics are written to the run summary. The resolved config and outputs are saved per run.

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

Tracking is controlled separately:

- `tracking=wandb`
- `tracking=none`

Example:

```bash
python scripts/run_with_hydra.py experiment=baseline paths=local tracking=none
```

## Output structure

Each run writes to:

```text
outputs/runs/<timestamp>_<exp_id>_<gitsha>/
```

The run directory includes the resolved config, manifest, logs, and metrics.

## Notes

The code still bridges to the legacy JSON configs for dataset/model setup, but Hydra is the main interface now. In the future, we may move fully to Hydra and remove the legacy JSONs entirely.