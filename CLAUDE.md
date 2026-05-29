# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**SAM2Depth** — video monocular depth estimation built on a fork of iDisc. The aim is to
reduce the temporal flicker that frame-by-frame depth shows on video by feeding iDisc's
internal discretization (IDR) bottleneck with object-centric tokens that are *tracked across
frames* by a frozen SAM3 backbone. SAM3 (~840M params, frozen) replaces iDisc's
ResNet/Swin pixel encoder; a small (~12M param) iDisc head is trained on top on KITTI.
See `Sam2depth.pdf` for the research proposal and `docs/SAM2Depth/` for architecture,
results, and the experiment log.

Current state (from `docs/SAM2Depth/README.md`): SAM3 FPN features are usable, but the
frozen SAM3 decoder queries do **not** yet beat iDisc's own AFP latents. The plain
ResNet-101 iDisc is kept as the baseline and still wins on `abs_rel`.

## Commands

Everything runs through the Hydra entrypoint `scripts/run_with_hydra.py`. There is no build
step (pure Python); `idisc/models/ops/` is a prebuilt CUDA deformable-attention extension.

```bash
# Cluster (SLURM) — launch.sh wraps run_with_hydra.py with account/time/GPU/venv.
# Invoke it DIRECTLY, not via sbatch; it submits the sbatch job itself.
./scripts/launch.sh experiment=finetune_sam3_image
./scripts/launch.sh experiment=finetune_sam3_video finetune.n_iters=100   # quick shakedown
./scripts/launch.sh experiment=finetune_sam3_image --name ablation1       # tags the W&B run

# Local run (no SLURM). paths=local; note sam_checkpoint is null locally, so SAM3
# experiments need the cluster — only the ResNet baseline runs locally.
PYTHONPATH=. python scripts/run_with_hydra.py experiment=eval_idisc_image paths=local

# Format + sort imports (black + isort over idisc/ and scripts/)
bash scripts/utils/lint.sh

# Tests (pytest is a dependency; tests are ad hoc, run a file directly)
pytest <path>           # whole file
pytest <path>::<test>   # single test

# Regenerate visualization GIFs (one GPU)
sbatch scripts/utils/visualize_all.sh
```

Live experiment keys: `eval_idisc_image`, `eval_idisc_video`, `finetune_idisc_image`,
`finetune_idisc_video`, `finetune_sam3_image`, `finetune_sam3_video`. Override anything on
the CLI (`finetune.n_iters=500`, `method.sam_mode=translate`, etc.).

## Architecture (the parts that span multiple files)

**Config is the spine.** Configuration is pure Hydra under `conf/` — there are no live JSON
configs (`configs/*.json` are legacy upstream artifacts kept only for the released-checkpoint
README tables). The flow:

1. `conf/config.yaml` composes `dataset / model / finetune / experiment / paths / tracking`,
   plus the `run` and `method` blocks. One file per live experiment in
   `conf/experiment/<name>.yaml` (`# @package _global_`); 25 retired ones live in
   `conf/experiment/legacy/`.
2. `run_with_hydra.py` calls `build_runtime_config` (`idisc/utils/config_bridge.py`), which
   **validates the schema and is the single gate** for the legal combinations of
   `run.task`, `run.dataset_mode`, `method.idr_source`, `method.sam_mode`, and
   `method.prompt.*` against the configured encoder. Add new invariants there, not scattered
   through the model code.
3. It snapshots the resolved config to `output/runs/<ts>_<exp>_<sha>/resolved_config.yaml`
   (alongside `manifest.json`, `metrics.json`, `stdout.log`) **before** dispatching to
   train/eval. Downstream viz/eval read that snapshot, not the `conf/` tree. Checkpoints go
   to `output/models/<exp_id>/`.

**Model wiring** (`idisc/models/idisc.py`). `IDisc.build(config)` is the canonical
constructor — do not call `__init__` directly. Data flow:

```
pixel_encoder → pixel_decoder (MSDeformAttn FPN) → IDR source → ISD → depth map
```

The IDR source is selected at `forward` time and is the heart of the project:
- no queries (baseline) → `AFP(decoder_outputs)`
- SAM3 queries + `sam_mode="replace"`   → `sam3_proj` Linear projection
- SAM3 queries + `sam_mode="translate"` → `Sam3QueryToIDR` cross-attention

Encoders share one `forward` path via the duck-typed `yields_instance_queries` flag
(`sam3_encoder.py`, `sam3_video_encoder.py` expose it; ResNet does not). `build()` is pure —
it copies the config before writing derived values (e.g. `embed_dims`) back for the
sub-builders, so the caller's config is never mutated.

**Weight save/load** is asymmetric and worth understanding before touching it
(`scripts/train.py`, `idisc/models/idisc.py`):
- Save (`_trainable_state_dict`): drops everything under `pixel_encoder.sam_model.` /
  `video_model.` — the frozen SAM3 backbone is rebuilt from `sam_checkpoint` at construction,
  so persisting it would bloat checkpoints ~16x.
- Load (`load_pretrained`, `strict=False`): missing frozen-backbone keys are expected and
  silent; missing **non-backbone** keys mean a *trainable* param was left at random init and
  are warned about, grouped per module. That warning is the main guard that loaded weights
  actually match the model — keep it meaningful.

## Coding guidelines

The bar is **readable without docs**, not heavily documented. We do not enforce strict
linting or docstring coverage; we do care about the following:

- **Don't over-abstract.** A helper, wrapper, or class that is used exactly once should be
  inlined. Reach for an abstraction when there are real second and third callers, not in
  anticipation of them. Prefer a few more lines of straightforward code over an indirection
  the reader has to chase.
- **Log and warn generously.** Surface what the code is doing, especially anything that would
  otherwise fail silently: weight loading mismatches, config resolution, freeze/unfreeze,
  fallbacks, NaN guards, skipped or truncated work. Use `warnings.warn(..., stacklevel=...)`
  (the codebase has no central logger) and `print(..., flush=True)` in the training loop.
  `IDisc.load_pretrained` is the model to follow: separate the expected case from the
  suspicious one and warn loudly only on the suspicious one, so the warning keeps its signal.
- **Readable on its own.** Match the style of the surrounding file (naming, comment density,
  idioms). Use comments and docstrings to explain *why* — the non-obvious decision, the
  invariant, the gotcha — not to restate *what* the code plainly does.
- **Keep invariants at the boundary.** New rules about valid config combinations belong in
  `config_bridge._validate`, validated once up front, not as defensive checks deep in the
  model.

## Gotchas

- `import sam3` resolves from an **installed venv package** (pinned in
  `requirements.txt` to the upstream repo), not from a `sam3/` source dir on `PYTHONPATH`.
  A fresh environment needs `pip install -r requirements.txt`; only the repo root needs to
  be on `PYTHONPATH` (for `idisc`/`scripts`).
- `paths=cluster` is the default; switch to `paths=local` off the cluster.
- `scripts/` root accumulates throwaway diagnostic scripts (`test_*.py`, `*.sh`) that are not
  committed — `scripts/experiments/` and `scripts/utils/` are the real homes.
