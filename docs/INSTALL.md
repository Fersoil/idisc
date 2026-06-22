# Installation

End-to-end setup for the SAM3 + iDisc fork. The ResNet-101 baseline runs with steps
1–3 only; the SAM3 experiments additionally need step 4.

## Prerequisites

- Linux, Python 3.12, an NVIDIA GPU with CUDA 12.8 (matching `requirements-lock.txt`).
- A C++/CUDA toolchain (`nvcc`, GCC) to build the deformable-attention op.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for the exact pinned env
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## 2. Deformable-attention op

```bash
cd idisc/models/ops/ && bash ./make.sh && cd -
```

## 3. iDisc checkpoints (baseline)

Download the released iDisc weights from the
[iDisc model zoo](https://github.com/SysCV/idisc) (`kitti_resnet101.pt`, `nyu_resnet101.pt`)
and point `paths.pretrained_model` at them (see step 5).

## 4. SAM3 (for the SAM3 experiments)

SAM3 is **not on PyPI**; install it from Meta's source release and download its checkpoint.

```bash
pip install -e /path/to/sam3            # Meta AI SAM3 source (https://github.com/facebookresearch/sam3)
```

Then place the SAM3 checkpoint (`sam3.pt`) anywhere and point `paths.sam_checkpoint` at it.
SAM3 is loaded via `sam3.model_builder` (`build_sam3_image_model` / `build_sam3_video_model`);
verify with `python -c "import sam3; print(sam3.__version__)"`.

## 5. Data and paths

Prepare KITTI / NYU as in [DATA.md](DATA.md). Set dataset and checkpoint locations in
`conf/paths/local.yaml` (single GPU) or `conf/paths/cluster.yaml` (SLURM):

```yaml
pretrained_model: /path/to/kitti_resnet101.pt
sam_checkpoint:   /path/to/sam3.pt        # null disables SAM3 → baseline only
kitti_root:       /path/to/datasets/kitti
```

## Quick check

```bash
# baseline eval, no SAM3 needed
PYTHONPATH=. python scripts/run_with_hydra.py experiment=eval_idisc_kitti_image paths=local
```

See the [README](../README.md) for the full list of experiments and `scripts/launch.sh`
(SLURM) usage.
