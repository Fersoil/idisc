# Installation

End-to-end setup for the SAM3 + iDisc fork. The ResNet-101 baseline needs steps 1–3;
the SAM3 experiments additionally need the SAM3 checkpoint (step 4).

## Prerequisites

- Linux, Python 3.12, an NVIDIA GPU with CUDA 12.8 (matching `requirements-lock.txt`).
- A C++/CUDA toolchain (`nvcc`, GCC) to build the deformable-attention op.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch + torchvision first, matching your CUDA (we used CUDA 12.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# SAM3 from source (requirements.txt lists `sam3`, which is not on PyPI):
pip install -e /path/to/sam3
pip install -r requirements-lock.txt
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

## 4. SAM3 checkpoint (for the SAM3 experiments)

The SAM3 package is installed in step 1. Download the SAM3 backbone checkpoint (`sam3.pt`)
from [Meta AI SAM3](https://github.com/facebookresearch/sam3), place it anywhere, and point
`paths.sam_checkpoint` at it (step 5). Verify the package with
`python -c "import sam3; print(sam3.__version__)"`.

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
