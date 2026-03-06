## Experimental setup for Euler

Based on https://docs.hpc.ethz.ch/software/proglang/python/ the recommended
approach is to use the Euler module system for Python + CUDA (matched toolchain)
and a **venv** on top for pip packages. This avoids conda file-quota
issues on Lustre and ensures CUDA headers match the toolkit.

Maybe it would be reasonable to switch to **uv** later on.

### 1. Load the stack modules


```bash
# internet access
module load eth_proxy
# python and tools
module load stack/2024-06 python_cuda/3.9.18
# python_cuda loads Python 3.9 + CUDA + NCCL + OpenBLAS with matched versions.
# Confirm:
python --version   # 3.9.18
nvcc --version     # should show the stack's CUDA version
echo $CUDA_HOME    # set by the module
```

### 2. Create a venv with access to system site-packages

```bash
cd ~/idisc
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 3. Install PyTorch + project requirements

Check the CUDA version from step 1 and pick a matching PyTorch wheel from
https://pytorch.org/get-started/locally/. For CUDA 12.4 on the 2024-06 stack it should be:

```bash
# not necessary ig, should already be in the setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 4. Build Deformable Attention CUDA extension

Must be done on a **compute node** (so nvcc + GPU driver are available):

```bash
module load stack/2024-06 python_cuda/3.9.18
source ~/idisc/.venv/bin/activate
export PYTHONPATH="$HOME/idisc:$PYTHONPATH"

cd ~/idisc/idisc/models/ops
rm -rf build                # clean any previous attempts
bash ./make.sh
```

### 5. Run training / testing

```bash
module load stack/2024-06 python_cuda/3.9.18
source ~/idisc/.venv/bin/activate
export PYTHONPATH="$HOME/idisc:$PYTHONPATH"

cd ~/idisc
python scripts/train.py ...
```

### Notes

- **Do NOT install CUDA packages via micromamba/pip** (nvidia-cuda-*, cuda-nvcc, cuda-libraries-dev).
  The Euler module provides a consistent, matched CUDA toolchain.
- If you change stacks or CUDA versions, rebuild the ops extension (`rm -rf build && bash ./make.sh`).