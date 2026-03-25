## Experimental setup for Euler

Based on https://docs.hpc.ethz.ch/software/proglang/python/ the recommended
approach is to use the Euler module system for Python + CUDA (matched toolchain)
and a **venv** on top for pip packages. This avoids conda file-quota
issues on Lustre and ensures CUDA headers match the toolkit.

Maybe it would be reasonable to switch to **uv** later on.

### 1. Load the stack modules


```bash
# python and tools
module load cuda/12.4
# Confirm:
python --version   # 3.9.18
nvcc --version     # should show the stack's CUDA version
echo $CUDA_HOME    # set by the module
```

### 2. Create a venv with access to system site-packages

```bash
cd /work/courses/3dv/team17/idisc/
python -m venv .venv
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
# The command below is how to log in to the GPU node, just for the reference
```bash 
srun --pty -A 3dv -t <number of minutes> --gpus 1 bash
```
Must be done on a **compute node** (so nvcc + GPU driver are available):

```bash
module unload cuda/12.4
module load cuda/12.8
source /work/courses/3dv/team17/idisc/.venv/bin/activate
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
# This is to check that cuda is available = True 
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
export PYTHONPATH="/work/courses/3dv/team17/idisc:$PYTHONPATH"

cd /work/courses/3dv/team17/idisc/idisc/models/ops
rm -rf build
bash ./make.sh
```

### 5. Run training / testing

# I did not do this part and it crashed because I did not have the .pt file for weights but something like this maybe? Idk...

```bash
cd /work/courses/3dv/team17/idisc
python scripts/test.py --model-file ../models/nyu_resnet101.pt \
  --config-file configs/nyu/nyu_r101.json \
  --base-path /work/courses/3dv/team17/idisc
```

### Notes

- **Do NOT install CUDA packages via micromamba/pip** (nvidia-cuda-*, cuda-nvcc, cuda-libraries-dev).
  The Euler module provides a consistent, matched CUDA toolchain.
- If you change stacks or CUDA versions, rebuild the ops extension (`rm -rf build && bash ./make.sh`).