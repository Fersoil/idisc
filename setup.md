# Experiment setup

This short doc is a tutorial on setting up the environment for SAM2Depth project.

## Setup for student cluster

For more detailed info please refer to (not too detailed) [docs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda).

For our student cluster setup we are going to use the simples python venv module along the `module` utility of the cluster.


### 1. Load the stack modules

Here we go:
```bash

module add cuda/12.8 # could also go with 13.0, this is a hard requirement for blackwell gpus (rtx 50 series)
```

I think this step could be ommited, but I am not sure why.

### 2. Create a venv 

Choose a directory for your venv. We have a common venv in the `~/store` directory, but I think recommended approach would be to use seperate python envs, as below:
```
cd ~/idisc/
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch + project requirements

Check the CUDA version from step 1 and pick a matching PyTorch wheel from
https://pytorch.org/get-started/locally/. 
Here we use cuda 12.8 that supports GPUs used in the student cluster:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 # this should match module version
pip install -r requirements.txt

```
### 4. Build Deformable Attention CUDA extension

Now, lets build the deformable attention CUDA extension. 

> This should be done on the GPU node. 
> For the reference, logging in command:
>
> `srun --pty -A 3dv -t <number of minutes> --gpus 1 --pty /bin/bash` 


Now, run:

```bash
source .venv/bin/activate
# point to correct cuda drivers dir
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# add the ground truth repo clone as a reference
export IDISC_REPO_PATH="$HOME/idisc"
export PYTHONPATH="$IDISC_REPO_PATH:$PYTHONPATH"
cd "$IDISC_REPO_PATH/idisc/models/ops"

# remove the stale build if exists
rm -rf build
bash make.sh

```
Now the module should be fresh and clean ;).
### 5. Run training / testing

```bash

export CHECKPOINTS_PATH="/work/courses/3dv/team17/models"
export BASE_PATH="/work/courses/3dv/team17/idisc"

cd $IDISC_REPO_PATH

python scripts/test.py --model-file $CHECKPOINTS_PATH/nyu_resnet101.pt \
  --config-file configs/nyu/nyu_r101.json \
  --base-path $BASE_PATH
```

### 6. Install SAM3

All of our current experiments use SAM3 module. SAM3 lies under `sam3/` and must be installed as a regular package.

```bash
source .venv/bin/activate
# install if not installed previously
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[notebooks]"
```

Now, we want to download the SAM3 checkpoint to scratch:

```bash
# auth to huggingface
hf auth login

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='facebook/sam3', local_dir=f'/work/scratch/{os.getenv("USER")}/sam3_checkpoints')
EOF
```


Finally, install the `sam3/` module.

```bash
pip install sam3/
```

### 7. Run the experiments!

Just run 
```bash
python scripts/run_with_hydra.py 
```

If you need some help, just type:
```bash
python scripts/run_with_hydra.py  --help
```


For a detailed description on how to run the experiments please refer to [experiments description](./EXPERIMENTS.md).

### Notes


If you change stacks or CUDA versions, rebuild the ops extension (`rm -rf build && bash ./make.sh`).

Remember to use cuda version supported by Blackwell GPU used in student cluster, some [docs](https://developer.nvidia.com/cuda/gpus), also look up the `TORCH_CUDA_ARCH_LIST` in [make.sh](idisc/models/ops/make.sh).


You might also need to adjust the source paths for you dataset. 


For faster inference you might want to install:
```bash
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```




## Experimental setup for Euler (deprecated)

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