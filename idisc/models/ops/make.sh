#!/usr/bin/env bash

export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" 
# export FORCE_CUDA=1 #if you do not actually have cuda, workaround

# Support both venv/virtualenv and conda/micromamba environments.
PREFIX="${VIRTUAL_ENV:-$CONDA_PREFIX}"
if [ -z "$PREFIX" ]; then
  echo "Error: neither VIRTUAL_ENV nor CONDA_PREFIX is set. Activate your environment first."
  exit 1
fi

python setup.py build install --prefix "$PREFIX"
