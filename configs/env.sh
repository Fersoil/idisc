#!/usr/bin/env bash
# configs/env.sh — source this at the top of every job script
# Override any variable by setting it before sourcing, e.g.:
#   export DATA_ROOT=/custom/path && source configs/env.sh

export SAM2DEPTH_ROOT="${SAM2DEPTH_ROOT:-$HOME/SAM2Depth}"

# Shared store — common for the whole group, read-only for data/ckpts
export STORE_ROOT="${STORE_ROOT:-$HOME/store}"
export RAW_DATA_ROOT="${STORE_ROOT}/datasets"
export PROCESSED_DATA_ROOT="${STORE_ROOT}/idisc/datasets"
export CKPT_ROOT="${STORE_ROOT}/checkpoints"

# Per-user outputs — each person writes only here
export RUN_ROOT="${STORE_ROOT}/runs/${USER}"
mkdir -p "$RUN_ROOT"
