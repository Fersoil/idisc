#!/bin/bash
set -euo pipefail
REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"

SAM_MODE="${SAM_MODE:-mask_linear}"
ITERS="${NYU_ITERS:-5000}"
NYU_DATA="${NYU_DATA:?set NYU_DATA to the export dir from export_nyu_hf.py}"

. /etc/profile.d/modules.sh
module add cuda/12.8
[[ -f "$REPO/.venv/bin/activate" ]] && source "$REPO/.venv/bin/activate"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -u scripts/run_with_hydra.py \
  experiment=finetune_sam3_image \
  dataset=nyu \
  data.data_root="$NYU_DATA" \
  run.exp_id="sam3_nyu_${SAM_MODE}" \
  method.sam_mode="$SAM_MODE" \
  model.pixel_encoder.pixel_source=sam3_memory \
  'model.pixel_encoder.sam3_trainable=[neck,encoder,decoder,head]' \
  'method.prompt.classes=[furniture, wall, floor, object]' \
  finetune.n_iters="$ITERS" \
  tracking=none
