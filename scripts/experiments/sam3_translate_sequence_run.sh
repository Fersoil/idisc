#!/bin/bash
#SBATCH --job-name=iDisc-sam3-translate-seq
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/sam3_translate_seq_%j.out
#SBATCH --error=logs/sam3_translate_seq_%j.err

set -euo pipefail

IDISC_REPO="$HOME/idisc"

. /etc/profile.d/modules.sh
module add cuda/12.8
if [[ -f "$IDISC_REPO/.venv/bin/activate" ]]; then
  source "$IDISC_REPO/.venv/bin/activate"
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:${PYTHONPATH:-}"

cd "$IDISC_REPO"

python -u scripts/run_with_hydra.py experiment=sam3_translate_sequence
