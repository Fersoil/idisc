#!/bin/bash
# SLURM launcher. Invoke directly, not via sbatch; wraps your overrides into an
# sbatch call with the right account, time, and GPU constraint for the chosen
# experiment. Experiment names are the files in conf/experiment/.
#
# Usage:
#   ./scripts/launch.sh experiment=<name> [hydra_overrides...] [--name TAG]
#
# Examples:
#   ./scripts/launch.sh experiment=eval_idisc_kitti_image
#   ./scripts/launch.sh experiment=finetune_sam3_kitti_linear_mem --name ablation1
#   ./scripts/launch.sh experiment=finetune_idisc_kitti_video_temporal finetune.temporal.weight=0.1
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IDISC_REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 experiment=<name> [hydra overrides...] [--name TAG]" >&2
    echo "Experiments: see conf/experiment/*.yaml" >&2
    exit 1
fi

RUN_NAME=""
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) RUN_NAME="$2"; shift 2 ;;
        *)      HYDRA_ARGS+=("$1"); shift ;;
    esac
done

EXPERIMENT=""
for a in "${HYDRA_ARGS[@]}"; do
    [[ "$a" == experiment=* ]] && EXPERIMENT="${a#experiment=}"
done
[[ -z "$EXPERIMENT" ]] && { echo "Missing experiment=<name>." >&2; exit 1; }

case "$EXPERIMENT" in
    eval_*)                       TIME="1:00:00";  CONSTRAINT="" ;;
    *video_temporal*)             TIME="14:00:00"; CONSTRAINT="--constraint=5060ti" ;;
    finetune_idisc_*video*)       TIME="10:00:00"; CONSTRAINT="" ;;
    finetune_sam3_*video*)        TIME="14:00:00"; CONSTRAINT="--constraint=5060ti" ;;
    finetune_sam3_*mask_pool*)    TIME="8:00:00";  CONSTRAINT="" ;;
    finetune_sam3_*)              TIME="4:00:00";  CONSTRAINT="" ;;
    finetune_idisc_*)             TIME="4:00:00";  CONSTRAINT="" ;;
    *)
        echo "Unknown experiment '$EXPERIMENT'; using defaults (4h, no constraint)." >&2
        TIME="4:00:00"; CONSTRAINT=""
        ;;
esac

mkdir -p "$IDISC_REPO/logs"
[[ -n "$RUN_NAME" ]] && HYDRA_ARGS+=("+run.wandb_name=${RUN_NAME}")

INNER_CMD="python -u scripts/run_with_hydra.py tracking=wandb ${HYDRA_ARGS[*]}"
JOB_NAME="iDisc-${EXPERIMENT}"

WRAP_CMD="set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.8
if [[ -f '${IDISC_REPO}/.venv/bin/activate' ]]; then
  source '${IDISC_REPO}/.venv/bin/activate'
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME=\$(dirname \"\$(dirname \"\$(which nvcc)\")\")
# sam3 is an installed venv package (see requirements.txt), not a source dir on
# the path; PYTHONPATH only needs the repo root for the idisc/scripts imports.
export PYTHONPATH='${IDISC_REPO}:\${PYTHONPATH:-}'
cd '${IDISC_REPO}'
${INNER_CMD}"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --account=3dv
    --gpus=1
    --time="$TIME"
    --mem=48G
    --output="$IDISC_REPO/logs/${JOB_NAME}_%j.out"
    --error="$IDISC_REPO/logs/${JOB_NAME}_%j.err"
)
[[ -n "$CONSTRAINT" ]] && SBATCH_ARGS+=($CONSTRAINT)

echo "Submitting: $EXPERIMENT"
echo "  SLURM args: ${SBATCH_ARGS[*]}"
echo "  Command:    $INNER_CMD"

TMPSCRIPT=$(mktemp "${TMPDIR:-/tmp}/idisc_slurm_XXXXX.sh")
printf '#!/bin/bash\n%s\n' "$WRAP_CMD" > "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
sbatch "${SBATCH_ARGS[@]}" "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
