#!/bin/bash
#
# Unified SLURM experiment launcher.
#
# Usage:
#   ./scripts/launch.sh <experiment> [--name <label>] [-- <hydra_overrides...>]
#
# Experiments:
#   baseline    – E1  iDisc-R101 pretrained baseline (eval)
#   e11         – E11 SAM3 pure replace, single-frame
#   e12         – E12 SAM3 translate (Sam3QueryToIDR), single-frame
#   e13         – E13 SAM3 pure replace, 4-frame sequence
#   e14         – E14 SAM3 video encoder, 4-frame sequence  (needs 16 GB GPU)
#   e15         – E15 SAM3 video encoder + translate, 4-frame sequence  (needs 16 GB GPU)
#   e21         – E21 SAM3 video encoder, FPN-capture fix re-run of E14  (needs 16 GB GPU)
#
# Examples:
#   ./scripts/launch.sh e11
#   ./scripts/launch.sh e11 --name ablation1
#   ./scripts/launch.sh e11 --name ablation1 -- finetune.n_iters=100
#
set -euo pipefail

# help with CUDA memory usage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IDISC_REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT="${1:-}"
shift

if [[ -z "$EXPERIMENT" ]]; then
    echo "Usage: $0 <experiment> [--name <label>] [-- <hydra_overrides...>]" >&2
    echo "Experiments: baseline e11 e12 e13 e14 e15 e21" >&2
    exit 1
fi

RUN_NAME=""
OVERRIDES=()
[[ $# -gt 0 && "$1" == "--name" ]] && { RUN_NAME="$2"; shift 2; }
[[ $# -gt 0 && "$1" == "--" ]] && { shift; OVERRIDES=("$@"); }

# Per-experiment SLURM settings
case "$EXPERIMENT" in
    baseline)
        JOB_NAME="iDisc-baseline"
        HYDRA_EXP="baseline"
        TIME="2:00:00"
        CONSTRAINT=""
        ;;
    e11)
        JOB_NAME="iDisc-e11-pure"
        HYDRA_EXP="sam3_pure"
        TIME="2:00:00"
        CONSTRAINT=""
        ;;
    e12)
        JOB_NAME="iDisc-e12-translate"
        HYDRA_EXP="sam3_translate"
        TIME="2:00:00"
        CONSTRAINT=""
        ;;
    e13)
        JOB_NAME="iDisc-e13-pure-seq"
        HYDRA_EXP="sam3_pure_sequence"
        TIME="10:00:00"
        CONSTRAINT=""
        ;;
    e14)
        JOB_NAME="iDisc-e14-video-seq"
        HYDRA_EXP="sam3_video_sequence"
        TIME="14:00:00"
        CONSTRAINT="--constraint=5060ti"
        ;;
    e15)
        JOB_NAME="iDisc-e15-video-translate"
        HYDRA_EXP="sam3_video_translate"
        TIME="14:00:00"
        CONSTRAINT="--constraint=5060ti"
        ;;
    e21)
        JOB_NAME="iDisc-e21-video-fixed"
        HYDRA_EXP="sam3_video_fixed"
        TIME="14:00:00"
        CONSTRAINT="--constraint=5060ti"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT" >&2
        echo "Valid: baseline e11 e12 e13 e14 e15 e21" >&2
        exit 1
        ;;
esac

mkdir -p "$IDISC_REPO/logs"

# Build the inner command that SLURM will execute
if [[ -n "$RUN_NAME" ]]; then
    OVERRIDES=("+run.wandb_name=${RUN_NAME}" "${OVERRIDES[@]}")
fi

OVERRIDE_STR=""
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    OVERRIDE_STR=" ${OVERRIDES[*]}"
fi
INNER_CMD="python -u scripts/run_with_hydra.py experiment=${HYDRA_EXP} tracking=wandb${OVERRIDE_STR}"

WRAP_CMD="set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.8
if [[ -f '${IDISC_REPO}/.venv/bin/activate' ]]; then
  source '${IDISC_REPO}/.venv/bin/activate'
else
  source /work/courses/3dv/team17/idisc/.venv/bin/activate
fi
export CUDA_HOME=\$(dirname \"\$(dirname \"\$(which nvcc)\")\")
export PYTHONPATH='${IDISC_REPO}:${IDISC_REPO}/sam3:\${PYTHONPATH:-}'
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

echo "Submitting: $EXPERIMENT ($HYDRA_EXP)"
echo "  SLURM args: ${SBATCH_ARGS[*]}"
echo "  Command:    $INNER_CMD"

TMPSCRIPT=$(mktemp "${TMPDIR:-/tmp}/idisc_slurm_XXXXX.sh")
printf '#!/bin/bash\n%s\n' "$WRAP_CMD" > "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
sbatch "${SBATCH_ARGS[@]}" "$TMPSCRIPT"
rm -f "$TMPSCRIPT"
