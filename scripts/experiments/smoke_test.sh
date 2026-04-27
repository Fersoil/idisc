#!/bin/bash
#SBATCH --job-name=iDisc-smoke-test
#SBATCH --account=3dv
#SBATCH --time=04:00:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

set -euo pipefail

IDISC_REPO="$HOME/idisc"

# ── environment ──
. /etc/profile.d/modules.sh
module add cuda/12.8
source "$IDISC_REPO/.venv/bin/activate"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PYTHONPATH="$IDISC_REPO:$IDISC_REPO/sam3:${PYTHONPATH:-}"

cd "$IDISC_REPO"

RUN="python scripts/run_with_hydra.py"

EXPERIMENTS=(
  # Detection (eval_sam task)
  detect_none
  detect_singleclass
  detect_multiclass
  detect_classonly
  # Depth eval (eval task) — no SAM required
  baseline
  # Depth eval — online SAM (requires sam_checkpoint)
  branch_empty
  branch_multiclass
  branch_singleclass
  replace_multiclass
  replace_singleclass
  concat_multiclass
  concat_singleclass
  concat_classonly
  # Video (requires prior C1 cache; skip if not yet cached)
  # concat_video
)

PASS=()
FAIL=()

for exp in "${EXPERIMENTS[@]}"; do
  echo ""
  echo "════════════════════════════════"
  echo " Smoke: experiment=$exp"
  echo "════════════════════════════════"
  if $RUN experiment="$exp"; then
    PASS+=("$exp")
  else
    echo "FAILED: $exp" >&2
    FAIL+=("$exp")
  fi
done

echo ""
echo "════════════ SUMMARY ════════════"
echo "PASSED (${#PASS[@]}): ${PASS[*]}"
echo "FAILED (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "═════════════════════════════════"

[[ ${#FAIL[@]} -eq 0 ]]  # exit 0 only if all passed