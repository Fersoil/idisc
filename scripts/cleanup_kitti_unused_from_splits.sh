#!/usr/bin/env bash
set -euo pipefail

# Remove files not referenced by KITTI split files.
#
# Default: dry-run (show only)
# Delete:  add --delete
#
# Usage:
#   bash scripts/cleanup_kitti_unused_from_splits.sh --data-root /work/courses/3dv/team17/idisc/datasets/kitti/train
#   bash scripts/cleanup_kitti_unused_from_splits.sh --data-root /work/courses/3dv/team17/idisc/datasets/kitti/train --delete
#
# Optional:
#   --split-dir /work/courses/3dv/team17/idisc/splits/kitti

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SPLIT_DIR="${REPO_ROOT}/splits/kitti"
DATA_ROOT=""
DELETE_MODE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --split-dir)
      SPLIT_DIR="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --delete)
      DELETE_MODE=1
      shift
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${DATA_ROOT}" ]; then
  echo "Missing required argument: --data-root" >&2
  exit 1
fi

TRAIN_SPLIT="${SPLIT_DIR}/kitti_eigen_train.txt"
TEST_SPLIT="${SPLIT_DIR}/kitti_eigen_test.txt"

if [ ! -f "${TRAIN_SPLIT}" ] || [ ! -f "${TEST_SPLIT}" ]; then
  echo "Split files not found in ${SPLIT_DIR}" >&2
  exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

# Build an unused-file list without writing temp files.
UNUSED_LIST="$(
  awk '{print $1; print $2}' "${TRAIN_SPLIT}" "${TEST_SPLIT}" | sed '/^$/d' | sort -u | awk '
    NR==FNR { keep[$0]=1; next }
    !($0 in keep) { print $0 }
  ' - <(
    cd "${DATA_ROOT}" && find . -type f | sed 's|^\./||' | sort -u
  )
)"

TOTAL_FILES="$(cd "${DATA_ROOT}" && find . -type f | wc -l)"
UNUSED_COUNT="$(printf '%s\n' "${UNUSED_LIST}" | sed '/^$/d' | wc -l)"

echo "Split dir:     ${SPLIT_DIR}"
echo "Data root:     ${DATA_ROOT}"
echo "Total files:   ${TOTAL_FILES}"
echo "Unused files:  ${UNUSED_COUNT}"

if [ "${UNUSED_COUNT}" -gt 0 ]; then
  echo
  echo "First 100 unused files:"
  awk 'NF && ++n <= 100 { print }' <<< "${UNUSED_LIST}"
fi

if [ "${DELETE_MODE}" -eq 1 ]; then
  echo
  echo "Deleting unused files..."
  while IFS= read -r rel_path; do
    [ -n "${rel_path}" ] || continue
    rm -f "${DATA_ROOT}/${rel_path}"
  done <<EOF
${UNUSED_LIST}
EOF

  # Clean up empty directories left behind.
  find "${DATA_ROOT}" -type d -empty -delete

  echo "Done. Deleted ${UNUSED_COUNT} files not referenced by split files."
else
  echo
  echo "Dry run only. Re-run with --delete to remove files."
fi
