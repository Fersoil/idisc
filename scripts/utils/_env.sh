REPO="${IDISC_REPO:-$HOME/idisc}"
cd "$REPO"
. /etc/profile.d/modules.sh
module add cuda/12.8
[[ -f "$REPO/.venv/bin/activate" ]] && source "$REPO/.venv/bin/activate"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
