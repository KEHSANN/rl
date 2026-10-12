#!/usr/bin/env bash
# Shared helpers for the VPS scripts. Source, don't run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
# Plain `uv run` re-syncs the project env first, and without `--extra cu128`
# that sync targets the *no-extra* resolution: it would replace the CUDA torch
# build that setup.sh installed (`uv sync --extra cu128`) with a different
# torch wheel on every call. setup.sh syncs once, explicitly; every other
# `uv run` uses the env exactly as installed (--no-sync, which implies
# --frozen). Re-run setup.sh after a `git pull` that changes uv.lock.
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
need() { command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found ($2)" >&2; exit 1; }; }
need uv "curl -LsSf https://astral.sh/uv/install.sh | sh"
start_tmux() {  # start_tmux <session> <command...>
  local s="$1"; shift
  need tmux "sudo apt install tmux"
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "tmux session '$s' already exists: tmux attach -t $s" >&2; exit 1
  fi
  # The command's exit code is captured *immediately* (a later `echo` would
  # otherwise reset $? to 0 and always report success). 'exec bash' keeps the
  # window open after the command ends so errors stay visible. The snippet is
  # run by bash explicitly: the user's tmux default-shell (sh, zsh, fish) may
  # not understand bash's %q quoting.
  tmux new-session -d -s "$s" -c "$REPO" "bash -c $(printf '%q' "$(tmux_wrap "$@")")"
  echo "started tmux session '$s'  ->  tmux attach -t $s   (detach: Ctrl-b d)"
}
# NOTE: keep tmux_wrap the last function in this file (tests source it alone).
tmux_wrap() {  # shell snippet: export GPU/GL/uv env, run <command...>, report its real exit code, keep a shell
  # A tmux server that is already running keeps its *own* (stale) environment,
  # so the variables that select the GPU / GL backend / uv behaviour are
  # passed explicitly.
  local v
  for v in CUDA_VISIBLE_DEVICES MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID UV_NO_SYNC; do
    if [ -n "${!v+x}" ]; then printf 'export %s=%q; ' "$v" "${!v}"; fi
  done
  printf '%q ' "$@"
  printf '%s' '; rc=$?; echo; echo "[exited with $rc]"; exec bash'
}
