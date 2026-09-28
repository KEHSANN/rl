#!/usr/bin/env bash
# TensorBoard on 127.0.0.1:6006 (reach it with scripts/vps/tunnel.sh from your laptop).
#   scripts/vps/tensorboard.sh                  # --logdir runs
#   PORT=6007 scripts/vps/tensorboard.sh runs/go2_contact
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then LOGDIR="$1"; shift; else LOGDIR=runs; fi
PORT="${PORT:-6006}"
start_tmux "${SESSION:-tb}" uv run tensorboard --logdir "$LOGDIR" --host 127.0.0.1 --port "$PORT" "$@"
