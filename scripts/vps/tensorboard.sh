#!/usr/bin/env bash
# TensorBoard on 127.0.0.1:6006 (reach it with scripts/vps/tunnel.sh from your laptop).
source "$(dirname "$0")/common.sh"
LOGDIR="${1:-runs}"; PORT="${PORT:-6006}"
start_tmux "${SESSION:-tb}" uv run tensorboard --logdir "$LOGDIR" --host 127.0.0.1 --port "$PORT"
