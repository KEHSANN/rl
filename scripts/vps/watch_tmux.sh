#!/usr/bin/env bash
# Start the checkpoint watcher (evaluation + videos) in tmux.
#   scripts/vps/watch_tmux.sh                          # newest run under runs/
#   scripts/vps/watch_tmux.sh runs/go2_contact/<run> --interval-s 120 --num-envs 128
source "$(dirname "$0")/common.sh"
RUN="${1:-latest}"; shift || true
start_tmux "${SESSION:-watch}" uv run contact-watch --run "$RUN" "$@"
