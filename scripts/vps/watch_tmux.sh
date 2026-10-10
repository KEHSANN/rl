#!/usr/bin/env bash
# Start the checkpoint watcher (evaluation + videos) in tmux.
#   scripts/vps/watch_tmux.sh                          # newest run under runs/
#   scripts/vps/watch_tmux.sh --interval-s 120         # newest run, extra contact-watch args
#   scripts/vps/watch_tmux.sh runs/go2_contact/<run> --interval-s 120 --num-envs 128
# The run is optional; a first argument starting with '-' is a contact-watch option.
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then RUN="$1"; shift; else RUN=latest; fi
start_tmux "${SESSION:-watch}" uv run contact-watch --run "$RUN" "$@"
