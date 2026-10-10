#!/usr/bin/env bash
# Graceful stop of a tmux job: sends Ctrl-C once (contact-train finishes the
# iteration, writes a checkpoint and exits; contact-watch finishes the current
# evaluation). A duplicate Ctrl-C within 1 s is ignored (uv run can deliver
# SIGINT twice); run this again after >1 s to abort immediately.
#   scripts/vps/stop.sh train | watch | tb | play
set -euo pipefail
S="${1:?session name}"
command -v tmux >/dev/null 2>&1 || { echo "error: 'tmux' not found" >&2; exit 1; }
tmux has-session -t "$S" 2>/dev/null || { echo "error: no tmux session '$S'" >&2; exit 1; }
tmux send-keys -t "$S" C-c
echo "sent SIGINT to '$S'; follow with: tmux attach -t $S"
