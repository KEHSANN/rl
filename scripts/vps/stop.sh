#!/usr/bin/env bash
# Graceful stop of a tmux job: sends Ctrl-C once (contact-train finishes the
# iteration, writes a checkpoint and exits; contact-watch finishes the current
# evaluation). Run twice to abort immediately.
#   scripts/vps/stop.sh train | watch | tb | play
S="${1:?session name}"
tmux send-keys -t "$S" C-c
echo "sent SIGINT to '$S'; follow with: tmux attach -t $S"
