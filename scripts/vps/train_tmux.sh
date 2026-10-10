#!/usr/bin/env bash
# Start training in a detached tmux session (survives SSH disconnects).
#   scripts/vps/train_tmux.sh Mjlab-Contact-Flat-Unitree-Go2                          # paper default: 8192 envs
#   scripts/vps/train_tmux.sh Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096  # lower-VRAM fallback
#   SESSION=train2 scripts/vps/train_tmux.sh <task> --resume-from runs/go2_contact/<run>:best
# Logs: runs/<experiment>/<run>/logs/train.log (persistent, also when detached).
# Stop cleanly: scripts/vps/stop.sh train   (SIGINT -> checkpoint -> exit)
source "$(dirname "$0")/common.sh"
if [ $# -lt 1 ] || [ "${1#-}" != "$1" ]; then
  echo "usage: $0 <task> [contact-train args]   (e.g. $0 Mjlab-Contact-Flat-Unitree-Go2)" >&2; exit 2
fi
start_tmux "${SESSION:-train}" uv run contact-train "$@"
