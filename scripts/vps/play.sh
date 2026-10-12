#!/usr/bin/env bash
# Interactive viewer on 127.0.0.1:8080 (tunnel with scripts/vps/tunnel.sh).
#   scripts/vps/play.sh                                   # best checkpoint of the newest run
#   scripts/vps/play.sh best --num-envs 16
#   scripts/vps/play.sh runs/go2_contact/<run>:1500
#   PORT=8081 TASK=Mjlab-Contact-Flat-Unitree-Go2-Improved scripts/vps/play.sh latest
# The selector is optional; a first argument starting with '-' is a contact-play option.
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CKPT="$1"; shift; else CKPT=best; fi
TASK="${TASK:-Mjlab-Contact-Flat-Unitree-Go2}"
start_tmux "${SESSION:-play}" uv run contact-play "$TASK" --checkpoint "$CKPT" --host 127.0.0.1 --port "${PORT:-8080}" "$@"
