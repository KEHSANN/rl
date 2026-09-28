#!/usr/bin/env bash
# Run this on YOUR LAPTOP, not on the VPS. Forwards TensorBoard and the viewer
# over SSH; nothing is exposed publicly on the VPS.
#   scripts/vps/tunnel.sh user@vps-host [ssh options]
#   then open http://localhost:6006 (TensorBoard) and http://localhost:8080 (contact-play)
set -euo pipefail
HOST="${1:?usage: $0 user@host [ssh opts]}"; shift
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "${TB_PORT:-6006}:127.0.0.1:6006" -L "${VISER_PORT:-8080}:127.0.0.1:8080" "$@" "$HOST"
