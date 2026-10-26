#!/usr/bin/env bash
# Runtime-only remote play: trained checkpoint + remote Liquid model (marimo).
#
# This branch installs NO training packages (no wandb/tensorboard/torchrunx,
# no contact-train) and NO Liquid packages on the robot side: the model
# lives on molab (bf16 d1-3B) behind the VPS relay, and only system_one
# moves over HTTP via contact_rl.runtime.remote_decider (stdlib only).
#
# Prerequisites (run once, in this order):
#   1. bash scripts/vps/setup-runtime.sh   (same as vps-runtime-only)
#   2. The molab side must be running: bf16 model_server + relay client
#      connected to the VPS relay (see docs/RUNTIME_REMOTE_MARIMO.md).
#   3. LIQUID_API_KEY must be set to the RELAY token (robot <-> VPS).
#      It is NOT the model token (that one stays inside molab only).
#      Never commit a token; never pass it as a CLI argument in a
#      shared log. If it is missing, this script refuses to start.
#
# LIQUID_API_URL defaults to the VPS relay used for the molab model;
# override it with the environment variable if the relay moves.
# Extra arguments are passed through, e.g.:
#   LIQUID_API_KEY=... bash scripts/vps/run-remote-play.sh \
#     --liquid-target 2.0 0.0 --liquid-stdin True
set -euo pipefail

if [ -z "${LIQUID_API_KEY:-}" ]; then
  echo "[run-remote-play] LIQUID_API_KEY is not set (relay token required)." >&2
  echo "[run-remote-play] Export it first, e.g.: export LIQUID_API_KEY='<relay-token>'" >&2
  exit 1
fi

export LIQUID_API_URL="${LIQUID_API_URL:-http://104.167.199.97:18002}"
export UV_NO_SYNC=1

exec uv run --no-sync contact-play Mjlab-Contact-Flat-Unitree-Go2 \
  --run-root weights --checkpoint best --num-envs 1 \
  --liquid True --liquid-decider remote \
  --liquid-api-url "$LIQUID_API_URL" \
  --liquid-mode closed_loop --liquid-decision-hz 1.0 \
  "$@"
