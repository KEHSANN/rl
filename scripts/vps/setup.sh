#!/usr/bin/env bash
# One-time setup on a fresh Linux x86_64 + NVIDIA VPS (no Docker needed).
# Re-run it after every `git pull` that changes uv.lock: the other scripts use
# the environment as installed (UV_NO_SYNC=1, see common.sh).
#   scripts/vps/setup.sh                 # uv sync --extra cu128 --frozen, then contact-doctor
#   EXTRA=cpu scripts/vps/setup.sh       # CPU-only smoke-test box
#   Extra arguments, if any, go to contact-doctor.
source "$(dirname "$0")/common.sh"
uv sync --extra "${EXTRA:-cu128}" --python 3.12 --frozen
.venv/bin/python scripts/patches/patch_mujoco_warp_sensor.py
uv run contact-doctor "$@"
