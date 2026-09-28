#!/usr/bin/env bash
# One-time setup on a fresh Linux x86_64 + NVIDIA VPS (no Docker needed).
#   scripts/vps/setup.sh            # uv sync --extra cu128 --frozen, then contact-doctor
source "$(dirname "$0")/common.sh"
uv sync --extra cu128 --python 3.12 --frozen
uv run contact-doctor "$@"
