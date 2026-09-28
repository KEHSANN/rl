#!/usr/bin/env bash
# One-off evaluation with metrics.csv + summary.json + video.
#   scripts/vps/eval.sh runs/go2_contact/<run>:best [more contact-eval args]
source "$(dirname "$0")/common.sh"
CKPT="${1:-best}"; shift || true
uv run contact-eval --checkpoint "$CKPT" --mode episodes --num-envs "${NUM_ENVS:-256}" --video True "$@"
