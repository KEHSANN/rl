#!/usr/bin/env bash
# One-off evaluation with metrics.csv + summary.json + video.
#   scripts/vps/eval.sh                                   # best checkpoint of the newest run
#   scripts/vps/eval.sh runs/go2_contact/<run>:best [more contact-eval args]
#   NUM_ENVS=512 TASK=Mjlab-Contact-Flat-Unitree-Go2-Improved scripts/vps/eval.sh latest
# The selector is optional; a first argument starting with '-' is a contact-eval option.
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CKPT="$1"; shift; else CKPT=best; fi
uv run contact-eval --task "${TASK:-Mjlab-Contact-Flat-Unitree-Go2}" --checkpoint "$CKPT" \
  --mode episodes --num-envs "${NUM_ENVS:-256}" --video True "$@"
