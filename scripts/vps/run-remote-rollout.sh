#!/usr/bin/env bash
# Headless version of run-remote-play.sh: same trained checkpoint and the
# same remote Liquid model, but no viewer; writes a JSON report to stdout
# (or to --json FILE). See docs/RUNTIME_REMOTE_MARIMO.md for prerequisites.
set -euo pipefail

if [ -z "${LIQUID_API_KEY:-}" ]; then
  echo "[run-remote-rollout] LIQUID_API_KEY is not set (relay token required)." >&2
  exit 1
fi

export LIQUID_API_URL="${LIQUID_API_URL:-http://104.167.199.97:18002}"
export UV_NO_SYNC=1

exec uv run --no-sync python -m contact_rl.runtime.rollout \
  --run-root weights --checkpoint best \
  --decider remote --api-url "$LIQUID_API_URL" \
  "$@"
