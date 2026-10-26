#!/usr/bin/env bash
# Runtime-only setup on a fresh Linux x86_64 + NVIDIA VPS (no Docker needed).
# This branch (vps-runtime-only) is for EXECUTION only, not training.
#   bash scripts/vps/setup-runtime.sh                 # GPU: uv lock + uv sync --extra cu128, then contact-doctor
#   EXTRA=cpu bash scripts/vps/setup-runtime.sh       # CPU-only smoke-test box
#   Extra arguments, if any, go to contact-doctor.
#
# What it does, and why it differs from scripts/vps/setup.sh:
# - `uv lock` FIRST: the committed uv.lock is the full-branch lock. This
#   branch moves wandb / tensorboard / torchrunx to mjlab[train] in
#   .mjlab_ref/pyproject.toml, so the lock must be re-resolved before a
#   frozen sync. Do NOT commit the re-resolved uv.lock back to this branch
#   unless you want to pin it (different uv versions rewrite it noisily).
# - `uv sync --extra cu128` WITHOUT mjlab[train]: no wandb, no tensorboard,
#   no torchrunx are installed. Verified by `uv sync --dry-run`: the runtime
#   tree contains torch, mujoco, mujoco-warp, warp-lang, mjlab, rsl-rl-lib,
#   viser, mediapy, imageio-ffmpeg, onnx -- and none of those three.
# - Then the same mujoco-warp sensor patch as setup.sh, then contact-doctor.
#   Expected doctor result: PASS except one WARN for tensorboard (not
#   installed on purpose; contact-eval --tensorboard only warns and continues).
#
# What is NOT available on this branch (by design):
# - contact-train (not installed; the script entry was removed from pyproject)
# - scripts/vps/train_tmux.sh, scripts/vps/tensorboard.sh
# - --wandb-run-path, --agent.logger wandb, --export-onnx, multi-GPU (--gpu-ids "[0, 1]")
# For training, check out feature/liquid-ai-runtime and run setup.sh there.
source "$(dirname "$0")/common.sh"
uv lock
uv sync --extra "${EXTRA:-cu128}" --python 3.12 --frozen
.venv/bin/python scripts/patches/patch_mujoco_warp_sensor.py
uv run contact-doctor "$@"
