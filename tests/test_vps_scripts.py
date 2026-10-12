"""Static checks for the VPS shell scripts / systemd units (no deps needed)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VPS = ROOT / "scripts" / "vps"


def test_train_unit_keeps_paper_num_envs():
  unit = (VPS / "systemd" / "contact-train@.service").read_text()
  exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
  assert "num-envs" not in exec_line  # 8192 comes from the task config


def test_gitignore_covers_runtime_artifacts():
  lines = (ROOT / ".gitignore").read_text().splitlines()
  for pat in ("runs/", "*.mp4", "*.pt", "*.onnx"):
    assert pat in lines


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", sorted(p.name for p in VPS.glob("*.sh")))
def test_shell_syntax(script):
  subprocess.run(["bash", "-n", str(VPS / script)], check=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("code", [0, 3])
def test_tmux_wrapper_reports_real_exit_code(code):
  # Source only the tmux_wrap helper (common.sh itself requires uv).
  src = (VPS / "common.sh").read_text()
  fn = src[src.index("tmux_wrap() {"):]
  snippet = subprocess.run(
    ["bash", "-c", fn + f'\ntmux_wrap bash -c "exit {code}"'], check=True, capture_output=True, text=True
  ).stdout
  snippet = snippet.replace("exec bash", "true")
  out = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True).stdout
  assert f"[exited with {code}]" in out
