"""Closed-loop Liquid runtime in the REAL simulator with the repository checkpoint.

Opt-in (slow: builds a MuJoCo-Warp env and runs ~10-40 s of sim, CPU works):

  CONTACT_SIM_TESTS=1 uv run pytest tests/test_liquid_closed_loop_sim.py

Only ``system_one`` is mocked (``--decider mock``); the env, ContactGoalCommand,
GaitPlanner, trained GRU policy and MuJoCo-Warp are real. This is evidence that
the integration path works, NOT that the Liquid model makes good decisions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = next(iter(sorted((ROOT / "weights").glob("*/checkpoints/best.pt"))), None)

pytestmark = pytest.mark.skipif(os.environ.get("CONTACT_SIM_TESTS") != "1" or CKPT is None,
                                reason="set CONTACT_SIM_TESTS=1 (needs the simulator stack and weights/*/best.pt)")


def _rollout(tmp_path, *args):
  out = tmp_path / "r.json"
  dev = os.environ.get("CONTACT_SIM_DEVICE", "cpu")
  cmd = [sys.executable, "-m", "contact_rl.runtime.rollout", "--checkpoint", str(CKPT.parent.parent) + ":best",
         "--run-root", str(ROOT), "--device", dev, "--decider", "mock", "--json", str(out), *args]
  subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, timeout=1800)
  return json.loads(out.read_text())


def test_target_mission_completes_on_measured_evidence(tmp_path):
  r = _rollout(tmp_path, "--instruction", "Move toward the target and turn left.", "--target", "1.5", "0.5",
               "--seconds", "40")
  m = r["mission"]
  assert m["state"] == "complete" and "measured" in m["outcome"], m
  assert m["decisions"] >= 3 and r["loop_counts"]["ticks"] >= 3
  assert r["interfaces"]["unchanged"] and r["interfaces"]["contact_goal_dim"] == 33
  assert r["policy_steps"] / max(r["loop_counts"]["ticks"], 1) >= 25  # policy at 50 Hz, decisions <= ~1 Hz


def test_walk_mission_keeps_deciding_from_feedback(tmp_path):
  r = _rollout(tmp_path, "--instruction", "Walk forward slowly.", "--seconds", "8")
  assert r["mission"]["state"] == "active" and r["loop_counts"]["ticks"] >= 5
  assert r["interfaces"]["unchanged"]
