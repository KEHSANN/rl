"""Integration contract and invalid-sample regression tests for posture output."""

import ast
import math
from pathlib import Path

import torch

from contact_rl.utils.posture_metrics import PostureAccumulator, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_state_is_counted_not_reported_as_healthy():
  acc = PostureAccumulator(torch.zeros(1))
  acc.step(torch.zeros(1), torch.zeros(1), knee_heights=torch.full((1, 4), float("nan")), illegal_found=torch.zeros(1, 19))
  row = acc.rows(0.02)[0]
  assert row["posture_invalid_samples"] == 1
  assert row["posture_samples"] == 0
  assert math.isnan(row["knee_below_fraction"])
  assert summarize([row])["posture_invalid_samples_total"] == 1


def test_rollout_wires_knees_and_contact_to_accumulator():
  tree = ast.parse((ROOT / "src/contact_rl/scripts/evaluate.py").read_text())
  rollout = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rollout")
  calls = [n for n in ast.walk(rollout) if isinstance(n, ast.Call)]
  step = next(n for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "acc" and n.func.attr == "step")
  assert {"knee_heights", "illegal_found"} <= {k.arg for k in step.keywords}
  find = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "find_bodies")
  assert any(k.arg == "preserve_order" and ast.literal_eval(k.value) is True for k in find.keywords)


def test_write_outputs_contains_posture_without_changing_legacy_csv(tmp_path):
  import json

  from contact_rl.scripts.evaluate import EvalConfig, write_outputs

  acc = PostureAccumulator(torch.zeros(1))
  acc.step(torch.zeros(1), torch.zeros(1), knee_heights=torch.full((1, 4), 0.04), illegal_found=torch.zeros(1, 19))
  cfg = EvalConfig()
  run_dir = tmp_path / "run"
  out_dir = run_dir / "metrics" / "iteration_1"
  result = write_outputs(cfg, Path("model_1.pt"), run_dir, out_dir, run_dir / "videos", 1, "cpu", acc.rows(0.02), {})
  assert result["overall"]["knee_below_fraction_pooled"] == 1.0
  disk = json.loads((out_dir / "summary.json").read_text())
  assert disk["overall"]["illegal_contact_fraction_pooled"] == 0.0
  assert "knee_RL_height_min_m" in (out_dir / "metrics.csv").read_text().splitlines()[0]
