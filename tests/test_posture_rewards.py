"""CPU characterization of the existing posture rewards and Go2 wiring.

Extract only the production function definitions to avoid importing the GPU
simulator stack. XML/config assertions complement, not replace, a runtime
sensor smoke test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "src/contact_rl/tasks/contact"


def reward_functions():
  tree = ast.parse((TASK / "mdp/rewards.py").read_text())
  names = {"knee_height_below", "base_height_below", "base_tilt_l2", "undesired_contact_count"}
  nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
  # Postpone production annotations; keep the function bodies unchanged.
  module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
  ns = {"torch": torch, "_DEFAULT_ASSET_CFG": SimpleNamespace(name="robot")}
  exec(compile(ast.fix_missing_locations(module), "posture_rewards", "exec"), ns)
  return ns


def scene(z):
  data = SimpleNamespace(body_link_pos_w=torch.zeros(len(z), 6, 3))
  data.body_link_pos_w[:, [1, 2, 4, 5], 2] = torch.tensor(z)
  env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
  asset = SimpleNamespace(name="robot", body_ids=[1, 2, 4, 5])
  return env, asset


def test_knee_reward_uses_selected_origins_and_sums_legs():
  env, asset = scene([[0.155] * 4, [0.08] * 4, [0.155, 0.155, 0.015, 0.015]])
  result = reward_functions()["knee_height_below"](env, asset)
  torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 1.625]))
  torch.testing.assert_close(-5.0 * result, torch.tensor([0.0, 0.0, -8.125]))


def test_hovering_knee_is_penalised_without_contact():
  env, asset = scene([[0.04, 0.155, 0.155, 0.155]])
  assert reward_functions()["knee_height_below"](env, asset).item() == pytest.approx(0.5)


def test_contact_count_counts_geoms_not_contact_points():
  env = SimpleNamespace(scene={"illegal": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0, 2, 5], [0, 0, 0]])))})
  fn = reward_functions()["undesired_contact_count"]
  torch.testing.assert_close(fn(env, "illegal"), torch.tensor([2.0, 0.0]))
  env.scene["illegal"].data.found = None
  with pytest.raises(RuntimeError):
    fn(env, "illegal")


def test_base_height_is_one_sided_and_tilt_is_squared_xy():
  env, _ = scene([[0.155] * 4] * 3)
  data = env.scene["robot"].data
  data.root_link_pos_w = torch.tensor([[0, 0, 0.20], [0, 0, 0.25], [0, 0, 0.5]])
  data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.6, 0.0, -0.8], [0.0, 0.8, -0.6]])
  functions = reward_functions()
  torch.testing.assert_close(functions["base_height_below"](env), torch.tensor([0.05, 0.0, 0.0]))
  torch.testing.assert_close(functions["base_tilt_l2"](env), torch.tensor([0.0, 0.36, 0.64]))


def test_go2_calf_origins_are_knees_and_illegal_geoms_exclude_feet():
  root = ElementTree.parse(ROOT / "src/assets/robots/unitree_go2/xmls/go2.xml").getroot()
  bodies = {b.attrib["name"]: b for b in root.iter("body")}
  geoms = {g.attrib.get("name") for g in root.iter("geom")}
  cfg_tree = ast.parse((TASK / "config/go2/env_cfgs.py").read_text())
  assignment = next(n for n in cfg_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ILLEGAL_GEOMS" for t in n.targets))
  scope = {"FOOT_ORDER": ("FL", "FR", "RL", "RR")}
  illegal = eval(compile(ast.Expression(assignment.value), "illegal_geoms", "eval"), scope)
  assert len(illegal) == len(set(illegal)) == 19
  assert set(illegal) <= geoms
  for leg in scope["FOOT_ORDER"]:
    calf = bodies[f"{leg}_calf"]
    assert calf.attrib["pos"] == "0 0 -0.213"
    assert calf.find("joint").attrib["name"] == f"{leg}_calf_joint"
    assert f"{leg}_foot_collision" not in illegal
  tree = ast.parse((TASK / "contact_env_cfg.py").read_text())
  reward_dict = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "rewards" for t in n.targets))
  terms = {ast.literal_eval(k): v for k, v in zip(reward_dict.keys, reward_dict.values, strict=True)}
  for name, expected in {"illegal_contact": -4.0, "knee_height": -5.0, "base_height_below": -10.0, "base_tilt": -1.0}.items():
    weight = next(k.value for k in terms[name].keywords if k.arg == "weight")
    assert ast.literal_eval(weight) == expected
