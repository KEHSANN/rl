"""Unit tests for the simulator-free contact planning / reward math.

Loads ``planning.py`` by file path so the tests need only ``torch`` + ``pytest``
(no mjlab / MuJoCo-Warp / GPU):

    uv run --with pytest pytest tests -q
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch

_PATH = (
  Path(__file__).resolve().parents[1]
  / "src/contact_rl/tasks/contact/mdp/planning.py"
)
_spec = importlib.util.spec_from_file_location("planning", _PATH)
P = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(P)


def _cfg(**kw):
  base = dict(
    stride_length_range=(0.0, 0.3),
    stance_width_range=(0.1, 0.3),
    leg_offset_range=(-0.15, 0.15),
    heading_range=(-math.pi, math.pi),
    yaw_rate_range=(-math.pi, math.pi),
    rel_yaw_rate_envs=0.5,
    gaits=None,
    resampling_time_range=(0.34, 0.36),
  )
  base.update(kw)
  return SimpleNamespace(**base)


def _planner(n=256, seed=0, **kw):
  g = torch.Generator(device="cpu")
  g.manual_seed(seed)
  pl = P.GaitPlanner(_cfg(**kw), n, "cpu", generator=g)
  ids = torch.arange(n)
  base_xy = torch.rand((n, 2), generator=g) * 2.0 - 1.0
  base_yaw = torch.rand((n,), generator=g) * 2 * math.pi - math.pi
  pl.reset(ids, base_xy, base_yaw)
  return pl, ids


def test_every_foot_swings_once_per_period():
  for name, pat in P.GAIT_PATTERNS.items():
    for f in range(P.NUM_FEET):
      assert sum(1 - row[f] for row in pat) == 1, name


def test_lookahead_goal_matches_next_goal():
  """p2 / I2 at switch k must equal p1 / I1 at switch k+1 (straight + curved)."""
  pl, ids = _planner()
  for _ in range(40):
    p2, i2 = pl.next_xy.clone(), pl.next_contact.clone()
    pl.advance(ids)
    assert torch.allclose(pl.cur_contact, i2)
    assert torch.allclose(pl.foot_xy, p2, atol=1e-4)


def test_stance_goal_is_constant_during_stance():
  pl, ids = _planner()
  for _ in range(30):
    before = pl.foot_xy.clone()
    prev = pl.cur_contact.clone()
    pl.advance(ids)
    stay = (prev > 0.5) & (pl.cur_contact > 0.5)
    diff = (pl.foot_xy - before).unbind(-1)
    moved = diff[0] * diff[0] + diff[1] * diff[1]
    assert float((moved * stay).sum()) < 1e-10


def test_front_hind_footholds_do_not_diverge():
  """Regression: independent front/hind strides used to pull the pairs apart."""
  pl, ids = _planner(stride_length_range=(0.0, 0.3), rel_yaw_rate_envs=0.0)
  for _ in range(300):
    pl.advance(ids)
  d = pl.foot_xy[:, 0] - pl.foot_xy[:, 2]  # FL - RL
  sep = torch.norm(d, dim=-1)
  # hip span 0.387 + leg offsets (<=0.3) + asymmetry (<=0.15) + one stride.
  assert float(sep.max()) < 1.2


def test_footholds_stay_near_reference_on_curved_paths():
  """Regression: the stance layout must rotate with the heading."""
  pl, ids = _planner(rel_yaw_rate_envs=1.0, leg_offset_range=(0.0, 0.0))
  for _ in range(200):
    pl.advance(ids)
    rel = pl.foot_xy - pl.ref_xy.unsqueeze(1)
    assert float(torch.norm(rel, dim=-1).max()) < 1.0
  # Left feet stay on the left of the (rotated) reference frame. Checked at a
  # moderate yaw rate: at |w| = pi rad/s the frame turns ~60 deg per switch,
  # so a stance foot planned for mid-stance is legitimately rotated w.r.t. the
  # reference at an arbitrary switch.
  pl, ids = _planner(
    rel_yaw_rate_envs=1.0, leg_offset_range=(0.0, 0.0), yaw_rate_range=(-0.5, 0.5)
  )
  for _ in range(200):
    pl.advance(ids)
  rel_b = P.rot2d((-pl.ref_yaw).unsqueeze(-1), pl.foot_xy - pl.ref_xy.unsqueeze(1))
  lat = rel_b.unbind(-1)[1]
  assert float((lat[:, 0] - lat[:, 1]).min()) > 0.0  # FL left of FR


def test_each_foot_advances_one_mean_stride_per_cycle():
  pl, ids = _planner(rel_yaw_rate_envs=0.0, gaits=("trot",))
  for _ in range(4):
    pl.advance(ids)
  a = pl.foot_xy.clone()
  for _ in range(2):  # trot period = 2 switches
    pl.advance(ids)
  disp = torch.norm(pl.foot_xy - a, dim=-1)  # [n, 4]
  expected = pl.stride.mean(dim=1).unsqueeze(-1)
  assert torch.allclose(disp, expected.expand(-1, 4), atol=1e-4)


def test_phase_masks_partition_swing_and_stance():
  i_con = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
  i_act = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
  reach, hold, detach = P.phase_masks(i_con, i_act, torch.tensor([0.1]), 0.18)
  assert reach.tolist() == [[True, True, False, False]]
  assert hold.tolist() == [[False, False, True, False]]
  assert detach.tolist() == [[False, False, False, False]]
  reach, hold, detach = P.phase_masks(i_con, i_act, torch.tensor([0.3]), 0.18)
  assert reach.tolist() == [[False, False, False, False]]
  assert detach.tolist() == [[True, False, False, False]]
  # Boundary: Eq. 1 uses s <= delta.
  reach, _, _ = P.phase_masks(i_con, i_act, torch.tensor([0.18]), 0.18)
  assert reach.tolist() == [[True, True, False, False]]


def test_proximity_kernel_matches_paper_eq1():
  d = torch.tensor([0.0, 0.1, 0.2])
  k = P.proximity_kernel(d, 0.1, "l2")
  assert torch.allclose(k, torch.exp(-d / 0.1))
  g = P.proximity_kernel(d, 0.01, "gaussian")
  assert torch.allclose(g, torch.exp(-(d * d) / 0.01))


def test_yaw_from_quat_wxyz():
  for yaw in (-3.0, -1.0, 0.0, 0.5, 2.5):
    q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    assert abs(float(P.yaw_from_quat_wxyz(q)[0]) - yaw) < 1e-5


def test_arc_displacement_is_additive():
  th = torch.tensor([0.3, -1.0])
  w = torch.tensor([0.5, 0.0])
  v = torch.tensor([0.1, 0.2])
  one = torch.ones(2)
  a = P.arc_displacement(th, w, v, one)
  b = P.arc_displacement(th + w, w, v, one)
  ab = P.arc_displacement(th, w, v, 2 * one)
  assert torch.allclose(a + b, ab, atol=1e-6)
