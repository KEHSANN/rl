"""User command -> planner params; OOD warnings; no silent clamping."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from conftest import load_mdp  # noqa: E402

P = load_mdp("planning")
C = load_mdp("command_override")


def planner(n=8):
  cfg = SimpleNamespace(stride_length_range=(0.0, 0.3), stance_width_range=(0.1, 0.3), leg_offset_range=(-0.15, 0.15),
                        heading_range=(-math.pi, math.pi), yaw_rate_range=(-math.pi, math.pi), rel_yaw_rate_envs=0.5,
                        gaits=None, resampling_time_range=(0.34, 0.36))
  g = torch.Generator().manual_seed(0)
  pl = P.GaitPlanner(cfg, n, "cpu", generator=g)
  ids = torch.arange(n)
  pl.reset(ids, torch.zeros(n, 2), torch.zeros(n))
  return pl, ids


def test_in_distribution_command_is_applied_without_warnings():
  pl, ids = planner()
  msgs = C.apply_user_command(pl, ids, C.UserCommand(gait="trot", speed=0.3, heading_offset=0.5))
  assert msgs == []
  assert torch.all(pl.gait == P.GAITS.index("trot"))
  assert torch.allclose(pl.stride, torch.full((8, 2), 0.3 * 2 * 0.35))
  assert torch.allclose(pl.heading_off, torch.full((8,), 0.5))
  assert abs(C.stride_to_speed(float(pl.stride[0, 0]), "trot", pl.switch_dt) - 0.3) < 1e-6


def test_out_of_distribution_speed_warns_and_is_not_clamped():
  pl, ids = planner()
  msgs = C.apply_user_command(pl, ids, C.UserCommand(gait="trot", speed=1.0))
  assert any("outside the training range" in m for m in msgs)
  assert torch.allclose(pl.stride, torch.full((8, 2), 0.7))  # applied as requested
  msgs = C.apply_user_command(pl, ids, C.UserCommand(gait="trot", speed=1.0), clamp_stride=True)
  assert torch.allclose(pl.stride, torch.full((8, 2), C.TRAIN_STRIDE_MAX))
  assert any("clamped" in m for m in msgs)


def test_heading_plus_yaw_and_large_yaw_warn():
  pl, ids = planner()
  msgs = C.apply_user_command(pl, ids, C.UserCommand(heading_offset=1.0, yaw_rate=4.0))
  assert len(msgs) == 2
  assert torch.all(pl.use_yaw_rate) and torch.allclose(pl.yaw_rate, torch.full((8,), 4.0))


def test_subset_of_envs_and_restore():
  pl, ids = planner()
  snap = C.snapshot_params(pl)
  C.apply_user_command(pl, torch.tensor([2]), C.UserCommand(gait="bound", stride=0.1))
  assert int(pl.gait[2]) == P.GAITS.index("bound")
  assert torch.equal(pl.gait[:2], snap["gait"][:2])
  C.restore_params(pl, snap)
  for k, v in snap.items():
    assert torch.equal(getattr(pl, k), v)


def test_invalid_commands_raise():
  pl, ids = planner()
  with pytest.raises(ValueError):
    C.apply_user_command(pl, ids, C.UserCommand(gait="gallop"))
  with pytest.raises(ValueError):
    C.apply_user_command(pl, ids, C.UserCommand(speed=0.1, stride=0.1))


def test_viewer_mapping_and_default_untouched():
  cmd = C.build_user_command(C.KEEP_TRAINED_GAIT, 90.0, 0.2, 0.0)
  assert cmd.gait is None and abs(cmd.heading_offset - math.pi / 2) < 1e-9
  pl, ids = planner()
  before = C.snapshot_params(pl)
  pl.advance(ids)  # default path never touches the params
  for k, v in before.items():
    assert torch.equal(getattr(pl, k), v)


def test_command_reaches_the_goals_after_reanchor():
  """User command -> planner -> contact goals (what the policy observes)."""
  pl, ids = planner(4)
  C.apply_user_command(pl, ids, C.UserCommand(gait="trot", stride=0.3, heading_offset=0.0, yaw_rate=0.0))
  pl.reset(ids, torch.zeros(4, 2), torch.zeros(4))
  x0 = pl.foot_xy[..., 0].mean(dim=1).clone()
  for _ in range(8):
    pl.advance(ids)
  moved = pl.foot_xy[..., 0].mean(dim=1) - x0
  assert torch.all(moved > 0.9)  # 8 switches * 0.3/2 per switch = 1.2 m forward
