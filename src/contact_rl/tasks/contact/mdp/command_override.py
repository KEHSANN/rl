"""User command overrides for the prespecified gait planner (pure torch).

Steers a trained policy through the training pathway: only the planner's
per-env command parameters change; the policy still only sees the contact
goals (p1, p2, I1, I2, s). Never called during training.

Training distribution (paper Sec. 4): stride U(0, 0.3) m -> speed
stride/(P*S); heading offset OR yaw rate U[-pi, pi], never both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from contact_rl.tasks.contact.mdp.planning import GAIT_PATTERNS, GAITS, GaitPlanner

TRAIN_STRIDE_MAX = 0.3
NOMINAL_STANCE_WIDTH = 0.2
_PARAM_NAMES = ("gait", "stride", "stance", "leg_off", "heading_off", "yaw_rate", "use_yaw_rate")


def gait_period(gait: str) -> int:
  return len(GAIT_PATTERNS[gait])


def speed_to_stride(speed_mps: float, gait: str, switch_dt: float) -> float:
  return float(speed_mps) * gait_period(gait) * float(switch_dt)


def stride_to_speed(stride_m: float, gait: str, switch_dt: float) -> float:
  return float(stride_m) / (gait_period(gait) * float(switch_dt))


def snapshot_params(planner: GaitPlanner) -> dict[str, torch.Tensor]:
  return {k: getattr(planner, k).clone() for k in _PARAM_NAMES}


def restore_params(planner: GaitPlanner, snap: dict[str, torch.Tensor]) -> None:
  for k, v in snap.items():
    getattr(planner, k).copy_(v)


@dataclass
class UserCommand:
  gait: str | None = None
  heading_offset: float | None = None  # rad, 0 fwd, +pi/2 left, pi back
  stride: float | None = None  # m (front = hind)
  speed: float | None = None  # m/s (alternative to stride)
  yaw_rate: float | None = None  # rad/s
  nominal_layout: bool | None = None

  def is_out_of_distribution(self) -> list[str]:
    msgs = []
    if (self.heading_offset or 0.0) != 0.0 and (self.yaw_rate or 0.0) != 0.0:
      msgs.append("heading offset + yaw rate combined (training samples one or the other)")
    if self.stride is not None and not 0.0 <= self.stride <= TRAIN_STRIDE_MAX:
      msgs.append(f"stride {self.stride:.2f} m outside the training range [0, 0.3]")
    if self.yaw_rate is not None and abs(self.yaw_rate) > math.pi:
      msgs.append("|yaw rate| > pi rad/s (outside the training range)")
    return msgs


def apply_user_command(planner: GaitPlanner, ids: torch.Tensor, cmd: UserCommand, clamp_stride: bool = True) -> None:
  """Write cmd into planner params of ids; takes effect at the next switch.
  Call GaitPlanner.reset afterwards to re-anchor immediately."""
  if len(ids) == 0:
    return
  dev = planner.device
  if cmd.gait is not None:
    if cmd.gait not in GAITS:
      raise ValueError(f"Unknown gait '{cmd.gait}'; choose from {GAITS}.")
    planner.gait[ids] = GAITS.index(cmd.gait)
  stride = cmd.stride
  if cmd.speed is not None:
    if stride is not None:
      raise ValueError("Give either stride or speed, not both.")
    if cmd.gait is None:
      periods = planner.period(ids)
      s = torch.full_like(periods, float(cmd.speed)) * periods * planner.switch_dt
      if clamp_stride:
        s = s.clamp(0.0, TRAIN_STRIDE_MAX)
      planner.stride[ids] = s.unsqueeze(-1).expand(-1, 2).clone()
    else:
      stride = speed_to_stride(cmd.speed, cmd.gait, planner.switch_dt)
  if stride is not None:
    if clamp_stride:
      stride = min(max(stride, 0.0), TRAIN_STRIDE_MAX)
    planner.stride[ids] = torch.full((len(ids), 2), float(stride), device=dev)
  if cmd.heading_offset is not None:
    planner.heading_off[ids] = math.remainder(float(cmd.heading_offset), 2.0 * math.pi)
  if cmd.yaw_rate is not None:
    planner.yaw_rate[ids] = float(cmd.yaw_rate)
    planner.use_yaw_rate[ids] = float(cmd.yaw_rate) != 0.0
  if cmd.nominal_layout:
    planner.leg_off[ids] = 0.0
    planner.stance[ids] = NOMINAL_STANCE_WIDTH
