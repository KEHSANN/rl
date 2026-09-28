"""User command overrides for the prespecified gait planner (pure torch).

Steers a trained policy through the *training* pathway: only the planner's
per-env command parameters change; the policy still only sees the contact
goals (p1, p2, I1, I2, s) produced by :class:`GaitPlanner`. There is no second
command system: ``ContactGoalCommand.apply_user_command`` is the only entry
point and it is never called during training, so default behaviour is
unchanged.

Control path::

  user (viewer / script) -> UserCommand -> apply_user_command -> GaitPlanner
  params -> ContactGoalCommand goals -> actor observation -> policy

Training distribution (paper Sec. 4): stride U(0, 0.3) m per pair -> speed
stride/(P*S); heading offset U[-pi, pi] OR yaw rate U[-pi, pi] rad/s, never
both. Out-of-distribution requests are **applied as given and reported**
(returned warnings + ``logging`` warning); nothing is clamped unless the
caller explicitly passes ``clamp_stride=True``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from contact_rl.tasks.contact.mdp.planning import GAIT_PATTERNS, GAITS, GaitPlanner

LOG = logging.getLogger("contact_rl.command")

TRAIN_STRIDE_MAX = 0.3
TRAIN_YAW_RATE_MAX = math.pi
NOMINAL_STANCE_WIDTH = 0.2
_PARAM_NAMES = ("gait", "stride", "stance", "leg_off", "heading_off", "yaw_rate", "use_yaw_rate")


def gait_period(gait: str) -> int:
  return len(GAIT_PATTERNS[gait])


def speed_to_stride(speed_mps: float, gait: str, switch_dt: float) -> float:
  return float(speed_mps) * gait_period(gait) * float(switch_dt)


def stride_to_speed(stride_m: float, gait: str, switch_dt: float) -> float:
  return float(stride_m) / (gait_period(gait) * float(switch_dt))


def max_train_speed(gait: str, switch_dt: float) -> float:
  return stride_to_speed(TRAIN_STRIDE_MAX, gait, switch_dt)


def snapshot_params(planner: GaitPlanner) -> dict[str, torch.Tensor]:
  return {k: getattr(planner, k).clone() for k in _PARAM_NAMES}


def restore_params(planner: GaitPlanner, snap: dict[str, torch.Tensor], ids: torch.Tensor | None = None) -> None:
  for k, v in snap.items():
    if ids is None:
      getattr(planner, k).copy_(v)
    else:
      getattr(planner, k)[ids] = v[ids]


@dataclass
class UserCommand:
  gait: str | None = None
  heading_offset: float | None = None  # rad, 0 fwd, +pi/2 left, pi back
  stride: float | None = None  # m (front = hind)
  speed: float | None = None  # m/s (alternative to stride)
  yaw_rate: float | None = None  # rad/s
  nominal_layout: bool | None = None

  def static_warnings(self) -> list[str]:
    """OOD checks that do not need the planner state."""
    msgs = []
    if (self.heading_offset or 0.0) != 0.0 and (self.yaw_rate or 0.0) != 0.0:
      msgs.append("heading offset + yaw rate combined (training samples one or the other)")
    if self.stride is not None and not 0.0 <= self.stride <= TRAIN_STRIDE_MAX:
      msgs.append(f"stride {self.stride:.2f} m outside the training range [0, {TRAIN_STRIDE_MAX}]")
    if self.speed is not None and self.speed < 0.0:
      msgs.append("negative speed (use heading_offset=pi to walk backwards)")
    if self.yaw_rate is not None and abs(self.yaw_rate) > TRAIN_YAW_RATE_MAX:
      msgs.append(f"|yaw rate| {abs(self.yaw_rate):.2f} > pi rad/s (outside the training range)")
    return msgs

  # Backwards-compatible name used by the first revision.
  def is_out_of_distribution(self) -> list[str]:
    return self.static_warnings()


KEEP_TRAINED_GAIT = "(as trained)"


def build_user_command(gait: str, heading_deg: float, speed: float, yaw_rate: float) -> UserCommand:
  """Viewer widget values -> UserCommand (direction in degrees)."""
  return UserCommand(
    gait=None if gait == KEEP_TRAINED_GAIT else gait,
    heading_offset=math.radians(float(heading_deg)),
    speed=float(speed),
    yaw_rate=float(yaw_rate),
  )


def apply_user_command(
  planner: GaitPlanner, ids: torch.Tensor, cmd: UserCommand, clamp_stride: bool = False
) -> list[str]:
  """Write ``cmd`` into the planner params of ``ids`` (takes effect at the
  next contact switch; call ``GaitPlanner.reset`` to re-anchor immediately).

  Returns human-readable out-of-distribution warnings (also logged). Values
  are applied as requested; ``clamp_stride=True`` clamps stride to the
  training range instead (and says so)."""
  if len(ids) == 0:
    return []
  msgs = cmd.static_warnings()
  if cmd.gait is not None:
    if cmd.gait not in GAITS:
      raise ValueError(f"Unknown gait '{cmd.gait}'; choose from {GAITS}.")
    planner.gait[ids] = GAITS.index(cmd.gait)
  if cmd.speed is not None and cmd.stride is not None:
    raise ValueError("Give either stride or speed, not both.")
  new_stride: torch.Tensor | None = None
  if cmd.speed is not None:
    periods = planner.period(ids)  # uses the (possibly just changed) gait
    new_stride = periods * float(cmd.speed) * planner.switch_dt
  elif cmd.stride is not None:
    new_stride = torch.full((len(ids),), float(cmd.stride), device=planner.device)
  if new_stride is not None:
    lo, hi = float(new_stride.min()), float(new_stride.max())
    if cmd.speed is not None and (lo < 0.0 or hi > TRAIN_STRIDE_MAX):
      msgs.append(
        f"speed {cmd.speed:.2f} m/s needs stride {lo:.2f}..{hi:.2f} m, outside the "
        f"training range [0, {TRAIN_STRIDE_MAX}] (max trained speed for this gait: "
        f"{TRAIN_STRIDE_MAX / (float(planner.period(ids).max()) * planner.switch_dt):.2f} m/s)"
      )
    if clamp_stride:
      clamped = new_stride.clamp(0.0, TRAIN_STRIDE_MAX)
      if not torch.equal(clamped, new_stride):
        msgs.append("stride clamped to the training range (clamp_stride=True)")
      new_stride = clamped
    planner.stride[ids] = new_stride.unsqueeze(-1).expand(-1, 2).clone()
  if cmd.heading_offset is not None:
    planner.heading_off[ids] = math.remainder(float(cmd.heading_offset), 2.0 * math.pi)
  if cmd.yaw_rate is not None:
    planner.yaw_rate[ids] = float(cmd.yaw_rate)
    planner.use_yaw_rate[ids] = float(cmd.yaw_rate) != 0.0
  if cmd.nominal_layout:
    planner.leg_off[ids] = 0.0
    planner.stance[ids] = NOMINAL_STANCE_WIDTH
  for m in msgs:
    LOG.warning("out-of-distribution command: %s", m)
  return msgs


# ------------------------------------------------ ContactGoalCommand wiring


def _all_ids(term) -> torch.Tensor:
  return torch.arange(term.num_envs, device=term.planner.device)


def reanchor(term, ids: torch.Tensor) -> None:
  """Re-anchor the plan of ``ids`` at the current base pose so a new command
  takes effect immediately (mid-episode, ``root_link_pos_w`` is valid)."""
  data = term.robot.data
  term.planner.reset(ids, data.root_link_pos_w[ids, :2], data.heading_w[ids])
  term._sync_goals(ids)


def apply_to_command_term(
  term, cmd: UserCommand, env_ids: torch.Tensor | None = None, reanchor_now: bool = True,
  clamp_stride: bool = False,
) -> list[str]:
  """Apply ``cmd`` to a live ``ContactGoalCommand`` (the only supported entry
  point for interactive control). The sampled training parameters are
  snapshotted on first use so :func:`restore_sampled_commands` can undo it."""
  ids = _all_ids(term) if env_ids is None else env_ids
  if getattr(term, "_user_cmd_snapshot", None) is None:
    term._user_cmd_snapshot = snapshot_params(term.planner)
  msgs = apply_user_command(term.planner, ids, cmd, clamp_stride=clamp_stride)
  if reanchor_now:
    reanchor(term, ids)
  return msgs


def restore_sampled_commands(term, env_ids: torch.Tensor | None = None, reanchor_now: bool = True) -> bool:
  snap = getattr(term, "_user_cmd_snapshot", None)
  if snap is None:
    return False
  ids = _all_ids(term) if env_ids is None else env_ids
  restore_params(term.planner, snap, ids)
  if reanchor_now:
    reanchor(term, ids)
  return True


def describe_command(planner: GaitPlanner, i: int) -> dict:
  """Human-readable command of env ``i`` (for status displays / HUDs)."""
  g = GAITS[int(planner.gait[i])]
  stride = float(planner.stride[i].mean())
  return {
    "gait": g,
    "stride_m": stride,
    "speed_mps": stride_to_speed(stride, g, planner.switch_dt),
    "heading_offset_deg": math.degrees(float(planner.heading_off[i])),
    "yaw_rate_rps": float(planner.yaw_rate[i]) if bool(planner.use_yaw_rate[i]) else 0.0,
  }
