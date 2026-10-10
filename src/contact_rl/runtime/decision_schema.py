"""Validation of high-level decisions before they reach the GaitPlanner.

Pure python on purpose: no torch / mjlab import, so it is unit-testable anywhere.
Gait names and limits mirror tasks/contact/mdp/planning.py and command_override.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

GAITS = ("trot", "pace", "bound", "jump", "crawl")  # == planning.GAITS
TRAIN_STRIDE_MAX = 0.3  # == command_override.TRAIN_STRIDE_MAX
TRAIN_YAW_RATE_MAX = math.pi
SWITCH_DT = 0.35  # nominal command duration S (paper Sec. 4)
MAX_SPEED_MPS = {g: TRAIN_STRIDE_MAX / (p * SWITCH_DT) for g, p in
                 (("trot", 2), ("pace", 2), ("bound", 2), ("jump", 2), ("crawl", 4))}
SPEED_LIMIT_MPS = 0.43  # hard cap for any gait: max trained speed of the P=2 gaits


class DecisionError(ValueError):
  """Raised for any model output that must not reach the planner."""


@dataclass(frozen=True)
class Limits:
  speed_max: float = SPEED_LIMIT_MPS
  yaw_rate_max: float = TRAIN_YAW_RATE_MAX
  max_speed_step: float = 0.1  # m/s change allowed per decision (no jumps)
  max_yaw_step: float = 0.5  # rad/s change allowed per decision


@dataclass
class Decision:
  command_id: int
  gait: str | None = None  # None -> keep current gait
  speed: float | None = None  # m/s
  heading_offset: float | None = None  # rad, 0 = forward
  yaw_rate: float | None = None  # rad/s
  explanation: str = ""
  meta: dict = field(default_factory=dict)


def _num(name: str, v, lo: float, hi: float) -> float:
  if isinstance(v, bool) or not isinstance(v, (int, float)):
    raise DecisionError(f"{name} must be a number, got {type(v).__name__}")
  f = float(v)
  if not math.isfinite(f):
    raise DecisionError(f"{name} is not finite")
  if not lo <= f <= hi:
    raise DecisionError(f"{name}={f} outside [{lo}, {hi}]")
  return f


def validate(raw: dict, command_id: int, current: Decision | None = None, limits: Limits = Limits()) -> Decision:
  """Turn a raw model answer into a Decision, or raise DecisionError.

  ``current`` (last applied decision) is used to bound per-step changes so the
  model cannot jump from 0 to the max speed in one decision.
  """
  if not isinstance(raw, dict):
    raise DecisionError("decision must be an object")
  gait = raw.get("gait")
  if gait is not None and gait not in GAITS:
    raise DecisionError(f"unknown gait {gait!r}; choose from {GAITS}")
  speed = None
  if raw.get("speed") is not None:
    speed = _num("speed", raw["speed"], 0.0, limits.speed_max)
    if gait is not None and speed > MAX_SPEED_MPS[gait] + 1e-9:
      speed = MAX_SPEED_MPS[gait]  # within the gait's trained range; do not extrapolate
  heading = None
  if raw.get("heading_offset") is not None:
    heading = _num("heading_offset", raw["heading_offset"], -math.pi, math.pi)
  yaw = None
  if raw.get("yaw_rate") is not None:
    yaw = _num("yaw_rate", raw["yaw_rate"], -limits.yaw_rate_max, limits.yaw_rate_max)
  if heading and yaw:
    raise DecisionError("heading_offset and yaw_rate are exclusive (training samples one or the other)")
  if current is not None:
    if speed is not None and current.speed is not None:
      speed = min(max(speed, current.speed - limits.max_speed_step), current.speed + limits.max_speed_step)
    if yaw is not None and current.yaw_rate is not None:
      yaw = min(max(yaw, current.yaw_rate - limits.max_yaw_step), current.yaw_rate + limits.max_yaw_step)
  expl = str(raw.get("explanation", ""))[:200]
  return Decision(command_id=command_id, gait=gait, speed=speed, heading_offset=heading, yaw_rate=yaw,
                  explanation=expl)


def to_user_command_kwargs(d: Decision) -> dict:
  """Keyword args for command_override.UserCommand (only fields that are set)."""
  out = {}
  if d.gait is not None:
    out["gait"] = d.gait
  if d.speed is not None:
    out["speed"] = d.speed
  if d.heading_offset is not None:
    out["heading_offset"] = d.heading_offset
  if d.yaw_rate is not None:
    out["yaw_rate"] = d.yaw_rate
  return out
