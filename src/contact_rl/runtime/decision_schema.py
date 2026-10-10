"""Typed decision contract between the Liquid AI decider and the GaitPlanner.

Pure python on purpose (no torch / mjlab), so it is unit-testable anywhere.
The constants mirror ``tasks/contact/mdp/planning.py`` and
``command_override.py``; ``tests/test_liquid_runtime_integration.py`` asserts
they stay in sync.

Only parameters the existing planner supports are expressible: gait, speed
(converted to stride by ``apply_user_command``), heading offset and yaw rate,
plus an explicit stop. There is no field for joint targets, torques or policy
actions, and unknown keys are rejected.

Default limits are PROVISIONAL (see docs/liquid_runtime.md): the Fig. 6
evaluation (eval_fig6.csv) sweeps the command duration, not the speed, so the
only speed evidence is the training range stride U(0, 0.3) m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GAITS = ("trot", "pace", "bound", "jump", "crawl")  # == planning.GAITS
GAIT_PERIODS = {"trot": 2, "pace": 2, "bound": 2, "jump": 2, "crawl": 4}  # == len(GAIT_PATTERNS[g])
TRAIN_STRIDE_MAX = 0.3  # == command_override.TRAIN_STRIDE_MAX
TRAIN_YAW_RATE_MAX = math.pi  # == command_override.TRAIN_YAW_RATE_MAX
SWITCH_DT = 0.35  # nominal command duration S; the live value comes from planner.switch_dt

ALLOWED_KEYS = frozenset({"gait", "speed", "heading_offset", "yaw_rate", "stop", "reason"})


def train_speed_max(gait: str, switch_dt: float = SWITCH_DT) -> float:
  """Largest speed whose stride is inside the training range for ``gait``."""
  return TRAIN_STRIDE_MAX / (GAIT_PERIODS[gait] * switch_dt)


MAX_SPEED_MPS = {g: train_speed_max(g) for g in GAITS}
SPEED_LIMIT_MPS = max(MAX_SPEED_MPS.values())  # ~0.43 m/s (P=2 gaits), absolute hard bound


class DecisionError(ValueError):
  """Raised for any model output that must not reach the planner."""


@dataclass(frozen=True)
class Limits:
  """Runtime safety caps. Never wider than the training distribution."""

  speed_fraction: float = 0.8  # PROVISIONAL: fraction of the per-gait trained max speed
  yaw_rate_max: float = 1.0  # PROVISIONAL rad/s (training range is +-pi)
  max_speed_step: float = 0.1  # m/s change allowed per decision
  max_yaw_step: float = 0.5  # rad/s change allowed per decision
  gaits: tuple[str, ...] = GAITS  # gaits the decider may select
  switch_dt: float = SWITCH_DT

  def __post_init__(self) -> None:
    if not 0.0 < self.speed_fraction <= 1.0:
      raise ValueError(f"speed_fraction must be in (0, 1] (never above the trained range), got {self.speed_fraction}")
    if not 0.0 <= self.yaw_rate_max <= TRAIN_YAW_RATE_MAX:
      raise ValueError(f"yaw_rate_max must be in [0, pi] (training range), got {self.yaw_rate_max}")
    if self.max_speed_step <= 0.0 or self.max_yaw_step <= 0.0:
      raise ValueError("per-decision step limits must be > 0")
    if not self.gaits or set(self.gaits) - set(GAITS):
      raise ValueError(f"gaits must be a non-empty subset of {GAITS}, got {self.gaits}")
    if self.switch_dt <= 0.0:
      raise ValueError("switch_dt must be > 0")

  def speed_cap(self, gaits) -> float:
    """Speed cap for a set of gaits (the most restrictive one wins)."""
    gs = tuple(gaits) or GAITS
    return self.speed_fraction * min(train_speed_max(g, self.switch_dt) for g in gs)


@dataclass(frozen=True)
class PlannerState:
  """Plain-python snapshot of the planner command, taken on the sim thread."""

  gait: str
  speed: float  # m/s
  heading_offset: float  # rad
  yaw_rate: float  # rad/s (0 when the env is not in yaw-rate mode)
  target_gaits: tuple[str, ...]  # gaits currently used by the envs a decision will touch


@dataclass(frozen=True)
class Decision:
  request_id: int
  generation: int
  gait: str | None = None  # None -> keep
  speed: float | None = None  # m/s
  heading_offset: float | None = None  # rad, 0 = forward
  yaw_rate: float | None = None  # rad/s
  stop: bool = False
  reason: str = ""  # diagnostic only, never used for control
  notes: tuple[str, ...] = ()  # deterministic adjustments made by the validator

  @property
  def is_noop(self) -> bool:
    return not self.stop and all(v is None for v in (self.gait, self.speed, self.heading_offset, self.yaw_rate))


def _num(name: str, v, lo: float, hi: float) -> float:
  if isinstance(v, bool) or not isinstance(v, (int, float)):
    raise DecisionError(f"{name} must be a number, got {type(v).__name__}")
  f = float(v)
  if not math.isfinite(f):
    raise DecisionError(f"{name} is not finite")
  if not lo <= f <= hi:
    raise DecisionError(f"{name}={f:.3f} outside [{lo:.3f}, {hi:.3f}]")
  return f


def validate(
  raw, *, request_id: int, generation: int, current: PlannerState | None = None, limits: Limits = Limits()
) -> Decision:
  """Turn a raw decision dict into a :class:`Decision`, or raise :class:`DecisionError`.

  Out-of-range values are rejected (not clamped). The only deterministic
  adjustments are the per-decision rate limits relative to ``current``, the
  speed reduction a gait change needs to stay in its trained range, and
  zeroing the other of heading offset / yaw rate (training never combines
  them). Each adjustment is recorded in ``Decision.notes``.
  """
  if not isinstance(raw, dict):
    raise DecisionError("decision must be an object")
  unknown = set(raw) - ALLOWED_KEYS
  if unknown:
    raise DecisionError(f"unknown decision fields {sorted(unknown)}")
  reason = raw.get("reason", "")
  reason = (reason if isinstance(reason, str) else "")[:200]
  stop = raw.get("stop", False)
  if not isinstance(stop, bool):
    raise DecisionError("stop must be a boolean")
  if stop:
    # Explicit stop: stride 0 (the planner keeps stepping in place, as in
    # training where stride ~ U(0, 0.3)), no turning. Not rate-limited.
    return Decision(request_id, generation, speed=0.0, yaw_rate=0.0, stop=True, reason=reason)

  gait = raw.get("gait")
  if gait is not None:
    if not isinstance(gait, str) or gait not in limits.gaits:
      raise DecisionError(f"unsupported gait {gait!r}; allowed: {limits.gaits}")
  notes: list[str] = []
  cap_gaits = (gait,) if gait is not None else (current.target_gaits if current is not None else GAITS)
  cap = limits.speed_cap(cap_gaits)

  speed = None
  if raw.get("speed") is not None:
    speed = _num("speed", raw["speed"], 0.0, cap)
  heading = None
  if raw.get("heading_offset") is not None:
    heading = _num("heading_offset", raw["heading_offset"], -math.pi, math.pi)
  yaw = None
  if raw.get("yaw_rate") is not None:
    yaw = _num("yaw_rate", raw["yaw_rate"], -limits.yaw_rate_max, limits.yaw_rate_max)
  if heading and yaw:
    raise DecisionError("heading_offset and yaw_rate are exclusive (training samples one or the other)")

  if current is not None:
    if speed is None and gait is not None and current.speed > cap + 1e-9:
      speed = cap
      notes.append(f"speed reduced to {cap:.2f} m/s for gait {gait}")
    if speed is not None:
      lo, hi = current.speed - limits.max_speed_step, current.speed + limits.max_speed_step
      if not lo <= speed <= hi:
        speed = min(max(speed, lo), hi)
        notes.append(f"speed change rate-limited to {speed:.2f} m/s")
      speed = min(max(speed, 0.0), cap)
    if yaw is not None:
      cur_yaw = current.yaw_rate
      lo, hi = cur_yaw - limits.max_yaw_step, cur_yaw + limits.max_yaw_step
      if not lo <= yaw <= hi:
        yaw = min(max(yaw, lo), hi)
        notes.append(f"yaw-rate change rate-limited to {yaw:.2f} rad/s")
    if yaw and heading is None and current.heading_offset != 0.0:
      heading = 0.0
      notes.append("heading offset zeroed (yaw-rate mode)")
    if heading and yaw is None and current.yaw_rate != 0.0:
      yaw = 0.0
      notes.append("yaw rate zeroed (heading-offset mode)")
  return Decision(request_id, generation, gait=gait, speed=speed, heading_offset=heading, yaw_rate=yaw,
                  reason=reason, notes=tuple(notes))


def to_user_command_kwargs(d: Decision) -> dict:
  """Keyword args for ``command_override.UserCommand`` (only fields that are set)."""
  out = {}
  for k in ("gait", "speed", "heading_offset", "yaw_rate"):
    v = getattr(d, k)
    if v is not None:
      out[k] = v
  return out
