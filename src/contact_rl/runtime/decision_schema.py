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
# Closed-loop ticks may additionally report the model's view of the mission.
TICK_KEYS = ALLOWED_KEYS | {"mission_status"}
MISSION_STATUSES = ("continue", "complete", "abort")


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


DEFAULT_LIMITS = Limits()


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
  # Closed-loop provenance (defaults = one-shot instruction decision).
  kind: str = "instruction"  # instruction | tick | safety | manual
  mission_id: int = 0  # mission the decision belongs to (0 = none)
  snapshot_seq: int = -1  # RobotFeedback.seq the decision was computed from (-1 = none)
  mission_status: str = "continue"  # model's view (tick only); completion is gated on measurements

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
  raw, *, request_id: int, generation: int, current: PlannerState | None = None, limits: Limits = DEFAULT_LIMITS,
  kind: str = "instruction", mission_id: int = 0, snapshot_seq: int = -1, speed_ceiling: float | None = None,
) -> Decision:
  """Turn a raw decision dict into a :class:`Decision`, or raise :class:`DecisionError`.

  Out-of-range values are rejected (not clamped). The only deterministic
  adjustments are the per-decision rate limits relative to ``current``, the
  speed reduction a gait change needs to stay in its trained range, and
  zeroing the other of heading offset / yaw rate (training never combines
  them). Each adjustment is recorded in ``Decision.notes``.

  Closed-loop ticks (``kind="tick"``) may also carry ``mission_status`` and
  are bounded by ``speed_ceiling`` (the mission's preferred speed): a tick may
  move the speed anywhere in ``[0, min(cap, speed_ceiling)]``, which is what
  keeps repeated "increase" answers from compounding past the user's intent.
  """
  if not isinstance(raw, dict):
    raise DecisionError("decision must be an object")
  allowed = TICK_KEYS if kind == "tick" else ALLOWED_KEYS
  unknown = set(raw) - allowed
  if unknown:
    raise DecisionError(f"unknown decision fields {sorted(unknown)}")
  reason = raw.get("reason", "")
  reason = (reason if isinstance(reason, str) else "")[:200]
  stop = raw.get("stop", False)
  if not isinstance(stop, bool):
    raise DecisionError("stop must be a boolean")
  status = raw.get("mission_status", "continue")
  if status not in MISSION_STATUSES:
    raise DecisionError(f"mission_status must be one of {MISSION_STATUSES}, got {status!r}")
  prov = dict(kind=kind, mission_id=mission_id, snapshot_seq=snapshot_seq, mission_status=status)
  if stop:
    # Explicit stop: stride 0 (the planner keeps stepping in place, as in
    # training where stride ~ U(0, 0.3)), no turning. Not rate-limited.
    return Decision(request_id, generation, speed=0.0, yaw_rate=0.0, stop=True, reason=reason, **prov)

  gait = raw.get("gait")
  if gait is not None:
    if not isinstance(gait, str) or gait not in limits.gaits:
      raise DecisionError(f"unsupported gait {gait!r}; allowed: {limits.gaits}")
  notes: list[str] = []
  cap_gaits = (gait,) if gait is not None else (current.target_gaits if current is not None else GAITS)
  cap = limits.speed_cap(cap_gaits)
  if speed_ceiling is not None:
    if not (isinstance(speed_ceiling, (int, float)) and math.isfinite(speed_ceiling) and speed_ceiling >= 0.0):
      raise DecisionError(f"invalid speed ceiling {speed_ceiling!r}")
    cap = min(cap, float(speed_ceiling))

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
      # De-escalation (same sign or zero, smaller magnitude) is not rate-limited: moving
      # toward "no turning" must never be slowed down, especially from a sampled training
      # command above the runtime cap. Any other change is rate-limited, then the result
      # is kept inside the cap (rate limiting from an out-of-cap value must not stay out).
      deescalate = abs(yaw) <= abs(cur_yaw) and yaw * cur_yaw >= 0.0
      lo, hi = cur_yaw - limits.max_yaw_step, cur_yaw + limits.max_yaw_step
      if not deescalate and not lo <= yaw <= hi:
        yaw = min(max(yaw, lo), hi)
        notes.append(f"yaw-rate change rate-limited to {yaw:.2f} rad/s")
      if abs(yaw) > limits.yaw_rate_max:
        yaw = math.copysign(limits.yaw_rate_max, yaw)
        notes.append(f"yaw rate held at the cap {yaw:.2f} rad/s")
    if yaw and heading is None and current.heading_offset != 0.0:
      heading = 0.0
      notes.append("heading offset zeroed (yaw-rate mode)")
    if heading and yaw is None and current.yaw_rate != 0.0:
      yaw = 0.0
      notes.append("yaw rate zeroed (heading-offset mode)")
  return Decision(request_id, generation, gait=gait, speed=speed, heading_offset=heading, yaw_rate=yaw,
                  reason=reason, notes=tuple(notes), **prov)


def to_user_command_kwargs(d: Decision) -> dict:
  """Keyword args for ``command_override.UserCommand`` (only fields that are set)."""
  out = {}
  for k in ("gait", "speed", "heading_offset", "yaw_rate"):
    v = getattr(d, k)
    if v is not None:
      out[k] = v
  return out


# ------------------------------------------------------------ closed loop
#
# Everything below is the contract of the closed-loop mission mode
# (``--liquid-mode closed_loop``): a persistent MissionSpec interpreted once
# from the instruction, periodic RobotFeedback snapshots taken on the sim
# thread, and the timing / validity / safety configuration. See
# docs/liquid_runtime.md, "Closed-loop mission mode".

OBJECTIVES = ("walk", "reach_target", "stop")
AFTER_ARRIVAL = ("stop", "turn_left", "turn_right")


@dataclass(frozen=True)
class RobotFeedback:
  """Measured state of the mission env, plain python, taken on the sim thread.

  Position / heading come from the simulator's ground-truth root state
  (``root_link_pos_w`` / ``heading_w``): there is no onboard state estimator in
  this repository. ``foot_contact`` is the contact sensor (``None`` if the env
  has none); ``contact_match`` is the fraction of feet whose measured contact
  equals the planned contact ``I^con_1``.
  """

  seq: int  # strictly increasing per runtime
  sim_time: float  # s since the runtime attached (sim clock, not wall clock)
  wall_time: float  # time.monotonic() at capture (application-latency measurement)
  env_idx: int
  episode_step: int
  x: float  # m, world (simulator ground truth)
  y: float
  yaw: float  # rad, world heading
  vx: float  # m/s, base frame
  vy: float
  yaw_rate: float  # rad/s, measured
  tilt: float  # rad, angle between body z and world z (from projected gravity)
  foot_contact: tuple[bool, ...] | None
  contact_match: float | None
  command: PlannerState

  def numbers(self) -> tuple[float, ...]:
    return (self.sim_time, self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate, self.tilt,
            self.command.speed, self.command.heading_offset, self.command.yaw_rate)

  @property
  def is_finite(self) -> bool:
    return all(math.isfinite(v) for v in self.numbers()) and (
      self.contact_match is None or math.isfinite(self.contact_match))

  @property
  def speed(self) -> float:
    return math.hypot(self.vx, self.vy)

  def along(self, heading_offset: float) -> float:
    """Measured velocity along a walking direction given in the body frame."""
    return self.vx * math.cos(heading_offset) + self.vy * math.sin(heading_offset)


@dataclass(frozen=True)
class MissionSpec:
  """Persistent intent of one instruction (interpreted once, never re-interpreted per tick)."""

  mission_id: int
  text: str
  objective: str  # OBJECTIVES
  preferred_speed: float  # m/s ceiling for every tick of this mission (already inside the caps)
  heading_offset: float | None  # rad, walking direction requested (None = keep)
  yaw_rate: float | None  # rad/s, continuous turning for "walk" missions (None = keep)
  gait: str | None  # requested gait (None = keep / leave it to the closed loop)
  gait_free: bool  # the user asked for "a suitable gait": ticks may pick any allowed gait
  cautious: bool
  after_arrival: str  # AFTER_ARRIVAL
  target_xy: tuple[float, float] | None = None  # world xy (set by the runtime from the configured target)
  reason: str = ""


def validate_mission(raw, *, mission_id: int, limits: Limits, current: PlannerState) -> MissionSpec:
  """Strict validation of an interpreted mission (labels already mapped to numbers)."""
  if not isinstance(raw, dict):
    raise DecisionError("mission must be an object")
  keys = {"text", "objective", "preferred_speed", "heading_offset", "yaw_rate", "gait", "gait_free",
          "cautious", "after_arrival", "reason"}
  unknown = set(raw) - keys
  if unknown:
    raise DecisionError(f"unknown mission fields {sorted(unknown)}")
  obj = raw.get("objective")
  if obj not in OBJECTIVES:
    raise DecisionError(f"objective must be one of {OBJECTIVES}, got {obj!r}")
  gait = raw.get("gait")
  if gait is not None and gait not in limits.gaits:
    raise DecisionError(f"unsupported gait {gait!r}; allowed: {limits.gaits}")
  cap = limits.speed_cap((gait,) if gait else current.target_gaits)
  sp = _num("preferred_speed", raw.get("preferred_speed", 0.0), 0.0, cap + 1e-9)
  h = raw.get("heading_offset")
  h = None if h is None else _num("heading_offset", h, -math.pi, math.pi)
  w = raw.get("yaw_rate")
  w = None if w is None else _num("yaw_rate", w, -limits.yaw_rate_max, limits.yaw_rate_max)
  if h and w:
    raise DecisionError("heading_offset and yaw_rate are exclusive (training samples one or the other)")
  aa = raw.get("after_arrival", "stop")
  if aa not in AFTER_ARRIVAL:
    raise DecisionError(f"after_arrival must be one of {AFTER_ARRIVAL}, got {aa!r}")
  for b in ("gait_free", "cautious"):
    if not isinstance(raw.get(b, False), bool):
      raise DecisionError(f"{b} must be a boolean")
  text = raw.get("text", "")
  reason = raw.get("reason", "")
  return MissionSpec(mission_id, (text if isinstance(text, str) else "")[:300], obj, min(sp, cap), h, w, gait,
                     bool(raw.get("gait_free", False)), bool(raw.get("cautious", False)), aa,
                     reason=(reason if isinstance(reason, str) else "")[:300])


@dataclass(frozen=True)
class ClosedLoopConfig:
  """Timing, validity and safety parameters of the closed-loop mode (all PROVISIONAL defaults).

  ``decision_hz`` is only an upper bound: the runtime never asks more often
  than ``latency_factor`` x the measured median inference latency, and never
  while a request is pending or in flight (single slot, no queue).
  """

  decision_hz: float = 1.0  # max model decisions per second of sim time (RL policy stays at 50 Hz)
  latency_factor: float = 2.0  # decision period >= latency_factor * median inference latency
  deadline_s: float = 2.0  # wall-clock deadline per tick inference; later answers are discarded
  max_snapshot_age_s: float = 1.5  # sim seconds between snapshot and application
  max_pos_drift_m: float = 0.25  # robot moved more than this since the snapshot -> stale
  max_yaw_drift_rad: float = 0.5  # robot turned more than this since the snapshot -> stale
  safety_every_steps: int = 5  # deterministic safety check period in env steps (10 Hz at 50 Hz)
  tilt_caution_rad: float = math.radians(20.0)  # above: deterministic speed reduction
  tilt_caution_cautious_rad: float = math.radians(12.0)  # same, for "cautious" missions
  tilt_abort_rad: float = math.radians(45.0)  # above: controlled stop, mission failed (env falls at 80 deg)
  caution_speed_factor: float = 0.5  # speed multiplier applied by the tilt guard
  no_progress_s: float = 8.0  # window for the lack-of-progress check
  min_progress_m: float = 0.1  # target missions: distance must shrink by this much per window
  min_tracking_ratio: float = 0.2  # walk missions: measured / commanded speed below this = no progress
  arrive_tol_m: float = 0.25  # target reached when the base is within this distance (ground truth)
  slow_radius_m: float = 0.6  # reported to the model as "near the target"
  final_turn_rad: float = math.pi / 2  # "turn left/right at the target" = this in-place turn
  final_turn_tol_rad: float = math.radians(15.0)
  max_consecutive_failures: int = 3  # failed / invalid / late ticks before the controlled-stop fallback
  fallback_hold_s: float = 4.0  # sim seconds without a valid decision before the controlled-stop fallback
  speed_step: float = 0.05  # m/s per "increase" / "decrease" tick answer
  turn_rate: float = 0.5  # rad/s for "left" / "right" tick answers

  def __post_init__(self) -> None:
    if not 0.0 < self.decision_hz <= 10.0:
      raise ValueError(f"decision_hz must be in (0, 10], got {self.decision_hz}")
    if self.deadline_s <= 0.0 or self.max_snapshot_age_s <= 0.0:
      raise ValueError("deadline_s and max_snapshot_age_s must be > 0")
    if self.latency_factor < 1.0:
      raise ValueError("latency_factor must be >= 1")
    if self.safety_every_steps < 1:
      raise ValueError("safety_every_steps must be >= 1")
    if not 0.0 < self.tilt_caution_cautious_rad <= self.tilt_caution_rad < self.tilt_abort_rad < math.radians(80):
      raise ValueError("need 0 < tilt_caution_cautious <= tilt_caution < tilt_abort < 80 deg (fall termination)")
    if not 0.0 < self.caution_speed_factor < 1.0:
      raise ValueError("caution_speed_factor must be in (0, 1)")
    if self.arrive_tol_m <= 0.0 or self.speed_step <= 0.0 or self.turn_rate <= 0.0:
      raise ValueError("arrive_tol_m, speed_step and turn_rate must be > 0")
    if self.max_consecutive_failures < 1 or self.fallback_hold_s <= 0.0:
      raise ValueError("max_consecutive_failures >= 1 and fallback_hold_s > 0 required")

  def tilt_caution(self, cautious: bool) -> float:
    return self.tilt_caution_cautious_rad if cautious else self.tilt_caution_rad

  def decision_period(self, median_latency_s: float | None) -> float:
    p = 1.0 / self.decision_hz
    if median_latency_s is not None and math.isfinite(median_latency_s):
      p = max(p, self.latency_factor * median_latency_s)
    return p


def wrap_angle(a: float) -> float:
  return math.remainder(a, 2.0 * math.pi)
