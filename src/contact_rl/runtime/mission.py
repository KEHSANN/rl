"""Persistent mission state, measured progress and deterministic safety (pure python).

The closed-loop mode separates two things the one-shot mode conflated:

* the **mission** (:class:`~contact_rl.runtime.decision_schema.MissionSpec`):
  what the user asked for, interpreted ONCE by the model from the instruction
  and never re-interpreted (so "go faster" cannot compound per tick);
* the **current action**: the planner command (gait / speed / direction /
  yaw rate) that the model re-decides at a low rate from fresh feedback.

:class:`MissionTracker` keeps the phase, measured progress and the outcome of
the previous action, and builds the model's decision context from real
measurements only. :class:`SafetySupervisor` runs on the sim thread
independently of the model; its actions take precedence over every model
decision. Completion is gated on measured state and explicit tolerances,
never on elapsed time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from contact_rl.runtime.decision_schema import (
  ClosedLoopConfig,
  Decision,
  MissionSpec,
  RobotFeedback,
  wrap_angle,
)

# Mission lifecycle states (``MissionTracker.state``).
ACTIVE, PAUSED, COMPLETE, FAILED, STOPPED = "active", "paused", "complete", "failed", "stopped"
# Phases while active.
APPROACH, FINAL_TURN, HOLD = "approach", "final_turn", "hold"


@dataclass
class SafetyAction:
  """Deterministic override. ``stop`` -> speed 0 / yaw 0; otherwise ``speed`` cap."""

  reason: str
  stop: bool = False
  speed: float | None = None
  fail_mission: bool = False


@dataclass
class MissionTracker:
  spec: MissionSpec
  start: RobotFeedback
  cfg: ClosedLoopConfig
  state: str = ACTIVE
  phase: str = APPROACH
  outcome: str = ""
  decisions: int = 0
  kept: int = 0
  changed: int = 0
  last_action: str = "mission start"
  last_metric: float | None = None  # distance to target (reach) or |speed error| (walk) when last applied
  last_metric_delta: float | None = None
  last_decision_sim_time: float = 0.0
  turn_start_yaw: float | None = None
  progress_ref: tuple[float, float] | None = None  # (sim_time, metric) at the start of the progress window
  history: list[str] = field(default_factory=list)

  # ------------------------------------------------------------ measurement
  def target_error(self, fb: RobotFeedback) -> tuple[float, float] | None:
    """(distance m, bearing rad in the body frame, + = left) or None without a target."""
    t = self.spec.target_xy
    if t is None:
      return None
    dx, dy = t[0] - fb.x, t[1] - fb.y
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - fb.yaw)

  def metric(self, fb: RobotFeedback) -> float:
    te = self.target_error(fb)
    if te is not None and self.phase == APPROACH:
      return te[0]
    if self.phase == FINAL_TURN and self.turn_start_yaw is not None:
      return self.cfg.final_turn_rad - abs(wrap_angle(fb.yaw - self.turn_start_yaw))
    along = fb.along(fb.command.heading_offset)
    return abs(along - fb.command.speed) if fb.command.speed > 0 else fb.speed

  def turned(self, fb: RobotFeedback) -> float:
    return 0.0 if self.turn_start_yaw is None else abs(wrap_angle(fb.yaw - self.turn_start_yaw))

  def evidence_complete(self, fb: RobotFeedback) -> bool:
    """Measured completion (the only way a mission can be reported complete)."""
    if self.spec.objective == "reach_target":
      te = self.target_error(fb)
      if te is None:
        return False
      if self.spec.after_arrival == "stop":
        return te[0] <= self.cfg.arrive_tol_m
      return self.phase == FINAL_TURN and self.turned(fb) >= self.cfg.final_turn_rad - self.cfg.final_turn_tol_rad
    if self.spec.objective == "stop":
      return fb.command.speed == 0.0 and fb.command.yaw_rate == 0.0
    return False  # "walk" missions have no measurable end: they run until stopped

  # ------------------------------------------------------------ transitions
  def phase_update(self, fb: RobotFeedback) -> str | None:
    """Advance the phase from measurements. Returns an event string or None."""
    if self.state != ACTIVE:
      return None
    te = self.target_error(fb)
    if self.spec.objective == "reach_target" and self.phase == APPROACH and te is not None \
        and te[0] <= self.cfg.arrive_tol_m:
      if self.spec.after_arrival == "stop":
        self.finish(COMPLETE, f"target reached: {te[0]:.2f} m <= {self.cfg.arrive_tol_m:.2f} m (measured)")
        return "complete"
      self.phase = FINAL_TURN
      self.turn_start_yaw = fb.yaw
      self.progress_ref = (fb.sim_time, self.metric(fb))
      return "arrived"
    if self.evidence_complete(fb):
      what = (f"turned {math.degrees(self.turned(fb)):.0f} deg at the target (measured)"
              if self.phase == FINAL_TURN else "stopped (commanded speed 0, yaw 0)")
      self.finish(COMPLETE, what)
      return "complete"
    return None

  def finish(self, state: str, outcome: str) -> None:
    self.state, self.outcome = state, outcome
    self.history.append(f"{state}: {outcome}")

  def record(self, dec: Decision, fb: RobotFeedback, applied: bool) -> None:
    """Remember the action taken at ``fb`` so the next context can say whether it helped."""
    self.decisions += 1
    m = self.metric(fb)
    self.last_metric_delta = None if self.last_metric is None else m - self.last_metric
    self.last_metric = m
    self.last_decision_sim_time = fb.sim_time
    if dec.is_noop:
      self.kept += 1
      self.last_action = "kept the current action"
    else:
      self.changed += 1
      parts = []
      if dec.stop:
        parts.append("stop")
      if dec.gait is not None:
        parts.append(f"gait {dec.gait}")
      if dec.speed is not None and not dec.stop:
        parts.append(f"speed {dec.speed:.2f} m/s")
      if dec.yaw_rate is not None and not dec.stop:
        parts.append(f"turning {dec.yaw_rate:+.2f} rad/s")
      if dec.heading_offset is not None:
        parts.append(f"direction {math.degrees(dec.heading_offset):.0f} deg")
      self.last_action = ", ".join(parts) + ("" if applied else " (not applied)")
    self.history.append(f"t={fb.sim_time:.1f}s {self.last_action}")
    del self.history[:-20]

  # ---------------------------------------------------------------- context
  def speed_ceiling(self) -> float:
    return 0.0 if self.phase in (FINAL_TURN, HOLD) else self.spec.preferred_speed

  def context(self, fb: RobotFeedback, gaits: tuple[str, ...], safety_notes: tuple[str, ...]) -> dict:
    """The decision state given to ``system_one`` (measured values only, no fabricated signals)."""
    s = self.spec
    cmd = fb.command
    mission = {
      "user instruction": s.text,
      "objective": {"walk": "walk as instructed (no destination)", "reach_target": "reach the target",
                    "stop": "stop"}[s.objective],
      "phase": self.phase,
      "cautious": "yes" if s.cautious else "no",
      "desired speed": f"{s.preferred_speed:.2f} m/s",
      "speed limit now": f"{self.speed_ceiling():.2f} m/s",
      "elapsed": f"{fb.sim_time - self.start.sim_time:.1f} s",
      "decisions so far": self.decisions,
    }
    if s.objective == "reach_target":
      mission["after reaching the target"] = {"stop": "stop", "turn_left": "turn left in place",
                                              "turn_right": "turn right in place"}[s.after_arrival]
    command = {
      "gait": cmd.gait,
      "allowed gaits": ", ".join(gaits) if s.gait_free or s.gait is None else s.gait,
      "speed": f"{cmd.speed:.2f} m/s",
      "walking direction": f"{math.degrees(cmd.heading_offset):.0f} deg (0 = forward, 90 = left)",
      "turning rate": f"{cmd.yaw_rate:+.2f} rad/s (+ = left)",
    }
    measured = {
      "forward velocity": f"{fb.vx:+.2f} m/s",
      "sideways velocity": f"{fb.vy:+.2f} m/s (+ = left)",
      "turning rate": f"{fb.yaw_rate:+.2f} rad/s",
      "body tilt": f"{math.degrees(fb.tilt):.1f} deg",
      "stability": self.stability_label(fb),
    }
    if fb.contact_match is not None:
      measured["feet matching the contact plan"] = f"{100 * fb.contact_match:.0f} %"
    if cmd.speed > 0.02:
      measured["speed tracking"] = f"{100 * fb.along(cmd.heading_offset) / cmd.speed:.0f} % of commanded"
    out = {"mission": mission, "current command": command, "measured state": measured}
    te = self.target_error(fb)
    if te is not None:
      d, b = te
      out["target"] = {
        "distance": f"{d:.2f} m",
        "direction": f"{math.degrees(b):+.0f} deg ({'left' if b > 0 else 'right'} of the body heading)",
        "near": "yes" if d <= self.cfg.slow_radius_m else "no",
      }
    elif s.objective == "reach_target":
      out["target"] = "not configured"
    if self.phase == FINAL_TURN:
      out["final turn"] = {"turned": f"{math.degrees(self.turned(fb)):.0f} deg",
                           "required": f"{math.degrees(self.cfg.final_turn_rad):.0f} deg",
                           "direction": "left" if s.after_arrival == "turn_left" else "right"}
    prev = {"action": self.last_action,
            "seconds since": f"{fb.sim_time - self.last_decision_sim_time:.1f}"}
    if self.last_metric is not None:
      delta = self.metric(fb) - self.last_metric
      if te is not None and self.phase == APPROACH:
        prev["result"] = (f"distance {'decreased' if delta < 0 else 'increased'} by {abs(delta):.2f} m")
      elif self.phase == FINAL_TURN:
        prev["result"] = f"remaining turn {'decreased' if delta < 0 else 'increased'} by {math.degrees(abs(delta)):.0f} deg"
      else:
        prev["result"] = f"speed error {'decreased' if delta < 0 else 'increased'} by {abs(delta):.2f} m/s"
    out["previous decision"] = prev
    out["safety"] = list(safety_notes) or ["no active constraint"]
    return out

  def stability_label(self, fb: RobotFeedback) -> str:
    c = self.cfg.tilt_caution(self.spec.cautious)
    if fb.tilt >= c:
      return "unstable (tilt above the caution limit)"
    if fb.tilt >= 0.6 * c or (fb.contact_match is not None and fb.contact_match < 0.5):
      return "marginal"
    return "stable"


class SafetySupervisor:
  """Deterministic checks on every safety tick; independent of the model."""

  def __init__(self, cfg: ClosedLoopConfig):
    self.cfg = cfg
    self.notes: tuple[str, ...] = ()

  def check(self, m: MissionTracker | None, fb: RobotFeedback, inference_failures: int,
            since_valid_decision_s: float | None) -> SafetyAction | None:
    cfg = self.cfg
    notes: list[str] = []
    if not fb.is_finite:
      self.notes = ("non-finite state",)
      return SafetyAction("non-finite robot state", stop=True, fail_mission=True)
    if fb.tilt >= cfg.tilt_abort_rad:
      self.notes = ("fall risk",)
      return SafetyAction(f"body tilt {math.degrees(fb.tilt):.0f} deg >= abort limit "
                          f"{math.degrees(cfg.tilt_abort_rad):.0f} deg", stop=True, fail_mission=True)
    if m is None or m.state != ACTIVE:
      self.notes = ()
      return None
    act: SafetyAction | None = None
    caution = cfg.tilt_caution(m.spec.cautious)
    if fb.tilt >= caution and fb.command.speed > 0.0:
      notes.append(f"tilt {math.degrees(fb.tilt):.0f} deg above caution {math.degrees(caution):.0f} deg: speed reduced")
      act = SafetyAction(notes[-1], speed=round(fb.command.speed * cfg.caution_speed_factor, 4))
    if inference_failures >= cfg.max_consecutive_failures:
      self.notes = (*notes, "decision model failing")
      return SafetyAction(f"{inference_failures} consecutive failed/late model decisions: controlled stop",
                          stop=True, fail_mission=True)
    if since_valid_decision_s is not None and since_valid_decision_s > cfg.fallback_hold_s:
      self.notes = (*notes, "no valid decision")
      return SafetyAction(f"no valid model decision for {since_valid_decision_s:.1f} s "
                          f"(> {cfg.fallback_hold_s:.1f} s): controlled stop", stop=True, fail_mission=True)
    # Lack of progress (measured): the best value of the mission metric must improve by a
    # minimum amount at least once per window. Using the best value (not a fixed window
    # from the start) tolerates an initial turn toward the target that first increases
    # the distance, but still stops a robot that circles or stalls.
    metric = m.metric(fb)
    if m.progress_ref is None:
      m.progress_ref = (fb.sim_time, metric)
    t_imp, best = m.progress_ref
    need = cfg.min_progress_m if m.phase == APPROACH else math.radians(10.0)
    if m.spec.objective == "reach_target" or m.phase == FINAL_TURN:
      if metric <= best - need:
        m.progress_ref = (fb.sim_time, metric)
      elif fb.sim_time - t_imp >= cfg.no_progress_s:
        unit = "m" if m.phase == APPROACH else "rad"
        self.notes = (*notes, "no progress")
        return SafetyAction(f"no progress: best {'distance' if m.phase == APPROACH else 'remaining turn'} "
                            f"{best:.2f} {unit} not improved by {need:.2f} {unit} in {fb.sim_time - t_imp:.0f} s",
                            stop=True, fail_mission=True)
    else:
      along = fb.along(fb.command.heading_offset)
      tracking = fb.command.speed <= 0.05 or along >= cfg.min_tracking_ratio * fb.command.speed
      if tracking:
        m.progress_ref = (fb.sim_time, metric)
      elif fb.sim_time - t_imp >= cfg.no_progress_s:
        self.notes = (*notes, "no progress")
        return SafetyAction(f"no progress: measured {along:.2f} m/s along the walking direction vs commanded "
                            f"{fb.command.speed:.2f} m/s for {fb.sim_time - t_imp:.0f} s", stop=True,
                            fail_mission=True)
    self.notes = tuple(notes)
    return act
