"""Glue between :class:`DecisionLoop` and the running ``ContactPlayViewer``.

Every method named ``on_*`` runs on the viewer's simulation thread (it is
called from ``_handle_custom_action``, i.e. from ``BaseViewer._process_actions``).
GUI callbacks, the stdin reader and the decision worker only *enqueue*
viewer actions; they never read or write simulation tensors.

Commands reach the planner exclusively through
``command_override.apply_to_command_term`` (the existing interactive entry
point). The policy, its observations, actions and GRU state are untouched.

Two execution modes (``mode``):

* ``oneshot``: one inference per instruction -> one command (original behaviour).
* ``closed_loop``: the instruction is interpreted once into a persistent
  :class:`MissionSpec`; while the mission is active, :meth:`on_sim_step`
  (called after every env step) takes a :class:`RobotFeedback` snapshot at a
  low, latency-bounded rate and asks the model for the next high-level action
  (``kind="tick"``). Deterministic safety checks run every
  ``safety_every_steps`` env steps and override the model.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Callable

from contact_rl.runtime.decision_loop import DecisionLoop, Result
from contact_rl.runtime.decision_schema import (
  ClosedLoopConfig,
  Decision,
  DecisionError,
  Limits,
  MissionSpec,
  PlannerState,
  RobotFeedback,
  to_user_command_kwargs,
  validate,
  wrap_angle,
)
from contact_rl.runtime.mission import (
  ACTIVE,
  COMPLETE,
  FAILED,
  PAUSED,
  STOPPED,
  MissionTracker,
  SafetyAction,
  SafetySupervisor,
)

LOG = logging.getLogger("contact_rl.runtime")
MAX_TEXT = 300
ACTION_PREFIX = "liquid_"
_UNSET = object()


def planner_state(term, ids, env_idx: int) -> PlannerState:
  """Snapshot (plain floats) of the command of ``env_idx`` + gaits of ``ids``. Sim thread only."""
  import torch

  from contact_rl.tasks.contact.mdp.command_override import describe_command
  from contact_rl.tasks.contact.mdp.planning import GAITS

  pl = term.planner
  d = describe_command(pl, env_idx)
  sel = pl.gait if ids is None else pl.gait[ids]
  gaits = tuple(GAITS[int(g)] for g in torch.unique(sel).tolist())
  return PlannerState(d["gait"], d["speed_mps"], float(pl.heading_off[env_idx]), d["yaw_rate_rps"], gaits)


def robot_feedback(term, env_idx: int, seq: int, sim_time: float, episode_step: int = -1) -> RobotFeedback:
  """Measured state of ``env_idx`` as plain floats (one device->host copy). Sim thread only."""
  import torch

  data = term.robot.data
  i = int(env_idx)
  g = data.projected_gravity_b[i]
  vec = torch.cat([data.root_link_pos_w[i, :2], data.heading_w[i].reshape(1), data.root_link_lin_vel_b[i, :2],
                   data.root_link_ang_vel_b[i, 2:3], g.reshape(-1)]).float().cpu().tolist()
  x, y, yaw, vx, vy, wz, gx, gy, gz = vec
  n = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
  tilt = math.acos(max(-1.0, min(1.0, -gz / n))) if math.isfinite(gz) else float("nan")
  contact = match = None
  try:
    act = term.actual_contact()[i]
    plan = term.planner.cur_contact[i]
    contact = tuple(bool(c > 0.5) for c in act.tolist())
    match = float(((act > 0.5) == (plan > 0.5)).float().mean())
  except Exception:  # noqa: BLE001  no contact sensor -> contact information unavailable (not fabricated)
    pass
  ids = torch.tensor([i], device=term.planner.device)
  return RobotFeedback(seq, sim_time, time.monotonic(), i, episode_step, x, y, yaw, vx, vy, wz, tilt, contact, match,
                       planner_state(term, ids, i))


MODES = ("oneshot", "closed_loop")


class LiquidRuntime:
  def __init__(self, decide_factory: Callable[[Limits], Callable], load: Callable[[], dict] | None,
               limits: Limits | None = None, timeout_s: float = 5.0, *, mode: str = "oneshot",
               closed_loop: ClosedLoopConfig | None = None, autostart: bool = True):
    if mode not in MODES:
      raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    self._decide_factory = decide_factory
    self._load = load
    self.limits = limits or Limits()
    self.timeout_s = timeout_s
    self.mode = mode
    self.cl = closed_loop or ClosedLoopConfig()
    self.autostart = autostart
    self.loop: DecisionLoop | None = None
    self._request_action: Callable[[str, Any], None] | None = None
    self._term = self._ids = self._env_idx = None
    self.last_text = ""
    self.last_decision: Decision | None = None
    self.last_msg = ""
    self.use_stdin = False
    # ---- closed-loop state (sim thread only)
    self.mission: MissionTracker | None = None
    self.target_rel: tuple[float, float] | None = None  # (forward m, left m) relative to the mission start pose
    self.supervisor = SafetySupervisor(self.cl)
    self.sim_time = 0.0
    self.step_dt = 0.02
    self._steps = 0
    self._seq = 0
    self._mission_env = 0
    self._last_tick_sim = -1e9
    self._last_valid_sim: float | None = None
    self._safety_cap: float | None = None
    self.last_feedback: RobotFeedback | None = None
    self.last_outcome = ""  # applied | kept | rejected (...) for the latest decision result
    self.last_choices = ""
    self.app_latencies: deque[float] = deque(maxlen=200)
    self.events: deque[str] = deque(maxlen=200)
    self._tick_snapshots: dict[int, RobotFeedback] = {}  # request_id -> snapshot it was computed from
    self._pending_env: int | None = None

  # ------------------------------------------------------------ lifecycle
  def attach(self, request_action, term_getter, ids_getter, env_idx_getter, step_dt: float | None = None) -> None:
    self._request_action = request_action
    self._term, self._ids, self._env_idx = term_getter, ids_getter, env_idx_getter
    # Caps use the live command duration of this env, not the nominal constant.
    self.limits = replace(self.limits, switch_dt=float(term_getter().planner.switch_dt))
    if step_dt is not None:
      self.step_dt = float(step_dt)
    timeout = min(self.timeout_s, self.cl.deadline_s) if self.mode == "closed_loop" else self.timeout_s
    self.loop = DecisionLoop(self._decide_factory(self.limits), self._deliver, load=self._load,
                             timeout_s=timeout)

  def start(self) -> None:
    assert self.loop is not None, "attach() first"
    self.loop.start()
    if self.use_stdin:
      self.start_stdin_reader()

  def stop(self) -> None:
    if self.loop is not None:
      self.loop.stop()

  def start_stdin_reader(self, stream=None) -> threading.Thread:
    """Read instructions line by line (``stop`` = immediate stop without the model;
    closed loop also: ``pause``, ``resume``, ``target FWD LEFT``, ``target clear``)."""
    stream = stream or sys.stdin

    def _read() -> None:
      for line in stream:
        text = line.strip()
        if not text:
          continue
        low = text.lower()
        if low in ("stop", "/stop"):
          self.send_stop()
        elif self.mode == "closed_loop" and low in ("pause", "/pause"):
          self.send_pause()
        elif self.mode == "closed_loop" and low in ("resume", "/resume", "start", "/start"):
          self.send_resume()
        elif self.mode == "closed_loop" and low.split()[0] in ("target", "/target"):
          parts = low.split()[1:]
          try:
            self.send_target(None) if parts == ["clear"] else self.send_target(float(parts[0]), float(parts[1]))
          except (IndexError, ValueError):
            print("[liquid] usage: target FWD LEFT  |  target clear", flush=True)
        else:
          self.send_text(text)

    t = threading.Thread(target=_read, name="liquid-stdin", daemon=True)
    t.start()
    extra = " 'pause', 'resume', 'target FWD LEFT' control the mission." if self.mode == "closed_loop" else ""
    print("[liquid] type instructions + Enter (e.g. 'walk forward slowly'); 'stop' halts without the model."
          + extra, flush=True)
    return t

  # ------------------------------------------------- any thread: enqueue only
  def send_text(self, text: str) -> None:
    self._request_action("CUSTOM", ("liquid_cmd", str(text)))

  def send_stop(self) -> None:
    self._request_action("CUSTOM", ("liquid_stop", None))

  def send_pause(self) -> None:
    self._request_action("CUSTOM", ("liquid_pause", None))

  def send_resume(self) -> None:
    self._request_action("CUSTOM", ("liquid_resume", None))

  def send_target(self, forward_m: float | None, left_m: float | None = None) -> None:
    payload = None if forward_m is None else (float(forward_m), float(left_m or 0.0))
    self._request_action("CUSTOM", ("liquid_target", payload))

  def _deliver(self, res: Result) -> None:  # worker thread
    self._request_action("CUSTOM", ("liquid_result", res))

  # ------------------------------------------------------- sim thread only
  def handle(self, kind: str, payload) -> str:
    if kind == "liquid_cmd":
      return self.on_command(payload)
    if kind == "liquid_result":
      return self.on_result(payload)
    if kind == "liquid_stop":
      return self.on_stop()
    if kind == "liquid_pause":
      return self.on_pause()
    if kind == "liquid_resume":
      return self.on_resume()
    if kind == "liquid_target":
      return self.on_target(payload)
    return f"unknown liquid action {kind}"

  def on_command(self, text) -> str:
    text = " ".join(str(text).split())[:MAX_TEXT]
    if not text:
      return self._set("empty instruction ignored")
    if self.mode == "closed_loop":
      env = int(self._env_idx())
      st = planner_state(self._term(), self._env_ids(env), env)
      req = self.loop.submit(text, st, kind="mission")  # bumps the generation: in-flight ticks become stale
      if req is None:
        return self._set(f"Liquid AI unavailable ({self.loop.last_error}); instruction not sent")
      if self.mission is not None and self.mission.state in (ACTIVE, PAUSED):
        self.mission.finish(STOPPED, "replaced by a new instruction")
      self._pending_env = env
      self.last_text = text
      return self._set(f"#{req.request_id} interpreting mission: {text!r}")
    st = planner_state(self._term(), self._ids(), self._env_idx())
    req = self.loop.submit(text, st)
    if req is None:
      return self._set(f"Liquid AI unavailable ({self.loop.last_error}); command not sent")
    self.last_text = text
    return self._set(f"#{req.request_id} sent: {text!r}")

  def on_result(self, res: Result) -> str:
    kind = res.request.kind
    if kind == "mission":
      return self._on_mission_result(res)
    if kind == "tick":
      return self._on_tick_result(res)
    if not self.loop.is_current(res):
      self.loop.reject_stale()
      return self._set(f"#{res.request.request_id} discarded (superseded by a newer command)")
    dec = res.decision
    self.loop.mark_applied(res)
    self.last_decision = dec
    if dec.is_noop:
      return self._set(f"#{dec.request_id} no change ({dec.reason})")
    warnings = self._apply(dec)
    note = "; ".join(dec.notes + tuple(warnings))
    return self._set(f"#{dec.request_id} applied in {res.latency_s * 1e3:.0f} ms: {self.describe(dec)}"
                     + (f" [{note}]" if note else ""))

  def on_stop(self) -> str:
    self.loop.supersede()  # drop any in-flight model answer
    dec = Decision(request_id=0, generation=0, speed=0.0, yaw_rate=0.0, stop=True, reason="manual stop",
                   kind="manual")
    if self.mode == "closed_loop":
      self._apply(dec, self._env_ids(self._mission_env))
      if self.mission is not None and self.mission.state in (ACTIVE, PAUSED):
        self.mission.finish(STOPPED, "stopped by the user")
    else:
      self._apply(dec)
    self.last_decision = dec
    return self._set("STOP applied (no model): speed 0, yaw rate 0")

  def _apply(self, dec: Decision, ids=_UNSET) -> list[str]:
    from contact_rl.tasks.contact.mdp.command_override import (
      UserCommand,
      apply_to_command_term,
    )

    cmd = UserCommand(**to_user_command_kwargs(dec))
    # Gait changes need a re-anchored plan; other parameters take effect at the
    # next contact switch (no plan discontinuity).
    ids = self._ids() if ids is _UNSET else ids
    return apply_to_command_term(self._term(), cmd, ids, reanchor_now=dec.gait is not None)

  # ------------------------------------------------- closed loop (sim thread)
  def _env_ids(self, env: int):
    import torch

    return torch.tensor([int(env)], device=self._term().planner.device)

  def _feedback(self, env: int | None = None) -> RobotFeedback:
    self._seq += 1
    term = self._term()
    ep = -1
    try:
      ep = int(term._env.episode_length_buf[self._mission_env if env is None else env])
    except Exception:  # noqa: BLE001
      pass
    fb = robot_feedback(term, self._mission_env if env is None else env, self._seq, self.sim_time, ep)
    self.last_feedback = fb
    return fb

  def on_target(self, payload) -> str:
    if payload is None:
      self.target_rel = None
      return self._set("target cleared")
    fwd, left = (float(v) for v in payload)
    if not (math.isfinite(fwd) and math.isfinite(left)) or math.hypot(fwd, left) > 20.0:
      return self._set("target rejected (non-finite or farther than 20 m)")
    self.target_rel = (fwd, left)
    return self._set(f"target set: {fwd:.2f} m forward, {left:.2f} m left of the robot at mission start")

  def on_pause(self) -> str:
    m = self.mission
    if self.mode != "closed_loop" or m is None or m.state != ACTIVE:
      return self._set("no active mission to pause")
    self.loop.supersede()
    m.state = PAUSED
    self._apply(Decision(0, 0, speed=0.0, yaw_rate=0.0, stop=True, kind="manual"), self._env_ids(self._mission_env))
    return self._set("mission paused (controlled stop; model decisions suspended)")

  def on_resume(self) -> str:
    m = self.mission
    if self.mode != "closed_loop" or m is None or m.state != PAUSED:
      return self._set("no paused mission to resume")
    self.loop.supersede()
    m.state = ACTIVE
    m.progress_ref = None
    self._last_valid_sim = self.sim_time
    self._last_tick_sim = -1e9  # ask the model right away how to continue
    return self._set("mission resumed; the model decides the next action from fresh feedback")

  def on_env_reset(self) -> str | None:
    """Viewer / episode reset: invalidate everything in flight and end the mission."""
    if self.loop is not None:
      self.loop.supersede()
    self._safety_cap = None
    m = self.mission
    if self.mode == "closed_loop" and m is not None and m.state in (ACTIVE, PAUSED):
      m.finish(FAILED, "environment reset during the mission")
      self._apply(Decision(0, 0, speed=0.0, yaw_rate=0.0, stop=True, kind="safety"), self._env_ids(self._mission_env))
      return self._set("mission ended: environment reset (controlled stop applied after reset)")
    return None

  def on_sim_step(self, dones=None) -> None:
    """Called on the sim thread after every env step (the RL policy keeps its own 50 Hz loop)."""
    self.sim_time += self.step_dt
    self._steps += 1
    m = self.mission
    if self.mode != "closed_loop" or m is None or m.state not in (ACTIVE, PAUSED):
      return
    if dones is not None:
      try:
        ended = bool(dones[self._mission_env] > 0)
      except Exception:  # noqa: BLE001
        ended = False
      if ended:
        self.loop.supersede()
        m.finish(FAILED, "episode terminated (fell over or time out) and the env was reset")
        self._safety_cap = None
        self._apply(Decision(0, 0, speed=0.0, yaw_rate=0.0, stop=True, kind="safety"), self._env_ids(self._mission_env))
        self._set("mission failed: episode terminated")
        return
    if m.state != ACTIVE:
      return
    period = self.cl.decision_period(self.loop.median_latency())
    tick_due = self.sim_time - self._last_tick_sim >= period and not self.loop.busy
    if self._steps % self.cl.safety_every_steps != 0 and not tick_due:
      return
    fb = self._feedback()
    # Deadline: an inference running longer than the deadline counts as a failure now
    # (its late answer is discarded by the loop when it eventually arrives).
    age = self.loop.inflight_age()
    if age is not None and age > self.cl.deadline_s:
      self.loop.flag_deadline()
    since = None if self._last_valid_sim is None else self.sim_time - self._last_valid_sim
    act = self.supervisor.check(m, fb, self.loop.consecutive_failures, since)
    if act is not None:
      self._apply_safety(act, fb)
      if m.state != ACTIVE:
        return
    elif self._safety_cap is not None and fb.tilt < self.cl.tilt_caution(m.spec.cautious):
      self._safety_cap = None
      self.events.append(f"t={self.sim_time:.1f}s safety speed cap released")
    ev = m.phase_update(fb)
    if ev == "complete":
      self.loop.supersede()
      self._apply(Decision(0, 0, speed=0.0, yaw_rate=0.0, stop=True, kind="safety"), self._env_ids(self._mission_env))
      self._set(f"mission COMPLETE: {m.outcome}")
      return
    if ev == "arrived":
      # Measured arrival: stop translating (deterministic), then let the model steer the final turn.
      self.loop.supersede()
      self._apply(Decision(0, 0, speed=0.0, kind="safety"), self._env_ids(self._mission_env))
      self._set(f"target reached (measured); final turn phase ({m.spec.after_arrival})")
      self._last_tick_sim = -1e9
      tick_due = not self.loop.busy
    if tick_due:
      self._submit_tick(fb)

  def _submit_tick(self, fb: RobotFeedback) -> None:
    m = self.mission
    gaits = self.limits.gaits if (m.spec.gait_free or m.spec.gait is None) else (m.spec.gait,)
    ceiling = m.speed_ceiling()
    if self._safety_cap is not None:
      ceiling = min(ceiling, self._safety_cap)
    ctx = m.context(fb, gaits, self.supervisor.notes)
    ctx["_ceiling"] = ceiling
    ctx["_allowed_gaits"] = gaits
    req = self.loop.submit_tick(m.spec.text, fb.command, mission_id=m.spec.mission_id, snapshot_seq=fb.seq,
                                context=ctx)
    if req is not None:
      self._last_tick_sim = self.sim_time
      self._tick_snapshots[req.request_id] = fb
      while len(self._tick_snapshots) > 8:
        self._tick_snapshots.pop(next(iter(self._tick_snapshots)))

  def _apply_safety(self, act: SafetyAction, fb: RobotFeedback) -> None:
    m = self.mission
    ids = self._env_ids(self._mission_env)
    if act.stop:
      self.loop.supersede()  # no in-flight model answer may override a safety stop
      self._apply(Decision(0, 0, speed=0.0, yaw_rate=0.0, stop=True, reason=act.reason, kind="safety"), ids)
      if m is not None and act.fail_mission:
        m.finish(FAILED, act.reason)
      self.events.append(f"t={self.sim_time:.1f}s SAFETY STOP: {act.reason}")
      self._set(f"SAFETY: {act.reason}")
      return
    if act.speed is not None and self._safety_cap is None:
      self._safety_cap = act.speed
      if fb.command.speed > act.speed:
        self._apply(Decision(0, 0, speed=act.speed, reason=act.reason, kind="safety"), ids)
      self.events.append(f"t={self.sim_time:.1f}s SAFETY: {act.reason} -> cap {act.speed:.2f} m/s")
      self._set(f"SAFETY: {act.reason}")

  def _on_mission_result(self, res: Result) -> str:
    if not self.loop.is_current(res):
      self.loop.reject_stale()
      self.last_outcome = "rejected (superseded)"
      return self._set(f"#{res.request.request_id} mission discarded (superseded)")
    spec: MissionSpec = res.decision
    self.loop.mark_applied(res)
    env = self._pending_env if self._pending_env is not None else int(self._env_idx())
    self._mission_env = env
    fb = self._feedback(env)
    if not fb.is_finite:
      return self._set("mission refused: robot state is not finite")
    if spec.objective == "reach_target":
      if self.target_rel is None:
        self.last_outcome = "rejected (no target configured)"
        return self._set("mission needs a target, but none is configured (set Target forward/left first); "
                         f"interpreted as: {spec.reason}")
      f, lft = self.target_rel
      c, s_ = math.cos(fb.yaw), math.sin(fb.yaw)
      spec = replace(spec, target_xy=(fb.x + c * f - s_ * lft, fb.y + s_ * f + c * lft))
    self.mission = MissionTracker(spec, fb, self.cl, state=ACTIVE if self.autostart else PAUSED)
    self._safety_cap = None
    self._last_valid_sim = self.sim_time
    self._last_tick_sim = self.sim_time  # first tick one period after the initial action
    self._tick_snapshots = {}
    self.last_choices = spec.reason
    # Initial action from the mission (validated + rate-limited like any decision).
    raw: dict = {"speed": spec.preferred_speed}
    if spec.objective == "stop":
      raw = {"stop": True}
    else:
      if spec.gait is not None and spec.gait != fb.command.gait:
        raw["gait"] = spec.gait
      if spec.yaw_rate:  # continuous turning (heading offset is zeroed by validate)
        raw["yaw_rate"] = spec.yaw_rate
      elif spec.heading_offset is not None:  # walking direction (yaw rate is zeroed by validate)
        raw["heading_offset"] = spec.heading_offset
        if spec.yaw_rate is not None:
          raw["yaw_rate"] = 0.0
      elif spec.yaw_rate is not None:
        raw["yaw_rate"] = 0.0
    if not self.autostart:
      self.last_outcome = "mission ready (paused)"
      return self._set(f"mission #{spec.mission_id} ready ({spec.objective}); press Start/Resume")
    try:
      dec = validate(raw, request_id=res.request.request_id, generation=res.request.generation,
                     current=fb.command, limits=self.limits, kind="instruction", mission_id=spec.mission_id,
                     snapshot_seq=fb.seq, speed_ceiling=spec.preferred_speed)
    except DecisionError as e:
      self.mission.finish(FAILED, f"initial action invalid: {e}")
      return self._set(f"mission refused: {e}")
    warnings = [] if dec.is_noop else self._apply(dec, self._env_ids(env))
    self.mission.record(dec, fb, applied=True)
    if spec.objective == "stop":
      self.mission.finish(COMPLETE, "stop requested; controlled stop applied (the planner keeps stepping in place)")
    self.last_decision = dec
    self.last_outcome = "applied (mission start)"
    tgt = "" if spec.target_xy is None else f", target ({spec.target_xy[0]:.2f}, {spec.target_xy[1]:.2f}) m world"
    note = "; ".join(dec.notes + tuple(warnings))
    return self._set(f"mission #{spec.mission_id} {spec.objective} v<= {spec.preferred_speed:.2f} m/s{tgt}: "
                     f"{self.describe(dec)}" + (f" [{note}]" if note else ""))

  def _on_tick_result(self, res: Result) -> str:
    req, dec = res.request, res.decision
    m = self.mission
    snap = self._tick_snapshots.pop(req.request_id, None)
    if m is None or m.spec.mission_id != req.mission_id or m.state != ACTIVE:
      self.loop.reject_stale("stale_mission")
      self.last_outcome = "rejected (stale mission)"
      return self._set(f"#{req.request_id} tick discarded: mission no longer active")
    if not self.loop.is_current(res):
      self.loop.reject_stale()
      self.last_outcome = "rejected (superseded / out of order)"
      return self._set(f"#{req.request_id} tick discarded (superseded or out of order)")
    fb = self._feedback()
    why = self._snapshot_invalid(snap, fb)
    if why:
      self.loop.reject_stale("stale_state")
      self.last_outcome = f"rejected (stale state: {why})"
      self._last_tick_sim = -1e9  # re-ask from fresh feedback
      return self._set(f"#{req.request_id} tick discarded: {why}")
    self.loop.mark_applied(res)
    self._last_valid_sim = self.sim_time
    self.last_choices = dec.reason
    notes = list(dec.notes)
    if dec.mission_status == "complete":
      if m.evidence_complete(fb):
        m.finish(COMPLETE, "model reported complete and the measurements confirm it")
        dec = Decision(dec.request_id, dec.generation, speed=0.0, yaw_rate=0.0, stop=True, kind="tick",
                       mission_id=dec.mission_id, snapshot_seq=dec.snapshot_seq, reason=dec.reason)
      else:
        notes.append("model said 'complete' but the measurements do not confirm it: mission continues")
    elif dec.mission_status == "abort":
      m.finish(FAILED, "the decision model judged that the mission cannot safely continue")
    if self._safety_cap is not None and dec.speed is not None and dec.speed > self._safety_cap:
      dec = replace(dec, speed=self._safety_cap)
      notes.append(f"speed held at the safety cap {self._safety_cap:.2f} m/s")
    warnings = [] if dec.is_noop else self._apply(dec, self._env_ids(self._mission_env))
    m.record(dec, fb, applied=True)
    self.last_decision = dec
    self.app_latencies.append(time.monotonic() - snap.wall_time if snap else float("nan"))
    self.last_outcome = "kept current action" if dec.is_noop else "applied"
    if m.state == FAILED:
      self.loop.supersede()
      return self._set(f"mission ABORTED by the model: {m.outcome}")
    if m.state == COMPLETE:
      return self._set(f"mission COMPLETE: {m.outcome}")
    note = "; ".join(notes + warnings)
    return self._set(f"#{req.request_id} tick ({res.latency_s * 1e3:.0f} ms): "
                     f"{'keep' if dec.is_noop else self.describe(dec)}" + (f" [{note}]" if note else ""))

  def _snapshot_invalid(self, snap: RobotFeedback | None, now: RobotFeedback) -> str:
    if snap is None:
      return "snapshot unknown"
    if now.sim_time - snap.sim_time > self.cl.max_snapshot_age_s:
      return f"snapshot {now.sim_time - snap.sim_time:.2f} s old > {self.cl.max_snapshot_age_s:.2f} s"
    d = math.hypot(now.x - snap.x, now.y - snap.y)
    if d > self.cl.max_pos_drift_m:
      return f"robot moved {d:.2f} m since the snapshot > {self.cl.max_pos_drift_m:.2f} m"
    dy = abs(wrap_angle(now.yaw - snap.yaw))
    if dy > self.cl.max_yaw_drift_rad:
      return f"robot turned {math.degrees(dy):.0f} deg since the snapshot"
    if now.command != snap.command:
      return "the planner command changed since the snapshot"
    return ""

  # --------------------------------------------------------------- status
  @staticmethod
  def describe(d: Decision) -> str:
    if d.stop:
      return "STOP"
    parts = []
    if d.gait is not None:
      parts.append(d.gait)
    if d.speed is not None:
      parts.append(f"v={d.speed:.2f} m/s")
    if d.heading_offset is not None:
      parts.append(f"dir={d.heading_offset:.2f} rad")
    if d.yaw_rate is not None:
      parts.append(f"yaw={d.yaw_rate:.2f} rad/s")
    return ", ".join(parts) or "no change"

  def _set(self, msg: str) -> str:
    self.last_msg = msg
    self.events.append(f"t={self.sim_time:.1f}s {msg}")
    LOG.info("[liquid] %s", msg)
    return msg

  def status_line(self) -> str:
    if self.loop is None:
      return "not attached"
    s = self.loop.summary()
    if self.loop.last_error:
      s += f" | last error: {self.loop.last_error}"
    return s

  def status_lines(self) -> list[str]:
    """Human-readable panel lines (mode, mission, decision, feedback, inference)."""
    out = [f"Mode: {self.mode}" + (" (one inference per instruction)" if self.mode == "oneshot"
                                     else " (persistent mission, periodic model decisions)")]
    if self.mode == "closed_loop":
      m = self.mission
      if m is None:
        out.append("Mission: none (type an instruction and press Send)")
      else:
        s = m.spec
        out.append(f"Mission #{s.mission_id}: {s.objective} [{m.state}"
                   + (f", phase {m.phase}" if m.state == ACTIVE else "") + f"] '{s.text}'")
        out.append(f"  desired v<= {s.preferred_speed:.2f} m/s | gait {s.gait or ('model' if s.gait_free else 'keep')}"
                   f" | cautious {s.cautious} | decisions {m.decisions} (kept {m.kept}, changed {m.changed})")
        if m.outcome:
          out.append(f"  outcome: {m.outcome}")
      tgt = "none" if self.target_rel is None else f"{self.target_rel[0]:.2f} m fwd, {self.target_rel[1]:.2f} m left"
      out.append(f"Target (relative to mission start): {tgt}")
      fb = self.last_feedback
      if fb is not None:
        line = (f"Feedback #{fb.seq} t={fb.sim_time:.1f}s: vx {fb.vx:+.2f} vy {fb.vy:+.2f} m/s, "
                f"yaw rate {fb.yaw_rate:+.2f}, tilt {math.degrees(fb.tilt):.1f} deg")
        if fb.contact_match is not None:
          line += f", contacts {100 * fb.contact_match:.0f}% as planned"
        if m is not None:
          te = m.target_error(fb)
          if te is not None:
            line += f" | target {te[0]:.2f} m at {math.degrees(te[1]):+.0f} deg"
        out.append(line)
        c = fb.command
        out.append(f"Command: {c.gait}, v={c.speed:.2f} m/s, dir {math.degrees(c.heading_offset):.0f} deg, "
                   f"yaw {c.yaw_rate:+.2f} rad/s" + (f" | safety cap {self._safety_cap:.2f} m/s"
                                                       if self._safety_cap is not None else ""))
      if self.last_choices:
        out.append(f"Latest model answer: {self.last_choices[:160]}")
      if self.last_outcome:
        out.append(f"Latest result: {self.last_outcome}")
      if self.loop is not None:
        period = self.cl.decision_period(self.loop.median_latency())
        app = sorted(x for x in self.app_latencies if math.isfinite(x))
        c = self.loop.counts
        out.append(f"Decision period {period:.2f} s (max {self.cl.decision_hz:g} Hz) | app latency p50 "
                   + (f"{1e3 * app[len(app) // 2]:.0f} ms" if app else "n/a")
                   + f" | stale state {c.get('stale_state', 0)} | stale mission {c.get('stale_mission', 0)}")
    out.append(self.status_line())
    return out


def build_runtime(*, model_id: str, device: str, compile: bool, limits: Limits, yaw_rate: float,
                  timeout_s: float, mode: str = "oneshot", closed_loop: ClosedLoopConfig | None = None,
                  autostart: bool = True) -> LiquidRuntime:
  """Real d1 runtime. Nothing heavy is imported until the worker loads the model."""
  if abs(yaw_rate) > limits.yaw_rate_max:
    raise ValueError(f"--liquid-yaw-rate {yaw_rate} exceeds --liquid-yaw-max {limits.yaw_rate_max}")
  from contact_rl.runtime.liquid_decider import LiquidDecider

  cl = replace(closed_loop or ClosedLoopConfig(), turn_rate=abs(yaw_rate))
  dec = LiquidDecider(model_id, device, compile)
  return LiquidRuntime(lambda lim: dec.make_decide(lim, yaw_rate, cl), dec.load, limits, timeout_s, mode=mode,
                       closed_loop=cl, autostart=autostart)
