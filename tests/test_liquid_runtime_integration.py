"""Liquid AI runtime wired to the *real* GaitPlanner / command_override path.

Uses the real planner and ``apply_to_command_term`` with a fake command term
(no simulator) and a mocked model. Needs torch; the viewer-dispatch tests
also need mjlab. The real LiquidAI model is not exercised here.
"""

from __future__ import annotations

import ast
import math
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from conftest import SRC, load_mdp  # noqa: E402

P = load_mdp("planning")
C = load_mdp("command_override")

from contact_rl.runtime import decision_schema as S  # noqa: E402
from contact_rl.runtime.viewer_bridge import LiquidRuntime, planner_state  # noqa: E402


class FakeTerm:
  """Just enough of ContactGoalCommand for apply_to_command_term / reanchor."""

  def __init__(self, n=4):
    cfg = SimpleNamespace(stride_length_range=(0.0, 0.3), stance_width_range=(0.1, 0.3),
                          leg_offset_range=(-0.15, 0.15), heading_range=(-math.pi, math.pi),
                          yaw_rate_range=(-math.pi, math.pi), rel_yaw_rate_envs=0.0, gaits=("trot",),
                          resampling_time_range=(0.34, 0.36))
    self.planner = P.GaitPlanner(cfg, n, "cpu", generator=torch.Generator().manual_seed(0))
    self.num_envs = n
    ids = torch.arange(n)
    self.planner.reset(ids, torch.zeros(n, 2), torch.zeros(n))
    self.planner.stride[:] = 0.1 * 2 * self.planner.switch_dt  # 0.1 m/s
    self.planner.heading_off[:] = 0.0
    self.robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(n, 3), heading_w=torch.zeros(n)))
    self.synced = 0

  def _sync_goals(self, ids):
    self.synced += 1


class Harness:
  """Mimics BaseViewer: thread-safe action deque drained on the 'sim thread'."""

  def __init__(self, rt: LiquidRuntime, term: FakeTerm):
    self.rt, self.term, self.actions, self.msgs = rt, term, deque(), []
    rt.attach(lambda name, payload: self.actions.append(payload), lambda: term, lambda: None, lambda: 0)

  def pump(self):
    while self.actions:
      kind, arg = self.actions.popleft()
      self.msgs.append(self.rt.handle(kind, arg))

  def pump_until(self, pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      self.pump()
      if pred():
        return True
      time.sleep(0.005)
    return False


def runtime_from_raw(raw_fn, gate: threading.Event | None = None):
  def factory(limits):
    def decide(req):
      if gate is not None:
        gate.wait(5)
      return S.validate(raw_fn(req), request_id=req.request_id, generation=req.generation,
                        current=req.state, limits=limits)
    return decide
  return LiquidRuntime(factory, load=None)


def test_schema_constants_match_planner():
  assert S.GAITS == P.GAITS
  assert S.GAIT_PERIODS == {g: len(p) for g, p in P.GAIT_PATTERNS.items()}
  assert S.TRAIN_STRIDE_MAX == C.TRAIN_STRIDE_MAX and S.TRAIN_YAW_RATE_MAX == C.TRAIN_YAW_RATE_MAX
  for g in P.GAITS:
    assert S.train_speed_max(g, 0.35) == pytest.approx(C.max_train_speed(g, 0.35))


def test_planner_state_snapshot_is_plain_python():
  st = planner_state(FakeTerm(), None, 0)
  assert st.gait == "trot" and st.speed == pytest.approx(0.1) and st.target_gaits == ("trot",)
  assert all(isinstance(v, float) for v in (st.speed, st.heading_offset, st.yaw_rate))


# ------------------------------------- 10./12. validated decisions reach planner
def test_text_command_reaches_planner_through_apply_to_command_term():
  term = FakeTerm()
  rt = runtime_from_raw(lambda req: {"speed": 0.18, "yaw_rate": 0.4})
  h = Harness(rt, term)
  rt.start()
  try:
    rt.send_text("turn left and walk a bit faster")
    assert h.pump_until(lambda: rt.loop.counts["applied"] == 1)
  finally:
    rt.stop()
  d = C.describe_command(term.planner, 0)
  assert d["speed_mps"] == pytest.approx(0.18) and d["yaw_rate_rps"] == pytest.approx(0.4)
  assert torch.all(term.planner.use_yaw_rate)
  assert term._user_cmd_snapshot is not None  # went through apply_to_command_term (restorable)
  assert term.synced == 0  # no gait change -> no re-anchor; applies at the next contact switch
  assert C.restore_sampled_commands(term, reanchor_now=False)


def test_gait_change_reanchors_and_stays_in_trained_stride():
  term = FakeTerm()
  rt = runtime_from_raw(lambda req: {"gait": "crawl"})
  h = Harness(rt, term)
  term.planner.stride[:] = 0.3  # trot at the trained max (0.43 m/s)
  rt.start()
  try:
    rt.send_text("crawl")
    assert h.pump_until(lambda: rt.loop.counts["applied"] == 1)
  finally:
    rt.stop()
  assert torch.all(term.planner.gait == P.GAITS.index("crawl")) and term.synced == 1
  assert float(term.planner.stride.max()) <= C.TRAIN_STRIDE_MAX + 1e-6


def test_invalid_model_output_never_touches_planner():
  term = FakeTerm()
  before = C.snapshot_params(term.planner)
  rt = runtime_from_raw(lambda req: {"speed": 5.0, "gait": "gallop"})
  h = Harness(rt, term)
  rt.start()
  try:
    rt.send_text("sprint")
    assert h.pump_until(lambda: rt.loop.counts["invalid"] == 1)
  finally:
    rt.stop()
  for k, v in before.items():
    assert torch.equal(getattr(term.planner, k), v)


def test_superseded_answer_is_discarded_and_manual_stop_wins():
  term = FakeTerm()
  gate = threading.Event()
  rt = runtime_from_raw(lambda req: {"speed": 0.2}, gate=gate)
  h = Harness(rt, term)
  rt.start()
  try:
    rt.send_text("walk faster")
    h.pump()
    time.sleep(0.05)
    rt.send_stop()  # deterministic, bypasses the model
    h.pump()
    gate.set()  # the slow answer for "walk faster" now arrives
    time.sleep(0.2)
    h.pump()
  finally:
    rt.stop()
  assert rt.loop.counts["applied"] == 0
  d = C.describe_command(term.planner, 0)
  assert d["speed_mps"] == 0.0 and d["yaw_rate_rps"] == 0.0


def test_sim_keeps_stepping_while_model_is_slow():
  term = FakeTerm()
  gate = threading.Event()
  rt = runtime_from_raw(lambda req: {"speed": 0.15}, gate=gate)
  h = Harness(rt, term)
  rt.start()
  ids = torch.arange(term.num_envs)
  try:
    rt.send_text("walk")
    t0 = time.monotonic()
    for _ in range(200):  # stand-in for env.step: planner keeps advancing
      h.pump()
      term.planner.advance(ids)
    assert time.monotonic() - t0 < 2.0 and rt.loop.counts["applied"] == 0
    gate.set()
    assert h.pump_until(lambda: rt.loop.counts["applied"] == 1)
  finally:
    rt.stop()


def test_stdin_reader_enqueues_only():
  import io

  term = FakeTerm()
  rt = runtime_from_raw(lambda req: {"speed": 0.12})
  h = Harness(rt, term)
  rt.start()
  try:
    rt.start_stdin_reader(io.StringIO("walk forward slowly\n\nstop\n")).join(2)
    kinds = [a[0] for a in h.actions]
    assert kinds == ["liquid_cmd", "liquid_stop"]
  finally:
    rt.stop()


# ------------------------------------------------ 11./13. isolation guarantees
ALLOWED_IMPORTERS = {"runtime", "play_viewer.py", "scripts/play.py"}


def _imports_runtime(path: Path) -> bool:
  tree = ast.parse(path.read_text())
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("contact_rl.runtime"):
      return True
    if isinstance(node, ast.Import) and any(a.name.startswith("contact_rl.runtime") for a in node.names):
      return True
  return False


def test_training_and_policy_code_never_import_the_runtime_layer():
  pkg = SRC / "contact_rl"
  offenders = []
  for f in pkg.rglob("*.py"):
    rel = f.relative_to(pkg).as_posix()
    if rel.split("/")[0] == "runtime" or rel in ALLOWED_IMPORTERS:
      continue
    if _imports_runtime(f):
      offenders.append(rel)
  assert offenders == []


def test_play_imports_runtime_only_lazily():
  tree = ast.parse((SRC / "contact_rl" / "scripts" / "play.py").read_text())
  top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
  mods = [n.module or "" for n in top if isinstance(n, ast.ImportFrom)] + \
    [a.name for n in top if isinstance(n, ast.Import) for a in n.names]
  assert not any(m.startswith("contact_rl.runtime") for m in mods)


def test_viewer_without_liquid_is_unchanged():
  pytest.importorskip("mjlab")
  pytest.importorskip("viser")
  import inspect

  from contact_rl.play_viewer import ContactPlayViewer

  assert inspect.signature(ContactPlayViewer).parameters["liquid"].default is None
  v = ContactPlayViewer.__new__(ContactPlayViewer)
  v._liquid, v._sim_lock, v._msg = None, threading.Lock(), ""
  v._update_status_display = lambda: None
  from mjlab.viewer.base import ViewerAction
  # without --liquid, liquid_* actions are unknown kinds, exactly as before
  assert v._handle_custom_action(ViewerAction.CUSTOM, ("liquid_cmd", "go")) is False
  assert v._msg == ""


def test_viewer_dispatches_liquid_actions_on_sim_thread():
  pytest.importorskip("mjlab")
  pytest.importorskip("viser")
  from mjlab.viewer.base import ViewerAction

  from contact_rl.play_viewer import ContactPlayViewer

  term = FakeTerm()
  rt = runtime_from_raw(lambda req: {"speed": 0.15})
  v = ContactPlayViewer.__new__(ContactPlayViewer)
  v._liquid, v._sim_lock, v._msg, v._actions = rt, threading.Lock(), "", deque()
  v._update_status_display = lambda: None
  rt.attach(v.request_action, lambda: term, lambda: None, lambda: 0)
  rt.start()
  try:
    rt.send_text("walk a little faster")
    deadline = time.monotonic() + 3
    while rt.loop.counts["applied"] == 0 and time.monotonic() < deadline:
      while v._actions:
        a, p = v._actions.popleft()
        v._handle_custom_action(a, p)
      time.sleep(0.005)
  finally:
    rt.stop()
  assert rt.loop.counts["applied"] == 1 and v._msg.startswith("liquid: #")
  assert C.describe_command(term.planner, 0)["speed_mps"] == pytest.approx(0.15)
  assert isinstance(ViewerAction.CUSTOM, ViewerAction)


def test_play_config_defaults_keep_policy_only_mode():
  pytest.importorskip("mjlab")
  from contact_rl.scripts import play

  cfg = play._config_cls()()
  assert cfg.liquid is False and play.build_liquid_runtime(cfg) is None


# =====================================================================
# Closed-loop mission mode wired to the real GaitPlanner (mocked model)
# =====================================================================
from contact_rl.runtime.decision_schema import ClosedLoopConfig  # noqa: E402
from contact_rl.runtime.liquid_decider import make_decide  # noqa: E402
from contact_rl.runtime.mission import COMPLETE, FAILED, FINAL_TURN, PAUSED, STOPPED  # noqa: E402


class RobotTerm(FakeTerm):
  """FakeTerm + the measured robot state robot_feedback() reads (mutable to emulate motion)."""

  def __init__(self, n=2):
    super().__init__(n)
    self.robot = SimpleNamespace(data=SimpleNamespace(
      root_link_pos_w=torch.zeros(n, 3), heading_w=torch.zeros(n), root_link_lin_vel_b=torch.zeros(n, 3),
      root_link_ang_vel_b=torch.zeros(n, 3), projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * n)))

  def move(self, x=None, y=None, yaw=None, vx=None, tilt_deg=None, env=0):
    d = self.robot.data
    if x is not None:
      d.root_link_pos_w[env, 0] = x
    if y is not None:
      d.root_link_pos_w[env, 1] = y
    if yaw is not None:
      d.heading_w[env] = yaw
    if vx is not None:
      d.root_link_lin_vel_b[env, 0] = vx
    if tilt_deg is not None:
      t = math.radians(tilt_deg)
      d.projected_gravity_b[env] = torch.tensor([math.sin(t), 0.0, -math.cos(t)])


def mission_ans(objective="walk", speed="medium", after_arrival="stop", gait="keep", caution="normal",
                direction="forward", turn="keep"):
  return {k: {"choice": v, "confidence": 0.9} for k, v in dict(
    objective=objective, speed=speed, direction=direction, turn=turn, after_arrival=after_arrival, gait=gait,
    caution=caution).items()}


def tick_ans(speed="keep", steer="keep", gait="keep", status="continue"):
  return {k: {"choice": v, "confidence": 0.9} for k, v in dict(speed=speed, steer=steer, gait=gait,
                                                                 status=status).items()}


class ScriptedModel:
  """Mock ``system_one`` answers; records every state it was asked about."""

  def __init__(self, mission, tick=None, gate: threading.Event | None = None, fail: Exception | None = None):
    self.mission, self.tick, self.gate, self.fail = mission, tick or (lambda st: tick_ans()), gate, fail
    self.mission_calls, self.tick_states = [], []

  def __call__(self, state, questions):
    if "objective" in questions:
      self.mission_calls.append(state)
      return self.mission
    if self.gate is not None:
      self.gate.wait(5)
    if self.fail is not None:
      raise self.fail
    self.tick_states.append(state)
    return self.tick(state)


class ClosedHarness:
  def __init__(self, model: ScriptedModel, term=None, cfg=None, gaits=None, **rt_kw):
    self.term = term or RobotTerm()
    self.cfg = cfg or ClosedLoopConfig()
    lim = S.Limits(gaits=gaits) if gaits else S.Limits()
    self.model = model
    self.rt = LiquidRuntime(lambda limits: make_decide(model, limits, 0.5, self.cfg), None, lim, mode="closed_loop",
                            closed_loop=self.cfg, **rt_kw)
    self.actions = deque()
    self.msgs = []
    self.rt.attach(lambda n, p: self.actions.append(p), lambda: self.term, lambda: None, lambda: 0, step_dt=0.02)
    self.rt.start()

  def pump(self):
    while self.actions:
      k, a = self.actions.popleft()
      self.msgs.append(self.rt.handle(k, a))

  def settle(self, timeout=2.0):
    """Wait for the worker (instant mock inference), then apply what it delivered."""
    deadline = time.monotonic() + timeout
    while self.rt.loop.busy and time.monotonic() < deadline:
      time.sleep(0.001)
    time.sleep(0.002)
    self.pump()

  def step(self, n=1, dones=None, settle=True):
    for _ in range(n):
      self.pump()
      self.rt.on_sim_step(dones)
      if settle:
        self.settle()

  def start(self, text="walk forward at a moderate speed"):
    self.rt.send_text(text)
    self.pump()
    self.settle()
    return self.rt.mission

  def close(self):
    self.rt.stop()


def cmd(term, i=0):
  return C.describe_command(term.planner, i)


def test_closed_loop_mission_then_repeated_state_aware_ticks():
  model = ScriptedModel(mission_ans(speed="medium"), tick=lambda st: tick_ans("increase"))
  h = ClosedHarness(model)
  try:
    m = h.start()
    assert m is not None and m.state == "active" and len(model.mission_calls) == 1
    v0 = cmd(h.term)["speed_mps"]
    h.term.move(vx=v0)
    h.step(50 * 4)  # 4 sim seconds at 50 Hz, decisions at <= 1 Hz
  finally:
    h.close()
  assert 3 <= len(model.tick_states) <= 5  # low rate, not per frame
  assert len(model.mission_calls) == 1  # the instruction was interpreted once, never re-interpreted
  assert all(s["mission"]["user instruction"] == "walk forward at a moderate speed" for s in model.tick_states)
  assert cmd(h.term)["speed_mps"] == pytest.approx(m.spec.preferred_speed)  # converged to intent, no compounding
  assert h.rt.loop.counts["applied"] >= 4


def test_ticks_only_touch_the_mission_env_and_reach_the_planner():
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans(steer="left"))
  h = ClosedHarness(model)
  other_before = C.snapshot_params(h.term.planner)
  try:
    h.start()
    h.step(60)
  finally:
    h.close()
  assert cmd(h.term, 0)["yaw_rate_rps"] == pytest.approx(0.5) and bool(h.term.planner.use_yaw_rate[0])
  for k, v in other_before.items():  # env 1 untouched
    assert torch.equal(getattr(h.term.planner, k)[1], v[1])
  assert h.term._user_cmd_snapshot is not None  # via apply_to_command_term


def test_keep_answer_leaves_the_planner_untouched():
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans())
  h = ClosedHarness(model)
  try:
    h.start()
    before = C.snapshot_params(h.term.planner)
    h.step(120)
  finally:
    h.close()
  for k, v in before.items():
    assert torch.equal(getattr(h.term.planner, k), v)
  assert h.rt.mission.kept >= 2 and h.rt.last_outcome == "kept current action"


def test_state_feedback_changes_subsequent_decisions():
  def tick(st):  # a "model" that reacts to the measured stability it is shown
    return tick_ans("decrease") if st["measured state"]["stability"] != "stable" else tick_ans("increase")
  model = ScriptedModel(mission_ans(speed="fast"), tick=tick)
  h = ClosedHarness(model, cfg=ClosedLoopConfig(tilt_caution_rad=math.radians(30),
                                                tilt_caution_cautious_rad=math.radians(30)))
  try:
    h.start()
    h.step(110)
    v_stable = cmd(h.term)["speed_mps"]
    h.term.move(tilt_deg=20)  # marginal (>= 0.6 x caution), below the deterministic guard
    h.step(110)
  finally:
    h.close()
  assert model.tick_states[0]["measured state"]["stability"] == "stable"
  assert model.tick_states[-1]["measured state"]["stability"] == "marginal"
  assert cmd(h.term)["speed_mps"] < v_stable


def test_previous_action_outcome_is_fed_back():
  model = ScriptedModel(mission_ans("reach_target"), tick=lambda st: tick_ans("increase", "straight"))
  h = ClosedHarness(model)
  h.rt.on_target((3.0, 0.0))
  try:
    h.start("go to the target")
    h.step(55)
    h.term.move(x=0.2)
    h.step(55)
  finally:
    h.close()
  last = model.tick_states[-1]
  assert last["target"]["distance"] == "2.80 m"
  assert last["previous decision"]["result"].startswith("distance decreased")


def test_target_arrival_final_turn_and_measured_completion():
  def tick(st):
    if st["mission"]["phase"] == "final_turn":
      return tick_ans(steer="left")
    return tick_ans("increase", "straight")
  model = ScriptedModel(mission_ans("reach_target", after_arrival="turn_left"), tick=tick)
  h = ClosedHarness(model)
  h.rt.on_target((1.0, 0.0))
  try:
    h.start("move toward the target and turn left")
    h.step(60)
    assert h.rt.mission.state == "active"
    h.term.move(x=0.9)  # arrives (within 0.25 m)
    h.step(10)
    m = h.rt.mission
    assert m.phase == FINAL_TURN and cmd(h.term)["speed_mps"] == 0.0
    assert cmd(h.term)["yaw_rate_rps"] == pytest.approx(0.5)  # the model steered the final turn
    h.term.move(yaw=math.radians(60))
    h.step(10)
    assert m.state == "active"
    h.term.move(yaw=math.radians(80))
    h.step(10)
  finally:
    h.close()
  assert m.state == COMPLETE and "measured" in m.outcome
  assert cmd(h.term)["speed_mps"] == 0.0 and cmd(h.term)["yaw_rate_rps"] == 0.0


def test_reach_target_without_a_target_is_refused():
  model = ScriptedModel(mission_ans("reach_target"))
  h = ClosedHarness(model)
  try:
    assert h.start("go to the target") is None
  finally:
    h.close()
  assert "no target" in h.rt.last_msg or "none is configured" in h.rt.last_msg


def test_model_complete_claim_without_evidence_is_ignored():
  model = ScriptedModel(mission_ans("reach_target"), tick=lambda st: tick_ans(status="complete"))
  h = ClosedHarness(model)
  h.rt.on_target((3.0, 0.0))
  try:
    h.start("go to the target")
    h.step(120)
  finally:
    h.close()
  assert h.rt.mission.state == "active"


def test_model_abort_stops_and_fails_the_mission():
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans(status="abort"))
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(60)
  finally:
    h.close()
  assert h.rt.mission.state == FAILED and cmd(h.term)["speed_mps"] == 0.0


def test_stale_state_is_rejected():
  gate = threading.Event()
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans("increase", "left"), gate=gate)
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(50, settle=False)  # first tick submitted, model blocked
    assert h.rt.loop.busy
    before = C.snapshot_params(h.term.planner)
    h.term.move(x=0.5)  # robot moved more than max_pos_drift since the snapshot
    gate.set()
    h.settle()
  finally:
    h.close()
  assert h.rt.loop.counts["stale_state"] == 1 and "stale state" in h.rt.last_outcome
  assert torch.equal(h.term.planner.yaw_rate, before["yaw_rate"])


def test_old_snapshot_age_is_rejected():
  gate = threading.Event()
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans("increase"), gate=gate)
  h = ClosedHarness(model, cfg=ClosedLoopConfig(deadline_s=30.0))
  try:
    h.start()
    h.step(50, settle=False)
    h.step(100, settle=False)  # 2 s of sim pass while the answer is pending
    gate.set()
    h.settle()
  finally:
    h.close()
  assert h.rt.loop.counts["stale_state"] >= 1 and "old" in h.rt.last_outcome


def test_new_instruction_invalidates_in_flight_tick():
  gate = threading.Event()
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans(steer="left"), gate=gate)
  h = ClosedHarness(model)
  try:
    first = h.start()
    h.step(50, settle=False)
    h.rt.send_text("walk forward slowly")
    h.pump()
    gate.set()
    h.settle()
    h.settle()
  finally:
    h.close()
  assert first.state == STOPPED and h.rt.mission is not first
  assert cmd(h.term)["yaw_rate_rps"] == 0.0  # the old mission's left turn never applied


@pytest.mark.parametrize("how", ["stop", "reset", "done", "pause"])
def test_stop_reset_and_pause_invalidate_in_flight_ticks(how):
  gate = threading.Event()
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans("increase", "left"), gate=gate)
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(50, settle=False)
    assert h.rt.loop.busy
    if how == "stop":
      h.rt.send_stop()
      h.pump()
    elif how == "reset":
      h.rt.on_env_reset()
    elif how == "pause":
      h.rt.send_pause()
      h.pump()
    else:
      h.step(1, dones=torch.tensor([1, 0]), settle=False)
    gate.set()
    h.settle()
    h.step(100)  # no further ticks
  finally:
    h.close()
  assert cmd(h.term)["speed_mps"] == 0.0 and cmd(h.term)["yaw_rate_rps"] == 0.0
  assert h.rt.mission.state == {"stop": STOPPED, "reset": FAILED, "done": FAILED, "pause": PAUSED}[how]
  assert len(model.tick_states) <= 1  # the in-flight one (if it had started), never answered into the planner


def test_pause_then_resume_asks_the_model_again():
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans("increase"))
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(60)
    h.rt.send_pause()
    h.step(200)
    n = len(model.tick_states)
    assert h.rt.mission.state == PAUSED and cmd(h.term)["speed_mps"] == 0.0
    h.rt.send_resume()
    h.step(5)
  finally:
    h.close()
  assert h.rt.mission.state == "active" and len(model.tick_states) == n + 1
  assert cmd(h.term)["speed_mps"] > 0.0


def test_inference_failures_trigger_the_documented_fallback():
  model = ScriptedModel(mission_ans(), fail=RuntimeError("cuda error"))
  h = ClosedHarness(model)
  try:
    h.start()
    v = cmd(h.term)["speed_mps"]
    h.step(int(50 * 1.2))  # one failure: last valid command kept
    assert h.rt.loop.counts["failed"] == 1 and cmd(h.term)["speed_mps"] == pytest.approx(v)
    h.step(50 * 4)
  finally:
    h.close()
  m = h.rt.mission
  assert m.state == FAILED and "controlled stop" in m.outcome
  assert cmd(h.term)["speed_mps"] == 0.0 and cmd(h.term)["yaw_rate_rps"] == 0.0


def test_inference_timeout_counts_as_failure_and_late_answer_is_dropped():
  gate = threading.Event()
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans(steer="left"), gate=gate)
  h = ClosedHarness(model, cfg=ClosedLoopConfig(deadline_s=0.05))
  try:
    h.start()
    h.step(50, settle=False)
    time.sleep(0.1)
    h.step(5, settle=False)  # sim side notices the deadline
    assert h.rt.loop.counts["timeout"] == 1
    gate.set()
    h.settle()
  finally:
    h.close()
  assert cmd(h.term)["yaw_rate_rps"] == 0.0


def test_safety_tilt_overrides_the_model():
  model = ScriptedModel(mission_ans(speed="fast"), tick=lambda st: tick_ans("increase"))
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(100)
    v = cmd(h.term)["speed_mps"]
    h.term.move(tilt_deg=25)  # above the 20 deg caution limit
    h.step(100)
    assert cmd(h.term)["speed_mps"] <= 0.5 * v + 1e-6  # model keeps saying "increase"; the cap wins
    h.term.move(tilt_deg=50)  # above the abort limit
    h.step(5)
  finally:
    h.close()
  assert h.rt.mission.state == FAILED and cmd(h.term)["speed_mps"] == 0.0


def test_non_finite_state_stops_the_robot():
  model = ScriptedModel(mission_ans())
  h = ClosedHarness(model)
  try:
    h.start()
    h.term.move(vx=float("nan"))
    h.step(10)
  finally:
    h.close()
  assert h.rt.mission.state == FAILED and cmd(h.term)["speed_mps"] == 0.0


def test_gait_choice_reaches_planner_reanchors_and_stays_in_range():
  model = ScriptedModel(mission_ans(gait="suitable"), tick=lambda st: tick_ans(gait="crawl"))
  h = ClosedHarness(model, gaits=("trot", "crawl"))
  h.term.planner.stride[:] = 0.3
  try:
    h.start("use a suitable gait")
    synced = h.term.synced
    h.step(60)
  finally:
    h.close()
  assert int(h.term.planner.gait[0]) == P.GAITS.index("crawl") and h.term.synced == synced + 1
  assert float(h.term.planner.stride[0].max()) <= C.TRAIN_STRIDE_MAX + 1e-6


def test_oneshot_mode_is_still_available_and_never_ticks():
  term = RobotTerm()
  rt = runtime_from_raw(lambda req: {"speed": 0.15})
  h = Harness(rt, term)
  rt.start()
  try:
    rt.send_text("walk")
    assert h.pump_until(lambda: rt.loop.counts["applied"] == 1)
    for _ in range(300):
      rt.on_sim_step(None)
  finally:
    rt.stop()
  assert rt.mode == "oneshot" and rt.loop.counts["ticks"] == 0 and rt.mission is None


def test_status_lines_report_mode_mission_and_results():
  model = ScriptedModel(mission_ans(), tick=lambda st: tick_ans("increase"))
  h = ClosedHarness(model)
  try:
    h.start()
    h.step(60)
    lines = "\n".join(h.rt.status_lines())
  finally:
    h.close()
  for s in ("Mode: closed_loop", "Mission #", "Feedback #", "Command:", "Latest model answer", "Latest result",
            "Decision period"):
    assert s in lines


def test_play_cli_builds_closed_loop_runtime_by_default():
  pytest.importorskip("mjlab")
  from dataclasses import replace

  from contact_rl.scripts import play

  cfg = replace(play._config_cls()(), liquid=True, liquid_target=(2.0, 0.5))
  rt = play.build_liquid_runtime(cfg)
  assert rt.mode == "closed_loop" and rt.target_rel == (2.0, 0.5) and rt.cl.decision_hz == 1.0
  with pytest.raises(SystemExit):
    play.build_liquid_runtime(replace(cfg, liquid_mode="oneshot"))  # target needs closed loop
