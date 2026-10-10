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
