"""Liquid AI runtime layer: decision contract, answer mapping, worker loop.

Pure python: no torch / GPU / model download. The model is mocked here; these
tests do NOT show that the real LiquidAI model works (see docs/liquid_runtime.md).
"""

from __future__ import annotations

import dataclasses
import math
import sys
import threading
import time

import pytest

from contact_rl.runtime.decision_loop import DecisionLoop, Request
from contact_rl.runtime.decision_schema import (
  MAX_SPEED_MPS,
  Decision,
  DecisionError,
  Limits,
  PlannerState,
  to_user_command_kwargs,
  validate,
)
from contact_rl.runtime.liquid_decider import (
  LiquidDecider,
  LiquidUnavailable,
  answers_to_raw,
  build_questions,
  build_state,
)

L = Limits()
ST = PlannerState("trot", 0.1, 0.0, 0.0, ("trot",))


def V(raw, current=ST, limits=L, rid=1, gen=1):
  return validate(raw, request_id=rid, generation=gen, current=current, limits=limits)


# ------------------------------------------------------------ 1. valid parsing
def test_valid_decision_passes():
  d = V({"gait": "trot", "speed": 0.15, "yaw_rate": 0.3})
  assert (d.gait, d.speed, d.yaw_rate, d.request_id, d.generation) == ("trot", 0.15, 0.3, 1, 1)
  assert to_user_command_kwargs(d) == {"gait": "trot", "speed": 0.15, "yaw_rate": 0.3}


def test_empty_decision_is_noop():
  assert V({}).is_noop


# --------------------------------------------- 2. invalid / incomplete responses
@pytest.mark.parametrize("raw", [["trot"], "trot", None, 3, {"torque": [0.1] * 12}, {"speed": 0.1, "exec": "rm"},
                                 {"stop": "yes"}, {"gait": 3}])
def test_malformed_rejected(raw):
  with pytest.raises(DecisionError):
    V(raw)


@pytest.mark.parametrize("answers", [
  None, {}, {"speed": {"choice": "slow"}},  # incomplete
  {"speed": {"choice": "warp9"}, "turn": {"choice": "keep"}, "direction": {"choice": "keep"}, "gait": {"choice": "keep"}},
  {"speed": "slow", "turn": {"choice": "keep"}, "direction": {"choice": "keep"}, "gait": {"choice": "keep"}},
])
def test_malformed_model_answers_rejected(answers):
  with pytest.raises(DecisionError):
    answers_to_raw(answers, build_questions(), ST, L, 0.5)


# ---------------------------------------------------- 3. out-of-range values
@pytest.mark.parametrize("raw", [{"speed": -0.1}, {"speed": 0.5}, {"speed": float("nan")}, {"speed": "fast"},
                                 {"speed": True}, {"heading_offset": 4.0}, {"yaw_rate": 1.5},
                                 {"yaw_rate": float("inf")}])
def test_out_of_range_rejected(raw):
  with pytest.raises(DecisionError):
    V(raw)


def test_speed_cap_is_gait_specific_and_below_training_max():
  cap_trot = L.speed_cap(("trot",))
  assert cap_trot == pytest.approx(0.8 * MAX_SPEED_MPS["trot"])
  with pytest.raises(DecisionError):  # crawl (P=4) has half the trot cap
    V({"gait": "crawl", "speed": cap_trot})
  with pytest.raises(DecisionError):  # mixed target envs: most restrictive gait wins
    V({"speed": 0.2}, current=PlannerState("trot", 0.15, 0.0, 0.0, ("trot", "crawl")))


def test_limits_cannot_exceed_training_range():
  for bad in ({"speed_fraction": 1.5}, {"speed_fraction": 0.0}, {"yaw_rate_max": 4.0}, {"gaits": ("gallop",)},
              {"gaits": ()}, {"max_speed_step": 0.0}):
    with pytest.raises(ValueError):
      Limits(**bad)


def test_heading_and_yaw_together_rejected():
  with pytest.raises(DecisionError):
    V({"heading_offset": 0.5, "yaw_rate": 0.5})


# ------------------------------------------------------- 4. unsupported gaits
@pytest.mark.parametrize("g", ["gallop", "TROT", "", "walk"])
def test_unknown_gait_rejected(g):
  with pytest.raises(DecisionError):
    V({"gait": g})


def test_gait_not_enabled_by_limits_rejected():
  with pytest.raises(DecisionError):
    V({"gait": "jump"}, limits=Limits(gaits=("trot", "crawl")))


# --------------------------------------------------------- 5. per-step limits
def test_per_step_change_is_bounded():
  d = V({"speed": 0.3, "yaw_rate": 1.0})
  assert d.speed == pytest.approx(0.2) and d.yaw_rate == pytest.approx(0.5)
  assert any("rate-limited" in n for n in d.notes)


def test_gait_change_reduces_speed_into_new_gait_range():
  cur = PlannerState("trot", 0.3, 0.0, 0.0, ("trot",))
  d = V({"gait": "crawl"}, current=cur)
  assert d.speed == pytest.approx(L.speed_cap(("crawl",)))


def test_turn_clears_heading_offset_and_vice_versa():
  d = V({"yaw_rate": 0.4}, current=PlannerState("trot", 0.1, 0.7, 0.0, ("trot",)))
  assert d.heading_offset == 0.0
  d = V({"heading_offset": 0.7}, current=PlannerState("trot", 0.1, 0.0, 0.4, ("trot",)))
  assert d.yaw_rate == 0.0


def test_stop_is_explicit_and_not_rate_limited():
  d = V({"stop": True, "speed": 9.0}, current=PlannerState("trot", 0.34, 0.0, 0.5, ("trot",)))
  assert d.stop and d.speed == 0.0 and d.yaw_rate == 0.0


def test_reason_is_diagnostic_only_and_truncated():
  d = V({"reason": "x" * 1000})
  assert len(d.reason) == 200 and d.is_noop


# ------------------------------------------------------------- answer mapping
def _ans(speed="keep", turn="keep", direction="keep", gait="keep", conf=0.99):
  return {k: {"choice": v, "confidence": conf} for k, v in
          dict(speed=speed, turn=turn, direction=direction, gait=gait).items()}


def test_examples_map_to_supported_parameters():
  q = build_questions()
  cap = L.speed_cap(("trot",))
  assert answers_to_raw(_ans("slow", direction="forward"), q, ST, L, 0.5) | {"reason": ""} == \
    {"speed": pytest.approx(0.3 * cap), "heading_offset": 0.0, "reason": ""}
  assert answers_to_raw(_ans("faster"), q, ST, L, 0.5)["speed"] == pytest.approx(0.15)
  assert answers_to_raw(_ans(turn="left"), q, ST, L, 0.5)["yaw_rate"] == 0.5
  assert answers_to_raw(_ans(turn="right"), q, ST, L, 0.5)["yaw_rate"] == -0.5
  assert answers_to_raw(_ans("stop", turn="left"), q, ST, L, 0.5)["stop"] is True


def test_slower_above_the_runtime_cap_slows_down_instead_of_failing():
  fast = PlannerState("trot", 0.42, 0.0, 0.0, ("trot",))  # sampled trained command, above 0.8 x max
  raw = answers_to_raw(_ans("slower"), build_questions(), fast, L, 0.5)
  d = V(raw, current=fast)
  assert d.speed == pytest.approx(L.speed_cap(("trot",))) and d.speed < fast.speed


def test_low_confidence_is_not_trusted_but_still_validated():
  raw = answers_to_raw(_ans("fast", conf=0.01), build_questions(), ST, L, 0.5)
  d = V(raw)
  assert d.speed == pytest.approx(0.2)  # still rate-limited from 0.1, confidence ignored for control


def test_questions_follow_decision_index_schema():
  qs = build_questions(("trot", "crawl"))
  for q in qs.values():
    assert q["type"] == "choice" and q["instructions"] and len(q["criteria"]) >= 2 and "keep" in q["criteria"]
  assert set(qs["gait"]["criteria"]) == {"keep", "trot", "crawl"}


def test_state_contains_command_and_real_values_only():
  s = build_state("turn left", ST)
  assert "User command: turn left" in s and "Current gait: trot" in s and "0.10 m/s" in s


# ------------------------------------------------- adapter (mocked model)
class FakeModel:
  def __init__(self, answers):
    self.answers, self.calls = answers, []

  def system_one(self, state, questions):
    self.calls.append((state, sorted(questions)))
    return {"answers": self.answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


def test_adapter_decide_uses_system_one_and_validates():
  pytest.importorskip("torch")
  dec = LiquidDecider()
  dec.model = FakeModel(_ans("slow", turn="left"))
  decide = dec.make_decide(L, 0.5)
  d = decide(Request(1, 1, "turn left slowly", ST, 0.0))
  assert dec.model.calls[0][1] == ["direction", "gait", "speed", "turn"]
  assert d.yaw_rate == 0.5 and d.speed == pytest.approx(0.3 * L.speed_cap(("trot",)))


def test_adapter_reports_missing_dependencies(monkeypatch):
  pytest.importorskip("torch")
  monkeypatch.setitem(sys.modules, "transformers", None)
  with pytest.raises(LiquidUnavailable):
    LiquidDecider(device="cpu").load()


def test_adapter_refuses_to_answer_before_load():
  with pytest.raises(LiquidUnavailable):
    LiquidDecider().answer("go", ST, build_questions())


# -------------------------------------------------------------- decision loop
def _loop(decide, **kw):
  out = []
  return DecisionLoop(decide, out.append, **kw), out


def _dec(speed):
  return lambda req: V({"speed": speed}, rid=req.request_id, gen=req.generation)


def test_valid_decision_is_delivered_with_latency():
  ticks = iter([0.0, 0.0, 0.25])
  loop, out = _loop(_dec(0.15), clock=lambda: next(ticks))
  req = loop.submit("faster", ST)
  loop.process(req)
  assert len(out) == 1 and out[0].latency_s == pytest.approx(0.25) and loop.is_current(out[0])


# ---------------------------------------------------- 6. failures are safe
def test_invalid_output_and_exceptions_never_deliver():
  loop, out = _loop(_dec(9.0))
  loop.process(loop.submit("x", ST))
  assert out == [] and loop.counts["invalid"] == 1 and "invalid" in loop.last_error

  def boom(req):
    raise RuntimeError("cuda oom")
  loop, out = _loop(boom)
  loop.process(loop.submit("x", ST))  # must not raise
  assert out == [] and "inference failed" in loop.last_error


def test_timeout_discards_late_answer():
  ticks = iter([0.0, 0.0, 9.0])
  loop, out = _loop(_dec(0.15), timeout_s=5.0, clock=lambda: next(ticks))
  loop.process(loop.submit("x", ST))
  assert out == [] and loop.counts["timeout"] == 1


def test_load_failure_marks_unavailable_and_refuses_requests():
  def bad_load():
    raise LiquidUnavailable("no GPU")
  loop, _ = _loop(_dec(0.1), load=bad_load)
  loop.start()
  deadline = time.monotonic() + 2
  while loop.status != "unavailable" and time.monotonic() < deadline:
    time.sleep(0.01)
  assert loop.status == "unavailable" and "no GPU" in loop.last_error
  assert loop.submit("go", ST) is None
  loop.stop()


# ----------------------------------------- 7./8. stale and superseded commands
def test_result_of_superseded_command_is_not_current():
  loop, out = _loop(_dec(0.15))
  r1 = loop.submit("walk", ST)
  release = threading.Event()
  loop._decide = lambda req: (release.wait(5), V({"speed": 0.15}, rid=req.request_id, gen=req.generation))[1]
  t = threading.Thread(target=loop.process, args=(r1,))
  t.start()
  loop.submit("stop moving", ST)  # newer user command arrives while r1 is in flight
  release.set()
  t.join(5)
  assert out == [] and loop.counts["superseded"] >= 1


def test_pending_request_is_replaced_by_newer_one():
  loop, out = _loop(_dec(0.15))
  r1 = loop.submit("walk", ST)
  loop.submit("turn left", ST)
  assert loop.process(r1) is None  # never started: superseded
  assert loop._pending.text == "turn left"


def test_late_delivery_is_rejected_at_apply_time():
  loop, out = _loop(_dec(0.15))
  loop.process(loop.submit("walk", ST))
  res = out[0]
  loop.supersede()  # e.g. manual STOP pressed after delivery, before the sim applied it
  assert not loop.is_current(res)


def test_older_result_never_overwrites_newer_applied_one():
  loop, out = _loop(_dec(0.15))
  loop.process(loop.submit("a", ST))
  old = out[0]
  loop.mark_applied(old)
  assert not loop.is_current(old)  # same request twice: applied exactly once


# --------------------------------------------------------- 9. non-blocking
def test_sim_side_calls_do_not_block_during_slow_inference():
  release = threading.Event()

  def slow(req):
    release.wait(5)
    return V({"speed": 0.15}, rid=req.request_id, gen=req.generation)

  loop, out = _loop(slow)
  loop.start()
  loop.submit("walk", ST)
  time.sleep(0.05)  # worker is now inside slow()
  t0 = time.monotonic()
  for i in range(2000):  # what the sim thread does per frame: submit / check, never wait
    if out:
      loop.is_current(out[0])
    if i % 500 == 0:
      loop.submit(f"cmd {i}", ST)
  assert time.monotonic() - t0 < 0.5
  release.set()
  deadline = time.monotonic() + 2
  while not out and time.monotonic() < deadline:
    time.sleep(0.01)
  loop.stop()
  assert out and out[-1].request.text == "cmd 1500"  # only the newest command is answered
  assert not loop.running


def test_no_inference_without_a_user_command():
  calls = []
  loop, _ = _loop(lambda req: calls.append(req) or V({}))
  loop.start()
  time.sleep(0.3)
  loop.stop()
  assert calls == []


def test_decision_dataclass_is_immutable():
  d = Decision(1, 1, speed=0.1)
  with pytest.raises(dataclasses.FrozenInstanceError):
    d.speed = 1.0  # type: ignore[misc]


def test_math_constants():
  assert MAX_SPEED_MPS["trot"] == pytest.approx(0.3 / 0.7) and MAX_SPEED_MPS["crawl"] == pytest.approx(0.3 / 1.4)
  assert math.isclose(Limits().yaw_rate_max, 1.0)


# =====================================================================
# Closed-loop mission mode (pure python; model answers are mocked)
# =====================================================================
from contact_rl.runtime.decision_loop import Result  # noqa: E402
from contact_rl.runtime.decision_schema import (  # noqa: E402
  ClosedLoopConfig,
  MissionSpec,
  RobotFeedback,
  validate_mission,
)
from contact_rl.runtime.liquid_decider import (  # noqa: E402
  answers_to_mission,
  answers_to_tick_raw,
  build_mission_questions,
  build_tick_questions,
  make_decide,
)
from contact_rl.runtime.mission import (  # noqa: E402
  ACTIVE,
  COMPLETE,
  FINAL_TURN,
  MissionTracker,
  SafetySupervisor,
)

CL = ClosedLoopConfig()
QM, QT = build_mission_questions(), build_tick_questions()


def _a(**kw):
  return {k: {"choice": v, "confidence": 0.9} for k, v in kw.items()}


def _mission_ans(objective="walk", speed="unspecified", direction="keep", turn="keep", after_arrival="stop",
                 gait="keep", caution="normal"):
  return _a(objective=objective, speed=speed, direction=direction, turn=turn, after_arrival=after_arrival,
            gait=gait, caution=caution)


def _tick_ans(speed="keep", steer="keep", gait="keep", status="continue"):
  return _a(speed=speed, steer=steer, gait=gait, status=status)


def FB(seq=1, t=0.0, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, tilt=0.02, cmd=ST, match=1.0, wz=0.0):
  return RobotFeedback(seq, t, time.monotonic(), 0, 0, x, y, yaw, vx, vy, wz, tilt, (True,) * 4, match, cmd)


def SPEC(objective="walk", v=0.2, target=None, after="stop", cautious=False, gait=None, gait_free=False):
  return MissionSpec(1, "test", objective, v, 0.0, 0.0, gait, gait_free, cautious, after, target)


# ------------------------------------------------ instruction -> persistent mission
def test_mission_questions_follow_decision_index_schema():
  for q in (*QM.values(), *QT.values()):
    assert q["type"] == "choice" and q["instructions"] and len(q["criteria"]) >= 2
  assert {"objective", "speed", "after_arrival", "gait", "caution"} <= set(QM)
  assert set(QT) == {"speed", "steer", "gait", "status"}


@pytest.mark.parametrize("answers,expect", [
  (_mission_ans(speed="slow", direction="forward"), dict(objective="walk", heading_offset=0.0)),
  (_mission_ans("reach_target", after_arrival="turn_left"), dict(objective="reach_target", after_arrival="turn_left",
                                                                yaw_rate=0.0)),
  (_mission_ans(speed="medium"), dict(objective="walk")),
  (_mission_ans(caution="cautious"), dict(cautious=True)),
  (_mission_ans("reach_target", gait="suitable", caution="cautious"), dict(gait_free=True, gait=None, cautious=True)),
])
def test_example_instructions_become_missions(answers, expect):
  raw = answers_to_mission(answers, QM, "x", ST, L, 0.5)
  spec = validate_mission(raw, mission_id=7, limits=L, current=ST)
  for k, v in expect.items():
    assert getattr(spec, k) == v
  assert 0.0 <= spec.preferred_speed <= L.speed_cap(("trot",))


def test_relative_speed_is_resolved_once_at_interpretation():
  raw = answers_to_mission(_mission_ans(speed="faster"), QM, "go faster", ST, L, 0.5)
  assert raw["preferred_speed"] == pytest.approx(ST.speed + 0.05)  # fixed number, not a per-tick delta


def test_cautious_mission_caps_the_desired_speed():
  raw = answers_to_mission(_mission_ans(speed="fast", caution="cautious"), QM, "x", ST, L, 0.5)
  assert raw["preferred_speed"] == pytest.approx(0.6 * L.speed_cap(("trot",)))


@pytest.mark.parametrize("raw", [{"objective": "fly"}, {"objective": "walk", "gait": "gallop"},
                                 {"objective": "walk", "preferred_speed": 9.0}, {"objective": "walk", "exec": 1},
                                 {"objective": "walk", "after_arrival": "dance"}, "walk",
                                 {"objective": "walk", "heading_offset": 1.0, "yaw_rate": 0.5}])
def test_invalid_missions_rejected(raw):
  with pytest.raises(DecisionError):
    validate_mission(raw, mission_id=1, limits=L, current=ST)


# ------------------------------------------------------------- tick mapping
def test_tick_increase_is_bounded_by_the_mission_ceiling():
  raw, _ = answers_to_tick_raw(_tick_ans("increase"), QT, ST, CL, ceiling=0.12, allowed_gaits=("trot",))
  assert raw["speed"] == pytest.approx(0.12)
  d = validate(raw, request_id=1, generation=1, current=ST, limits=L, kind="tick", speed_ceiling=0.12)
  assert d.speed == pytest.approx(0.12)
  with pytest.raises(DecisionError):  # a raw speed above the ceiling never passes validation
    validate({"speed": 0.2}, request_id=1, generation=1, current=ST, limits=L, kind="tick", speed_ceiling=0.12)


def test_repeated_increase_converges_to_the_ceiling_instead_of_compounding():
  st = ST
  for _ in range(20):
    raw, _ = answers_to_tick_raw(_tick_ans("increase"), QT, st, CL, ceiling=0.2, allowed_gaits=("trot",))
    d = validate(raw, request_id=1, generation=1, current=st, limits=L, kind="tick", speed_ceiling=0.2)
    st = dataclasses.replace(st, speed=d.speed)
  assert st.speed == pytest.approx(0.2)


def test_keep_answer_is_a_noop_and_explicit_change_is_not():
  raw, _ = answers_to_tick_raw(_tick_ans(), QT, ST, CL, 0.3, ("trot",))
  assert validate(raw, request_id=1, generation=1, current=ST, limits=L, kind="tick").is_noop
  raw, _ = answers_to_tick_raw(_tick_ans(steer="left"), QT, ST, CL, 0.3, ("trot",))
  d = validate(raw, request_id=1, generation=1, current=ST, limits=L, kind="tick")
  assert not d.is_noop and d.yaw_rate == pytest.approx(CL.turn_rate)


def test_tick_stop_abort_and_gait_rules():
  raw, _ = answers_to_tick_raw(_tick_ans("stop"), QT, ST, CL, 0.3, ("trot",))
  assert raw["stop"] is True
  raw, _ = answers_to_tick_raw(_tick_ans(status="abort"), QT, ST, CL, 0.3, ("trot",))
  assert raw["stop"] is True and raw["mission_status"] == "abort"
  with pytest.raises(DecisionError):  # gait not allowed by the mission
    answers_to_tick_raw(_tick_ans(gait="bound"), QT, ST, CL, 0.3, ("trot",))
  raw, _ = answers_to_tick_raw(_tick_ans(gait="crawl"), QT, ST, CL, 0.3, ("trot", "crawl"))
  assert raw["gait"] == "crawl"


def test_tick_validation_is_strict():
  with pytest.raises(DecisionError):
    validate({"mission_status": "done"}, request_id=1, generation=1, kind="tick")
  with pytest.raises(DecisionError):  # mission_status is only accepted from ticks
    validate({"mission_status": "complete"}, request_id=1, generation=1)
  with pytest.raises(DecisionError):
    answers_to_tick_raw(_a(speed="sprint", steer="keep", gait="keep", status="continue"), QT, ST, CL, 0.3, ("trot",))
  d = validate({"mission_status": "complete"}, request_id=3, generation=2, kind="tick", mission_id=5, snapshot_seq=9)
  assert (d.kind, d.mission_id, d.snapshot_seq, d.mission_status) == ("tick", 5, 9, "complete")


def test_yaw_deescalation_is_immediate_and_result_stays_inside_the_cap():
  sampled = PlannerState("trot", 0.1, 0.0, 1.94, ("trot",))  # training-sampled yaw above the runtime cap
  assert V({"yaw_rate": 0.0}, current=sampled).yaw_rate == 0.0
  d = V({"yaw_rate": 1.0}, current=sampled)
  assert abs(d.yaw_rate) <= L.yaw_rate_max
  d = V({"yaw_rate": -1.0}, current=sampled)  # reversal is rate-limited, then capped
  assert d.yaw_rate == pytest.approx(1.0) and abs(d.yaw_rate) <= L.yaw_rate_max


def test_model_answers_change_the_decision():
  """The model genuinely contributes: different answers over the same state -> different commands."""
  ctx = {"mission": {}, "_ceiling": 0.3, "_allowed_gaits": ("trot",)}
  outs = []
  for ans in (_tick_ans("increase", "left"), _tick_ans("decrease", "right"), _tick_ans()):
    decide = make_decide(lambda s, q, a=ans: a, L, 0.5, CL)
    outs.append(decide(Request(1, 1, "x", ST, 0.0, kind="tick", mission_id=1, snapshot_seq=1, context=ctx)))
  assert outs[0].speed > ST.speed and outs[0].yaw_rate > 0
  assert outs[1].speed < ST.speed and outs[1].yaw_rate < 0
  assert outs[2].is_noop


def test_tick_request_sends_the_feedback_context_to_system_one():
  seen = []
  ctx = {"mission": {"phase": "approach"}, "measured state": {"body tilt": "3.0 deg"}, "_ceiling": 0.3,
         "_allowed_gaits": ("trot",)}
  decide = make_decide(lambda s, q: seen.append((s, sorted(q))) or _tick_ans(), L, 0.5, CL)
  decide(Request(1, 1, "x", ST, 0.0, kind="tick", mission_id=1, snapshot_seq=1, context=ctx))
  state, names = seen[0]
  assert names == ["gait", "speed", "status", "steer"]
  assert state["measured state"]["body tilt"] == "3.0 deg" and "_ceiling" not in state


# ------------------------------------------------------------ config / timing
def test_closed_loop_config_validation_and_latency_bounded_period():
  with pytest.raises(ValueError):
    ClosedLoopConfig(decision_hz=50.0)
  with pytest.raises(ValueError):
    ClosedLoopConfig(tilt_abort_rad=math.radians(85))
  assert CL.decision_period(None) == pytest.approx(1.0)
  assert CL.decision_period(0.8) == pytest.approx(1.6)  # never faster than 2 x measured latency


def test_feedback_finiteness():
  assert FB().is_finite and not FB(vx=float("nan")).is_finite and not FB(match=float("inf")).is_finite


# --------------------------------------------------- decision loop: ticks
def test_tick_does_not_bump_generation_and_is_bounded():
  loop, out = _loop(lambda req: V({}, rid=req.request_id, gen=req.generation))
  mreq = loop.submit("walk", ST, kind="mission")
  assert loop.submit_tick("walk", ST, mission_id=1, snapshot_seq=0, context={}) is None  # mission pending
  loop.process(mreq)
  g = mreq.generation
  t1 = loop.submit_tick("walk", ST, mission_id=g, snapshot_seq=1, context={})
  assert t1 is not None and loop.generation == g
  assert loop.submit_tick("walk", ST, mission_id=g, snapshot_seq=2, context={}) is None  # slot busy
  assert loop.counts["skipped_busy"] == 2


def test_out_of_order_snapshot_is_not_current():
  loop, out = _loop(lambda req: V({}, rid=req.request_id, gen=req.generation))
  loop._pending = None
  newer = Result(Request(5, 0, "", ST, 0.0, kind="tick", snapshot_seq=10), Decision(5, 0), 0.0)
  older = Result(Request(6, 0, "", ST, 0.0, kind="tick", snapshot_seq=9), Decision(6, 0), 0.0)
  assert loop.is_current(newer)
  loop.mark_applied(newer)
  assert not loop.is_current(older)  # higher request id, but computed from an older snapshot


def test_deadline_is_counted_once_and_late_answer_discarded():
  release = threading.Event()
  loop, out = _loop(lambda req: (release.wait(5), V({}, rid=req.request_id, gen=req.generation))[1], timeout_s=0.05)
  loop.start()
  loop.submit_tick("x", ST, mission_id=0, snapshot_seq=1, context={})
  deadline = time.monotonic() + 2
  while loop.inflight_age() is None and time.monotonic() < deadline:
    time.sleep(0.005)
  time.sleep(0.08)
  assert loop.flag_deadline() and not loop.flag_deadline()
  release.set()
  time.sleep(0.2)
  loop.stop()
  assert out == [] and loop.counts["timeout"] == 1 and loop.consecutive_failures == 1


def test_consecutive_failures_reset_after_a_delivered_answer():
  answers = iter([RuntimeError("x"), RuntimeError("y"), None])

  def decide(req):
    a = next(answers)
    if a is not None:
      raise a
    return V({}, rid=req.request_id, gen=req.generation)
  loop, out = _loop(decide)
  for i in range(2):
    loop.process(loop.submit_tick("x", ST, mission_id=0, snapshot_seq=i, context={}))
  assert loop.consecutive_failures == 2
  loop.process(loop.submit_tick("x", ST, mission_id=0, snapshot_seq=5, context={}))
  assert loop.consecutive_failures == 0 and len(out) == 1


# ------------------------------------------------------------ mission tracker
def test_context_uses_measured_values_and_reports_previous_outcome():
  m = MissionTracker(SPEC("reach_target", target=(2.0, 1.0)), FB(), CL)
  m.record(Decision(1, 1, speed=0.2), FB(t=0.0), applied=True)
  ctx = m.context(FB(seq=2, t=1.0, x=0.5, vx=0.18, tilt=math.radians(4)), ("trot",), ())
  assert ctx["measured state"]["forward velocity"] == "+0.18 m/s"
  assert ctx["measured state"]["body tilt"] == "4.0 deg"
  assert ctx["target"]["distance"] == f"{math.hypot(1.5, 1.0):.2f} m" and "left" in ctx["target"]["direction"]
  assert ctx["previous decision"]["result"].startswith("distance decreased")
  assert ctx["mission"]["user instruction"] == "test"


def test_context_without_target_says_so_instead_of_inventing_one():
  m = MissionTracker(SPEC("walk"), FB(), CL)
  ctx = m.context(FB(), ("trot",), ())
  assert "target" not in ctx
  m2 = MissionTracker(SPEC("reach_target"), FB(), CL)
  assert m2.context(FB(), ("trot",), ())["target"] == "not configured"


def test_arrival_final_turn_and_measured_completion():
  m = MissionTracker(SPEC("reach_target", target=(1.0, 0.0), after="turn_left"), FB(), CL)
  assert m.phase_update(FB(x=0.5)) is None and m.state == ACTIVE
  assert m.phase_update(FB(x=0.9, yaw=0.1)) == "arrived" and m.phase == FINAL_TURN and m.speed_ceiling() == 0.0
  assert m.phase_update(FB(x=0.9, yaw=0.1 + math.radians(50))) is None
  assert m.phase_update(FB(x=0.9, yaw=0.1 + math.radians(80))) == "complete" and m.state == COMPLETE


def test_reach_and_stop_completes_on_tolerance_only():
  m = MissionTracker(SPEC("reach_target", target=(1.0, 0.0)), FB(), CL)
  assert m.phase_update(FB(t=500.0, x=0.0)) is None  # elapsed time alone never completes a mission
  assert m.phase_update(FB(x=1.0 - CL.arrive_tol_m + 0.01)) == "complete"


def test_walk_missions_never_complete_on_their_own():
  m = MissionTracker(SPEC("walk"), FB(), CL)
  assert not m.evidence_complete(FB(t=1e4, x=100.0))


# ------------------------------------------------------------- safety
SV = SafetySupervisor(CL)


def test_safety_non_finite_and_tilt_abort_always_win():
  m = MissionTracker(SPEC(), FB(), CL)
  a = SV.check(m, FB(vx=float("nan")), 0, 0.0)
  assert a.stop and a.fail_mission
  a = SV.check(None, FB(tilt=math.radians(50)), 0, None)  # even without a mission
  assert a.stop


def test_safety_tilt_caution_reduces_speed_and_is_tighter_when_cautious():
  cmd = dataclasses.replace(ST, speed=0.2)
  a = SV.check(MissionTracker(SPEC(), FB(), CL), FB(cmd=cmd, tilt=math.radians(22), vx=0.2), 0, 0.0)
  assert not a.stop and a.speed == pytest.approx(0.1)
  assert SV.check(MissionTracker(SPEC(), FB(), CL), FB(cmd=cmd, tilt=math.radians(15), vx=0.2), 0, 0.0) is None
  a = SV.check(MissionTracker(SPEC(cautious=True), FB(), CL), FB(cmd=cmd, tilt=math.radians(15), vx=0.2), 0, 0.0)
  assert a.speed == pytest.approx(0.1)


def test_safety_inference_failures_and_missing_decisions_stop_the_robot():
  m = MissionTracker(SPEC(), FB(), CL)
  assert SV.check(m, FB(), CL.max_consecutive_failures, 0.0).stop
  assert SV.check(m, FB(), 0, CL.fallback_hold_s + 0.1).stop
  assert SV.check(m, FB(), CL.max_consecutive_failures - 1, CL.fallback_hold_s - 0.1) is None


def test_safety_no_progress_toward_target():
  m = MissionTracker(SPEC("reach_target", target=(3.0, 0.0)), FB(), CL)
  assert SV.check(m, FB(t=0.0), 0, 0.0) is None
  assert SV.check(m, FB(t=5.0, x=-0.3), 0, 0.0) is None  # first turns away: tolerated
  assert SV.check(m, FB(t=7.0, x=0.2), 0, 0.0) is None  # improved -> window restarts
  a = SV.check(m, FB(t=7.0 + CL.no_progress_s, x=0.25), 0, 0.0)
  assert a.stop and "no progress" in a.reason


def test_safety_no_progress_walk_mission():
  cmd = dataclasses.replace(ST, speed=0.2)
  m = MissionTracker(SPEC(), FB(), CL)
  assert SV.check(m, FB(t=0.0, cmd=cmd, vx=0.0), 0, 0.0) is None
  assert SV.check(m, FB(t=CL.no_progress_s + 0.1, cmd=cmd, vx=0.01), 0, 0.0).stop
  m = MissionTracker(SPEC(), FB(), CL)
  SV.check(m, FB(t=0.0, cmd=cmd, vx=0.0), 0, 0.0)
  assert SV.check(m, FB(t=CL.no_progress_s + 0.1, cmd=cmd, vx=0.15), 0, 0.0) is None
