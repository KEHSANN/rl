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
