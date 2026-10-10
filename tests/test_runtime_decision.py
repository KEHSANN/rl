"""Runtime decision layer: validation, scheduling, stale rejection, fail-safe.
Pure python, no torch / GPU / model download needed."""

from __future__ import annotations

import math
import threading
import time

import pytest

from contact_rl.runtime.decision_loop import DecisionLoop, DecisionRecord
from contact_rl.runtime.decision_schema import (
  MAX_SPEED_MPS,
  Decision,
  DecisionError,
  validate,
  to_user_command_kwargs,
)
from contact_rl.runtime.liquid_decider import answers_to_raw, build_questions, build_state


# ---------------------------------------------------------- validation
def test_valid_decision_passes():
  d = validate({"gait": "trot", "speed": 0.2, "yaw_rate": 0.0}, 1)
  assert d.gait == "trot" and d.speed == 0.2 and d.command_id == 1


@pytest.mark.parametrize("raw", [
  {"speed": -0.1}, {"speed": 0.5}, {"speed": float("nan")}, {"speed": "fast"}, {"speed": True},
])
def test_invalid_speed_rejected(raw):
  with pytest.raises(DecisionError):
    validate(raw, 1)


def test_invalid_heading_rejected():
  with pytest.raises(DecisionError):
    validate({"heading_offset": 4.0}, 1)


def test_invalid_yaw_rate_rejected():
  with pytest.raises(DecisionError):
    validate({"yaw_rate": 4.0}, 1)


def test_unknown_gait_rejected():
  with pytest.raises(DecisionError):
    validate({"gait": "gallop"}, 1)


def test_heading_and_yaw_together_rejected():
  with pytest.raises(DecisionError):
    validate({"heading_offset": 0.5, "yaw_rate": 0.5}, 1)


def test_non_object_rejected():
  with pytest.raises(DecisionError):
    validate(["trot"], 1)


def test_speed_capped_to_gait_range_not_extrapolated():
  d = validate({"gait": "trot", "speed": 0.43}, 1)
  assert d.speed == pytest.approx(MAX_SPEED_MPS["trot"])


def test_per_step_change_is_bounded():
  prev = Decision(command_id=1, speed=0.1, yaw_rate=0.0)
  d = validate({"speed": 0.4, "yaw_rate": 3.0}, 2, prev)
  assert d.speed == pytest.approx(0.2)
  assert d.yaw_rate == pytest.approx(0.5)


def test_explanation_is_truncated():
  d = validate({"explanation": "x" * 1000}, 1)
  assert len(d.explanation) == 200


def test_to_user_command_kwargs_only_set_fields():
  d = validate({"speed": 0.2}, 1)
  assert to_user_command_kwargs(d) == {"speed": 0.2}


# ------------------------------------------------------ answer mapping
def test_answers_map_to_raw_without_inventing_values():
  answers = {"speed_action": {"choice": "increase"}, "turn_action": {"choice": "left"},
             "gait_action": {"choice": "keep"}}
  raw = answers_to_raw(answers, {"increase": 0.3, "decrease": 0.1, "hold": 0.2})
  assert raw == {"speed": 0.3, "yaw_rate": 0.5}


def test_questions_follow_schema():
  for q in build_questions().values():
    assert q["type"] == "choice" and q["instructions"] and len(q["criteria"]) >= 2


def test_state_omits_missing_fields():
  s = build_state("move forward", {"speed": 0.2, "target": None})
  assert "target" not in s and "speed: 0.2" in s


# ------------------------------------------------------- scheduler
def _loop(infer, **kw):
  return DecisionLoop(infer, period_s=0.01, **kw)


def test_valid_inference_is_applied_once():
  loop = _loop(lambda cid: {"speed": 0.2})
  loop.step_once()
  applied = []
  assert loop.poll_and_apply(applied.append) is True
  assert loop.poll_and_apply(applied.append) is False
  assert len(applied) == 1 and applied[0].speed == 0.2


def test_latency_is_recorded():
  ticks = iter([0.0, 0.25, 0.25])
  loop = DecisionLoop(lambda cid: {"speed": 0.1}, clock=lambda: next(ticks))
  loop.step_once()
  assert loop.applied[-1].latency_s == pytest.approx(0.25)


def test_invalid_output_keeps_last_valid_command():
  outputs = iter([{"speed": 0.2}, {"speed": 9.0}])
  loop = _loop(lambda cid: next(outputs))
  loop.step_once()
  loop.step_once()
  assert loop.latest().speed == 0.2
  assert loop.rejected_invalid == 1
  assert "invalid" in loop.last_error


def test_model_exception_does_not_propagate():
  def boom(cid):
    raise RuntimeError("cuda oom")
  loop = _loop(boom)
  loop.step_once()  # must not raise
  assert loop.latest() is None
  assert "inference failed" in loop.last_error


def test_stale_result_never_overwrites_newer():
  loop = _loop(lambda cid: {"speed": 0.1})
  loop._offer(DecisionRecord(5, validate({"speed": 0.3}, 5), 0.0, 0.0))
  loop._offer(DecisionRecord(3, validate({"speed": 0.1}, 3), 0.0, 0.0))
  assert loop.latest().speed == 0.3
  assert loop.rejected_stale == 1


def test_control_loop_not_blocked_during_slow_inference():
  release = threading.Event()

  def slow(cid):
    release.wait(5)
    return {"speed": 0.2}

  loop = _loop(slow)
  t = threading.Thread(target=loop.step_once, daemon=True)
  t.start()
  t0 = time.monotonic()
  for _ in range(1000):  # simulated physics-side calls while inference is pending
    loop.latest()
    loop.poll_and_apply(lambda d: None)
  assert time.monotonic() - t0 < 1.0
  release.set()
  t.join(5)
  assert loop.latest().speed == 0.2


def test_worker_thread_applies_and_stops():
  loop = _loop(lambda cid: {"speed": 0.15})
  loop.start()
  deadline = time.monotonic() + 2
  while loop.latest() is None and time.monotonic() < deadline:
    time.sleep(0.01)
  loop.stop()
  assert loop.latest().speed == 0.15


def test_commands_update_without_restart():
  seq = iter([{"speed": 0.1}, {"speed": 0.2}, {"speed": 0.3}])
  loop = _loop(lambda cid: next(seq))
  seen = []
  for _ in range(3):
    loop.step_once()
    loop.poll_and_apply(lambda d: seen.append(d.speed))
  assert seen == [0.1, 0.2, 0.3]
