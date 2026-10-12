"""Adapter for LiquidAI/d1-3B-w8a8 (a decision model, not a chat model).

Verified against the official model card (huggingface.co/LiquidAI/d1-3B-w8a8):

* load: ``AutoModel.from_pretrained(id, trust_remote_code=True).to("cuda")``,
  then ``model.compile(mode="reduce-overhead")`` (torchao INT8 kernels are only
  fast compiled);
* call: ``model.system_one(state, questions)`` where ``state`` is a string or
  JSON value and ``questions`` follow the Decision Index schema
  (``type`` / ``instructions`` / ``criteria``);
* answer: ``{"answers": {name: {"choice", "confidence", "probabilities"}},
  "usage": {...}}`` for ``choice`` questions, zero output tokens. Choice
  answers are argmax over named options, so inference is deterministic for a
  given input; there is no sampling to configure;
* requirements: NVIDIA GPU with INT8 tensor cores (Ampere, sm_80, or newer),
  PyTorch, ``transformers>=5.19``, ``torchao>=0.18``, ``pillow``. On other
  hardware the card points to LiquidAI/d1-3B; this adapter never switches
  model on its own.

Heavy imports (torch, transformers) happen only inside :meth:`LiquidDecider.load`.
The model's answers are only *labels*; every number that reaches the planner is
computed here from the planner state and then checked by
:func:`decision_schema.validate`. Confidence values are logged, never trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import time

from contact_rl.runtime.decision_schema import (
  DEFAULT_LIMITS,
  ClosedLoopConfig,
  Decision,
  DecisionError,
  Limits,
  MissionSpec,
  PlannerState,
  validate,
  validate_mission,
)

MODEL_ID = "LiquidAI/d1-3B-w8a8"
INSTALL_HINT = 'uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow ...'

SPEED_LEVELS = {"slow": 0.3, "medium": 0.6, "fast": 1.0}  # fractions of the validated speed cap
SPEED_DELTA = 0.05  # m/s for "faster" / "slower"
DIRECTIONS = {"forward": 0.0, "backward": math.pi, "left": math.pi / 2, "right": -math.pi / 2}


class LiquidUnavailable(RuntimeError):
  """The model cannot be loaded or run in this environment (with a diagnostic)."""


def build_questions(gaits=DEFAULT_LIMITS.gaits) -> dict:
  """Questions in the Decision Index schema from the model card."""
  return {
    "speed": {
      "type": "choice",
      "instructions": "What walking speed does the user's command ask for?",
      "criteria": {
        "keep": "The command does not mention speed or stopping",
        "stop": "Stop moving, halt, stand still",
        "slow": "Walk slowly, carefully",
        "medium": "Walk at a normal pace",
        "fast": "Walk fast, run, hurry",
        "faster": "Speed up a little compared to now",
        "slower": "Slow down a little compared to now",
      },
    },
    "turn": {
      "type": "choice",
      "instructions": "Should the robot keep rotating its body while walking?",
      "criteria": {
        "keep": "The command does not mention turning",
        "straight": "Stop turning, go straight, keep the current heading",
        "left": "Turn left, rotate counter-clockwise",
        "right": "Turn right, rotate clockwise",
      },
    },
    "direction": {
      "type": "choice",
      "instructions": "In which direction relative to its body should the robot walk, without rotating?",
      "criteria": {
        "keep": "The command does not ask for a new walking direction",
        "forward": "Walk forward",
        "backward": "Walk backward, reverse",
        "left": "Step sideways to the left without turning",
        "right": "Step sideways to the right without turning",
      },
    },
    "gait": {
      "type": "choice",
      "instructions": "Which gait does the command ask for?",
      "criteria": {"keep": "The command does not name a gait"} | {g: f"The {g} gait" for g in gaits},
    },
  }


def build_state(command: str, state: PlannerState) -> str:
  """Text state: the user command plus the current planner command (real values only)."""
  return "\n".join([
    f"User command: {command}",
    f"Current gait: {state.gait}",
    f"Current speed: {state.speed:.2f} m/s",
    f"Current walking direction offset: {math.degrees(state.heading_offset):.0f} deg",
    f"Current turning rate: {state.yaw_rate:.2f} rad/s",
  ])


def _choice(answers: dict, questions: dict, name: str) -> str:
  a = answers.get(name)
  if not isinstance(a, dict) or "choice" not in a:
    raise DecisionError(f"malformed answer for {name!r}: {a!r}")
  c = a["choice"]
  if not isinstance(c, str) or c not in questions[name]["criteria"]:
    raise DecisionError(f"answer {name}={c!r} is not one of {list(questions[name]['criteria'])}")
  return c


def answers_to_raw(answers, questions: dict, current: PlannerState, limits: Limits, yaw_rate: float) -> dict:
  """Map choice labels to a raw decision dict (validated afterwards).

  Absolute speeds are fractions of the validated cap for the target gait;
  relative ones move the *current* planner speed by ``SPEED_DELTA``.
  """
  if not isinstance(answers, dict):
    raise DecisionError("answers must be an object")
  ch = {n: _choice(answers, questions, n) for n in questions}
  conf = {n: answers[n].get("confidence") for n in questions}
  reason = " ".join(f"{n}={c}" for n, c in ch.items()) + " | conf " + json.dumps(conf, default=str)
  if ch["speed"] == "stop":
    return {"stop": True, "reason": reason}
  raw: dict = {"reason": reason}
  gait = None if ch["gait"] == "keep" else ch["gait"]
  if gait is not None:
    raw["gait"] = gait
  cap = limits.speed_cap((gait,) if gait else current.target_gaits)
  sp = ch["speed"]
  if sp in SPEED_LEVELS:
    raw["speed"] = SPEED_LEVELS[sp] * cap
  elif sp == "faster":
    raw["speed"] = min(current.speed + SPEED_DELTA, cap)
  elif sp == "slower":
    # Also capped: a sampled training command may be above the runtime cap, and
    # "slower" must then slow down to the cap instead of being rejected.
    raw["speed"] = min(max(current.speed - SPEED_DELTA, 0.0), cap)
  t = ch["turn"]
  if t == "left":
    raw["yaw_rate"] = abs(yaw_rate)
  elif t == "right":
    raw["yaw_rate"] = -abs(yaw_rate)
  elif t == "straight":
    raw["yaw_rate"] = 0.0
  if ch["direction"] in DIRECTIONS:
    raw["heading_offset"] = DIRECTIONS[ch["direction"]]
  return raw


# ------------------------------------------------- closed loop: mission interpretation
#
# Asked ONCE per instruction. The answers become a persistent MissionSpec; the
# instruction is never re-interpreted on later ticks.

MISSION_SPEED = {"slow": 0.3, "medium": 0.6, "fast": 1.0}  # fractions of the validated cap


def build_mission_questions(gaits=DEFAULT_LIMITS.gaits) -> dict:
  return {
    "objective": {
      "type": "choice",
      "instructions": "What is the overall goal of the user's instruction?",
      "criteria": {
        "walk": "Walk, move, turn or keep a speed, with no destination or target",
        "reach_target": "Go to, reach, approach or move toward the target, goal or waypoint",
        "stop": "Stop, halt, stand still, do nothing",
      },
    },
    "speed": {
      "type": "choice",
      "instructions": "What speed does the instruction ask for?",
      "criteria": {
        "unspecified": "No speed is mentioned",
        "slow": "Slowly, carefully, cautiously, gently",
        "medium": "Normal, moderate or medium pace",
        "fast": "Fast, quickly, hurry, run",
        "faster": "Faster than now",
        "slower": "Slower than now",
      },
    },
    "direction": {
      "type": "choice",
      "instructions": "In which direction relative to its body should the robot walk?",
      "criteria": {
        "keep": "No walking direction is mentioned, or the robot should head to a target",
        "forward": "Forward, ahead",
        "backward": "Backward, reverse, back up",
        "left": "Sideways to the left without turning",
        "right": "Sideways to the right without turning",
      },
    },
    "turn": {
      "type": "choice",
      "instructions": "Should the robot keep rotating while it walks (not counting a turn at the target)?",
      "criteria": {
        "keep": "Turning is not mentioned, or the turn happens only at the target or waypoint",
        "straight": "Go straight, stop turning",
        "left": "Keep turning left while walking",
        "right": "Keep turning right while walking",
      },
    },
    "after_arrival": {
      "type": "choice",
      "instructions": "What should the robot do once it reaches the target or waypoint?",
      "criteria": {
        "stop": "Stop there, or nothing is said about the target",
        "turn_left": "Turn left at or after the target or waypoint",
        "turn_right": "Turn right at or after the target or waypoint",
      },
    },
    "gait": {
      "type": "choice",
      "instructions": "Which gait does the instruction ask for?",
      "criteria": {"keep": "No gait is mentioned",
                   "suitable": "Use a suitable, appropriate or best gait (the robot chooses)"}
      | {g: f"The {g} gait" for g in gaits},
    },
    "caution": {
      "type": "choice",
      "instructions": "Does the instruction ask the robot to be careful about stability?",
      "criteria": {
        "normal": "No mention of care, caution or stability",
        "cautious": "Be careful or cautious, or slow down if the robot becomes unstable",
      },
    },
  }


def answers_to_mission(answers, questions: dict, text: str, current: PlannerState, limits: Limits,
                       turn_rate: float) -> dict:
  """Mission labels -> raw mission dict (validated afterwards by ``validate_mission``).

  Relative speed words are resolved ONCE against the speed at interpretation
  time, so they cannot compound on later ticks.
  """
  if not isinstance(answers, dict):
    raise DecisionError("answers must be an object")
  ch = {n: _choice(answers, questions, n) for n in questions}
  conf = {n: answers[n].get("confidence") for n in questions}
  gait = ch["gait"] if ch["gait"] not in ("keep", "suitable") else None
  cap = limits.speed_cap((gait,) if gait else current.target_gaits)
  sp = ch["speed"]
  if ch["objective"] == "stop":
    speed = 0.0
  elif sp in MISSION_SPEED:
    speed = MISSION_SPEED[sp] * cap
  elif sp == "faster":
    speed = min(current.speed + SPEED_DELTA, cap)
  elif sp == "slower":
    speed = min(max(current.speed - SPEED_DELTA, 0.0), cap)
  else:  # unspecified: keep the current speed if moving, otherwise a slow start
    speed = min(current.speed, cap) if current.speed > 0.02 else MISSION_SPEED["slow"] * cap
  if ch["caution"] == "cautious":
    speed = min(speed, MISSION_SPEED["medium"] * cap)
  raw = {"text": text, "objective": ch["objective"], "preferred_speed": speed, "gait": gait,
         "gait_free": ch["gait"] == "suitable", "cautious": ch["caution"] == "cautious",
         "after_arrival": ch["after_arrival"] if ch["objective"] == "reach_target" else "stop",
         "reason": " ".join(f"{n}={c}" for n, c in ch.items()) + " | conf " + json.dumps(conf, default=str)}
  if ch["objective"] == "reach_target":
    raw["heading_offset"] = 0.0  # face the target and steer with the yaw rate
  elif ch["direction"] in DIRECTIONS:
    raw["heading_offset"] = DIRECTIONS[ch["direction"]]
  t = ch["turn"]
  if ch["objective"] != "reach_target" and t in ("left", "right"):
    raw["yaw_rate"] = abs(turn_rate) if t == "left" else -abs(turn_rate)
    if raw.get("heading_offset"):
      raw["heading_offset"] = 0.0  # exclusive in training; turning wins (validate would reject both)
  elif t == "straight" or ch["objective"] == "reach_target":
    raw["yaw_rate"] = 0.0
  if ch["objective"] == "stop":
    raw["yaw_rate"] = 0.0
  return raw


# ------------------------------------------------------- closed loop: decision tick
#
# Asked periodically while a mission is active, over a fresh state snapshot
# (MissionTracker.context). Answers are actions relative to the CURRENT
# command, bounded by the mission's preferred speed.

GAIT_HINTS = {
  "trot": "diagonal pairs, the standard all-round gait",
  "pace": "lateral pairs",
  "bound": "front and hind pairs",
  "jump": "all four feet together, with a flight phase",
  "crawl": "one foot at a time, the most statically stable and slowest gait",
}


def build_tick_questions(gaits=DEFAULT_LIMITS.gaits) -> dict:
  return {
    "speed": {
      "type": "choice",
      "instructions": "Given the mission, the measured state and the previous decision, how should the speed change now?",
      "criteria": {
        "keep": "The speed is right for the mission and the robot is stable and tracking well",
        "increase": "Slower than the desired speed while stable and tracking well, and not near the target",
        "decrease": "Faster than wanted, unstable or tilting, tracking poorly, or near the target",
        "stop": "The robot must stop moving now",
      },
    },
    "steer": {
      "type": "choice",
      "instructions": "How should the robot steer now?",
      "criteria": {
        "keep": "The current turning is right for the mission",
        "straight": "Go straight: the target is roughly straight ahead, or the mission says go straight",
        "left": "Turn left: the target is to the left, or the mission asks to turn left",
        "right": "Turn right: the target is to the right, or the mission asks to turn right",
      },
    },
    "gait": {
      "type": "choice",
      "instructions": "Which gait should the robot use now? Only gaits listed as allowed may be chosen.",
      "criteria": {"keep": "Keep the current gait"} | {g: f"Switch to {g}: {GAIT_HINTS[g]}" for g in gaits},
    },
    "status": {
      "type": "choice",
      "instructions": "What is the status of the mission according to the measurements?",
      "criteria": {
        "continue": "The mission is still in progress",
        "complete": "The measurements show the mission's goal is achieved",
        "abort": "The robot cannot safely continue the mission",
      },
    },
  }


def answers_to_tick_raw(answers, questions: dict, current: PlannerState, cfg: ClosedLoopConfig,
                        ceiling: float, allowed_gaits: tuple[str, ...],
                        limits: Limits = DEFAULT_LIMITS) -> tuple[dict, dict]:
  """Tick labels -> raw decision dict (validated afterwards), plus the raw choices.

  A gait change also lowers the ceiling to the new gait's validated cap, so the
  mapped speed is always inside the range ``validate`` accepts."""
  if not isinstance(answers, dict):
    raise DecisionError("answers must be an object")
  ch = {n: _choice(answers, questions, n) for n in questions}
  conf = {n: answers[n].get("confidence") for n in questions}
  raw: dict = {"reason": " ".join(f"{n}={c}" for n, c in ch.items()) + " | conf " + json.dumps(conf, default=str),
               "mission_status": ch["status"]}
  if ch["speed"] == "stop" or ch["status"] == "abort":
    raw["stop"] = True
    return raw, ch
  if ch["gait"] != "keep":
    if ch["gait"] not in allowed_gaits:
      raise DecisionError(f"model chose gait {ch['gait']!r}, not allowed for this mission {allowed_gaits}")
    if ch["gait"] != current.gait:
      raw["gait"] = ch["gait"]
      ceiling = min(ceiling, limits.speed_cap((ch["gait"],)))
  if ch["speed"] == "increase":
    raw["speed"] = min(current.speed + cfg.speed_step, ceiling)
  elif ch["speed"] == "decrease":
    raw["speed"] = max(min(current.speed, ceiling) - cfg.speed_step, 0.0)
  elif current.speed > ceiling + 1e-9:  # "keep" above the mission ceiling (e.g. after arrival)
    raw["speed"] = ceiling
  if ch["steer"] == "left":
    raw["yaw_rate"] = abs(cfg.turn_rate)
  elif ch["steer"] == "right":
    raw["yaw_rate"] = -abs(cfg.turn_rate)
  elif ch["steer"] == "straight":
    raw["yaw_rate"] = 0.0
  return raw, ch


def build_tick_state(context: dict) -> dict:
  """The JSON state passed to ``system_one`` (rendered by the model's own state formatter)."""
  return context


class LiquidDecider:
  """Loads the model once (lazily) and turns text commands into validated decisions."""

  def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", compile: bool = True):
    self.model_id = model_id
    self.device = device
    self.compile = compile
    self.model = None
    self.load_report: dict = {}

  def load(self) -> dict:
    if self.model is not None:
      return self.load_report
    t0 = time.perf_counter()
    try:
      import torch
    except ImportError as e:
      raise LiquidUnavailable(f"torch is not importable ({e})") from e
    if self.device.startswith("cuda"):
      if not torch.cuda.is_available():
        raise LiquidUnavailable(f"{self.model_id} needs CUDA (INT8 tensor cores); torch.cuda.is_available() is False")
      cap = torch.cuda.get_device_capability(torch.device(self.device))
      if cap < (8, 0):
        raise LiquidUnavailable(
          f"{self.model_id} needs an Ampere+ GPU (sm_80+, INT8 tensor cores); this GPU is sm_{cap[0]}{cap[1]}. "
          "The model card recommends LiquidAI/d1-3B on such hardware (pass --liquid-model LiquidAI/d1-3B); "
          "no model is substituted automatically."
        )
      torch.cuda.reset_peak_memory_stats(torch.device(self.device))
      mem0 = torch.cuda.memory_allocated(torch.device(self.device))
    try:
      import torchao  # noqa: F401  (needed by the w8a8 checkpoint)
      from transformers import AutoModel
    except ImportError as e:
      raise LiquidUnavailable(f"missing optional dependency ({e}). Run with: {INSTALL_HINT}") from e
    try:
      model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
      model.eval()
    except Exception as e:  # noqa: BLE001
      raise LiquidUnavailable(f"loading {self.model_id} failed: {type(e).__name__}: {e}") from e
    if not callable(getattr(model, "system_one", None)):
      raise LiquidUnavailable(f"{self.model_id} has no system_one(); not a d1 decision model")
    if self.compile:
      try:
        model.compile(mode="reduce-overhead")
      except Exception as e:  # noqa: BLE001
        raise LiquidUnavailable(f"model.compile failed: {type(e).__name__}: {e}") from e
    self.model = model
    rep = {"model_id": self.model_id, "device": self.device, "load_s": time.perf_counter() - t0}
    if self.device.startswith("cuda"):
      d = torch.device(self.device)
      rep["vram_model_mb"] = (torch.cuda.memory_allocated(d) - mem0) / 2**20
      rep["vram_peak_mb"] = torch.cuda.max_memory_allocated(d) / 2**20
    try:
      import resource

      rep["cpu_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001
      pass
    self.load_report = rep
    return rep

  def answer(self, command: str, state: PlannerState, questions: dict) -> dict:
    return self.ask(build_state(command, state), questions)

  def ask(self, state, questions: dict) -> dict:
    """One ``system_one`` call (one forward pass); returns ``answers``."""
    if self.model is None:
      raise LiquidUnavailable("model not loaded")
    import torch

    with torch.inference_mode():
      out = self.model.system_one(state, questions)
    if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
      raise DecisionError(f"unexpected system_one output: {type(out).__name__}")
    return out["answers"]

  def make_decide(self, limits: Limits, yaw_rate: float, cfg: ClosedLoopConfig | None = None):
    """``decide(request)`` for :class:`DecisionLoop` (runs on the worker thread).

    ``req.kind`` selects the question set: ``instruction`` (one-shot command,
    -> Decision), ``mission`` (interpret the instruction once, -> MissionSpec)
    or ``tick`` (closed-loop decision from a feedback snapshot, -> Decision).
    """
    return make_decide(self.ask, limits, yaw_rate, cfg or ClosedLoopConfig(turn_rate=abs(yaw_rate) or 0.5))


def make_decide(ask, limits: Limits, yaw_rate: float, cfg: ClosedLoopConfig):
  """Request dispatcher around ``ask(state, questions) -> answers`` (the real model or a mock)."""
  q_one = build_questions(limits.gaits)
  q_mission = build_mission_questions(limits.gaits)
  q_tick = build_tick_questions(limits.gaits)

  def decide(req) -> Decision | MissionSpec:
    kind = getattr(req, "kind", "instruction")
    if kind == "instruction":
      raw = answers_to_raw(ask(build_state(req.text, req.state), q_one), q_one, req.state, limits, yaw_rate)
      return validate(raw, request_id=req.request_id, generation=req.generation, current=req.state, limits=limits)
    if kind == "mission":
      raw = answers_to_mission(ask(build_state(req.text, req.state), q_mission), q_mission, req.text, req.state,
                               limits, cfg.turn_rate)
      return validate_mission(raw, mission_id=req.mission_id, limits=limits, current=req.state)
    if kind == "tick":
      ctx = req.context or {}
      ceiling = float(ctx.get("_ceiling", limits.speed_cap(req.state.target_gaits)))
      allowed = tuple(ctx.get("_allowed_gaits", limits.gaits))
      state = {k: v for k, v in ctx.items() if not k.startswith("_")}
      raw, _ = answers_to_tick_raw(ask(build_tick_state(state), q_tick), q_tick, req.state, cfg, ceiling, allowed,
                                   limits)
      return validate(raw, request_id=req.request_id, generation=req.generation, current=req.state, limits=limits,
                      kind="tick", mission_id=req.mission_id, snapshot_seq=req.snapshot_seq, speed_ceiling=ceiling)
    raise DecisionError(f"unknown request kind {kind!r}")

  return decide


# --------------------------------------------------------------- benchmark


def bench(model_id: str, device: str, n: int, compile: bool) -> dict:
  """Load time, memory and warm latency on this machine (real model, no sim)."""
  dec = LiquidDecider(model_id, device, compile)
  rep = dict(dec.load())
  st = PlannerState("trot", 0.1, 0.0, 0.0, ("trot",))
  qs = build_questions()
  cmds = ["Walk forward slowly.", "Increase speed slightly.", "Turn left.",
          "Turn right and continue walking.", "Stop moving."]
  t0 = time.perf_counter()
  dec.answer(cmds[0], st, qs)  # first call compiles this shape
  rep["first_call_s"] = time.perf_counter() - t0
  lat, results = [], {}
  for i in range(n):
    c = cmds[i % len(cmds)]
    t0 = time.perf_counter()
    a = dec.answer(c, st, qs)
    lat.append(time.perf_counter() - t0)
    results[c] = {k: v.get("choice") for k, v in a.items()}
  lat.sort()
  rep.update(n=n, latency_ms_median=1e3 * lat[len(lat) // 2], latency_ms_max=1e3 * lat[-1], answers=results)
  # Closed-loop question sets: mission interpretation (once per instruction)
  # and the periodic tick over a representative feedback context.
  rep["closed_loop"] = bench_closed_loop(dec, st, n)
  rep["recommended_decision_hz"] = round(1.0 / max(1.0, 2.0 * rep["closed_loop"]["tick_ms_median"] / 1e3), 2)
  try:
    import torch

    if device.startswith("cuda"):
      rep["vram_peak_mb"] = torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
  except Exception:  # noqa: BLE001
    pass
  return rep


def bench_closed_loop(dec: LiquidDecider, st: PlannerState, n: int) -> dict:
  qm, qt = build_mission_questions(), build_tick_questions()
  missions = ["Walk forward slowly.", "Move toward the target and turn left.", "Maintain a moderate speed.",
              "Slow down if the robot becomes unstable.", "Use a suitable gait and reach the target cautiously."]
  ctx = {"mission": {"user instruction": missions[1], "objective": "reach the target", "phase": "approach",
                     "desired speed": "0.20 m/s"},
         "current command": {"gait": "trot", "speed": "0.10 m/s", "turning rate": "+0.00 rad/s"},
         "measured state": {"forward velocity": "+0.09 m/s", "body tilt": "3.0 deg", "stability": "stable"},
         "target": {"distance": "1.80 m", "direction": "+35 deg (left of the body heading)", "near": "no"},
         "previous decision": {"action": "speed 0.10 m/s", "result": "distance decreased by 0.08 m"},
         "safety": ["no active constraint"]}
  t0 = time.perf_counter()
  dec.ask(build_state(missions[0], st), qm)
  dec.ask(ctx, qt)
  first = time.perf_counter() - t0
  ml, tl, mans = [], [], {}
  for i in range(n):
    t0 = time.perf_counter()
    a = dec.ask(build_state(missions[i % len(missions)], st), qm)
    ml.append(time.perf_counter() - t0)
    mans[missions[i % len(missions)]] = {k: v.get("choice") for k, v in a.items()}
    t0 = time.perf_counter()
    ta = dec.ask(ctx, qt)
    tl.append(time.perf_counter() - t0)
  ml.sort()
  tl.sort()
  return {"first_calls_s": first, "mission_ms_median": 1e3 * ml[len(ml) // 2], "mission_ms_max": 1e3 * ml[-1],
          "tick_ms_median": 1e3 * tl[len(tl) // 2], "tick_ms_max": 1e3 * tl[-1], "mission_answers": mans,
          "tick_answers": {k: v.get("choice") for k, v in ta.items()}}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description="Benchmark LiquidAI d1 for the contact-play runtime.")
  ap.add_argument("--model", default=MODEL_ID)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--n", type=int, default=20)
  ap.add_argument("--no-compile", action="store_true")
  a = ap.parse_args(argv)
  try:
    print(json.dumps(bench(a.model, a.device, a.n, not a.no_compile), indent=2, default=str))
  except LiquidUnavailable as e:
    raise SystemExit(f"[liquid] unavailable: {e}") from e


if __name__ == "__main__":
  main()
