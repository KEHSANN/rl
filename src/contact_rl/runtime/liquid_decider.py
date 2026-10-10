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
  Decision,
  DecisionError,
  Limits,
  PlannerState,
  validate,
)

MODEL_ID = "LiquidAI/d1-3B-w8a8"
INSTALL_HINT = 'uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow ...'

SPEED_LEVELS = {"slow": 0.3, "medium": 0.6, "fast": 1.0}  # fractions of the validated speed cap
SPEED_DELTA = 0.05  # m/s for "faster" / "slower"
DIRECTIONS = {"forward": 0.0, "backward": math.pi, "left": math.pi / 2, "right": -math.pi / 2}


class LiquidUnavailable(RuntimeError):
  """The model cannot be loaded or run in this environment (with a diagnostic)."""


def build_questions(gaits=Limits().gaits) -> dict:
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
    if self.model is None:
      raise LiquidUnavailable("model not loaded")
    import torch

    with torch.inference_mode():
      out = self.model.system_one(build_state(command, state), questions)
    if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
      raise DecisionError(f"unexpected system_one output: {type(out).__name__}")
    return out["answers"]

  def make_decide(self, limits: Limits, yaw_rate: float):
    """``decide(request) -> Decision`` for :class:`DecisionLoop` (runs on the worker thread)."""
    questions = build_questions(limits.gaits)

    def decide(req) -> Decision:
      answers = self.answer(req.text, req.state, questions)
      raw = answers_to_raw(answers, questions, req.state, limits, yaw_rate)
      return validate(raw, request_id=req.request_id, generation=req.generation, current=req.state, limits=limits)

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
  try:
    import torch

    if device.startswith("cuda"):
      rep["vram_peak_mb"] = torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
  except Exception:  # noqa: BLE001
    pass
  return rep


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
