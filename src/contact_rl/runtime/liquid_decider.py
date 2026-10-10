"""Adapter for LiquidAI/d1-3B-w8a8 (decision model; no text output).

Heavy imports (transformers, torch) happen only inside ``LiquidDecider.load``,
so importing this module and building questions needs no GPU.
UNVERIFIED on hardware in this branch: the load/answer path needs an
Ampere+ NVIDIA GPU (INT8 tensor cores) and is not exercised by the unit tests.
"""

from __future__ import annotations

from contact_rl.runtime.decision_schema import GAITS

MODEL_ID = "LiquidAI/d1-3B-w8a8"


def build_questions() -> dict:
  """Questions in the Decision Index schema from the model card."""
  return {
    "speed_action": {
      "type": "choice",
      "instructions": "How should the target speed change? Consider the user's command and the robot state.",
      "criteria": {
        "increase": "Speed up toward the requested target",
        "decrease": "Slow down (robot unstable, too fast, or target reached)",
        "hold": "Keep the current speed",
      },
    },
    "turn_action": {
      "type": "choice",
      "instructions": "How should the heading change?",
      "criteria": {
        "left": "Turn left (positive heading / yaw rate)",
        "right": "Turn right (negative heading / yaw rate)",
        "straight": "Keep the current heading",
      },
    },
    "gait_action": {
      "type": "choice",
      "instructions": "Which gait fits the command?",
      "criteria": {g: f"gait {g}" for g in GAITS} | {"keep": "Keep the current gait"},
    },
  }


def build_state(command: str, state: dict) -> str:
  """Text state. Only fields that really exist are passed; missing ones are omitted, never invented."""
  lines = [f"User command: {command}"]
  for k, v in state.items():
    if v is not None:
      lines.append(f"{k}: {v}")
  return "\n".join(lines)


def answers_to_raw(answers: dict, speed_map: dict[str, float]) -> dict:
  """Map model answers (choice labels) to the raw decision dict for validation.

  ``speed_map`` gives the new absolute speed for increase/decrease/hold,
  computed by the caller from the current speed (never by the model).
  """
  raw: dict = {}
  sp = answers.get("speed_action", {}).get("choice")
  if sp in speed_map:
    raw["speed"] = speed_map[sp]
  turn = answers.get("turn_action", {}).get("choice")
  if turn == "left":
    raw["yaw_rate"] = 0.5
  elif turn == "right":
    raw["yaw_rate"] = -0.5
  elif turn == "straight":
    raw["yaw_rate"] = 0.0
  g = answers.get("gait_action", {}).get("choice")
  if g in GAITS:
    raw["gait"] = g
  return raw


class LiquidDecider:
  """Loads the model once and answers questions. Not run in unit tests."""

  def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
    self.model_id = model_id
    self.device = device
    self.model = None

  def load(self) -> None:
    from transformers import AutoModel  # heavy: only here

    self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
    self.model.compile(mode="reduce-overhead")

  def decide(self, command: str, state: dict) -> dict:
    if self.model is None:
      raise RuntimeError("model not loaded")
    out = self.model.system_one(build_state(command, state), build_questions())
    return out["answers"]
