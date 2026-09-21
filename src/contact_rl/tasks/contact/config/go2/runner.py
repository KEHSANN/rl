"""On-policy runner for the contact-explicit task.

Extends mjlab's :class:`MjlabOnPolicyRunner` with:

1. ONNX export on checkpoint save (same behaviour as mjlab's
   ``VelocityOnPolicyRunner``), and
2. the paper's **linear entropy decay** (Section 4, "Training").

rsl-rl's ``PPO.entropy_coef`` is a plain mutable float with no built-in
schedule, so the decay is applied by chunking ``learn`` and re-setting
``self.alg.entropy_coef`` between chunks (each ``super().learn`` call continues
from ``current_learning_iteration``, so the chunks compose into one run). A
guard completes any remaining iterations in a single call if the decay
bookkeeping ever raises, so a training run never dies on account of the
schedule. Tune the schedule via the module-level constants below.
"""

from __future__ import annotations

import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

# --- Entropy-decay schedule (realises the paper's "entropy decay"). ---
ENTROPY_DECAY_ENABLED = True
ENTROPY_START = 0.01
ENTROPY_END = 0.001
ENTROPY_DECAY_ITERS = 5000  # Iterations over which to anneal START -> END.
ENTROPY_UPDATE_EVERY = 50  # Re-set the coefficient every N iterations.


class ContactOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  # --- ONNX export on save (mirrors VelocityOnPolicyRunner). ---
  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      onnx_path = os.path.join(policy_path, filename)
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(onnx_path, metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  # --- Entropy decay. ---
  def _entropy_at(self, iteration: int) -> float:
    frac = min(max(iteration / float(ENTROPY_DECAY_ITERS), 0.0), 1.0)
    return ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if not ENTROPY_DECAY_ENABLED or not hasattr(self.alg, "entropy_coef"):
      return super().learn(num_learning_iterations, init_at_random_ep_len)

    try:
      remaining = num_learning_iterations
      first = True
      while remaining > 0:
        chunk = min(ENTROPY_UPDATE_EVERY, remaining)
        self.alg.entropy_coef = self._entropy_at(self.current_learning_iteration)
        super().learn(chunk, init_at_random_ep_len and first)
        remaining -= chunk
        first = False
    except Exception as e:
      print(f"[WARN] entropy-decay loop failed ({e}); continuing without decay.")
      if remaining > 0:
        super().learn(remaining, init_at_random_ep_len and first)
