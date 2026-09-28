"""On-policy runner for the contact-explicit task.

Extends mjlab's :class:`MjlabOnPolicyRunner` with ONNX export on save and the
paper's **entropy decay** (Section 4, "Training").

Entropy decay is applied by wrapping ``self.alg.update`` so the coefficient is
set right before every PPO update, inside a *single* ``learn()`` call. The
previous revision chunked ``learn()`` into 50-iteration calls instead, which
(i) re-ran the last iteration of every chunk (rsl-rl stores
``current_learning_iteration = it``, and the next call starts from it), (ii)
reset the episode reward/length buffers every chunk, (iii) triggered rsl-rl's
end-of-``learn`` checkpoint every chunk, and (iv) wrapped everything in a
``try/except`` that silently restarted training after *any* exception.
"""

from __future__ import annotations

import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from .rl_cfg import (
  ENTROPY_DECAY_ITERS,
  ENTROPY_END,
  ENTROPY_START,
)


def entropy_at(iteration: int) -> float:
  """Linear entropy schedule, clamped to ``ENTROPY_END`` after decay."""
  frac = min(max(iteration / float(max(ENTROPY_DECAY_ITERS, 1)), 0.0), 1.0)
  return ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)


class ContactOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

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
    except Exception as e:  # Export is a convenience; never kill training.
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def _install_entropy_schedule(self) -> None:
    if getattr(self, "_entropy_schedule_installed", False):
      return
    if not hasattr(self.alg, "entropy_coef"):
      raise AttributeError("PPO algorithm has no 'entropy_coef'; cannot decay it.")
    original_update = self.alg.update
    self._entropy_updates = 0

    def update_with_entropy_decay(*args, **kwargs):
      it = self._entropy_it0 + self._entropy_updates
      self.alg.entropy_coef = entropy_at(it)
      self._entropy_updates += 1
      return original_update(*args, **kwargs)

    self.alg.update = update_with_entropy_decay  # type: ignore[method-assign]
    self._entropy_schedule_installed = True

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    # Resume-aware: the schedule continues from the loaded iteration.
    self._entropy_it0 = int(self.current_learning_iteration)
    self._install_entropy_schedule()
    self._entropy_updates = 0
    return super().learn(num_learning_iterations, init_at_random_ep_len)
