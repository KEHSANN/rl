"""Recurrent-policy state handling shared by eval / play.

rsl-rl's ``RNNModel.reset(dones)`` zeroes the GRU hidden state of the envs
whose ``dones == 1``; ``reset()`` (no argument) drops the whole state. PPO
does this during training (``process_env_step``) but mjlab's viewers never do,
so without these helpers the hidden state of a finished episode leaks into
the next one.
"""

from __future__ import annotations

from typing import Any


def reset_recurrent_state(policy: Any, dones: Any | None = None) -> bool:
  """Reset the policy's recurrent state for ``dones`` (all envs if None).
  Returns True if a reset was issued. Non-recurrent policies are a no-op."""
  fn = getattr(policy, "reset", None)
  if fn is None or not callable(fn):
    return False
  if dones is None:
    fn()
    return True
  if not bool((dones > 0).any()):
    return False
  fn(dones)
  return True


def is_recurrent(policy: Any) -> bool:
  return bool(getattr(policy, "is_recurrent", False))
