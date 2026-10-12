"""Small training statistics helpers (torch only, no mjlab)."""

from __future__ import annotations

import torch


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
  """1 - Var[R - V] / Var[R]; NaN if the returns have (almost) no variance.
  1 = perfect value function, 0 = no better than a constant, < 0 = worse."""
  v, r = values.flatten().float(), returns.flatten().float()
  var_r = torch.var(r)
  if float(var_r) <= 1e-12:
    return float("nan")
  return float(1.0 - torch.var(r - v) / var_r)
