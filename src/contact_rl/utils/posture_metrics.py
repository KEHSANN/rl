"""Additive posture diagnostics for flat-ground Go2 evaluation.

Legacy success means survival, not healthy gait. Terminal post-step states
are auto-reset poses and are excluded. Contact is sampled at policy frequency,
not over all physics substeps. Importing summaries does not require torch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from contact_rl.utils.episode_metrics import EpisodeAccumulator
from contact_rl.utils.episode_metrics import summarize as summarize_legacy

if TYPE_CHECKING:
  import torch

FOOT_ORDER = ("FL", "FR", "RL", "RR")


class PostureAccumulator(EpisodeAccumulator):
  """Torch rollout accumulator; preserves legacy fields and success semantics."""

  def __init__(self, zeros: torch.Tensor, min_height: float = 0.08):
    import torch

    if not math.isfinite(min_height) or min_height <= 0:
      raise ValueError("min_height must be finite and positive")
    if zeros.ndim != 1:
      raise ValueError("zeros must have shape [N]")
    super().__init__(zeros)
    self.min_height = float(min_height)
    self.posture_samples = torch.zeros_like(zeros)
    self.invalid_samples = torch.zeros_like(zeros)
    self.height_sum = zeros.new_zeros((len(zeros), 4))
    self.height_min = zeros.new_full((len(zeros), 4), float("inf"))
    self.below_sum = torch.zeros_like(self.height_sum)
    self.any_below_sum = torch.zeros_like(zeros)
    self.contact_sum = torch.zeros_like(zeros)
    self.contact_count_sum = torch.zeros_like(zeros)

  def step(
    self,
    reward: torch.Tensor,
    reason: torch.Tensor,
    *,
    knee_heights: torch.Tensor,
    illegal_found: torch.Tensor,
    **kwargs: Any,
  ) -> None:
    import torch

    n = len(self.active)
    if knee_heights.shape != (n, 4):
      raise ValueError("knee_heights must have shape [N, 4] in FL/FR/RL/RR order")
    if illegal_found.ndim != 2 or illegal_found.shape[0] != n or illegal_found.shape[1] == 0:
      raise ValueError("illegal_found must have shape [N, G] with G > 0")
    if reward.shape != (n,) or reason.shape != (n,):
      raise ValueError("reward and reason must have shape [N]")
    valid = (self.active > 0.5) & (reason < 0.5)
    finite = torch.isfinite(knee_heights).all(dim=1) & torch.isfinite(illegal_found).all(dim=1)
    self.invalid_samples += (valid & ~finite).to(self.invalid_samples.dtype)
    valid = valid & finite
    self.posture_samples += valid.to(self.posture_samples.dtype)
    self.height_sum += torch.where(valid[:, None], knee_heights, 0.0)
    self.height_min = torch.minimum(
      self.height_min,
      torch.where(valid[:, None], knee_heights, float("inf")),
    )
    below = knee_heights < self.min_height
    self.below_sum += (valid[:, None] & below).to(self.below_sum.dtype)
    self.any_below_sum += (valid & below.any(dim=1)).to(self.any_below_sum.dtype)
    contacts = illegal_found > 0
    self.contact_sum += (valid & contacts.any(dim=1)).to(self.contact_sum.dtype)
    self.contact_count_sum += torch.where(valid, contacts.sum(dim=1), 0)
    super().step(reward, reason, **kwargs)

  def rows(self, step_dt: float, extra: dict | None = None) -> list[dict]:
    rows = super().rows(step_dt, extra)
    samples = self.posture_samples.tolist()
    invalid = self.invalid_samples.tolist()
    heights = self.height_sum.tolist()
    minima = self.height_min.tolist()
    below = self.below_sum.tolist()
    any_below = self.any_below_sum.tolist()
    contacts = self.contact_sum.tolist()
    counts = self.contact_count_sum.tolist()
    for i, row in enumerate(rows):
      n = samples[i]
      denom = n if n else float("nan")
      row.update({
        "posture_samples": int(n),
        "posture_invalid_samples": int(invalid[i]),
        "knee_min_height_threshold_m": self.min_height,
        "knee_below_fraction": any_below[i] / denom,
        "knee_below_steps": int(any_below[i]),
        "illegal_contact_fraction": contacts[i] / denom,
        "illegal_contact_steps": int(contacts[i]),
        "illegal_contact_count_mean": counts[i] / denom,
        "illegal_contact_count_sum": int(counts[i]),
      })
      for j, leg in enumerate(FOOT_ORDER):
        row[f"knee_{leg}_height_mean_m"] = heights[i][j] / denom
        row[f"knee_{leg}_height_min_m"] = minima[i][j] if n else float("nan")
        row[f"knee_{leg}_below_fraction"] = below[i][j] / denom
    return rows


def summarize(rows: list[dict]) -> dict:
  """Legacy summary plus posture aggregates; legacy-only inputs are unchanged."""
  out = summarize_legacy(rows)
  measured = [r for r in rows if "posture_samples" in r]
  if not measured:
    return out
  samples = sum(r["posture_samples"] for r in measured)
  out["posture_samples_total"] = samples
  out["posture_invalid_samples_total"] = sum(r["posture_invalid_samples"] for r in measured)
  for name, numerator in (
    ("knee_below_fraction", "knee_below_steps"),
    ("illegal_contact_fraction", "illegal_contact_steps"),
    ("illegal_contact_count_mean", "illegal_contact_count_sum"),
  ):
    values = [float(r[name]) for r in measured if math.isfinite(r[name])]
    out[f"{name}_mean"] = sum(values) / len(values) if values else float("nan")
    out[f"{name}_pooled"] = sum(r[numerator] for r in measured) / samples if samples else float("nan")
  for leg in FOOT_ORDER:
    for metric in ("height_mean_m", "height_min_m", "below_fraction"):
      key = f"knee_{leg}_{metric}"
      values = [float(r[key]) for r in measured if math.isfinite(r[key])]
      out[f"{key}_mean"] = sum(values) / len(values) if values else float("nan")
    minima = [r[f"knee_{leg}_height_min_m"] for r in measured if r["posture_samples"]]
    out[f"knee_{leg}_height_min_m_min"] = min(minima) if minima else float("nan")
  return out
