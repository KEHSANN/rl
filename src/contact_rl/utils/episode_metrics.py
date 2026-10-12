"""Per-episode evaluation accounting (backend-agnostic: torch or numpy).

Semantics (fixes of the previous contact-eval):

* **Termination reasons are exclusive and explicit.** ``fall`` = the
  ``fell_over`` term fired; ``other_termination`` = any other non-time-out
  term fired; ``timeout`` = only a time-out (truncation) term fired;
  ``running`` = the episode was still alive when evaluation stopped. A
  time-out is *never* counted as a fall.
* **The terminal transition is excluded from state metrics.** mjlab
  auto-resets done envs inside ``step()``, so the post-step state of a done env
  already belongs to the next episode. Rewards of the terminal step are real
  and are counted; contact / tracking metrics of that step are not.
* Each env contributes its **first** episode only (unbiased w.r.t. episode
  length: short failing episodes cannot be over-represented).
* :func:`summarize` reports both the per-episode mean (``<metric>_mean``) and
  the **pooled** estimate (``<metric>_pooled``: sums over all counted steps /
  in-contact samples divided by their totals). The pooled numbers are the
  aggregation the Fig. 6 table of this repository has always used (one
  step-weighted average over all episodes), so the legacy ``eval_fig6.csv``
  keeps its meaning.

All inputs are ``[N]`` / ``[N, F]`` arrays of the same backend. Only
arithmetic and comparisons are used, so numpy arrays work in tests.
"""

from __future__ import annotations

import math
from typing import Any

REASONS = ("running", "timeout", "fall", "other_termination")
RUNNING, TIMEOUT, FALL, OTHER = 0, 1, 2, 3


def _f(x: Any) -> Any:
  """bool/int array -> float array (torch or numpy)."""
  return x * 1.0


def classify(truncated: Any, terminated: Any, fell: Any) -> Any:
  """Reason code per env for this step (0 where nothing happened).

  ``fell`` must be the fall term only; ``terminated`` = any non-time-out term.
  Precedence: fall > other termination > timeout.
  """
  tr, te, fe = _f(truncated) > 0.5, _f(terminated) > 0.5, _f(fell) > 0.5
  fall = _f(fe)
  other = _f(te) * (1.0 - fall)
  timeout = _f(tr) * (1.0 - _f(te))
  return fall * FALL + other * OTHER + timeout * TIMEOUT


class EpisodeAccumulator:
  STATE_KEYS = (
    "contact_hamming",
    "contact_location_error",
    "tracking_error",
    "lin_vel_error",
    "yaw_rate_error",
  )

  def __init__(self, zeros: Any):
    """``zeros``: an ``[N]`` float zero array on the right device."""
    z = zeros
    self.active = z + 1.0
    self.reason = z + 0.0
    self.reward = z + 0.0
    self.length = z + 0.0
    self.state_steps = z + 0.0
    self.sums = {k: z + 0.0 for k in self.STATE_KEYS}
    self.loc_cnt = z + 0.0
    self.discoveries = z + 0.0

  def step(
    self,
    reward: Any,
    reason: Any,
    hamming: Any | None = None,
    loc_err_sum: Any | None = None,
    loc_cnt: Any | None = None,
    tracking: Any | None = None,
    lin_vel_err: Any | None = None,
    yaw_rate_err: Any | None = None,
    discovered: Any | None = None,
  ) -> None:
    """Account one ``env.step``. ``reason`` from :func:`classify` (0 = alive)."""
    a = self.active
    done = _f(reason > 0.5)
    self.reward = self.reward + a * reward
    self.length = self.length + a
    valid = a * (1.0 - done)  # exclude the auto-reset (terminal) transition
    self.state_steps = self.state_steps + valid
    if hamming is not None:
      self.sums["contact_hamming"] = self.sums["contact_hamming"] + valid * hamming
    if loc_err_sum is not None and loc_cnt is not None:
      self.sums["contact_location_error"] = self.sums["contact_location_error"] + valid * loc_err_sum
      self.loc_cnt = self.loc_cnt + valid * loc_cnt
    if tracking is not None:
      self.sums["tracking_error"] = self.sums["tracking_error"] + valid * tracking
    if lin_vel_err is not None:
      self.sums["lin_vel_error"] = self.sums["lin_vel_error"] + valid * lin_vel_err
    if yaw_rate_err is not None:
      self.sums["yaw_rate_error"] = self.sums["yaw_rate_error"] + valid * yaw_rate_err
    if discovered is not None:
      self.discoveries = self.discoveries + valid * discovered
    newly = a * done
    self.reason = self.reason * (1.0 - newly) + reason * newly
    self.active = a * (1.0 - done)

  def all_done(self) -> bool:
    return float(self.active.sum()) == 0.0

  def rows(self, step_dt: float, extra: dict | None = None) -> list[dict]:
    def L(x):
      return [float(v) for v in x.tolist()]

    reason, reward, length, steps = L(self.reason), L(self.reward), L(self.length), L(self.state_steps)
    sums = {k: L(v) for k, v in self.sums.items()}
    cnt, disc = L(self.loc_cnt), L(self.discoveries)
    out = []
    for i in range(len(reason)):
      r = REASONS[int(round(reason[i]))]
      n = max(steps[i], 1.0)
      row = {
        "env": i,
        "termination": r,
        "fell": r == "fall",
        "success": r in ("timeout", "running"),
        "reward": reward[i],
        "length_steps": int(length[i]),
        "length_s": length[i] * step_dt,
        "contact_plan_hamming": sums["contact_hamming"][i] / n,
        "contact_location_error_cm": 100.0 * sums["contact_location_error"][i] / max(cnt[i], 1.0),
        "foot_tracking_error_cm": 100.0 * sums["tracking_error"][i] / n,
        "lin_vel_error_mps": sums["lin_vel_error"][i] / n,
        "yaw_rate_error_rps": sums["yaw_rate_error"][i] / n,
        "goals_discovered": disc[i],
        # Raw counters so pooled (step-weighted) aggregates can be rebuilt.
        "state_steps": int(steps[i]),
        "contact_samples": int(cnt[i]),
        "contact_hamming_sum": sums["contact_hamming"][i],
        "contact_location_error_sum_m": sums["contact_location_error"][i],
      }
      if extra:
        row.update(extra)
      out.append(row)
    return out


def _mean(xs: list[float]) -> float:
  return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
  if len(xs) < 2:
    return 0.0
  m = _mean(xs)
  return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


NUMERIC = (
  "reward",
  "length_s",
  "contact_plan_hamming",
  "contact_location_error_cm",
  "foot_tracking_error_cm",
  "lin_vel_error_mps",
  "yaw_rate_error_rps",
  "goals_discovered",
)


def summarize(rows: list[dict]) -> dict:
  """Machine-readable aggregate for summary.json (rates + mean/std)."""
  n = len(rows)
  out: dict[str, Any] = {"episodes": n}
  for r in REASONS:
    out[f"{r}_rate"] = sum(1 for x in rows if x["termination"] == r) / n if n else float("nan")
  out["success_rate"] = sum(1 for x in rows if x["success"]) / n if n else float("nan")
  for k in NUMERIC:
    xs = [float(x[k]) for x in rows if k in x and x[k] == x[k]]
    out[f"{k}_mean"] = _mean(xs)
    out[f"{k}_std"] = _std(xs)
  steps = sum(int(x.get("state_steps", 0)) for x in rows)
  samples = sum(int(x.get("contact_samples", 0)) for x in rows)
  if any("contact_hamming_sum" in x for x in rows):
    ham = sum(float(x.get("contact_hamming_sum", 0.0)) for x in rows)
    loc = sum(float(x.get("contact_location_error_sum_m", 0.0)) for x in rows)
    out["contact_plan_hamming_pooled"] = ham / steps if steps else float("nan")
    out["contact_location_error_cm_pooled"] = 100.0 * loc / samples if samples else float("nan")
  out["state_steps_total"] = steps
  out["contact_samples_total"] = samples
  return out
