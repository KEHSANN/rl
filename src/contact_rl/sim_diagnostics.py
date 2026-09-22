"""Read-only simulation diagnostics for the contact-explicit locomotion task.

This module measures the quantities Phase 1 of the optimisation plan needs
*before* any simulation parameter is changed:

* **Contact-sensor match counts.** Exact, not estimated. mjlab's
  :class:`~mjlab.sensor.ContactSensor` documents its ``found`` field as
  "0=no contact, >0=match count **before reduction**", which is precisely the
  quantity ``contact_sensor_maxmatch`` bounds. So ``max(found)`` over a run is
  the true requirement, and saturation at the configured cap is directly
  detectable.
* **Per-world constraint counts** (the ``njmax`` question) with explicit
  overflow detection. MJWarp does not raise on constraint overflow -- it drops
  rows, which shows up much later as feet sinking and a corrupted hold reward.
* **Per-world contact counts** (context for ``nconmax``, which we do not tune).
* **Foot penetration depth**, which is how you prove a ``ccd_iterations``
  reduction was safe rather than merely fast.

Design notes
------------
*Exact statistics, not sampled.* Counts are small non-negative integers, so
every metric is accumulated into a GPU histogram via ``torch.bincount``. Mean,
max and percentiles are then computed exactly over every sample of the whole
run, with O(1) memory and no host transfers per step.

*Defensive field probing.* MJWarp's ``Data`` layout is not a stable public API
and differs between ``mujoco-warp`` releases. Rather than hardcode field names
and crash on a version bump, the collector probes a list of candidates once,
reports what it found, and silently drops metrics whose source is unavailable.
A run that can only measure the contact sensor is still a useful run -- and the
sensor is the one source that is guaranteed present, because the task defines
the sensor itself.

*Zero footprint on training.* Nothing here is wired into the env, the managers
or the runner. ``scripts/benchmark.py`` is the only consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "CountStats",
  "SimStatsCollector",
  "recommend_maxmatch",
  "recommend_njmax",
]

# Upper bound for the per-world contact histogram. Contacts are pooled across
# worlds in MJWarp, so there is no per-world cap to read from the config; this
# is only a histogram bound and anything above it lands in the overflow bin.
_CONTACT_HIST_CAP = 256

# Candidate MJWarp field names, most-preferred first. See module docstring.
_NEFC_FIELDS = ("nefc",)
_NCON_FIELDS = ("ncon",)
_EFC_WORLDID_PATHS = (("efc", "worldid"), ("efc_worldid",))
_CONTACT_WORLDID_PATHS = (("contact", "worldid"), ("contact_worldid",))
_CONTACT_DIST_PATHS = (("contact", "dist"), ("contact_dist",))


##
# Helpers.
##


def _as_tensor(value: Any) -> torch.Tensor | None:
  """Coerce an mjlab ``TorchArray`` / warp array / tensor to a torch tensor."""
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    return value
  # mjlab wraps warp arrays in TorchArray, which exposes .wp_array.
  wp_array = getattr(value, "wp_array", None)
  if wp_array is not None:
    try:
      import warp as wp

      return wp.to_torch(wp_array)
    except Exception:
      return None
  try:
    return torch.as_tensor(value)
  except Exception:
    return None


def _resolve_path(root: Any, path: Sequence[str]) -> torch.Tensor | None:
  """Walk an attribute path on the MJWarp data bridge, tolerating absences."""
  node: Any = root
  for attr in path:
    try:
      node = getattr(node, attr)
    except Exception:
      return None
    if node is None:
      return None
  return _as_tensor(node)


def _first_available(
  root: Any, paths: Sequence[Sequence[str]]
) -> tuple[str | None, torch.Tensor | None]:
  for path in paths:
    tensor = _resolve_path(root, path)
    if tensor is not None:
      return ".".join(path), tensor
  return None, None


def _next_power_of_two(value: int) -> int:
  out = 1
  while out < value:
    out *= 2
  return out


def _round_up(value: int, multiple: int) -> int:
  return int(np.ceil(value / multiple) * multiple)


##
# Statistics.
##


@dataclass
class CountStats:
  """Exact statistics over an integer-valued quantity, from a histogram."""

  name: str
  samples: int
  minimum: int
  maximum: int
  mean: float
  p50: int
  p95: int
  p99: int
  saturated_samples: int
  """Number of samples that landed at or above the histogram cap. For the
  contact sensor the cap IS ``contact_sensor_maxmatch``, so a non-zero value
  here means the sensor was clipping and the measurement is a lower bound."""
  cap: int

  @property
  def saturated(self) -> bool:
    return self.saturated_samples > 0

  def as_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "samples": self.samples,
      "min": self.minimum,
      "max": self.maximum,
      "mean": self.mean,
      "p50": self.p50,
      "p95": self.p95,
      "p99": self.p99,
      "cap": self.cap,
      "saturated_samples": self.saturated_samples,
    }


@dataclass
class _Histogram:
  """Exact integer histogram accumulated on device."""

  name: str
  cap: int
  device: str
  _bins: torch.Tensor = field(init=False)

  def __post_init__(self) -> None:
    # cap + 2 bins: values 0..cap, plus one overflow bin for > cap.
    self._bins = torch.zeros(self.cap + 2, dtype=torch.int64, device=self.device)

  def update(self, values: torch.Tensor) -> None:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
      return
    clamped = flat.to(torch.int64).clamp_(min=0, max=self.cap + 1)
    self._bins += torch.bincount(clamped, minlength=self.cap + 2)

  def stats(self) -> CountStats | None:
    bins = self._bins.detach().cpu().numpy().astype(np.int64)
    total = int(bins.sum())
    if total == 0:
      return None
    values = np.arange(bins.size, dtype=np.int64)
    nonzero = np.flatnonzero(bins)
    cumulative = np.cumsum(bins)

    def quantile(q: float) -> int:
      threshold = q * total
      idx = int(np.searchsorted(cumulative, threshold, side="left"))
      return int(values[min(idx, bins.size - 1)])

    return CountStats(
      name=self.name,
      samples=total,
      minimum=int(values[nonzero[0]]),
      maximum=int(values[nonzero[-1]]),
      mean=float((bins * values).sum() / total),
      p50=quantile(0.50),
      p95=quantile(0.95),
      p99=quantile(0.99),
      saturated_samples=int(bins[self.cap] + bins[self.cap + 1]),
      cap=self.cap,
    )


##
# Collector.
##


class SimStatsCollector:
  """Accumulates contact / constraint statistics over a simulation run.

  Usage::

      collector = SimStatsCollector(env)
      for _ in range(n):
        env.step(actions)
        collector.record()
      print(collector.report())
  """

  def __init__(
    self,
    env: "ManagerBasedRlEnv",
    sensor_name: str = "feet_ground_contact",
    verbose: bool = True,
  ) -> None:
    self._env = env
    self._sensor_name = sensor_name
    self._device = str(env.device)
    self._num_envs = int(env.num_envs)
    self._records = 0

    sim = env.sim
    self._data = sim.data
    self._configured_maxmatch = int(env.cfg.sim.contact_sensor_maxmatch)
    self._configured_njmax = env.cfg.sim.njmax

    # --- Contact sensor (always available: the task defines it). ---
    self._sensor = None
    try:
      self._sensor = env.scene[sensor_name]
    except Exception:
      if verbose:
        print(
          f"[WARN] diagnostics: contact sensor {sensor_name!r} not found; "
          f"skipping match-count measurement."
        )

    # --- Probe MJWarp data sources once. ---
    self._sources: dict[str, str] = {}

    self._nefc_field = None
    for name in _NEFC_FIELDS:
      if _resolve_path(self._data, (name,)) is not None:
        self._nefc_field = name
        self._sources["constraints"] = f"data.{name}"
        break

    self._ncon_field = None
    for name in _NCON_FIELDS:
      if _resolve_path(self._data, (name,)) is not None:
        self._ncon_field = name
        self._sources["contacts_total"] = f"data.{name}"
        break

    self._efc_worldid_path, _ = _first_available(self._data, _EFC_WORLDID_PATHS)
    if self._efc_worldid_path:
      self._sources["constraints_per_world"] = f"data.{self._efc_worldid_path}"

    self._contact_worldid_path, _ = _first_available(
      self._data, _CONTACT_WORLDID_PATHS
    )
    if self._contact_worldid_path:
      self._sources["contacts_per_world"] = f"data.{self._contact_worldid_path}"

    self._contact_dist_path, _ = _first_available(self._data, _CONTACT_DIST_PATHS)
    if self._contact_dist_path:
      self._sources["penetration"] = f"data.{self._contact_dist_path}"

    # --- Histograms. ---
    njmax_cap = int(self._configured_njmax) if self._configured_njmax else 1024
    self._hist: dict[str, _Histogram] = {}
    if self._sensor is not None:
      self._hist["sensor_match_count"] = _Histogram(
        name="sensor_match_count",
        cap=self._configured_maxmatch,
        device=self._device,
      )
      self._hist["feet_in_contact"] = _Histogram(
        name="feet_in_contact", cap=4, device=self._device
      )
    if self._nefc_field or self._efc_worldid_path:
      self._hist["constraints_per_world"] = _Histogram(
        name="constraints_per_world", cap=njmax_cap, device=self._device
      )
    if self._ncon_field or self._contact_worldid_path:
      self._hist["contacts_per_world"] = _Histogram(
        name="contacts_per_world", cap=_CONTACT_HIST_CAP, device=self._device
      )

    # --- Penetration tracking (float, so not a histogram). ---
    self._worst_penetration = 0.0

    # --- Overflow counters. ---
    self._njmax_overflow_steps = 0
    self._maxmatch_saturation_steps = 0

    if verbose:
      print("[INFO] diagnostics sources:")
      if not self._sources and self._sensor is None:
        print("  (none available -- nothing will be measured)")
      for key, src in sorted(self._sources.items()):
        print(f"  {key:<24} <- {src}")
      if self._sensor is not None:
        print(f"  {'sensor_match_count':<24} <- scene[{sensor_name!r}].data.found")
      missing = {
        "constraints",
        "constraints_per_world",
        "contacts_total",
        "contacts_per_world",
        "penetration",
      } - set(self._sources)
      if missing:
        print(
          f"  [note] unavailable in this mujoco-warp build: "
          f"{', '.join(sorted(missing))}"
        )

  ##
  # Recording.
  ##

  def record(self) -> None:
    """Sample every available metric. Call once per env step."""
    self._records += 1
    self._record_sensor()
    self._record_constraints()
    self._record_contacts()
    self._record_penetration()

  def _record_sensor(self) -> None:
    if self._sensor is None:
      return
    try:
      found = self._sensor.data.found
    except Exception:
      return
    if found is None:
      return
    found_t = _as_tensor(found)
    if found_t is None:
      return
    # `found` is the per-sensor match count BEFORE reduction, so this is the
    # exact quantity contact_sensor_maxmatch bounds.
    self._hist["sensor_match_count"].update(found_t)
    if bool((found_t >= self._configured_maxmatch).any()):
      self._maxmatch_saturation_steps += 1
    self._hist["feet_in_contact"].update((found_t > 0).sum(dim=-1))

  def _record_constraints(self) -> None:
    hist = self._hist.get("constraints_per_world")
    if hist is None:
      return
    per_world = self._per_world_counts(
      count_field=self._nefc_field,
      worldid_path=self._efc_worldid_path,
    )
    if per_world is None:
      return
    hist.update(per_world)
    if self._configured_njmax and bool(
      (per_world >= int(self._configured_njmax)).any()
    ):
      self._njmax_overflow_steps += 1

  def _record_contacts(self) -> None:
    hist = self._hist.get("contacts_per_world")
    if hist is None:
      return
    per_world = self._per_world_counts(
      count_field=self._ncon_field,
      worldid_path=self._contact_worldid_path,
    )
    if per_world is None:
      return
    hist.update(per_world)

  def _per_world_counts(
    self, count_field: str | None, worldid_path: str | None
  ) -> torch.Tensor | None:
    """Best-effort per-world counts.

    Two possible shapes, handled in order of usefulness:

    1. The count field is already per-world (``shape[0] == num_envs``): use it.
    2. The count field is a global total (``shape == (1,)``) and a ``worldid``
       array exists: bincount the first ``total`` world ids. This is the exact
       per-world distribution.
    3. Only a global total: fall back to reporting the total spread evenly is
       *not* acceptable (it would fabricate a distribution), so return None.
    """
    count_t = None
    if count_field is not None:
      count_t = _resolve_path(self._data, (count_field,))

    if count_t is not None and count_t.ndim >= 1 and count_t.shape[0] == self._num_envs:
      return count_t.reshape(self._num_envs)

    if worldid_path is None:
      return None
    worldid = _resolve_path(self._data, tuple(worldid_path.split(".")))
    if worldid is None:
      return None

    total = int(worldid.numel())
    if count_t is not None and count_t.numel() >= 1:
      total = min(total, int(count_t.reshape(-1)[0].item()))
    if total <= 0:
      return torch.zeros(self._num_envs, dtype=torch.int64, device=self._device)

    ids = worldid.reshape(-1)[:total].to(torch.int64).clamp_(0, self._num_envs - 1)
    return torch.bincount(ids, minlength=self._num_envs)[: self._num_envs]

  def _record_penetration(self) -> None:
    if self._contact_dist_path is None:
      return
    dist = _resolve_path(self._data, tuple(self._contact_dist_path.split(".")))
    if dist is None or dist.numel() == 0:
      return
    # MuJoCo contact `dist` is negative when geoms interpenetrate. Only the
    # active portion of the buffer is meaningful; inactive slots hold large
    # positive values, so taking the minimum is safe.
    worst = float(dist.min().item())
    if worst < 0.0:
      self._worst_penetration = min(self._worst_penetration, worst)

  ##
  # Reporting.
  ##

  def summary(self) -> dict[str, Any]:
    stats = {}
    for key, hist in self._hist.items():
      result = hist.stats()
      if result is not None:
        stats[key] = result.as_dict()

    out: dict[str, Any] = {
      "records": self._records,
      "num_envs": self._num_envs,
      "sources": dict(self._sources),
      "configured": {
        "contact_sensor_maxmatch": self._configured_maxmatch,
        "njmax": self._configured_njmax,
        "nconmax": self._env.cfg.sim.nconmax,
        "ccd_iterations": self._env.cfg.sim.mujoco.ccd_iterations,
      },
      "stats": stats,
      "warnings": {
        "njmax_overflow_steps": self._njmax_overflow_steps,
        "maxmatch_saturation_steps": self._maxmatch_saturation_steps,
      },
      "worst_penetration_m": self._worst_penetration,
    }
    out["recommendations"] = self.recommendations()
    return out

  def recommendations(self) -> dict[str, Any]:
    """Suggest safe values from the measurements, with explicit margins."""
    rec: dict[str, Any] = {}
    stats = {k: h.stats() for k, h in self._hist.items()}

    match = stats.get("sensor_match_count")
    if match is not None:
      if match.saturated:
        rec["contact_sensor_maxmatch"] = {
          "value": None,
          "reason": (
            f"Measurement is a LOWER BOUND: 'found' hit the configured cap "
            f"of {self._configured_maxmatch} on "
            f"{self._maxmatch_saturation_steps} steps. Re-measure with a "
            f"higher contact_sensor_maxmatch before trusting any reduction."
          ),
        }
      else:
        suggested = max(4, _next_power_of_two(max(1, match.maximum) * 4))
        rec["contact_sensor_maxmatch"] = {
          "value": suggested,
          "reason": (
            f"observed max match count = {match.maximum} (mean "
            f"{match.mean:.2f}); 4x margin rounded to a power of two."
          ),
        }

    njmax = stats.get("constraints_per_world")
    if njmax is not None:
      if self._njmax_overflow_steps > 0:
        rec["njmax"] = {
          "value": None,
          "reason": (
            f"OVERFLOW DETECTED: some world reached the configured njmax of "
            f"{self._configured_njmax} on {self._njmax_overflow_steps} "
            f"steps. Constraints were likely dropped -- RAISE njmax, do not "
            f"lower it, and treat any reward data from this run as suspect."
          ),
        }
      else:
        by_max = int(njmax.maximum) * 2
        by_p99 = int(njmax.p99) * 3
        suggested = max(64, _round_up(max(by_max, by_p99), 32))
        rec["njmax"] = {
          "value": suggested,
          "reason": (
            f"observed max = {njmax.maximum}, p99 = {njmax.p99}, p95 = "
            f"{njmax.p95}, mean = {njmax.mean:.1f}; max(2x max, 3x p99) "
            f"rounded up to a multiple of 32, floor 64."
          ),
        }

    if self._contact_dist_path is not None:
      rec["ccd_iterations"] = {
        "value": None,
        "reason": (
          f"Not inferable from a single run. Worst penetration observed was "
          f"{self._worst_penetration * 1000.0:.3f} mm at ccd_iterations="
          f"{self._env.cfg.sim.mujoco.ccd_iterations}. Compare this number "
          f"between presets: a reduction is safe only if penetration and the "
          f"hold-reward trace are unchanged."
        ),
      }

    return rec

  def report(self) -> str:
    """Human-readable measurement report."""
    data = self.summary()
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SIMULATION STATISTICS")
    lines.append("=" * 78)
    lines.append(
      f"records: {data['records']} env steps x {data['num_envs']} envs"
    )
    cfg = data["configured"]
    lines.append(
      f"configured: contact_sensor_maxmatch={cfg['contact_sensor_maxmatch']}, "
      f"njmax={cfg['njmax']}, nconmax={cfg['nconmax']}, "
      f"ccd_iterations={cfg['ccd_iterations']}"
    )
    lines.append("")

    if data["stats"]:
      header = (
        f"{'metric':<24}{'min':>7}{'mean':>9}{'p50':>7}{'p95':>7}"
        f"{'p99':>7}{'max':>7}{'cap':>7}"
      )
      lines.append(header)
      lines.append("-" * len(header))
      for key in sorted(data["stats"]):
        s = data["stats"][key]
        flag = "  <-- SATURATED" if s["saturated_samples"] else ""
        lines.append(
          f"{key:<24}{s['min']:>7}{s['mean']:>9.2f}{s['p50']:>7}"
          f"{s['p95']:>7}{s['p99']:>7}{s['max']:>7}{s['cap']:>7}{flag}"
        )
      lines.append("")
    else:
      lines.append("No statistics collected (no data sources available).")
      lines.append("")

    if data["worst_penetration_m"] < 0.0:
      lines.append(
        f"worst penetration: {data['worst_penetration_m'] * 1000.0:.4f} mm"
      )
      lines.append("")

    warn = data["warnings"]
    if warn["njmax_overflow_steps"]:
      lines.append(
        f"!! njmax OVERFLOW on {warn['njmax_overflow_steps']} steps. "
        f"Constraints were dropped; physics and rewards are unreliable."
      )
    if warn["maxmatch_saturation_steps"]:
      lines.append(
        f"!! contact sensor saturated on {warn['maxmatch_saturation_steps']} "
        f"steps. I_act is wrong and the hold reward is corrupted."
      )
    if warn["njmax_overflow_steps"] or warn["maxmatch_saturation_steps"]:
      lines.append("")

    if data["recommendations"]:
      lines.append("RECOMMENDATIONS")
      lines.append("-" * 78)
      for key, rec in sorted(data["recommendations"].items()):
        value = rec["value"]
        shown = "(cannot recommend)" if value is None else str(value)
        lines.append(f"{key} = {shown}")
        lines.append(f"  {rec['reason']}")
      lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


##
# Standalone margin helpers (used by scripts/benchmark.py --verify-margins).
##


def recommend_maxmatch(observed_max: int) -> int:
  """Safe ``contact_sensor_maxmatch`` for an observed max match count."""
  return max(4, _next_power_of_two(max(1, observed_max) * 4))


def recommend_njmax(observed_max: int, observed_p99: int) -> int:
  """Safe ``njmax`` for observed constraint counts."""
  return max(64, _round_up(max(observed_max * 2, observed_p99 * 3), 32))
