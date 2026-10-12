"""Pure-torch contact planning + reward math (no mjlab dependency).

This module holds everything about the contact-explicit task that can be
expressed without a simulator, so it can be unit-tested in isolation:

* the per-gait **contact sequences** (Section 3.2: the stacked ``I^con``),
* :class:`GaitPlanner`, the prespecified high-level planner that turns the
  once-per-environment sampled command parameters (Section 4, "Locomotion")
  into contact goals for a horizon of two contact switches,
* the paper's proximity kernel ``exp(-d / sigma^2)`` (Eq. 1-2) and the
  reach / hold / detach phase masks (Eq. 1-3),
* ``yaw_from_quat_wxyz`` to read the base heading straight from ``qpos``.

Planner design (the paper specifies the goal *contents* and the sampling
distributions, not the foothold propagation; this is the documented choice):

A per-environment **reference frame** ``(ref_xy, ref_yaw)`` -- a virtual base
-- advances by ``mean_stride / P`` per contact switch along the travel
direction (``P`` = gait period in switches, so every foot moves exactly one
mean stride per gait cycle) and rotates by ``yaw_rate * S`` per switch on
curved paths. A foot that lifts off is assigned the foothold

    ref(t_mid) (+) R(yaw(t_mid)) @ nominal_e  +  0.5 * (stride_pair - mean) * travel_dir

where ``t_mid`` is the middle of the stance that follows the swing (the hip
passes over the foot at mid-stance) and ``nominal_e`` is the sampled stance
layout (hip x, stance width, per-leg offsets). Consequences, all of which the
previous implementation violated:

* front and hind footholds never drift apart, even when the front and hind
  strides are sampled differently (previously each pair marched by its own
  stride, so the pairs separated by ``|s_f - s_h|`` every cycle);
* the stance layout rotates with the heading on curved paths (previously the
  foothold rectangle kept its initial world orientation while the path turned);
* the robot can be asked to move in *any* direction relative to its body
  ("contact locations are sampled to move in all directions"): the layout is
  expressed in the body-facing frame and the travel direction is facing +
  sampled heading offset.
"""

from __future__ import annotations

import math
from typing import Any

import torch

NUM_FEET = 4
NUM_SWITCHES = 2  # Horizon of two contact switches (current + next).
FOOT_ORDER = ("FL", "FR", "RL", "RR")

GAITS = ("trot", "pace", "bound", "jump", "crawl")

# Per-gait contact sequences over one period P: 1 = stance/contact.
# Rows are switches, columns are feet (FL, FR, RL, RR). Every foot swings for
# exactly one switch per period (the planner relies on this).
GAIT_PATTERNS: dict[str, list[list[int]]] = {
  "trot": [[1, 0, 0, 1], [0, 1, 1, 0]],  # Diagonal pairs.
  "pace": [[1, 0, 1, 0], [0, 1, 0, 1]],  # Lateral pairs.
  "bound": [[1, 1, 0, 0], [0, 0, 1, 1]],  # Front / hind pairs.
  "jump": [[1, 1, 1, 1], [0, 0, 0, 0]],  # Full flight phase.
  # One foot swings at a time (statically stable): FL, RR, FR, RL.
  "crawl": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
}

# Go2 hip x-offsets (from go2.xml body positions) and lateral side sign.
GO2_HIP_X = (0.1934, 0.1934, -0.1934, -0.1934)
SIDE = (1.0, -1.0, 1.0, -1.0)  # Left +, right -.
PAIR = (0, 0, 1, 1)  # 0 = front pair, 1 = hind pair.


def _validate_patterns() -> None:
  for name, pat in GAIT_PATTERNS.items():
    for foot in range(NUM_FEET):
      swings = sum(1 - row[foot] for row in pat)
      if swings != 1:
        raise ValueError(f"Gait '{name}': foot {foot} swings {swings}x per period.")


_validate_patterns()


# --------------------------------------------------------------------- math


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
  """Heading (yaw) of a MuJoCo ``(w, x, y, z)`` quaternion, ``[..., 4] -> [...]``.

  Identical to ``atan2`` of the body x-axis projected on the ground, i.e. to
  mjlab's ``EntityData.heading_w``.
  """
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot2d(yaw: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
  """Rotate ``xy [..., 2]`` by ``yaw`` (broadcast over the leading dims)."""
  c, s = torch.cos(yaw), torch.sin(yaw)
  x, y = xy[..., 0], xy[..., 1]
  return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def unit(yaw: torch.Tensor) -> torch.Tensor:
  return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def arc_displacement(
  theta0: torch.Tensor, w: torch.Tensor, v: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
  """Exact displacement ``[n, 2]`` after ``tau`` switches along a constant-
  curvature path: speed ``v`` and yaw rate ``w`` (both per switch), initial
  travel direction ``theta0``. Additive in ``tau``, which keeps the planner's
  lookahead goal ``p2`` identical to the goal ``p1`` realised one switch later.
  """
  eps = 1e-6
  straight = torch.abs(w) < eps
  w_safe = torch.where(straight, torch.full_like(w, eps), w)
  th1 = theta0 + w_safe * tau
  r = v / w_safe
  arc = torch.stack(
    [r * (torch.sin(th1) - torch.sin(theta0)), r * (torch.cos(theta0) - torch.cos(th1))],
    dim=-1,
  )
  line = (v * tau).unsqueeze(-1) * unit(theta0)
  return torch.where(straight.unsqueeze(-1), line, arc)


def proximity_kernel(dist: torch.Tensor, sigma_sq: float, kernel: str) -> torch.Tensor:
  """Spatial kernel of Eq. 1-2.

  ``kernel="l2"`` is the paper's literal ``exp(-d / sigma^2)`` with ``d`` the
  (unsquared) L2 distance. ``kernel="gaussian"`` is ``exp(-d^2 / sigma^2)``,
  the form used by the previous revision of this repository (kept for
  ablations).
  """
  if sigma_sq <= 0.0:
    raise ValueError(f"sigma_sq must be > 0, got {sigma_sq}.")
  if kernel == "l2":
    return torch.exp(-dist / sigma_sq)
  if kernel == "gaussian":
    return torch.exp(-torch.square(dist) / sigma_sq)
  raise ValueError(f"Unknown kernel '{kernel}' (expected 'l2' or 'gaussian').")


def phase_masks(
  i_con1: torch.Tensor, i_act: torch.Tensor, s: torch.Tensor, delta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Boolean reach / hold / detach masks ``[N, F]`` of Eq. 1-3.

  * reach  : ``I^con_1 = 0  and  s <= delta``           (Eq. 1)
  * hold   : ``I^con_1 = I^act = 1``                    (Eq. 2)
  * detach : ``I^con_1 = I^act = 0  and  s > delta``    (Eq. 3)
  """
  s_ = s.unsqueeze(-1)
  swing = i_con1 < 0.5
  act = i_act > 0.5
  reach = swing & (s_ <= delta)
  hold = (~swing) & act
  detach = swing & (~act) & (s_ > delta)
  return reach, hold, detach


# ------------------------------------------------------------------ planner


class GaitPlanner:
  """Prespecified contact planner (the paper's "high-level planner").

  ``cfg`` is duck-typed; it needs ``stride_length_range``,
  ``stance_width_range``, ``leg_offset_range``, ``heading_range``,
  ``yaw_rate_range``, ``rel_yaw_rate_envs``, ``gaits`` and
  ``resampling_time_range``. Command parameters are sampled **once** here
  (paper: sampling once at initialisation beats per-reset sampling).
  """

  def __init__(
    self,
    cfg: Any,
    num_envs: int,
    device: str | torch.device,
    hip_x: tuple[float, ...] = GO2_HIP_X,
    generator: torch.Generator | None = None,
  ):
    N, dev = num_envs, device
    self.num_envs = N
    self.device = dev
    g = generator

    def U(shape, lo_hi):
      lo, hi = lo_hi
      return torch.rand(shape, device=dev, generator=g) * (hi - lo) + lo

    self._hip_x = torch.tensor(hip_x, device=dev)
    self._side = torch.tensor(SIDE, device=dev)
    self._pair = torch.tensor(PAIR, device=dev, dtype=torch.long)

    periods = [len(GAIT_PATTERNS[name]) for name in GAITS]
    self._periods = torch.tensor(periods, device=dev, dtype=torch.long)
    max_p = max(periods)
    pats = torch.zeros(len(GAITS), max_p, NUM_FEET, device=dev)
    for gi, name in enumerate(GAITS):
      pat = GAIT_PATTERNS[name]
      for k in range(max_p):
        pats[gi, k] = torch.tensor(pat[k % len(pat)], dtype=torch.float, device=dev)
    self._patterns = pats  # [G, max_p, 4]

    # --- Per-environment command parameters (sampled once). ---
    allowed = tuple(cfg.gaits) if cfg.gaits is not None else GAITS
    unknown = set(allowed) - set(GAITS)
    if unknown:
      raise ValueError(f"Unknown gaits {sorted(unknown)}; choose from {GAITS}.")
    allowed_ids = torch.tensor([GAITS.index(x) for x in allowed], device=dev)
    self.gait = allowed_ids[
      torch.randint(0, len(allowed_ids), (N,), device=dev, generator=g)
    ]
    self.stride = U((N, 2), cfg.stride_length_range)  # [N, (front, hind)]
    self.stance = U((N, 2), cfg.stance_width_range)  # [N, (front, hind)]
    self.leg_off = U((N, NUM_FEET, 2), cfg.leg_offset_range)  # [N, 4, (lon, lat)]
    use_yaw_rate = torch.rand(N, device=dev, generator=g) < cfg.rel_yaw_rate_envs
    self.use_yaw_rate = use_yaw_rate
    # Straight envs: fixed travel offset w.r.t. facing, no rotation.
    # Curved envs: travel along the facing direction, rotating at yaw_rate.
    self.heading_off = torch.where(
      use_yaw_rate, torch.zeros(N, device=dev), U((N,), cfg.heading_range)
    )
    self.yaw_rate = torch.where(
      use_yaw_rate, U((N,), cfg.yaw_rate_range), torch.zeros(N, device=dev)
    )
    lo, hi = cfg.resampling_time_range
    self.switch_dt = 0.5 * (lo + hi)  # Nominal command duration S.

    # --- Per-episode plan state. ---
    self.ref_xy = torch.zeros(N, 2, device=dev)
    self.ref_yaw = torch.zeros(N, device=dev)
    self.switch = torch.zeros(N, device=dev, dtype=torch.long)
    self.foot_xy = torch.zeros(N, NUM_FEET, 2, device=dev)  # p1 (world xy)
    self.next_xy = torch.zeros(N, NUM_FEET, 2, device=dev)  # p2 (world xy)
    self.cur_contact = torch.zeros(N, NUM_FEET, device=dev)  # I^con_1
    self.next_contact = torch.zeros(N, NUM_FEET, device=dev)  # I^con_2

  # ------------------------------------------------------------ helpers

  def pattern_at(self, ids: torch.Tensor, switch: torch.Tensor) -> torch.Tensor:
    gait = self.gait[ids]
    k = torch.remainder(switch, self._periods[gait])
    return self._patterns[gait, k]  # [n, 4]

  def period(self, ids: torch.Tensor) -> torch.Tensor:
    return self._periods[self.gait[ids]].float()  # [n]

  def mean_stride(self, ids: torch.Tensor) -> torch.Tensor:
    return self.stride[ids].mean(dim=1)  # [n]

  def nominal_offsets(self, ids: torch.Tensor) -> torch.Tensor:
    """Stance layout in the body-facing frame, ``[n, 4, 2]`` (lon, lat)."""
    n = len(ids)
    pair_idx = self._pair.unsqueeze(0).expand(n, -1)
    stance_f = self.stance[ids].gather(1, pair_idx)  # [n, 4]
    lon = self._hip_x.unsqueeze(0) + self.leg_off[ids, :, 0]
    lat = self._side.unsqueeze(0) * 0.5 * stance_f + self.leg_off[ids, :, 1]
    return torch.stack([lon, lat], dim=-1)

  def footholds(self, ids: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Footholds of all feet for a stance centred ``tau`` switches ahead of
    the current reference, ``tau: [n]`` -> ``[n, 4, 2]`` (world xy)."""
    n = len(ids)
    w_s = self.yaw_rate[ids] * self.switch_dt  # yaw change per switch
    yaw0 = self.ref_yaw[ids]
    yaw_pred = yaw0 + w_s * tau
    mean = self.mean_stride(ids)
    v = mean / self.period(ids)  # reference speed, metres per switch
    ref_pred = self.ref_xy[ids] + arc_displacement(
      yaw0 + self.heading_off[ids], w_s, v, tau
    )
    layout = rot2d(yaw_pred.view(n, 1), self.nominal_offsets(ids))  # [n, 4, 2]
    pair_idx = self._pair.unsqueeze(0).expand(n, -1)
    asym = 0.5 * (self.stride[ids].gather(1, pair_idx) - mean.unsqueeze(-1))
    travel_pred = unit(yaw_pred + self.heading_off[ids])  # [n, 2]
    return (
      ref_pred.unsqueeze(1)
      + layout
      + asym.unsqueeze(-1) * travel_pred.unsqueeze(1)
    )

  def _landing_tau(self, ids: torch.Tensor) -> torch.Tensor:
    # A foot swinging during the current switch lands at the next switch and
    # stands for P-1 switches; the hip is over it at mid-stance.
    return 1.0 + 0.5 * (self.period(ids) - 1.0)

  def _refresh_goals(self, ids: torch.Tensor, lifted: torch.Tensor) -> None:
    cur = self.pattern_at(ids, self.switch[ids])
    nxt = self.pattern_at(ids, self.switch[ids] + 1)
    tau_land = self._landing_tau(ids)
    land = self.footholds(ids, tau_land)
    foot = torch.where(lifted.unsqueeze(-1), land, self.foot_xy[ids])
    lifts_next = (cur > 0.5) & (nxt < 0.5)
    land_next = self.footholds(ids, tau_land + 1.0)
    self.foot_xy[ids] = foot
    self.next_xy[ids] = torch.where(lifts_next.unsqueeze(-1), land_next, foot)
    self.cur_contact[ids] = cur
    self.next_contact[ids] = nxt

  # --------------------------------------------------------------- API

  def reset(self, ids: torch.Tensor, base_xy: torch.Tensor, base_yaw: torch.Tensor):
    """Anchor the plan to the (freshly reset) base pose of ``ids``."""
    if len(ids) == 0:
      return
    self.ref_xy[ids] = base_xy
    self.ref_yaw[ids] = base_yaw
    self.switch[ids] = 0
    self.foot_xy[ids] = self.footholds(ids, torch.zeros(len(ids), device=self.device))
    cur = self.pattern_at(ids, self.switch[ids])
    # Feet that swing at switch 0 are treated as having just lifted off.
    self._refresh_goals(ids, lifted=cur < 0.5)

  def advance(self, ids: torch.Tensor) -> None:
    """Advance the plan by one contact switch for ``ids``."""
    if len(ids) == 0:
      return
    prev = self.cur_contact[ids]
    w_s = self.yaw_rate[ids] * self.switch_dt
    v = self.mean_stride(ids) / self.period(ids)
    one = torch.ones(len(ids), device=self.device)
    self.ref_xy[ids] = self.ref_xy[ids] + arc_displacement(
      self.ref_yaw[ids] + self.heading_off[ids], w_s, v, one
    )
    self.ref_yaw[ids] = (
      torch.remainder(self.ref_yaw[ids] + w_s + math.pi, 2 * math.pi) - math.pi
    )
    self.switch[ids] = self.switch[ids] + 1
    cur = self.pattern_at(ids, self.switch[ids])
    lifted = (prev > 0.5) & (cur < 0.5)
    self._refresh_goals(ids, lifted=lifted)
