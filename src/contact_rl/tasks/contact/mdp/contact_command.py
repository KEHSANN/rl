"""Contact-goal command for the contact-explicit locomotion task.

Implements the contact goals of Omar & Khadiv, *Learning to Act Through Contact:
A Unified View of Multi-Task Robot Learning* (LDC 2026, arXiv:2510.03599v2),
Section 3, for a quadruped end-effector set (the four feet).

For each end-effector ``e`` and time ``t`` the high-level planner provides
(Section 3.1):

* contact locations for a horizon of two contact switches
  ``p^con_{e,t} = {p^con_{e,t,1}, p^con_{e,t,2}}`` (current + next), and
* a binary contact indicator for two switches
  ``I^con_{e,t} = {I^con_{e,t,1}, I^con_{e,t,2}}`` (current + next), and
* the command duration ``S`` of the current contact goal.

The contact *phase* of an end-effector (Fig. 3) is derived from the current
indicator ``I^con_{e,t,1}`` and the remaining command time ``s`` against a
threshold ``nu``:

* **reach**  : ``I^con_{e,t,1} == 0``  and  ``s <  nu``
* **hold**   : ``I^con_{e,t,1} == 1``
* **detach** : ``I^con_{e,t,1} == 0``  and  ``s >  nu``

Locomotion command sampling (Section 4, "Locomotion"):

* stride lengths ``U(0.0, 0.3) m`` and stance widths ``U(0.1, 0.3) m`` per
  front / hind pair, **sampled once at initialisation** (not per reset) --
  the paper reports this yields a better policy;
* a sampled heading direction ``U[-pi, pi] rad`` **or** a sampled yaw rate
  ``U[-pi, pi] rad/s`` for curved paths;
* additional per-leg lateral + longitudinal offsets ``U(-0.15, 0.15) m``;
* command durations ``U[0.34, 0.36] s``.

The multi-gait behaviour (trot / pace / bound / jump / crawl) is produced by a
per-environment **contact sequence** (which feet are in stance at each switch);
the gait is likewise sampled once at initialisation.

Design notes (the paper specifies the goal *contents* and sampling, not the
exact foothold propagation; the following realises its description and is
documented so it can be tuned):

* Feet are laid out in a *travel frame* whose yaw is the desired travel/facing
  direction; the frame integrates the yaw rate for curved paths.
* A foot's world foothold marches forward by its pair's stride at *lift-off*
  (contact sequence transitions 1 -> 0), so that for the whole swing window
  ``p^con_{e,t,1}`` is the foothold the foot is travelling *to*. This matches
  Fig. 3: with ``I^con_{e,t,1} == 0`` the window runs *detach* (``s > nu``) and
  then *reach* (``s < nu``), and the reach reward must pull the foot forward to
  its upcoming contact -- not back to the one it just left. Between switches
  the goal is piecewise-constant, giving the policy a stable target.
* Goals also advance early (with a discovery bonus) when the base, projected to
  the ground, comes within a threshold of the current footholds -- the paper's
  "update the goals if the robot's base ... remains within a threshold ... and
  provide a bonus reward for discovering more goals". The base must *remain*
  within the threshold until at least ``goal_dwell_frac`` of the command window
  has elapsed, which bounds the early-advance rate (see
  :attr:`ContactGoalCommandCfg.goal_dwell_frac`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Feet ordered FL, FR, RL, RR (matches go2.xml sites / geoms).
_FOOT_ORDER = ("FL", "FR", "RL", "RR")
_NUM_FEET = 4
_NUM_SWITCHES = 2  # Horizon of two contact switches (current + next).

# Gait identifiers.
GAITS = ("trot", "pace", "bound", "jump", "crawl")

# Per-gait current-contact patterns over one period P: 1 = stance/contact.
# Rows are switches (length P), columns are feet (FL, FR, RL, RR).
_GAIT_PATTERNS: dict[str, list[list[int]]] = {
  # Diagonal pairs.
  "trot": [[1, 0, 0, 1], [0, 1, 1, 0]],
  # Lateral pairs.
  "pace": [[1, 0, 1, 0], [0, 1, 0, 1]],
  # Front / hind pairs.
  "bound": [[1, 1, 0, 0], [0, 0, 1, 1]],
  # Full flight phase.
  "jump": [[1, 1, 1, 1], [0, 0, 0, 0]],
  # One foot swings at a time (statically stable): FL, RR, FR, RL.
  "crawl": [[0, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
}


class ContactGoalCommand(CommandTerm):
  """Generates per-foot contact goals (locations + indicators + duration)."""

  cfg: ContactGoalCommandCfg

  def __init__(self, cfg: ContactGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._dt = env.step_dt

    # ``preserve_order=True`` is required: the resolver defaults to *model*
    # order, so without it the goal / reward / observation columns would follow
    # the MJCF declaration order instead of ``cfg.foot_site_names``. Today they
    # happen to coincide for go2.xml; a reordered MJCF would silently permute
    # the feet.
    site_ids, site_names = self.robot.find_sites(
      list(cfg.foot_site_names), preserve_order=True
    )
    if tuple(site_names) != tuple(cfg.foot_site_names):
      raise ValueError(
        f"Resolved foot sites {tuple(site_names)} do not match the requested "
        f"order {tuple(cfg.foot_site_names)}."
      )
    self._foot_site_ids = site_ids

    N, dev = self.num_envs, self.device

    # --- Foot geometry (constant), feet ordered FL, FR, RL, RR. ---
    # Longitudinal hip offset (from go2.xml body positions) and lateral side.
    self._hip_x = torch.tensor([0.1934, 0.1934, -0.1934, -0.1934], device=dev)
    self._side = torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)  # L:+  R:-
    self._pair = torch.tensor([0, 0, 1, 1], device=dev, dtype=torch.long)  # front/hind

    # --- Gait pattern lookup tensors. ---
    # Pad each pattern to the max period so we can index by gait and switch.
    self._periods = torch.tensor(
      [len(_GAIT_PATTERNS[g]) for g in GAITS], device=dev, dtype=torch.long
    )
    max_p = int(self._periods.max().item())
    patterns = torch.zeros(len(GAITS), max_p, _NUM_FEET, device=dev)
    for gi, g in enumerate(GAITS):
      pat = _GAIT_PATTERNS[g]
      for k in range(max_p):
        patterns[gi, k] = torch.tensor(pat[k % len(pat)], device=dev, dtype=torch.float)
    self._patterns = patterns  # [G, max_p, 4]

    # --- Per-environment constants (sampled once, at construction). ---
    r = torch.empty(N, device=dev)
    self._gait = torch.randint(0, len(GAITS), (N,), device=dev)
    if cfg.gaits is not None:
      allowed = torch.tensor(
        [GAITS.index(g) for g in cfg.gaits], device=dev, dtype=torch.long
      )
      self._gait = allowed[torch.randint(0, len(allowed), (N,), device=dev)]
    self._stride = torch.stack(
      [
        r.clone().uniform_(*cfg.stride_length_range),  # front
        torch.empty(N, device=dev).uniform_(*cfg.stride_length_range),  # hind
      ],
      dim=1,
    )  # [N, 2]
    self._stance = torch.stack(
      [
        torch.empty(N, device=dev).uniform_(*cfg.stance_width_range),
        torch.empty(N, device=dev).uniform_(*cfg.stance_width_range),
      ],
      dim=1,
    )  # [N, 2]
    self._use_yaw_rate = torch.empty(N, device=dev).uniform_(0, 1) < cfg.rel_yaw_rate_envs
    self._heading = torch.empty(N, device=dev).uniform_(*cfg.heading_range)
    self._yaw_rate = torch.empty(N, device=dev).uniform_(*cfg.yaw_rate_range)
    self._leg_off = torch.empty(N, _NUM_FEET, 2, device=dev).uniform_(
      *cfg.leg_offset_range
    )  # [N, 4, (lon, lat)]

    # --- Per-episode plan state. ---
    self._travel_yaw = torch.zeros(N, device=dev)
    self._foot_xy = torch.zeros(N, _NUM_FEET, 2, device=dev)  # current world foothold
    self._switch = torch.zeros(N, device=dev, dtype=torch.long)
    self._prev_cur_contact = torch.zeros(N, _NUM_FEET, device=dev)

    # --- Goal buffers (world). ---
    self._goal_pos_w = torch.zeros(N, _NUM_FEET, _NUM_SWITCHES, 3, device=dev)
    self._goal_contact = torch.zeros(N, _NUM_FEET, _NUM_SWITCHES, device=dev)

    # --- Discovery bonus bookkeeping. ---
    self.just_discovered = torch.zeros(N, device=dev)
    self._discovered_this_switch = torch.zeros(N, device=dev, dtype=torch.bool)
    self.goals_discovered = torch.zeros(N, device=dev)
    # An early goal advance requires ``goal_dwell_frac`` of the command window
    # to have elapsed, i.e. ``time_left`` to have dropped below this value. The
    # upper end of the resampling range is used as the reference so that
    # ``goal_dwell_frac == 0`` opens the gate for every env on every tick.
    self._dwell_time_left_max = (
      1.0 - cfg.goal_dwell_frac
    ) * cfg.resampling_time_range[1]

    # Flat command vector: per foot [p1(3), p2(3), I1, I2] + duration.
    self._command = torch.zeros(N, _NUM_FEET * 8 + 1, device=dev)

    self.metrics["tracking_error"] = torch.zeros(N, device=dev)
    self.metrics["goals_discovered"] = torch.zeros(N, device=dev)

  # ----------------------------------------------------------------- helpers

  def _pattern_at(self, gait: torch.Tensor, switch: torch.Tensor) -> torch.Tensor:
    """Current-contact row [len, 4] for given gait ids and (cyclic) switch idx."""
    period = self._periods[gait]
    k = torch.remainder(switch, period)
    return self._patterns[gait, k]  # [len, 4]

  def _foot_nominal_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Nominal stance foothold (world xy) in the travel frame, [len, 4, 2]."""
    yaw = self._travel_yaw[env_ids]  # [n]
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd = torch.stack([c, s], dim=-1)  # [n, 2]
    lat = torch.stack([-s, c], dim=-1)  # [n, 2]
    stance = self._stance[env_ids]  # [n, 2] (front, hind)
    stance_f = stance.gather(1, self._pair.unsqueeze(0).expand(len(env_ids), -1))  # [n,4]
    lon = self._hip_x.unsqueeze(0) + self._leg_off[env_ids, :, 0]  # [n, 4]
    off_lat = self._side.unsqueeze(0) * (stance_f * 0.5) + self._leg_off[env_ids, :, 1]
    base_xy = self.robot.data.root_link_pos_w[env_ids, :2]  # [n, 2]
    # world = base + fwd * lon + lat * off_lat  (broadcast over feet)
    return (
      base_xy.unsqueeze(1)
      + fwd.unsqueeze(1) * lon.unsqueeze(-1)
      + lat.unsqueeze(1) * off_lat.unsqueeze(-1)
    )  # [n, 4, 2]

  def _travel_dir(self, env_ids: torch.Tensor) -> torch.Tensor:
    yaw = self._travel_yaw[env_ids]
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # [n, 2]

  # ------------------------------------------------------------- CommandTerm

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def num_feet(self) -> int:
    return _NUM_FEET

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Advance the contact plan by one switch for ``env_ids``.

    Called at construction-time reset and whenever the command duration expires
    (``resampling_time_range == (0.34, 0.36) s``) or a goal is discovered early.
    """
    n = len(env_ids)
    if n == 0:
      return

    first = self.command_counter[env_ids] == 0  # True on (re)initialisation.

    # (Re)initialise the travel frame + footholds for freshly reset envs.
    if first.any():
      init_ids = env_ids[first]
      heading0 = self.robot.data.heading_w[init_ids]
      straight = ~self._use_yaw_rate[init_ids]
      self._travel_yaw[init_ids] = torch.where(
        straight, heading0 + self._heading[init_ids], heading0
      )
      self._switch[init_ids] = 0
      self._foot_xy[init_ids] = self._foot_nominal_xy(init_ids)
      # Treat every foot as having "just lifted" so that the feet which are in
      # swing at switch 0 march their foothold once below, exactly like they
      # would at any later lift-off. Feet in stance at switch 0 are unaffected
      # (the lift-off mask needs ``cur == 0``).
      self._prev_cur_contact[init_ids] = 1.0
      self.goals_discovered[init_ids] = 0.0

    # Advance the switch counter for envs already running.
    cont = env_ids[~first]
    if len(cont) > 0:
      self._switch[cont] += 1

    # New current-contact row after advancing.
    cur = self._pattern_at(self._gait[env_ids], self._switch[env_ids])  # [n, 4]

    # A foot that just lifted off (1 -> 0) marches its foothold forward by its
    # pair's stride, so p1 is the foothold it is swinging *to* (see module
    # docstring). Marching at landing instead would leave p1 at the departure
    # point for the whole swing, and the reach reward (Eq. 1) would pull the
    # swing foot backwards.
    lifted = (self._prev_cur_contact[env_ids] == 1) & (cur == 0)  # [n, 4]
    stride = self._stride[env_ids]  # [n, 2]
    stride_f = stride.gather(1, self._pair.unsqueeze(0).expand(n, -1))  # [n, 4]
    step = self._travel_dir(env_ids).unsqueeze(1) * stride_f.unsqueeze(-1)  # [n,4,2]
    self._foot_xy[env_ids] = self._foot_xy[env_ids] + lifted.unsqueeze(-1) * step

    # Next-switch current-contact row; a foot that lifts off at that switch gets
    # a marched foothold, everything else keeps p1 (a swinging foot lands on p1
    # and then holds it).
    nxt = self._pattern_at(self._gait[env_ids], self._switch[env_ids] + 1)  # [n, 4]
    lifts_next = (cur == 1) & (nxt == 0)  # [n, 4]
    p1_xy = self._foot_xy[env_ids]  # [n, 4, 2]
    p2_xy = p1_xy + lifts_next.unsqueeze(-1) * step  # [n, 4, 2]

    z = self.cfg.foot_target_height
    self._goal_pos_w[env_ids, :, 0, :2] = p1_xy
    self._goal_pos_w[env_ids, :, 1, :2] = p2_xy
    self._goal_pos_w[env_ids, :, :, 2] = z
    self._goal_contact[env_ids, :, 0] = cur
    self._goal_contact[env_ids, :, 1] = nxt

    self._prev_cur_contact[env_ids] = cur
    self._discovered_this_switch[env_ids] = False

  def _update_command(self) -> None:
    """Per-step update: integrate curved heading, build obs vector, discovery."""
    # Integrate yaw rate for curved-path environments.
    turning = self._use_yaw_rate
    if turning.any():
      self._travel_yaw = self._travel_yaw + turning.float() * self._yaw_rate * self._dt

    # Base-frame (yaw-only) contact locations for the observation.
    goal_b = self._world_to_base(self._goal_pos_w)  # [N, 4, 2, 3]

    # Fill the flat command in place (the observation manager clones it).
    n_pos = _NUM_FEET * _NUM_SWITCHES * 3
    n_con = _NUM_FEET * _NUM_SWITCHES
    self._command[:, :n_pos] = goal_b.reshape(self.num_envs, n_pos)
    self._command[:, n_pos : n_pos + n_con] = self._goal_contact.reshape(
      self.num_envs, n_con
    )
    self._command[:, -1] = torch.clamp(self.time_left, min=0.0)

    # Goal discovery: base (projected to ground) within threshold of the
    # centroid of the *current* footholds -> advance goals early + bonus.
    #
    # The dwell gate implements the paper's "*remains* within a threshold": an
    # early advance is only allowed once ``goal_dwell_frac`` of the nominal
    # command window has elapsed. Without it the plan can advance on every
    # control step (50 Hz instead of ~2.9 Hz) because ``_resample_command``
    # clears ``_discovered_this_switch``; for a near-zero stride the centroid
    # barely moves, so it would never stop, and for a normal stride the plan
    # jumps several switches at once -- an odd jump inverts the phase of a
    # two-phase gait.
    base_xy = self.robot.data.root_link_pos_w[:, :2]
    centroid = self._goal_pos_w[:, :, 0, :2].mean(dim=1)  # [N, 2]
    dist = torch.norm(base_xy - centroid, dim=-1)
    dwell_ok = self.time_left < self._dwell_time_left_max
    reached = (
      (dist < self.cfg.goal_proximity_threshold)
      & (~self._discovered_this_switch)
      & dwell_ok
    )
    self.just_discovered.copy_(reached)
    if reached.any():
      self._discovered_this_switch[reached] = True
      self.goals_discovered[reached] += 1.0
      # Force an early resample (goal update) on the next compute() tick.
      self.time_left[reached] = 0.0

  def _update_metrics(self) -> None:
    foot_w = self.robot.data.site_pos_w[:, self._foot_site_ids, :]  # [N, 4, 3]
    err = torch.norm(self._goal_pos_w[:, :, 0, :] - foot_w, dim=-1).mean(dim=1)
    self.metrics["tracking_error"] = err
    self.metrics["goals_discovered"] = self.goals_discovered

  # ------------------------------------------------------------- debug vis

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the contact goals: p1 as a solid sphere, p2 faded and smaller.

    Green = contact commanded (hold), red = no contact commanded (detach then
    reach). For a swinging foot an orange arrow points from the foot site to
    ``p1``, i.e. exactly the vector the reach reward shrinks -- it should point
    *forward*, ahead of the robot.
    """
    goal = self._goal_pos_w
    con = self._goal_contact
    feet = self.robot.data.site_pos_w
    radius = 0.03
    for i in visualizer.get_env_indices(self.num_envs):
      for f in range(_NUM_FEET):
        stance = bool(con[i, f, 0] > 0.5)
        c1 = (0.1, 0.9, 0.2, 0.85) if stance else (0.95, 0.25, 0.2, 0.85)
        visualizer.add_sphere(goal[i, f, 0], radius, c1, label=f"{_FOOT_ORDER[f]}_p1")
        stance_next = bool(con[i, f, 1] > 0.5)
        c2 = (0.1, 0.9, 0.2, 0.3) if stance_next else (0.95, 0.25, 0.2, 0.3)
        visualizer.add_sphere(
          goal[i, f, 1], 0.6 * radius, c2, label=f"{_FOOT_ORDER[f]}_p2"
        )
        if not stance:
          visualizer.add_arrow(
            feet[i, self._foot_site_ids[f]],
            goal[i, f, 0],
            (1.0, 0.6, 0.0, 0.9),
            width=0.008,
            label=f"{_FOOT_ORDER[f]}_reach",
          )

  # ---------------------------------------------------------------- utils

  def _world_to_base(self, pos_w: torch.Tensor) -> torch.Tensor:
    """Yaw-only world->base transform of positions ``[N, ..., 3]``."""
    base_pos = self.robot.data.root_link_pos_w  # [N, 3]
    yaw = self.robot.data.heading_w  # [N]
    c, s = torch.cos(yaw), torch.sin(yaw)
    extra = pos_w.dim() - 2
    view = (self.num_envs,) + (1,) * extra + (3,)
    rel = pos_w - base_pos.view(view)
    rx, ry, rz = rel[..., 0], rel[..., 1], rel[..., 2]
    cc = c.view((self.num_envs,) + (1,) * extra)
    ss = s.view((self.num_envs,) + (1,) * extra)
    xb = cc * rx + ss * ry
    yb = -ss * rx + cc * ry
    return torch.stack([xb, yb, rz], dim=-1)

  # --------------------------------------------------------- goal accessors
  # (Read by the reward / observation terms via
  #  ``env.command_manager.get_term("contact")``.)

  @property
  def goal_pos_w(self) -> torch.Tensor:
    """World contact locations, ``[N, 4, 2, 3]`` (foot, {current,next}, xyz)."""
    return self._goal_pos_w

  @property
  def current_contact_goal(self) -> torch.Tensor:
    """Current binary contact indicator ``I^con_{e,t,1}``, ``[N, 4]``."""
    return self._goal_contact[:, :, 0]

  @property
  def next_contact_goal(self) -> torch.Tensor:
    return self._goal_contact[:, :, 1]

  @property
  def time_remaining(self) -> torch.Tensor:
    """Remaining command duration ``s``, ``[N]``."""
    return self.time_left

  @property
  def foot_site_ids(self):
    return self._foot_site_ids


@dataclass(kw_only=True)
class ContactGoalCommandCfg(CommandTermCfg):
  """Configuration for :class:`ContactGoalCommand`."""

  entity_name: str = "robot"
  foot_site_names: tuple[str, ...] = _FOOT_ORDER

  # Command duration S ~ U[0.34, 0.36] s (paper, Section 4).
  resampling_time_range: tuple[float, float] = (0.34, 0.36)

  # Sampled once at construction (paper: "sampling ... once during
  # initialisation leads to a better policy").
  stride_length_range: tuple[float, float] = (0.0, 0.3)
  stance_width_range: tuple[float, float] = (0.1, 0.3)
  leg_offset_range: tuple[float, float] = (-0.15, 0.15)
  heading_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
  yaw_rate_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
  rel_yaw_rate_envs: float = 0.5
  """Fraction of environments that follow a curved path (yaw rate) instead of a
  fixed heading."""

  gaits: tuple[str, ...] | None = None
  """Subset of gaits to sample from; ``None`` uses all of
  ``('trot', 'pace', 'bound', 'jump', 'crawl')``."""

  # Contact phase threshold nu (on remaining command duration s).
  nu: float = 0.18

  foot_target_height: float = 0.02
  """Target z of a planted foot site (approx. foot sphere radius)."""

  goal_proximity_threshold: float = 0.15
  """Base-to-goal distance under which goals advance early (discovery)."""

  goal_dwell_frac: float = 0.5
  """Fraction of the nominal command window that must elapse before a goal is
  allowed to advance early.

  Implements the paper's "*remains* within a threshold". ``0.0`` disables the
  gate (and lets the plan advance every control step); ``0.5`` caps the
  early-advance rate at twice the nominal switch rate."""

  hold_lambda: float = 1.0
  """``lambda_hold`` weight on the contact-location term of the hold reward."""

  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> ContactGoalCommand:
    return ContactGoalCommand(self, env)
