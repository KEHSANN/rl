"""Contact-explicit rewards for the locomotion task.

Implements the reach / hold / detach rewards of Omar & Khadiv (Section 3.2,
Eq. 1-3) plus the paper's auxiliary locomotion penalties and the goal-discovery
bonus (Section 4, "Locomotion"):

  r_reach  = exp(-d(p1, p_act) / sigma^2) * 1[I^con_1 = 0 and s <= delta]      (1)
  r_hold   = (1 + a_hold * exp(-d / sigma^2)) * 1[I^con_1 = I^act = 1]         (2)
  r_detach = 1[I^con_1 = I^act = 0 and s > delta]                               (3)

``d`` is the *unsquared* L2 distance (paper, Eq. 1-2). ``kernel="gaussian"``
selects ``exp(-d^2 / sigma^2)`` -- the form of the previous revision -- for
ablations. Each term is summed over feet and exposed as its own reward term so
it can be weighted / logged independently.

Foot-column ordering is asserted at startup by ``check_foot_ordering``.

Posture regularisation (not in the paper, added after the 2000-iteration run
learned to walk on its hind knees): the contact rewards only see the foot
sites / foot geoms, so a kneeling leg whose foot sphere still touches the goal
is paid in full. ``undesired_contact_count``, ``base_height_below``,
``base_tilt_l2`` and ``knee_height_below`` close that loophole. The binary
contact count alone is not enough: a knee hovering a few mm above the floor
costs nothing, so ``knee_height_below`` adds a dense, geometric penalty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from contact_rl.tasks.contact.mdp.contact_command import ContactGoalCommand
from contact_rl.tasks.contact.mdp.planning import phase_masks, proximity_kernel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _term(env: ManagerBasedRlEnv, command_name: str) -> ContactGoalCommand:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ContactGoalCommand):
    raise TypeError(f"Command '{command_name}' is not a ContactGoalCommand.")
  return term


def _phase_inputs(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
):
  """Return (term, dist [N,4], reach, hold, detach masks [N,4])."""
  term = _term(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.found is None:
    raise RuntimeError(f"Sensor '{sensor_name}' must record the 'found' field.")
  goal = term.goal_pos_w[:, :, 0, :]  # current target, [N, 4, 3]
  foot = asset.data.site_pos_w[:, term.foot_site_ids, :]  # [N, 4, 3]
  dist = torch.norm(goal - foot, dim=-1)  # [N, 4]
  i_act = (sensor.data.found > 0).float()
  reach, hold, detach = phase_masks(
    term.current_contact_goal, i_act, term.time_remaining, term.cfg.delta
  )
  return term, dist, reach, hold, detach


def contact_reach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  sigma_sq: float = 0.1,
  kernel: str = "l2",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 1, summed over feet: guide a swinging foot to its upcoming contact."""
  _, dist, reach, _, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  return (proximity_kernel(dist, sigma_sq, kernel) * reach.float()).sum(dim=1)


def contact_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  hold_alpha: float = 1.0,
  sigma_sq: float = 0.1,
  kernel: str = "l2",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 2, summed over feet: keep commanded contact, bonus at the right spot."""
  _, dist, _, hold, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  k = proximity_kernel(dist, sigma_sq, kernel)
  return ((1.0 + hold_alpha * k) * hold.float()).sum(dim=1)


def contact_detach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 3, summed over feet: no contact during the detach phase."""
  _, _, _, _, detach = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  return detach.float().sum(dim=1)


def goal_discovery_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Bonus for discovering more goals (at most once per contact switch)."""
  return _term(env, command_name).just_discovered


##
# Auxiliary locomotion penalties named by the paper that are not already in
# ``mjlab.envs.mdp`` (joint vel / acc / torques / action rate come from there).
##


def base_angular_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base angular velocities (L2, body frame)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)


def joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviations from the default pose (L2)."""
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  jnt = asset_cfg.joint_ids
  return torch.sum(torch.square(asset.data.joint_pos[:, jnt] - default[:, jnt]), dim=1)


##
# Posture regularisation (anti-kneeling, see module docstring).
##


def undesired_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of non-foot collision geoms (base, hips, thighs, calves) touching
  the ground. ``sensor_name`` must be a ContactSensor recording ``found``."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.found is None:
    raise RuntimeError(f"Sensor '{sensor_name}' must record the 'found' field.")
  return (sensor.data.found > 0).float().sum(dim=1)


def base_height_below(
  env: ManagerBasedRlEnv,
  target: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """One-sided, linear: ``max(target - z_base, 0)`` (flat terrain).

  Linear rather than squared so small drops still register. One-sided so
  jumps are never penalised. NOTE: measured at the base centre, so kneeling on
  the hind legs only (front legs straight) barely moves it; the dense
  anti-kneeling signal is ``knee_height_below``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  return torch.clamp(target - z, min=0.0)


def base_tilt_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize a constant roll / pitch of the base (xy of projected gravity)."""
  asset: Entity = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b
  return torch.sum(torch.square(g[:, :2]), dim=1)


def knee_height_below(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_height: float = 0.08,
) -> torch.Tensor:
  """Dense anti-kneeling penalty (flat terrain), summed over legs.

  ``sum_legs max(min_height - z_knee, 0) / min_height`` in [0, n_legs], where
  ``z_knee`` is the world height of the calf body origin (= knee joint).
  ``asset_cfg.body_names`` must select the calf bodies (e.g. ``".*_calf"``).

  Go2: nominal knee height ~0.155 m, kneeling ~0.015 m. Unlike the binary
  ``undesired_contact_count`` it cannot be gamed by hovering the knee a few mm
  above the floor, and it gives a gradient before touch-down.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (torch.clamp(min_height - z, min=0.0) / min_height).sum(dim=1)
