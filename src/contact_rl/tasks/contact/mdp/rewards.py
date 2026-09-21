"""Contact-explicit rewards for the locomotion task.

Implements the reach / hold / detach rewards of Omar & Khadiv (Section 3.2,
Eq. 1-3) plus the paper's auxiliary locomotion penalties and the goal-discovery
bonus (Section 4, "Locomotion").

Notation (per end-effector ``e``, time ``t``):
  * ``p^con_{e,t,1}`` : current desired contact location (goal, world frame).
  * ``p^act_{e,t}``   : actual foot (site) position, world frame.
  * ``I^con_{e,t,1}`` : current desired contact indicator (0/1).
  * ``I^act_{e,t}``   : actual contact indicator (0/1), from the contact sensor.
  * ``s``             : remaining command duration; ``nu`` the phase threshold.

The three feet rewards are summed over feet here (matching
``r^con = ... + sum_e (r_reach + r_hold + r_detach)``) and exposed as three
reward terms so each can be weighted / logged independently.

Contact ordering: the ``foot_site_ids`` from the command term and the columns
of ``ContactSensor.data.found`` must both follow the same foot order
(FL, FR, RL, RR). Neither resolver guarantees this on its own, so the
``check_foot_ordering`` startup event (see :mod:`~contact_rl.tasks.contact.mdp
.events`) asserts it at construction time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from contact_rl.tasks.contact.mdp.contact_command import ContactGoalCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _phase_inputs(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
):
  """Return (term, d2 [N,4], I_con1 [N,4], I_act [N,4], s [N])."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, ContactGoalCommand)
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]

  goal = term.goal_pos_w[:, :, 0, :]  # current target, [N, 4, 3]
  foot = asset.data.site_pos_w[:, term.foot_site_ids, :]  # [N, 4, 3]
  d2 = torch.sum(torch.square(goal - foot), dim=-1)  # [N, 4]

  i_con1 = term.current_contact_goal  # [N, 4]
  assert sensor.data.found is not None
  i_act = (sensor.data.found > 0).float()  # [N, 4]
  s = term.time_remaining  # [N]
  return term, d2, i_con1, i_act, s


def contact_reach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 1: ``exp(-d(p1, p_act)^2) * 1(I^con_1 == 0 and s < nu)``, summed over feet.

  Guides a swinging foot toward its upcoming contact location. ``std=1``
  reproduces the paper's literal ``exp(-d^2)``; smaller values sharpen it.
  """
  term, d2, i_con1, _, s = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  reach_mask = (i_con1 == 0) & (s.unsqueeze(-1) < term.cfg.nu)
  reward = torch.exp(-d2 / (std**2)) * reach_mask.float()
  return torch.sum(reward, dim=1)


def contact_hold(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  hold_lambda: float = 1.0,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 2: ``(1 + lambda*exp(-d^2)) * 1(I^con_1 == I^act == 1)``, summed over feet.

  Rewards maintaining commanded contact, with a bonus for holding at the right
  location (``lambda_hold > 0``).
  """
  _, d2, i_con1, i_act, _ = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  hold_mask = ((i_con1 == 1) & (i_act == 1)).float()
  reward = (1.0 + hold_lambda * torch.exp(-d2 / (std**2))) * hold_mask
  return torch.sum(reward, dim=1)


def contact_detach(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Eq. 3: ``1(I^con_1 == I^act == 0 and s > nu)``, summed over feet.

  A scalar reward incentivising the foot to make no contact during the detach
  phase.
  """
  term, _, i_con1, i_act, s = _phase_inputs(env, command_name, sensor_name, asset_cfg)
  detach_mask = (i_con1 == 0) & (i_act == 0) & (s.unsqueeze(-1) > term.cfg.nu)
  return torch.sum(detach_mask.float(), dim=1)


def goal_discovery_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Bonus for discovering more goals (base reaches the desired contacts).

  Fires once per switch when the base, projected to the ground, comes within a
  threshold of the current footholds (see :class:`ContactGoalCommand`).
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, ContactGoalCommand)
  return term.just_discovered


##
# Auxiliary locomotion penalties named by the paper that are not already in
# ``mjlab.envs.mdp`` (joint velocities / accelerations / torques / action rate
# are reused from there; these two are added here).
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
  assert default is not None
  jnt = asset_cfg.joint_ids
  return torch.sum(
    torch.square(asset.data.joint_pos[:, jnt] - default[:, jnt]), dim=1
  )
