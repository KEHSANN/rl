"""Contact-explicit observations for the locomotion task.

The bulk of the task observation (current + next contact sequence, current +
next contact locations in base frame, command duration) is produced by the
:class:`ContactGoalCommand` and read through ``mdp.generated_commands``. This
module adds the remaining piece the paper lists -- the *relative distance of the
robot's feet to their desired contact locations* -- and a privileged actual
contact indicator for the (asymmetric) critic.

Per the paper we deliberately do **not** expose actual foot-contact sensing to
the actor ("we do not use any contact sensing on the robot's feet since it did
not make any difference in simulation performance"); the contact-state term
below is intended for the critic only.
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


def feet_to_goal_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Relative vector from each foot to its current desired contact location.

  Expressed in the base frame (yaw-only), flattened to ``[N, 4*3 = 12]``.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, ContactGoalCommand)
  asset: Entity = env.scene[asset_cfg.name]

  goal_w = term.goal_pos_w[:, :, 0, :]  # [N, 4, 3]
  foot_w = asset.data.site_pos_w[:, term.foot_site_ids, :]  # [N, 4, 3]
  rel_w = goal_w - foot_w  # [N, 4, 3]

  yaw = asset.data.heading_w  # [N]
  c, s = torch.cos(yaw), torch.sin(yaw)
  rx, ry, rz = rel_w[..., 0], rel_w[..., 1], rel_w[..., 2]
  xb = c.unsqueeze(-1) * rx + s.unsqueeze(-1) * ry
  yb = -s.unsqueeze(-1) * rx + c.unsqueeze(-1) * ry
  rel_b = torch.stack([xb, yb, rz], dim=-1)  # [N, 4, 3]
  return rel_b.reshape(env.num_envs, -1)


def foot_contact_state(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Actual per-foot contact indicator ``I^act`` (privileged; critic only)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()
