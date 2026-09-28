"""Unitree Go2 contact-explicit locomotion environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from contact_rl.robots.go2 import GO2_ACTION_SCALE, get_go2_robot_cfg
from contact_rl.tasks.contact.contact_env_cfg import (
  FEET_SENSOR_NAME,
  FOOT_ORDER,
  make_contact_env_cfg,
)

# Foot collision geoms, ordered to match ``FOOT_ORDER`` (FL, FR, RL, RR).
_FOOT_GEOMS = tuple(f"{name}_foot_collision" for name in FOOT_ORDER)


def unitree_go2_flat_env_cfg(
  play: bool = False, improved: bool = False
) -> ManagerBasedRlEnvCfg:
  """Go2 flat-terrain contact-explicit configuration.

  Args:
    play: Evaluation / visualisation variant (no pushes, no obs noise).
    improved: EXPERIMENTAL variant -- adds IMU signals to the actor (see
      ``make_contact_env_cfg``). ``False`` is the paper-faithful task.
  """
  cfg = make_contact_env_cfg(actor_imu=improved)

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Foot / ground contact sensor. ``ContactSensor`` resolves ``pattern`` in MJCF
  # declaration order; ``check_foot_ordering`` asserts it matches FOOT_ORDER.
  feet_ground_cfg = ContactSensorCfg(
    name=FEET_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.viewer.body_name = "base_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 64

  return cfg
