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
  play: bool = False, sim_preset: str | None = None
) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 flat-terrain contact-explicit configuration.

  Args:
    play: Use the play/evaluation variant (no corruption, no pushes, endless
      episodes, 64 envs).
    sim_preset: Simulation preset name from :mod:`contact_rl.sim_presets`.
      ``None`` resolves via ``$CONTACT_RL_SIM_PRESET`` and then ``baseline``.
  """
  cfg = make_contact_env_cfg(sim_preset=sim_preset)

  # Robot entity.
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Foot / ground contact sensor. NOTE: ``ContactSensor`` resolves ``pattern``
  # with ``find_geoms(..., preserve_order=False)``, so the ``found`` columns
  # follow the MJCF *declaration* order, not the order of this tuple. go2.xml
  # declares the feet FL, FR, RL, RR, so the two coincide; the
  # ``check_foot_ordering`` startup event asserts it rather than assuming it.
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

  # Action scale (per-actuator, from the Go2 constants).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  # Per-robot event targets.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Viewer follows the base.
  cfg.viewer.body_name = "base_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 64

  return cfg
