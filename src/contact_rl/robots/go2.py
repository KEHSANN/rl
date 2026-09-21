"""Unitree Go2 entity configuration for the contact-explicit task.

This mirrors mjlab's ``asset_zoo/robots/unitree_go1/go1_constants.py`` but points
at the Go2 MJCF shipped with this project (``src/assets/robots/unitree_go2``).

Notes
-----
* The Go2 and Go1 are close in size and use very similar direct-drive motors, so
  the actuator model (reflected inertia -> PD gains) is *adapted from* the mjlab
  Go1 config. Effort limits use published Go2 values (hip/thigh 23.7 Nm, calf
  45.43 Nm). Refine with a Go2 datasheet if exact numbers are required; the
  contact-explicit task itself does not depend on these.
* Joint / site / geom names are taken verbatim from ``go2.xml``:
    - joints:   ``{FL,FR,RL,RR}_{hip,thigh,calf}_joint``
    - feet sites (end-effectors ``e``): ``FL FR RL RR``
    - feet collision geoms:             ``{FL,FR,RL,RR}_foot_collision``
    - free joint on ``base_link``:      ``floating_base_joint``
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Assets live at ``paper-contact/src/assets`` (sibling of the ``contact_rl``
# package). Resolve relative to this file so a source / editable install works
# the same way mjlab resolves its own assets via ``MJLAB_SRC_PATH``.
_ASSETS_ROOT: Path = Path(__file__).resolve().parents[2] / "assets"
GO2_XML: Path = _ASSETS_ROOT / "robots" / "unitree_go2" / "xmls" / "go2.xml"
assert GO2_XML.exists(), f"Go2 MJCF not found at {GO2_XML}"

# End-effector (foot) definitions, ordered FL, FR, RL, RR.
FOOT_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
FOOT_SITE_NAMES: tuple[str, ...] = FOOT_NAMES
FOOT_GEOM_NAMES: tuple[str, ...] = tuple(f"{n}_foot_collision" for n in FOOT_NAMES)


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (adapted from mjlab Go1; see module docstring).
##

ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.1,
  effort_limit=23.7,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=15.70,
  effort_limit=45.43,
)

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz.
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ
STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Initial state (go2.xml has no <keyframe>; set a nominal standing pose).
#
# Standing base height from the go2.xml kinematics: thigh and calf are each
# 0.213 m and the thigh joint sits at the base-link z. With thigh=+0.9 and
# calf=-1.8 the foot site lands 0.213*cos(0.9) + 0.213*cos(-0.9) = 0.265 m below
# the base, plus the 0.022 m foot-collision sphere radius -> ~0.287 m. Spawning
# at 0.30 leaves ~13 mm of clearance, enough to absorb the +-0.1 rad joint
# randomisation without initial penetration and without a long free fall.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.30),
  joint_pos={
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
    ".*_hip_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# Enable all collisions except self-collisions; feet get their own
# condim / friction / solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_HIP_ACTUATOR_CFG, GO2_KNEE_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Return a fresh Go2 ``EntityCfg`` (new instance to avoid shared mutation)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


# Per-actuator action scale, following mjlab's convention: 0.25 * effort / stiffness.
GO2_ACTION_SCALE: dict[str, float] = {}
for _a in GO2_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    GO2_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())
  viewer.launch(robot.spec.compile())
