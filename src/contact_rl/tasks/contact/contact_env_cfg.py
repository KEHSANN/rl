"""Contact-explicit locomotion task configuration.

Factory ``make_contact_env_cfg`` builds the base (robot-agnostic) config for the
contact-explicit multi-gait locomotion task of Omar & Khadiv
(arXiv:2510.03599v2). Robot-specific configs (see ``config/go2``) call the
factory and fill in the per-robot names / scales.

Faithfulness notes:
  * Actor observations = proprioception (joint pos, joint vel, last action) +
    task observations (contact goal = current+next contact sequence & locations
    in base frame + command duration, and relative feet->goal distance). No
    height scan and no foot-contact sensing in the actor, exactly as the paper
    states.
  * The critic is asymmetric (privileged): it additionally sees base linear /
    angular velocity, projected gravity, and actual foot contact. This does not
    change the deployed policy's input contract.
  * Rewards = reach + hold + detach (Eq. 1-3) + goal-discovery bonus, plus the
    auxiliary penalties the paper lists (base angular velocity, joint velocity,
    acceleration, torque, joint deviation, action rate). Reward weights are
    per-second rates (mjlab dt-scales them by ``step_dt``); tune as needed.

Simulation parameters:
  The MuJoCo / MJWarp settings are **not** written inline here. They come from
  the named presets in :mod:`contact_rl.sim_presets`, which default to
  ``baseline`` -- the exact values this task has always used. This keeps
  performance experiments non-destructive: an optimisation is a preset you opt
  into, not an edit that silently changes the physics of every future run. See
  that module for what each parameter does and which ones must never change.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from contact_rl.sim_presets import make_sim_cfg
from contact_rl.tasks.contact import mdp

# Foot order shared by the command term, the contact sensor, and the feet
# rewards / observations. Keep these aligned.
FOOT_ORDER: tuple[str, ...] = ("FL", "FR", "RL", "RR")

COMMAND_NAME = "contact"
FEET_SENSOR_NAME = "feet_ground_contact"


def make_contact_env_cfg(sim_preset: str | None = None) -> ManagerBasedRlEnvCfg:
  """Create the base contact-explicit locomotion task configuration.

  Args:
    sim_preset: Name of a preset in :mod:`contact_rl.sim_presets`
      (``baseline`` / ``safe`` / ``optimized``). ``None`` resolves via the
      ``CONTACT_RL_SIM_PRESET`` environment variable and then falls back to
      ``baseline``, whose values are identical to the ones this task used
      before the preset system existed.
  """

  ##
  # Observations.
  ##

  # Actor: proprioception + task observations only (paper-faithful).
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "contact_goal": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": COMMAND_NAME},
    ),
    "feet_to_goal": ObservationTermCfg(
      func=mdp.feet_to_goal_distance,
      params={"command_name": COMMAND_NAME},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
  }

  # Critic: privileged (asymmetric) observations.
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact_state,
      params={"sensor_name": FEET_SENSOR_NAME},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands (the contact goal).
  ##

  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: mdp.ContactGoalCommandCfg(
      entity_name="robot",
      foot_site_names=FOOT_ORDER,
      resampling_time_range=(0.34, 0.36),
      debug_vis=True,
    ),
  }

  ##
  # Events (extensive domain randomisation, per the paper).
  ##

  events = {
    # NOTE: ``z`` is an offset on top of the robot's nominal standing height, so
    # keep the range small -- a large offset makes every episode start with a
    # free fall that eats a sizeable fraction of the first command window.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    # Fail fast if the goal buffers, the foot sites and the contact sensor ever
    # stop agreeing on the foot column order (both name resolvers default to
    # model order, so this is a real failure mode, not a theoretical one).
    "check_foot_ordering": EventTermCfg(
      mode="startup",
      func=mdp.check_foot_ordering,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR_NAME,
        "foot_order": FOOT_ORDER,
      },
    ),
  }

  ##
  # Rewards.
  ##

  rewards = {
    # Contact-explicit rewards (Section 3.2, Eq. 1-3).
    "reach": RewardTermCfg(
      func=mdp.contact_reach,
      weight=1.0,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR_NAME,
        "std": 0.1,  # std=1 reproduces the literal exp(-d^2); smaller sharpens.
      },
    ),
    "hold": RewardTermCfg(
      func=mdp.contact_hold,
      weight=1.0,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR_NAME,
        "hold_lambda": 1.0,
        "std": 0.1,
      },
    ),
    "detach": RewardTermCfg(
      func=mdp.contact_detach,
      weight=0.5,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": FEET_SENSOR_NAME,
      },
    ),
    "goal_discovery": RewardTermCfg(
      func=mdp.goal_discovery_bonus,
      # This is an *event* indicator (fires on the step a goal is discovered),
      # but mjlab dt-scales every reward term by step_dt=0.02, so the weight is
      # 50x smaller than it reads: 25.0 -> 0.5 per discovery. Together with
      # ``goal_dwell_frac`` the rate is capped at ~2x the nominal switch rate,
      # which keeps this well below the hold reward's ~8/s ceiling.
      weight=25.0,
      params={"command_name": COMMAND_NAME},
    ),
    # Auxiliary locomotion penalties named by the paper.
    "base_ang_vel": RewardTermCfg(
      func=mdp.base_angular_velocity_l2,
      weight=-0.05,
    ),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-1.0e-3,
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-2.0e-4,
    ),
    "joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.05,
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    # Standard joint-limit safety penalty (kept minimal).
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
    ),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(80.0)},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4096,  # Paper uses 8192; override via CLI as GPU memory allows.
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    # Every MuJoCo / MJWarp knob lives in contact_rl.sim_presets. The default
    # ``baseline`` preset is value-identical to the literal this replaced:
    #   nconmax=None, njmax=300, contact_sensor_maxmatch=64,
    #   timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50.
    sim=make_sim_cfg(sim_preset),
    decimation=4,  # 50 Hz control.
    episode_length_s=20.0,
  )
