"""MDP terms for the contact-explicit task.

Re-exports the generic mjlab MDP terms and adds the contact-explicit command,
planner, rewards, observations, and a startup foot-ordering check.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .contact_command import (  # noqa: F401
  GAITS,
  ContactGoalCommand,
  ContactGoalCommandCfg,
)
from .events import (  # noqa: F401
  check_foot_ordering,
)
from .observations import (  # noqa: F401
  feet_to_goal_distance,
  foot_contact_state,
)
from .planning import (  # noqa: F401
  GAIT_PATTERNS,
  GaitPlanner,
  phase_masks,
  proximity_kernel,
)
from .rewards import (  # noqa: F401
  base_angular_velocity_l2,
  base_height_below,
  base_tilt_l2,
  contact_detach,
  contact_hold,
  contact_reach,
  goal_discovery_bonus,
  joint_deviation_l2,
  knee_height_below,
  undesired_contact_count,
)
