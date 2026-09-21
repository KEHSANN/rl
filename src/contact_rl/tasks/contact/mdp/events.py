"""Startup consistency checks for the contact-explicit task.

The reach / hold / detach rewards (Eq. 1-3) index three independent things by
the *same* foot column: the command term's goal buffers, the robot's foot sites,
and the contact sensor's ``found`` field. Nothing in mjlab guarantees these
agree -- both name resolvers default to *model* order -- so a reordered MJCF, an
extra site, or a changed geom pattern would silently permute the feet and
produce a policy that trains against scrambled goals.

This module provides a ``startup`` event that fails loudly instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.sensor import ContactSensor

from contact_rl.tasks.contact.mdp.contact_command import ContactGoalCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def check_foot_ordering(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  sensor_name: str,
  foot_order: Sequence[str],
) -> None:
  """Verify the command term and the contact sensor use ``foot_order``.

  Args:
    env: The environment.
    env_ids: Unused (startup events receive ``None``).
    command_name: Key of the :class:`ContactGoalCommand` term.
    sensor_name: Key of the feet :class:`ContactSensor`.
    foot_order: The canonical foot order, e.g. ``("FL", "FR", "RL", "RR")``.

  Raises:
    ValueError: If either the command term's foot sites or the sensor's contact
      slots deviate from ``foot_order``.
  """
  del env_ids  # Startup event.
  expected = tuple(foot_order)

  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ContactGoalCommand):
    raise ValueError(
      f"Command term '{command_name}' is {type(term).__name__}, "
      f"expected ContactGoalCommand."
    )
  if tuple(term.cfg.foot_site_names) != expected:
    raise ValueError(
      f"Command term '{command_name}' foot sites {tuple(term.cfg.foot_site_names)} "
      f"!= expected foot order {expected}."
    )
  if len(term.foot_site_ids) != len(expected):
    raise ValueError(
      f"Command term '{command_name}' resolved {len(term.foot_site_ids)} foot "
      f"sites, expected {len(expected)}."
    )

  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor):
    raise ValueError(
      f"Sensor '{sensor_name}' is {type(sensor).__name__}, expected ContactSensor."
    )

  # ``ContactSensor`` does not publish its resolved primary names; ``_slots`` is
  # built in column order (primary-major, then field) in ``edit_spec``. Skip the
  # check rather than crash if that internal ever changes.
  slots = getattr(sensor, "_slots", None)
  if not slots:
    return
  primaries = [s.primary_name for s in slots if s.field_name == "found"]
  if len(primaries) != len(expected):
    raise ValueError(
      f"Sensor '{sensor_name}' has {len(primaries)} 'found' columns "
      f"({primaries}), expected {len(expected)} for feet {expected}."
    )
  # Foot geoms are named ``<FOOT>_foot_collision``; compare the leg prefix.
  resolved = tuple(p.rsplit("/", 1)[-1].split("_", 1)[0] for p in primaries)
  if resolved != expected:
    raise ValueError(
      f"Sensor '{sensor_name}' contact columns resolve to feet {resolved} "
      f"(from {primaries}), but the rewards and goals assume {expected}. "
      f"Fix the sensor's geom pattern order or the MJCF declaration order."
    )
