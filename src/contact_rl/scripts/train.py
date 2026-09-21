"""``contact-train`` entry point.

Registers the contact-explicit tasks, then hands control to mjlab's training
CLI. All of mjlab's ``train`` arguments are supported unchanged, e.g.::

    uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096
"""

from __future__ import annotations


def main() -> None:
  # Import for the registration side effect: adds the contact tasks to mjlab's
  # shared task registry before mjlab's CLI reads it via list_tasks().
  import contact_rl  # noqa: F401
  from mjlab.scripts.train import main as _train_main

  _train_main()


if __name__ == "__main__":
  main()
