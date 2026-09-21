"""``contact-play`` entry point.

Registers the contact-explicit tasks, then hands control to mjlab's play CLI.
All of mjlab's ``play`` arguments are supported unchanged, e.g.::

    uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 \
        --wandb-run-path <entity/project/run>
"""

from __future__ import annotations


def main() -> None:
  # Import for the registration side effect (see contact_rl.scripts.train).
  import contact_rl  # noqa: F401
  from mjlab.scripts.play import main as _play_main

  _play_main()


if __name__ == "__main__":
  main()
