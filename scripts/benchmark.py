#!/usr/bin/env python3
"""Launcher for the benchmark harness.

The implementation lives in :mod:`contact_rl.scripts.benchmark` so it ships
inside the installed package. This file exists only so the conventional path
works::

    uv run python scripts/benchmark.py --preset baseline --collect-stats

The installed console script is equivalent and preferred::

    uv run contact-benchmark --preset baseline --collect-stats

Run either with ``--help`` for the full option list, or ``--list-presets`` to
see the available simulation presets and how they differ from baseline.
"""

from __future__ import annotations

import sys


def _main() -> None:
  try:
    from contact_rl.scripts.benchmark import main
  except ModuleNotFoundError as exc:  # pragma: no cover
    print(
      f"Could not import contact_rl ({exc}).\n"
      f"Install the project first:\n"
      f"  uv sync --extra cu128 --python 3.12 --frozen\n"
      f"and run this script through uv:\n"
      f"  uv run python scripts/benchmark.py --help",
      file=sys.stderr,
    )
    raise SystemExit(1) from exc
  main()


if __name__ == "__main__":
  _main()
