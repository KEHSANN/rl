"""Test setup: import ``contact_rl`` from ``src`` without needing a GPU.

``contact_rl/__init__`` tolerates a missing simulator stack (it records the
error for ``require_tasks``), so the pure utilities import anywhere. Modules
under ``tasks/contact/mdp`` would pull in mjlab through the package
``__init__`` chain; :func:`load_mdp` loads them by file path instead.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))


def load_mdp(name: str):
  """Load ``contact_rl.tasks.contact.mdp.<name>`` without its package __init__."""
  import contact_rl  # noqa: F401

  pkgs = ["contact_rl.tasks", "contact_rl.tasks.contact", "contact_rl.tasks.contact.mdp"]
  base = SRC / "contact_rl"
  for i, pk in enumerate(pkgs):
    mod = sys.modules.get(pk)
    if mod is None or not hasattr(mod, "__path__"):
      m = types.ModuleType(pk)
      m.__path__ = [str(base.joinpath(*pk.split(".")[1:]))]  # type: ignore[attr-defined]
      sys.modules[pk] = m
  full = f"contact_rl.tasks.contact.mdp.{name}"
  if full in sys.modules:
    return sys.modules[full]
  spec = importlib.util.spec_from_file_location(full, base / "tasks" / "contact" / "mdp" / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[full] = mod
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  return mod
