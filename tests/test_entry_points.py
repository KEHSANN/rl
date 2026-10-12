"""Every [project.scripts] entry point resolves to a real module + function."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"contact-train", "contact-eval", "contact-watch", "contact-play", "contact-doctor", "contact-bench"}


def _scripts() -> dict[str, str]:
  try:
    import tomllib
  except ModuleNotFoundError:  # python 3.10
    tomllib = pytest.importorskip("tomli")
  return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]


def test_required_commands_are_registered():
  assert REQUIRED <= set(_scripts())


def test_entry_points_resolve():
  for name, target in _scripts().items():
    mod, _, fn = target.partition(":")
    path = ROOT / "src" / Path(*mod.split(".")).with_suffix(".py")
    assert path.is_file(), f"{name}: {path} missing"
    tree = ast.parse(path.read_text())
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert fn in funcs, f"{name}: {mod} has no function {fn}"
    src = path.read_text()
    assert '__name__ == "__main__"' in src or name != "contact-eval", "watch runs eval via python -m"


def test_headless_backend_selected_before_mujoco_in_every_entry_module():
  """Each contact-* module is imported as contact_rl.scripts.<x>, which runs
  contact_rl/__init__ (GL selection) before anything imports mujoco; the
  modules themselves must not import mujoco / mjlab / warp at module level."""
  heavy = {"mujoco", "mjlab", "warp"}  # torch alone does not touch the GL backend
  for target in _scripts().values():
    mod = target.partition(":")[0]
    path = ROOT / "src" / Path(*mod.split(".")).with_suffix(".py")
    for node in ast.parse(path.read_text()).body:
      names = []
      if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
      elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module]
      for n in names:
        assert n.split(".")[0] not in heavy, f"{mod} imports {n} at module level"
