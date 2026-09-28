"""Headless GL selection, bind guard, run info."""

from __future__ import annotations

import pytest

from contact_rl.utils import runtime as rt


def test_headless_selects_egl_without_display(monkeypatch):
  for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
    monkeypatch.delenv(k, raising=False)
  assert rt.configure_headless_rendering() == "egl"
  import os

  assert os.environ["MUJOCO_GL"] == "egl" and os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_user_choice_is_respected(monkeypatch):
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
  assert rt.configure_headless_rendering() == "osmesa"


def test_display_keeps_default(monkeypatch):
  monkeypatch.delenv("MUJOCO_GL", raising=False)
  monkeypatch.setenv("DISPLAY", ":0")
  assert rt.configure_headless_rendering() == "default"


def test_contact_rl_import_sets_gl_before_mujoco():
  """contact_rl/__init__ must select the backend before the task import."""
  import pathlib

  src = (pathlib.Path(rt.__file__).parents[1] / "__init__.py").read_text()
  assert src.index("_configure_gl()") < src.index("from contact_rl import tasks")


def test_private_bind_guard():
  rt.assert_private_bind("127.0.0.1", False)
  rt.assert_private_bind("0.0.0.0", True)
  with pytest.raises(SystemExit):
    rt.assert_private_bind("0.0.0.0", False)


def test_run_info_roundtrip(tmp_path):
  rt.update_run_info(tmp_path, a=1)
  d = rt.update_run_info(tmp_path, b=2)
  assert d == {"a": 1, "b": 2} and not list(tmp_path.glob("*.tmp"))
  assert rt.distribution_versions(("definitely-not-a-package",)) == {"definitely-not-a-package": None}


def test_resolve_device_explicit_errors():
  torch = pytest.importorskip("torch")
  assert rt.resolve_device("cpu") == "cpu"
  if not torch.cuda.is_available():
    with pytest.raises(SystemExit):
      rt.resolve_device("cuda:0")
  with pytest.raises(SystemExit):
    rt.resolve_device("tpu")
