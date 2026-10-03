"""contact-play: GRU reset on episode end for the native viewer path, and
the localhost-only default (no torch / mjlab needed)."""

from __future__ import annotations

import numpy as np
import pytest

from contact_rl.scripts import play
from contact_rl.utils import runtime as rt


class Policy:
  def __init__(self):
    self.calls = []

  def reset(self, dones=None):
    self.calls.append(None if dones is None else np.asarray(dones).tolist())


class Env:
  unwrapped = "base"
  num_envs = 3

  def __init__(self):
    self.dones = [np.array([0, 0, 0]), np.array([0, 1, 0])]

  def step(self, a):
    return ("obs", 0.0, self.dones.pop(0), {})


def test_native_viewer_env_proxy_resets_gru_of_done_envs_only():
  pol = Policy()
  env = play._RecurrentResetEnv(Env(), lambda: pol)
  assert env.unwrapped == "base" and env.num_envs == 3
  env.step(None)
  assert pol.calls == []  # nothing ended
  env.step(None)
  assert pol.calls == [[0, 1, 0]]


def test_public_bind_requires_explicit_flag():
  for host in ("127.0.0.1", "localhost", "::1"):
    rt.assert_private_bind(host, False)
  with pytest.raises(SystemExit):
    rt.assert_private_bind("0.0.0.0", False)
  rt.assert_private_bind("0.0.0.0", True)
