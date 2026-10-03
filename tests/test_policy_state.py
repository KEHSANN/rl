"""GRU hidden-state reset wiring (fake policy + real rsl-rl RNN when available)."""

from __future__ import annotations

import numpy as np
import pytest

from contact_rl.utils.policy_state import reset_recurrent_state


class FakeRecurrent:
  is_recurrent = True

  def __init__(self):
    self.calls = []

  def reset(self, dones=None):
    self.calls.append(None if dones is None else np.asarray(dones).tolist())


def test_reset_only_when_an_episode_ended():
  p = FakeRecurrent()
  assert reset_recurrent_state(p, np.array([0, 0, 0])) is False
  assert p.calls == []
  assert reset_recurrent_state(p, np.array([0, 1, 0])) is True
  assert p.calls == [[0, 1, 0]]
  assert reset_recurrent_state(p) is True  # manual reset: whole state
  assert p.calls[-1] is None


def test_non_recurrent_policy_is_noop():
  assert reset_recurrent_state(lambda obs: obs, np.array([1])) is False


def test_rsl_rl_gru_hidden_state_zeroed_only_for_done_envs():
  torch = pytest.importorskip("torch")
  rnn_mod = pytest.importorskip("rsl_rl.modules")

  class Holder:
    def __init__(self):
      self.rnn = rnn_mod.RNN(input_size=3, hidden_dim=4, num_layers=1, type="gru")

    def reset(self, dones=None):
      self.rnn.reset(dones)

  h = Holder()
  with torch.inference_mode():
    h.rnn(torch.randn(3, 3))
    h.rnn(torch.randn(3, 3))
    before = h.rnn.hidden_state.clone()
    assert float(before.abs().sum()) > 0
    reset_recurrent_state(h, torch.tensor([0, 1, 0]))
    after = h.rnn.hidden_state
    assert torch.all(after[:, 1] == 0)
    assert torch.equal(after[:, 0], before[:, 0]) and torch.equal(after[:, 2], before[:, 2])
    reset_recurrent_state(h)
    assert h.rnn.hidden_state is None
