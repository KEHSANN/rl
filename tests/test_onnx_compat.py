"""GRU ONNX export: rsl-rl 5.0.1 returns (actions, h, None) with 2 names."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from contact_rl.utils.onnx_compat import DropNoneOutputs, export_policy_onnx, make_exportable, verify_onnx  # noqa: E402


class FakeOnnxGRU(nn.Module):
  """Same interface / return contract as rsl-rl 5.0.1 ``_OnnxRNNModel`` (GRU)."""

  rnn_type = "gru"

  def __init__(self):
    super().__init__()
    self.rnn = nn.GRU(5, 4)
    self.head = nn.Linear(4, 3)

  def forward(self, obs, h_in, c_in=None):
    x, h = self.rnn(obs.unsqueeze(0), h_in)
    return self.head(x.squeeze(0)), h, None

  def get_dummy_inputs(self):
    return (torch.zeros(1, 5), torch.zeros(1, 1, 4))

  input_names = ["obs", "h_in"]
  output_names = ["actions", "h_out"]


class FakePolicy:
  def as_onnx(self, verbose=False):
    return FakeOnnxGRU()


def test_none_output_is_dropped():
  m = make_exportable(FakeOnnxGRU())
  assert isinstance(m, DropNoneOutputs)
  out = m(*m.get_dummy_inputs())
  assert len(out) == 2 and all(isinstance(o, torch.Tensor) for o in out)


def test_real_mismatch_still_raises():
  class Bad(FakeOnnxGRU):
    output_names = ["actions"]

  with pytest.raises(ValueError):
    make_exportable(Bad())


def test_export_and_verify(tmp_path):
  pytest.importorskip("onnx")
  path = export_policy_onnx(FakePolicy(), str(tmp_path), "policy.onnx")
  rep = verify_onnx(path, make_exportable(FakeOnnxGRU().eval()))
  assert rep["checker"] == "ok" and rep["outputs"] == ["actions", "h_out"]


def test_against_installed_rsl_rl_gru(tmp_path):
  """Regression against the real rsl-rl RNNModel if the package is installed."""
  rnn_model = pytest.importorskip("rsl_rl.models.rnn_model")
  td = pytest.importorskip("tensordict")
  obs = td.TensorDict({"policy": torch.zeros(2, 6)}, batch_size=[2])
  model = rnn_model.RNNModel(obs, {"actor": ["policy"]}, "actor", 3, hidden_dims=(8,), rnn_type="gru", rnn_hidden_dim=4)
  raw = model.as_onnx()
  out = raw(*raw.get_dummy_inputs())
  assert len(out) == 3 and out[2] is None and len(raw.output_names) == 2  # the upstream issue
  m = make_exportable(raw)
  assert len(m(*m.get_dummy_inputs())) == len(m.output_names)
  pytest.importorskip("onnx")
  path = export_policy_onnx(model, str(tmp_path), "gru.onnx")
  assert verify_onnx(path)["outputs"] == ["actions", "h_out"]
