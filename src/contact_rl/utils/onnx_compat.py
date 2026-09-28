"""ONNX export compatibility for rsl-rl 5.0.1 recurrent policies.

Issue (verified against rsl-rl-lib 5.0.1, ``rsl_rl/models/rnn_model.py``):
``_OnnxRNNModel.forward`` returns ``(actions, h, None)`` for a GRU while
``output_names`` is ``["actions", "h_out"]``. The trailing ``None`` is not a
tensor, so ``torch.onnx.export`` (legacy / TorchScript tracer) cannot emit it
and the GRU policy export fails -- rsl-rl's own LSTM path returns 3 tensors
with 3 names and is fine.

Fix: wrap the export module so ``None`` outputs are dropped before tracing.
Nothing in rsl-rl is modified; training is untouched (the wrapper only exists
on the CPU copy that ``as_onnx()`` creates for export).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


class DropNoneOutputs(nn.Module):
  """Forward the wrapped module and drop ``None`` entries of its output tuple."""

  def __init__(self, inner: nn.Module):
    super().__init__()
    self.inner = inner

  def forward(self, *args: torch.Tensor):  # type: ignore[override]
    out = self.inner(*args)
    if isinstance(out, (tuple, list)):
      kept = tuple(o for o in out if o is not None)
      return kept if len(kept) != 1 else kept[0]
    return out

  def get_dummy_inputs(self):
    return self.inner.get_dummy_inputs()  # type: ignore[operator]

  @property
  def input_names(self) -> list[str]:
    return list(self.inner.input_names)  # type: ignore[arg-type]

  @property
  def output_names(self) -> list[str]:
    return list(self.inner.output_names)  # type: ignore[arg-type]


def make_exportable(onnx_model: nn.Module) -> nn.Module:
  """Return a module whose number of tensor outputs equals ``output_names``.
  Raises ValueError if they still disagree (a real incompatibility)."""
  model: nn.Module = onnx_model
  with torch.no_grad():
    out = model(*model.get_dummy_inputs())  # type: ignore[operator]
  outs = out if isinstance(out, (tuple, list)) else (out,)
  if any(o is None for o in outs):
    model = DropNoneOutputs(model)
    outs = tuple(o for o in outs if o is not None)
  names = list(model.output_names)  # type: ignore[arg-type]
  if len(outs) != len(names):
    raise ValueError(f"ONNX export: model returns {len(outs)} tensors but declares {len(names)} output names {names}.")
  return model


def export_policy_onnx(policy: Any, path: str, filename: str = "policy.onnx", verbose: bool = False) -> str:
  """Export ``policy`` (an rsl-rl model with ``as_onnx``) to ``path/filename``
  with the legacy exporter (``dynamo=False``, as mjlab does)."""
  onnx_model = policy.as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()
  model = make_exportable(onnx_model)
  model.eval()
  os.makedirs(path, exist_ok=True)
  out = os.path.join(path, filename)
  tmp = out + ".tmp"
  kwargs: dict[str, Any] = dict(
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
  )
  try:
    torch.onnx.export(model, model.get_dummy_inputs(), tmp, dynamo=False, **kwargs)
  except TypeError:  # torch without the ``dynamo`` kwarg
    torch.onnx.export(model, model.get_dummy_inputs(), tmp, **kwargs)
  os.replace(tmp, out)
  return out


def verify_onnx(path: str, model: nn.Module | None = None, atol: float = 1e-4) -> dict:
  """Structural check with ``onnx.checker``; numeric comparison against
  ``model`` (the exportable module) when onnxruntime is installed."""
  report: dict[str, Any] = {"path": path}
  try:
    import onnx

    m = onnx.load(path)
    onnx.checker.check_model(m)
    report["outputs"] = [o.name for o in m.graph.output]
    report["inputs"] = [i.name for i in m.graph.input]
    report["checker"] = "ok"
  except ImportError:
    report["checker"] = "skipped (onnx not installed)"
  try:
    import numpy as np
    import onnxruntime as ort

    if model is not None:
      sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
      dummy = model.get_dummy_inputs()  # type: ignore[operator]
      feeds = {n: d.numpy() for n, d in zip(model.input_names, dummy)}  # type: ignore[arg-type]
      got = sess.run(None, feeds)
      with torch.no_grad():
        ref = model(*dummy)
      ref = ref if isinstance(ref, (tuple, list)) else (ref,)
      report["max_abs_diff"] = float(max(np.abs(g - r.numpy()).max() for g, r in zip(got, ref)))
      report["numeric"] = "ok" if report["max_abs_diff"] <= atol else "MISMATCH"
  except ImportError:
    report["numeric"] = "skipped (onnxruntime not installed)"
  return report
