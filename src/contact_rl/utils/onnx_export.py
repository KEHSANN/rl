"""Torch-free bookkeeping around ONNX policy export.

The runner's export has two independent steps: writing ``policy.onnx``
(rsl-rl's exporter, wrapped by :class:`contact_rl.utils.onnx_compat.DropNoneOutputs`
for GRU policies) and attaching metadata to the written file (which may need
the W&B run name). Before this module both steps shared one ``try`` block
that also did ``import wandb``, so a machine without wandb -- the default
TensorBoard setup -- reported a successfully written ONNX file as an
*export failure*. :func:`export_onnx_with_metadata` keeps the outcomes apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def wandb_run_name(logger_type: str | None, default: str = "local") -> str:
  """Current W&B run name when logging to W&B, else ``default``. Never raises
  (wandb is optional)."""
  if logger_type != "wandb":
    return default
  try:
    import wandb  # optional dependency
  except Exception:
    return default
  run = getattr(wandb, "run", None)
  name = getattr(run, "name", None)
  return str(name) if name else default


def export_onnx_with_metadata(
  export_fn: Callable[[], Path | str],
  attach_fn: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
  """Run ``export_fn`` (returns the written path), then ``attach_fn(path)``.

  Returns ``{"status", "path", "error", "metadata"}`` where ``status`` is
  ``"ok"`` (file written, metadata attached or not requested),
  ``"ok_without_metadata"`` (file written, metadata step failed -- the file is
  still usable) or ``"failed"`` (no usable file). ``error`` is ``None`` unless
  the export itself failed; a metadata failure is reported in ``metadata``.
  """
  res: dict[str, Any] = {"status": "failed", "path": None, "error": None, "metadata": None}
  try:
    path = Path(export_fn())
  except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    return res
  if not path.is_file() or path.stat().st_size == 0:
    res["error"] = f"exporter returned but {path} was not written"
    return res
  res["path"] = str(path)
  if attach_fn is None:
    res["status"] = "ok"
    res["metadata"] = "skipped"
    return res
  try:
    attach_fn(path)
  except Exception as e:  # noqa: BLE001
    res["status"] = "ok_without_metadata"
    res["metadata"] = f"failed: {type(e).__name__}: {e}"
    return res
  res["status"] = "ok"
  res["metadata"] = "attached"
  return res
