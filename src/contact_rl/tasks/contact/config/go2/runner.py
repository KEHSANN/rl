"""On-policy runner for the contact-explicit task.

Extends mjlab's :class:`MjlabOnPolicyRunner` with

* the paper's **entropy decay** (Section 4, "Training"), applied by wrapping
  ``self.alg.update`` so the coefficient is set right before every PPO update
  inside a *single* ``learn()`` call (resume-aware);
* **observability**: explained variance, entropy + entropy coefficient,
  iteration / collection / learning time, FPS, GPU memory, written to
  TensorBoard as ``training/*`` and ``system/*`` next to rsl-rl's own scalars;
* the project **checkpoint layer** (:mod:`contact_rl.utils.checkpoints`):
  atomic ``checkpoints/model_<it>.pt``, ``latest.pt``, ``best.pt`` (by mean
  training episode reward), ``index.json`` and retention;
* **safe resume**: :meth:`load_for_resume` restores the full training state
  and continues at ``loaded_iter + 1`` (rsl-rl alone would re-run the loaded
  iteration and overwrite its checkpoint); the resume source can never be
  written to;
* **graceful stop**: :meth:`request_stop` (SIGTERM / SIGINT) finishes the
  current iteration, always writes a checkpoint, closes the logger (so a W&B
  run is finished) and raises :class:`TrainingStopped`;
* **GRU ONNX export** through :mod:`contact_rl.utils.onnx_compat` (rsl-rl
  5.0.1 declares 2 output names but returns 3 values for GRUs). The export
  status is reported by :mod:`contact_rl.utils.onnx_export`: a written
  ``policy.onnx`` is never reported as a failure because the optional wandb
  package (only used for the metadata run name) is missing.

Without :meth:`configure_run` (e.g. used through mjlab's own CLI) the runner
behaves like before, with checkpoints under ``<log_dir>/checkpoints``.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from contact_rl.utils import checkpoints as ckpt
from contact_rl.utils.onnx_compat import export_policy_onnx
from contact_rl.utils.onnx_export import export_onnx_with_metadata, wandb_run_name
from contact_rl.utils.runtime import gpu_memory_stats, now_iso, update_run_info
from contact_rl.utils.train_stats import explained_variance

from .rl_cfg import (
  ENTROPY_DECAY_ITERS,
  ENTROPY_END,
  ENTROPY_START,
)

BEST_METRIC = "mean_episode_reward"


class TrainingStopped(Exception):
  """Raised after a graceful stop request once the checkpoint is written."""


def entropy_at(iteration: int) -> float:
  """Linear entropy schedule, clamped to ``ENTROPY_END`` after decay."""
  frac = min(max(iteration / float(max(ENTROPY_DECAY_ITERS, 1)), 0.0), 1.0)
  return ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)


class ContactOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.run_dir: Path | None = Path(log_dir) if log_dir else None
    self.retention = ckpt.RetentionPolicy()
    self.export_onnx = True
    self._resume_source: Path | None = None
    self._stop_reason: str | None = None
    self._last_ev = float("nan")
    self._last_saved_it: int | None = None
    self._wrap_logger()

  # ----------------------------------------------------------- configuration

  def configure_run(
    self,
    run_dir: Path,
    retention: ckpt.RetentionPolicy | None = None,
    export_onnx: bool = True,
  ) -> None:
    self.run_dir = Path(run_dir)
    if retention is not None:
      self.retention = retention
    self.export_onnx = export_onnx

  def request_stop(self, reason: str) -> None:
    if self._stop_reason is None:
      print(f"\n[INFO] {reason} received: stopping after the current iteration (Ctrl-C again to abort).", flush=True)
    self._stop_reason = reason

  # ----------------------------------------------------------------- resume

  def load_for_resume(self, path: str | Path, map_location: str | None = None) -> int:
    """Load full training state; the next iteration is ``iter + 1``."""
    path = Path(path).resolve()
    ckpt.validate_checkpoint(path)
    self.load(str(path), load_cfg=None, strict=True, map_location=map_location or self.device)
    loaded_it = int(self.current_learning_iteration)
    self.current_learning_iteration = loaded_it + 1
    self._resume_source = path
    return loaded_it

  # --------------------------------------------------------------- logging

  def _writer(self):
    """rsl-rl creates ``logger.writer`` lazily in ``init_logging_writer``
    (inside ``learn``); None before that or when not logging."""
    return getattr(self.logger, "writer", None)

  def _wrap_logger(self) -> None:
    orig_log = self.logger.log

    def log(**kw):
      orig_log(**kw)
      self._log_extra(kw)

    self.logger.log = log  # type: ignore[method-assign]

  def _mean_reward(self) -> float | None:
    buf = getattr(self.logger, "rewbuffer", None)
    return float(statistics.mean(buf)) if buf else None

  def _log_extra(self, kw: dict) -> None:
    it = int(kw["it"])
    ct, lt = float(kw["collect_time"]), float(kw["learn_time"])
    loss = kw.get("loss_dict") or {}
    writer = self._writer()
    steps = self.cfg["num_steps_per_env"] * self.env.num_envs
    scalars: dict[str, float] = {
      "training/iteration_time": ct + lt,
      "training/collection_time": ct,
      "training/learning_time": lt,
      "training/fps": steps / max(ct + lt, 1e-9),
      "training/explained_variance": self._last_ev,
      "training/entropy": float(loss.get("entropy", float("nan"))),
      "training/entropy_coef": float(getattr(self.alg, "entropy_coef", float("nan"))),
      "training/learning_rate": float(kw.get("learning_rate", float("nan"))),
      "training/value_loss": float(loss.get("value", float("nan"))),
      "training/surrogate_loss": float(loss.get("surrogate", float("nan"))),
    }
    r = self._mean_reward()
    if r is not None:
      scalars["training/reward"] = r
      scalars["training/episode_length"] = float(statistics.mean(self.logger.lenbuffer))
    mem = gpu_memory_stats(str(self.device))
    for k, v in mem.items():
      scalars[f"system/gpu_memory_{k}"] = v
    if mem:
      scalars["system/gpu_memory"] = mem["allocated_gib"]
    if writer is not None:
      for k, v in scalars.items():
        if v == v:  # skip NaN
          writer.add_scalar(k, v, it)
      print(
        f"[iter {it}] reward={scalars.get('training/reward', float('nan')):.3f} "
        f"ev={self._last_ev:.3f} ent_coef={scalars['training/entropy_coef']:.5f} "
        f"t={ct + lt:.2f}s fps={scalars['training/fps']:.0f} "
        f"gpu={mem.get('allocated_gib', 0.0):.2f}/{mem.get('reserved_gib', 0.0):.2f}GiB",
        flush=True,
      )
    if self._stop_reason is not None:
      # Always write the stop checkpoint: it must not depend on the logger
      # having a writer (the TensorBoard/W&B extras above are optional).
      # ``save`` maps the name onto <run_dir>/checkpoints/model_<it>.pt.
      log_dir = getattr(self.logger, "log_dir", None) or (str(self.run_dir) if self.run_dir else ".")
      self.save(os.path.join(log_dir, f"model_{it}.pt"))
      if writer is not None:
        writer.flush()
        # rsl-rl's learn() closes the logger only after its loop; we leave the
        # loop by raising, so close it here (finishes a W&B run cleanly).
        try:
          self.logger.stop_logging_writer()
        except Exception as e:  # noqa: BLE001
          print(f"[WARN] closing the logger failed: {type(e).__name__}: {e}")
      raise TrainingStopped(self._stop_reason)

  # ------------------------------------------------------------------ save

  def _target_path(self, path: str) -> Path:
    it = ckpt.iteration_of(path)
    it = int(self.current_learning_iteration) if it is None else it
    base = self.run_dir if self.run_dir is not None else Path(path).parent
    return ckpt.checkpoint_dir(base) / ckpt.checkpoint_name(it)

  def save(self, path: str, infos=None) -> None:
    target = self._target_path(path)
    it = ckpt.iteration_of(target)
    assert it is not None
    if self._resume_source is not None and target.resolve() == self._resume_source:
      raise RuntimeError(f"Refusing to overwrite the resume source checkpoint {target}.")
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = infos
    saved["contact_rl"] = {
      "iteration": it,
      "time": now_iso(),
      "resumed_from": str(self._resume_source) if self._resume_source else None,
      "mean_episode_reward": self._mean_reward(),
    }
    ckpt.atomic_torch_save(saved, target)
    run_dir = target.parent.parent
    protect = {ckpt.iteration_of(self._resume_source)} if self._resume_source and \
      ckpt.run_dir_of_checkpoint(self._resume_source) == run_dir else set()
    res = ckpt.CheckpointIndex(run_dir).record(
      target, it, BEST_METRIC, self._mean_reward(), self.retention, protect={p for p in protect if p is not None}
    )
    self._last_saved_it = it
    writer = self._writer()
    if writer is not None:
      writer.add_scalar("checkpoint/iteration", it, it)
      if res["is_best"]:
        writer.add_scalar("checkpoint/best_reward", float(self._mean_reward() or 0.0), it)
    if (run_dir / "run_info.json").exists():
      update_run_info(run_dir, last_checkpoint=target.name, last_iteration=it, updated=now_iso())
    if self.cfg.get("upload_model"):
      self.logger.save_model(str(target), it)
    if self.export_onnx:
      self._export_onnx(run_dir)

  def _export_onnx(self, run_dir: Path) -> None:
    out_dir = run_dir / "exported"
    run_name = wandb_run_name(getattr(self.logger, "logger_type", None))
    res = export_onnx_with_metadata(
      lambda: export_policy_onnx(self.alg.get_policy(), str(out_dir), "policy.onnx"),
      lambda p: attach_metadata_to_onnx(str(p), get_base_metadata(self.env.unwrapped, run_name)),
    )
    if res["status"] == "failed":  # never kill training for an export; record it loudly
      print(f"[ERROR] ONNX export failed (training continues): {res['error']}")
    elif res["status"] == "ok_without_metadata":
      print(f"[WARN] wrote {res['path']} but could not attach metadata: {res['metadata']}")
    if (run_dir / "run_info.json").exists():
      update_run_info(run_dir, onnx_export=res, onnx_export_error=res["error"])

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
    export_policy_onnx(self.alg.get_policy(), path, filename, verbose)

  # --------------------------------------------------- PPO update wrapping

  def _install_update_hooks(self) -> None:
    if getattr(self, "_update_hooks_installed", False):
      return
    if not hasattr(self.alg, "entropy_coef"):
      raise AttributeError("PPO algorithm has no 'entropy_coef'; cannot decay it.")
    original_update = self.alg.update
    self._entropy_updates = 0

    def update_with_hooks(*args, **kwargs):
      it = self._entropy_it0 + self._entropy_updates
      self.alg.entropy_coef = entropy_at(it)
      self._entropy_updates += 1
      st = self.alg.storage
      try:  # returns / values are final here (compute_returns ran)
        self._last_ev = explained_variance(st.values, st.returns)
      except Exception:
        self._last_ev = float("nan")
      return original_update(*args, **kwargs)

    self.alg.update = update_with_hooks  # type: ignore[method-assign]
    self._update_hooks_installed = True

  # Backwards-compatible name.
  _install_entropy_schedule = _install_update_hooks

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    self._entropy_it0 = int(self.current_learning_iteration)
    self._install_update_hooks()
    self._entropy_updates = 0
    t0 = time.time()
    try:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      if self.run_dir is not None and (self.run_dir / "run_info.json").exists():
        update_run_info(
          self.run_dir, wall_time_s=round(time.time() - t0, 1), stopped=self._stop_reason, updated=now_iso()
        )
