"""Checkpoint layout, resolution, validation, retention and atomic saving.

This is the single checkpoint-management layer used by contact-train /
contact-watch / contact-eval / contact-play.

Run layout (created by :func:`create_run_dir`)::

  runs/<experiment>/<YYYY-mm-dd_HH-MM-SS>[_<name>]/
    run_info.json            git / versions / hardware / resume metadata
    events.out.tfevents.*    TensorBoard (training/*, system/*, rsl-rl scalars)
    config/                  env.yaml, agent.yaml, cli.json
    checkpoints/             model_<it>.pt, latest.pt, best.pt, index.json
    exported/                policy.onnx (GRU-compatible export)
    videos/                  iteration_<it:06d>/*.mp4 (watcher / eval),
                             train/iteration_<it:06d>/*.mp4 (--video)
    logs/                    train.log, watch.log
    metrics/                 iteration_<it:06d>/{metrics.csv,summary.json},
                             evaluations.csv, watch_state.json, tb/

Checkpoint files are written to ``<name>.tmp`` then ``os.replace``-d, so a
reader never observes a half-written ``model_<it>.pt``. ``latest``
resolution skips files that fail validation (empty / truncated / not a torch
archive) instead of crashing.

Pure python (torch imported lazily). Paths stored on disk are relative to the
run directory so runs can be moved between machines.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from contact_rl.utils.runtime import now_iso, read_json, write_json

CKPT_DIRNAME = "checkpoints"
LATEST_NAME = "latest.pt"
BEST_NAME = "best.pt"
INDEX_NAME = "index.json"
RUN_SUBDIRS = ("config", CKPT_DIRNAME, "videos", "logs", "metrics", "exported")
_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


class CorruptCheckpointError(RuntimeError):
  """A checkpoint file exists but is empty, truncated or unreadable."""


def checkpoint_dir(run_dir: Path) -> Path:
  return Path(run_dir) / CKPT_DIRNAME


def checkpoint_name(iteration: int) -> str:
  return f"model_{int(iteration)}.pt"


def iteration_of(path: Path | str) -> int | None:
  m = _MODEL_RE.match(Path(path).name)
  return int(m.group(1)) if m else None


def create_run_dir(
  root: Path | str, experiment: str, run_name: str | None = None, now: _dt.datetime | None = None
) -> Path:
  """Create ``root/experiment/<timestamp>[_name]`` with the standard subdirs.
  Never reuses an existing directory (appends ``-1``, ``-2`` ...)."""
  stamp = (now or _dt.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
  base = stamp + (f"_{run_name}" if run_name else "")
  parent = Path(root) / experiment
  parent.mkdir(parents=True, exist_ok=True)
  run = parent / base
  k = 0
  while True:
    try:
      run.mkdir(parents=False, exist_ok=False)
      break
    except FileExistsError:
      k += 1
      run = parent / f"{base}-{k}"
  for sub in RUN_SUBDIRS:
    (run / sub).mkdir(exist_ok=True)
  return run


def validate_checkpoint(path: Path | str, deep: bool = False) -> None:
  """Raise :class:`CorruptCheckpointError` if ``path`` is not a loadable
  checkpoint. The cheap check (default) catches empty files, leftovers of an
  interrupted non-atomic copy and truncated zip archives without loading
  tensors; ``deep=True`` additionally ``torch.load``-s on CPU and checks keys."""
  p = Path(path)
  if not p.is_file():
    raise CorruptCheckpointError(f"'{p}' does not exist or is not a file.")
  if p.name.endswith(".tmp"):
    raise CorruptCheckpointError(f"'{p}' is an in-progress temporary file.")
  size = p.stat().st_size
  if size == 0:
    raise CorruptCheckpointError(f"'{p}' is empty.")
  with open(p, "rb") as f:
    head = f.read(4)
  if head.startswith(b"PK"):
    try:
      with zipfile.ZipFile(p) as z:
        names = z.namelist()
    except (zipfile.BadZipFile, OSError) as e:
      raise CorruptCheckpointError(f"'{p}' is a truncated/corrupt torch archive: {e}") from e
    if not any(n.endswith("data.pkl") for n in names):
      raise CorruptCheckpointError(f"'{p}' is a zip file but not a torch checkpoint.")
  elif not head.startswith(b"\x80"):  # legacy (non-zip) pickle protocol header
    raise CorruptCheckpointError(f"'{p}' is not a torch checkpoint (bad header {head!r}).")
  if deep:
    import torch

    try:
      d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
      raise CorruptCheckpointError(f"torch.load('{p}') failed: {e}") from e
    if not isinstance(d, dict) or "actor_state_dict" not in d and "model_state_dict" not in d:
      raise CorruptCheckpointError(f"'{p}' has no actor_state_dict.")


def is_valid_checkpoint(path: Path | str) -> bool:
  try:
    validate_checkpoint(path)
    return True
  except (CorruptCheckpointError, OSError):
    # Retention may remove the file between is_file(), stat() and open().
    return False


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  """``(iteration, path)`` for every ``model_<it>.pt`` in the run directory
  or its ``checkpoints/`` subdir (legacy mjlab layout supported)."""
  found: dict[int, Path] = {}
  for d in (Path(run_dir), checkpoint_dir(run_dir)):
    if not d.is_dir():
      continue
    for p in d.iterdir():
      it = iteration_of(p)
      if it is not None and p.is_file():
        found[it] = p
  return sorted(found.items())


def is_run_dir(path: Path) -> bool:
  path = Path(path)
  return path.is_dir() and (
    (path / "run_info.json").exists() or checkpoint_dir(path).is_dir() or bool(list_checkpoints(path))
  )


def list_runs(root: Path) -> list[Path]:
  root = Path(root)
  if not root.is_dir():
    return []
  if is_run_dir(root):
    return [root]
  runs: list[Path] = []
  for p in sorted(root.iterdir()):
    if not p.is_dir():
      continue
    if is_run_dir(p):
      runs.append(p)
    else:
      runs.extend(q for q in sorted(p.iterdir()) if q.is_dir() and is_run_dir(q))
  return sorted(runs, key=lambda r: (r.stat().st_mtime, r.name))


def latest_run(root: Path) -> Path | None:
  runs = list_runs(root)
  return runs[-1] if runs else None


def _latest_valid(target: Path) -> Path:
  ckpts = list_checkpoints(target)
  if not ckpts:
    raise FileNotFoundError(f"Run '{target}' has no checkpoints yet.")
  for it, path in reversed(ckpts):
    try:
      validate_checkpoint(path)
      return path.resolve()
    except CorruptCheckpointError as e:
      print(f"[WARN] skipping corrupt checkpoint (iteration {it}): {e}")
  raise CorruptCheckpointError(f"Run '{target}': every checkpoint failed validation.")


def resolve_checkpoint(spec: str, search_root: Path | str = "runs") -> Path:
  """Resolve a checkpoint selector.

  ``file.pt`` | ``run_dir`` (=latest) | ``run_dir:best|latest|<it>`` |
  ``latest`` | ``best`` (newest run under ``search_root``).
  """
  spec = spec.strip()
  which = "latest"
  if spec in ("latest", "best"):
    which = spec
    target = latest_run(Path(search_root))
    if target is None:
      raise FileNotFoundError(f"No runs found under '{search_root}'.")
  else:
    p = Path(spec).expanduser()
    if p.is_file():
      validate_checkpoint(p)
      return p.resolve()
    if ":" in spec and not p.exists():
      base, _, which = spec.rpartition(":")
      p = Path(base).expanduser()
    if not p.exists():
      raise FileNotFoundError(f"Checkpoint or run directory not found: '{spec}'")
    target = p if is_run_dir(p) else latest_run(p)
    if target is None:
      raise FileNotFoundError(f"No checkpoints below '{p}'.")
  if which == "best":
    best = checkpoint_dir(target) / BEST_NAME
    if best.exists():
      validate_checkpoint(best)
      return best.resolve()
    idx = CheckpointIndex(target)
    if idx.best is not None:
      cand = checkpoint_dir(target) / checkpoint_name(int(idx.best["iteration"]))
      if cand.exists():
        validate_checkpoint(cand)
        return cand.resolve()
    print(f"[WARN] run '{target}' has no best checkpoint yet; using latest.")
    which = "latest"
  if which == "latest":
    return _latest_valid(target)
  if which.isdigit():
    for i, path in list_checkpoints(target):
      if i == int(which):
        validate_checkpoint(path)
        return path.resolve()
    raise FileNotFoundError(f"Run '{target}' has no checkpoint for iteration {which}.")
  raise ValueError(f"Unknown checkpoint selector ':{which}' (use best/latest/<it>).")


def run_dir_of_checkpoint(path: Path) -> Path:
  parent = Path(path).resolve().parent
  return parent.parent if parent.name == CKPT_DIRNAME else parent


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
  """``write_fn(tmp)`` then fsync + ``os.replace(tmp, path)``; the temp file
  is removed if ``write_fn`` raises, and ``path`` is left untouched."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  try:
    write_fn(tmp)
    with open(tmp, "rb") as f:
      os.fsync(f.fileno())
    os.replace(tmp, path)
  finally:
    if tmp.exists():
      tmp.unlink(missing_ok=True)


def atomic_torch_save(obj: Any, path: Path) -> None:
  import torch

  atomic_write(Path(path), lambda tmp: torch.save(obj, tmp))


def update_latest_link(ckpt: Path) -> None:
  ckpt = Path(ckpt)
  link = ckpt.parent / LATEST_NAME
  tmp = ckpt.parent / (LATEST_NAME + ".tmp")
  try:
    if tmp.is_symlink() or tmp.exists():
      tmp.unlink()
    os.symlink(ckpt.name, tmp)
    os.replace(tmp, link)
  except OSError:
    shutil.copy2(ckpt, tmp)
    os.replace(tmp, link)


def copy_atomic(src: Path, dst: Path) -> None:
  atomic_write(Path(dst), lambda tmp: shutil.copy2(src, tmp))


@dataclass
class RetentionPolicy:
  keep_last: int = 5
  """Keep the newest N ``model_<it>.pt`` (<= 0 keeps everything)."""
  keep_every: int = 1000
  """Additionally keep every checkpoint whose iteration is a multiple of this."""
  keep_best: bool = True
  """Maintain ``best.pt`` and never delete the best iteration's file."""


def select_for_deletion(iterations: list[int], policy: RetentionPolicy, protect: set[int] | None = None) -> list[int]:
  if policy.keep_last <= 0 or not iterations:
    return []
  its = sorted(set(iterations))
  keep = set(its[-policy.keep_last :])
  keep.add(its[-1])
  if policy.keep_every > 0:
    keep.update(i for i in its if i % policy.keep_every == 0)
  keep |= protect or set()
  return [i for i in its if i not in keep]


class CheckpointIndex:
  """``checkpoints/index.json``: one entry per saved iteration + best."""

  def __init__(self, run_dir: Path):
    self.dir = checkpoint_dir(run_dir)
    self.path = self.dir / INDEX_NAME
    data = read_json(self.path, default=None) or {}
    self.entries: list[dict] = list(data.get("checkpoints", []))
    self.best: dict | None = data.get("best")

  def _flush(self) -> None:
    write_json(self.path, {"checkpoints": self.entries, "best": self.best})

  def record(
    self,
    ckpt: Path,
    iteration: int,
    metric_name: str,
    metric: float | None,
    policy: RetentionPolicy,
    protect: set[int] | None = None,
  ) -> dict:
    ckpt = Path(ckpt)
    self.entries = [e for e in self.entries if e.get("iteration") != iteration]
    self.entries.append({"iteration": iteration, "file": ckpt.name, "time": now_iso(), metric_name: metric})
    self.entries.sort(key=lambda e: e["iteration"])
    update_latest_link(ckpt)
    is_best = False
    if policy.keep_best and metric is not None and metric == metric:
      if self.best is None or metric > float(self.best.get("value", -float("inf"))):
        copy_atomic(ckpt, self.dir / BEST_NAME)
        self.best = {"iteration": iteration, "metric": metric_name, "value": metric}
        is_best = True
    keep = set(protect or set())
    if self.best and policy.keep_best:
      keep.add(int(self.best["iteration"]))
    on_disk = dict(list_checkpoints(self.dir.parent))
    doomed = select_for_deletion(list(on_disk), policy, keep)
    for it in doomed:
      try:
        on_disk[it].unlink()
      except OSError:
        pass
    doomed_set = set(doomed)
    for e in self.entries:
      if e["iteration"] in doomed_set:
        e["deleted"] = True
    self._flush()
    return {"is_best": is_best, "deleted": doomed}


def prune_dirs(parent: Path, keep: int, prefix: str = "iteration_") -> list[Path]:
  parent = Path(parent)
  if keep <= 0 or not parent.is_dir():
    return []
  dirs = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))
  doomed = dirs[:-keep] if len(dirs) > keep else []
  for d in doomed:
    shutil.rmtree(d, ignore_errors=True)
  return doomed
