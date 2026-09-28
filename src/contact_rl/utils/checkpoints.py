"""Checkpoint layout, resolution, retention and atomic saving.

Run layout: runs/<experiment>/<timestamp>[_<name>]/{events, run_info.json,
params/, checkpoints/{model_<it>.pt, latest.pt, best.pt, index.json},
exported/, videos/iteration_<it:06d>/, eval/, train.log}.

Pure python (torch imported lazily). Paths stored on disk are relative to the
run directory so runs can be moved between machines.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contact_rl.utils.runtime import now_iso, read_json, write_json

CKPT_DIRNAME = "checkpoints"
LATEST_NAME = "latest.pt"
BEST_NAME = "best.pt"
INDEX_NAME = "index.json"
_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_dir(run_dir: Path) -> Path:
  return Path(run_dir) / CKPT_DIRNAME


def iteration_of(path: Path | str) -> int | None:
  m = _MODEL_RE.match(Path(path).name)
  return int(m.group(1)) if m else None


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
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


def resolve_checkpoint(spec: str, search_root: Path | str = "runs") -> Path:
  """file.pt | run_dir | run_dir:best|latest|<it> | latest | best."""
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
      return p.resolve()
    if ":" in spec and not p.exists():
      base, _, which = spec.rpartition(":")
      p = Path(base).expanduser()
    if not p.exists():
      raise FileNotFoundError(f"Checkpoint or run directory not found: '{spec}'")
    target = p if is_run_dir(p) else latest_run(p)
    if target is None:
      raise FileNotFoundError(f"No checkpoints below '{p}'.")
  ckpts = list_checkpoints(target)
  if which == "best":
    best = checkpoint_dir(target) / BEST_NAME
    if best.exists():
      return best.resolve()
    which = "latest"
  if which == "latest":
    if not ckpts:
      raise FileNotFoundError(f"Run '{target}' has no checkpoints yet.")
    return ckpts[-1][1].resolve()
  if which.isdigit():
    for i, path in ckpts:
      if i == int(which):
        return path.resolve()
    raise FileNotFoundError(f"Run '{target}' has no checkpoint for iteration {which}.")
  raise ValueError(f"Unknown checkpoint selector ':{which}' (use best/latest/<it>).")


def run_dir_of_checkpoint(path: Path) -> Path:
  parent = Path(path).resolve().parent
  return parent.parent if parent.name == CKPT_DIRNAME else parent


def atomic_torch_save(obj: Any, path: Path) -> None:
  import torch

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  try:
    torch.save(obj, tmp)
    os.replace(tmp, path)
  finally:
    if tmp.exists():
      tmp.unlink(missing_ok=True)


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
  tmp = Path(dst).with_name(Path(dst).name + ".tmp")
  shutil.copy2(src, tmp)
  os.replace(tmp, dst)


@dataclass
class RetentionPolicy:
  keep_last: int = 5
  keep_every: int = 1000


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
  def __init__(self, run_dir: Path):
    self.dir = checkpoint_dir(run_dir)
    self.path = self.dir / INDEX_NAME
    data = read_json(self.path, default=None) or {}
    self.entries: list[dict] = list(data.get("checkpoints", []))
    self.best: dict | None = data.get("best")

  def _flush(self) -> None:
    write_json(self.path, {"checkpoints": self.entries, "best": self.best})

  def record(self, ckpt: Path, iteration: int, metric_name: str, metric: float | None, policy: RetentionPolicy) -> dict:
    ckpt = Path(ckpt)
    self.entries = [e for e in self.entries if e.get("iteration") != iteration]
    self.entries.append({"iteration": iteration, "file": ckpt.name, "time": now_iso(), metric_name: metric})
    self.entries.sort(key=lambda e: e["iteration"])
    update_latest_link(ckpt)
    is_best = False
    if metric is not None and metric == metric:
      if self.best is None or metric > float(self.best.get("value", -float("inf"))):
        copy_atomic(ckpt, self.dir / BEST_NAME)
        self.best = {"iteration": iteration, "metric": metric_name, "value": metric}
        is_best = True
    protect = {int(self.best["iteration"])} if self.best else set()
    on_disk = dict(list_checkpoints(self.dir.parent))
    doomed = select_for_deletion(list(on_disk), policy, protect)
    for it in doomed:
      try:
        on_disk[it].unlink()
      except OSError:
        pass
    for e in self.entries:
      if e["iteration"] in set(doomed):
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
