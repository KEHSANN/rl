"""Checkpoint layer: layout, selection, validation, retention, atomicity."""

from __future__ import annotations

import datetime as dt
import os
import zipfile
from pathlib import Path

import pytest

from contact_rl.utils import checkpoints as ck


def fake_ckpt(path: Path, payload: bytes = b"x") -> Path:
  """A minimal torch-zip-shaped file (torch.save writes <name>/data.pkl)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with zipfile.ZipFile(path, "w") as z:
    z.writestr("archive/data.pkl", payload)
  return path


def make_run(root: Path, name: str, its: list[int]) -> Path:
  run = ck.create_run_dir(root, "exp", name)
  for it in its:
    fake_ckpt(ck.checkpoint_dir(run) / ck.checkpoint_name(it))
  return run


def test_create_run_dir_layout_and_no_reuse(tmp_path):
  now = dt.datetime(2026, 1, 2, 3, 4, 5)
  a = ck.create_run_dir(tmp_path, "exp", "x", now=now)
  b = ck.create_run_dir(tmp_path, "exp", "x", now=now)
  assert a.name == "2026-01-02_03-04-05_x" and b.name == "2026-01-02_03-04-05_x-1"
  for sub in ("config", "checkpoints", "videos", "logs", "metrics", "exported"):
    assert (a / sub).is_dir()


def test_resolve_latest_best_and_specific(tmp_path):
  run = make_run(tmp_path, "r", [0, 50, 100])
  assert ck.resolve_checkpoint(str(run)).name == "model_100.pt"
  assert ck.resolve_checkpoint(f"{run}:latest").name == "model_100.pt"
  assert ck.resolve_checkpoint(f"{run}:50").name == "model_50.pt"
  with pytest.raises(FileNotFoundError):
    ck.resolve_checkpoint(f"{run}:75")
  # best falls back to latest when no best exists yet
  assert ck.resolve_checkpoint(f"{run}:best").name == "model_100.pt"
  idx = ck.CheckpointIndex(run)
  idx.record(ck.checkpoint_dir(run) / "model_50.pt", 50, "r", 9.0, ck.RetentionPolicy(keep_last=0))
  assert ck.resolve_checkpoint(f"{run}:best").name == "best.pt"
  # bare 'latest' / 'best' pick the newest run under the root
  assert ck.resolve_checkpoint("latest", tmp_path).name == "model_100.pt"
  with pytest.raises(ValueError):
    ck.resolve_checkpoint(f"{run}:bogus")


def test_latest_skips_corrupt_and_incomplete(tmp_path):
  run = make_run(tmp_path, "r", [10, 20])
  d = ck.checkpoint_dir(run)
  (d / "model_30.pt").write_bytes(b"")  # empty
  (d / "model_40.pt").write_bytes(b"PK\x03\x04garbage")  # truncated zip
  (d / "model_50.pt.tmp").write_bytes(b"PK")  # in-flight write is not a checkpoint
  assert ck.resolve_checkpoint(str(run)).name == "model_20.pt"
  with pytest.raises(ck.CorruptCheckpointError):
    ck.resolve_checkpoint(f"{run}:40")
  with pytest.raises(ck.CorruptCheckpointError):
    ck.resolve_checkpoint(str(d / "model_30.pt"))
  assert ck.is_valid_checkpoint(d / "model_20.pt")
  assert not ck.is_valid_checkpoint(d / "model_50.pt.tmp")


def test_retention_selection():
  p = ck.RetentionPolicy(keep_last=2, keep_every=100)
  its = [0, 50, 100, 150, 200, 250]
  assert ck.select_for_deletion(its, p) == [50, 150]
  assert ck.select_for_deletion(its, p, protect={50}) == [150]
  assert ck.select_for_deletion(its, ck.RetentionPolicy(keep_last=0)) == []


def test_index_records_best_and_prunes_but_keeps_best(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  pol = ck.RetentionPolicy(keep_last=2, keep_every=0, keep_best=True)
  idx = ck.CheckpointIndex(run)
  rewards = {0: 1.0, 50: 5.0, 100: 2.0, 150: 3.0}
  for it, r in rewards.items():
    p = fake_ckpt(ck.checkpoint_dir(run) / ck.checkpoint_name(it), payload=str(it).encode())
    res = idx.record(p, it, "reward", r, pol)
  assert idx.best == {"iteration": 50, "metric": "reward", "value": 5.0}
  left = [i for i, _ in ck.list_checkpoints(run)]
  assert left == [50, 100, 150]  # 0 pruned, best (50) protected, last 2 kept
  assert res["deleted"] == []
  best = ck.checkpoint_dir(run) / ck.BEST_NAME
  assert zipfile.ZipFile(best).read("archive/data.pkl") == b"50"
  latest = ck.checkpoint_dir(run) / ck.LATEST_NAME
  assert os.path.realpath(latest).endswith("model_150.pt")
  # index survives a reload
  assert ck.CheckpointIndex(run).best["iteration"] == 50
  # keep_best=False: no best.pt maintenance
  run2 = ck.create_run_dir(tmp_path, "exp2")
  idx2 = ck.CheckpointIndex(run2)
  idx2.record(fake_ckpt(ck.checkpoint_dir(run2) / "model_0.pt"), 0, "reward", 1.0, ck.RetentionPolicy(keep_best=False))
  assert idx2.best is None and not (ck.checkpoint_dir(run2) / ck.BEST_NAME).exists()


def test_atomic_write_never_leaves_partial_file(tmp_path):
  target = tmp_path / "checkpoints" / "model_5.pt"
  fake_ckpt(target, b"old")

  def boom(tmp: Path) -> None:
    tmp.write_bytes(b"PK half written")
    raise RuntimeError("disk full")

  with pytest.raises(RuntimeError):
    ck.atomic_write(target, boom)
  assert zipfile.ZipFile(target).read("archive/data.pkl") == b"old"
  assert not (target.parent / "model_5.pt.tmp").exists()
  ck.atomic_write(target, lambda tmp: fake_ckpt(tmp, b"new"))
  assert zipfile.ZipFile(target).read("archive/data.pkl") == b"new"


def test_run_dir_of_checkpoint(tmp_path):
  run = make_run(tmp_path, "r", [7])
  assert ck.run_dir_of_checkpoint(ck.checkpoint_dir(run) / "model_7.pt") == run.resolve()
  legacy = tmp_path / "legacy"
  fake_ckpt(legacy / "model_3.pt")
  assert ck.run_dir_of_checkpoint(legacy / "model_3.pt") == legacy.resolve()
  assert ck.resolve_checkpoint(str(legacy)).name == "model_3.pt"


def test_prune_dirs(tmp_path):
  for it in (100, 200, 300):
    (tmp_path / f"iteration_{it:06d}").mkdir()
  gone = ck.prune_dirs(tmp_path, keep=2)
  assert [g.name for g in gone] == ["iteration_000100"]


def test_checkpoint_disappears_during_validation(tmp_path, monkeypatch):
  import builtins
  import zipfile

  path = tmp_path / "model_1.pt"
  with zipfile.ZipFile(path, "w") as archive:
    archive.writestr("a/data.pkl", b"x")

  original_open = builtins.open

  def disappearing_open(file, *args, **kwargs):
    if file == path:
      path.unlink()
      raise FileNotFoundError(path)
    return original_open(file, *args, **kwargs)

  monkeypatch.setattr(builtins, "open", disappearing_open)
  assert ck.is_valid_checkpoint(path) is False
