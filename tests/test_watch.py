"""contact-watch scheduling: once per checkpoint, settle, retries, failures."""

from __future__ import annotations

import json
import zipfile

from contact_rl.scripts.watch import WatchConfig, Watcher
from contact_rl.utils import checkpoints as ck


def ckpt(run, it):
  p = ck.checkpoint_dir(run) / ck.checkpoint_name(it)
  with zipfile.ZipFile(p, "w") as z:
    z.writestr("a/data.pkl", b"x")
  return p


class Launcher:
  def __init__(self, fail=()):
    self.calls, self.fail = [], set(fail)

  def __call__(self, cmd, log_path):
    it = int(cmd[cmd.index("--out-dir") + 1].rsplit("_", 1)[1])
    self.calls.append(it)
    assert cmd[cmd.index("--checkpoint") + 1].endswith(f"model_{it}.pt")
    assert cmd[cmd.index("--video-dir") + 1].endswith(f"videos/iteration_{it:06d}")
    if it in self.fail:
      return 1
    out = log_path.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"overall": {"fall_rate": 0.1, "reward_mean": 1.0}}))
    return 0


def test_evaluates_each_checkpoint_once_after_settling(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 0)
  ckpt(run, 50)
  la = Launcher()
  w = Watcher(WatchConfig(settle_s=5.0), run, launcher=la)
  assert w.poll_once(now=100.0) == 0  # first sighting: wait to settle
  assert w.poll_once(now=103.0) == 0
  assert w.poll_once(now=106.0) == 2
  assert la.calls == [0, 50]
  assert w.poll_once(now=200.0) == 0  # never re-evaluated
  ckpt(run, 100)
  w.poll_once(now=300.0)
  w.poll_once(now=310.0)
  assert la.calls == [0, 50, 100]
  # state persists: a restarted watcher does not redo work
  w2 = Watcher(WatchConfig(settle_s=0.0), run, launcher=la)
  assert w2.poll_once() == 0


def test_failures_are_retried_then_skipped_and_watching_continues(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 10)
  ckpt(run, 20)
  la = Launcher(fail={10})
  w = Watcher(WatchConfig(settle_s=0.0, max_retries=1), run, launcher=la)
  w.poll_once()
  assert la.calls == [10, 20] and "20" in w.state["evaluated"] and "10" not in w.state["failed"]
  w.poll_once()
  assert "10" in w.state["failed"]
  w.poll_once()
  assert la.calls == [10, 20, 10]  # gave up after max_retries + 1 attempts


def test_launcher_exception_does_not_kill_watcher(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 1)

  def boom(cmd, log):
    raise OSError("fork failed")

  w = Watcher(WatchConfig(settle_s=0.0, max_retries=0), run, launcher=boom)
  assert w.poll_once() == 1
  assert "1" in w.state["failed"]


def test_every_only_latest_and_corrupt(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  for it in (0, 50, 100, 150):
    ckpt(run, it)
  (ck.checkpoint_dir(run) / "model_200.pt").write_bytes(b"")  # incomplete -> skipped
  la = Launcher()
  Watcher(WatchConfig(settle_s=0.0, every=100), run, launcher=la).poll_once()
  assert la.calls == [0, 100]
  la2 = Launcher()
  run2 = ck.create_run_dir(tmp_path, "exp2")
  for it in (0, 50):
    ckpt(run2, it)
  Watcher(WatchConfig(settle_s=0.0, only_latest=True), run2, launcher=la2).poll_once()
  assert la2.calls == [50]


def test_snapshot_protects_against_retention_and_is_removed(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  src = ckpt(run, 5)
  seen = {}

  def la(cmd, log_path):
    snap = cmd[cmd.index("--checkpoint") + 1]
    assert cmd[cmd.index("--run-dir") + 1] == str(run)
    assert ".snapshot" in snap and snap != str(src)
    src.unlink()  # retention deletes the original mid-evaluation ...
    seen["ok"] = zipfile.ZipFile(snap).read("a/data.pkl") == b"x"  # ... snapshot still readable
    (log_path.parent / "summary.json").write_text(json.dumps({"overall": {}}))
    return 0

  w = Watcher(WatchConfig(settle_s=0.0), run, launcher=la)
  assert w.poll_once() == 1 and seen["ok"] and "5" in w.state["evaluated"]
  assert not (run / "metrics" / "iteration_000005" / ".snapshot").exists()


def test_checkpoint_deleted_before_its_turn_is_skipped(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  p = ckpt(run, 7)
  w = Watcher(WatchConfig(settle_s=0.0), run, launcher=Launcher())
  (it, path), = w.pending()
  p.unlink()
  assert w.evaluate(it, path) is False and "7" in w.state["skipped"]
  assert w.pending() == []


def test_real_launcher_runs_child_in_own_session(tmp_path):
  import sys

  run = ck.create_run_dir(tmp_path, "exp")
  w = Watcher(WatchConfig(timeout_s=30), run)
  log = tmp_path / "eval.log"
  code = "import os,sys; sys.exit(0 if os.getsid(0) == os.getpid() else 3)"
  assert w._launch([sys.executable, "-c", code], log) == 0
  assert w._launch([sys.executable, "-c", "import sys; sys.exit(4)"], log) == 4
  w.cfg.timeout_s = 0.5
  assert w._launch([sys.executable, "-c", "import time; time.sleep(30)"], log) == -9
  assert "exceeded --timeout-s" in log.read_text()


def test_snapshot_failure_never_evaluates_source_or_consumes_retry(tmp_path, monkeypatch):
  run = ck.create_run_dir(tmp_path, "exp")
  src = ckpt(run, 8)
  la = Launcher()
  w = Watcher(WatchConfig(settle_s=0.0, max_retries=0), run, launcher=la)

  def fail_copy(source, dest):
    raise PermissionError("snapshot unavailable")

  monkeypatch.setattr("contact_rl.scripts.watch.os.link",
                      lambda source, dest: (_ for _ in ()).throw(OSError("no hardlinks")))
  monkeypatch.setattr(ck, "copy_atomic", fail_copy)
  assert w.poll_once() == 1
  assert la.calls == []
  assert src.exists()
  assert "8" not in w.state["attempts"]
  assert "8" not in w.state["failed"]
  assert not (run / "metrics" / "iteration_000008" / ".snapshot").exists()


def test_invalid_snapshot_never_launches_or_consumes_retry(tmp_path, monkeypatch):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 9)
  la = Launcher()
  w = Watcher(WatchConfig(settle_s=0.0), run, launcher=la)
  original = w._snapshot

  def corrupt_snapshot(it, path):
    snap = original(it, path)
    snap.unlink()  # break the hard link; do not corrupt the source checkpoint
    snap.write_bytes(b"invalid")
    return snap

  monkeypatch.setattr(w, "_snapshot", corrupt_snapshot)
  assert w.poll_once() == 1
  assert la.calls == []
  assert "9" not in w.state["attempts"]
  assert not (run / "metrics" / "iteration_000009" / ".snapshot").exists()
