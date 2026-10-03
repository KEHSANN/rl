"""contact-watch edge cases: multiple watchers, stale summaries, aborted evaluations."""

from __future__ import annotations

import json
import zipfile

from contact_rl.scripts.watch import LOCKED_EXIT_CODE, WatchConfig, Watcher, acquire_watch_lock
from contact_rl.utils import checkpoints as ck


def ckpt(run, it):
  p = ck.checkpoint_dir(run) / ck.checkpoint_name(it)
  with zipfile.ZipFile(p, "w") as z:
    z.writestr("a/data.pkl", b"x")
  return p


def test_second_watcher_on_same_run_exits_with_lock_code(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  lock = acquire_watch_lock(run)
  try:
    try:
      acquire_watch_lock(run)
      raise AssertionError("second watcher acquired the lock")
    except SystemExit as e:
      assert e.code == LOCKED_EXIT_CODE
  finally:
    lock.close()
  acquire_watch_lock(run).close()  # released with its holder: no stale lock


def test_stale_summary_from_crashed_attempt_is_not_accepted(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 3)
  out = run / "metrics" / "iteration_000003"
  out.mkdir(parents=True)
  (out / "summary.json").write_text(json.dumps({"iteration": 3, "overall": {}}))
  w = Watcher(WatchConfig(settle_s=0.0, max_retries=5), run, launcher=lambda cmd, log: 0)
  w.poll_once()
  assert "3" not in w.state["evaluated"] and not (out / "summary.json").exists()


def test_aborted_evaluation_is_not_counted_as_a_retry(tmp_path):
  run = ck.create_run_dir(tmp_path, "exp")
  ckpt(run, 4)
  w = Watcher(WatchConfig(settle_s=0.0, max_retries=0), run, launcher=None)

  def interrupted(cmd, log):
    w.aborted = True
    return -15

  w.launcher = interrupted
  w.poll_once()
  assert "4" not in w.state["failed"] and "4" in w.state["interrupted"]
  assert w.state["attempts"]["4"] == 0
