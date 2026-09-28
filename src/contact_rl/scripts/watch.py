"""``contact-watch``: evaluate new checkpoints of a running training job.

Runs as a separate process (own tmux window / systemd unit). It never touches
the training process: it only *reads* ``checkpoints/`` and writes under
``metrics/`` and ``videos/``.

Loop::

  poll run dir -> new, stable, valid model_<it>.pt? -> contact-eval subprocess
  (--mode episodes --video) -> metrics/iteration_<it>/{metrics.csv,summary.json},
  videos/iteration_<it>/*.mp4, metrics/evaluations.csv, TensorBoard evaluation/*
  -> record in metrics/watch_state.json

* **One watcher per run.** ``metrics/watch.lock`` is held (``flock``) for the
  watcher's lifetime; a second ``contact-watch`` on the same run exits with
  code 3. The kernel releases the lock when the holder dies (no stale locks).
* Each evaluation is a **subprocess**: only one policy/env is alive at a time,
  GPU memory is returned to the driver after every evaluation, and a crash
  (CUDA OOM, EGL, ffmpeg...) cannot kill the watcher.
* A checkpoint is evaluated **once**; failures are retried up to
  ``--max-retries`` times, then marked failed and skipped. An attempt that
  was aborted by the user (second Ctrl-C) is not counted.
* A checkpoint is considered complete when it passes validation and its size
  and mtime have not changed for ``--settle-s`` seconds. A checkpoint that
  disappears or becomes unreadable while being polled is simply not pending.
* The evaluated file is a **hard-linked snapshot** of the checkpoint
  (``metrics/iteration_<it>/.snapshot/``; copy if hard links are not
  supported), so training's retention (``--keep-last``) cannot delete it
  while the evaluation is still loading it. The snapshot is removed after the
  evaluation; ``--source-checkpoint`` makes ``summary.json`` /
  ``evaluations.csv`` record the original ``checkpoints/model_<it>.pt``, not
  the deleted snapshot. A checkpoint deleted before its turn is recorded as
  skipped.
* A stale ``summary.json`` from an earlier (crashed) attempt is removed
  before each attempt, and a summary is only accepted if the evaluation
  exited 0 and reports the expected iteration.
* The evaluation child runs in its **own session / process group**, so a
  Ctrl-C in the watcher's terminal does not kill the running evaluation; the
  watcher then exits after it finishes. A second Ctrl-C (more than 1 s later,
  see :class:`contact_rl.utils.runtime.StopSignalHandler`) terminates the
  child's whole process group. ``--timeout-s`` kills a hung evaluation.
* ``--every N`` evaluates only iterations that are multiples of N;
  ``--only-latest`` skips the backlog and evaluates the newest checkpoint.

Example::

  uv run contact-watch --run runs/go2_contact/<timestamp> --interval-s 60 --num-envs 128
  uv run contact-watch --run latest              # newest run under runs/
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from contact_rl.utils import checkpoints as ck
from contact_rl.utils import runtime as rt

LOCKED_EXIT_CODE = 3
"""Exit code when another watcher already holds the run's lock."""


@dataclass
class WatchConfig:
  run: str = "latest"
  """Run directory, or 'latest' (newest run under --run-root)."""
  run_root: str = "runs"
  task: str | None = None
  """Default: the task recorded in run_info.json."""
  interval_s: float = 60.0
  settle_s: float = 10.0
  every: int = 0
  """Only evaluate iterations divisible by this (0 = every checkpoint)."""
  only_latest: bool = False
  num_envs: int = 128
  episode_s: float = 15.0
  gaits: tuple[str, ...] = ("trot", "pace", "bound", "jump", "crawl")
  device: str | None = None
  """Default: the training device from run_info.json (evaluation is small)."""
  video: bool = True
  video_steps: int = 500
  keep_videos: int = 0
  """Keep only the newest N iteration video dirs (0 = keep all)."""
  max_retries: int = 2
  timeout_s: float = 3600.0
  once: bool = False
  """Evaluate what is pending, then exit (for cron / tests)."""
  extra_args: list[str] = field(default_factory=list)
  """Extra arguments forwarded verbatim to contact-eval."""


def iteration_tag(it: int) -> str:
  return f"iteration_{it:06d}"


class Watcher:
  def __init__(self, cfg: WatchConfig, run_dir: Path, launcher=None):
    self.cfg = cfg
    self.run_dir = run_dir
    self.state_path = run_dir / "metrics" / "watch_state.json"
    self.state = rt.read_json(self.state_path, default=None) or {}
    for key in ("evaluated", "failed", "attempts", "skipped", "interrupted"):
      self.state.setdefault(key, {})
    self._seen: dict[int, tuple[int, float, float]] = {}  # it -> (size, mtime, first_seen)
    self.stop = False
    self.aborted = False
    self.child: subprocess.Popen | None = None
    self.launcher = launcher or self._launch

  # ----------------------------------------------------------- selection

  def _done(self, it: int) -> bool:
    k = str(it)
    return k in self.state["evaluated"] or k in self.state["failed"] or k in self.state["skipped"]

  def pending(self, now: float | None = None) -> list[tuple[int, Path]]:
    now = time.time() if now is None else now
    out = []
    for it, path in ck.list_checkpoints(self.run_dir):
      if self._done(it) or (self.cfg.every > 0 and it % self.cfg.every != 0):
        continue
      try:
        st = path.stat()
      except OSError:  # deleted by retention meanwhile
        self._seen.pop(it, None)
        continue
      sig = (st.st_size, st.st_mtime)
      prev = self._seen.get(it)
      if prev is None or (prev[0], prev[1]) != sig:
        self._seen[it] = (sig[0], sig[1], now)
        if self.cfg.settle_s > 0:
          continue
      elif now - prev[2] < self.cfg.settle_s:
        continue
      if not ck.is_valid_checkpoint(path):  # also False if it vanished meanwhile
        continue
      out.append((it, path))
    if self.cfg.only_latest and out:
      out = out[-1:]
    return out

  # ----------------------------------------------------------- execution

  def _cmd(self, path: Path, it: int, source: Path | None = None) -> list[str]:
    info = rt.read_json(self.run_dir / "run_info.json", default={}) or {}
    task = self.cfg.task or info.get("task") or "Mjlab-Contact-Flat-Unitree-Go2"
    device = self.cfg.device or info.get("device") or "cuda:0"
    tag = iteration_tag(it)
    cmd = [
      sys.executable, "-m", "contact_rl.scripts.evaluate",
      "--task", task, "--checkpoint", str(path), "--run-dir", str(self.run_dir), "--mode", "episodes",
      "--num-envs", str(self.cfg.num_envs), "--episode-s", str(self.cfg.episode_s),
      "--gaits", *self.cfg.gaits, "--device", device, "--tensorboard", "True",
      "--out-dir", str(self.run_dir / "metrics" / tag),
      "--video", str(self.cfg.video), "--video-steps", str(self.cfg.video_steps),
      "--video-dir", str(self.run_dir / "videos" / tag),
    ]
    if source is not None and Path(source) != Path(path):
      cmd += ["--source-checkpoint", str(source)]
    return cmd + list(self.cfg.extra_args)

  def _launch(self, cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
      log.write(f"\n$ {' '.join(cmd)}\n")
      log.flush()
      # Own session: a terminal Ctrl-C reaches the watcher only (graceful
      # "stop after this evaluation"); the child is signalled explicitly.
      self.child = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True
      )
      try:
        return self.child.wait(timeout=self.cfg.timeout_s)
      except subprocess.TimeoutExpired:
        log.write(f"\n[watch] evaluation exceeded --timeout-s {self.cfg.timeout_s:.0f}s; killed\n")
        self._signal_child(signal.SIGKILL)
        self.child.wait()
        return -9
      finally:
        self.child = None

  def _signal_child(self, sig: int) -> None:
    child = self.child
    if child is None or child.poll() is not None:
      return
    try:
      os.killpg(child.pid, sig)  # the child's whole group (eval + its ffmpeg)
    except (ProcessLookupError, PermissionError, OSError):
      try:
        child.send_signal(sig)
      except OSError:
        pass

  def _snapshot(self, it: int, path: Path) -> Path:
    """Hard link (or copy) ``path`` to ``metrics/<tag>/.snapshot/model_<it>.pt``."""
    snap_dir = self.run_dir / "metrics" / iteration_tag(it) / ".snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / ck.checkpoint_name(it)
    snap.unlink(missing_ok=True)
    try:
      os.link(path, snap)
    except FileNotFoundError:
      raise
    except OSError:
      ck.copy_atomic(path, snap)
    return snap

  def evaluate(self, it: int, path: Path) -> bool:
    key = str(it)
    tag = iteration_tag(it)
    out_dir = self.run_dir / "metrics" / tag
    log_path = out_dir / "eval.log"
    summary_path = out_dir / "summary.json"
    self.state["attempts"][key] = int(self.state["attempts"].get(key, 0)) + 1
    t0 = time.time()
    try:
      snap = self._snapshot(it, path)
    except FileNotFoundError:  # retention deleted it before its turn
      shutil.rmtree(out_dir / ".snapshot", ignore_errors=True)
      self.state["skipped"][key] = {"time": rt.now_iso(), "reason": "checkpoint deleted before evaluation"}
      print(f"[watch] iteration {it}: checkpoint vanished (retention); skipped", flush=True)
      rt.write_json(self.state_path, self.state)
      return False
    except OSError as e:
      print(f"[watch] could not snapshot {path} ({e}); evaluating it in place", flush=True)
      snap = path
    # A summary.json left by an earlier crashed / killed attempt must not be
    # mistaken for the result of this attempt.
    try:
      summary_path.unlink(missing_ok=True)
    except OSError as e:
      print(f"[watch] could not remove stale {summary_path} ({e})", flush=True)
    print(f"[watch] {rt.now_iso()} evaluating iteration {it} ({path.name}), attempt {self.state['attempts'][key]}", flush=True)
    try:
      rc = self.launcher(self._cmd(snap, it, source=path), log_path)
    except Exception as e:  # noqa: BLE001  (launcher failure = failed attempt, keep watching)
      print(f"[watch] launcher error: {e}")
      rc = -1
    finally:
      if snap != path:
        shutil.rmtree(snap.parent, ignore_errors=True)
    dt = time.time() - t0
    summary = rt.read_json(summary_path) if rc == 0 else None
    if summary is not None and summary.get("iteration", it) != it:
      print(f"[watch] iteration {it}: summary.json reports iteration {summary.get('iteration')}; rejected", flush=True)
      summary = None
    if rc == 0 and summary is not None:
      o = summary.get("overall", {})
      self.state["evaluated"][key] = {"time": rt.now_iso(), "seconds": round(dt, 1), "fall_rate": o.get("fall_rate"),
                                      "reward_mean": o.get("reward_mean"), "success_rate": o.get("success_rate"),
                                      "checkpoint": str(path)}
      self.state["interrupted"].pop(key, None)
      print(f"[watch] iteration {it}: fall={o.get('fall_rate')} success={o.get('success_rate')} "
            f"reward={o.get('reward_mean')} ({dt:.0f}s)", flush=True)
      ok = True
    elif self.aborted:
      # The user aborted this evaluation: not the checkpoint's fault, so the
      # attempt does not count towards --max-retries.
      self.state["attempts"][key] = max(0, int(self.state["attempts"][key]) - 1)
      self.state["interrupted"][key] = {"time": rt.now_iso(), "exit_code": rc, "log": str(log_path)}
      print(f"[watch] iteration {it}: evaluation aborted (exit {rc}); will be retried on restart", flush=True)
      ok = False
    else:
      if rc == 0:
        print(f"[watch] iteration {it} FAILED (exit 0 but no valid summary.json); see {log_path}", flush=True)
      else:
        print(f"[watch] iteration {it} FAILED (exit {rc}); see {log_path}", flush=True)
      if self.state["attempts"][key] > self.cfg.max_retries:
        self.state["failed"][key] = {"time": rt.now_iso(), "exit_code": rc, "log": str(log_path)}
        print(f"[watch] giving up on iteration {it} after {self.state['attempts'][key]} attempts", flush=True)
      ok = False
    rt.write_json(self.state_path, self.state)
    if self.cfg.keep_videos > 0:
      ck.prune_dirs(self.run_dir / "videos", self.cfg.keep_videos)
    return ok

  def poll_once(self, now: float | None = None) -> int:
    n = 0
    for it, path in self.pending(now):
      if self.stop:
        break
      self.evaluate(it, path)
      n += 1
    return n

  def run(self) -> None:
    print(f"[watch] watching {self.run_dir} every {self.cfg.interval_s:.0f}s "
          f"({len(self.state['evaluated'])} already evaluated)", flush=True)
    while not self.stop:
      try:
        self.poll_once()
      except Exception as e:  # noqa: BLE001  (e.g. transient FS error)
        print(f"[watch] poll error (continuing): {type(e).__name__}: {e}", flush=True)
      if self.cfg.once:
        break
      end = time.time() + self.cfg.interval_s
      while not self.stop and time.time() < end:
        time.sleep(0.5)
    print("[watch] stopped.", flush=True)

  def request_stop(self, reason: str) -> None:
    """Graceful stop: finish the current evaluation, then exit."""
    if not self.stop:
      print(f"[watch] {reason}: stopping after the current evaluation "
            "(Ctrl-C again to abort it).", flush=True)
    self.stop = True

  def abort(self, reason: str) -> None:
    """Hard stop: terminate the running evaluation's process group."""
    self.stop = True
    self.aborted = True
    if self.child is not None:
      print(f"[watch] {reason} again: terminating the running evaluation.", flush=True)
      self._signal_child(signal.SIGTERM)
    else:
      print(f"[watch] {reason} again: exiting.", flush=True)


def resolve_run(cfg: WatchConfig) -> Path:
  if cfg.run in ("latest", ""):
    run = ck.latest_run(Path(cfg.run_root))
    if run is None:
      raise SystemExit(f"No runs under '{cfg.run_root}'.")
    return run
  p = Path(cfg.run).expanduser()
  if not p.is_dir():
    raise SystemExit(f"Run directory '{p}' does not exist.")
  return p


def acquire_watch_lock(run_dir: Path):
  """Hold ``metrics/watch.lock`` or exit with :data:`LOCKED_EXIT_CODE`."""
  path = Path(run_dir) / "metrics" / "watch.lock"
  try:
    return rt.acquire_lock(path)
  except rt.LockHeld as e:
    print(f"[watch] another contact-watch is already watching {run_dir} ({e}); exiting.",
          file=sys.stderr, flush=True)
    raise SystemExit(LOCKED_EXIT_CODE) from e


def main(argv: list[str] | None = None) -> None:
  import tyro

  cfg = tyro.cli(WatchConfig, args=sys.argv[1:] if argv is None else argv, config=(tyro.conf.FlagConversionOff,))
  run_dir = resolve_run(cfg)
  lock = acquire_watch_lock(run_dir)
  try:
    rt.install_console_tee(run_dir / "logs" / "watch.log")
    w = Watcher(cfg, run_dir)
    rt.install_stop_handlers(w.request_stop, on_abort=w.abort)
    w.run()
  finally:
    lock.close()


if __name__ == "__main__":
  main()
