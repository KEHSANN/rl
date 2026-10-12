"""Regression tests for the second audit pass (VPS scripts, systemd units,
``contact-train`` startup / shutdown, runtime helpers).

Every test here runs without a GPU, torch, mjlab, uv or tmux:

* shell scripts run against **fake ``uv`` / ``tmux``** executables that only
  record how they were called (see :func:`fake_tools`);
* ``contact-train``'s startup path runs against **stub modules** for torch /
  mjlab / the project runner (see :func:`stub_train_stack`), so the real
  signal handling, ``run_info.json`` finalisation and env cleanup code of
  :func:`contact_rl.scripts.train.run_contact_train` is exercised;
* the ONNX temp-file test needs torch and is skipped without it.

Each test names the bug it guards against in its docstring. Keep the fake
executables tiny: they document the *interface* the scripts rely on
(``tmux has-session`` / ``new-session ... <command>``, ``uv <args>``).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from contact_rl.scripts import train
from contact_rl.utils import runtime as rt

ROOT = Path(__file__).resolve().parents[1]
VPS = ROOT / "scripts" / "vps"
UNITS = VPS / "systemd"
HAS_BASH = shutil.which("bash") is not None
needs_bash = pytest.mark.skipif(not HAS_BASH, reason="bash not available")


# --------------------------------------------------------------------- helpers


def _unit(name: str) -> dict[str, list[str]]:
  """``key -> [values]`` of a systemd unit file (comments ignored)."""
  out: dict[str, list[str]] = {}
  for line in (UNITS / name).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "[")) or "=" not in line:
      continue
    k, v = line.split("=", 1)
    out.setdefault(k, []).append(v)
  return out


@pytest.fixture
def fake_tools(tmp_path):
  """A ``bin/`` dir with fake ``uv`` and ``tmux`` plus a ``run(script, *args)``
  helper that executes a VPS script with that dir first on PATH.

  * ``uv`` appends one record per call to ``uv.log``: ``UV_NO_SYNC=<value>``
    followed by one ``ARG=<arg>`` line per argument, then ``---``.
  * ``tmux has-session`` always fails (no session yet); ``tmux new-session``
    stores its last argument -- the shell command run inside the session --
    in ``tmux.cmd``.

  ``run_session()`` then executes that stored command the way tmux would
  (``sh -c``) with stdin closed, so the trailing ``exec bash`` exits at once.
  """
  bindir = tmp_path / "bin"
  bindir.mkdir()
  log = tmp_path / "uv.log"
  cmd = tmp_path / "tmux.cmd"
  (bindir / "uv").write_text(
    "#!/usr/bin/env bash\n"
    '{ printf "UV_NO_SYNC=%s\\n" "${UV_NO_SYNC-<unset>}"; for a in "$@"; do printf "ARG=%s\\n" "$a"; done; '
    'echo ---; } >> "$FAKE_UV_LOG"\n'
  )
  (bindir / "tmux").write_text(
    "#!/usr/bin/env bash\n"
    'case "$1" in\n'
    "  has-session) exit 1 ;;\n"
    '  new-session) for a in "$@"; do last="$a"; done; printf "%s" "$last" > "$FAKE_TMUX_CMD" ;;\n'
    "  *) exit 0 ;;\n"
    "esac\n"
  )
  for f in bindir.iterdir():
    f.chmod(0o755)
  env = {k: v for k, v in os.environ.items() if k not in ("UV_NO_SYNC", "SESSION", "TASK", "EXTRA", "PORT")}
  env.update(PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", FAKE_UV_LOG=str(log), FAKE_TMUX_CMD=str(cmd))

  def run(script: str, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
      ["bash", str(VPS / script), *args], env={**env, **extra_env}, capture_output=True, text=True,
      stdin=subprocess.DEVNULL, timeout=30,
    )

  def run_session() -> subprocess.CompletedProcess:
    return subprocess.run(
      ["sh", "-c", cmd.read_text()], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
    )

  def uv_calls() -> list[dict]:
    calls, cur = [], {"args": []}
    for line in log.read_text().splitlines() if log.exists() else []:
      if line == "---":
        calls.append(cur)
        cur = {"args": []}
      elif line.startswith("UV_NO_SYNC="):
        cur["UV_NO_SYNC"] = line.split("=", 1)[1]
      elif line.startswith("ARG="):
        cur["args"].append(line[4:])
    return calls

  return SimpleNamespace(run=run, run_session=run_session, uv_calls=uv_calls, cmd=cmd)


@pytest.fixture
def restore_signals():
  """Restore SIGINT / SIGTERM / SIGHUP handlers changed by the code under test."""
  sigs = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
  saved = {s: signal.getsignal(s) for s in sigs}
  yield
  for s, h in saved.items():
    signal.signal(s, h)


@pytest.fixture
def stub_train_stack(monkeypatch, tmp_path, restore_signals):
  """Stub torch / mjlab / the project runner so ``run_contact_train`` can run
  on a machine without the simulator stack. Returns a namespace whose
  ``env_cls`` / ``wrapper_cls`` a test replaces to inject behaviour (e.g.
  sending a signal while the env is "being built"), a ready ``cfg`` and a
  ``finalisations`` list recording every final ``run_info.json`` write."""

  def mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # type: ignore[attr-defined]  (behave like a package)
    m.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, m)
    return m

  ns = SimpleNamespace(env_cls=None, wrapper_cls=None, finalisations=[])

  class TrainingStopped(Exception):
    pass

  class ContactOnPolicyRunner:  # never constructed by these tests
    pass

  mod("torch", cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0, set_device=lambda d: None))
  mod("mjlab")
  mod("mjlab.envs", ManagerBasedRlEnv=lambda *a, **k: ns.env_cls(*a, **k))
  mod("mjlab.rl", MjlabOnPolicyRunner=object, RslRlVecEnvWrapper=lambda *a, **k: ns.wrapper_cls(*a, **k))
  mod("mjlab.tasks")
  mod("mjlab.tasks.registry", load_runner_cls=lambda task: None)
  mod("mjlab.utils")
  mod("mjlab.utils.os", dump_yaml=lambda *a, **k: None)
  mod("mjlab.utils.torch", configure_torch_backends=lambda: None)
  for name in ("contact_rl.tasks", "contact_rl.tasks.contact", "contact_rl.tasks.contact.config",
               "contact_rl.tasks.contact.config.go2"):
    mod(name)
  mod("contact_rl.tasks.contact.config.go2.runner", ContactOnPolicyRunner=ContactOnPolicyRunner,
      TrainingStopped=TrainingStopped)
  mod("contact_rl.utils.video", StreamingVideoRecorder=None)

  monkeypatch.setattr(rt, "install_console_tee", lambda path: None)  # keep pytest's stdout
  real_update = rt.update_run_info

  def counting_update(run_dir, **fields):
    if "finished" in fields:
      ns.finalisations.append(fields)
    return real_update(run_dir, **fields)

  monkeypatch.setattr(rt, "update_run_info", counting_update)
  ns.cfg = SimpleNamespace(
    device="cpu", gpu_ids=[0], resume_from=None, run_root=str(tmp_path / "runs"), video=False,
    enable_nan_guard=False, keep_last=5, keep_every=1000, keep_best=True, export_onnx=False,
    agent=SimpleNamespace(resume=False, experiment_name="exp", run_name="", seed=1, num_steps_per_env=24,
                          clip_actions=None),
    env=SimpleNamespace(seed=None, sim=SimpleNamespace(nan_guard=SimpleNamespace(enabled=False))),
  )
  ns.run_info = lambda: json.loads(next((tmp_path / "runs" / "exp").glob("*/run_info.json")).read_text())
  return ns


def _wait_for_signal() -> None:
  """Give the interpreter a chance to run a pending Python signal handler."""
  for _ in range(200):
    time.sleep(0.005)
  raise AssertionError("the signal handler did not interrupt startup")


# ---------------------------------------------------------- 1. systemd units


def test_systemd_units_environment_and_stop_policy():
  """All units: unbuffered logs, UV_NO_SYNC=1 (a plain ``uv run`` would swap
  the CUDA torch build), no ``network-online.target`` (user managers do not
  provide it). Train / watch stop gracefully via SIGTERM to uv only
  (KillMode=mixed); the trainer is never restarted; the watcher does not
  restart-loop on exit 3; TensorBoard has a stop timeout and restart delay."""
  for name in ("contact-train@.service", "contact-watch.service", "contact-tensorboard.service"):
    u = _unit(name)
    raw = (UNITS / name).read_text()
    assert "PYTHONUNBUFFERED=1" in u["Environment"], name
    assert "UV_NO_SYNC=1" in u["Environment"], name
    for key in ("After", "Wants", "Requires"):
      assert not any("network-online.target" in v for v in u.get(key, [])), name
    assert "network-online.target" not in "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    assert u["ExecStart"][0].split()[1] == "run", name
  train_u, watch_u, tb_u = (_unit(n) for n in ("contact-train@.service", "contact-watch.service",
                                                "contact-tensorboard.service"))
  assert train_u["KillMode"] == ["mixed"] and train_u["KillSignal"] == ["SIGTERM"] and train_u["Restart"] == ["no"]
  assert watch_u["KillMode"] == ["mixed"] and watch_u["RestartPreventExitStatus"] == ["3"]
  assert int(tb_u["TimeoutStopSec"][0]) > 0 and int(tb_u["RestartSec"][0]) > 0


# -------------------------------------------------- 2-6. VPS shell scripts


@needs_bash
def test_tmux_wrap_propagates_uv_no_sync():
  """A running tmux server keeps its own stale environment, so tmux_wrap must
  export UV_NO_SYNC (and the GPU / GL variables) inside the session."""
  src = (VPS / "common.sh").read_text()
  assert 'export UV_NO_SYNC="${UV_NO_SYNC:-1}"' in src
  fn = src[src.index("tmux_wrap() {"):]
  out = subprocess.run(
    ["bash", "-c", fn + "\ntmux_wrap uv run contact-train T"], env={"PATH": os.environ.get("PATH", ""),
    "UV_NO_SYNC": "1", "CUDA_VISIBLE_DEVICES": "1"}, capture_output=True, text=True, check=True,
  ).stdout
  assert "export UV_NO_SYNC=1;" in out and "export CUDA_VISIBLE_DEVICES=1;" in out
  assert out.index("export UV_NO_SYNC") < out.index("uv run")


@needs_bash
def test_setup_sh_syncs_once_with_selected_extra(fake_tools):
  """setup.sh is the only script that syncs: ``--extra cu128`` by default,
  ``EXTRA=cpu`` for a CPU box; the doctor call runs with UV_NO_SYNC=1."""
  r = fake_tools.run("setup.sh", "--skip-env", EXTRA="cpu")
  assert r.returncode == 0, r.stderr
  sync, doctor = fake_tools.uv_calls()
  assert sync["args"][:3] == ["sync", "--extra", "cpu"] and "--frozen" in sync["args"]
  assert doctor["args"] == ["run", "contact-doctor", "--skip-env"] and doctor["UV_NO_SYNC"] == "1"
  r = fake_tools.run("setup.sh")
  assert r.returncode == 0 and fake_tools.uv_calls()[2]["args"][:3] == ["sync", "--extra", "cu128"]


@needs_bash
@pytest.mark.parametrize(
  "args, rc",
  [((), 2), (("--env.scene.num-envs", "4096"), 2), (("Mjlab-Contact-Flat-Unitree-Go2", "--env.scene.num-envs", "4096"), 0)],
)
def test_train_tmux_task_handling(fake_tools, args, rc):
  """train_tmux.sh without a task -- or with an option where the task should
  be -- exits 2 with usage instead of starting ``contact-train --option``.
  With a task, the tmux session (fake tmux + fake uv, end to end) runs
  ``uv run contact-train <task> <args>`` with UV_NO_SYNC=1 and reports the
  real exit code."""
  r = fake_tools.run("train_tmux.sh", *args)
  assert r.returncode == rc, r.stderr
  if rc:
    assert "usage:" in r.stderr and not fake_tools.cmd.exists()
    return
  assert "[exited with 0]" in fake_tools.run_session().stdout
  (call,) = fake_tools.uv_calls()
  assert call["UV_NO_SYNC"] == "1" and call["args"] == ["run", "contact-train", *args]


@needs_bash
@pytest.mark.parametrize(
  "script, args, expected",
  [
    ("play.sh", ["--num-envs", "16"],
     ["run", "contact-play", "Mjlab-Contact-Flat-Unitree-Go2", "--checkpoint", "best", "--host", "127.0.0.1",
      "--port", "8080", "--num-envs", "16"]),
    ("play.sh", ["runs/exp/r:1500"],
     ["run", "contact-play", "Mjlab-Contact-Flat-Unitree-Go2", "--checkpoint", "runs/exp/r:1500", "--host",
      "127.0.0.1", "--port", "8080"]),
    ("watch_tmux.sh", ["--interval-s", "60"], ["run", "contact-watch", "--run", "latest", "--interval-s", "60"]),
    ("watch_tmux.sh", ["runs/exp/r", "--num-envs", "8"],
     ["run", "contact-watch", "--run", "runs/exp/r", "--num-envs", "8"]),
    ("tensorboard.sh", ["--reload_interval", "5"],
     ["run", "tensorboard", "--logdir", "runs", "--host", "127.0.0.1", "--port", "6006", "--reload_interval", "5"]),
  ],
)
def test_leading_option_is_not_taken_as_selector(fake_tools, script, args, expected):
  """The optional first argument (checkpoint / run / logdir) is only taken if
  it does not start with '-': ``play.sh --num-envs 16`` used to run
  ``--checkpoint --num-envs 16``."""
  r = fake_tools.run(script, *args)
  assert r.returncode == 0, r.stderr
  fake_tools.run_session()
  (call,) = fake_tools.uv_calls()
  assert call["args"] == expected and call["UV_NO_SYNC"] == "1"


@needs_bash
@pytest.mark.parametrize(
  "args, ckpt, tail",
  [(["--seed", "3"], "best", ["--seed", "3"]), (["latest", "--seed", "3"], "latest", ["--seed", "3"]), ([], "best", [])],
)
def test_eval_sh_selector_and_options(fake_tools, args, ckpt, tail):
  """eval.sh (runs uv directly, no tmux): same selector rule as play.sh."""
  r = fake_tools.run("eval.sh", *args)
  assert r.returncode == 0, r.stderr
  (call,) = fake_tools.uv_calls()
  assert call["args"] == ["run", "contact-eval", "--task", "Mjlab-Contact-Flat-Unitree-Go2", "--checkpoint", ckpt,
                          "--mode", "episodes", "--num-envs", "256", "--video", "True", *tail]
  assert call["UV_NO_SYNC"] == "1"


# -------------------------------------------------- 7-8. train.py helpers


def test_run_status_mapping():
  """A run is only "finished" if training completed; a stop request is
  "stopped", 130 "aborted", anything else "failed"."""
  assert train.run_status(0) == "finished"
  assert train.run_status(0, "SIGTERM during startup") == "stopped"
  assert train.run_status(130) == "aborted"
  assert train.run_status(1) == train.run_status(2) == "failed"
  assert train.RunOutcome(code=0, stop_reason="SIGINT").status == "stopped"


def test_gpu_counting_and_multi_gpu_detection(monkeypatch):
  """CUDA_VISIBLE_DEVICES is authoritative ("" = none, "-1" hides the rest);
  only >1 selected GPU is multi-GPU (``all`` on a 1-GPU box is not)."""
  for value, n in (("", 0), ("0", 1), ("0,1", 2), (" 2 , 3 ,", 2), ("0,-1,1", 1), ("-1", 0)):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert train.count_visible_gpus() == n, value
  assert train.is_multi_gpu([0, 1]) and not train.is_multi_gpu([1]) and not train.is_multi_gpu(None)
  assert train.is_multi_gpu("all", visible=lambda: 2) and not train.is_multi_gpu("all", visible=lambda: 1)


# -------------------------------------------- 9-11. contact-train startup stop


def test_sigterm_during_env_construction_finalises_as_stopped(stub_train_stack):
  """SIGTERM (``systemctl --user stop``) while the env is being built used to
  kill the process with run_info.json left at "starting". Now: exit 0,
  status "stopped", exactly one finalisation, default handlers restored."""
  ns = stub_train_stack

  def building_env(*a, **k):
    os.kill(os.getpid(), signal.SIGTERM)
    _wait_for_signal()

  ns.env_cls = building_env
  assert train.run_contact_train("T", ns.cfg) == 0
  info = ns.run_info()
  assert info["status"] == "stopped" and info["exit_code"] == 0 and info["error"] is None
  assert info["stop_reason"] == "SIGTERM during startup" and info["finished"]
  assert info["last_checkpoint_iteration"] is None
  assert len(ns.finalisations) == 1
  assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
  assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_duplicate_ctrl_c_during_startup_is_a_clean_stop(stub_train_stack):
  """Ctrl-C through ``uv run`` / ``tmux send-keys`` can arrive twice. During
  startup the first one stops cleanly (exit 0, not a 130 hard abort); a
  second one arriving while the env is being closed is ignored, the env is
  still closed and run_info.json is finalised exactly once."""
  ns = stub_train_stack

  class Env:
    closed = False

    def close(self):
      os.kill(os.getpid(), signal.SIGINT)  # the duplicate delivery, mid-cleanup
      time.sleep(0.05)
      Env.closed = True

  def wrapper(env, **k):
    os.kill(os.getpid(), signal.SIGINT)
    _wait_for_signal()

  ns.env_cls = lambda *a, **k: Env()
  ns.wrapper_cls = wrapper
  assert train.run_contact_train("T", ns.cfg) == 0
  info = ns.run_info()
  assert info["status"] == "stopped" and info["stop_reason"] == "SIGINT during startup"
  assert Env.closed and len(ns.finalisations) == 1


def test_startup_guard_and_finalisation_are_idempotent(tmp_path, restore_signals):
  """Unit level: the guard raises StartupStop once (a BaseException, so no
  ``except Exception`` in library code can swallow it) and ignores every
  later signal; finalize_run_info writes once, and never without a run dir."""
  guard = train.StartupSignalGuard()
  with pytest.raises(train.StartupStop) as ei:
    guard(signal.SIGTERM)
  assert not isinstance(ei.value, Exception) and ei.value.signal_name == "SIGTERM"
  guard(signal.SIGINT)
  guard(signal.SIGTERM)
  assert guard.ignored == 2

  outcome = train.RunOutcome(code=0, stop_reason=str(ei.value))
  assert train.finalize_run_info(None, outcome) is False and not outcome.finalized
  assert train.finalize_run_info(tmp_path, outcome, 7) is True
  outcome.code = 130  # a later (buggy) second finalisation must not overwrite the status
  assert train.finalize_run_info(tmp_path, outcome) is False
  info = json.loads((tmp_path / "run_info.json").read_text())
  assert info["status"] == "stopped" and info["exit_code"] == 0 and info["last_checkpoint_iteration"] == 7

  with train.stop_signals_ignored():
    os.kill(os.getpid(), signal.SIGINT)  # ignored, no KeyboardInterrupt
    time.sleep(0.02)
  assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# -------------------------------------------------- 12-15. runtime / ONNX helpers


def test_write_json_is_atomic(tmp_path):
  """A failing serialisation leaves the previous document intact and no
  temp file behind; temp names are per-process (no interleaving)."""
  path = tmp_path / "run_info.json"
  rt.write_json(path, {"status": "starting"})

  class Unserialisable:
    def __repr__(self):
      raise RuntimeError("boom")

  with pytest.raises(RuntimeError):
    rt.write_json(path, {"status": "running", "bad": Unserialisable()})
  assert json.loads(path.read_text()) == {"status": "starting"}
  assert sorted(p.name for p in tmp_path.iterdir()) == ["run_info.json"]
  assert rt.update_run_info(tmp_path, status="stopped")["status"] == "stopped"


def test_duplicate_sigint_window_during_training():
  """StopSignalHandler: 1st signal stops, a SIGINT within the window is a
  duplicate (ignored), a later one aborts; repeated SIGTERM never aborts."""
  now = [0.0]
  stops: list[str] = []
  h = rt.StopSignalHandler(stops.append, window_s=1.0, clock=lambda: now[0])
  h(signal.SIGINT)
  now[0] = 0.5
  h(signal.SIGINT)
  assert stops == ["SIGINT"] and h.ignored == 1
  h(signal.SIGTERM)
  assert stops == ["SIGINT", "SIGTERM"]
  now[0] = 1.5
  with pytest.raises(KeyboardInterrupt):
    h(signal.SIGINT)


def test_run_lock_is_exclusive_and_released(tmp_path):
  """metrics/watch.lock: a second holder is refused (the watcher then exits
  3), and the lock is free again as soon as the holder closes / exits."""
  lock = tmp_path / "metrics" / "watch.lock"
  held = rt.acquire_lock(lock)
  try:
    with pytest.raises(rt.LockHeld, match=f"pid {os.getpid()}"):
      rt.acquire_lock(lock)
  finally:
    held.close()
  rt.acquire_lock(lock).close()


def test_onnx_export_failure_removes_temp_file(tmp_path, monkeypatch):
  """A failed export must not leave ``policy.onnx.tmp`` behind nor replace a
  previous good ``policy.onnx``; the GRU ``(actions, h, None)`` output of
  rsl-rl 5.0.1 is made exportable by dropping ``None``."""
  torch = pytest.importorskip("torch")
  from contact_rl.utils import onnx_compat

  class OnnxGRU(torch.nn.Module):
    input_names = ["obs", "h_in"]
    output_names = ["actions", "h_out"]

    def forward(self, obs, h):
      return obs, h, None

    def get_dummy_inputs(self):
      return (torch.zeros(1, 3), torch.zeros(1, 1, 4))

  policy = SimpleNamespace(as_onnx=lambda verbose=False: OnnxGRU())
  assert isinstance(onnx_compat.make_exportable(OnnxGRU()), onnx_compat.DropNoneOutputs)

  (tmp_path / "policy.onnx").write_bytes(b"previous-good")

  def failing_export(model, args, f, **kwargs):
    Path(f).write_bytes(b"partial")
    raise RuntimeError("exporter crashed")

  monkeypatch.setattr(torch.onnx, "export", failing_export)
  with pytest.raises(RuntimeError, match="exporter crashed"):
    onnx_compat.export_policy_onnx(policy, str(tmp_path), "policy.onnx")
  assert not (tmp_path / "policy.onnx.tmp").exists()
  assert (tmp_path / "policy.onnx").read_bytes() == b"previous-good"
