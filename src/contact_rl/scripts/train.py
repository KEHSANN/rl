"""``contact-train``: production training entry point.

Registers the contact tasks, then trains with mjlab's env + the project
runner, adding run management on top of mjlab's ``train`` CLI. Every mjlab
``TrainConfig`` flag still works (``--env.*``, ``--agent.*``, ``--video``,
``--video-length``, ``--video-interval``, ``--gpu-ids``,
``--enable-nan-guard`` ...), plus::

  --resume-from SPEC   checkpoint file | run dir | run_dir:<it>|best|latest |
                       latest | best  (full state; continues at iter+1 in a
                       NEW run dir, the source is never modified)
  --device DEV         cpu | cuda | cuda:<i> (default: first of --gpu-ids)
  --run-root DIR       default "runs" -> runs/<experiment>/<timestamp>[_name]
  --keep-last N        keep newest N checkpoints (<=0: keep all)
  --keep-every N       also keep multiples of N
  --keep-best BOOL     maintain checkpoints/best.pt (by mean episode reward)
  --export-onnx BOOL   write exported/policy.onnx at every save

Examples::

  uv run contact-train Mjlab-Contact-Flat-Unitree-Go2          # paper default: 8192 envs
  uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096   # lower-VRAM fallback

On the VPS run it with ``UV_NO_SYNC=1`` (exported by ``scripts/vps/common.sh``
and the systemd units) or as ``uv run --no-sync contact-train ...``: a plain
``uv run`` re-syncs the env without ``--extra cu128`` and swaps out the CUDA
torch build that ``scripts/vps/setup.sh`` installed.

``--enable-nan-guard True`` turns on mjlab's simulation NaN guard
(``env.sim.nan_guard.enabled``), exactly like mjlab's own ``train`` CLI.

GPUs: mjlab parses collections with Python syntax
(``tyro.conf.UsePythonSyntaxForLiteralCollections``), so ``--gpu-ids`` takes
one quoted list literal: ``--gpu-ids "[0]"`` (default), ``--gpu-ids "[1]"``,
``--gpu-ids "[0, 1]"`` or ``--gpu-ids all`` (``--gpu-ids 0 1`` is *not*
accepted). The ids index into ``CUDA_VISIBLE_DEVICES``. A single selected GPU
-- including ``all`` on a machine with one visible GPU -- trains in this
process with full run management. Only when more than one GPU is actually
selected (``"[0, 1]"``, or ``all`` with several visible GPUs) is training
delegated to mjlab's torchrunx launcher with a plain mjlab ``TrainConfig``
(legacy logs/rsl_rl layout, no run management; the contact-train-only flags
are ignored with a warning).

run_info.json: written as soon as the run dir exists (``status="starting"``),
updated to ``"running"`` once training starts, and always finalised with
``status`` (finished | stopped | aborted | failed), ``exit_code``, ``error``
(``null`` on success), ``finished`` and ``last_checkpoint_iteration``. A failed
run can never be recorded as a success; the traceback also goes to
``logs/error.txt``.

Signals: from the start of :func:`run_contact_train` until the runner's
graceful-stop handlers take over, :class:`StartupSignalGuard` is installed.
The first SIGINT / SIGTERM during startup (heavy imports, CUDA init, env
construction / Warp kernel compilation, runner construction, resume load)
raises :class:`StartupStop`; the run is finalised as ``status="stopped"``
(exit 0, no checkpoint) instead of being left at ``"starting"`` (SIGTERM used
to kill the process without running any cleanup) or recorded as a hard abort
(first Ctrl-C). Repeated signals during the stop are ignored. During training
the first SIGINT / SIGTERM finishes the iteration and writes a checkpoint, a
duplicate SIGINT within 1 s is ignored and a later Ctrl-C aborts (exit 130).
While ``run_info.json`` is finalised and the env is closed, SIGINT / SIGTERM
are ignored so the final status is written exactly once and never torn;
SIGKILL (systemd ``TimeoutStopSec``) still ends a hung cleanup.

Exit codes: 0 = finished or stopped cleanly (SIGINT/SIGTERM, also during
startup), 130 = hard abort (second Ctrl-C during training), 1 = any other
error (the traceback is re-raised). The env is closed even if env
construction, the runner, the resume load or training itself fails.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from contact_rl.utils import runtime as rt  # torch / mujoco free
from contact_rl.utils.runtime import configure_headless_rendering

configure_headless_rendering()  # before anything can import mujoco

# Flags that only contact-train understands (ignored by mjlab's launcher).
CONTACT_ONLY_FLAGS = ("resume_from", "device", "run_root", "keep_last", "keep_every", "keep_best", "export_onnx")


def _config_cls():
  from mjlab.scripts.train import TrainConfig

  @dataclass(frozen=True)
  class ContactTrainConfig(TrainConfig):
    resume_from: str | None = None
    """Checkpoint selector to resume from (see module docstring)."""
    device: str | None = None
    """cpu | cuda | cuda:<index>. Default: cuda:<first --gpu-ids> or cpu."""
    run_root: str = "runs"
    keep_last: int = 5
    keep_every: int = 1000
    keep_best: bool = True
    export_onnx: bool = True

  return ContactTrainConfig


# ------------------------------------------------------------ torch-free helpers


def count_visible_gpus() -> int:
  """Number of CUDA devices this process can use. ``CUDA_VISIBLE_DEVICES``
  (if set) is authoritative; otherwise ask torch (0 if torch/CUDA is
  unavailable)."""
  env = os.environ.get("CUDA_VISIBLE_DEVICES")
  if env is not None:
    ids = [s.strip() for s in env.split(",") if s.strip()]
    out = []
    for s in ids:
      if s.startswith("-"):  # "-1" hides this and all following devices
        break
      out.append(s)
    return len(out)
  try:
    import torch

    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
  except Exception:
    return 0


def is_multi_gpu(gpu_ids: Any, visible: Callable[[], int] = count_visible_gpus) -> bool:
  """True only if training would really use more than one GPU. ``"all"`` on a
  single-GPU machine is a single-GPU run (keeps contact-train run management)."""
  if gpu_ids == "all":
    return visible() > 1
  if isinstance(gpu_ids, (list, tuple)):
    return len(gpu_ids) > 1
  return False


def apply_nan_guard(cfg: Any) -> bool:
  """Connect ``--enable-nan-guard`` to the env config (mjlab's ``run_train``
  does the same). Returns whether the guard is enabled."""
  if getattr(cfg, "enable_nan_guard", False):
    cfg.env.sim.nan_guard.enabled = True
  guard = getattr(getattr(cfg.env, "sim", None), "nan_guard", None)
  return bool(getattr(guard, "enabled", False))


def close_quietly(env: Any) -> None:
  """Close ``env`` (if any) without letting a close failure mask the real error."""
  if env is None:
    return
  try:
    env.close()
  except Exception as e:  # noqa: BLE001
    print(f"[WARN] env close failed: {e}")


def run_status(code: int, stop_reason: str | None = None) -> str:
  """Final ``run_info.json`` status for an exit code: never "finished" unless
  training really completed (code 0 without a stop request)."""
  if code == 0:
    return "stopped" if stop_reason else "finished"
  if code == 130:
    return "aborted"
  return "failed"


def base_config(cfg: Any, base_cls: type) -> Any:
  """Copy the ``base_cls`` fields of ``cfg`` into a plain ``base_cls``
  instance. Used for multi-GPU delegation: torchrunx pickles the config for
  its workers, and the function-local ``ContactTrainConfig`` class cannot be
  pickled, while mjlab's module-level ``TrainConfig`` can."""
  kwargs = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(base_cls) if f.init}
  return base_cls(**kwargs)


def ignored_contact_flags(cfg: Any, defaults: Any) -> list[str]:
  """contact-train-only flags whose value differs from the default (these are
  ignored when training is delegated to mjlab's multi-GPU launcher)."""
  return [n for n in CONTACT_ONLY_FLAGS if getattr(cfg, n, None) != getattr(defaults, n, None)]


# ------------------------------------------------ startup-stop / finalisation


def _sig_name(signum: int) -> str:
  try:
    return signal.Signals(signum).name
  except ValueError:
    return f"signal {signum}"


class StartupStop(BaseException):
  """SIGINT / SIGTERM received before the runner's graceful-stop handlers were
  installed -- typically while the env is being built, which can take minutes
  on a first run while Warp compiles its kernels.

  Derives from ``BaseException`` (like ``KeyboardInterrupt``) so that broad
  ``except Exception`` blocks in mjlab, MuJoCo-Warp or our own helpers cannot
  swallow the stop request: it always unwinds to :func:`run_contact_train`,
  which records ``status="stopped"``."""

  def __init__(self, signal_name: str, phase: str = "startup"):
    super().__init__(f"{signal_name} during {phase}")
    self.signal_name = signal_name
    self.phase = phase


class StartupSignalGuard:
  """SIGINT / SIGTERM policy while contact-train starts up.

  Without it the default dispositions applied: SIGTERM (``systemctl --user
  stop``, ``kill``) terminated the process at once, skipping every ``finally``
  block, so ``run_info.json`` stayed at ``status="starting"``; the first
  Ctrl-C raised ``KeyboardInterrupt`` and was recorded as a hard abort (130)
  although nothing had started.

  * The first SIGINT / SIGTERM raises :class:`StartupStop` (clean stop).
  * Every later SIGINT / SIGTERM is ignored with a message: a stop is already
    in progress, and a duplicate delivery (``uv run`` forwarding the signal
    that the terminal also sent, ``tmux send-keys C-c``, systemd signalling
    both uv and its child) must neither interrupt the cleanup nor turn the
    stop into an abort.
  * SIGHUP is ignored (dropped SSH session), as during training.

  ``phase`` only labels the stop reason (``"<SIGNAL> during <phase>"``).
  """

  def __init__(self) -> None:
    self.signal_name: str | None = None
    self.ignored = 0
    self.phase = "startup"

  def __call__(self, signum: int, _frame: Any = None) -> None:
    name = _sig_name(signum)
    if self.signal_name is not None:
      self.ignored += 1
      print(f"[INFO] {name} ignored: already stopping on {self.signal_name}.", file=sys.stderr, flush=True)
      return
    self.signal_name = name
    raise StartupStop(name, self.phase)

  def install(self) -> "StartupSignalGuard":
    """Install for SIGINT / SIGTERM (and ignore SIGHUP). Signal handlers can
    only be set from the main thread; elsewhere this warns and does nothing."""
    try:
      signal.signal(signal.SIGINT, self)
      signal.signal(signal.SIGTERM, self)
      if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except ValueError as e:  # not the main thread
      print(f"[WARN] startup signal guard not installed: {e}", file=sys.stderr)
    return self


@contextlib.contextmanager
def stop_signals_ignored(what: str = "finalising the run") -> Iterator[None]:
  """Ignore SIGINT / SIGTERM inside the block (the final ``run_info.json``
  write and the env close), so a late or duplicate signal can neither tear
  the finalisation half-way nor cause a second one. On exit the process
  defaults are restored (Ctrl-C -> ``KeyboardInterrupt``, SIGTERM ->
  terminate): the run is finalised, nothing is left to protect. SIGKILL
  (e.g. systemd's ``TimeoutStopSec``) still ends a hung cleanup."""

  def _ignore(signum: int, _frame: Any = None) -> None:
    print(
      f"[INFO] {_sig_name(signum)} ignored while {what} (kill -9 {os.getpid()} to force).",
      file=sys.stderr,
      flush=True,
    )

  def _set(int_handler: Any, term_handler: Any) -> None:
    try:
      signal.signal(signal.SIGINT, int_handler)
      signal.signal(signal.SIGTERM, term_handler)
    except ValueError:  # not the main thread: leave the handlers alone
      pass

  _set(_ignore, _ignore)
  try:
    yield
  finally:
    _set(signal.default_int_handler, signal.SIG_DFL)


@dataclass
class RunOutcome:
  """What :func:`run_contact_train` records in ``run_info.json`` and returns."""

  code: int = 1
  error: str | None = None
  stop_reason: str | None = None
  finalized: bool = False

  @property
  def status(self) -> str:
    return run_status(self.code, self.stop_reason)


def finalize_run_info(run_dir: Path | None, outcome: RunOutcome, last_checkpoint_iteration: int | None = None) -> bool:
  """Record the final status in ``run_info.json`` exactly once.

  Returns ``False`` without writing if there is no run dir yet (the process
  was stopped / failed before it was created) or the run is already
  finalised. The write itself is atomic (:func:`runtime.write_json`); a write
  failure is reported but never masks the real error, and is not retried."""
  if run_dir is None or outcome.finalized:
    return False
  outcome.finalized = True
  try:
    rt.update_run_info(
      run_dir,
      finished=rt.now_iso(),
      exit_code=outcome.code,
      status=outcome.status,
      error=outcome.error,
      stop_reason=outcome.stop_reason,
      last_checkpoint_iteration=last_checkpoint_iteration,
    )
  except Exception as e:  # noqa: BLE001 - never mask the real error
    print(f"[WARN] could not finalise run_info.json: {e}")
  return True


def _pick_device(cfg) -> str:
  from contact_rl.utils.runtime import resolve_device

  if cfg.device:
    return resolve_device(cfg.device)
  ids = cfg.gpu_ids
  if ids is None:
    return "cpu"
  if ids == "all":
    if count_visible_gpus() == 0:
      print("[WARN] --gpu-ids all but no CUDA device is visible; training on cpu.")
      return "cpu"
    return resolve_device("cuda:0")
  return resolve_device(f"cuda:{ids[0]}") if ids else "cpu"


def _legacy_resume_path(cfg) -> Path | None:
  """mjlab's ``--agent.resume True`` (+ ``--agent.load-run`` /
  ``--agent.load-checkpoint`` or ``--wandb-run-path``). Looks in mjlab's
  ``logs/rsl_rl/<experiment>`` first, then in ``<run-root>/<experiment>``."""
  if not cfg.agent.resume or cfg.resume_from:
    return None
  from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path

  from contact_rl.utils.checkpoints import resolve_checkpoint

  legacy_root = Path("logs/rsl_rl") / cfg.agent.experiment_name
  if cfg.wandb_run_path:
    path, _ = get_wandb_checkpoint_path(legacy_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name)
    return Path(path)
  try:
    return Path(get_checkpoint_path(legacy_root, cfg.agent.load_run, cfg.agent.load_checkpoint))
  except Exception:
    return resolve_checkpoint(str(Path(cfg.run_root) / cfg.agent.experiment_name))


def run_contact_train(task_id: str, cfg) -> int:
  # The startup guard goes in first, before anything slow (heavy imports, CUDA
  # init, resume resolution, env construction): from here on SIGINT / SIGTERM
  # always unwind through the handlers below, so run_info.json is finalised
  # (see the module docstring, "Signals").
  guard = StartupSignalGuard()
  outcome = RunOutcome()
  run_dir: Path | None = None
  closable: Any = None
  runner = None
  try:
    guard.install()
    import torch

    import contact_rl
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    from contact_rl.tasks.contact.config.go2.runner import ContactOnPolicyRunner, TrainingStopped
    from contact_rl.utils import checkpoints as ck
    from contact_rl.utils.video import StreamingVideoRecorder

    device = _pick_device(cfg)
    rt.set_egl_device_for(device)
    if device.startswith("cuda"):
      torch.cuda.set_device(device)
    configure_torch_backends()

    resume_path: Path | None = None
    if cfg.resume_from:
      resume_path = ck.resolve_checkpoint(cfg.resume_from, Path(cfg.run_root))
    else:
      resume_path = _legacy_resume_path(cfg)

    run_dir = ck.create_run_dir(cfg.run_root, cfg.agent.experiment_name, cfg.agent.run_name or None)
    rt.install_console_tee(run_dir / "logs" / "train.log")
    rt.update_run_info(run_dir, status="starting", task=task_id, started=rt.now_iso(), pid=os.getpid(), argv=sys.argv)

    # From here on a failing env construction, runner construction, resume
    # load, CUDA OOM or NaN still closes whatever was built (EGL context,
    # ffmpeg) and records the real status / exit code / error.
    cfg.env.seed = cfg.agent.seed
    nan_guard = apply_nan_guard(cfg)  # must happen before the env (and its sim) is built
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
    closable = env
    nspe = int(cfg.agent.num_steps_per_env)
    if cfg.video:
      env = StreamingVideoRecorder(
        env,
        path_fn=lambda s: run_dir / "videos" / "train" / f"iteration_{s // nspe:06d}" / f"step_{s:09d}.mp4",
        step_trigger=lambda s: s % cfg.video_interval == 0,
        video_length=cfg.video_length,
        hud_fn=lambda s: [f"train it {s // nspe}", f"step {s}"],
      )
      closable = env
    venv = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    closable = venv

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(venv, agent_cfg, str(run_dir), device)
    managed = isinstance(runner, ContactOnPolicyRunner)
    if managed:
      runner.configure_run(
        run_dir, ck.RetentionPolicy(cfg.keep_last, cfg.keep_every, cfg.keep_best), export_onnx=cfg.export_onnx
      )
    else:
      print(f"[WARN] runner {runner_cls.__name__} is not a ContactOnPolicyRunner: no checkpoint management.")
    runner.add_git_repo_to_log(contact_rl.__file__)

    resume_meta = None
    if resume_path is not None:
      if managed:
        loaded = runner.load_for_resume(resume_path, map_location=device)
      else:
        runner.load(str(resume_path), map_location=device)
        loaded = int(runner.current_learning_iteration)
      resume_meta = {
        "source": str(resume_path),
        "source_run": str(ck.run_dir_of_checkpoint(resume_path)),
        "source_iteration": loaded,
        "start_iteration": int(runner.current_learning_iteration),
      }

    dump_yaml(run_dir / "config" / "env.yaml", env_cfg)
    dump_yaml(run_dir / "config" / "agent.yaml", agent_cfg)
    rt.write_json(run_dir / "config" / "cli.json", {"argv": sys.argv, "task": task_id})
    gpu = rt.hardware_info()
    info = rt.update_run_info(
      run_dir,
      status="running",
      task=task_id,
      experiment=cfg.agent.experiment_name,
      device=device,
      gpu_ids=cfg.gpu_ids,
      num_envs=int(cfg.env.scene.num_envs),
      max_iterations=int(cfg.agent.max_iterations),
      seed=int(cfg.agent.seed),
      logger=agent_cfg.get("logger"),
      nan_guard=nan_guard,
      git=rt.git_info(),
      versions=rt.distribution_versions(),
      hardware=gpu,
      mujoco_gl=os.environ.get("MUJOCO_GL"),
      resume=resume_meta,
      retention={"keep_last": cfg.keep_last, "keep_every": cfg.keep_every, "keep_best": cfg.keep_best},
    )
    start = int(runner.current_learning_iteration)
    print("=" * 78)
    print(f"[contact-train] task={task_id}  device={device}  num_envs={cfg.env.scene.num_envs}  "
          f"logger={agent_cfg.get('logger')}  nan_guard={nan_guard}")
    print(f"[contact-train] run dir: {run_dir}")
    print(f"[contact-train] iterations {start} .. {start + cfg.agent.max_iterations - 1} (save every {cfg.agent.save_interval})")
    print(f"[contact-train] git: {json.dumps(info.get('git'))}")
    v = info["versions"]
    print(f"[contact-train] torch={v.get('torch')} mjlab={v.get('mjlab')} rsl-rl={v.get('rsl-rl-lib')} mujoco={v.get('mujoco')} warp={v.get('warp-lang')}")
    for g in gpu.get("gpus", []):
      print(f"[contact-train] GPU{g['index']}: {g['name']} {g['vram_gib']} GiB (sm_{g['capability']})")
    if resume_meta:
      print(f"[contact-train] resumed from {resume_meta['source']} (iter {resume_meta['source_iteration']}) -> starting at {start}")
    print(f"[contact-train] TensorBoard: uv run tensorboard --logdir {Path(cfg.run_root)} --host 127.0.0.1 --port 6006")
    print("=" * 78, flush=True)

    if managed:
      # First SIGINT/SIGTERM: stop after this iteration; a duplicate SIGINT
      # (uv run / tmux double delivery) within 1 s is ignored; a later Ctrl-C
      # raises KeyboardInterrupt (hard abort, exit 130). Replaces the guard.
      rt.install_stop_handlers(runner.request_stop)
    else:
      # No graceful stop without ContactOnPolicyRunner: the guard stays and a
      # signal interrupts training immediately (recorded as "stopped").
      guard.phase = "training (runner without graceful stop)"
    try:
      runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
      print("[contact-train] finished.")
    except TrainingStopped as e:
      outcome.stop_reason = str(e) or "stop requested"
      print(f"[contact-train] stopped cleanly on {e}; last checkpoint iteration {runner._last_saved_it}.")
    outcome.code = 0
  except StartupStop as e:
    outcome.code = 0
    outcome.stop_reason = str(e)
    where = "before training started (no checkpoint written)" if e.phase == "startup" else f"during {e.phase}"
    print(f"[contact-train] stopped on {e.signal_name} {where}.", flush=True)
  except KeyboardInterrupt:
    print("[contact-train] aborted (second Ctrl-C). The last complete checkpoint is intact.")
    outcome.code = 130
    outcome.error = "KeyboardInterrupt (hard abort)"
  except BaseException as e:
    # Record the failure, then re-raise so the traceback is shown and the
    # process exits non-zero (SystemExit keeps its own code).
    outcome.code = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) and e.code != 0 else 1
    outcome.error = f"{type(e).__name__}: {e}"
    if run_dir is not None:
      try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs" / "error.txt").write_text(traceback.format_exc())
      except Exception:
        pass
    raise
  finally:
    with stop_signals_ignored("finalising run_info.json / closing the env"):
      finalize_run_info(run_dir, outcome, getattr(runner, "_last_saved_it", None))
      close_quietly(closable)
  return outcome.code


def parse_args(argv: list[str] | None = None):
  import tyro

  import contact_rl
  contact_rl.require_tasks()  # registers the contact tasks (explicit error if broken)
  import mjlab
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  cls = _config_cls()
  argv = sys.argv[1:] if argv is None else argv
  task, rest = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=argv, add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  default = cls(env=load_env_cfg(task), agent=load_rl_cfg(task))
  args = tyro.cli(cls, args=rest, default=default, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
  return task, args


def main() -> None:
  task, args = parse_args()
  if is_multi_gpu(args.gpu_ids):
    from mjlab.scripts.train import TrainConfig, launch_training

    defaults = type(args)(env=args.env, agent=args.agent)
    ignored = ignored_contact_flags(args, defaults)
    print("[WARN] multi-GPU: delegating to mjlab's launcher (logs/rsl_rl layout, no contact-train run management).")
    if ignored:
      print(f"[WARN] multi-GPU: ignoring contact-train-only flags: {', '.join('--' + n.replace('_', '-') for n in ignored)}")
    launch_training(task_id=task, args=base_config(args, TrainConfig))
    return
  sys.exit(run_contact_train(task, args))


if __name__ == "__main__":
  main()
