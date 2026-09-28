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

``--enable-nan-guard True`` turns on mjlab's simulation NaN guard
(``env.sim.nan_guard.enabled``), exactly like mjlab's own ``train`` CLI.

GPUs: ``--gpu-ids 0`` (default) / ``--gpu-ids 1`` / ``--gpu-ids all`` with one
visible GPU all train in this process with full run management. Only when
more than one GPU is actually selected (``--gpu-ids 0 1``, or ``all`` with
several visible GPUs) is training delegated to mjlab's torchrunx launcher
with a plain mjlab ``TrainConfig`` (legacy logs/rsl_rl layout, no run
management; the contact-train-only flags are ignored with a warning).

run_info.json: written as soon as the run dir exists (``status="starting"``),
updated to ``"running"`` once training starts, and always finalised with
``status`` (finished | stopped | aborted | failed), ``exit_code``, ``error``
(``null`` on success), ``finished`` and ``last_checkpoint_iteration``. A failed
run can never be recorded as a success; the traceback also goes to
``logs/error.txt``.

Exit codes: 0 = finished or stopped cleanly (SIGINT/SIGTERM), 130 = hard abort
(second Ctrl-C), 1 = any other error (the traceback is re-raised). The env is
closed even if env construction, the runner, the resume load or training
itself fails.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
  import torch

  import contact_rl
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_runner_cls
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  from contact_rl.tasks.contact.config.go2.runner import ContactOnPolicyRunner, TrainingStopped
  from contact_rl.utils import checkpoints as ck
  from contact_rl.utils import runtime as rt
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

  # Everything from here on runs under try/finally: a failing env
  # construction, runner construction, resume load, CUDA OOM or NaN still
  # closes whatever was built (EGL context, ffmpeg) and records the real
  # status / exit code / error in run_info.json.
  closable: Any = None
  runner = None
  code = 1
  error: str | None = None
  stop_reason: str | None = None
  try:
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
      # raises KeyboardInterrupt (hard abort, exit 130).
      rt.install_stop_handlers(runner.request_stop)
    try:
      runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
      print("[contact-train] finished.")
    except TrainingStopped as e:
      stop_reason = str(e) or "stop requested"
      print(f"[contact-train] stopped cleanly on {e}; last checkpoint iteration {runner._last_saved_it}.")
    code = 0
  except KeyboardInterrupt:
    print("[contact-train] aborted (second Ctrl-C). The last complete checkpoint is intact.")
    code = 130
    error = "KeyboardInterrupt (hard abort)"
  except BaseException as e:
    # Record the failure, then re-raise so the traceback is shown and the
    # process exits non-zero (SystemExit keeps its own code).
    code = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) and e.code != 0 else 1
    error = f"{type(e).__name__}: {e}"
    try:
      (run_dir / "logs").mkdir(parents=True, exist_ok=True)
      (run_dir / "logs" / "error.txt").write_text(traceback.format_exc())
    except Exception:
      pass
    raise
  finally:
    try:
      rt.update_run_info(
        run_dir,
        finished=rt.now_iso(),
        exit_code=code,
        status=run_status(code, stop_reason),
        error=error,
        stop_reason=stop_reason,
        last_checkpoint_iteration=getattr(runner, "_last_saved_it", None),
      )
    except Exception as e:  # noqa: BLE001 - never mask the real error
      print(f"[WARN] could not finalise run_info.json: {e}")
    finally:
      close_quietly(closable)
  return code


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
