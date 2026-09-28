"""``contact-train``: production training entry point.

Registers the contact tasks, then trains with mjlab's env + the project
runner, adding run management on top of mjlab's ``train`` CLI. Every mjlab
``TrainConfig`` flag still works (``--env.*``, ``--agent.*``, ``--video``,
``--video-length``, ``--video-interval``, ``--gpu-ids`` ...), plus::

  --resume-from SPEC   checkpoint file | run dir | run_dir:<it>|best|latest |
                       latest | best  (full state; continues at iter+1 in a
                       NEW run dir, the source is never modified)
  --device DEV         cpu | cuda | cuda:<i> (default: first of --gpu-ids)
  --run-root DIR       default "runs" -> runs/<experiment>/<timestamp>[_name]
  --keep-last N        keep newest N checkpoints (<=0: keep all)
  --keep-every N       also keep multiples of N
  --keep-best BOOL     maintain checkpoints/best.pt (by mean episode reward)
  --export-onnx BOOL   write exported/policy.onnx at every save

Example::

  uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096

Multi-GPU (``--gpu-ids`` with more than one id) is delegated unchanged to
mjlab's torchrunx launcher (legacy logs/rsl_rl layout, no run management).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from contact_rl.utils.runtime import configure_headless_rendering

configure_headless_rendering()  # before anything can import mujoco


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


def _pick_device(cfg) -> str:
  from contact_rl.utils.runtime import resolve_device

  if cfg.device:
    return resolve_device(cfg.device)
  ids = cfg.gpu_ids
  if ids is None:
    return "cpu"
  if ids == "all":
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

  cfg.env.seed = cfg.agent.seed
  env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  nspe = int(cfg.agent.num_steps_per_env)
  if cfg.video:
    env = StreamingVideoRecorder(
      env,
      path_fn=lambda s: run_dir / "videos" / "train" / f"iteration_{s // nspe:06d}" / f"step_{s:09d}.mp4",
      step_trigger=lambda s: s % cfg.video_interval == 0,
      video_length=cfg.video_length,
      hud_fn=lambda s: [f"train it {s // nspe}", f"step {s}"],
    )
  venv = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

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
    task=task_id,
    experiment=cfg.agent.experiment_name,
    device=device,
    num_envs=int(cfg.env.scene.num_envs),
    max_iterations=int(cfg.agent.max_iterations),
    seed=int(cfg.agent.seed),
    git=rt.git_info(),
    versions=rt.distribution_versions(),
    hardware=gpu,
    mujoco_gl=__import__("os").environ.get("MUJOCO_GL"),
    resume=resume_meta,
    started=rt.now_iso(),
    retention={"keep_last": cfg.keep_last, "keep_every": cfg.keep_every, "keep_best": cfg.keep_best},
  )
  start = int(runner.current_learning_iteration)
  print("=" * 78)
  print(f"[contact-train] task={task_id}  device={device}  num_envs={cfg.env.scene.num_envs}")
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
    rt.install_stop_handlers(runner.request_stop)
  code = 0
  try:
    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    print("[contact-train] finished.")
  except TrainingStopped as e:
    print(f"[contact-train] stopped cleanly on {e}; last checkpoint iteration {runner._last_saved_it}.")
  except KeyboardInterrupt:
    print("[contact-train] aborted (second Ctrl-C). The last complete checkpoint is intact.")
    code = 130
  finally:
    rt.update_run_info(run_dir, finished=rt.now_iso(), exit_code=code)
    try:
      venv.close()
    except Exception as e:  # noqa: BLE001
      print(f"[WARN] env close failed: {e}")
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
  ids = args.gpu_ids
  if (ids == "all") or (isinstance(ids, list) and len(ids) > 1):
    from mjlab.scripts.train import launch_training

    print("[WARN] multi-GPU: delegating to mjlab's launcher (logs/rsl_rl layout, no contact-train run management).")
    launch_training(task_id=task, args=args)
    return
  sys.exit(run_contact_train(task, args))


if __name__ == "__main__":
  main()
