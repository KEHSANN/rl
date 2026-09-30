"""``contact-play``: interactive policy viewer for a headless GPU VPS.

The Viser server binds to **127.0.0.1:8080** by default (mjlab's own play
binds 0.0.0.0 -- unauthenticated). Reach it through an SSH tunnel::

  ssh -N -L 8080:127.0.0.1:8080 user@vps      # then open http://localhost:8080

``--host 0.0.0.0`` is refused unless ``--allow-public True`` is also given.

Examples::

  uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best
  uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint runs/go2_contact/<run>:1500
  uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint-file model.pt   # mjlab flag, still works

All mjlab ``play`` flags are accepted (``--agent``, ``--checkpoint-file``,
``--wandb-run-path``, ``--num-envs``, ``--device``, ``--video``, ``--viewer``,
``--no-terminations`` ...). See :mod:`contact_rl.play_viewer` for the controls.

The env (and its video recorder / EGL context) is closed in a ``finally``
block, so a failing checkpoint load, a busy port or a viewer crash never
leaves it open.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from contact_rl.scripts import taskless_help
from contact_rl.utils.runtime import configure_headless_rendering

configure_headless_rendering()  # before mujoco is imported

# Shown by ``contact-play --help`` (no task id yet, so the per-task option list
# cannot be built).
_USAGE = """\
contact-play-only options (all mjlab play flags also work: --agent,
--checkpoint-file, --wandb-run-path, --num-envs, --device, --video, --viewer,
--no-terminations ...):
  --checkpoint SPEC    checkpoint file | run dir | run_dir:<it>|best|latest | latest | best
  --run-root DIR       default "runs"
  --host HOST          viewer bind address                (default: 127.0.0.1)
  --port PORT          viewer port                        (default: 8080)
  --allow-public BOOL  required to bind a non-loopback host (no authentication!)

examples:
  contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best
  contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint runs/go2_contact/<run>:1500
  ssh -N -L 8080:127.0.0.1:8080 <user>@<vps>    # then open http://localhost:8080"""


def _config_cls():
  from mjlab.scripts.play import PlayConfig

  @dataclass(frozen=True)
  class ContactPlayConfig(PlayConfig):
    checkpoint: str | None = None
    """Selector: file | run dir | run_dir:best|latest|<it> | latest | best."""
    run_root: str = "runs"
    host: str = "127.0.0.1"
    port: int = 8080
    allow_public: bool = False
    """Required to bind a non-loopback host (the viewer has no authentication)."""

  return ContactPlayConfig


def check_port(host: str, port: int) -> None:
  with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
      s.bind((host, port))
    except OSError as e:
      raise SystemExit(f"Cannot bind {host}:{port} ({e}). Is another viewer running? Use --port.") from e


class _RecurrentResetEnv:
  """Env proxy that zeroes the policy's recurrent state for envs whose episode
  ended in ``step`` (used for mjlab's native viewer, which never does this;
  :class:`~contact_rl.play_viewer.ContactPlayViewer` handles it itself)."""

  def __init__(self, env, get_policy):
    self._env = env
    self._get_policy = get_policy

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def unwrapped(self):
    return self._env.unwrapped

  def step(self, actions):
    from contact_rl.utils.policy_state import reset_recurrent_state

    out = self._env.step(actions)
    reset_recurrent_state(self._get_policy(), out[2])
    return out


def run(task_id: str, cfg) -> None:
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import get_wandb_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends

  from contact_rl.utils import checkpoints as ck
  from contact_rl.utils import runtime as rt
  from contact_rl.utils.policy_state import reset_recurrent_state
  from contact_rl.utils.video import StreamingVideoRecorder

  rt.assert_private_bind(cfg.host, cfg.allow_public)
  if not rt.is_loopback(cfg.host):
    print("!" * 70, file=sys.stderr)
    print(f"[play] WARNING: binding the viewer to {cfg.host} (--allow-public). It has NO", file=sys.stderr)
    print("[play] authentication: anyone who can reach this port can control the", file=sys.stderr)
    print("[play] simulation. Prefer 127.0.0.1 + an SSH tunnel.", file=sys.stderr)
    print("!" * 70, file=sys.stderr, flush=True)
  configure_torch_backends()
  device = rt.resolve_device(cfg.device)
  rt.set_egl_device_for(device)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.no_terminations:
    env_cfg.terminations = {}
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  trained = cfg.agent == "trained"
  ckpt: Path | None = None
  if trained:
    if cfg.checkpoint or cfg.checkpoint_file:
      ckpt = ck.resolve_checkpoint(cfg.checkpoint or cfg.checkpoint_file, cfg.run_root)
    elif cfg.wandb_run_path:
      ckpt, _ = get_wandb_checkpoint_path(
        (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve(), Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
    else:
      raise SystemExit("Pass --checkpoint (e.g. 'best'), --checkpoint-file or --wandb-run-path.")
    print(f"[play] checkpoint: {ckpt}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if (cfg.video and trained) else None)
  closable = env
  try:
    if cfg.video and trained and ckpt is not None:
      vdir = ck.run_dir_of_checkpoint(ckpt) / "videos" / "play"
      env = StreamingVideoRecorder(env, path_fn=lambda s: vdir / f"{ckpt.stem}_step{s}.mp4",
                                   step_trigger=lambda s: s == 0, video_length=cfg.video_length)
      closable = env
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    closable = venv

    runner = None
    if trained:
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(venv, asdict(agent_cfg), device=device)
      runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
      policy = runner.get_inference_policy(device=device)
    else:
      shape = venv.unwrapped.action_space.shape
      zero = cfg.agent == "zero"

      def policy(obs):  # noqa: ARG001
        return torch.zeros(shape, device=device) if zero else 2 * torch.rand(shape, device=device) - 1

    reset_recurrent_state(policy)

    def load_policy(path: Path):
      assert runner is not None
      ck.validate_checkpoint(path)
      runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
      p = runner.get_inference_policy(device=device)
      reset_recurrent_state(p)
      return p

    def list_ckpts() -> list[Path]:
      if ckpt is None:
        return []
      run_dir = ck.run_dir_of_checkpoint(ckpt)
      out = [p for _, p in ck.list_checkpoints(run_dir)]
      best = ck.checkpoint_dir(run_dir) / ck.BEST_NAME
      return ([best] if best.exists() else []) + out[::-1]

    viewer = cfg.viewer
    if viewer == "auto":
      viewer = "native" if rt.has_display() else "viser"
    if viewer == "native":
      from mjlab.viewer import NativeMujocoViewer

      NativeMujocoViewer(_RecurrentResetEnv(venv, lambda: policy), policy).run()
    else:
      import viser

      from contact_rl.play_viewer import ContactPlayViewer

      check_port(cfg.host, cfg.port)
      server = viser.ViserServer(host=cfg.host, port=cfg.port, label="contact-play")
      where = f"http://{cfg.host}:{cfg.port}"
      print("=" * 70)
      print(f"[play] Viser listening on {where}" + ("" if rt.is_loopback(cfg.host) else "  (PUBLIC, no auth!)"))
      print(f"[play] from your laptop:  ssh -N -L {cfg.port}:127.0.0.1:{cfg.port} <user>@<vps>")
      print(f"[play] then open          http://localhost:{cfg.port}")
      print("=" * 70, flush=True)
      try:
        ContactPlayViewer(venv, policy, viser_server=server, load_policy=load_policy if trained else None,
                          checkpoints=list_ckpts, current_checkpoint=ckpt).run()
      finally:
        server.stop()
  finally:
    try:
      closable.close()
    except Exception as e:  # noqa: BLE001  (never mask the original error)
      print(f"[WARN] env close failed: {e}", file=sys.stderr)


def parse_args(argv: list[str] | None = None):
  import tyro

  import contact_rl
  contact_rl.require_tasks()  # registers the contact tasks (explicit error if broken)
  import mjlab
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  cls = _config_cls()
  argv = sys.argv[1:] if argv is None else argv
  tasks = list_tasks()
  taskless_help(sys.argv[0], tasks, argv, usage=_USAGE)
  task, rest = tyro.cli(
    tyro.extras.literal_type_from_choices(tasks),
    args=argv, add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(cls, args=rest, default=cls(), prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
  return task, args


def main() -> None:
  task, args = parse_args()
  try:
    run(task, args)
  except KeyboardInterrupt:
    print("\n[play] bye.")


if __name__ == "__main__":
  main()
