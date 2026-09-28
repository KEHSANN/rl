"""``contact-eval``: per-episode evaluation of a trained contact policy.

Modes
-----
``--mode fig6`` (default, backwards compatible): the paper's Fig. 6 protocol.
For every gait and command duration ``S`` the policy is rolled out on
``num_envs`` parallel episodes of ``episode_s`` seconds (paper: 1000 x 15 s,
S in [0.2, 0.9] s) and the legacy table ``--out-csv`` is written as before.

``--mode episodes``: one rollout with the training command distribution (used
by ``contact-watch``).

Both modes write, to ``--out-dir`` (default ``<run>/metrics/iteration_<it>``)::

  metrics.csv    one row per episode (first episode of every env)
  summary.json   rates + mean/std per metric, overall and per (gait, S)

Per-episode fields: reward, length, termination (fall | timeout |
other_termination | running), fell, success, contact_plan_hamming,
contact_location_error_cm, foot_tracking_error_cm, lin_vel_error_mps
(base xy velocity vs. the planner's reference velocity), yaw_rate_error_rps,
goals_discovered. The terminal (auto-reset) transition is excluded from state
metrics and time-outs are never counted as falls (see
:mod:`contact_rl.utils.episode_metrics`).

Videos (``--video True``): env 0 of the first rollout, streamed to
``--video-dir`` (default ``<run>/videos/iteration_<it>``) with a text HUD.

Examples::

  uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 --checkpoint runs/go2_contact/<run>:best \
      --mode episodes --num-envs 256 --video True
  uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 --checkpoint-file <model.pt>   # Fig. 6
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from contact_rl.utils.runtime import configure_headless_rendering

configure_headless_rendering()  # before mujoco is imported


@dataclass
class EvalConfig:
  task: str = "Mjlab-Contact-Flat-Unitree-Go2"
  checkpoint_file: str = ""
  """Legacy: explicit checkpoint path."""
  checkpoint: str | None = None
  """Selector: file | run dir | run_dir:best|latest|<it> | latest | best."""
  run_root: str = "runs"
  mode: Literal["fig6", "episodes"] = "fig6"
  gaits: tuple[str, ...] = ("trot", "pace", "bound", "jump", "crawl")
  durations: tuple[float, ...] = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  num_envs: int = 1000
  episode_s: float = 15.0
  seed: int = 0
  device: str | None = None
  out_csv: str = "eval_fig6.csv"
  """Legacy Fig. 6 table (fig6 mode only)."""
  out_dir: str | None = None
  video: bool = False
  video_dir: str | None = None
  video_steps: int = 500
  hud: bool = True
  tensorboard: bool = False
  """Also write evaluation/* scalars to <run>/metrics/tb (used by contact-watch)."""
  run_dir: str | None = None
  """Run directory the results belong to. Default: derived from the checkpoint
  path (``<run>/checkpoints/model_<it>.pt``). contact-watch evaluates a
  hard-linked snapshot of the checkpoint (so retention cannot delete it
  mid-evaluation) and passes the real run directory here."""


def _checkpoint(cfg: EvalConfig) -> Path:
  from contact_rl.utils.checkpoints import resolve_checkpoint

  spec = cfg.checkpoint or cfg.checkpoint_file
  if not spec:
    raise SystemExit("Pass --checkpoint <selector> or --checkpoint-file <model.pt>.")
  return resolve_checkpoint(spec, cfg.run_root)


def rollout(cfg: EvalConfig, ckpt: Path, device: str, gait: str | None, duration: float | None,
            video_path: Path | None, label: str | None = None) -> list[dict]:
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  from contact_rl.tasks.contact.mdp import ContactGoalCommand
  from contact_rl.tasks.contact.mdp.command_override import describe_command
  from contact_rl.tasks.contact.mdp.planning import unit
  from contact_rl.utils.episode_metrics import EpisodeAccumulator, classify
  from contact_rl.utils.policy_state import reset_recurrent_state
  from contact_rl.utils.video import Hud, StreamingVideoWriter, VideoEncodeError

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.episode_length_s = cfg.episode_s  # finite horizon -> real time-outs
  cmd = env_cfg.commands["contact"]
  if gait is not None:
    cmd.gaits = (gait,)
  elif cfg.gaits:
    cmd.gaits = tuple(cfg.gaits)
  if duration is not None:
    cmd.resampling_time_range = (duration, duration)
    cmd.delta = min(cmd.delta, 0.5 * duration)  # delta only shapes rewards; must stay < S
  cmd.debug_vis = video_path is not None

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if video_path else None)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(venv, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  reset_recurrent_state(policy)

  term = env.command_manager.get_term("contact")
  assert isinstance(term, ContactGoalCommand)
  robot = term.robot
  pl = term.planner
  tm = env.termination_manager
  n = env.num_envs
  zeros_b = torch.zeros(n, dtype=torch.bool, device=device)
  acc = EpisodeAccumulator(torch.zeros(n, device=device))
  writer = StreamingVideoWriter(video_path, fps=1.0 / env.step_dt) if video_path else None
  hud = Hud(cfg.hud)
  video_err = None

  obs = venv.get_observations()
  max_steps = int(env.max_episode_length) + 1
  t0 = time.time()
  with torch.inference_mode():
    for k in range(max_steps):
      i_con = term.current_contact_goal.clone()
      goal = term.goal_pos_w[:, :, 0, :].clone()
      speed = pl.stride.mean(dim=1) / (pl.period(torch.arange(n, device=device)) * pl.switch_dt)
      v_cmd = speed.unsqueeze(-1) * unit(pl.ref_yaw + pl.heading_off)
      w_cmd = pl.yaw_rate.clone()

      obs, rew, dones, _ = venv.step(policy(obs))
      fell = tm.get_term("fell_over") if "fell_over" in tm.active_terms else zeros_b
      reason = classify(tm.time_outs, tm.terminated, fell)

      i_act = term.actual_contact()
      dist = torch.norm(term.foot_pos_w() - goal, dim=-1)
      in_c = ((i_con > 0.5) & (i_act > 0.5)).float()
      acc.step(
        rew,
        reason,
        hamming=(i_con - i_act).abs().sum(dim=1),
        loc_err_sum=(dist * in_c).sum(dim=1),
        loc_cnt=in_c.sum(dim=1),
        tracking=dist.mean(dim=1),
        lin_vel_err=torch.norm(robot.data.root_link_lin_vel_w[:, :2] - v_cmd, dim=-1),
        yaw_rate_err=(robot.data.root_link_ang_vel_w[:, 2] - w_cmd).abs(),
        discovered=term.just_discovered.float(),
      )
      reset_recurrent_state(policy, dones)  # no GRU state leaks into the next episode

      if writer is not None and k < cfg.video_steps:
        try:
          frame = env.render()
          if frame is not None:
            d = describe_command(pl, 0)
            lines = [
              label or f"{ckpt.parent.parent.name}/{ckpt.name}",
              f"gait {d['gait']}  v_cmd {d['speed_mps']:.2f} m/s  hdg {d['heading_offset_deg']:.0f} deg",
              f"yaw_rate {d['yaw_rate_rps']:.2f} rad/s  t={k * env.step_dt:5.2f}s",
            ]
            if bool(dones[0]):
              lines.append("EPISODE END: " + ["", "timeout", "FALL", "terminated"][int(reason[0])])
            writer.add(hud.draw(frame, lines))
        except VideoEncodeError as e:
          video_err = str(e)
          print(f"[ERROR] video disabled: {e}")
          writer.abort()
          writer = None
      if acc.all_done() and (writer is None or k >= cfg.video_steps):
        break
  if writer is not None:
    try:
      writer.close()
    except VideoEncodeError as e:
      video_err = str(e)
      print(f"[ERROR] {e}")
  env.close()
  rows = acc.rows(env.step_dt, extra={
    "gait": gait or "all",
    "command_duration_s": duration if duration is not None else 0.5 * sum(env_cfg.commands["contact"].resampling_time_range),
  })
  print(f"[eval] {len(rows)} episodes in {time.time() - t0:.1f}s" + (f" (video error: {video_err})" if video_err else ""))
  return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    raise RuntimeError(f"no rows to write to {path} (evaluation produced no episodes)")
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".csv.tmp")
  with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  tmp.replace(path)


def evaluate(cfg: EvalConfig) -> dict:
  from contact_rl.utils import checkpoints as ck
  from contact_rl.utils import runtime as rt
  from contact_rl.utils.episode_metrics import summarize

  ckpt = _checkpoint(cfg)
  device = rt.resolve_device(cfg.device)
  rt.set_egl_device_for(device)
  run_dir = Path(cfg.run_dir) if cfg.run_dir else ck.run_dir_of_checkpoint(ckpt)
  it = ck.iteration_of(ckpt)
  if it is None:  # best.pt / latest.pt -> read the iteration from the file
    import torch

    it = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("iter", -1))
  tag = f"iteration_{it:06d}"
  out_dir = Path(cfg.out_dir) if cfg.out_dir else run_dir / "metrics" / tag
  video_dir = Path(cfg.video_dir) if cfg.video_dir else run_dir / "videos" / tag
  print(f"[eval] checkpoint {ckpt} (iteration {it}) on {device}; results -> {out_dir}")

  rows: list[dict] = []
  groups: dict[str, dict] = {}
  if cfg.mode == "episodes":
    vp = video_dir / "eval_env0.mp4" if cfg.video else None
    rows = rollout(cfg, ckpt, device, None, None, vp, label=f"{run_dir.name} it {it}")
  else:
    first = True
    legacy = []
    for gait in cfg.gaits:
      for s in cfg.durations:
        vp = video_dir / f"{gait}_S{s:.2f}.mp4" if (cfg.video and first) else None
        first = False
        r = rollout(cfg, ckpt, device, gait, s, vp, label=f"{run_dir.name} it {it}")
        rows += r
        sm = summarize(r)
        groups[f"{gait}/S={s:.2f}"] = sm
        # Pooled (step-weighted over all episodes), exactly the aggregation the
        # Fig. 6 table used before; per-episode means/stds are in summary.json.
        legacy.append({
          "gait": gait,
          "command_duration_s": s,
          "contact_location_error_cm": sm["contact_location_error_cm_pooled"],
          "contact_plan_hamming": sm["contact_plan_hamming_pooled"],
          "fall_rate": sm["fall_rate"],
          "timeout_rate": sm["timeout_rate"],
          "episodes": sm["episodes"],
        })
        print(f"{gait:6s} S={s:.2f}s  loc_err={legacy[-1]['contact_location_error_cm']:.2f} cm  "
              f"hamming={legacy[-1]['contact_plan_hamming']:.3f}  fall={sm['fall_rate']:.3f}  "
              f"timeout={sm['timeout_rate']:.3f}")
    _write_csv(Path(cfg.out_csv), legacy)
    print(f"[INFO] wrote {cfg.out_csv}")

  for r in rows:
    r["iteration"] = it
  _write_csv(out_dir / "metrics.csv", rows)
  summary = {
    "checkpoint": str(ckpt),
    "iteration": it,
    "task": cfg.task,
    "mode": cfg.mode,
    "device": device,
    "time": rt.now_iso(),
    "config": asdict(cfg),
    "overall": summarize(rows),
    "groups": groups,
    "videos": sorted(str(p) for p in video_dir.glob("*.mp4")) if cfg.video else [],
    "git": rt.git_info(),
  }
  rt.write_json(out_dir / "summary.json", summary)
  o = summary["overall"]
  print(f"[eval] fall={o['fall_rate']:.3f} timeout={o['timeout_rate']:.3f} success={o['success_rate']:.3f} "
        f"reward={o['reward_mean']:.2f} loc_err={o['contact_location_error_cm_mean']:.2f}cm "
        f"hamming={o['contact_plan_hamming_mean']:.3f} v_err={o['lin_vel_error_mps_mean']:.3f}")
  if cfg.tensorboard:
    _log_tensorboard(run_dir, it, o)
  _append_history(run_dir, it, ckpt, o)
  return summary


def _log_tensorboard(run_dir: Path, it: int, overall: dict) -> None:
  try:
    from torch.utils.tensorboard import SummaryWriter
  except Exception as e:  # noqa: BLE001
    print(f"[WARN] tensorboard unavailable: {e}")
    return
  w = SummaryWriter(log_dir=str(run_dir / "metrics" / "tb"))
  for k, v in overall.items():
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
      w.add_scalar(f"evaluation/{k}", v, it)
  w.flush()
  w.close()


def _append_history(run_dir: Path, it: int, ckpt: Path, overall: dict) -> None:
  """metrics/evaluations.csv: one row per evaluated checkpoint (for comparing
  checkpoints over time). An existing header is reused so rows stay aligned
  even if the summary gains fields in a later version."""
  path = run_dir / "metrics" / "evaluations.csv"
  row = {"iteration": it, "checkpoint": ckpt.name, **{k: v for k, v in overall.items() if not isinstance(v, dict)}}
  path.parent.mkdir(parents=True, exist_ok=True)
  fields = list(row.keys())
  new = not path.exists() or path.stat().st_size == 0
  if not new:
    with open(path, newline="") as f:
      header = next(csv.reader(f), None)
    if header:
      fields = header
  with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
      w.writeheader()
    w.writerow(row)


def main(argv: list[str] | None = None) -> None:
  import tyro

  import contact_rl
  contact_rl.require_tasks()  # registers the contact tasks (explicit error if broken)
  from mjlab.utils.torch import configure_torch_backends

  cfg = tyro.cli(EvalConfig, args=sys.argv[1:] if argv is None else argv, config=(tyro.conf.FlagConversionOff,))
  configure_torch_backends()
  evaluate(cfg)


if __name__ == "__main__":
  main()
