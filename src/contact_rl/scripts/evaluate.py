"""``contact-eval``: reproduce the paper's Fig. 6 evaluation protocol.

For every gait and command duration ``S`` the trained policy is rolled out on
``num_envs`` parallel episodes of ``episode_s`` seconds (paper: 1000 episodes x
15 s, S in [0.2, 0.9] s, training S ~ U[0.34, 0.36] s) and two metrics are
reported:

* **contact location error** -- L2 distance between the actual foot position and
  the planned contact location while the foot is (commanded and actually) in
  contact, in cm;
* **contact plan deviation** -- Hamming distance between the desired contact
  plan and the actual contact state (feet in disagreement per control step).

Example::

    uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 \
        --checkpoint-file logs/rsl_rl/go2_contact/<run>/model_9999.pt
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class EvalConfig:
  task: str = "Mjlab-Contact-Flat-Unitree-Go2"
  checkpoint_file: str = ""
  gaits: tuple[str, ...] = ("trot", "pace", "bound", "jump", "crawl")
  durations: tuple[float, ...] = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  num_envs: int = 1000
  episode_s: float = 15.0
  seed: int = 0
  device: str | None = None
  out_csv: str = "eval_fig6.csv"


def _evaluate_one(cfg: EvalConfig, gait: str, duration: float, device: str) -> dict:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  from contact_rl.tasks.contact.mdp import ContactGoalCommand

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.episode_length_s = cfg.episode_s
  cmd = env_cfg.commands["contact"]
  cmd.gaits = (gait,)
  cmd.resampling_time_range = (duration, duration)
  cmd.debug_vis = False
  # delta only shapes rewards; it must stay below S for the cfg check.
  cmd.delta = min(cmd.delta, 0.5 * duration)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(venv, asdict(agent_cfg), device=device)
  runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = env.command_manager.get_term("contact")
  assert isinstance(term, ContactGoalCommand)
  n = env.num_envs
  loc_sum = torch.zeros(n, device=device)
  loc_cnt = torch.zeros(n, device=device)
  ham_sum = torch.zeros(n, device=device)
  steps = torch.zeros(n, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  alive = torch.ones(n, dtype=torch.bool, device=device)

  num_steps = env.max_episode_length - 1  # stay clear of the time-out reset
  obs = venv.get_observations()
  with torch.inference_mode():
    for _ in range(num_steps):
      # Metrics on the goal the policy acts on, evaluated after the step.
      i_con = term.current_contact_goal.clone()
      goal = term.goal_pos_w[:, :, 0, :].clone()
      obs, _, dones, _ = venv.step(policy(obs))
      i_act = term.actual_contact()
      dist = torch.norm(term.foot_pos_w() - goal, dim=-1)
      m = alive.float()
      in_c = ((i_con > 0.5) & (i_act > 0.5)).float()
      loc_sum += m * (dist * in_c).sum(dim=1)
      loc_cnt += m * in_c.sum(dim=1)
      ham_sum += m * (i_con - i_act).abs().sum(dim=1)
      steps += m
      d = dones.bool()
      fell |= d & alive
      alive &= ~d  # count each env's first episode only
      if hasattr(policy, "reset"):
        policy.reset(dones)
  env.close()

  loc_cm = 100.0 * (loc_sum.sum() / loc_cnt.sum().clamp(min=1)).item()
  ham = (ham_sum.sum() / steps.sum().clamp(min=1)).item()
  return {
    "gait": gait,
    "command_duration_s": duration,
    "contact_location_error_cm": loc_cm,
    "contact_plan_hamming": ham,
    "fall_rate": fell.float().mean().item(),
    "episodes": n,
  }


def main() -> None:
  import contact_rl  # noqa: F401  (registers the tasks)
  from mjlab.utils.torch import configure_torch_backends

  cfg = tyro.cli(EvalConfig)
  if not cfg.checkpoint_file or not Path(cfg.checkpoint_file).exists():
    raise FileNotFoundError(f"Checkpoint not found: '{cfg.checkpoint_file}'")
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  rows = []
  for gait in cfg.gaits:
    for s in cfg.durations:
      row = _evaluate_one(cfg, gait, s, device)
      rows.append(row)
      print(
        f"{gait:6s} S={s:.2f}s  loc_err={row['contact_location_error_cm']:.2f} cm  "
        f"hamming={row['contact_plan_hamming']:.3f}  fall={row['fall_rate']:.3f}"
      )
  with open(cfg.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f"[INFO] wrote {cfg.out_csv}")


if __name__ == "__main__":
  main()
