# CLAUDE.md: mental map of this repository

Read this first. It is the condensed model of the code that an assistant (or a
new contributor) needs before touching anything. Details live in `README.md`,
`docs/AUDIT.md`, `docs/VPS_DEPLOYMENT.md`, `scripts/vps/README.md`.

## 1. What this is

Implementation of *Learning to Act Through Contact* (Omar & Khadiv, L4DC 2026,
arXiv:2510.03599v2, full text in `paper.txt`) on top of **mjlab**
(Isaac-Lab-style manager-based RL on **MuJoCo-Warp**, GPU-parallel physics) +
**rsl-rl-lib 5.0.1** PPO. Robot: **Unitree Go2**, flat ground, 5 gaits
(trot, pace, bound, jump, crawl). The policy is conditioned on **contact goals**
(where/when each foot must touch or leave the ground), not on velocity commands.

Branch `audit/paper-alignment` = corrected implementation. `main` @ `23ad625` =
pre-audit baseline (known bugs, see AUDIT.md table section 2).

## 2. Directory map

```
pyproject.toml            uv workspace root (contact-rl); mjlab vendored as member .mjlab_ref
uv.lock                   pinned graph, Linux x86_64 only. COMMIT IT, never ignore it
.mjlab_ref/               vendored mjlab (2 local fixes: scipy dep, wp.get_cuda_driver_version)
paper.txt                 the paper (source of truth for faithfulness questions)
src/assets/               robot assets (Go2 MJCF / meshes)
src/contact_rl/
  __init__.py             sets MUJOCO_GL (EGL headless) BEFORE mujoco import; require_tasks()
  play_viewer.py          ContactPlayViewer (Viser): contact control panel, checkpoint picker, GRU reset
  robots/go2.py           get_go2_robot_cfg(), GO2_ACTION_SCALE (actuators, joints)
  tasks/contact/
    contact_env_cfg.py    make_contact_env_cfg(): obs, actions, command, events (DR), rewards, terminations, sim
    config/go2/
      __init__.py         registers the 2 task ids (register_mjlab_task)
      env_cfgs.py         Go2 specifics: PAPER_NUM_ENVS=8192, contact sensor, play variant (64 envs)
      rl_cfg.py           PPO + GRU cfg, entropy schedule constants, logger=tensorboard
      runner.py           ContactOnPolicyRunner: entropy decay, extra TB scalars, checkpoints, resume, graceful stop, ONNX
    mdp/
      planning.py         PURE TORCH: GAITS, GAIT_PATTERNS, GaitPlanner, proximity_kernel, phase_masks (unit tested)
      contact_command.py  ContactGoalCommand (mjlab CommandTerm) wraps GaitPlanner; builds the 33-dim goal obs; Fig.6 metrics
      command_override.py UserCommand / apply_to_command_term: steer a trained policy (viewer / scripts)
      rewards.py          reach / hold / detach / goal_discovery (+ mjlab penalties re-exported)
      observations.py     feet_to_goal_distance, foot_contact_state
      events.py           check_foot_ordering (startup assertion)
  scripts/                CLI entry points (see section 5)
  utils/
    checkpoints.py        run dirs, selectors (best/latest/run:it), atomic save, index.json, retention
    runtime.py            device/EGL, run_info.json, signal handlers, console tee, git/hardware info
    onnx_compat.py        DropNoneOutputs: fixes rsl-rl 5.0.1 GRU export returning (a, h, None)
    onnx_export.py        export + metadata status reporting
    episode_metrics.py    Fig. 6 metric aggregation
    policy_state.py       reset_recurrent_state (zero GRU hidden state on done)
    video.py              StreamingVideoRecorder (EGL, ffmpeg)
    train_stats.py        explained_variance
scripts/patches/patch_mujoco_warp_sensor.py   REQUIRED after every uv sync (patches .venv, SHA-gated)
scripts/vps/              setup / train_tmux / watch_tmux / eval / play / tensorboard / stop / tunnel + systemd units
tests/                    pytest; many run without GPU (planning, checkpoints, CLI, VPS scripts, regressions)
docs/                     AUDIT.md (paper vs code), VPS_DEPLOYMENT.md, VPS_READINESS_STATUS.md
```

## 3. Task ids

| id | meaning | run dir |
| --- | --- | --- |
| `Mjlab-Contact-Flat-Unitree-Go2` | paper-faithful (default) | `runs/go2_contact/` |
| `Mjlab-Contact-Flat-Unitree-Go2-Improved` | EXPERIMENTAL: + IMU in actor, obs normalisation | `runs/go2_contact_improved/` |

Each id has a train cfg (8192 envs, pushes, obs noise, 20 s episodes) and a
**play cfg** (`play=True`: **64 envs**, no pushes, no obs noise, ~infinite
episodes, debug vis of footholds on). `contact-play` / `contact-eval` use the
play cfg, so a viewer shows 64 robots: that is NOT training.

## 4. Data flow (one env step, 50 Hz control, sim dt 0.005, decimation 4)

```
GaitPlanner (per env, params sampled ONCE at env construction:
  gait, stride U(0,0.3) front/hind, stance width U(0.1,0.3), leg offsets U(-0.15,0.15),
  heading U[-pi,pi] OR yaw rate U[-pi,pi] (50/50), S ~ U[0.34,0.36] s)
   -> ContactGoalCommand: p1,p2 (4 feet x 2 switches x xyz, yaw-aligned base frame),
      I1,I2 (4x2), s  => 24 + 8 + 1 = 33 dims
   -> actor obs = joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) + contact_goal(33)
                  + feet_to_goal  [+ base_ang_vel, projected_gravity if -Improved]
   -> GRU(256) + MLP 512-256-128 -> 12 joint position targets (scale GO2_ACTION_SCALE, default offset)
critic obs = actor obs + base_lin_vel, base_ang_vel, projected_gravity, foot_contact (privileged)
rewards (weights are per-second, mjlab multiplies by step_dt=0.02):
  reach 1.0, hold 1.0 (alpha_hold=1), detach 0.5, goal_discovery 25.0,
  base_ang_vel -0.05, joint_vel -1e-3, joint_acc -2.5e-7, torques -2e-4,
  joint_deviation -0.05, action_rate -0.01, dof_pos_limits -1.0
kernel exp(-d/sigma^2), d = unsquared L2, sigma^2 = 0.1, delta = 0.18 s
phases: reach I1=0 & s<=delta | hold I1=1 | detach I1=0 & s>delta
terminations: time_out (20 s), fell_over (tilt > 80 deg)
```

Gait patterns (rows = switches, cols = FL FR RL RR, 1 = stance): trot 2
switches, pace 2, bound 2, jump 2 (full flight), crawl 4. Every foot swings
exactly once per period (validated at import).

Speed <-> stride: `speed = stride / (P * S)`, S ~= 0.35 s. Max trained speed:
~0.43 m/s for P=2 gaits, ~0.21 m/s for crawl. Higher = out of distribution
(applied, warned, not clamped).

## 5. CLIs (`[project.scripts]`)

| cmd | module | notes |
| --- | --- | --- |
| `contact-train <task>` | scripts/train.py | run management, resume into NEW run dir, graceful stop, run_info.json |
| `contact-play <task>` | scripts/play.py | Viser on 127.0.0.1:8080 (SSH tunnel), `--num-envs`, `--checkpoint best` |
| `contact-eval` | scripts/evaluate.py | `--mode fig6` (default, 1000 envs x 5 gaits x 9 durations) or `--mode episodes` |
| `contact-watch` | scripts/watch.py | evaluates each new checkpoint once (subprocess, flock) |
| `contact-doctor` | scripts/doctor.py | PASS/WARN/FAIL env diagnostics |
| `contact-bench` | scripts/benchmark.py | env throughput vs num_envs (pick the GPU knee) |

All mjlab flags pass through (`--env.*`, `--agent.*`, `--video`, `--gpu-ids "[0]"`).

## 6. PPO / training constants (rl_cfg.py)

24 steps/env/iteration, 5 epochs x 4 mini-batches, lr 1e-3 adaptive (KL 0.01),
gamma 0.99, lambda 0.95, clip 0.2, max_grad_norm 1.0, init_std 1.0, seed 42,
**max_iterations 10000**, save_interval 50, entropy 0.01 -> 0.001 linear over
5000 iterations (applied by wrapping `alg.update`).

Samples per iteration = 24 x num_envs (8192 -> 196,608). Full run = ~1.97 B
env steps. Wall time = 10000 x per-iteration time (printed as `[iter N] ... t=`).

## 7. Run / checkpoint layout

```
runs/<experiment>/<timestamp>[_name]/
  run_info.json              status starting|running|finished|stopped|aborted|failed, exit_code
  checkpoints/model_<it>.pt  full state (actor, critic, optimizer, iter, infos, contact_rl meta)
  checkpoints/latest.pt      SYMLINK (copy fallback) -> use cp -L when copying
  checkpoints/best.pt        copy of best by mean training episode reward
  checkpoints/index.json
  exported/policy.onnx       actor only, GRU hidden state as extra input/output
  config/{env.yaml,agent.yaml,cli.json}  logs/{train.log,error.txt}  metrics/  videos/
```
Retention: keep last 5, multiples of 1000, best. Selectors: `best`, `latest`,
`<run_dir>`, `<run_dir>:<it>|best|latest`, `file.pt`. `runs/`, `logs/`, `*.pt`,
`*.onnx`, `*.mp4` are git-ignored (use `git add -f` or a Release to publish weights).

## 8. Gotchas (do not regress these)

* `UV_NO_SYNC=1` always after the one explicit `uv sync --extra cu128`, or the
  CUDA torch gets silently replaced.
* Re-run `scripts/patches/patch_mujoco_warp_sensor.py` after any sync that
  reinstalls mujoco-warp.
* `mujoco-warp==3.5.0.2` from PyPI, not git (see pyproject notes). Do not restore the git rev.
* `--gpu-ids` takes a quoted Python list: `"[0, 1]"`, never `0 1`. >1 GPU hands
  off to mjlab's launcher (legacy `logs/rsl_rl`, no run management).
* Reset anchoring must read root pose from **qpos** (derived xpos is stale at reset).
* Foot order FL, FR, RL, RR everywhere (goals, sites, sensor); checked at startup.
* Goal discovery gives a bonus only; it must NOT shorten the command window
  (`discovery_advances_goals=False`).
* Command params are sampled once per env: fewer envs = fewer distinct
  gait/stride/heading combinations seen in training.
* GRU hidden state must be zeroed on episode end (viewer and deployment).
* ONNX export must go through `onnx_compat.export_policy_onnx`.
* W&B blocks on headless servers; logger default is TensorBoard.

## 9. Verification status

Authoring environment had no GPU: unit tests pass, but training performance,
throughput and final gait quality are NOT runtime verified by the authors.
Treat numbers (time per iteration, reward curves) as things to measure.

## 10. Not in this repo

No sim-to-real / Unitree SDK deployment code. `exported/policy.onnx` plus a
re-implementation of the observation pipeline (GaitPlanner + joint state + foot
forward kinematics) would be needed on the real robot.
