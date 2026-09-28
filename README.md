# Learning to Act Through Contact — mjlab implementation

An implementation of

> **Learning to Act Through Contact: A Unified View of Multi-Task Robot Learning**
> Shafeef Omar, Majid Khadiv (TUM), L4DC 2026 — arXiv:2510.03599v2

on top of [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style,
manager-based RL on MuJoCo-Warp).

The policy is conditioned on **contact goals** — where and when each
end-effector should make or break contact — instead of velocity targets or
reference motions. One reward formulation (reach / hold / detach) covers a whole
family of gaits.

Scope: **Unitree Go2**, flat terrain, multi-gait locomotion (trot / pace /
bound / jump / crawl). Humanoid / manipulation results are out of scope.

> **Audit:** see [`docs/AUDIT.md`](docs/AUDIT.md) for the paper↔code gap
> analysis, the bugs fixed (stale reset anchoring, diverging front/hind
> footholds, non-rotating layout on curved paths, discovery halving the command
> duration, Eq. 1 kernel, entropy-decay runner) and what was / was not verified.

---

## Tasks

| Task id | What |
| --- | --- |
| `Mjlab-Contact-Flat-Unitree-Go2` | Paper-faithful implementation (default). |
| `Mjlab-Contact-Flat-Unitree-Go2-Improved` | **Experimental**: + IMU (projected gravity, base angular velocity) in the actor and running observation normalisation. Everything else identical. |

The pre-audit implementation is commit `23ad625` on `main`.

## The contact goal

For each foot `e`, over a horizon of **two contact switches** (§3.2):

| Symbol | Meaning | Shape (per env) |
| --- | --- | --- |
| `p^con_{e,1}`, `p^con_{e,2}` | current / next contact location (yaw-aligned base frame) | 4 × 2 × 3 |
| `I^con_{e,1}`, `I^con_{e,2}` | current / next contact indicator | 4 × 2 |
| `s` | remaining time of the current goal (`S ~ U[0.34, 0.36] s`) | 1 |

Phases: **reach** `I^con_1 = 0 ∧ s ≤ δ` · **hold** `I^con_1 = 1` ·
**detach** `I^con_1 = 0 ∧ s > δ` (δ = 0.18 s).

### Rewards (Eq. 1–3, `d` = unsquared L2 distance)

```
reach  = exp(-d(p1, p_act) / σ²) · 1[I^con_1 = 0 ∧ s ≤ δ]
hold   = (1 + α_hold · exp(-d / σ²)) · 1[I^con_1 = I^act = 1]
detach = 1[I^con_1 = I^act = 0 ∧ s > δ]
```

σ² = 0.1 m, α_hold = 1 (unreported in the paper). `kernel="gaussian"`
(`exp(-d²/σ²)`, the pre-audit form) is available for ablations. Plus the
paper's locomotion penalties and a goal-discovery bonus (once per switch when
the base stays near the foothold centroid; it does **not** shorten `S`).

### Planner (prespecified, `mdp/planning.py`)

Command parameters are sampled **once per env** (§4): stride per front/hind
pair `U(0, 0.3)`, stance width `U(0.1, 0.3)`, per-leg offsets
`U(-0.15, 0.15)`, heading offset `U[-π, π]` **or** yaw rate `U[-π, π]` rad/s.
A virtual base advances `mean_stride / P` per switch (P = gait period in
switches) along facing + heading offset, rotating on curved paths. A foot that
lifts off gets the foothold under its hip at the middle of its next stance;
the per-pair stride shifts the landing point by ±½(s_pair − mean). Front and
hind footholds therefore stay consistent, the layout rotates with the path,
and `p2` is exactly the next `p1`.

### Observations

Actor: joint pos, joint vel, last action, contact goal (above), feet→goal
vectors — no height scan, no foot-contact sensing. Critic (privileged): + base
lin/ang velocity, projected gravity, actual foot contact.

### Logged metrics (paper Fig. 6)

`Metrics/contact/contact_plan_hamming` (feet whose contact state disagrees
with the plan, per step) and `Metrics/contact/contact_location_error` (L2, m,
feet in contact), both episode-averaged.

---

## Setup

`uv` workspace whose root is `contact-rl` and whose only member is the vendored
`mjlab` (`.mjlab_ref`). The committed `uv.lock` pins the full graph.

```bash
uv sync --extra cu128 --python 3.12 --frozen   # Linux x86_64 + NVIDIA GPU
uv sync --extra cpu --frozen                   # CPU smoke tests only
```

Dependency-resolution notes (do not undo; details in `pyproject.toml`):
`mujoco-warp==3.5.0.2` comes from PyPI rather than mjlab's git rev (the git
source pins mujoco to a nightly-only index), and the `cu128` / `cpu` extras
exist only on the workspace root (extras on both packages make torch indexes
conflict). `[tool.uv] environments` narrows resolution to linux/x86_64.

Headless servers: the default logger is `wandb` — run `wandb login`, export
`WANDB_MODE=offline`, or pass `--agent.logger tensorboard`. Use `tmux`.

## Commands

```bash
# Unit tests (planner, phase masks, Eq. 1 kernel) -- torch only, no GPU needed
uv run --with pytest pytest tests -q

# Hardware report + env throughput sweep: pick num_envs for your GPU
uv run contact-bench --num-envs 2048 4096 8192

# Smoke run (Warp compiles kernels on the first run)
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 \
    --env.scene.num-envs 512 --agent.max-iterations 5 --agent.logger tensorboard

# Full run (paper: 8192 envs)
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 8192

# Visualise
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint-file <model.pt>

# Fig. 6 protocol: per gait x command duration, 1000 x 15 s episodes -> CSV
uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 --checkpoint-file <model.pt>
```

`contact-train` / `contact-play` register the tasks and then delegate to
mjlab's CLIs, so all mjlab flags work.

## Training setup

PPO, recurrent GRU actor & critic (rsl-rl `RNNModel`, 256 hidden, MLP head
512-256-128), 24 steps/env, 5 epochs × 4 mini-batches, adaptive LR 1e-3,
γ = 0.99, λ = 0.95. **Entropy decay**: linear 0.01 → 0.001 over 5000 iterations
(`rl_cfg.py`), applied before every PPO update by `ContactOnPolicyRunner`
(resume-aware). Seed 42.

## Project layout

```
src/contact_rl/
  robots/go2.py                       Go2 EntityCfg + action scale
  tasks/contact/
    mdp/planning.py                   GaitPlanner, gait patterns, Eq. 1-3 math (pure torch)
    mdp/contact_command.py            ContactGoalCommand (mjlab wiring + metrics)
    mdp/rewards.py                    reach / hold / detach + penalties
    mdp/observations.py               feet->goal vectors; privileged foot contact
    mdp/events.py                     startup foot-order assertion
    contact_env_cfg.py                robot-agnostic task factory
    config/go2/                       env / PPO cfgs, runner, task registration
  scripts/{train,play,evaluate,benchmark}.py
tests/test_planning.py
docs/AUDIT.md
```
