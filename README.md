# Learning to Act Through Contact — mjlab implementation

An implementation of

> **Learning to Act Through Contact: A Unified View of Multi-Task Robot Learning**
> Shafeef Omar, Majid Khadiv (TUM), LDC 2026 — arXiv:2510.03599v2

on top of [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style,
manager-based RL powered by MuJoCo-Warp).

The paper's central idea is **contact-explicit, goal-conditioned RL**: instead of
commanding a *velocity target* (locomotion) or a *reference motion* (imitation),
the policy is conditioned on **contact goals** — where and when each
end-effector should make or break contact. A single reward formulation
(reach / hold / detach) then covers a whole family of tasks and gaits.

This repository implements **only that paper**. The `.mjlab_ref` recipe and any
other bundled references are used for framework plumbing only; none of their
*tasks* (mjlab's velocity target, mjlab's motion tracking) are mixed in — those
are exactly the paradigms this paper argues against.

Scope of this implementation: the **Unitree Go2 quadruped**, flat terrain,
multi-gait locomotion (trot / pace / bound / jump / crawl) via contact goals.
Manipulation (the paper's humanoid results) is intentionally out of scope for
now (see *Future work*).

---

## The contact goal (what conditions the policy)

For each foot `e` and control step `t` (Section 3.1), over a horizon of **two
contact switches** (current + next):

| Symbol | Meaning | Shape (per env) |
| --- | --- | --- |
| `p^con_{e,t,1}`, `p^con_{e,t,2}` | current / next desired contact location (base frame) | 2 × 4 × 3 |
| `I^con_{e,t,1}`, `I^con_{e,t,2}` | current / next desired contact indicator (0/1) | 2 × 4 |
| `S` | command duration of the current goal | 1 |

Stacking `I^con` over the feet is the **contact sequence**; the per-gait
sequences produce trot / pace / bound / jump / crawl.

The **contact phase** of a foot (Fig. 3) follows from `I^con_{e,t,1}` and the
remaining command time `s` vs a threshold `ν`:

- **reach**  — `I^con_1 == 0` and `s < ν`
- **hold**   — `I^con_1 == 1`
- **detach** — `I^con_1 == 0` and `s > ν`

### Rewards (Section 3.2, Eq. 1–3)

With `d(·)` the L2 distance, `p^act` / `I^act` the *actual* foot position /
contact:

```
reach  :  exp(-d(p^con_1, p^act)^2) · 1(I^con_1 = 0 ∧ s < ν)              (Eq. 1)
hold   :  (1 + λ_hold · exp(-d(p^con_1, p^act)^2)) · 1(I^con_1 = I^act = 1) (Eq. 2)
detach :  1(I^con_1 = I^act = 0 ∧ s > ν)                                  (Eq. 3)
r^con  :  r^obj_pose + Σ_e (reach + hold + detach)
```

`r^obj_pose` is the manipulation-only object term and is omitted for locomotion.
Each of reach / hold / detach is a separate, independently weighted mjlab reward
term (see `contact_env_cfg.py`). In code, `exp(-d^2)` is generalised to
`exp(-d^2 / std^2)`; `std = 1` reproduces the paper's literal form, and the
default `std = 0.1` simply sharpens the spatial tolerance — set it to `1.0` for
the exact expression.

### Observations (Section 3.3)

- **Actor** (paper-faithful): proprioception (joint positions, joint velocities,
  last action) **+** task obs (current & next contact sequence of all feet;
  current & next contact locations of all feet in the base frame; command
  duration; relative distance of feet to their desired contacts). No height
  scan; **no foot-contact sensing in the actor** ("did not make any difference
  in simulation performance").
- **Critic** (asymmetric / privileged): the actor obs **+** base linear &
  angular velocity, projected gravity, and actual foot contact. This changes
  nothing about the deployed policy's input contract.

### Command sampling (Section 4, "Locomotion")

Per the paper, the locomotion command distribution is **sampled once at
initialisation** (it reports this yields a better policy than per-reset
sampling): stride lengths `U(0.0, 0.3) m` and stance widths `U(0.1, 0.3) m` per
front/hind pair; a heading `U[-π, π] rad` *or* (for curved paths) a yaw rate
`U[-π, π] rad/s`; per-leg longitudinal + lateral offsets `U(-0.15, 0.15) m`;
command durations `U[0.34, 0.36] s`. Extensive domain randomisation
(friction, encoder bias, base-CoM offset, pushes) is applied on top.

Goals **advance early with a bonus** when the base, projected to the ground,
comes within a threshold of the current footholds — the paper's "update the
goals … and provide a bonus reward for discovering more goals".

---

## Project layout

```
src/contact_rl/
  robots/go2.py                       Unitree Go2 EntityCfg + action scale
  tasks/contact/
    mdp/
      contact_command.py              ContactGoalCommand (the contact goal)
      rewards.py                      reach / hold / detach (Eq. 1–3) + penalties
      observations.py                 feet→goal distance; privileged foot contact
    contact_env_cfg.py                robot-agnostic task factory
    config/go2/
      env_cfgs.py                     Go2 flat env (train + play)
      rl_cfg.py                       PPO runner cfg (recurrent GRU actor/critic)
      runner.py                       ContactOnPolicyRunner (ONNX + entropy decay)
      __init__.py                     register_mjlab_task(...)
  scripts/{train,play}.py             entry points (register task, then delegate)
src/assets/robots/unitree_go2/        Go2 MJCF + meshes
paper.txt                             the paper (source text)
```

Task id: **`Mjlab-Contact-Flat-Unitree-Go2`**.

---

## Setup

This directory is a **uv workspace** whose root is `contact-rl` and whose only
member is the vendored `mjlab` (`.mjlab_ref`). uv reads resolver-wide settings
(`conflicts`, `environments`) and index definitions only from the workspace root,
so mjlab's custom indexes (NVIDIA warp, PyTorch CUDA) are mirrored in
`pyproject.toml`. Two upstream choices are deliberately *not* mirrored — the
`mujoco-warp` git revision and mjlab's own torch extras; both are explained in
comments in `pyproject.toml` and summarised below.

A committed `uv.lock` (224 packages) pins the whole graph, so `uv sync` installs
instead of re-resolving. Keep it with the tree when you copy the project
anywhere.

```bash
# CUDA (Linux, recommended for training — matches the paper's setup):
uv sync --extra cu128

# CPU-only (small smoke tests):
uv sync --extra cpu
```

> mjlab targets Linux-x86_64 (CUDA) and macOS-arm64, and this workspace narrows
> the resolution further to **Linux-x86_64 only** (the deployment target). On
> other platforms `uv sync` reports that the current platform is not compatible
> with the lockfile's supported environments; widen `[tool.uv] environments` if
> you need one of them. Large-scale training needs an NVIDIA GPU.

### Running on a Linux GPU server

Two dependency-resolution details make the difference between "resolves in
seconds" and "unsatisfiable"; both are already encoded in `pyproject.toml`, so
the steps below are all you need — but do not undo them:

* **`mujoco-warp` comes from PyPI (`==3.5.0.2`), not from mjlab's git rev.** uv
  honours a *git* dependency's own `[tool.uv.sources]`, and `mujoco_warp` pins
  `mujoco` to `py.mujoco.org`, which serves only `.dev` nightlies (oldest listed:
  `3.10.1.dev939631378`). With the git rev in place, `mujoco>=3.5.0,<3.6` is
  unsatisfiable, and a root-level `mujoco = { index = "pypi" }` pin does not
  override it — uv fails with *conflicting indexes for package `mujoco`*. The
  PyPI release is the same revision with identical requirements, as a
  pure-python wheel, so the server needs neither `git` nor a source build.
* **The `cu128` / `cpu` extras exist only on the workspace root.** They were
  removed from `.mjlab_ref` (where they only re-added `torch>=2.7.0`, already a
  base dependency). With extras on both packages uv splits the resolution 9 ways
  and cross-splits such as `contact-rl[cu128] + mjlab[cpu]` activate both torch
  indexes → *conflicting indexes for package `torch`*.

`[tool.uv] environments` also narrows the resolution to `linux/x86_64`, which is
why `uv sync` on Windows/macOS refuses the lockfile — that is expected.

```bash
# 0. Prerequisites: an NVIDIA driver new enough for CUDA 12.8 wheels
#    (>= 525.60.13; >= 550 recommended) and ~25 GB free disk.
nvidia-smi

# 1. Install uv (brings its own Python; no system Python needed).
curl -LsSf https://astral.sh/uv/install.sh | sh && . "$HOME/.local/bin/env"

# 2. Install from the committed lockfile (--frozen = fail rather than silently
#    re-resolve if uv.lock is missing or stale).
uv sync --extra cu128 --python 3.12 --frozen

# 3. Verify the GPU is actually visible to torch (a broken driver otherwise
#    surfaces much later as an IndexError inside select_gpus).
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Verify the task registered.
uv run python -c "import contact_rl; from mjlab.tasks.registry import list_tasks; print(list_tasks())"

# 5. Short smoke run (Warp compiles kernels on the first run: a few minutes).
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 \
    --env.scene.num-envs 512 --agent.max-iterations 5 --agent.logger tensorboard
```

Two things that bite on a headless box:

* **The default logger is `wandb`**, so an un-authenticated server stalls or
  errors on `wandb.init`. Either `wandb login` once, or export
  `WANDB_MODE=offline`, or pass `--agent.logger tensorboard`.
* **Long runs need `tmux`/`screen`** — the training process dies with the SSH
  session otherwise.

## Train

```bash
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2
```

`contact-train` registers this task into mjlab's registry and then hands off to
mjlab's own training CLI, so **all** of mjlab's `train` flags work unchanged,
e.g. limit parallel envs to fit your GPU:

```bash
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096
```

## Play / visualise a checkpoint

```bash
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --wandb-run-path <entity/project/run>
```

---

## Training setup

- **Recurrent (GRU) actor & critic.** The policy is recurrent, as in the paper.
  On the pinned stack this uses rsl-rl's unified `class_name="RNNModel"`
  (`rsl_rl/models/rnn_model.py`, which inherits `MLPModel` and adds
  `rnn_type`/`rnn_hidden_dim`/`rnn_num_layers`). mjlab forwards the runner cfg to
  rsl-rl via `dataclasses.asdict` without dropping unknown keys, so the
  `RslRlRnnModelCfg` subclass in `rl_cfg.py` flows straight through; PPO
  auto-detects recurrence and manages hidden states / BPTT itself. Defaults:
  GRU, `rnn_hidden_dim=256`, `rnn_num_layers=1`, MLP head `(512, 256, 128)`.
- **Entropy decay.** The paper anneals the entropy coefficient. rsl-rl's
  `PPO.entropy_coef` is a plain mutable float with no built-in schedule, so
  `ContactOnPolicyRunner` applies a **linear decay** by chunking `learn()` and
  re-setting `alg.entropy_coef` between chunks. Tune via the module constants in
  `runner.py`.

## Minor configuration notes

- Default `num_envs = 4096` (paper uses 8192); raise/lower via
  `--env.scene.num-envs` as GPU memory allows.

## Design choices where the paper is underspecified

The paper specifies the *contents* of a contact goal and the command sampling,
but not the exact **foothold propagation**. This implementation (documented in
`contact_command.py`):

- lays footholds out in a **travel frame** whose yaw is the desired
  travel/facing direction and which integrates the yaw rate for curved paths;
- marches a foot's world foothold forward by its pair's stride each time it
  *lifts off* (contact sequence 1 → 0), so the very first goal a swinging foot
  sees is already one stride ahead of where it departed; goals are
  piecewise-constant within a switch so the target is stable;
- re-anchors footholds to a nominal stance at each reset;
- advances goals early on discovery (see above).

The gait contact patterns (`_GAIT_PATTERNS` in `contact_command.py`), the
phase threshold `ν = 0.18`, the reward `std`, and all reward weights are
explicit, documented constants intended for tuning.

## Future work

- Manipulation / humanoid tasks (the object-pose term `r^obj_pose` and the
  contact goals for a humanoid's end-effectors), building on the same
  `ContactGoalCommand` and reach/hold/detach rewards.
