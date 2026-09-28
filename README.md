# Learning to Act Through Contact: mjlab implementation

An implementation of

> **Learning to Act Through Contact: A Unified View of Multi-Task Robot Learning**
> Shafeef Omar, Majid Khadiv (TUM), L4DC 2026, arXiv:2510.03599v2

on top of [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style,
manager-based RL on MuJoCo-Warp).

## Project overview

The policy is conditioned on **contact goals** (where and when each
end-effector should make or break contact) instead of velocity targets or
reference motions. One reward formulation (reach / hold / detach) covers a whole
family of gaits. Scope: **Unitree Go2**, flat terrain, multi-gait locomotion
(trot / pace / bound / jump / crawl).

Architecture: task + MDP code under `src/contact_rl/tasks/contact/`, a project
runner (`config/go2/runner.py`: entropy decay, checkpoint management, ONNX
export, graceful stop), and six CLIs under `src/contact_rl/scripts/` with
shared utilities in `src/contact_rl/utils/`. See [`docs/AUDIT.md`](docs/AUDIT.md)
for the paper/code gap analysis.

| Task id | What |
| --- | --- |
| `Mjlab-Contact-Flat-Unitree-Go2` | Paper-faithful implementation (default). |
| `Mjlab-Contact-Flat-Unitree-Go2-Improved` | **Experimental**: + IMU in the actor and running observation normalisation. |

### The contact goal

For each foot `e`, over a horizon of **two contact switches** (paper 3.2):
current / next contact location `p^con_{e,1..2}` (yaw-aligned base frame,
4 x 2 x 3), indicators `I^con_{e,1..2}` (4 x 2) and remaining goal time `s`
(`S ~ U[0.34, 0.36] s`). Phases: **reach** `I^con_1 = 0 and s <= delta`,
**hold** `I^con_1 = 1`, **detach** `I^con_1 = 0 and s > delta` (delta = 0.18 s).

```
reach  = exp(-d(p1, p_act) / sigma^2) * 1[I^con_1 = 0 and s <= delta]
hold   = (1 + alpha_hold * exp(-d / sigma^2)) * 1[I^con_1 = I^act = 1]
detach = 1[I^con_1 = I^act = 0 and s > delta]
```

sigma^2 = 0.1 m, alpha_hold = 1. Planner: `mdp/planning.py`. Actor observations:
joint pos/vel, last action, contact goal, feet-to-goal vectors; the critic adds
privileged base velocities, projected gravity and foot contact.

PPO, recurrent GRU actor & critic (rsl-rl `RNNModel`, 256 hidden, MLP head
512-256-128), 24 steps/env, 5 epochs x 4 mini-batches, adaptive LR 1e-3,
gamma 0.99, lambda 0.95, entropy decay 0.01 -> 0.001 over 5000 iterations, seed 42.

## Installation

`uv` workspace whose root is `contact-rl` and whose only member is the vendored
`mjlab` (`.mjlab_ref`). The committed `uv.lock` pins the full graph; resolution
is narrowed to Linux x86_64 (`[tool.uv] environments`).

```bash
uv sync --extra cu128 --python 3.12 --frozen   # Linux x86_64 + NVIDIA GPU
uv sync --extra cpu --frozen                   # CPU smoke tests only
uv run contact-doctor                          # machine-readiness report
```

Dependency-resolution notes (do not undo; details in `pyproject.toml`):
`mujoco-warp==3.5.0.2` comes from PyPI rather than mjlab's git rev, and the
`cu128` / `cpu` extras exist only on the workspace root.

Headless rendering: importing `contact_rl` configures `MUJOCO_GL` (EGL on
headless machines) *before* MuJoCo is imported; an explicitly exported
`MUJOCO_GL` is always respected.

| Command | Module | Purpose |
| --- | --- | --- |
| `contact-train` | `contact_rl.scripts.train` | training with run management |
| `contact-eval` | `contact_rl.scripts.evaluate` | per-episode evaluation (Fig. 6 protocol) |
| `contact-watch` | `contact_rl.scripts.watch` | evaluates new checkpoints of a running job |
| `contact-play` | `contact_rl.scripts.play` | viewer / video for a checkpoint |
| `contact-doctor` | `contact_rl.scripts.doctor` | environment diagnostics |
| `contact-bench` | `contact_rl.scripts.benchmark` | hardware report + env-throughput sweep |

Every command supports `--help`.

## Training

```bash
# Smoke run (Warp compiles kernels on the first run)
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 512 --agent.max-iterations 5

# Full run: paper default, 8192 envs
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2

# Lower-VRAM fallback
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 4096

# Resume (full state, continues in a NEW run dir; the source is never modified)
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --resume-from runs/go2_contact/<run>:best
```

All mjlab `TrainConfig` flags work (`--env.*`, `--agent.*`, `--video`,
`--gpu-ids`, `--enable-nan-guard True` -> `env.sim.nan_guard.enabled`), plus
`--resume-from`, `--device`, `--run-root`, `--keep-last`, `--keep-every`,
`--keep-best`, `--export-onnx`. First Ctrl-C / SIGTERM: finish the iteration,
save a checkpoint, exit. A duplicate SIGINT within 1 s (e.g. from `uv run`) is
ignored; a later Ctrl-C aborts hard (exit 130).

## Evaluation

```bash
uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 --checkpoint <selector|model.pt>
uv run contact-eval --help        # all options (--run-dir, --video, --out-dir ...)
```

Per gait x command duration, 15 s episodes. Fig. 6 metrics
(`contact_plan_hamming`, `contact_location_error`) are step-weighted averages;
per-episode means are also stored in `summary.json` next to the CSV.

## Watcher

```bash
uv run contact-watch --run latest   # or --run runs/go2_contact/<run> --interval-s 60 --num-envs 128
```

Polls a run for new, settled, valid checkpoints and evaluates each once in an
isolated `contact-eval` subprocess (own process group, `--timeout-s`). The
evaluated file is a hard-linked snapshot, so retention cannot delete it
mid-evaluation; `summary.json` / `evaluations.csv` record the original
checkpoint path (`--source-checkpoint`). Stale `summary.json` files are removed
before each attempt. `metrics/watch.lock` (flock) makes a second watcher on the
same run exit with code 3; checkpoints deleted by retention are skipped.
Results: `metrics/iteration_<it>/`, `videos/iteration_<it>/`, `metrics/watch_state.json`.

## Play

```bash
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint-file model.pt
```

GRU hidden state is reset at episode boundaries (Viser and native viewer). The
Viser viewer binds to `127.0.0.1:8080`; a non-loopback `--host` is refused
unless `--allow-public True` is given, and then a prominent security warning is
printed (the viewer has no authentication, prefer an SSH tunnel).

## Diagnostics

`uv run contact-doctor` checks Python, torch/CUDA, MuJoCo/EGL, mjlab and task
registration.

## Benchmark

`uv run contact-bench --num-envs 2048 4096 8192` reports hardware and env
throughput to help pick `num_envs`.

## Checkpoints

`runs/<experiment>/<timestamp>[_name]/` contains `checkpoints/`
(`model_<iter>.pt`, `best.pt`), `config/`, `logs/train.log`,
`exported/policy.onnx`, `metrics/`, `videos/` and `run_info.json`. Retention
keeps the newest `--keep-last` (5), multiples of `--keep-every` (1000) and
`best.pt` (by mean episode reward). Selectors: a file, a run dir,
`run_dir:<iter>|best|latest`, `latest`, `best`. `runs/` is git-ignored.

## Video

`contact-train --video` streams training clips to `videos/train/iteration_<it>/`;
`contact-watch` records evaluation videos to `videos/iteration_<it>/`;
`contact-play --video` writes to `<run>/videos/play/`. Rendering is headless
via EGL on servers. `*.mp4` is git-ignored.

## GRU / ONNX

rsl-rl-lib 5.0.1's GRU export path returns `(actions, h, None)` while declaring
two outputs. `utils/onnx_compat.py` wraps the export module (`DropNoneOutputs`)
so `None` outputs are dropped, output order is preserved and a real
output-count mismatch raises. Non-recurrent / LSTM policies pass through
unchanged. The file is written atomically (`.tmp` + rename).

## Logging

The project default is **TensorBoard** (`rl_cfg.py`: `logger="tensorboard"`);
mjlab's own default (W&B) is overridden because it blocks on a headless VPS.
Opt in with `--agent.logger wandb` (project `contact-rl`).

```bash
uv run tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

## GPU / environment count

- **8192** envs = paper / project default (`PAPER_NUM_ENVS` in `env_cfgs.py`).
- **4096** = lower-VRAM fallback; pass `--env.scene.num-envs 4096` explicitly.
- `--gpu-ids 0` (default), `--gpu-ids 1`, and `--gpu-ids all` on a one-GPU
  machine train in-process with full run management. Only when more than one
  GPU is actually selected is training delegated to mjlab's multi-GPU launcher
  (legacy `logs/rsl_rl` layout, no run management).

## VPS

See [`docs/VPS_DEPLOYMENT.md`](docs/VPS_DEPLOYMENT.md) and
[`scripts/vps/README.md`](scripts/vps/README.md): tmux wrappers that report the
command's real exit code, graceful `stop.sh`, TensorBoard, SSH tunnel and
optional systemd user units (which use the 8192 default).

## Testing

```bash
uv run --with pytest pytest tests -q
```

- **Lightweight / local** (no GPU): planner math, checkpoint lifecycle, episode
  metrics, watcher logic, CLI helpers, VPS script syntax / tmux exit codes,
  systemd and `.gitignore` static checks (`tests/test_vps_scripts.py`).
- **Linux/VPS-only**: anything importing mjlab / MuJoCo-Warp (the lock is
  Linux x86_64 only), EGL rendering, `contact-doctor` end to end.
- **GPU-dependent**: training, `contact-bench`, full `contact-eval` runs.
