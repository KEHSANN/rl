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
| `Mjlab-Contact-Flat-Unitree-Go2` | Paper-faithful implementation (default). Runs under `runs/go2_contact/`. |
| `Mjlab-Contact-Flat-Unitree-Go2-Improved` | **Experimental**: + IMU (base angular velocity, projected gravity) in the actor and running observation normalisation on actor *and* critic. Runs under `runs/go2_contact_improved/`. |

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

### Posture regularisation (not in the paper)

The contact rewards only look at the foot sites / foot geoms. With
thigh = calf = 0.213 m a kneeling hind leg still has its foot sphere on the
goal, so reach / hold / detach pay out in full, and a 2000-iteration run learned
to walk on its hind knees in every gait. Three terms close that loophole
(`contact_env_cfg.py`, functions in `mdp/rewards.py`):

| Term | Function | Weight (per s) | What |
| --- | --- | --- | --- |
| `illegal_contact` | `undesired_contact_count` | -2.0 | number of non-foot collision geoms (base, hips, thighs, calves) touching the ground; sensor `illegal_contact` in `config/go2/env_cfgs.py` |
| `base_height_below` | `base_height_below` | -10.0 | `max(0.25 - z_base, 0)`, one-sided and linear (jumps are never penalised) |
| `base_tilt` | `base_tilt_l2` | -1.0 | squared xy of the projected gravity (constant roll / pitch) |

Watch `Episode_Reward/illegal_contact` in TensorBoard: it should go to ~0.
They change no observation, action or network shape, so older checkpoints can
be resumed with them (next section).

## Installation

`uv` workspace whose root is `contact-rl` and whose only member is the vendored
`mjlab` (`.mjlab_ref`). The committed `uv.lock` pins the full graph; resolution
is narrowed to Linux x86_64 (`[tool.uv] environments`).

```bash
uv sync --extra cu128 --python 3.12 --frozen   # Linux x86_64 + NVIDIA GPU
uv sync --extra cpu --frozen                   # CPU smoke tests only
.venv/bin/python scripts/patches/patch_mujoco_warp_sensor.py   # required, see below
export UV_NO_SYNC=1                            # see below
uv run contact-doctor                          # machine-readiness report
```

`scripts/vps/setup.sh` performs the sync, the patch and the doctor check for
you (and `scripts/vps/common.sh` exports `UV_NO_SYNC=1`); run the commands
above by hand only if you are not using it.

**Post-sync patch (required):** `mujoco-warp==3.5.0.2` generates its sensor
kernels from `mujoco_warp/_src/sensor.py`, whose `UNKNOWN` frame-axis branch
reads `xmat` before it is assigned; Warp rejects the generated kernel and env
construction fails. `scripts/patches/patch_mujoco_warp_sensor.py` inserts the
missing `xmat` initialisation into the **installed** package. It is SHA256-gated
(it refuses to touch an unexpected file), idempotent, atomic, and must be re-run
after every `uv sync` that reinstalls `mujoco-warp`, because the edit lives in
`.venv`, not in this repo.

**`UV_NO_SYNC=1`:** `uv run` normally re-syncs the project env before every
command, and without `--extra cu128` that sync targets the *no-extra*
resolution: it silently replaces the CUDA torch build installed above with a
different torch wheel. Sync once, explicitly (`uv sync --extra ...` or
`scripts/vps/setup.sh`), then run everything with `UV_NO_SYNC=1` exported or
as `uv run --no-sync ...` (`--no-sync` implies `--frozen`). The VPS scripts
(`scripts/vps/common.sh`, also inside tmux sessions) and the systemd units set
it for you. Re-sync after a `git pull` that changes `uv.lock`. All `uv run`
commands below assume it is set.

Dependency-resolution notes (do not undo; details in `pyproject.toml`):
`mujoco-warp==3.5.0.2` comes from PyPI rather than mjlab's git rev, and the
`cu128` / `cpu` extras exist only on the workspace root.

Two fixes carried in the vendored mjlab (`.mjlab_ref`):
`scipy` is declared as a dependency because `mjlab.terrains` imports it while
upstream never listed it, and `sim/sim.py` uses `wp.get_cuda_driver_version()`
instead of `wp.context.runtime.driver_version`, which warp 1.16 no longer
exposes (it raises `AttributeError`, disabling CUDA graphs).

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

Every command supports `--help`. `contact-train` and `contact-play` take the
task id as a positional argument, so their help is two-level: `contact-train
--help` lists the task ids plus the contact-train-only flags, while the full
`--env.*` / `--agent.*` option list is built from the selected task and needs
`contact-train <task> --help`.

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
ignored; a later Ctrl-C aborts hard (exit 130). A Ctrl-C / SIGTERM *before*
training has started (e.g. while the env is being built and Warp compiles its
kernels) is a clean stop too: exit 0, `run_info.json` finalised as
`status: "stopped"` (no checkpoint yet), further signals during that stop are
ignored.

## Continuing training from a finished checkpoint

A run that reached `max_iterations` (`status: "finished"`) is resumed exactly
like a stopped one. Example: the committed 2000-iteration checkpoint
(`model_1999.pt` = iterations 0..1999), 3000 more iterations up to 5000:

```bash
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 \
  --resume-from runs/go2_contact/2026-10-01_04-21-51/checkpoints/model_1999.pt \
  --agent.max-iterations 3000
# equivalent selector: runs/go2_contact/2026-10-01_04-21-51:1999
```

- `--agent.max-iterations` is the number of **additional** iterations
  (`start .. start + N - 1`, printed at startup), not the final iteration.
- Restored from the checkpoint: actor + critic weights (including the GRUs),
  optimizer state and the iteration counter (training continues at
  `iter + 1`). The entropy schedule uses the absolute iteration, so it simply
  continues (0.0064 at iteration 2000, 0.001 from 5000).
- **Not** restored: the env config. It is rebuilt from the current code and the
  CLI flags, so reward / termination / domain-randomisation edits made after
  the checkpoint take effect on resume.
- Output goes to a **new** run dir; the source checkpoint is never written.
  `best.pt` of the new run is ranked by the *new* reward, which is not
  comparable with the old run's numbers.
- If the checkpoint came from git, it must be the real file, not a Git LFS
  pointer (`ls -lh` should show MBs, not ~130 bytes; `git lfs pull` if needed).

What may change between the checkpoint and the resumed run:

| Change | OK? | Note |
| --- | --- | --- |
| reward terms / weights | yes | expect a value-loss spike and a reward dip for ~100-200 iterations while the critic re-fits |
| terminations, events, pushes, friction ranges | yes | add a new hard termination gradually (a penalty first), or many episodes end at once |
| sensors that are not observations (e.g. `illegal_contact`) | yes | |
| `--env.scene.num-envs` | yes | see below |
| PPO hyper-parameters (`--agent.algorithm.*`) | yes | |
| actor / critic observations | **no** | input size changes, `strict=True` load fails |
| action space, network sizes, GRU size | **no** | same reason |
| default task <-> `-Improved` task | **no** | different observations (IMU + normalisation) |

**Different `num_envs` (e.g. 12000 -> 20000):** fine. The network does not
depend on the number of envs, and each env's command (gait, strides, stance,
heading / yaw rate) is sampled once when the env is built, so a larger count
just covers more of the command space. A PPO iteration then collects
`24 x num_envs` transitions and the 4 mini-batches get larger; the adaptive LR
handles that. The real limits are VRAM and throughput: check first with
`uv run contact-bench --num-envs 12000 16000 20000`, and if `env-steps/s` no
longer grows, more envs only make each iteration slower.

If the old policy is stuck in a bad local optimum (e.g. kneeling) and the new
penalty does not remove it within a few hundred iterations, train from scratch
instead: by iteration 2000 the action noise has already shrunk, which makes
escaping it slow.

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
checkpoint path (`--source-checkpoint`). If the snapshot cannot be taken, or no
longer validates once taken (retention replaced the source after it was
queued), the checkpoint is retried on the next poll rather than evaluated in
place, and the `--max-retries` budget is not consumed. Stale `summary.json`
files are removed before each attempt. `metrics/watch.lock` (flock) makes a
second watcher on the same run exit with code 3; checkpoints deleted by
retention are skipped.
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

```
runs/<experiment>/<timestamp>[_name]/      # e.g. runs/go2_contact/2026-01-31_12-00-00
  run_info.json                            # status: starting|running|finished|stopped|aborted|failed
  checkpoints/model_<iter>.pt              # atomic writes (.tmp + rename)
  checkpoints/latest.pt                    # symlink (copy if symlinks fail) to the newest model_<iter>.pt
  checkpoints/best.pt                      # copy of the best checkpoint (mean episode reward)
  checkpoints/index.json                   # every saved iteration + best
  config/  logs/train.log  logs/error.txt (on crash)  exported/policy.onnx  metrics/  videos/
```

Retention keeps the newest `--keep-last` (5), multiples of `--keep-every`
(1000) and the best iteration. Selectors (`--checkpoint`, `--resume-from`):
a `.pt` file, a run dir (= its latest), `<run_dir>:<iter>|best|latest`, or
`latest` / `best` (newest run under `runs/`). Corrupt or
half-written files are skipped by `latest`. mjlab's legacy
`logs/rsl_rl/<experiment>/<run>/model_<iter>.pt` layout is only produced by a
multi-GPU run (see below) and is still readable (`--checkpoint-file`,
`--agent.resume True`). `runs/` and `logs/` are git-ignored.

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
- `--gpu-ids` takes a quoted Python list (mjlab's tyro config uses
  `UsePythonSyntaxForLiteralCollections`): `--gpu-ids "[0]"` (default),
  `--gpu-ids "[1]"`, `--gpu-ids "[0, 1]"`, or `--gpu-ids all`. `--gpu-ids 0 1`
  is **not** valid. The ids index into `CUDA_VISIBLE_DEVICES`; alternatively
  `CUDA_VISIBLE_DEVICES=1` or `--device cuda:1`.
- One selected GPU (including `all` on a one-GPU machine) trains in-process
  with full run management. Only when more than one GPU is actually selected
  is training delegated to mjlab's multi-GPU launcher (legacy `logs/rsl_rl`
  layout, no run management; contact-train-only flags are ignored with a
  warning).

## VPS

See [`docs/VPS_DEPLOYMENT.md`](docs/VPS_DEPLOYMENT.md) and
[`scripts/vps/README.md`](scripts/vps/README.md): tmux wrappers that report the
command's real exit code, graceful `stop.sh`, TensorBoard, SSH tunnel and
optional systemd user units (which use the 8192 default).

## Testing

```bash
UV_NO_SYNC=1 uv run --with pytest pytest tests -q   # after the env was synced once
```

- **Lightweight / local** (no GPU): planner math, checkpoint lifecycle, episode
  metrics, watcher logic, CLI helpers, VPS script syntax / tmux exit codes,
  systemd and `.gitignore` static checks (`tests/test_vps_scripts.py`), and
  the second-pass regressions (`tests/test_second_pass_regressions.py`: VPS
  scripts against fake `uv` / `tmux`, `UV_NO_SYNC` propagation, systemd unit
  settings, `contact-train` startup stop against stubbed torch / mjlab,
  atomic JSON, duplicate Ctrl-C, run lock, ONNX temp cleanup).
- **Linux/VPS-only**: anything importing mjlab / MuJoCo-Warp (the lock is
  Linux x86_64 only), EGL rendering, `contact-doctor` end to end.
- **GPU-dependent**: training, `contact-bench`, full `contact-eval` runs.

## Why many parallel environments (and why `contact-play` shows 64 robots)

`contact-play` and `contact-eval` load the task's **play** config
(`env_cfgs.py`, `play=True`), which uses **64** envs, no pushes, no observation
noise and practically endless episodes. Nothing is being trained there: the
robots only run the loaded checkpoint. Use `--num-envs 1` to see a single robot:

```bash
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best --num-envs 1
```

Training (`contact-train`) uses **8192** envs by default. What the env count
does and does not change:

- **Iteration time barely changes, sample throughput does.** MuJoCo-Warp steps
  all envs in one batched GPU kernel. Until the GPU is saturated, stepping 8192
  envs takes about as long as stepping 512, so seconds per iteration look
  "the same". But one PPO iteration collects `24 x num_envs` transitions
  (512 envs: 12,288; 8192 envs: 196,608), i.e. 16x more experience in the same
  wall-clock time. Compare the `fps` value printed per iteration (and
  `training/fps` in TensorBoard), not the iteration time.
- **Better gradients.** PPO splits each iteration into 4 mini-batches; with more
  envs they are larger and less noisy, so learning per iteration is more stable.
- **Command coverage (specific to this task).** Each env samples its gait,
  front/hind stride, stance width, leg offsets and heading *or* yaw rate
  **once**, when the env is built (paper Sec. 4). With 8192 envs every one of the
  5 gaits gets ~1600 different command combinations; with 512 envs only ~100.
  Fewer envs means the policy sees a thinner slice of the command space and
  generalises worse, even when the iteration speed looks identical.
- **Past the knee it stops helping.** Once the GPU is saturated, iteration time
  grows roughly linearly with `num_envs` and throughput stops improving (or VRAM
  runs out). Find the knee with
  `uv run contact-bench --num-envs 1024 2048 4096 8192` and pick the largest size
  whose `env-steps/s` still grows clearly.

If you change `num_envs`, the number of iterations needed to converge changes
too: with fewer envs you need more iterations for the same amount of experience.

## How long training takes and when it ends

- A run ends after `max_iterations` = **10,000** PPO iterations (`rl_cfg.py`),
  i.e. iterations `start .. start + 9999` (printed at startup). At the end the log
  prints `[contact-train] finished.` and `run_info.json` gets
  `"status": "finished"`, `"exit_code": 0`.
- Total time = `10,000 x (seconds per iteration)`. Every iteration prints a line
  like `[iter 1234] reward=... t=1.80s fps=...`; multiply `t` by the remaining
  iterations. Examples: 1 s/iter ~= 2.8 h, 2 s/iter ~= 5.6 h, 4 s/iter ~= 11 h.
  The first iterations are slower (Warp kernel compilation / CUDA graph capture).
  The same number is in TensorBoard as `training/iteration_time`.
- Check progress at any time:

```bash
tail -f runs/go2_contact/<run>/logs/train.log      # per-iteration line
cat runs/go2_contact/<run>/run_info.json           # status, last_iteration
cat runs/go2_contact/<run>/checkpoints/index.json  # saved iterations + best
```

- A checkpoint is written every 50 iterations (`save_interval`), so you can stop
  early at any time (Ctrl-C once, or `bash scripts/vps/stop.sh train`): the
  current iteration finishes, a checkpoint is saved and `best.pt` stays usable.
  The entropy coefficient decays until iteration 5000; if `training/reward` and
  the evaluation metrics have plateaued well after that, stopping early is fine.
- Shorter run: `--agent.max-iterations 3000`. Continue a stopped run:
  `--resume-from runs/go2_contact/<run>:latest` (continues in a new run dir).

## Pushing the trained weights to GitHub

`runs/`, `*.pt` and `*.onnx` are git-ignored on purpose (checkpoints are written
every 50 iterations). Publish only the final artefacts. `latest.pt` may be a
symlink, so copy with `cp -L`.

Option A: commit them into the repo (the Go2 GRU checkpoint is a few MB, well
below GitHub's 100 MB per-file limit):

```bash
RUN=runs/go2_contact/<run>                       # the finished run dir
mkdir -p weights/go2_contact
cp -L $RUN/checkpoints/best.pt       weights/go2_contact/best.pt
cp -L $RUN/exported/policy.onnx      weights/go2_contact/policy.onnx
cp    $RUN/run_info.json $RUN/checkpoints/index.json weights/go2_contact/
cp -r $RUN/config                    weights/go2_contact/config
git add -f weights/go2_contact                   # -f: bypasses the *.pt / *.onnx ignore rules
git commit -m "weights: go2_contact <run> best checkpoint + ONNX"
git push origin audit/paper-alignment
```

After cloning, play it directly: `uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint weights/go2_contact/best.pt`.

Option B: attach them to a GitHub Release (keeps the git history small,
recommended if you publish many runs), with the GitHub CLI:

```bash
gh release create go2-contact-v1 weights/go2_contact/best.pt weights/go2_contact/policy.onnx \
  --title "Go2 contact policy v1" --notes "run <run>, iteration <it>"
```

For files above 100 MB use Git LFS (`git lfs track "*.pt"`) or a Release.

## Commanding the robot after training

**In simulation (Viser viewer).** Start the viewer on the server, open an SSH
tunnel from your laptop, then open `http://localhost:8080`:

```bash
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best --num-envs 1   # on the server
ssh -N -L 8080:127.0.0.1:8080 <user>@<vps>                                          # on your laptop
```

The **Contact control** panel steers the policy through the same planner that
generated the training goals:

- **Gait**: `(as trained)`, trot, pace, bound, jump, crawl.
- **Direction (deg)**: travel direction relative to the body (0 forward, 90
  left, 180 backwards).
- **Speed (m/s)**: converted to a stride (`stride = speed x period x S`). Speed
  0 = step in place.
- **Turning (rad/s)**: yaw rate. Training samples a direction *or* a turning
  rate, never both, so combining them is out of distribution.
- **Apply to all envs** (or only the selected env), **Apply command**,
  **Restore trained commands**, **Restart episode (reset GRU)**.
- **Checkpoint** panel: switch between saved checkpoints of the run without
  restarting.

Trained range: stride 0 to 0.3 m, i.e. about **0.43 m/s** max for trot / pace /
bound / jump and about **0.21 m/s** for crawl, and turning up to pi rad/s.
Larger values are applied but flagged as OUT OF TRAINING DISTRIBUTION in the
panel (nothing is clamped).

**From Python** (same pathway, e.g. for scripted tests):

```python
import math
from contact_rl.tasks.contact.mdp.command_override import UserCommand, apply_to_command_term

term = env.unwrapped.command_manager.get_term("contact")
warnings = apply_to_command_term(term, UserCommand(gait="trot", speed=0.3, heading_offset=0.0, yaw_rate=0.0))
# walk left: heading_offset=math.pi / 2 ; turn on the spot: speed=0.0, yaw_rate=1.0
```

**On the real Go2.** This repo contains no hardware deployment code. The
exported `exported/policy.onnx` is the actor only (GRU hidden state is an extra
input / output that must be carried between steps and zeroed on reset). A
deployment has to rebuild the actor observation at 50 Hz exactly as in training
(joint positions relative to default, joint velocities, last action, the 33-dim
contact goal from `GaitPlanner` in the yaw-aligned base frame, feet-to-goal
vectors from forward kinematics) and send the 12 outputs, scaled by
`GO2_ACTION_SCALE` and added to the default joint positions, as PD position
targets. Validate in simulation first.
