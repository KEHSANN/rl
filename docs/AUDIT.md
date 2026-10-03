# Paper ↔ code audit: *Learning to Act Through Contact* (arXiv:2510.03599v2)

Scope: Unitree Go2, flat terrain, multi-gait contact-explicit locomotion
(trot / pace / bound / jump / crawl). The humanoid / manipulation results of the
paper remain out of scope. The pre-audit implementation is preserved on `main`
(commit `23ad625`); this branch contains the corrected implementation.

## 1. What the paper specifies

| Item | Paper |
| --- | --- |
| Goal | per end-effector: contact locations + binary contact indicators for **two contact switches**, remaining command time `s` (§3.2) |
| Phases | reach `I=0 ∧ s≤δ`, hold `I=1`, detach `I=0 ∧ s>δ` (§3.2, Fig. 3) |
| Eq. 1 | `r_reach = exp(-d(p1,p_act)/σ²)·1[I=0 ∧ s≤δ]`, **d = unsquared L2 norm** |
| Eq. 2 | `r_hold = (1 + α_hold·exp(-d/σ²))·1[I_con = I_act = 1]` |
| Eq. 3 | `r_detach = 1[I_con = I_act = 0 ∧ s>δ]` |
| Sampling | once per env: stride `U(0,0.3)` & stance width `U(0.1,0.3)` per front/hind pair; heading `U[-π,π]` **or** yaw rate `U[-π,π]` rad/s; per-leg offsets `U(-0.15,0.15)`; `S ~ U[0.34,0.36]` s; contact locations "move in all directions" (§4) |
| Obs | joint pos/vel + current & next contact sequence, current & next contact locations (base frame), command duration, feet→goal distance; no foot contact sensing (§4) |
| Extra | locomotion penalties; goal update + bonus when the base stays within a threshold of the contact locations (§4) |
| Training | PPO, GRU, entropy decay, 8192 envs (IsaacLab) |
| Eval (Fig. 6) | 1000 eps × 15 s; S ∈ [0.2, 0.9]; L2 contact-location error while in contact; Hamming distance plan vs actual contact |
| Unspecified | σ, δ, α_hold, reward weights, foothold propagation, network sizes |

## 2. Discrepancies found and what was done

Category key: **A** correctness · **B** paper reproduction · **C** performance ·
**D** engineering · **E** experimental (opt-in only).

| # | Cat | Finding (original code) | Impact | Fix |
| --- | --- | --- | --- | --- |
| 1 | A | **Stale reset anchor.** `ContactGoalCommand` anchored new footholds to `root_link_pos_w` / `heading_w` inside `command_manager.reset`. mjlab calls that *after* reset events write `qpos` but *before* `sim.forward()`, so those derived (`xpos`/`xquat`) values belong to the **terminated** episode. | Every episode after the first started with footholds (and travel frame) around the previous episode's final pose, up to metres away; reach/hold rewards ≈ 0 until the goals drifted by chance. | Read root xy / yaw directly from `qpos` (`_root_pose_from_qpos`), which is current at that point. |
| 2 | A | **Front/hind footholds diverge.** Each foot marched by *its pair's* stride at lift-off; front and hind strides are sampled independently. | Pairs separate by \|s_f − s_h\| per gait cycle (E = 0.1 m); over a 20 s trot episode ≈ 2.9 m on average → infeasible goals for most of each episode. | New `GaitPlanner`: a virtual base advances `mean_stride / P` per switch; footholds are placed relative to it at mid-stance; the pair stride only shifts the landing point by ±½(s_pair − mean). Regression test `test_front_hind_footholds_do_not_diverge`. |
| 3 | A | **Stance layout never rotates on curved paths.** Footholds marched along the turning direction but kept their initial world-frame lateral offsets. | For the 50 % yaw-rate envs the "left" feet end up in front/right of the body after a turn → inconsistent goals. | Layout expressed in the (rotating) reference frame; exact constant-curvature arc integration so the look-ahead `p2` equals the next `p1` exactly. Tests `test_footholds_stay_near_reference_on_curved_paths`, `test_lookahead_goal_matches_next_goal`. |
| 4 | A/B | **Goal discovery halved the command duration.** On discovery the window was forced to expire; with the dwell gate at S/2 ≈ δ and a threshold (0.15 m) that a tracking base almost always satisfies, it fired nearly every switch at `s ≈ 0.18`. | Effective S ≈ 0.17–0.18 s instead of 0.34–0.36 s, reach phase shrinks to ~1 step, and the Fig. 6 duration sweep becomes meaningless. | Discovery now only grants the bonus (once per switch). Old behaviour: `discovery_advances_goals=True`. |
| 5 | B | Eq. 1–2 kernel implemented as `exp(-d²/std²)`; README claimed `std=1` reproduced the paper. The paper uses `exp(-d/σ²)` with unsquared d. | Different reward shape (no gradient at d→0 vs Laplacian; much thinner tails). | `proximity_kernel(kernel="l2")` default, σ² = 0.1 m (same 10 cm e-fold length); `kernel="gaussian"` kept for ablation. |
| 6 | B | Eq. 1 uses `s ≤ δ`; code used `s < δ`. | Negligible. | `phase_masks` uses `≤`. |
| 7 | B | Layout was expressed in the *travel* frame, so a straight env with heading offset θ asked the robot to rotate in place by θ first and then always walk forward. Paper: locations "move in all directions". | No omnidirectional (sideways/backwards) walking was trained. | Layout in the body-facing frame, travel = facing + sampled offset. (Documented design decision.) |
| 8 | B | No implementation of the paper's evaluation metrics; `tracking_error` logged the *last-step* distance of all feet (incl. swing) at reset. | Not comparable to Fig. 6. | Episode-averaged `contact_plan_hamming` and `contact_location_error` (in-contact feet) metrics + `contact-eval` script implementing the Fig. 6 protocol. |
| 9 | A | Entropy decay chunked `learn()` every 50 iterations: last iteration of every chunk re-run (`current_learning_iteration = it`), episode stat buffers reset per chunk, extra end-of-learn checkpoint per chunk, and a `try/except` that silently restarted training after **any** exception (e.g. NaNs / OOM). | Silent failures, duplicated iterations/log steps, skewed logged returns. | Schedule applied by wrapping `alg.update` inside a single `learn()`; resume-aware; no exception swallowing. |
| 10 | C | Per-step `.any()` host syncs in `_update_command` (yaw-rate integration and discovery branch). | Two GPU→CPU syncs per env step. | Branch-free tensor ops (`masked_fill_`, boolean algebra). |
| 11 | D | `dump_project.py`: hard-coded `C:\Users\A\...` Windows paths, unrelated to the pipeline. | Dead code. | Removed. |
| 12 | D | No tests, no evaluation or benchmarking tooling. | — | `tests/test_planning.py` (10 tests), `contact-eval`, `contact-bench`. |
| 13 | E | Actor has no IMU signals (projected gravity / base angular velocity). Paper lists observations "such as" joint pos/vel. | Likely harder to learn balance, esp. jump / bound. | Opt-in `Mjlab-Contact-Flat-Unitree-Go2-Improved` (+IMU in actor, running obs normalisation). Paper task unchanged. |

Items checked and found consistent with the paper: two-switch horizon, phase
logic, detach/hold use of the actual contact, sampling ranges, once-per-env
sampling, PPO + GRU, no foot-contact in the actor, asymmetric critic (does not
change the deployed policy), foot-order startup assertion, reward dt-scaling.

There is no dataset / train–test split in this RL setting; the relevant
"leakage" check is that the actor sees no privileged signal — verified (foot
contact, base velocities only reach the critic).

## 3. Verification actually performed (in the authoring environment)

The authoring sandbox had **no GPU, no internet and no torch/mjlab install**,
so no simulation, training, or throughput measurement could be run there.

* All changed Python files byte-compile.
* `tests/test_planning.py` (10 tests) was executed against a numpy-backed torch
  stand-in: **10/10 pass** — look-ahead consistency, stance-goal constancy,
  bounded front/hind separation, rotating layout, one-mean-stride-per-cycle,
  phase masks, Eq. 1 kernel, quaternion yaw, arc additivity.
* Not executed: env construction, training, checkpoint/ONNX, evaluation,
  benchmarks. Run the commands below to verify on the target GPU.

## 4. Reproduce

```bash
uv sync --extra cu128 --python 3.12 --frozen
export UV_NO_SYNC=1        # a plain `uv run` would re-sync without cu128 and swap torch
uv run --with pytest pytest tests -q                       # unit tests
uv run contact-bench --num-envs 2048 4096 8192             # hardware + throughput
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.scene.num-envs 8192 \
    --agent.logger tensorboard                             # paper task
uv run contact-train Mjlab-Contact-Flat-Unitree-Go2-Improved --env.scene.num-envs 8192 \
    --agent.logger tensorboard                             # experimental variant
uv run contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 \
    --checkpoint runs/go2_contact/<run>:best                # Fig. 6 (or .../checkpoints/model_<it>.pt)
```

Checkpoints are written by `contact-train` to
`runs/<experiment>/<timestamp>/checkpoints/` (`model_<it>.pt`, `latest.pt`,
`best.pt`, `index.json`); the `logs/rsl_rl/...` layout used by earlier
versions of this document is only produced by a multi-GPU hand-off to mjlab's
launcher. Seed: `agent.seed = 42` (override with `--agent.seed`). Pre-audit
baseline: `git checkout 23ad625`.

## 5. Second pass: VPS / runtime robustness

Engineering fixes (no change to the task, rewards or training algorithm):

| # | Finding | Fix |
| --- | --- | --- |
| S1 | SIGTERM (`systemctl --user stop`) while the env was being built killed `contact-train` without running any cleanup, leaving `run_info.json` at `status: "starting"`; a first Ctrl-C at that point was recorded as a hard abort (130). | `StartupSignalGuard` is installed at the start of `run_contact_train`: the first SIGINT / SIGTERM raises `StartupStop` (a `BaseException`, so library `except Exception` blocks cannot swallow it), the run is finalised as `stopped` with exit 0; repeated signals are ignored. The final `run_info.json` write and env close run with SIGINT / SIGTERM ignored, so the status is written exactly once (`finalize_run_info`, atomic `write_json`). |
| S2 | Plain `uv run` re-synced the env without `--extra cu128` and replaced the CUDA torch build. | `UV_NO_SYNC=1` exported by `scripts/vps/common.sh`, forwarded into tmux sessions, set in all systemd units; `setup.sh` is the only sync (`EXTRA=cpu` supported). |
| S3 | VPS wrappers took a leading option as checkpoint / run / logdir (`play.sh --num-envs 16`); `train_tmux.sh` started without a task. | Optional first argument only if it does not start with `-`; `train_tmux.sh` exits 2 without a task. |
| S4 | systemd user units depended on `network-online.target` (not provided by user managers), buffered stdout. | Dependency removed; `PYTHONUNBUFFERED=1`; TensorBoard unit gets `TimeoutStopSec` / `RestartSec`. |
| S5 | Docs: `--gpu-ids 0 1` (invalid with mjlab's `UsePythonSyntaxForLiteralCollections`), stale `logs/rsl_rl` checkpoint paths. | `--gpu-ids "[0, 1]"`, `runs/<exp>/<run>/checkpoints/` everywhere. |

**rsl-rl 5.0.1 compatibility (source review, not executed).** The project
runner and entry points were compared against upstream `rsl-rl` tag `v5.0.1`
(`runners/on_policy_runner.py`, `utils/logger.py`, `algorithms/ppo.py`,
`models/rnn_model.py`) and the vendored mjlab runner: `OnPolicyRunner(env,
train_cfg, log_dir, device)`, `load(path, load_cfg, strict, map_location)`
(with `load_cfg=None` the iteration is restored, which `load_for_resume`
relies on before continuing at `iter + 1`), `current_learning_iteration`
being set to `it` after each update (the stop checkpoint therefore stores
`iter = it`), the keyword-only `Logger.log(it=..., collect_time=...,
learn_time=..., loss_dict=..., learning_rate=...)` call wrapped by the runner,
the lazily created `logger.writer` / `logger_type`, `rewbuffer` / `lenbuffer`,
`stop_logging_writer`, `alg.entropy_coef`, `alg.storage.values / returns`,
`alg.save()` keys (`actor_state_dict`, `critic_state_dict`,
`optimizer_state_dict`) and the GRU `_OnnxRNNModel.forward` returning
`(actions, h, None)` with two output names (handled by `onnx_compat`). No
incompatibility was found. This is a static check only.

Verification of this pass: see `docs/VPS_READINESS_STATUS.md`.
