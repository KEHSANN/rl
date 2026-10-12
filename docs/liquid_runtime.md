# Liquid AI runtime (optional, runtime only)

LiquidAI **d1-3B-w8a8** makes *high-level* decisions from a natural-language
instruction **and, in closed-loop mode, from ongoing robot feedback**; the
trained PPO/GRU policy keeps producing every joint action with its original
observation and action spaces.

```
instruction + robot state -> d1 system_one() (local) -> validated high-level action
  -> apply_to_command_term -> GaitPlanner (gait timing, footholds, contact goals)
  -> unchanged GRU policy (50 Hz) -> IK/actuators -> MuJoCo-Warp Go2
  -> RobotFeedback snapshot -> next d1 decision (<= 1 Hz by default)
```

Two execution modes (`--liquid-mode`):

| mode | what the model does | when it is asked |
| --- | --- | --- |
| `closed_loop` (default) | interprets the instruction ONCE into a persistent mission, then repeatedly decides the next action from fresh feedback | once per instruction + periodic ticks while the mission is active |
| `oneshot` (original) | translates one instruction into one command | once per instruction, never again | Nothing in training, rewards,
observations, the network, checkpoints or `contact-train` / `contact-eval` is
touched. Without `--liquid True`, `contact-play` behaves exactly as before and
never imports the runtime package or `transformers`.

## Data flow (actual integration points)

```
Viser "Liquid AI" text box / stdin
  -> BaseViewer action queue (request_action, CUSTOM "liquid_cmd")
  -> sim thread: PlannerState snapshot (describe_command, plain floats)
  -> DecisionLoop worker thread (single-slot, latest-wins) -> d1 system_one()
  -> answers_to_raw (labels -> numbers computed from the planner state)
  -> decision_schema.validate (types, ranges, gaits, rate limits)
  -> action queue (CUSTOM "liquid_result")
  -> sim thread: stale/superseded check
  -> command_override.apply_to_command_term (the existing interactive entry point)
  -> GaitPlanner params -> contact goals -> unchanged trained policy -> MuJoCo-Warp
```

| Piece | File |
| --- | --- |
| Decision contract + validation | `src/contact_rl/runtime/decision_schema.py` |
| d1 adapter + benchmark | `src/contact_rl/runtime/liquid_decider.py` |
| Non-blocking worker | `src/contact_rl/runtime/decision_loop.py` |
| Viewer glue (sim-thread handlers, stdin) | `src/contact_rl/runtime/viewer_bridge.py` |
| GUI panel, dispatch (opt-in `liquid=` kwarg) | `src/contact_rl/play_viewer.py` |
| CLI flags (`--liquid ...`) | `src/contact_rl/scripts/play.py` |

The worker thread never reads or writes simulation tensors. Speed/heading/yaw
changes take effect at the next contact switch (no re-anchor); gait changes
re-anchor the plan, as the existing "Apply command" button does.

## Model interface (from the official model card)

`AutoModel.from_pretrained("LiquidAI/d1-3B-w8a8", trust_remote_code=True).to("cuda")`,
`model.compile(mode="reduce-overhead")`, then `model.system_one(state, questions)`
with Decision Index questions (`type: "choice"`, `instructions`, `criteria`).
It returns `{"answers": {name: {"choice", "confidence", "probabilities"}}, "usage": ...}`
in one forward pass with zero output tokens; it is not a chat model. Four
choice questions are asked per instruction: `speed`
(keep/stop/slow/medium/fast/faster/slower), `turn` (keep/straight/left/right),
`direction` (keep/forward/backward/left/right) and `gait` (keep + allowed gaits).
Confidence is logged only.

Requirements (model card): NVIDIA GPU with INT8 tensor cores (Ampere sm_80+),
PyTorch, `transformers>=5.19`, `torchao>=0.18`, `pillow`. On older GPUs / CPU the
card recommends `LiquidAI/d1-3B`; the adapter refuses to start and says so, it
never swaps models on its own (`--liquid-model LiquidAI/d1-3B` is an explicit opt-in).
The model card reports ~7.9 GB peak process memory and 45 ms for three questions
on a Jetson AGX Orin; **not measured on this project's hardware yet.**

## Running

Policy only (unchanged):

```bash
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best
```

Liquid AI assisted, closed-loop (the extra packages are added for this run
only, `uv.lock` is not changed). One env is recommended: closed-loop missions
control and measure only the env selected when the mission starts.

```bash
uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow \
  contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best --num-envs 1 \
  --liquid True --liquid-mode closed_loop --liquid-decision-hz 1.0 \
  --liquid-target 2.0 0.0 --liquid-stdin True
```

Panel (closed loop): *Send mission to Liquid AI*, *Pause mission*,
*Start / resume mission*, *Target forward/left* + *Set target*,
*STOP (no model)*; the status shows the mode, mission (objective, phase,
state, outcome), target, latest feedback, current command, latest model
answer, whether the latest result was applied / kept / rejected (and why), the
decision period, application latency and drop counters. Stdin: `pause`,
`resume`, `target FWD LEFT`, `target clear`, `stop`.

One-shot mode (original behaviour): add `--liquid-mode oneshot`.

Headless closed-loop rollout (same runtime, no viewer; JSON report):

```bash
uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow \
  python -m contact_rl.runtime.rollout --checkpoint best --decider liquid \
  --instruction "Move toward the target and turn left." --target 1.5 0.5 --seconds 40
```

`--decider mock` replaces only `system_one` with a transparent stub (for
machines that cannot run the model); its report says so and is not evidence
about the Liquid model.

Open the viewer (SSH tunnel as usual), "Liquid AI" folder: type an instruction,
press **Send to Liquid AI**. **STOP (no model)** applies speed 0 / yaw 0
immediately and cancels any in-flight decision. With `--liquid-stdin True` the
same works from the terminal (`stop` = the deterministic stop). Examples:
"Walk forward slowly.", "Increase speed slightly.", "Turn left.", "Turn right and
continue walking.", "Stop moving.". Decisions apply to all envs or the selected
env according to the existing "Apply to all envs" checkbox. The model loads in
the background; the policy keeps running on its current command meanwhile and
if loading fails (status line shows the reason).

In `oneshot` mode decisions are event-triggered: one inference per
instruction, never per frame and never periodically. Closed-loop mode is
described next.

## Closed-loop mission mode

### What was wrong with the first implementation

`DecisionLoop` was event-triggered only: one `system_one()` call per text
instruction mapped to one planner command, after which the model never saw the
robot again. There was no mission object, no measured state in the model's
input (only the planner command), no periodic decisions, no completion /
progress / tilt logic and no reset handling. That is an instruction
translator, not a closed-loop high-level controller.

### Separation of mission and action

* **Mission** (`decision_schema.MissionSpec`, `mission.MissionTracker`): the
  instruction is interpreted once with the *mission* question set
  (`objective` walk / reach_target / stop, `speed`, `direction`, `turn`,
  `after_arrival` stop / turn_left / turn_right, `gait` keep / suitable / named,
  `caution`). Relative words ("faster") are resolved once against the speed at
  that moment into a fixed `preferred_speed` (the mission's speed ceiling).
  The instruction is never re-interpreted.
* **Action**: while the mission is active, `LiquidRuntime.on_sim_step`
  (called after every `env.step`, sim thread) takes a `RobotFeedback`
  snapshot and submits a *tick*. The tick question set is `speed`
  (keep / increase / decrease / stop), `steer` (keep / straight / left /
  right), `gait` (keep / allowed gaits) and `status` (continue / complete /
  abort). The state given to `system_one` is a JSON object built only from
  measurements: mission + phase, current command and allowed gaits, measured
  base velocity, yaw rate, body tilt, stability label, contact-plan match,
  speed tracking, target distance / bearing (if a target is configured), the
  previous action and whether it improved the mission metric, time since the
  last decision and the active safety constraints.
* Answers are mapped to actions relative to the CURRENT command
  (`increase` = +0.05 m/s, bounded by the mission ceiling, so repeated
  "increase" converges to the user's speed instead of compounding;
  `left/right` = +-`--liquid-yaw-rate`), then validated by
  `decision_schema.validate(kind="tick", speed_ceiling=...)`. A "keep" answer
  is a no-op and the planner is not touched.

Phases: `approach` -> (measured arrival within `arrive_tol_m` = 0.25 m)
-> `final_turn` (only if the instruction asked to turn at the target; speed
ceiling 0, the model steers; complete when the measured heading change reaches
90 deg - 15 deg) -> complete. "Turn left at the waypoint" is implemented as a
90 deg in-place turn (`final_turn_rad`). `walk` missions have no measurable end
and run until stopped, paused, replaced or failed. Completion is only ever
reported from measurements; a model answer `status=complete` is accepted only
if the measurements confirm it, otherwise it is ignored (logged).

### Timing (configurable, provisional)

| parameter | default | meaning |
| --- | --- | --- |
| `--liquid-decision-hz` | 1.0 | upper bound on decisions per *sim* second; the RL policy stays at 50 Hz |
| latency bound | 2 x median measured inference latency | the decision period is never shorter |
| `--liquid-decision-deadline-s` | 2.0 s | an inference running longer counts as a failure; its late answer is discarded |
| safety checks | every 5 env steps (10 Hz) | deterministic, independent of the model |

1 Hz is a conservative start: commands take effect at the next contact switch
(S = 0.35 s) and a gait cycle is 0.7-1.4 s, so the robot cannot show the
effect of a decision much faster than that. Raise it only after measuring the
real latency (benchmark below prints `recommended_decision_hz`).

### Asynchronous safety

* Every snapshot has a strictly increasing `seq`, a sim time and a wall time;
  every tick request/decision carries `mission_id`, `snapshot_seq` and the
  generation.
* Single slot: a tick is refused while any request is pending or in flight
  (`skipped_busy`), so no queue can build up.
* A result is applied only if: same generation (instructions, STOP, pause,
  resume and resets bump it), same active mission, snapshot newer than the
  last applied one, snapshot younger than 1.5 sim-s, the robot moved < 0.25 m
  and turned < 0.5 rad since the snapshot, and the planner command did not
  change since the snapshot. Otherwise it is counted as `stale`,
  `stale_mission` or `stale_state` and a fresh tick is requested.
* The worker thread never touches simulation tensors; snapshots are plain
  floats copied on the sim thread in one device->host transfer.
* Measured: inference latency, application latency (snapshot -> applied),
  and counts for applied / kept / superseded / stale / timeout / invalid /
  failed / busy-skips (viewer panel and rollout report).

### Fallback and safety (deterministic, precede every model decision)

| check | action |
| --- | --- |
| non-finite state | controlled stop, mission failed |
| tilt >= 45 deg (env terminates at 80 deg) | controlled stop, mission failed |
| tilt >= 20 deg (12 deg for cautious missions) | speed capped at 50 % of the current speed until tilt recovers; model answers above the cap are clipped |
| episode termination / env reset | in-flight decisions invalidated, mission failed, controlled stop |
| no progress: best target distance not improved by 0.1 m in 8 s (walk: < 20 % speed tracking for 8 s; final turn: < 10 deg in 8 s) | controlled stop, mission failed |
| invalid / failed / late model answer | discarded, last valid command kept |
| 3 consecutive failures, or 4 sim-s without a valid decision | controlled stop, mission failed |
| user STOP | immediate (no model), cancels in-flight work |
| model `abort` | controlled stop, mission failed |

"Controlled stop" = speed 0 / yaw 0 through the same planner path. **It is not
a stand-still**: the contact planner keeps stepping in place (there is no
standing gait in this repository).

### Localization limitation

Position and heading come from the simulator's ground-truth root state
(`root_link_pos_w`, `heading_w`). There is no state estimator or target
detection: the target is given by the user (`--liquid-target FWD LEFT`, viewer
"Target forward/left", stdin `target FWD LEFT`) relative to the robot pose at
mission start. Without a target, `reach_target` missions are refused instead
of inventing one. There is no obstacle perception or avoidance.

## Safety limits and evidence

| Limit | Default | Evidence |
| --- | --- | --- |
| Gaits | all 5 | `planning.GAITS`; `eval_fig6.csv` at the nominal S=0.35 s: fall rate <= 0.2 % for every gait (1000 episodes each). Runtime never changes S. Pace falls 20-47 % at S >= 0.5 s, irrelevant while S is fixed. |
| Speed cap | 0.8 x per-gait trained max (trot/pace/bound/jump 0.34 m/s, crawl 0.17 m/s) | Training stride U(0, 0.3) m -> v = stride/(P*S). **Provisional**: Fig. 6 sweeps command duration, not speed, so no per-speed success data exists. `--liquid-speed-fraction` can be raised to 1.0, never above. |
| Yaw rate cap | 1.0 rad/s | Training U[-pi, pi]. **Provisional**, no per-yaw-rate evaluation. Max pi. |
| Turn magnitude | 0.5 rad/s | provisional |
| Per-decision change | 0.1 m/s, 0.5 rad/s | provisional |
| Heading vs yaw | never both | training samples one or the other |
| Stop | stride 0, yaw 0 | inside training range (stride 0); the planner keeps stepping in place, there is no stand-still gait |
| Timeout | 5 s | late answers discarded |

Out-of-range model outputs are rejected, not clamped; unknown fields/gaits are
rejected; the last valid command stays in force. A gait change lowers the speed
to the new gait's cap.

## Benchmark (not run yet: needs an Ampere+ GPU)

```bash
uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow \
  python -m contact_rl.runtime.liquid_decider --n 50
```

Prints load time, model/peak VRAM, CPU max RSS, first-call (compile) time,
median/max warm latency and the answers to the five example instructions,
plus `closed_loop`: latency of the mission-interpretation and tick question
sets and `recommended_decision_hz` (= 1 / max(1 s, 2 x tick latency)).

Verified model interface (model card and its `api.py` / `modeling_d1.py`):
`system_one(state, questions, images=None)`, state = string / any JSON value;
question types `choice` (answer: `choice`, `confidence`, `probabilities`),
`noul` (P(yes)) and `score`; one forward pass, zero output tokens; the d1-3B-w8a8
needs `torch._int_mm` on INT8 tensor cores (NVIDIA Ampere+ / Jetson Orin). This
runtime uses only `choice` questions. For
sim impact compare "Actual RT" in the viewer Info panel with and without
`--liquid True` while sending instructions.

## Tests

```bash
uv run pytest tests/test_runtime_decision.py tests/test_liquid_runtime_integration.py
CONTACT_SIM_TESTS=1 uv run pytest tests/test_liquid_closed_loop_sim.py   # real env + checkpoint, mock system_one
uv run pytest            # whole suite
uv run ruff check src tests
```

Model calls are mocked; passing tests do not prove the real model works.

## Limitations / troubleshooting

* Requires the Viser viewer (`--viewer viser`); refused with the native viewer.
* Closed loop: no localization (simulator ground truth only), no obstacle
  avoidance, no verified stand-still (zero speed = stepping in place), no
  recovery behaviour after a fall (the env resets, the mission fails), a
  single tracked env per mission, the final turn is a fixed 90 deg.
* `reduce-overhead` compile uses CUDA graphs recorded on the worker thread; load
  and all calls stay on that thread. If compile fails, try `--liquid-compile False`.
* The model shares the GPU with MuJoCo-Warp; expect some sim slowdown during the
  first (compiling) call.
* "Liquid AI unavailable: missing optional dependency" -> use the `uv run --with ...` command above.
* "needs an Ampere+ GPU" -> different GPU, or explicitly `--liquid-model LiquidAI/d1-3B`.
