# Liquid AI runtime (optional, runtime only)

LiquidAI **d1-3B-w8a8** makes *high-level* decisions from a natural-language
instruction; the trained PPO/GRU policy keeps producing every joint action with
its original observation and action spaces. Nothing in training, rewards,
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

Liquid AI assisted (the extra packages are added for this run only, `uv.lock`
is not changed):

```bash
uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow \
  contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best --liquid True --liquid-stdin True
```

Open the viewer (SSH tunnel as usual), "Liquid AI" folder: type an instruction,
press **Send to Liquid AI**. **STOP (no model)** applies speed 0 / yaw 0
immediately and cancels any in-flight decision. With `--liquid-stdin True` the
same works from the terminal (`stop` = the deterministic stop). Examples:
"Walk forward slowly.", "Increase speed slightly.", "Turn left.", "Turn right and
continue walking.", "Stop moving.". Decisions apply to all envs or the selected
env according to the existing "Apply to all envs" checkbox. The model loads in
the background; the policy keeps running on its current command meanwhile and
if loading fails (status line shows the reason).

Decisions are event-triggered: one inference per instruction, never per frame
and never periodically (relative instructions like "faster" would compound).

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
median/max warm latency and the answers to the five example instructions. For
sim impact compare "Actual RT" in the viewer Info panel with and without
`--liquid True` while sending instructions.

## Tests

```bash
uv run pytest tests/test_runtime_decision.py tests/test_liquid_runtime_integration.py
uv run pytest            # whole suite
uv run ruff check src tests
```

Model calls are mocked; passing tests do not prove the real model works.

## Limitations / troubleshooting

* Requires the Viser viewer (`--viewer viser`); refused with the native viewer.
* `reduce-overhead` compile uses CUDA graphs recorded on the worker thread; load
  and all calls stay on that thread. If compile fails, try `--liquid-compile False`.
* The model shares the GPU with MuJoCo-Warp; expect some sim slowdown during the
  first (compiling) call.
* "Liquid AI unavailable: missing optional dependency" -> use the `uv run --with ...` command above.
* "needs an Ampere+ GPU" -> different GPU, or explicitly `--liquid-model LiquidAI/d1-3B`.
