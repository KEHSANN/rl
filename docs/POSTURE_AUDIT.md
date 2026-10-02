# Posture audit: knee-walking diagnostics

## Status and scope

Code changes on `audit/paper-alignment` add diagnostics and tests, not a new
reward schedule or a trained policy. The pre-change snapshot is
`backup/paper-alignment-2026-10-02-pre-posture-audit` at `9835aa0`.
No Python tests, lint/type checks, GPU smoke tests or training were executed
in the authoring environment. Do not interpret committed tests as passing
results or claim that knee-walking is solved without rollout evidence.

## Current rewards (supersedes the older README posture table)

| Term | Weight per second | Definition |
| --- | ---: | --- |
| `illegal_contact` | -4 | Count of non-foot collision geoms touching terrain |
| `knee_height` | -5 | Sum over four calf origins of max(0.08 - world_z, 0) / 0.08 |
| `base_height_below` | -10 | max(0.25 - base_z, 0) |
| `base_tilt` | -1 | Squared xy components of projected gravity |

The calf body origin is the knee joint, not the calf centre of mass. At a
knee height of 0.015 m, the knee penalty is -4.0625 per second per knee.
The environment multiplies reward rates by 0.02 s. A hovering knee is still
penalised. An 8 cm threshold does not prove all crouches are unaffected, nor
that every posture above it is healthy. These definitions assume ground z=0.
No observation, action, architecture, reward weight, planner distribution or
termination condition was changed by this audit.

## New evaluation fields

`contact-eval` records these automatically in `metrics.csv` and `summary.json`.
With `--tensorboard True`, summaries also appear under `evaluation/`.

| Field | Meaning |
| --- | --- |
| `knee_FL_height_mean_m` (also FR/RL/RR) | Per-episode mean knee origin height |
| `knee_FL_height_min_m` (also FR/RL/RR) | Minimum sampled height |
| `knee_FL_below_fraction` (also FR/RL/RR) | Fraction of valid samples below the reward threshold |
| `knee_below_fraction` | Fraction with at least one low knee |
| `illegal_contact_fraction` | Fraction with any non-foot terrain contact |
| `illegal_contact_count_mean` | Mean number of touching non-foot geoms, not contact points |
| `posture_samples` | Number of valid posture samples |
| `posture_invalid_samples` | Nonterminal samples with nonfinite height/contact data |
| `knee_min_height_threshold_m` | Threshold read from this evaluation's reward config |

Summaries include episode means and pooled fractions (sums divided by valid
samples across episodes). Per-leg minimum summaries distinguish the mean of
episode minima from the minimum over all episodes.

Legacy `success`/`success_rate` still mean survival (`timeout` or `running`),
not healthy gait. All old columns and the Fig. 6 CSV layout are preserved.
Only the first episode per environment is counted. Auto-reset terminal states
are excluded, including from minima; therefore these diagnostics can miss a
contact on the terminal transition. Sampling is at policy frequency (50 Hz),
not every physics substep. Missing/invalid-only data produce NaN, not zero;
nonfinite samples are counted explicitly. NaN follows the existing output
convention; strict-JSON consumers need to handle that existing convention.

An existing `evaluations.csv` keeps its original header and therefore omits
new columns when appended to. Full posture results remain in each evaluation's
`metrics.csv`, `summary.json` and optional TensorBoard. New history files
include the new fields. No historical file is migrated or overwritten.

## Validation on the configured machine

Keep the existing CUDA environment; do not re-sync without its CUDA extra.
Install pytest/ruff/type-checker in a controlled development environment if
not already available. Run the following from the repository root:

```bash
export UV_NO_SYNC=1
uv run --no-sync python -m compileall -q src/contact_rl tests
uv run --no-sync python -m pytest tests/test_posture_rewards.py tests/test_posture_metrics.py tests/test_posture_integration.py tests/test_episode_metrics.py -q
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src/contact_rl/utils/posture_metrics.py tests/test_posture_rewards.py tests/test_posture_metrics.py tests/test_posture_integration.py
uv run --no-sync pyright src/contact_rl/utils/posture_metrics.py src/contact_rl/scripts/evaluate.py
```

Type-check results may include pre-existing untyped simulator interfaces;
review them rather than treating compilation alone as a type check. Reward
tests execute production function bodies extracted with AST to avoid loading
the GPU stack; they do not replace testing real sensor binding on the GPU.

A small GPU rollout using the tracked, old checkpoint validates wiring only:

```bash
uv run --no-sync contact-eval \
  --checkpoint runs/go2_contact/2026-10-01_04-21-51/checkpoints/model_1999.pt \
  --mode episodes --num-envs 16 --episode-s 2 \
  --out-dir runs/posture_smoke --tensorboard True
```

A longer baseline (all default gaits, training command duration) is:

```bash
uv run --no-sync contact-eval \
  --checkpoint runs/go2_contact/2026-10-01_04-21-51/checkpoints/model_1999.pt \
  --mode episodes --num-envs 256 --episode-s 15 \
  --out-dir runs/posture_baseline --video True --tensorboard True
```

In episodes mode results are aggregated as `all`. Use Fig. 6 mode for per-gait
summaries, or run each gait separately. Default Fig. 6 videos cover only its
first gait/duration; do not mistake one video for coverage of all gaits.
Use separate output directories when comparing checkpoints.

## Training experiment, not a guarantee

Compare fresh training and full-state resume with the same environment count,
command distribution, evaluation seeds and additional sample budget. Use
several seeds and evaluate every gait. Do not compare total rewards across
old and new reward definitions as if they measured the same objective.

`load_for_resume` restores full state and resumes at loaded iteration + 1.
The runner overwrites `entropy_coef` before every PPO update with its schedule:
0.01 to 0.001 over 5000 iterations. A CLI entropy override alone will not
restart exploration; this audit deliberately leaves the schedule unchanged.
Changing to the Improved task changes actor inputs and is not a compatible
substitute for resuming the default task's checkpoint.

Accept improvement only when low-knee/contact fractions decrease together
with maintained episode length, contact-plan tracking and visual gait quality.
Always inspect invalid sample counts and per-leg measurements. A low illegal
contact rate alone can mean hovering, not healthy posture. A shorter episode
can reduce logged reward penalties without fixing the gait. No universal
pass threshold is imposed here without measured gait/crouch distributions.
