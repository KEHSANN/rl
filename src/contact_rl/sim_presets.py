"""Simulation parameter presets for the contact-explicit locomotion task.

Why this module exists
----------------------
Every MuJoCo / MJWarp knob used to live as a literal inside
``contact_env_cfg.py``. That made performance experiments destructive: changing
a value to try it out silently changed the physics of *every* subsequent run,
including reruns of old experiments. Presets make the choice explicit, named,
versioned and reversible.

The presets
-----------
``baseline``
  Byte-for-byte the values this repository trained with before the preset
  system existed. **Never change these.** Every optimisation is measured
  against this preset, and any checkpoint produced before the preset system
  was introduced was produced under exactly these numbers.

``safe``
  ``baseline`` plus the single change that is provably safe for *this* model
  (see ``contact_sensor_maxmatch`` below). Physics is bit-identical to
  ``baseline``; only the contact-sensor kernel does less scanning work.
  Use this if ``optimized`` ever shows artefacts.

``optimized``
  ``safe`` plus reduced ``njmax`` and ``ccd_iterations``. These two values are
  **provisional** (``measured=False``) and derived from static analysis of
  ``go2.xml``, not from measurement. Run::

      uv run contact-benchmark --preset baseline --collect-stats

  and then::

      uv run contact-benchmark --verify-margins --preset optimized

  before trusting them for a real training run. The benchmark prints the
  measured maxima and the values it recommends.

How to select a preset
----------------------
Three ways, in increasing precedence:

1. Default: ``baseline``.
2. Environment variable, which works through mjlab's task *registry* (the
   registry is populated at import time, so this is the only way to change the
   preset for ``contact-train`` without editing code)::

       CONTACT_RL_SIM_PRESET=optimized uv run contact-train Mjlab-Contact-Flat-Unitree-Go2

3. mjlab's tyro CLI, which overrides individual fields after the registry is
   built and therefore wins over both of the above::

       uv run contact-train Mjlab-Contact-Flat-Unitree-Go2 --env.sim.njmax 128

The resolved preset name is recorded in ``logs/.../params/env.yaml`` by way of
the ``SimulationCfg`` values themselves, so an experiment is always
reconstructable from its log directory.

What is deliberately NOT tunable here
-------------------------------------
``timestep`` (0.005) and the env's ``decimation`` (4) define the paper's 50 Hz
control rate and the physics fidelity the reward functions were tuned against.
They are identical in every preset and must stay that way: changing either one
changes the MDP, invalidates every existing checkpoint, and rescales every
dt-scaled reward weight. Same for the solver block (``solver``, ``iterations``,
``ls_iterations``, ``tolerance``, ``cone``, ``impratio``) -- Phase 1's rule is
that solver parameters change only after benchmarking proves a need.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from mjlab.sim import MujocoCfg, SimulationCfg

__all__ = [
  "ENV_VAR",
  "DEFAULT_PRESET",
  "PRESETS",
  "PARAMETER_DOCS",
  "SimPreset",
  "get_preset",
  "make_sim_cfg",
  "resolve_preset_name",
  "describe_presets",
]

ENV_VAR = "CONTACT_RL_SIM_PRESET"
"""Environment variable read by :func:`resolve_preset_name`."""

DEFAULT_PRESET = "baseline"
"""Preset used when nothing else is specified. Must stay ``baseline``."""


@dataclass(frozen=True)
class SimPreset:
  """A named, fully explicit simulation configuration.

  Every field is spelled out rather than left to a dataclass default, so that
  reading one preset tells you the complete physics configuration without
  having to cross-reference mjlab's defaults. See :data:`PARAMETER_DOCS` for
  what each field does and why it is set the way it is.
  """

  name: str
  description: str

  # --- Frozen across all presets: defines the MDP. Do not vary. ---
  timestep: float = 0.005

  # --- Frozen across all presets: solver. Phase 1 forbids blind changes. ---
  integrator: str = "implicitfast"
  solver: str = "newton"
  cone: str = "pyramidal"
  jacobian: str = "auto"
  impratio: float = 1.0
  iterations: int = 10
  tolerance: float = 1e-8
  ls_iterations: int = 20
  ls_tolerance: float = 0.01
  ls_parallel: bool = True

  # --- Tunable under Phase 1, only with measurements. ---
  ccd_iterations: int = 50
  multiccd: bool = False
  njmax: int | None = 300
  nconmax: int | None = None
  contact_sensor_maxmatch: int = 64

  # --- Provenance. ---
  measured: bool = True
  """``False`` means at least one tunable value is a provisional estimate that
  has not yet been validated against measured headroom on real hardware.
  ``make_sim_cfg`` emits a warning when an unmeasured preset is used."""

  def to_sim_cfg(self) -> SimulationCfg:
    """Materialise this preset as an mjlab :class:`SimulationCfg`."""
    return SimulationCfg(
      nconmax=self.nconmax,
      njmax=self.njmax,
      ls_parallel=self.ls_parallel,
      contact_sensor_maxmatch=self.contact_sensor_maxmatch,
      mujoco=MujocoCfg(
        timestep=self.timestep,
        integrator=self.integrator,  # type: ignore[arg-type]
        impratio=self.impratio,
        cone=self.cone,  # type: ignore[arg-type]
        jacobian=self.jacobian,  # type: ignore[arg-type]
        solver=self.solver,  # type: ignore[arg-type]
        iterations=self.iterations,
        tolerance=self.tolerance,
        ls_iterations=self.ls_iterations,
        ls_tolerance=self.ls_tolerance,
        ccd_iterations=self.ccd_iterations,
        multiccd=self.multiccd,
      ),
    )

  def diff_against(self, other: "SimPreset") -> dict[str, tuple[object, object]]:
    """Return ``{field: (other_value, self_value)}`` for differing fields."""
    skip = {"name", "description", "measured"}
    out: dict[str, tuple[object, object]] = {}
    for f in fields(self):
      if f.name in skip:
        continue
      mine = getattr(self, f.name)
      theirs = getattr(other, f.name)
      if mine != theirs:
        out[f.name] = (theirs, mine)
    return out


##
# The presets.
##

_BASELINE = SimPreset(
  name="baseline",
  description=(
    "The values this project trained with before presets existed. Reference "
    "point for every benchmark comparison. Do not change."
  ),
  # Reproduces the original literal block in contact_env_cfg.py:
  #   SimulationCfg(nconmax=None, njmax=300, contact_sensor_maxmatch=64,
  #                 mujoco=MujocoCfg(timestep=0.005, iterations=10,
  #                                  ls_iterations=20, ccd_iterations=50))
  # All other fields below match mjlab's own defaults for those cfgs, so this
  # preset is value-identical to the pre-preset configuration.
  timestep=0.005,
  iterations=10,
  ls_iterations=20,
  ccd_iterations=50,
  njmax=300,
  nconmax=None,
  contact_sensor_maxmatch=64,
  measured=True,
)

_SAFE = SimPreset(
  name="safe",
  description=(
    "Baseline physics, bit-identical, with only the contact-sensor match "
    "budget trimmed. The safe fallback if `optimized` shows artefacts."
  ),
  timestep=_BASELINE.timestep,
  iterations=_BASELINE.iterations,
  ls_iterations=_BASELINE.ls_iterations,
  # Unchanged from baseline -- this preset touches no physics.
  ccd_iterations=50,
  njmax=300,
  nconmax=None,
  # 16 is 4x the theoretical worst case for this model (see PARAMETER_DOCS).
  # This does not affect physics at all: maxmatch only bounds how many contact
  # candidates the *sensor* kernel scans per sensor.
  contact_sensor_maxmatch=16,
  measured=True,
)

_OPTIMIZED = SimPreset(
  name="optimized",
  description=(
    "PROVISIONAL. Safe preset plus reduced njmax and ccd_iterations. Verify "
    "with `contact-benchmark --verify-margins` before a real training run."
  ),
  timestep=_BASELINE.timestep,
  iterations=_BASELINE.iterations,
  ls_iterations=_BASELINE.ls_iterations,
  ccd_iterations=20,
  njmax=128,
  nconmax=None,
  contact_sensor_maxmatch=8,
  # Provisional: njmax and ccd_iterations are static-analysis estimates.
  measured=False,
)

PRESETS: dict[str, SimPreset] = {
  p.name: p for p in (_BASELINE, _SAFE, _OPTIMIZED)
}


##
# Parameter documentation (Phase 3: "document every parameter").
##

PARAMETER_DOCS: dict[str, str] = {
  "timestep": (
    "Physics integration step, seconds. FROZEN at 0.005. With the env's "
    "decimation=4 this gives the paper's 50 Hz control rate. Changing it "
    "changes the MDP, rescales every dt-scaled reward weight, and invalidates "
    "existing checkpoints."
  ),
  "integrator": (
    "FROZEN at 'implicitfast'. Semi-implicit integration of joint damping; "
    "markedly more stable than 'euler' for stiff position actuators like the "
    "Go2's, at essentially no extra cost."
  ),
  "solver": (
    "FROZEN at 'newton'. Constraint solver. Newton converges in far fewer "
    "iterations than 'cg'/'pgs' for contact-rich scenes, which is why "
    "iterations=10 suffices."
  ),
  "cone": (
    "FROZEN at 'pyramidal'. Friction cone approximation. Cheaper than "
    "'elliptic' and standard for quadruped locomotion. Switching to elliptic "
    "would change foot slip behaviour and therefore the hold reward."
  ),
  "jacobian": (
    "FROZEN at 'auto'. Lets MuJoCo pick dense vs sparse constraint Jacobians "
    "based on model size. The Go2 is small (19 DoF), so this resolves to "
    "dense, which is the fast path on GPU."
  ),
  "impratio": (
    "FROZEN at 1.0. Ratio of frictional-to-normal constraint impedance. "
    "Raising it reduces foot slip but stiffens the solve; it interacts "
    "directly with the hold reward, so it is not a performance knob."
  ),
  "iterations": (
    "FROZEN at 10. Newton solver iterations per step. This is the single "
    "biggest solver cost multiplier, and also the one most likely to silently "
    "degrade contact quality. Do not reduce without measuring foot "
    "penetration and the hold-reward trace side by side against baseline."
  ),
  "tolerance": "FROZEN at 1e-8. Solver convergence tolerance.",
  "ls_iterations": (
    "FROZEN at 20. Line-search iterations inside each Newton iteration. "
    "Total line-search work is iterations * ls_iterations, so this is a "
    "tempting target -- and for the same reason a risky one."
  ),
  "ls_tolerance": "FROZEN at 0.01. Line-search convergence tolerance.",
  "ls_parallel": (
    "FROZEN at True (mjlab's default, its comment: 'Boosts perf quite "
    "noticeably'). Parallelises the line search across warps."
  ),
  "ccd_iterations": (
    "TUNABLE. Iterations for convex collision detection (GJK/EPA), used for "
    "geom pairs MuJoCo cannot solve analytically. go2.xml contains NO "
    "collision meshes -- every collision geom is a primitive (box, cylinder, "
    "sphere) and the terrain is a plane -- and contype=1/conaffinity=0 "
    "disables self-collision, so the robot only ever collides with the "
    "ground. Most of those pairs have analytic solutions. 50 is mjlab's "
    "default, chosen for mesh-heavy scenes, and is very likely unnecessary "
    "here. Verify by checking for foot penetration (contact dist) and "
    "unchanged hold reward, not just by watching throughput."
  ),
  "multiccd": (
    "FROZEN at False. Multi-point CCD produces more contact points per pair. "
    "Enabling it would increase both contact count and constraint count."
  ),
  "njmax": (
    "TUNABLE, and the riskiest value in this file. Per-world constraint-row "
    "budget. Constraint arrays are batched by world, so no world may exceed "
    "njmax. Static worst case for the Go2 on flat ground: 4 feet at "
    "condim=3 (12 rows) + up to 11 other collision geoms at condim=1 during "
    "a fall (11 rows) + active joint-limit rows (<=12) ~= 35, with p95 in "
    "normal locomotion far lower. 300 is generous. The danger: overflow does "
    "NOT raise -- constraints are silently dropped, feet sink through the "
    "floor, and the hold/reach rewards quietly corrupt, which you may only "
    "notice thousands of iterations later. Always keep >=2x measured max, "
    "and measure across ALL FIVE gaits with pushes enabled."
  ),
  "nconmax": (
    "Left at None (mjlab heuristic) in every preset. Total contact budget, "
    "pooled across worlds rather than per-world, so it is both less "
    "memory-critical and harder to reason about than njmax. Not worth the "
    "risk for the return."
  ),
  "contact_sensor_maxmatch": (
    "TUNABLE, and the safest win available. Bounds how many contact "
    "candidates each MuJoCo contact sensor scans before reduction. This task "
    "has exactly one contact sensor ('feet_ground_contact'): 4 primary foot "
    "geoms, secondary = the 'terrain' body, num_slots=1, reduce='netforce'. "
    "Each foot is a single sphere of radius 0.022 and the terrain is one "
    "plane geom, so a foot-ground pair yields exactly ONE contact. 64 is "
    "mjlab's default and scans ~64 slots to find 1 match. Crucially this is "
    "NOT a physics parameter -- it cannot change the simulation, only what "
    "the sensor reports. And it is directly measurable: ContactSensor's "
    "'found' field is documented as the match count BEFORE reduction, so "
    "max(found) over a run is the exact match count needed. If ever "
    "undershot, 'found' saturates, I_act goes wrong, and the hold reward "
    "collapses -- so keep a real margin (safe=16, optimized=8 vs. an "
    "expected max of 1)."
  ),
}


##
# Accessors.
##


def resolve_preset_name(name: str | None = None) -> str:
  """Resolve a preset name from an explicit argument, the env var, or default.

  Precedence: explicit ``name`` > ``$CONTACT_RL_SIM_PRESET`` > ``baseline``.
  """
  candidate = name or os.environ.get(ENV_VAR) or DEFAULT_PRESET
  candidate = candidate.strip().lower()
  if candidate not in PRESETS:
    raise ValueError(
      f"Unknown simulation preset {candidate!r}. "
      f"Available: {sorted(PRESETS)}. "
      f"(Set via the {ENV_VAR} environment variable or the sim_preset argument.)"
    )
  return candidate


def get_preset(name: str | None = None) -> SimPreset:
  """Return the :class:`SimPreset` selected by :func:`resolve_preset_name`."""
  return PRESETS[resolve_preset_name(name)]


def make_sim_cfg(name: str | None = None, quiet: bool = False) -> SimulationCfg:
  """Build a :class:`SimulationCfg` from the selected preset.

  Prints the resolved preset and, for non-baseline presets, the exact diff
  against baseline -- so a training log always records which physics ran.
  Warns loudly when an unverified (``measured=False``) preset is used.
  """
  preset = get_preset(name)

  if not quiet:
    print(f"[INFO] Simulation preset: {preset.name} -- {preset.description}")
    if preset.name != DEFAULT_PRESET:
      delta = preset.diff_against(_BASELINE)
      if delta:
        changes = ", ".join(
          f"{k}: {base!r} -> {new!r}" for k, (base, new) in sorted(delta.items())
        )
        print(f"[INFO] Diff vs baseline: {changes}")
    if not preset.measured:
      print(
        f"[WARN] Preset {preset.name!r} contains PROVISIONAL values that have "
        f"not been validated against measured headroom on this hardware. "
        f"Run `contact-benchmark --preset baseline --collect-stats` then "
        f"`contact-benchmark --verify-margins --preset {preset.name}` before "
        f"trusting it for a long training run."
      )

  return preset.to_sim_cfg()


def describe_presets() -> str:
  """Human-readable table of every preset and how it differs from baseline."""
  lines: list[str] = []
  for name, preset in PRESETS.items():
    flag = "" if preset.measured else "  [PROVISIONAL]"
    lines.append(f"{name}{flag}")
    lines.append(f"  {preset.description}")
    if name == DEFAULT_PRESET:
      lines.append("  (reference preset)")
    else:
      delta = preset.diff_against(_BASELINE)
      if not delta:
        lines.append("  identical to baseline")
      for key, (base, new) in sorted(delta.items()):
        lines.append(f"  {key}: {base!r} -> {new!r}")
    lines.append("")
  return "\n".join(lines)


if __name__ == "__main__":
  print(describe_presets())
