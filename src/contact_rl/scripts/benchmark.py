"""Benchmark and measurement harness for the contact-explicit locomotion task.

Phase 2 of the optimisation plan. The point of this script is to make the
Phase 1 tuning decisions *measured* rather than guessed, and to make any claimed
speedup reproducible.

What it measures
----------------
* **Throughput** -- environment steps/sec, physics steps/sec, env-loop
  iterations/sec, and per-env realtime factor.
* **Training throughput** (``--mode train``) -- the same, but including the PPO
  learning update, which is the number that actually determines wall-clock time
  to a trained policy.
* **GPU** -- utilisation and VRAM, sampled in the background from
  ``nvidia-smi``, plus torch's own allocator peak.
* **CPU** -- process CPU seconds per wall second, i.e. how many cores the
  rollout is burning. On a well-behaved MJWarp run this should be low; a high
  number means the GPU is being starved by Python.
* **Contact and constraint statistics** (``--collect-stats``) -- see
  :mod:`contact_rl.sim_diagnostics`.

Quick start on the target machine
---------------------------------
::

    # 1. Baseline throughput + the measurements Phase 1 needs, all five gaits.
    uv run contact-benchmark --preset baseline --collect-stats --per-gait \\
        --out bench/baseline.json

    # 2. Check the provisional `optimized` values against measured headroom.
    uv run contact-benchmark --verify-margins --preset optimized \\
        --stats-file bench/baseline.json

    # 3. Only if step 2 passes: measure the optimised preset.
    uv run contact-benchmark --preset optimized --collect-stats --per-gait \\
        --out bench/optimized.json

    # 4. Compare.
    uv run contact-benchmark --compare bench/baseline.json bench/optimized.json

Methodology notes (read before trusting a number)
-------------------------------------------------
* **Warmup is not optional.** Warp compiles kernels on first use and mjlab
  captures CUDA graphs lazily; the first few hundred steps are dominated by
  compilation. ``--warmup`` steps are run and discarded.
* **CUDA graph capture is reported explicitly.** If mjlab falls back to
  per-kernel dispatch (old driver, mempool disabled) it only prints a warning
  and throughput drops hard, so the result JSON records whether graphs were
  active. Comparing a graph-captured run against a non-captured one is
  meaningless.
* **Every timed region is bracketed by ``torch.cuda.synchronize()``.** Without
  it you measure kernel launch time, not execution time.
* **Gait matters.** ``ContactGoalCommand`` samples gait once per env at
  construction (the paper's choice), so a default run is a random mix and its
  maxima under-report the worst gait. ``--per-gait`` runs each of trot / pace /
  bound / jump / crawl separately and reports the worst case across all of
  them. Use it for any decision about ``njmax``.
* **Pushes stay enabled.** The ``push_robot`` interval event is part of the
  training distribution and is exactly what drives peak constraint counts, so
  the benchmark deliberately does not disable it.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

# MuJoCo must be told to use EGL before any rendering-capable object is built;
# the target machine is a headless GPU server with no X display.
os.environ.setdefault("MUJOCO_GL", "egl")
# Keep wandb out of a benchmark run: an unauthenticated server stalls on init.
os.environ.setdefault("WANDB_MODE", "offline")

TASK_ID = "Mjlab-Contact-Flat-Unitree-Go2"
# Mirrors ``contact_rl.tasks.contact.mdp.contact_command.GAITS``. Duplicated as a
# literal so ``--help`` and ``--list-presets`` work without importing torch.
ALL_GAITS = ("trot", "pace", "bound", "jump", "crawl")


##
# Configuration.
##


@dataclass(frozen=True)
class BenchmarkConfig:
  """Benchmark options."""

  preset: str = "baseline"
  """Simulation preset to measure: baseline, safe, or optimized."""

  mode: Literal["env", "policy", "train"] = "policy"
  """env: random actions (pure sim throughput). policy: GRU inference + sim
  (rollout throughput). train: full PPO iterations including the update."""

  num_envs: int = 4096
  """Parallel environments. The single biggest throughput and VRAM knob on an
  RTX 3080 (10 GB); sweep it with --num-envs-sweep."""

  steps: int = 300
  """Timed environment steps (ignored in train mode, which uses --train-iters)."""

  warmup: int = 100
  """Discarded steps, to absorb Warp kernel compilation and graph capture."""

  train_iters: int = 20
  """PPO iterations to time in train mode."""

  collect_stats: bool = False
  """Collect contact / constraint statistics. Adds a small per-step cost, so
  do not mix a stats run with a pure throughput claim -- run it separately."""

  per_gait: bool = False
  """Run once per gait and report the worst case. Required for njmax decisions."""

  gaits: tuple[str, ...] | None = None
  """Restrict gait sampling to these gaits (default: all five, mixed)."""

  num_envs_sweep: tuple[int, ...] | None = None
  """Measure throughput at several env counts, e.g. 1024 2048 4096 8192."""

  device: str | None = None
  seed: int = 0
  label: str | None = None
  """Free-text label stored in the result JSON."""

  out: str | None = None
  """Write the result JSON here."""

  gpu_poll_interval: float = 0.25

  # --- Alternate entry points. ---
  compare: tuple[str, ...] = ()
  """Two or more result JSONs to compare instead of running a benchmark."""

  verify_margins: bool = False
  """Check --preset's tunable values against --stats-file's measurements.
  Exits non-zero if any margin is unsafe."""

  stats_file: str | None = None
  """Result JSON from a previous --collect-stats run, for --verify-margins."""

  list_presets: bool = False


##
# Telemetry.
##


def _nvidia_smi_available() -> bool:
  return shutil.which("nvidia-smi") is not None


def _physical_gpu_index(torch_index: int) -> int:
  """Map a torch device ordinal to the physical index nvidia-smi reports."""
  visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if not visible:
    return torch_index
  try:
    ids = [int(part) for part in visible.split(",") if part != ""]
  except ValueError:
    return torch_index
  if torch_index < len(ids):
    return ids[torch_index]
  return torch_index


class GpuSampler:
  """Background ``nvidia-smi`` poller.

  Uses a subprocess rather than pynvml on purpose: pynvml is not in this
  project's locked dependency graph, and Phase 1 is not a good reason to touch
  ``uv.lock``.
  """

  def __init__(self, gpu_index: int, interval: float = 0.25) -> None:
    self._index = gpu_index
    self._interval = interval
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._util: list[float] = []
    self._mem: list[float] = []
    self._total_mem: float | None = None
    self.available = _nvidia_smi_available()

  def _poll_once(self) -> None:
    try:
      raw = subprocess.check_output(
        [
          "nvidia-smi",
          "--query-gpu=utilization.gpu,memory.used,memory.total",
          "--format=csv,noheader,nounits",
          "-i",
          str(self._index),
        ],
        stderr=subprocess.DEVNULL,
        timeout=5,
      )
    except Exception:
      return
    line = raw.decode().strip().splitlines()
    if not line:
      return
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
      return
    try:
      self._util.append(float(parts[0]))
      self._mem.append(float(parts[1]))
      self._total_mem = float(parts[2])
    except ValueError:
      return

  def _loop(self) -> None:
    while not self._stop.wait(self._interval):
      self._poll_once()

  def start(self) -> None:
    if not self.available:
      return
    self._util.clear()
    self._mem.clear()
    self._poll_once()
    self._stop.clear()
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()

  def stop(self) -> dict[str, Any]:
    if self._thread is not None:
      self._stop.set()
      self._thread.join(timeout=2.0)
      self._thread = None
    if not self._util:
      return {"available": False}
    return {
      "available": True,
      "gpu_index": self._index,
      "samples": len(self._util),
      "util_mean_pct": sum(self._util) / len(self._util),
      "util_max_pct": max(self._util),
      "mem_used_mean_mib": sum(self._mem) / len(self._mem),
      "mem_used_max_mib": max(self._mem),
      "mem_total_mib": self._total_mem,
    }


class CpuSampler:
  """Process CPU time, via stdlib ``resource`` (no psutil dependency)."""

  def __init__(self) -> None:
    self._start: tuple[float, float] | None = None

  @staticmethod
  def _cpu_seconds() -> float:
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
      usage = resource.getrusage(who)
      total += usage.ru_utime + usage.ru_stime
    return total

  def start(self) -> None:
    self._start = (self._cpu_seconds(), time.perf_counter())

  def stop(self) -> dict[str, Any]:
    if self._start is None:
      return {}
    cpu0, wall0 = self._start
    cpu = self._cpu_seconds() - cpu0
    wall = time.perf_counter() - wall0
    return {
      "cpu_seconds": cpu,
      "wall_seconds": wall,
      "cores_used": (cpu / wall) if wall > 0 else 0.0,
      "logical_cores": os.cpu_count(),
    }


##
# Environment construction.
##


def _resolve_device(requested: str | None) -> str:
  import torch

  if requested:
    return requested
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _build_env(cfg: BenchmarkConfig, num_envs: int, gaits: Sequence[str] | None):
  """Build a training-shaped env for the given preset and gait restriction."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.torch import configure_torch_backends

  from contact_rl.sim_presets import ENV_VAR
  from contact_rl.tasks.contact.config.go2.env_cfgs import unitree_go2_flat_env_cfg

  # Also export it, so anything constructed later agrees with --preset.
  os.environ[ENV_VAR] = cfg.preset

  configure_torch_backends()

  env_cfg = unitree_go2_flat_env_cfg(play=False, sim_preset=cfg.preset)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed

  if gaits:
    # ContactGoalCommandCfg.gaits already exists for exactly this purpose, so
    # no planner code is touched.
    env_cfg.commands["contact"].gaits = tuple(gaits)

  device = _resolve_device(cfg.device)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return env, env_cfg, device


def _graph_capture_active(env) -> bool:
  sim = env.unwrapped.sim
  return bool(
    getattr(sim, "use_cuda_graph", False) and getattr(sim, "step_graph", None)
  )


##
# Single measurement run.
##


def _measure(
  cfg: BenchmarkConfig, num_envs: int, gaits: Sequence[str] | None
) -> dict[str, Any]:
  import torch

  from contact_rl.sim_diagnostics import SimStatsCollector

  env, env_cfg, device = _build_env(cfg, num_envs, gaits)
  gait_label = ",".join(gaits) if gaits else "mixed"
  print(
    f"\n[BENCH] preset={cfg.preset} mode={cfg.mode} num_envs={num_envs} "
    f"gaits={gait_label} device={device}"
  )

  graphs = _graph_capture_active(env)
  if not graphs and device.startswith("cuda"):
    print(
      "[WARN] CUDA graph capture is NOT active. Throughput will be far below "
      "what this hardware can do, and this result is not comparable with a "
      "graph-captured run. Check the driver version (>= 12.4 required) and "
      "that Warp memory pools are enabled."
    )

  result: dict[str, Any] = {
    "num_envs": num_envs,
    "gaits": list(gaits) if gaits else list(ALL_GAITS),
    "cuda_graph_capture": graphs,
  }

  runner = None
  wrapped = env
  policy = None
  tmp_log_dir: str | None = None

  if cfg.mode in ("policy", "train"):
    from dataclasses import asdict as _asdict

    from mjlab.rl import RslRlVecEnvWrapper

    from contact_rl.tasks.contact.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
    from contact_rl.tasks.contact.config.go2.runner import ContactOnPolicyRunner

    agent_cfg = unitree_go2_ppo_runner_cfg()
    # Never let a benchmark try to reach wandb.
    if hasattr(agent_cfg, "logger"):
      agent_cfg.logger = "tensorboard"  # type: ignore[attr-defined]
    agent_cfg.max_iterations = cfg.train_iters

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    tmp_log_dir = tempfile.mkdtemp(prefix="contact_bench_")
    runner = ContactOnPolicyRunner(wrapped, _asdict(agent_cfg), tmp_log_dir, device)
    policy = runner.get_inference_policy(device=device)

  gpu = GpuSampler(
    _physical_gpu_index(
      torch.cuda.current_device() if device.startswith("cuda") else 0
    ),
    cfg.gpu_poll_interval,
  )
  cpu = CpuSampler()

  try:
    if cfg.mode == "train":
      assert runner is not None
      print("[BENCH] warmup: 1 PPO iteration (kernel compile + graph capture)")
      runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
      if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

      gpu.start()
      cpu.start()
      t0 = time.perf_counter()
      runner.learn(num_learning_iterations=cfg.train_iters)
      if device.startswith("cuda"):
        torch.cuda.synchronize()
      wall = time.perf_counter() - t0

      steps_per_iter = agent_cfg.num_steps_per_env * num_envs
      env_steps = cfg.train_iters * steps_per_iter
      result.update(
        {
          "train_iters": cfg.train_iters,
          "num_steps_per_env": agent_cfg.num_steps_per_env,
          "wall_seconds": wall,
          "seconds_per_iteration": wall / cfg.train_iters,
          "env_steps": env_steps,
          "env_steps_per_s": env_steps / wall,
          "physics_steps_per_s": env_steps * env_cfg.decimation / wall,
        }
      )

    else:
      action_dim = env.action_space.shape[-1]
      actions = torch.zeros((num_envs, action_dim), device=device)

      def next_actions():
        if cfg.mode == "policy":
          assert policy is not None
          with torch.no_grad():
            return policy(wrapped.get_observations())
        return actions.uniform_(-1.0, 1.0)

      print(f"[BENCH] warmup: {cfg.warmup} steps")
      wrapped.reset()
      for _ in range(cfg.warmup):
        wrapped.step(next_actions())
      if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

      collector = SimStatsCollector(env, verbose=True) if cfg.collect_stats else None

      print(f"[BENCH] timing: {cfg.steps} steps")
      gpu.start()
      cpu.start()
      t0 = time.perf_counter()
      for _ in range(cfg.steps):
        wrapped.step(next_actions())
        if collector is not None:
          collector.record()
      if device.startswith("cuda"):
        torch.cuda.synchronize()
      wall = time.perf_counter() - t0

      env_steps = cfg.steps * num_envs
      result.update(
        {
          "steps": cfg.steps,
          "wall_seconds": wall,
          "env_steps": env_steps,
          "env_steps_per_s": env_steps / wall,
          "physics_steps_per_s": env_steps * env_cfg.decimation / wall,
          "env_loop_fps": cfg.steps / wall,
          "realtime_factor_per_env": (cfg.steps * env.step_dt) / wall,
          "stats_collected": collector is not None,
        }
      )
      if collector is not None:
        result["sim_stats"] = collector.summary()
        print()
        print(collector.report())

    result["gpu"] = gpu.stop()
    result["cpu"] = cpu.stop()
    if device.startswith("cuda"):
      result["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (
        1024**2
      )
      result["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024**2)

  finally:
    gpu.stop()
    try:
      wrapped.close()
    except Exception:
      pass
    if tmp_log_dir:
      shutil.rmtree(tmp_log_dir, ignore_errors=True)

  return result


##
# Orchestration.
##


def _environment_info(device: str) -> dict[str, Any]:
  import mujoco
  import torch

  info: dict[str, Any] = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "mujoco": mujoco.__version__,
    "device": device,
    "cuda_available": torch.cuda.is_available(),
  }
  try:
    import warp as wp

    info["warp"] = wp.config.version
  except Exception:
    pass
  try:
    import mujoco_warp

    info["mujoco_warp"] = getattr(mujoco_warp, "__version__", "unknown")
  except Exception:
    pass
  if torch.cuda.is_available():
    info["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["gpu_total_mib"] = props.total_memory / (1024**2)
    info["cuda_driver"] = torch.version.cuda
  return info


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
  from contact_rl.sim_presets import get_preset

  preset = get_preset(cfg.preset)
  device = _resolve_device(cfg.device)

  runs: list[dict[str, Any]] = []
  gait_sets: list[Sequence[str] | None]
  if cfg.per_gait:
    gait_sets = [(g,) for g in ALL_GAITS]
  elif cfg.gaits:
    gait_sets = [tuple(cfg.gaits)]
  else:
    gait_sets = [None]

  env_counts = list(cfg.num_envs_sweep) if cfg.num_envs_sweep else [cfg.num_envs]

  for num_envs in env_counts:
    for gaits in gait_sets:
      runs.append(_measure(cfg, num_envs, gaits))

  payload: dict[str, Any] = {
    "schema": 1,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "label": cfg.label or f"{cfg.preset}-{cfg.mode}",
    "preset": {
      "name": preset.name,
      "measured": preset.measured,
      "njmax": preset.njmax,
      "nconmax": preset.nconmax,
      "contact_sensor_maxmatch": preset.contact_sensor_maxmatch,
      "ccd_iterations": preset.ccd_iterations,
      "timestep": preset.timestep,
      "iterations": preset.iterations,
      "ls_iterations": preset.ls_iterations,
    },
    "mode": cfg.mode,
    "environment": _environment_info(device),
    "runs": runs,
    "worst_case": _worst_case(runs),
  }

  print()
  print(_summarize(payload))

  if cfg.out:
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[BENCH] wrote {out_path}")

  return payload


def _worst_case(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Aggregate the pessimistic view across runs -- what a safe value must cover."""
  worst: dict[str, Any] = {}
  match_max = 0
  njmax_max = 0
  njmax_p99 = 0
  contacts_max = 0
  penetration = 0.0
  saturated = False
  overflowed = False
  found_any = False

  for run in runs:
    stats = run.get("sim_stats")
    if not stats:
      continue
    found_any = True
    table = stats.get("stats", {})
    if "sensor_match_count" in table:
      match_max = max(match_max, int(table["sensor_match_count"]["max"]))
    if "constraints_per_world" in table:
      njmax_max = max(njmax_max, int(table["constraints_per_world"]["max"]))
      njmax_p99 = max(njmax_p99, int(table["constraints_per_world"]["p99"]))
    if "contacts_per_world" in table:
      contacts_max = max(contacts_max, int(table["contacts_per_world"]["max"]))
    penetration = min(penetration, float(stats.get("worst_penetration_m", 0.0)))
    warnings = stats.get("warnings", {})
    saturated = saturated or bool(warnings.get("maxmatch_saturation_steps"))
    overflowed = overflowed or bool(warnings.get("njmax_overflow_steps"))

  if not found_any:
    return {}

  from contact_rl.sim_diagnostics import recommend_maxmatch, recommend_njmax

  worst["sensor_match_count_max"] = match_max
  worst["constraints_per_world_max"] = njmax_max
  worst["constraints_per_world_p99"] = njmax_p99
  worst["contacts_per_world_max"] = contacts_max
  worst["worst_penetration_m"] = penetration
  worst["maxmatch_saturated"] = saturated
  worst["njmax_overflowed"] = overflowed
  if not saturated and match_max > 0:
    worst["recommended_contact_sensor_maxmatch"] = recommend_maxmatch(match_max)
  if not overflowed and njmax_max > 0:
    worst["recommended_njmax"] = recommend_njmax(njmax_max, njmax_p99)
  return worst


def _summarize(payload: dict[str, Any]) -> str:
  lines: list[str] = []
  lines.append("=" * 78)
  lines.append(f"BENCHMARK: {payload['label']}")
  lines.append("=" * 78)
  env = payload["environment"]
  lines.append(
    f"{env.get('gpu_name', 'cpu')} | torch {env['torch']} | "
    f"mujoco {env['mujoco']} | warp {env.get('warp', '?')}"
  )
  p = payload["preset"]
  flag = "" if p["measured"] else "  [PROVISIONAL]"
  lines.append(
    f"preset {p['name']}{flag}: njmax={p['njmax']} "
    f"maxmatch={p['contact_sensor_maxmatch']} ccd={p['ccd_iterations']}"
  )
  lines.append("")

  header = (
    f"{'envs':>7}  {'gaits':<10}{'env steps/s':>14}{'phys steps/s':>15}"
    f"{'GPU %':>8}{'VRAM MiB':>11}{'cores':>7}{'graph':>7}"
  )
  lines.append(header)
  lines.append("-" * len(header))
  for run in payload["runs"]:
    gpu = run.get("gpu", {})
    cpu = run.get("cpu", {})
    gaits = run.get("gaits", [])
    gait_label = gaits[0] if len(gaits) == 1 else "mixed"
    lines.append(
      f"{run['num_envs']:>7}  {gait_label:<10}"
      f"{run.get('env_steps_per_s', 0):>14,.0f}"
      f"{run.get('physics_steps_per_s', 0):>15,.0f}"
      f"{gpu.get('util_mean_pct', float('nan')):>8.0f}"
      f"{gpu.get('mem_used_max_mib', float('nan')):>11,.0f}"
      f"{cpu.get('cores_used', float('nan')):>7.2f}"
      f"{'yes' if run.get('cuda_graph_capture') else 'NO':>7}"
    )
  lines.append("")

  worst = payload.get("worst_case") or {}
  if worst:
    lines.append("WORST CASE ACROSS ALL RUNS")
    lines.append("-" * 78)
    lines.append(
      f"contact sensor match count : max {worst['sensor_match_count_max']}"
    )
    lines.append(
      f"constraints per world      : max "
      f"{worst['constraints_per_world_max']}, p99 "
      f"{worst['constraints_per_world_p99']}"
    )
    lines.append(
      f"contacts per world         : max {worst['contacts_per_world_max']}"
    )
    if worst["worst_penetration_m"] < 0:
      lines.append(
        f"worst penetration          : "
        f"{worst['worst_penetration_m'] * 1000:.4f} mm"
      )
    if worst.get("njmax_overflowed"):
      lines.append("!! njmax overflowed: RAISE njmax. Reward data is suspect.")
    if worst.get("maxmatch_saturated"):
      lines.append("!! contact sensor saturated: I_act and hold reward corrupt.")
    if "recommended_contact_sensor_maxmatch" in worst:
      lines.append(
        f"-> contact_sensor_maxmatch = "
        f"{worst['recommended_contact_sensor_maxmatch']}"
      )
    if "recommended_njmax" in worst:
      lines.append(f"-> njmax = {worst['recommended_njmax']}")
    lines.append("")

  lines.append("=" * 78)
  return "\n".join(lines)


##
# Comparison.
##


def _mean_throughput(payload: dict[str, Any]) -> float:
  values = [r.get("env_steps_per_s", 0.0) for r in payload.get("runs", [])]
  return (sum(values) / len(values)) if values else 0.0


def _peak_vram(payload: dict[str, Any]) -> float:
  values = [
    r.get("gpu", {}).get("mem_used_max_mib", 0.0) or 0.0
    for r in payload.get("runs", [])
  ]
  return max(values) if values else 0.0


def compare_results(paths: Sequence[str]) -> int:
  payloads = []
  for path in paths:
    data = json.loads(Path(path).read_text())
    payloads.append((Path(path).name, data))

  reference_name, reference = payloads[0]
  ref_mean = _mean_throughput(reference)
  ref_vram = _peak_vram(reference)

  print("=" * 78)
  print("BENCHMARK COMPARISON")
  print("=" * 78)
  print(f"reference: {reference_name} (preset {reference['preset']['name']})")
  print()

  header = (
    f"{'result':<28}{'preset':<12}{'mean steps/s':>14}{'delta':>9}"
    f"{'peak VRAM':>12}{'delta':>9}"
  )
  print(header)
  print("-" * len(header))

  graph_mismatch = False
  mode_mismatch = False
  ref_graphs = {r.get("cuda_graph_capture") for r in reference.get("runs", [])}

  for name, payload in payloads:
    mean = _mean_throughput(payload)
    vram = _peak_vram(payload)
    d_thr = ((mean - ref_mean) / ref_mean * 100.0) if ref_mean else 0.0
    d_vram = ((vram - ref_vram) / ref_vram * 100.0) if ref_vram else 0.0
    print(
      f"{name[:27]:<28}{payload['preset']['name']:<12}{mean:>14,.0f}"
      f"{d_thr:>8.1f}%{vram:>12,.0f}{d_vram:>8.1f}%"
    )
    if {r.get("cuda_graph_capture") for r in payload.get("runs", [])} != ref_graphs:
      graph_mismatch = True
    if payload.get("mode") != reference.get("mode"):
      mode_mismatch = True

  print()

  if mode_mismatch:
    print(
      "!! These results used different --mode values. env / policy / train "
      "measure different things and are not comparable."
    )
  if graph_mismatch:
    print(
      "!! CUDA graph capture differed between results. This dwarfs every "
      "parameter change measured here; fix the driver/mempool setup and "
      "re-run before drawing conclusions."
    )

  # Quality gate: a speedup is only real if the physics did not degrade.
  for name, payload in payloads[1:]:
    worst = payload.get("worst_case") or {}
    ref_worst = reference.get("worst_case") or {}
    if not worst or not ref_worst:
      continue
    pen = worst.get("worst_penetration_m", 0.0)
    ref_pen = ref_worst.get("worst_penetration_m", 0.0)
    if ref_pen < 0 and pen < ref_pen * 1.5:
      print(
        f"!! {name}: worst penetration {pen * 1000:.4f} mm vs baseline "
        f"{ref_pen * 1000:.4f} mm -- contact quality degraded. Reject this "
        f"configuration regardless of its throughput."
      )
    if worst.get("njmax_overflowed"):
      print(f"!! {name}: njmax overflowed. Reject.")
    if worst.get("maxmatch_saturated"):
      print(f"!! {name}: contact sensor saturated. Reject.")

  print("=" * 78)
  return 0


##
# Margin verification.
##


def verify_margins(preset_name: str, stats_path: str) -> int:
  from contact_rl.sim_diagnostics import recommend_maxmatch, recommend_njmax
  from contact_rl.sim_presets import get_preset

  payload = json.loads(Path(stats_path).read_text())
  worst = payload.get("worst_case") or {}
  if not worst:
    print(
      f"[FAIL] {stats_path} contains no statistics. Produce it with "
      f"`contact-benchmark --preset baseline --collect-stats --per-gait "
      f"--out {stats_path}`."
    )
    return 2

  preset = get_preset(preset_name)
  gaits_covered = sorted({g for run in payload.get("runs", []) for g in run["gaits"]})
  problems: list[str] = []
  notes: list[str] = []

  print("=" * 78)
  print(f"MARGIN CHECK: preset {preset.name} vs {Path(stats_path).name}")
  print("=" * 78)
  print(f"gaits covered by the measurement: {', '.join(gaits_covered)}")
  if set(gaits_covered) != set(ALL_GAITS):
    problems.append(
      f"measurement covers only {gaits_covered}; njmax decisions need all "
      f"five gaits. Re-run the baseline with --per-gait."
    )
  print()

  if worst.get("njmax_overflowed"):
    problems.append(
      "the baseline measurement itself overflowed njmax -- the numbers in it "
      "are a lower bound and the run's rewards are unreliable."
    )
  if worst.get("maxmatch_saturated"):
    problems.append(
      "the baseline measurement saturated the contact sensor -- max match "
      "count is a lower bound."
    )

  # contact_sensor_maxmatch.
  match_max = int(worst.get("sensor_match_count_max", 0))
  if match_max:
    required = recommend_maxmatch(match_max)
    ok = preset.contact_sensor_maxmatch >= required
    ratio = preset.contact_sensor_maxmatch / max(1, match_max)
    print(
      f"contact_sensor_maxmatch: preset={preset.contact_sensor_maxmatch} "
      f"observed_max={match_max} required>={required} "
      f"margin={ratio:.1f}x  {'OK' if ok else 'UNSAFE'}"
    )
    if not ok:
      problems.append(
        f"contact_sensor_maxmatch={preset.contact_sensor_maxmatch} is below "
        f"the recommended {required} for an observed max of {match_max}. "
        f"If undershot, the sensor's `found` clips, I_act goes wrong, and "
        f"the hold reward silently collapses."
      )
    elif preset.contact_sensor_maxmatch > required * 4:
      notes.append(
        f"contact_sensor_maxmatch={preset.contact_sensor_maxmatch} is far "
        f"above the required {required}; there is more headroom to reclaim."
      )

  # njmax.
  nj_max = int(worst.get("constraints_per_world_max", 0))
  nj_p99 = int(worst.get("constraints_per_world_p99", 0))
  if nj_max and preset.njmax:
    required = recommend_njmax(nj_max, nj_p99)
    ok = preset.njmax >= required
    ratio = preset.njmax / max(1, nj_max)
    print(
      f"njmax:                   preset={preset.njmax} "
      f"observed_max={nj_max} p99={nj_p99} required>={required} "
      f"margin={ratio:.1f}x  {'OK' if ok else 'UNSAFE'}"
    )
    if not ok:
      problems.append(
        f"njmax={preset.njmax} is below the recommended {required} for an "
        f"observed max of {nj_max}. Constraint overflow does not raise -- it "
        f"drops rows, feet sink, and rewards corrupt silently."
      )
  elif preset.njmax:
    notes.append(
      "no per-world constraint statistics in the measurement (this "
      "mujoco-warp build did not expose them), so njmax cannot be verified. "
      "Leave njmax at the baseline 300."
    )

  # ccd_iterations is not verifiable from counts; it needs an A/B on physics.
  if preset.ccd_iterations != 50:
    pen = float(worst.get("worst_penetration_m", 0.0))
    notes.append(
      f"ccd_iterations={preset.ccd_iterations} cannot be verified from this "
      f"file. Baseline worst penetration was {pen * 1000:.4f} mm; run this "
      f"preset with --collect-stats and compare penetration and the "
      f"hold-reward trace before accepting it."
    )

  print()
  for note in notes:
    print(f"[NOTE] {note}")
  if notes:
    print()

  if problems:
    print("RESULT: UNSAFE")
    for problem in problems:
      print(f"  - {problem}")
    print("=" * 78)
    return 1

  print(f"RESULT: preset {preset.name} has adequate margin for this measurement.")
  if not preset.measured:
    print(
      "Note: the preset is still flagged measured=False in sim_presets.py. "
      "Flip it to True once you have also confirmed locomotion quality."
    )
  print("=" * 78)
  return 0


##
# Entry point.
##


def main() -> None:
  import tyro

  cfg = tyro.cli(BenchmarkConfig)

  if cfg.list_presets:
    from contact_rl.sim_presets import describe_presets

    print(describe_presets())
    return

  if cfg.compare:
    if len(cfg.compare) < 2:
      print("--compare needs at least two result JSON files.")
      raise SystemExit(2)
    raise SystemExit(compare_results(cfg.compare))

  if cfg.verify_margins:
    if not cfg.stats_file:
      print(
        "--verify-margins needs --stats-file pointing at a result JSON from "
        "a previous `--collect-stats` run."
      )
      raise SystemExit(2)
    raise SystemExit(verify_margins(cfg.preset, cfg.stats_file))

  run_benchmark(cfg)


if __name__ == "__main__":
  main()
