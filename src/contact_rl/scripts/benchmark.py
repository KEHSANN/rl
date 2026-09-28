"""``contact-bench``: hardware report + environment-throughput sweep.

Prints the GPU / CUDA / CPU / RAM of the machine and measures raw environment
throughput (physics + all managers, random actions, no learning) for a sweep of
``num_envs``. Use it to pick the largest ``--env.scene.num-envs`` that still
scales on your GPU (the paper uses 8192; 4096 is the lower-VRAM fallback)
before launching a long run.

A size that runs out of GPU memory is reported as ``OOM`` and the sweep
continues with the next size (the env is always closed and the CUDA cache
emptied). The exit code is 1 only if every size failed.

    uv run contact-bench --num-envs 1024 2048 4096 8192
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
import tyro


@dataclass
class BenchConfig:
  task: str = "Mjlab-Contact-Flat-Unitree-Go2"
  num_envs: tuple[int, ...] = (1024, 2048, 4096, 8192)
  warmup_steps: int = 50
  steps: int = 200
  device: str | None = None


def hardware_report(device: str) -> None:
  print("=" * 60)
  print(f"python {platform.python_version()}  torch {torch.__version__}")
  print(f"cpu: {platform.processor() or platform.machine()}  cores: {os.cpu_count()}")
  try:
    with open("/proc/meminfo") as f:
      kb = int(f.readline().split()[1])
    print(f"ram: {kb / 1024**2:.1f} GiB")
  except OSError:
    pass
  if device.startswith("cuda") and torch.cuda.is_available():
    p = torch.cuda.get_device_properties(device)
    print(f"gpu: {p.name}  vram: {p.total_memory / 1024**3:.1f} GiB  "
          f"sm: {p.multi_processor_count}  cuda: {torch.version.cuda}")
  else:
    print("gpu: none (CPU run -- numbers are not representative)")
  print("=" * 60)


def is_oom(e: BaseException) -> bool:
  """CUDA / Warp out-of-memory, recognised by type or message."""
  oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
  if oom_cls is not None and isinstance(e, oom_cls):
    return True
  msg = str(e).lower()
  return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def bench_one(cfg: BenchConfig, num_envs: int, device: str) -> dict:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs = num_envs
  env_cfg.commands["contact"].debug_vis = False
  cuda = device.startswith("cuda")
  env = None
  try:
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    act_dim = env.action_manager.total_action_dim

    def run(n):
      for _ in range(n):
        env.step(2.0 * torch.rand(num_envs, act_dim, device=device) - 1.0)

    run(cfg.warmup_steps)  # includes Warp kernel compilation / graph capture
    if cuda:
      torch.cuda.synchronize(device)
      torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    run(cfg.steps)
    if cuda:
      torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if cuda else float("nan")
  finally:
    if env is not None:
      try:
        env.close()
      except Exception as e:  # noqa: BLE001
        print(f"[WARN] env close failed: {e}")
  return {
    "num_envs": num_envs,
    "step_ms": 1000.0 * dt / cfg.steps,
    "env_steps_per_s": num_envs * cfg.steps / dt,
    "torch_peak_gib": peak,
  }


def main() -> None:
  import contact_rl

  contact_rl.require_tasks()  # registers the tasks (explicit error if broken)
  from mjlab.utils.torch import configure_torch_backends

  from contact_rl.utils.runtime import resolve_device, set_egl_device_for

  cfg = tyro.cli(BenchConfig)
  configure_torch_backends()
  device = resolve_device(cfg.device)
  set_egl_device_for(device)
  hardware_report(device)
  print(f"{'num_envs':>9} {'step_ms':>9} {'env-steps/s':>13} {'torch GiB':>10}")
  ok = 0
  for n in cfg.num_envs:
    try:
      r = bench_one(cfg, n, device)
    except Exception as e:  # noqa: BLE001
      if not is_oom(e):
        raise
      print(f"{n:>9} {'OOM':>9} {'-':>13} {'-':>10}   ({type(e).__name__})")
      if device.startswith("cuda"):
        torch.cuda.empty_cache()
      continue
    ok += 1
    print(f"{r['num_envs']:>9} {r['step_ms']:>9.2f} {r['env_steps_per_s']:>13.0f} "
          f"{r['torch_peak_gib']:>10.2f}")
  print("PPO iteration time ~= num_steps_per_env (24) x step_ms + update time.")
  if ok == 0:
    print("[contact-bench] every num_envs size ran out of memory.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
