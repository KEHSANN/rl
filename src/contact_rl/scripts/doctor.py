"""``contact-doctor``: is this machine ready to train / evaluate / play?

Every check reports PASS, WARN or FAIL. Exit code: 0 = no FAIL, 1 = at least
one FAIL (2 with ``--strict`` and a WARN). Heavy checks (EGL render, ffmpeg
encode, env config) run in subprocesses so a crash in a GL driver cannot take
the doctor down and the parent never imports mujoco with the wrong backend.

  uv run contact-doctor                       # default task, cuda:0
  uv run contact-doctor --device cuda:1 --run-root /data/runs --port 8080
  uv run contact-doctor --skip-env            # skip building the env (fast)
  uv run contact-doctor --json report.json
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
  section: str
  name: str
  status: str
  detail: str = ""


def _ver(name: str) -> str | None:
  try:
    return md.version(name)
  except md.PackageNotFoundError:
    return None


def _run(cmd: list[str], timeout: float = 120, env: dict | None = None) -> tuple[int, str]:
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout + p.stderr).strip()
  except FileNotFoundError:
    return 127, f"{cmd[0]} not found"
  except subprocess.TimeoutExpired:
    return 124, f"timed out after {timeout:.0f}s"


# ------------------------------------------------------------------ python


def check_python() -> list[Check]:
  out = []
  v = sys.version_info
  ok = (3, 10) <= v[:2] < (3, 14)
  out.append(Check("python", "version", PASS if ok else FAIL, sys.version.split()[0] + ("" if ok else " (need >=3.10,<3.14)")))
  in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
  out.append(Check("python", "virtualenv", PASS if in_venv else WARN, sys.prefix if in_venv else "not in a venv; use `uv run`"))
  return out


REQUIRED = {"mjlab": None, "rsl-rl-lib": "5.0.1", "mujoco": None, "mujoco-warp": "3.5.0.2", "warp-lang": None, "torch": None}
OPTIONAL = ("tensorboard", "viser", "imageio-ffmpeg", "mediapy", "onnx", "onnxruntime", "PyOpenGL", "tyro")


def check_dependencies(ver: Callable[[str], str | None] = _ver) -> list[Check]:
  out = []
  for name, pin in REQUIRED.items():
    v = ver(name)
    if v is None:
      out.append(Check("deps", name, FAIL, "not installed (run `uv sync --extra cu128`)"))
    elif pin and v != pin:
      out.append(Check("deps", name, WARN, f"{v} (project pins {pin})"))
    else:
      out.append(Check("deps", name, PASS, v))
  mj = ver("mujoco")
  if mj and not mj.startswith("3.5."):
    out.append(Check("deps", "mujoco series", FAIL, f"{mj}: mujoco-warp 3.5.0.2 needs mujoco 3.5.x"))
  for name in OPTIONAL:
    v = ver(name)
    need = name in ("tensorboard", "viser", "imageio-ffmpeg")
    out.append(Check("deps", name, PASS if v else (WARN if need else SKIP), v or "not installed"))
  return out


# --------------------------------------------------------------------- gpu


def parse_nvidia_smi(text: str) -> list[dict]:
  gpus = []
  for line in text.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 4:
      gpus.append({"index": parts[0], "name": parts[1], "driver": parts[2], "memory_mib": parts[3]})
  return gpus


def check_gpu(device: str, run=_run) -> list[Check]:
  out = []
  rc, txt = run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"], 30)
  gpus = parse_nvidia_smi(txt) if rc == 0 else []
  if gpus:
    for g in gpus:
      out.append(Check("gpu", f"GPU{g['index']}", PASS, f"{g['name']} | driver {g['driver']} | {int(float(g['memory_mib'])) / 1024:.1f} GiB"))
  else:
    out.append(Check("gpu", "nvidia-smi", FAIL if device.startswith("cuda") else WARN, txt[:200] or "no GPU found"))
  cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
  if cvd is not None:
    out.append(Check("gpu", "CUDA_VISIBLE_DEVICES", WARN if cvd == "" else PASS,
                     repr(cvd) + (" hides every GPU" if cvd == "" else " (cuda:0 = first listed GPU)")))
  try:
    import torch
  except Exception as e:  # noqa: BLE001
    out.append(Check("gpu", "torch import", FAIL, repr(e)))
    return out
  out.append(Check("gpu", "torch build", PASS if torch.version.cuda else WARN,
                   f"torch {torch.__version__}, CUDA runtime {torch.version.cuda or 'none (CPU build)'}"))
  if not torch.cuda.is_available():
    out.append(Check("gpu", "torch.cuda", FAIL if device.startswith("cuda") else WARN,
                     "not available (driver too old for this CUDA runtime, or CPU torch)"))
    return out
  idx = int(device.split(":")[1]) if ":" in device else 0
  n = torch.cuda.device_count()
  if idx >= n:
    out.append(Check("gpu", "selected device", FAIL, f"{device} requested, {n} visible"))
    return out
  props = torch.cuda.get_device_properties(idx)
  vram = props.total_memory / 1024**3
  out.append(Check("gpu", "selected device", PASS, f"{device}: {props.name}, {vram:.1f} GiB, sm_{props.major}{props.minor}"))
  try:
    x = torch.ones(1024, 1024, device=device)
    float((x @ x).sum())
    free, total = torch.cuda.mem_get_info(idx)
    out.append(Check("gpu", "CUDA kernel", PASS, f"matmul ok, free {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB"))
  except Exception as e:  # noqa: BLE001
    out.append(Check("gpu", "CUDA kernel", FAIL, repr(e)[:300]))
  # Heuristic thresholds (NOT measured for this task on every GPU): the Go2
  # task defaults to the paper's 8192 envs; measure with `contact-bench`.
  out.append(Check("gpu", "VRAM for 8192 envs (default)", PASS if vram >= 20 else WARN,
                   f"{vram:.1f} GiB" + ("" if vram >= 20 else " (heuristic; if training OOMs use --env.scene.num-envs 4096)")))
  out.append(Check("gpu", "VRAM for 4096 envs", PASS if vram >= 10 else WARN,
                   f"{vram:.1f} GiB" + ("" if vram >= 10 else " (heuristic; try --env.scene.num-envs 2048; measure with contact-bench)")))
  try:
    import warp as wp

    wp.init()
    out.append(Check("gpu", "warp", PASS, f"warp {wp.config.version}, devices {[str(d) for d in wp.get_cuda_devices()]}"))
  except Exception as e:  # noqa: BLE001
    out.append(Check("gpu", "warp", FAIL, repr(e)[:300]))
  return out


# --------------------------------------------------------------- rendering

_EGL_SNIPPET = r"""
import os, numpy as np
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size=".2" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d); img = r.render(); r.close()
assert img.shape == (64, 64, 3) and img.max() > 0, "blank frame"
print("mujoco", mujoco.__version__, "MUJOCO_GL=" + os.environ.get("MUJOCO_GL", ""), "mean", float(img.mean()))
"""


def check_rendering(device: str, run=_run) -> list[Check]:
  out = []
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
  if device.startswith("cuda"):
    env.setdefault("MUJOCO_EGL_DEVICE_ID", device.split(":")[1] if ":" in device else "0")
  out.append(Check("render", "MUJOCO_GL", PASS if env["MUJOCO_GL"] == "egl" else WARN, env["MUJOCO_GL"]))
  rc, txt = run([sys.executable, "-c", _EGL_SNIPPET], 120, env)
  out.append(Check("render", "MuJoCo headless render", PASS if rc == 0 else FAIL,
                   txt.splitlines()[-1][:300] if txt else f"exit {rc}"))
  try:
    from contact_rl.utils.video import find_ffmpeg

    exe = find_ffmpeg()
    rc, txt = run([exe, "-version"], 30)
    out.append(Check("render", "ffmpeg", PASS if rc == 0 else FAIL, f"{exe}: {txt.splitlines()[0] if txt else rc}"))
    out.append(check_encode(exe))
  except Exception as e:  # noqa: BLE001
    out.append(Check("render", "ffmpeg", FAIL, str(e)))
  return out


def check_encode(exe: str | None = None) -> Check:
  import numpy as np

  from contact_rl.utils.video import StreamingVideoWriter

  with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "t.mp4"
    try:
      w = StreamingVideoWriter(p, fps=30, ffmpeg=exe)
      for k in range(10):
        w.add(np.full((48, 64, 3), 20 * k, np.uint8))
      w.close()
      return Check("render", "MP4 encode (libx264)", PASS, f"{p.stat().st_size} bytes")
    except Exception as e:  # noqa: BLE001
      return Check("render", "MP4 encode (libx264)", FAIL, str(e)[:300])


# -------------------------------------------------------------- filesystem


def check_filesystem(run_root: Path, min_free_gib: float = 20.0) -> list[Check]:
  out = []
  for sub in ("", "_doctor/checkpoints", "_doctor/videos"):
    d = run_root / sub
    try:
      d.mkdir(parents=True, exist_ok=True)
      probe = d / ".write_test"
      probe.write_text("ok")
      probe.unlink()
      out.append(Check("fs", f"writable {d}", PASS))
    except OSError as e:
      out.append(Check("fs", f"writable {d}", FAIL, str(e)))
  shutil.rmtree(run_root / "_doctor", ignore_errors=True)
  try:
    free = shutil.disk_usage(run_root).free / 1024**3
    out.append(Check("fs", "free disk", PASS if free >= min_free_gib else (WARN if free >= 2 else FAIL), f"{free:.1f} GiB at {run_root}"))
  except OSError as e:
    out.append(Check("fs", "free disk", FAIL, str(e)))
  return out


# ---------------------------------------------------------------- network


def check_network(host: str, port: int, tb_port: int = 6006) -> list[Check]:
  from contact_rl.utils.runtime import is_loopback

  out = [Check("net", "viser bind address", PASS if is_loopback(host) else FAIL,
               f"{host}" + ("" if is_loopback(host) else " is public: viewer has no auth; use 127.0.0.1 + SSH tunnel"))]
  for name, p in (("viser port", port), ("tensorboard port", tb_port)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      try:
        s.bind(("127.0.0.1", p))
        out.append(Check("net", name, PASS, f"127.0.0.1:{p} free"))
      except OSError as e:
        out.append(Check("net", name, WARN, f"127.0.0.1:{p} in use ({e.strerror}); pass another --port"))
  if os.environ.get("SSH_CONNECTION") and not (os.environ.get("TMUX") or os.environ.get("STY")):
    out.append(Check("net", "tmux", WARN, "SSH session without tmux/screen: use scripts/vps/train_tmux.sh"))
  elif shutil.which("tmux") is None:
    out.append(Check("net", "tmux", WARN, "tmux not installed (apt install tmux)"))
  else:
    out.append(Check("net", "tmux", PASS, "available"))
  return out


# ---------------------------------------------------------------- project


def check_project(task: str, skip_env: bool, device: str, run=_run) -> list[Check]:
  from contact_rl.utils.runtime import git_info

  out = []
  g = git_info()
  if not g:
    out.append(Check("project", "git", WARN, "not a git checkout; runs will not record a commit"))
  else:
    out.append(Check("project", "git commit", PASS, f"{g['commit'][:10]} ({g.get('branch')})"))
    out.append(Check("project", "working tree", WARN if g["dirty"] else PASS, "dirty: uncommitted changes" if g["dirty"] else "clean"))
  if skip_env:
    out.append(Check("project", "task config", SKIP, "--skip-env"))
    return out
  snippet = (
    "import contact_rl\ncontact_rl.require_tasks()\nfrom mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg\n"
    f"t={task!r}\nassert t in list_tasks(), f'{{t}} not registered: {{list_tasks()}}'\n"
    "e=load_env_cfg(t); a=load_rl_cfg(t); p=load_env_cfg(t, play=True)\n"
    "print('ok', 'num_envs', e.scene.num_envs, 'actor', a.actor.class_name, getattr(a.actor,'rnn_type',''))\n"
  )
  rc, txt = run([sys.executable, "-c", snippet], 300)
  out.append(Check("project", f"task config {task}", PASS if rc == 0 else FAIL, (txt.splitlines() or [""])[-1][:300]))
  return out


# ------------------------------------------------------------------- main


def render_report(checks: list[Check]) -> str:
  colors = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
  tty = sys.stdout.isatty()
  lines, sec = [], None
  for c in checks:
    if c.section != sec:
      sec = c.section
      lines.append(f"\n[{sec}]")
    tag = f"{colors[c.status]}{c.status}\033[0m" if tty else c.status
    lines.append(f"  {tag:<4}  {c.name:<28} {c.detail}")
  n = {s: sum(c.status == s for c in checks) for s in (PASS, WARN, FAIL, SKIP)}
  lines.append(f"\n{n[PASS]} PASS, {n[WARN]} WARN, {n[FAIL]} FAIL, {n[SKIP]} SKIP")
  return "\n".join(lines)


def exit_code(checks: list[Check], strict: bool = False) -> int:
  if any(c.status == FAIL for c in checks):
    return 1
  if strict and any(c.status == WARN for c in checks):
    return 2
  return 0


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="contact-doctor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Mjlab-Contact-Flat-Unitree-Go2")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--run-root", default="runs")
  p.add_argument("--host", default="127.0.0.1", help="viser bind address to validate")
  p.add_argument("--port", type=int, default=8080)
  p.add_argument("--tb-port", type=int, default=6006)
  p.add_argument("--min-free-gib", type=float, default=20.0)
  p.add_argument("--skip-env", action="store_true")
  p.add_argument("--skip-render", action="store_true")
  p.add_argument("--strict", action="store_true", help="exit 2 on WARN")
  p.add_argument("--json", default=None, help="also write the report as JSON")
  return p


def main(argv: list[str] | None = None) -> None:
  a = build_parser().parse_args(argv)
  checks: list[Check] = []
  checks += check_python()
  checks += check_dependencies()
  checks += check_gpu(a.device)
  if a.skip_render:
    checks.append(Check("render", "rendering", SKIP, "--skip-render"))
  else:
    checks += check_rendering(a.device)
  checks += check_filesystem(Path(a.run_root), a.min_free_gib)
  checks += check_network(a.host, a.port, a.tb_port)
  checks += check_project(a.task, a.skip_env, a.device)
  print(render_report(checks))
  if a.json:
    Path(a.json).write_text(json.dumps([asdict(c) for c in checks], indent=2))
  code = exit_code(checks, a.strict)
  print("READY" if code == 0 else "NOT READY" if code == 1 else "READY WITH WARNINGS (strict)")
  sys.exit(code)


if __name__ == "__main__":
  main()
