"""Process-level runtime helpers: headless GL, console tee, signals, locks,
run info, device selection and GPU memory statistics.

Nothing in this module imports torch / mujoco / mjlab at module level:
:func:`configure_headless_rendering` has to run *before* ``import mujoco``
because MuJoCo selects its OpenGL backend from ``MUJOCO_GL`` at import time.
``contact_rl/__init__.py`` calls it before importing the task package (which
imports mjlab -> mujoco), so every ``contact-*`` entry point gets the right
backend no matter which ``contact_rl`` submodule it imports first.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
  "contact-rl",
  "mjlab",
  "torch",
  "mujoco",
  "mujoco-warp",
  "warp-lang",
  "rsl-rl-lib",
  "numpy",
  "tensordict",
  "tensorboard",
  "tyro",
  "viser",
  "mediapy",
  "imageio-ffmpeg",
  "pillow",
  "onnx",
  "onnxscript",
  "onnxruntime",
  "PyOpenGL",
  "GitPython",
  "wandb",
)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

DUPLICATE_SIGNAL_WINDOW_S = 1.0
"""A repeated SIGINT within this many seconds of the first stop request is
treated as a duplicate delivery of the same Ctrl-C (see :class:`StopSignalHandler`)."""


def has_display() -> bool:
  return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def configure_headless_rendering() -> str:
  """Select a headless OpenGL backend unless the user already chose one.

  No DISPLAY and no (or an empty) MUJOCO_GL -> egl. An explicit non-empty
  MUJOCO_GL is always respected; with a DISPLAY and no MUJOCO_GL MuJoCo keeps
  its default (GLFW) so local graphical use is unchanged. MUJOCO_GL=egl/osmesa
  implies the same PYOPENGL_PLATFORM (unless that is set explicitly). Must be
  called before ``mujoco`` is imported; if mujoco is already imported the
  backend can no longer change and a warning is printed.
  """
  already = "mujoco" in sys.modules
  before = os.environ.get("MUJOCO_GL")
  if before is not None and not before.strip():
    del os.environ["MUJOCO_GL"]  # MUJOCO_GL="" is not a valid MuJoCo backend
  if not os.environ.get("MUJOCO_GL") and not has_display():
    os.environ["MUJOCO_GL"] = "egl"
  backend = os.environ.get("MUJOCO_GL", "").lower()
  if backend in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
  if already and os.environ.get("MUJOCO_GL") != before:
    print(
      "[WARN] mujoco was imported before MUJOCO_GL was set; offscreen rendering "
      "may use the wrong GL backend.",
      file=sys.stderr,
    )
  return backend or "default"


def set_egl_device_for(device: str) -> None:
  """Point MuJoCo's EGL context at the GPU used for simulation."""
  if device.startswith("cuda") and "MUJOCO_EGL_DEVICE_ID" not in os.environ:
    idx = device.split(":", 1)[1] if ":" in device else "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = idx


def assert_private_bind(host: str, allow_public: bool) -> None:
  """Refuse to bind an unauthenticated control server to a public interface."""
  if host in _LOOPBACK_HOSTS or allow_public:
    return
  raise SystemExit(
    f"Refusing to bind the interactive server to '{host}'. It has no "
    "authentication; keep it on 127.0.0.1 and use an SSH tunnel "
    "(ssh -L 8080:127.0.0.1:8080 user@vps), or pass --allow-public explicitly."
  )


def is_loopback(host: str) -> bool:
  return host in _LOOPBACK_HOSTS


def resolve_device(requested: str | None) -> str:
  """Validate / pick the torch device. Explicit errors instead of silent CPU
  fallback when a GPU was requested."""
  import torch

  if requested in (None, "", "auto"):
    return "cuda:0" if torch.cuda.is_available() else "cpu"
  req = str(requested)
  if req == "cpu":
    return "cpu"
  if req.startswith("cuda"):
    if not torch.cuda.is_available():
      raise SystemExit(
        f"Device '{req}' requested but torch.cuda.is_available() is False "
        f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
        "Run `uv run contact-doctor`."
      )
    idx = int(req.split(":", 1)[1]) if ":" in req else 0
    n = torch.cuda.device_count()
    if idx >= n:
      raise SystemExit(f"Device '{req}' requested but only {n} CUDA device(s) visible.")
    return f"cuda:{idx}"
  raise SystemExit(f"Unknown device '{req}' (use cpu, cuda or cuda:<index>).")


def gpu_memory_stats(device: str) -> dict[str, float]:
  """Allocated / reserved / peak GiB for a CUDA device, {} on CPU."""
  if not str(device).startswith("cuda"):
    return {}
  try:
    import torch

    if not torch.cuda.is_available():
      return {}
    g = 1024.0**3
    out = {
      "allocated_gib": torch.cuda.memory_allocated(device) / g,
      "reserved_gib": torch.cuda.memory_reserved(device) / g,
      "max_allocated_gib": torch.cuda.max_memory_allocated(device) / g,
    }
    try:
      free, total = torch.cuda.mem_get_info(device)
      out["device_used_gib"] = (total - free) / g
      out["device_total_gib"] = total / g
    except Exception:
      pass
    return out
  except Exception:
    return {}


class _Tee(io.TextIOBase):
  """Write to a log file and (best effort) to the original stream; survives a
  dropped SSH terminal."""

  def __init__(self, stream: Any, file: Any):
    self._stream = stream
    self._file = file
    self._stream_ok = True

  def write(self, s: str) -> int:  # type: ignore[override]
    self._file.write(s)
    if self._stream_ok:
      try:
        self._stream.write(s)
      except (OSError, ValueError):
        self._stream_ok = False
    return len(s)

  def flush(self) -> None:  # type: ignore[override]
    self._file.flush()
    if self._stream_ok:
      try:
        self._stream.flush()
      except (OSError, ValueError):
        self._stream_ok = False

  def isatty(self) -> bool:  # type: ignore[override]
    return False

  def fileno(self) -> int:  # type: ignore[override]
    return self._file.fileno()

  @property
  def encoding(self) -> str:  # type: ignore[override]
    return "utf-8"


def install_console_tee(log_file: Path) -> None:
  log_file.parent.mkdir(parents=True, exist_ok=True)
  f = open(log_file, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
  sys.stdout = _Tee(sys.stdout, f)  # type: ignore[assignment]
  sys.stderr = _Tee(sys.stderr, f)  # type: ignore[assignment]


def _signal_name(signum: int) -> str:
  try:
    return signal.Signals(signum).name
  except ValueError:
    return f"signal {signum}"


class StopSignalHandler:
  """Graceful-stop signal policy shared by ``contact-train`` and ``contact-watch``.

  * The first SIGINT / SIGTERM calls ``on_stop(name)``: finish the current unit
    of work (training iteration / evaluation), then exit cleanly.
  * SIGTERM never escalates: systemd and ``kill`` may deliver it more than once
    (e.g. to both ``uv`` and its child), and a stop request is idempotent.
  * A repeated SIGINT within ``window_s`` of the first stop request is treated
    as a **duplicate delivery** of the same Ctrl-C and ignored. Launchers such
    as ``uv run`` (and ``tmux send-keys C-c``) can deliver one Ctrl-C twice --
    once from the terminal to the whole foreground process group and once
    forwarded by the launcher -- which used to turn a clean stop into a hard
    abort.
  * A SIGINT arriving ``window_s`` or more after the stop request is a real
    second Ctrl-C and calls ``on_abort(name)`` (default: raise
    :class:`KeyboardInterrupt`). Every later SIGINT escalates again, so
    legitimate interrupts are never suppressed indefinitely.
  """

  def __init__(
    self,
    on_stop: Callable[[str], None],
    on_abort: Callable[[str], None] | None = None,
    window_s: float = DUPLICATE_SIGNAL_WINDOW_S,
    clock: Callable[[], float] = time.monotonic,
  ):
    self._on_stop = on_stop
    self._on_abort = on_abort
    self.window_s = float(window_s)
    self._clock = clock
    self.stop_requested_at: float | None = None
    self.ignored = 0

  def __call__(self, signum: int, _frame: Any = None) -> None:
    name = _signal_name(signum)
    now = self._clock()
    if self.stop_requested_at is None:
      self.stop_requested_at = now
      self._on_stop(name)
      return
    if signum != signal.SIGINT:
      self._on_stop(name)  # repeated SIGTERM: idempotent stop request
      return
    if now - self.stop_requested_at < self.window_s:
      self.ignored += 1
      print(
        f"[INFO] duplicate {name} within {self.window_s:.1f}s of the stop request ignored "
        "(launcher double delivery); press Ctrl-C again to abort.",
        file=sys.stderr,
        flush=True,
      )
      return
    if self._on_abort is None:
      raise KeyboardInterrupt
    self._on_abort(name)


def install_stop_handlers(
  on_stop: Callable[[str], None],
  on_abort: Callable[[str], None] | None = None,
  window_s: float = DUPLICATE_SIGNAL_WINDOW_S,
) -> StopSignalHandler:
  """Install :class:`StopSignalHandler` for SIGTERM / SIGINT; SIGHUP is ignored
  (dropped SSH session). Returns the handler."""
  handler = StopSignalHandler(on_stop, on_abort, window_s)
  signal.signal(signal.SIGTERM, handler)
  signal.signal(signal.SIGINT, handler)
  if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
  return handler


class LockHeld(RuntimeError):
  """Another process holds the lock."""


def acquire_lock(path: Path) -> Any:
  """Take an exclusive, non-blocking advisory lock (``flock``) on ``path``.

  The lock lives as long as the returned file object is open and is released
  by the kernel when the process exits (also on SIGKILL), so a crashed holder
  never leaves a stale lock behind. Raises :class:`LockHeld` if another open
  file description (another process) holds it.
  """
  import fcntl  # Linux / macOS only; the VPS target is Linux

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  f = open(path, "a+", encoding="utf-8")  # noqa: SIM115
  try:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError as e:
    try:
      f.seek(0)
      holder = f.read().strip()
    finally:
      f.close()
    raise LockHeld(f"{path} is held by {holder or 'another process'}") from e
  f.seek(0)
  f.truncate()
  f.write(f"pid {os.getpid()} since {now_iso()}\n")
  f.flush()
  return f


def repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def git_info(path: Path | None = None) -> dict[str, Any]:
  cwd = str(path or repo_root())

  def _git(*args: str) -> str | None:
    try:
      out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
      return None
    return out.stdout.strip() if out.returncode == 0 else None

  commit = _git("rev-parse", "HEAD")
  if commit is None:
    return {}
  status = _git("status", "--porcelain")
  return {"commit": commit, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status)}


def distribution_versions(names: tuple[str, ...] = TRACKED_DISTRIBUTIONS) -> dict:
  out: dict[str, str | None] = {}
  for n in names:
    try:
      out[n] = _md.version(n)
    except _md.PackageNotFoundError:
      out[n] = None
  return out


def hardware_info() -> dict[str, Any]:
  info: dict[str, Any] = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
  }
  try:
    with open("/proc/meminfo") as f:
      info["ram_gib"] = round(int(f.readline().split()[1]) / 1024**2, 1)
  except (OSError, ValueError, IndexError):
    pass
  try:
    import torch

    info["torch_cuda"] = torch.version.cuda
    if torch.cuda.is_available():
      info["gpus"] = [
        {
          "index": i,
          "name": torch.cuda.get_device_properties(i).name,
          "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
          "capability": "%d.%d" % torch.cuda.get_device_capability(i),
        }
        for i in range(torch.cuda.device_count())
      ]
  except Exception as e:  # pragma: no cover
    info["torch_error"] = repr(e)
  return info


def now_iso() -> str:
  return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=str)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
  try:
    with open(path, encoding="utf-8") as f:
      return json.load(f)
  except (OSError, ValueError):
    return default


def update_run_info(run_dir: Path, **fields: Any) -> dict:
  path = Path(run_dir) / "run_info.json"
  data = read_json(path, default={}) or {}
  data.update(fields)
  write_json(path, data)
  return data
