"""Streaming MP4 writing, optional text HUD, and a constant-memory recorder
wrapper for training videos.

Frames are piped one by one into an ffmpeg child process (``imageio-ffmpeg``'s
bundled binary, falling back to ``ffmpeg`` on PATH). Memory use is one frame,
independent of video length -- this replaces mjlab's ``VideoRecorder``, which
keeps every frame of a clip in a Python list before encoding.

Encoder failures raise :class:`VideoEncodeError` (with ffmpeg's stderr);
callers that must not die because of a video (training, the watcher) catch it
and log it explicitly.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np


class VideoEncodeError(RuntimeError):
  pass


def even(x: int) -> int:
  return max(2, int(x) - (int(x) % 2))


def find_ffmpeg() -> str:
  """Path of the ffmpeg binary: imageio-ffmpeg's bundled one, else PATH."""
  try:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if exe and Path(exe).exists():
      return exe
  except Exception:
    pass
  exe = shutil.which("ffmpeg")
  if exe:
    return exe
  raise VideoEncodeError("No ffmpeg found (install imageio-ffmpeg or put ffmpeg on PATH).")


def to_uint8_rgb(frame: Any) -> np.ndarray:
  frame = np.asarray(frame)
  if frame.ndim == 4:
    frame = frame[0]
  if frame.ndim != 3 or frame.shape[-1] < 3:
    raise VideoEncodeError(f"Expected an HxWx3 frame, got shape {frame.shape}.")
  frame = frame[..., :3]
  if frame.dtype != np.uint8:
    f = frame.astype(np.float32)
    if f.size and float(np.nanmax(f)) <= 1.0:
      f = f * 255.0
    frame = np.clip(f, 0, 255).astype(np.uint8)
  return frame


class StreamingVideoWriter:
  """``add(frame)`` pipes raw RGB into ffmpeg; ``close()`` finalises the MP4.
  The output is written to ``<name>.part.mp4`` and renamed on success, so a
  crash never leaves a truncated file under the final name."""

  def __init__(self, path: Path | str, fps: float, crf: int = 23, ffmpeg: str | None = None):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.fps = float(fps)
    self.crf = int(crf)
    self._ffmpeg = ffmpeg
    self._proc: subprocess.Popen | None = None
    self._err: Any = None
    self._shape: tuple[int, int] | None = None
    self._part = self.path.with_name(self.path.stem + ".part" + self.path.suffix)
    self.frames = 0
    self.failed: str | None = None

  def _open(self, h: int, w: int) -> None:
    self._shape = (h, w)
    exe = self._ffmpeg or find_ffmpeg()
    cmd = [
      exe, "-y", "-loglevel", "error",
      "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-",
      "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self._part),
    ]
    self._err = tempfile.TemporaryFile()
    try:
      self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err)
    except OSError as e:
      raise VideoEncodeError(f"could not start ffmpeg ({exe}): {e}") from e

  def _stderr(self) -> str:
    if self._err is None:
      return ""
    try:
      self._err.seek(0)
      return self._err.read().decode(errors="replace").strip()
    except Exception:
      return ""

  def add(self, frame: np.ndarray) -> None:
    frame = to_uint8_rgb(frame)
    if self._shape is None:
      self._open(even(frame.shape[0]), even(frame.shape[1]))
    eh, ew = self._shape  # type: ignore[misc]
    frame = np.ascontiguousarray(frame[:eh, :ew])
    if frame.shape[:2] != (eh, ew):
      raise VideoEncodeError(f"frame size changed mid-video: {frame.shape[:2]} vs {(eh, ew)}")
    assert self._proc is not None and self._proc.stdin is not None
    try:
      self._proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError) as e:
      self.failed = self._stderr() or str(e)
      self._kill()
      raise VideoEncodeError(f"ffmpeg died while encoding {self.path}: {self.failed}") from e
    self.frames += 1

  def _kill(self) -> None:
    if self._proc is not None:
      try:
        self._proc.kill()
        self._proc.wait(timeout=5)
      except Exception:
        pass
      self._proc = None
    self._part.unlink(missing_ok=True)

  def close(self) -> Path | None:
    """Finalise; returns the MP4 path, or None if no frame was written."""
    if self._proc is None:
      return self.path if (self.frames and self.path.exists()) else None
    proc, self._proc = self._proc, None
    try:
      if proc.stdin is not None:
        proc.stdin.close()
      rc = proc.wait(timeout=120)
    except Exception as e:
      proc.kill()
      self._part.unlink(missing_ok=True)
      raise VideoEncodeError(f"ffmpeg did not finish {self.path}: {e}") from e
    if rc != 0 or not self._part.exists():
      self.failed = self._stderr() or f"exit code {rc}"
      self._part.unlink(missing_ok=True)
      raise VideoEncodeError(f"ffmpeg failed for {self.path}: {self.failed}")
    self._part.replace(self.path)
    if self._err is not None:
      self._err.close()
    return self.path if self.frames else None

  def abort(self) -> None:
    self._kill()

  def __enter__(self) -> "StreamingVideoWriter":
    return self

  def __exit__(self, exc_type, *exc) -> None:
    if exc_type is not None:
      self.abort()
    else:
      self.close()


class Hud:
  """Top-left text overlay via Pillow; no-op if Pillow is unavailable."""

  def __init__(self, enabled: bool = True):
    self.enabled = enabled
    self._ok = False
    if enabled:
      try:
        from PIL import Image, ImageDraw, ImageFont

        self._Image, self._Draw = Image, ImageDraw
        self._font = ImageFont.load_default()
        self._ok = True
      except Exception:
        self._ok = False

  def draw(self, frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if not (self.enabled and self._ok and lines):
      return frame
    img = self._Image.fromarray(to_uint8_rgb(frame))
    d = self._Draw.Draw(img, "RGBA")
    pad, lh = 4, 12
    d.rectangle([0, 0, 8 + 6 * max(len(s) for s in lines), pad * 2 + lh * len(lines)], fill=(0, 0, 0, 140))
    for k, s in enumerate(lines):
      d.text((pad, pad + k * lh), s, fill=(255, 255, 255, 255), font=self._font)
    return np.asarray(img)


def time_bar(frame: np.ndarray, frac: float, color=(255, 200, 0)) -> np.ndarray:
  frame = np.array(frame, copy=True)
  h, w = frame.shape[:2]
  frac = float(min(max(frac, 0.0), 1.0))
  frame[h - 6 : h] = (frame[h - 6 : h] * 0.4).astype(frame.dtype)
  frame[h - 6 : h, : int(w * frac), :3] = color
  return frame


class StreamingVideoRecorder:
  """Drop-in replacement for ``mjlab.utils.wrappers.VideoRecorder`` (step
  trigger only) that streams frames to disk.

  ``path_fn(step)`` returns the MP4 path of a clip started at env step
  ``step``. Encoder failures are logged and disable recording for that clip;
  they never propagate into the training loop.
  """

  def __init__(
    self,
    env: Any,
    path_fn: Callable[[int], Path],
    step_trigger: Callable[[int], bool],
    video_length: int,
    fps: float | None = None,
    hud_fn: Callable[[int], list[str]] | None = None,
  ):
    self._wrapped_env = env
    self._path_fn = path_fn
    self._trigger = step_trigger
    self._length = int(video_length)
    meta = getattr(env, "metadata", None) or {}
    self._fps = float(fps or meta.get("render_fps", 30))
    self._hud = Hud(hud_fn is not None)
    self._hud_fn = hud_fn
    self._writer: StreamingVideoWriter | None = None
    self.step_count = 0
    self.saved: list[Path] = []
    self.errors: list[str] = []

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped_env, name)

  @property
  def unwrapped(self) -> Any:
    return self._wrapped_env.unwrapped

  def reset(self, **kwargs: Any) -> Any:
    return self._wrapped_env.reset(**kwargs)

  def render(self) -> Any:
    return self._wrapped_env.render()

  def step(self, action: Any) -> Any:
    if self._writer is None and self._trigger(self.step_count):
      self._writer = StreamingVideoWriter(self._path_fn(self.step_count), self._fps)
    out = self._wrapped_env.step(action)
    if self._writer is not None:
      try:
        frame = self._wrapped_env.render()
        if frame is not None:
          if self._hud_fn is not None:
            frame = self._hud.draw(frame, self._hud_fn(self.step_count))
          self._writer.add(frame)
        if self._writer.frames >= self._length:
          self._finish()
      except VideoEncodeError as e:
        self.errors.append(str(e))
        print(f"[ERROR] training video disabled for this clip: {e}")
        self._writer.abort()
        self._writer = None
    self.step_count += 1
    return out

  def _finish(self) -> None:
    w, self._writer = self._writer, None
    if w is None:
      return
    try:
      p = w.close()
      if p is not None:
        self.saved.append(p)
        print(f"[INFO] saved video {p}")
    except VideoEncodeError as e:
      self.errors.append(str(e))
      print(f"[ERROR] {e}")

  def close(self) -> None:
    self._finish()
    self._wrapped_env.close()
