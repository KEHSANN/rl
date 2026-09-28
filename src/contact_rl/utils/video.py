"""Streaming MP4 writing + optional text HUD for headless evaluation videos.

Frames stream into an ffmpeg child process (imageio-ffmpeg's bundled binary,
fallback mediapy) instead of being buffered in RAM like mjlab's VideoRecorder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def even(x: int) -> int:
  return max(2, int(x) - (int(x) % 2))


class StreamingVideoWriter:
  def __init__(self, path: Path | str, fps: float, crf: int = 23):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.fps = float(fps)
    self.crf = int(crf)
    self._gen: Any = None
    self._mp: Any = None
    self._shape: tuple[int, int] | None = None
    self.frames = 0

  def _open(self, h: int, w: int) -> None:
    self._shape = (h, w)
    try:
      import imageio_ffmpeg

      self._gen = imageio_ffmpeg.write_frames(
        str(self.path), size=(w, h), fps=self.fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1, quality=None,
        output_params=["-crf", str(self.crf), "-preset", "veryfast"], ffmpeg_log_level="error",
      )
      self._gen.send(None)
      return
    except ImportError:
      pass
    import mediapy

    self._mp = mediapy.VideoWriter(self.path, shape=(h, w), fps=self.fps, crf=self.crf)
    self._mp.__enter__()

  def add(self, frame: np.ndarray) -> None:
    frame = np.asarray(frame)
    if frame.ndim == 4:
      frame = frame[0]
    if frame.dtype != np.uint8:
      frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame = frame[..., :3]
    if self._shape is None:
      self._open(even(frame.shape[0]), even(frame.shape[1]))
    eh, ew = self._shape  # type: ignore[misc]
    frame = np.ascontiguousarray(frame[:eh, :ew])
    if self._gen is not None:
      self._gen.send(frame)
    else:
      self._mp.add_image(frame)
    self.frames += 1

  def close(self) -> Path | None:
    if self._gen is not None:
      self._gen.close()
      self._gen = None
    if self._mp is not None:
      self._mp.__exit__(None, None, None)
      self._mp = None
    return self.path if self.frames else None

  def __enter__(self) -> "StreamingVideoWriter":
    return self

  def __exit__(self, *exc) -> None:
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
    img = self._Image.fromarray(np.asarray(frame)[..., :3])
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
