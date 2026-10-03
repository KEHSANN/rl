"""Streaming MP4 writer + recorder wrapper (needs an ffmpeg binary)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from contact_rl.utils import video as V


def _ffmpeg():
  try:
    return V.find_ffmpeg()
  except V.VideoEncodeError:
    pytest.skip("no ffmpeg available")


def _nframes(path) -> int:
  probe = shutil.which("ffprobe")
  if probe is None:
    pytest.skip("no ffprobe")
  out = subprocess.run([probe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
  return int(out.stdout.strip())


def test_streams_frames_to_mp4_with_odd_size_and_float_frames(tmp_path):
  _ffmpeg()
  p = tmp_path / "v" / "a.mp4"
  w = V.StreamingVideoWriter(p, fps=25)
  for k in range(30):
    f = np.full((51, 67, 3), k / 30.0, dtype=np.float32)  # floats in [0,1], odd size
    w.add(V.Hud().draw(f, [f"frame {k}"]))
  assert not p.exists()  # written to .part until close
  assert w.close() == p
  assert p.stat().st_size > 0 and _nframes(p) == 30
  assert not list(p.parent.glob("*.part.mp4"))


def test_encoder_failure_raises_and_leaves_no_file(tmp_path):
  p = tmp_path / "b.mp4"
  w = V.StreamingVideoWriter(p, fps=25, ffmpeg=str(tmp_path / "no-such-ffmpeg"))
  with pytest.raises(V.VideoEncodeError):
    w.add(np.zeros((8, 8, 3), np.uint8))
  assert not p.exists()


def test_bad_frame_shape_is_explicit(tmp_path):
  with pytest.raises(V.VideoEncodeError):
    V.StreamingVideoWriter(tmp_path / "c.mp4", 25).add(np.zeros((8, 8), np.uint8))


class FakeEnv:
  metadata = {"render_fps": 50}
  unwrapped = "base"

  def __init__(self):
    self.closed = False
    self.t = 0

  def step(self, a):
    self.t += 1
    return self.t

  def reset(self):
    return 0

  def render(self):
    return np.full((32, 48, 3), (self.t * 10) % 255, np.uint8)

  def close(self):
    self.closed = True


def test_recorder_wrapper_is_constant_memory_and_clips(tmp_path):
  _ffmpeg()
  env = FakeEnv()
  rec = V.StreamingVideoRecorder(env, path_fn=lambda s: tmp_path / f"it_{s}" / "clip.mp4",
                                 step_trigger=lambda s: s % 20 == 0, video_length=5)
  for _ in range(45):
    rec.step(None)
  rec.close()
  assert [p.parent.name for p in rec.saved] == ["it_0", "it_20", "it_40"]
  assert all(_nframes(p) == 5 for p in rec.saved)
  assert env.closed and rec.unwrapped == "base" and rec.errors == []
  assert not hasattr(rec, "current_video_frames")  # no in-memory frame list
