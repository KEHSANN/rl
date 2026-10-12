"""contact-doctor: PASS/WARN/FAIL decisions with mocked probes."""

from __future__ import annotations

import pytest

from contact_rl.scripts import doctor as D


def test_parse_nvidia_smi():
  g = D.parse_nvidia_smi("0, NVIDIA A10G, 550.54.15, 23028\n1, NVIDIA L4, 550.54.15, 23034\n")
  assert [x["name"] for x in g] == ["NVIDIA A10G", "NVIDIA L4"] and g[0]["driver"] == "550.54.15"


def test_missing_gpu_is_fail_for_cuda_and_warn_for_cpu():
  no_smi = lambda cmd, timeout=0, env=None: (127, "nvidia-smi not found")  # noqa: E731
  c = D.check_gpu("cuda:0", run=no_smi)[0]
  assert c.status == D.FAIL
  c = D.check_gpu("cpu", run=no_smi)[0]
  assert c.status == D.WARN


def test_dependencies_pins_and_missing():
  vers = {"mjlab": "1.2.0", "rsl-rl-lib": "5.0.0", "mujoco": "3.6.0", "mujoco-warp": "3.5.0.2", "warp-lang": "1.11.0"}
  checks = {c.name: c.status for c in D.check_dependencies(ver=vers.get)}
  assert checks["torch"] == D.FAIL  # missing
  assert checks["rsl-rl-lib"] == D.WARN  # pin mismatch
  assert checks["mujoco series"] == D.FAIL
  assert checks["mjlab"] == D.PASS
  assert checks["viser"] == D.WARN and checks["onnx"] == D.SKIP


def test_public_bind_is_fail_and_busy_port_is_warn():
  import socket

  s = socket.socket()
  s.bind(("127.0.0.1", 0))
  s.listen(1)
  port = s.getsockname()[1]
  try:
    checks = D.check_network("0.0.0.0", port, tb_port=0)
  finally:
    s.close()
  st = {c.name: c.status for c in checks}
  assert st["viser bind address"] == D.FAIL and st["viser port"] == D.WARN
  assert {c.name: c.status for c in D.check_network("127.0.0.1", 0, 0)}["viser bind address"] == D.PASS


def test_filesystem_and_exit_codes(tmp_path):
  checks = D.check_filesystem(tmp_path / "runs", min_free_gib=0.0)
  assert all(c.status == D.PASS for c in checks)
  checks = D.check_filesystem(tmp_path / "runs", min_free_gib=1e9)
  assert checks[-1].status in (D.WARN, D.FAIL)
  ok = [D.Check("a", "x", D.PASS), D.Check("a", "y", D.WARN)]
  assert D.exit_code(ok) == 0 and D.exit_code(ok, strict=True) == 2
  assert D.exit_code(ok + [D.Check("a", "z", D.FAIL)]) == 1
  assert "1 PASS, 1 WARN, 0 FAIL" in D.render_report(ok)


def test_encode_probe():
  from contact_rl.utils.video import VideoEncodeError, find_ffmpeg

  try:
    find_ffmpeg()
  except VideoEncodeError:
    pytest.skip("no ffmpeg")
  assert D.check_encode().status == D.PASS
  assert D.check_encode("/nonexistent/ffmpeg").status == D.FAIL


def test_cli_parsing():
  a = D.build_parser().parse_args(["--device", "cpu", "--skip-env", "--port", "9000"])
  assert a.device == "cpu" and a.skip_env and a.port == 9000 and a.host == "127.0.0.1"
