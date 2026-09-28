# GPU VPS readiness: status

Tested in a sandbox WITHOUT torch/mjlab/GPU. 35 unit tests pass there; 7 skip (need torch/tyro/mjlab) and must run on the VPS: `uv run --with pytest pytest tests -q`.

Implemented: runner (EV, entropy coef, iteration time, GPU mem, training/* system/* TB, best.pt, atomic checkpoints, resume at iter+1 into a new run dir, SIGTERM/SIGINT graceful stop, git/version run_info), GRU ONNX wrapper (utils/onnx_compat.py), contact-train/-eval/-watch/-play/-doctor, streaming video, command override wired via apply_to_command_term, GRU reset on dones + manual reset, MUJOCO_GL set in contact_rl/__init__ before mujoco import, scripts/vps.

Not runtime-verified (needs VPS): EGL render, Warp, training start, VRAM at 4096/8192, checkpoint/resume on real runs, watcher+eval on GPU, ONNX export of real GRU, Viser/tunnel/controls, learning behaviour.

Manual step: add `contact-watch` and `contact-doctor` to [project.scripts] in pyproject.toml (see PYPROJECT_SCRIPTS.txt).

## Integration status (part 2)
VERIFIED (sandbox, no torch/mujoco): 44 unit tests passed (checkpoints, retention, watcher, episode metrics, doctor, video/ffmpeg, runtime, entry points, play GRU proxy); py_compile; bash -n.
SKIPPED: 7 tests needing torch/tyro/mjlab (CLI --help, command override, ONNX export, rsl-rl GRU, resolve_device), plus tests/test_planning.py.
NOT RUNTIME VERIFIED: training, resume, TensorBoard, EGL rendering, ONNX export, contact-play, GPU/VRAM checks.
PERFORMANCE NOT RUNTIME VERIFIED.
