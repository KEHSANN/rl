# GPU VPS readiness: status (work in progress)

This branch is being made production-ready for a headless GPU VPS. **Part 1
is committed; Parts 2-3 are NOT done yet.** Nothing here has been runtime
verified (the authoring sandbox had no torch/GPU/simulator).

## Done (part 1), VERIFIED BY STATIC ANALYSIS (py_compile) only

- `src/contact_rl/utils/runtime.py`: headless EGL selection before mujoco import,
  SSH-safe console tee, SIGTERM/SIGINT graceful-stop hooks, SIGHUP ignored,
  git/version/hardware capture, atomic JSON, localhost-only bind guard.

## Written but not yet pushed / pending

- `utils/checkpoints.py` (resolve latest/best/:<it>, atomic save, index.json,
  retention), `utils/video.py` (streaming ffmpeg MP4 via imageio-ffmpeg, HUD),
  `mdp/command_override.py` (user commands through the planner).
- Runner rewrite (explained variance, entropy coef, iteration time, GPU mem,
  checkpoint TB scalars, checkpoints/ dir, best.pt, fixed GRU ONNX export).
- `contact-train` rewrite (runs/<exp>/<ts>, resume-from, gpu, video, retention),
  `contact-watch` async eval/video sidecar, `contact-eval` protocol mode,
  `contact-play` localhost viser UI with command/checkpoint controls,
  `contact-doctor`, scripts/vps/*, README section.

## Audit findings to fix in the pending parts

1. rsl-rl 5.0.1 `_OnnxRNNModel` (GRU) returns 3 outputs but declares 2 names:
   ONNX export of the recurrent policy fails; current runner swallows it.
2. mjlab train calls `runner.load(path)` without `map_location` (resume across
   devices) and sets `MUJOCO_GL=egl` after mujoco is imported.
3. mjlab play binds viser to 0.0.0.0 (unauthenticated) on headless hosts.
4. mjlab play viewer never resets the GRU hidden state on episode dones.
5. contact-eval counts the auto-reset transition in contact metrics and treats
   time-outs as falls.
6. rsl-rl resume repeats the loaded iteration and overwrites its checkpoint.
7. mjlab VideoRecorder buffers all frames in RAM.
