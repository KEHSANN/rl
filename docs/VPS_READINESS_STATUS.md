# GPU VPS readiness: status

Run the full suite on the VPS: `uv run --with pytest pytest tests -q`.

Implemented: runner (EV, entropy coef, iteration time, GPU mem, training/* system/* TB, best.pt, atomic checkpoints, resume at iter+1 into a new run dir, SIGTERM/SIGINT graceful stop, git/version run_info), GRU ONNX wrapper (utils/onnx_compat.py), contact-train/-eval/-watch/-play/-doctor/-bench (all six registered in `[project.scripts]`), streaming video, command override wired via apply_to_command_term, GRU reset on dones + manual reset, MUJOCO_GL set in contact_rl/__init__ before mujoco import, scripts/vps.

Not runtime-verified (needs VPS): EGL render, Warp, training start, VRAM at 4096/8192, checkpoint/resume on real runs, watcher+eval on GPU, ONNX export of real GRU, Viser/tunnel/controls, learning behaviour.

## Final compatibility audit
Fixed: `contact-train` recorded `exit_code: 0` in run_info.json when training crashed (OOM/NaN/bad resume) and leaked the env if the runner or resume load failed -- now try/finally, exit 1 on error, env always closed. Graceful stop now always writes its checkpoint (no longer conditional on a logger writer). `contact-play` closes the env in `finally`. systemd: `KillMode=mixed` (SIGTERM to uv only, which forwards it), watcher unit no longer restart-loops on exit 3 (lock held). Docs: stale no-task `train_tmux.sh` / `contact-play` commands, clone of the wrong branch, removed obsolete PYPROJECT_SCRIPTS.txt step.

VERIFIED in a sandbox without torch/mjlab/GPU/pytest (tests executed with a minimal pytest-compatible runner): test_checkpoints (8), test_watch (7), test_watch_extra (3, new: second watcher exits 3 + lock released, stale summary rejected, aborted eval not counted), tmux exit-code wrapper (2); train shutdown paths against stubbed mjlab (finish/stop -> 0, second Ctrl-C -> 130, crash/runner failure -> 1 re-raised; env closed in all 5); py_compile; `bash -n` on every script; `systemd-analyze verify` on all three units.
NOT RUN (missing deps): everything importing torch / tyro / mjlab (CLI `--help`, command override, ONNX export, rsl-rl GRU, resolve_device, planner tests), real pytest.
PERFORMANCE NOT RUNTIME VERIFIED.
