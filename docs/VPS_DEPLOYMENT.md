# Remote GPU VPS workflow

```bash
git pull origin audit/paper-alignment
bash scripts/vps/setup.sh            # uv sync + sanity
uv run contact-doctor                # PASS/WARN/FAIL, exit 1 only on FAIL
uv run contact-bench --num-envs 256  # smoke test
bash scripts/vps/train_tmux.sh       # contact-train inside tmux
bash scripts/vps/tensorboard.sh      # TensorBoard on 127.0.0.1:6006
# on your laptop:
ssh -N -L 6006:127.0.0.1:6006 -L 8080:127.0.0.1:8080 user@vps   # or scripts/vps/tunnel.sh
bash scripts/vps/watch_tmux.sh       # contact-watch: evaluates each new checkpoint once
uv run contact-eval --checkpoint best
uv run contact-play                  # viser on 127.0.0.1:8080
```

* GPU: `--device cuda:N` or `CUDA_VISIBLE_DEVICES=N`; EGL device follows the chosen GPU (`MUJOCO_EGL_DEVICE_ID`).
* Envs: Go2 defaults to 8192 (paper value). On <20 GiB VRAM use `--env.scene.num-envs 4096`.
* Headless: `MUJOCO_GL=egl` is set in `contact_rl/__init__.py` before mujoco is imported. No X server needed.
* Run layout: `runs/<exp>/<timestamp>/{checkpoints/model_<it>.pt, latest.pt, best.pt, index.json, videos/iteration_<it>/, metrics/, logs/train.log, exported/policy.onnx, run_info.json}`.
* Retention: `--keep-last`, `--keep-every`, `--keep-best`. Writes are atomic; corrupt checkpoints are skipped.
* Resume: `--resume-from <run_dir|checkpoint|latest|best>` continues into a NEW run dir; the source is never modified.
* Stop: SIGINT/SIGTERM (`scripts/vps/stop.sh`) saves a final checkpoint.
* Eval: `metrics.csv` + `summary.json`; outcomes fall / timeout / other / running are distinct; reset steps are excluded from contact metrics.
* Play: gait, direction, speed, yaw rate, stop (speed 0), reset (clears GRU), restore trained commands, checkpoint picker, pause. User input goes through the existing planner (no clamping; out-of-range values warn). `--allow-public` binds 0.0.0.0 with a loud warning; there is no auth.

PERFORMANCE NOT RUNTIME VERIFIED.
