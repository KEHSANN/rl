# Remote GPU VPS workflow

```bash
git clone -b audit/paper-alignment https://github.com/KEHSANN/rl.git rl && cd rl   # or: git pull origin audit/paper-alignment
bash scripts/vps/setup.sh            # uv sync --extra cu128 --frozen + contact-doctor
export UV_NO_SYNC=1                  # for commands typed by hand (the scripts set it themselves)
uv run contact-doctor                # PASS/WARN/FAIL, exit 1 only on FAIL
uv run contact-bench --num-envs 256  # smoke test
bash scripts/vps/train_tmux.sh Mjlab-Contact-Flat-Unitree-Go2   # contact-train inside tmux (8192 envs)
bash scripts/vps/tensorboard.sh      # TensorBoard on 127.0.0.1:6006
# on your laptop:
ssh -N -L 6006:127.0.0.1:6006 -L 8080:127.0.0.1:8080 user@vps   # or scripts/vps/tunnel.sh
# once train.log shows the run dir (`--run latest` is resolved at watcher start):
bash scripts/vps/watch_tmux.sh       # contact-watch: evaluates each new checkpoint once
uv run contact-eval --checkpoint best --mode episodes --num-envs 256   # or: bash scripts/vps/eval.sh best
uv run contact-play Mjlab-Contact-Flat-Unitree-Go2 --checkpoint best   # viser on 127.0.0.1:8080
```

* `UV_NO_SYNC=1`: a plain `uv run` re-syncs the env without `--extra cu128` and replaces the CUDA torch build. `setup.sh` is the only sync; `scripts/vps/common.sh` exports `UV_NO_SYNC=1` (also inside tmux sessions) and the systemd units set it. By hand: `export UV_NO_SYNC=1` or `uv run --no-sync ...`. Re-run `setup.sh` after a `git pull` that changes `uv.lock`.
* GPU: `--device cuda:N`, `CUDA_VISIBLE_DEVICES=N` or `--gpu-ids "[N]"` (quoted Python list; `--gpu-ids 0 1` is invalid, use `--gpu-ids "[0, 1]"`). More than one selected GPU hands off to mjlab's launcher (legacy `logs/rsl_rl/` layout, no run management). EGL device follows the chosen GPU (`MUJOCO_EGL_DEVICE_ID`).
* Envs: Go2 defaults to 8192 (paper value). On <20 GiB VRAM use `--env.scene.num-envs 4096`.
* Headless: `MUJOCO_GL=egl` is set in `contact_rl/__init__.py` before mujoco is imported. No X server needed.
* Run layout: `runs/<exp>/<timestamp>[_name]/` with `checkpoints/{model_<it>.pt, latest.pt, best.pt, index.json}`, `config/`, `logs/{train.log, error.txt}`, `metrics/iteration_<it>/`, `videos/iteration_<it>/`, `exported/policy.onnx`, `run_info.json`. Selectors: `best`, `latest`, `<run_dir>`, `<run_dir>:best|latest|<it>` or a `.pt` file.
* Retention: `--keep-last`, `--keep-every`, `--keep-best`. Writes are atomic; corrupt checkpoints are skipped.
* Resume: `--resume-from <run_dir|checkpoint|latest|best>` continues into a NEW run dir; the source is never modified.
* Stop: SIGINT/SIGTERM (`scripts/vps/stop.sh train`, `systemctl --user stop`) saves a final checkpoint. Before training has started (env construction, Warp compile) the same signal stops cleanly without a checkpoint and `run_info.json` is finalised as `stopped` (no longer stuck at `starting`). `run_info.json` records `status` and `exit_code` (0 finished/stopped, 130 hard abort, 1 error).
* Eval: `contact-eval` defaults to the Fig. 6 protocol (`--mode fig6`, 1000 envs x 5 gaits x 9 durations); `--mode episodes` is the quick single rollout the watcher uses. `metrics.csv` + `summary.json`; outcomes fall / timeout / other / running are distinct; reset steps are excluded from contact metrics.
* Play: gait, direction, speed, yaw rate, stop (speed 0), reset (clears GRU), restore trained commands, checkpoint picker, pause. User input goes through the existing planner (no clamping; out-of-range values warn). `--host 0.0.0.0 --allow-public True` binds publicly with a loud warning; there is no auth.

PERFORMANCE NOT RUNTIME VERIFIED.
