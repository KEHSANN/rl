# VPS scripts

Linux x86_64 + NVIDIA, no Docker. All scripts `source common.sh`, which `cd`s to the repo, defaults `MUJOCO_GL=egl` and exports `UV_NO_SYNC=1`.

**Why `UV_NO_SYNC=1`:** a plain `uv run` re-syncs the env to the *no-extra* resolution and would swap out the CUDA torch installed by `uv sync --extra cu128`. `setup.sh` syncs once; everything else runs the env as installed. Re-run `setup.sh` after a `git pull` that changes `uv.lock`. When typing commands by hand use `uv run --no-sync ...` (or `export UV_NO_SYNC=1`).

| Script | What it does | Knobs |
|---|---|---|
| `setup.sh [doctor args]` | `uv sync --extra cu128 --python 3.12 --frozen`, then `contact-doctor` | `EXTRA=cpu` |
| `train_tmux.sh <task> [contact-train args]` | training in tmux session `train` (task is required) | `SESSION` |
| `watch_tmux.sh [run] [contact-watch args]` | checkpoint watcher (default run: `latest`) | `SESSION` |
| `eval.sh [selector] [contact-eval args]` | one-off episodes eval + video (default `best`) | `TASK`, `NUM_ENVS` (256) |
| `play.sh [selector] [contact-play args]` | viser viewer on 127.0.0.1 (default `best`) | `TASK`, `PORT` (8080), `SESSION` |
| `tensorboard.sh [logdir] [args]` | TensorBoard on 127.0.0.1 (default `runs`) | `PORT` (6006), `SESSION` |
| `stop.sh <session>` | graceful stop of a tmux session (checkpoint, then exit) | |
| `tunnel.sh` | SSH tunnel from your laptop for TensorBoard + viewer | `TB_PORT`, `VISER_PORT`, `REMOTE_*` |

The optional first argument (task / run / selector / logdir) is only taken if it does not start with `-`, so `play.sh --num-envs 16` means "best checkpoint, 16 envs". The tmux window prints `[exited with <code>]` with the real exit code and stays open.

Defaults: training uses the task's 8192 environments (paper setting). Only on a lower-VRAM GPU add `--env.scene.num-envs 4096`. GPU selection: `CUDA_VISIBLE_DEVICES=1 scripts/vps/train_tmux.sh <task>` or `--gpu-ids "[1]"`; more than one GPU (`--gpu-ids "[0, 1]"` / `all`) hands off to mjlab's launcher without contact-train run management.

Checkpoints and logs: `runs/<experiment>/<timestamp>/` with `checkpoints/model_<it>.pt`, `checkpoints/latest.pt`, `checkpoints/best.pt`, `checkpoints/index.json`, `logs/train.log`, `logs/error.txt` (on crash), `run_info.json`, `exported/policy.onnx`, `metrics/`, `videos/`.

## systemd (optional, instead of tmux)

```bash
bash scripts/vps/setup.sh
mkdir -p ~/.config/systemd/user && cp scripts/vps/systemd/*.service ~/.config/systemd/user/
loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user start contact-train@Mjlab-Contact-Flat-Unitree-Go2
systemctl --user start contact-watch contact-tensorboard   # after the run dir exists
journalctl --user -u contact-train@Mjlab-Contact-Flat-Unitree-Go2 -f
```

Units assume the repo at `~/rl` and uv at `~/.local/bin/uv`. They set `PYTHONUNBUFFERED=1` and `UV_NO_SYNC=1`; train/watch use `KillMode=mixed` + SIGTERM (graceful stop, SIGKILL after the timeout), the trainer is never auto-restarted, the watcher is restarted on failure except exit 3 (lock held). `systemctl --user stop` during env construction is also recorded as `stopped` in `run_info.json`.
