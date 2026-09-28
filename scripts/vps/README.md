# VPS operations

Plain Linux x86_64 + NVIDIA driver, SSH access. No Docker, no desktop.
Scripts are committed without the executable bit: run them with `bash`
(or `chmod +x scripts/vps/*.sh` once).

| Script | Where | What |
| --- | --- | --- |
| `setup.sh` | VPS | `uv sync --extra cu128 --frozen` + `contact-doctor` |
| `train_tmux.sh <task> [args]` | VPS | training in tmux session `train` |
| `watch_tmux.sh [run] [args]` | VPS | checkpoint watcher in tmux session `watch` |
| `tensorboard.sh [logdir]` | VPS | TensorBoard on 127.0.0.1:6006 (session `tb`) |
| `eval.sh <selector> [args]` | VPS | one-off evaluation (+video) |
| `play.sh <selector> [args]` | VPS | viewer on 127.0.0.1:8080 (session `play`) |
| `stop.sh <session>` | VPS | graceful stop (Ctrl-C once = checkpoint + exit; again after >1 s = abort) |
| `tunnel.sh user@host` | **laptop** | forwards 6006 and 8080 over SSH |
| `systemd/*.service` | VPS | optional user units instead of tmux |

tmux windows stay open after the command ends and show its real exit code
(`[exited with N]`).

Typical session:

```bash
# laptop
ssh user@vps
# VPS
git clone <repo> rl && cd rl
bash scripts/vps/setup.sh
bash scripts/vps/train_tmux.sh Mjlab-Contact-Flat-Unitree-Go2      # paper default: 8192 envs
# lower-VRAM GPU: append --env.scene.num-envs 4096
bash scripts/vps/tensorboard.sh
bash scripts/vps/watch_tmux.sh            # newest run
exit                                      # training keeps running
# laptop
bash scripts/vps/tunnel.sh user@vps       # http://localhost:6006
```
