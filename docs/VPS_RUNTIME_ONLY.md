# اجرای فقط-Runtime روی VPS (بدون ترینینگ)

شاخه: `vps-runtime-only` — از `feature/liquid-ai-runtime` (کامیت `1e62e8f`) ساخته شده،
فقط برای اجرا. ترینینگ روی این شاخه عمداً غیرفعال است؛ برای ترین، شاخه‌ی اصلی را
چک‌اوت کن و `scripts/vps/setup.sh` را بزن.

## نصب روی VPS (Linux x86_64 + GPU انویدیا)

```bash
git clone -b vps-runtime-only https://github.com/KEHSANN/rl.git rl && cd rl
bash scripts/vps/setup-runtime.sh
# CPU-only برای تست دود:  EXTRA=cpu bash scripts/vps/setup-runtime.sh
```

`setup-runtime.sh` اول `uv lock` می‌زند (چون `uv.lock` کامیت‌شده، لاک شاخه‌ی کامل است)،
بعد `uv sync --extra cu128`، بعد پچ سنسور mujoco-warp، بعد `contact-doctor`.

نتیجه‌ی مورد انتظار `contact-doctor`: همه PASS، به‌جز یک WARN برای `tensorboard`
(عمداً نصب نشده؛ `contact-eval --tensorboard` فقط هشدار می‌دهد و ادامه می‌دهد).
اگر چیزی FAIL شد، همان خطا را بفرست؛ WARN مربوط به tensorboard طبیعی است.

نکته‌ی `uv`: مثل بقیه‌ی اسکریپت‌ها، `scripts/vps/common.sh` خودش `UV_NO_SYNC=1`
می‌گذارد. با دست هم `uv run --no-sync ...` بزن، وگرنه یک `uv run` ساده، torch
مخصوص CUDA را عوض می‌کند.

## اجرا (چک‌پوینت آماده، بدون ترین)

چک‌پوینت آماده داخل خود ریپو هست: `weights/go2_contact_2026-10-04_03-24-00/checkpoints/best.pt`.
پس `--run-root weights` را بده؛ `--checkpoint best` بدون آن، به ران قدیمی
`runs/go2_contact/2026-10-01_04-21-51` می‌رسد.

```bash
export UV_NO_SYNC=1
# ویوئر (Viser روی 127.0.0.1:8080، با تانل SSH مثل قبل):
bash scripts/vps/play.sh
# یا مستقیم، با چک‌پوینت آماده:
uv run --no-sync contact-play Mjlab-Contact-Flat-Unitree-Go2 \
  --run-root weights --checkpoint best --num-envs 16

# ارزیابی یک‌باره:
bash scripts/vps/eval.sh
# یا مستقیم:
uv run --no-sync contact-eval --task Mjlab-Contact-Flat-Unitree-Go2 \
  --run-root weights --checkpoint best --mode episodes --num-envs 256 --video True

# رول‌آوت هدلس Liquid (بدون ویوئر، گزارش JSON):
uv run --no-sync python -m contact_rl.runtime.rollout \
  --run-root weights --checkpoint best --decider mock \
  --instruction "Move toward the target and turn left." --target 1.5 0.5 --seconds 40
```

## Liquid AI (اختیاری، جدا نصب می‌شود)

پکیج‌های Liquid (`transformers>=5.19` ،`torchao>=0.18` ،`pillow`) جزو
runtime-sync نیستند؛ مثل قبل فقط برای همان اجرا با `--with` اضافه می‌شوند
(و `uv.lock` عوض نمی‌شود):

```bash
uv run --no-sync --with "transformers>=5.19" --with "torchao>=0.18" --with pillow \
  contact-play Mjlab-Contact-Flat-Unitree-Go2 --run-root weights --checkpoint best --num-envs 1 \
  --liquid True --liquid-mode closed_loop --liquid-decision-hz 1.0 \
  --liquid-target "(2.0, 0.0)" --liquid-stdin True
```

شرط GPU طبق `docs/liquid_runtime.md`: انویدیا با INT8 tensor cores (Ampere sm_80 به بالا).
روی CPU یا GPU قدیمی، آداپتور خودش جلوی اجرا را می‌گیرد و مدل را عوض نمی‌کند.

## چی نصب می‌شود، چی نمی‌شود (بررسی‌شده، نه حدسی)

می‌ماند (چون `contact-play` / `contact-eval` / rollout واقعاً بهشان نیاز دارند):

- `torch` (cu128 یا cpu)، `mujoco` 3.5.x، `mujoco-warp` 3.5.0.2، `warp-lang`، `mjlab`
- `rsl-rl-lib` 5.0.1 — با وجود اسمش، فقط ترینینگ نیست؛ لودر پالیسی در `contact-play`
  و `contact-eval` همان `MjlabOnPolicyRunner` ریشه‌ی rsl است. بدون آن، اجرا بالا نمی‌آید.
- `tyro`، `viser` (ویوئر)، `mediapy` + `imageio-ffmpeg` (ویدیو)، `onnx`
  (چون `mjlab/rl/exporter_utils.py` در سطح ماژول `import onnx` می‌کند و
  `ContactOnPolicyRunner` همان را ایمپورت می‌کند؛ حذف onnx، خود play را می‌شکند)
- `onnxscript`: حذفش از mjlab به‌تنهایی فایده ندارد، چون `rsl-rl-lib` خودش آن را
  declare می‌کند؛ کوچک است و در مسیر play/eval ایمپورت نمی‌شود، پس مانده است.

حذف شده (در `mjlab[train]` رفته، و در `uv sync --dry-run` شاخه‌ی runtime دیده نمی‌شود):

- `wandb` — همه‌ی ایمپورت‌هایش lazy است؛ فقط `--wandb-run-path` و `logger=wandb`
  به آن نیاز دارند، و کانفیگ این پروژه `logger=tensorboard` است و play/eval اصلاً
  `learn()` را صدا نمی‌زنند، پس writer لاگ هیچ‌وقت ساخته نمی‌شود.
- `tensorboard` — فقط writer ترینینگ در `learn()` و فلگ اختیاری `contact-eval --tensorboard`
  (که بدون آن فقط هشدار می‌دهد).
- `torchrunx` — فقط لانچر چند-GPU ترینینگ mjlab.

## چی روی این شاخه عمداً کار نمی‌کند

- `contact-train` نصب نمی‌شود (ورودی‌اش از `pyproject.toml` حذف شده).
- `scripts/vps/train_tmux.sh` و `scripts/vps/tensorboard.sh` را روی این شاخه نزن.
- `--wandb-run-path`، `--agent.logger wandb`، `--export-onnx` و چند-GPU کار نمی‌کند.
- اگر به هرکدام نیاز داشتی، برگرد به `feature/liquid-ai-runtime` و `setup.sh`.

## برگشت به ترینینگ

```bash
git checkout feature/liquid-ai-runtime
bash scripts/vps/setup.sh
```
