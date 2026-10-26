# اجرا با وزن‌های آماده + مدل ریموت روی marimo (بدون ترین)

شاخه: `runtime-remote-marimo` — از `vps-runtime-only` (کامیت `4119b94`) ساخته شده.
فقط برای اجراست، با همان وزن‌های ترین‌شده‌ی قبلی؛ ترینینگ روی این شاخه عمداً
غیرفعال است. مدل تصمیم‌گیر (Liquid) هم اینجا لوکال اجرا نمی‌شود؛ روی marimo
است و فقط همان یک تماس `system_one` از طریق رله‌ی VPS می‌آید.

## این شاخه چه فرقی با شاخه‌های قبلی دارد؟

| | `vps-runtime-only` | `runtime-remote-marimo` (این شاخه) |
|---|---|---|
| نصب ترینینگ | نه (`wandb`/`tensorboard`/`torchrunx` در `mjlab[train]`) | همان، نه |
| `contact-train` | نصب نمی‌شود | نصب نمی‌شود |
| پکیج‌های Liquid روی ماشین اجرا | جدا با `--with`، فقط برای اجرای لوکال | اصلاً لازم نیست |
| مدل Liquid | لوکال (`--liquid-decider` نداشت) | ریموت (`--liquid-decider remote`) |
| ویوئر `contact-play` با مدل ریموت | نداشت | دارد (قسمت جدید این شاخه) |
| رول‌آوت هدلس با مدل ریموت | نداشت | دارد (`--decider remote`، از شاخه‌ی ریموت قبلی) |

## نصب روی ماشین اجرا (فقط پکیج‌های لازم)

```bash
git clone -b runtime-remote-marimo https://github.com/KEHSANN/rl.git rl && cd rl
bash scripts/vps/setup-runtime.sh
# CPU-only برای تست دود:  EXTRA=cpu bash scripts/vps/setup-runtime.sh
```

همان `setup-runtime.sh` شاخه‌ی قبلی است: اول `uv lock` می‌زند (چون `uv.lock`
کامیت‌شده، لاک شاخه‌ی کامل است)، بعد `uv sync --extra cu128` بدون
`mjlab[train]`، بعد پچ سنسور mujoco-warp، بعد `contact-doctor`. نتیجه‌ی
مورد انتظار `contact-doctor`: همه PASS، به‌جز یک WARN برای `tensorboard`
(عمداً نصب نشده).

چی نصب می‌شود (چون `contact-play` / `contact-eval` / rollout واقعاً بهشان
نیاز دارند): `torch`، `mujoco`، `mujoco-warp`، `warp-lang`، `mjlab`،
`rsl-rl-lib` (لودر پالیسی است، فقط ترینینگ نیست)، `viser`، `mediapy` +
`imageio-ffmpeg`، `onnx`/`onnxscript` (به همان دلیلی که در
`docs/VPS_RUNTIME_ONLY.md` آمده).

چی نصب نمی‌شود: `wandb`، `tensorboard`، `torchrunx`، `contact-train`، و
روی این ماشین `transformers`/`torchao`/`pillow` برای Liquid هم نصب
نمی‌شود — چون `remote_decider.py` فقط از کتابخانه‌ی استاندارد پایتون
استفاده می‌کند و `liquid_decider.py` هم `torch`/`transformers` را فقط
داخل `LiquidDecider.load` (مسیر لوکال) ایمپورت می‌کند، نه در سطح ماژول.

## مدل سمت marimo باید روشن باشد

قبل از اجرا، سمت marimo باید روشن باشد (این شاخه آن را روشن نمی‌کند):

1. مدل `LiquidAI/d1-3B` (غیرکوانتومی، `bf16`) با `model_server.py` روی
   marimo بالا باشد — نسخه‌ی کوانتومی `d1-3B-w8a8` در تست قبلی در ساخت
   موتور با خطای `Couldn't swap Linear.weight` می‌شکست و جواب نداد؛
   `bf16` همان تست را از مسیر رله با جواب واقعی گذراند.
2. رله‌کلاینت marimo به رله‌ی VPS وصل باشد تا `GET /health` رله
   `molab_connected True` نشان بدهد.

اگر marimo خاموش شده باشد (نشست حداکثر ۱۲ ساعت، یا ۹۰ دقیقه بی‌کاری)،
سمت ربات جواب نمی‌گیرد و بعد از چند تصمیم ناموفق، ناظر ایمنی (همان
منطق قبلی) ماموریت را با توقف کنترل‌شده تمام می‌کند؛ مدل هیچ‌وقت
به‌جای جواب واقعی، جواب حدسی نمی‌سازد.

## اجرا (چک‌پوینت آماده، بدون ترین)

چک‌پوینت آماده داخل خود ریپو هست:
`weights/go2_contact_2026-10-04_03-24-00/checkpoints/best.pt`؛ پس
`--run-root weights --checkpoint best` را بده (اسکریپت‌ها خودشان همین
را می‌گذارند).

توکن را هیچ‌وقت داخل ریپو، اسکریپت یا لاگ مشترک نگذار. توکن لازم،
توکن **رله** است (ربات ↔ VPS)، نه توکن مدل (آن فقط داخل marimo می‌ماند):

```bash
export LIQUID_API_KEY='<relay-token>'
# LIQUID_API_URL اگر ست نشود، اسکریپت‌ها رله‌ی VPS فعلی را می‌گذارند؛
# اگر رله جابه‌جا شد، همین متغیر را عوض کن، نه کد را.

# ویوئر، دقیقاً مثل اجرای لوکال قبلی ولی با مدل ریموت:
bash scripts/vps/run-remote-play.sh --liquid-target 2.0 0.0 --liquid-stdin True

# یا مستقیم، معادل همان دستور:
uv run --no-sync contact-play Mjlab-Contact-Flat-Unitree-Go2 \
  --run-root weights --checkpoint best --num-envs 1 \
  --liquid True --liquid-decider remote \
  --liquid-mode closed_loop --liquid-decision-hz 1.0 \
  --liquid-target 2.0 0.0 --liquid-stdin True

# رول‌آوت هدلس، بدون ویوئر:
bash scripts/vps/run-remote-rollout.sh \
  --instruction "Move toward the target and turn left." --target 1.5 0.5 --seconds 40
```

فلگ‌های جدید `contact-play` (بقیه‌ی فلگ‌های `--liquid ...` همان قبلی‌اند):

- `--liquid-decider local|remote` (پیش‌فرض `local`؛ برای این شاخه `remote` بده)
- `--liquid-api-url URL` (خالی یعنی `$LIQUID_API_URL`، وگرنه `http://127.0.0.1:8001`)
- `--liquid-api-key KEY` (خالی یعنی `$LIQUID_API_KEY`)
- `--liquid-api-timeout-s S` (پیش‌فرض `1.8`؛ باید زیر مهلت تیک ۲٫۰ ثانیه بماند)

## چه چیزی مثل قبل می‌ماند، چه چیزی عوض شده؟

مثل قبل می‌ماند: سیمولاتور، پالیسی GRU ترین‌شده، `GaitPlanner`،
اعتبارسنجی (`validate`)، `SafetySupervisor`، رد جواب‌های کهنه، و اینکه
روی هر خطای مدل، آخرین دستور معتبر سر جایش می‌ماند. فقط یک تماس
`system_one` جابه‌جا شده: به‌جای لود لوکال مدل، یک درخواست HTTP چند
کیلوبایتی با همان `state` و همان سؤال‌های `type`/`instructions`/`criteria`
می‌رود و فقط دیکشنری `answers` برمی‌گردد؛ هیچ تصویر، تنسور یا وزنی
فرستاده نمی‌شود.

عوض شده: سؤال‌ها باید با فرمت درست مدل باشند
(`{"type", "instructions", "criteria"}`)؛ فرمت قدیمی نوت‌بوک
(`{"type": "choice", "options": [...]}`) را مدل واقعی با
`KeyError: 'instructions'` رد می‌کند (موک قبلاً آن را تحمل می‌کرد).
سؤال‌های خود runtime از اول با همین فرمت درست ساخته می‌شوند.

## محدودیت‌های این نسخه (برای تست، نه محصول نهایی)

- انتقال با HTTP ساده است، بدون TLS، و توکن تستی فقط در متغیر محیطی
  است. قبل از استفاده‌ی واقعی، توکن‌ها را تازه کن، HTTPS بگذار و رله را
  به `systemd` ببر (الان رله با `nohup` است و بعد از ری‌بوت VPS خودش
  برنمی‌گردد).
- اولین استنتاج بعد از بالا آمدن مدل کند است (در تست حدود ۲۱ ثانیه)؛
  بعد از گرم شدن، warmup/mission/tick در تست از مسیر رله به‌ترتیب
  ۰٫۲۱ / ۰٫۵۶ / ۰٫۱۴ ثانیه شدند که داخل مهلت رله (۱٫۵ ثانیه) جا می‌شوند.
- حلقه‌ی بسته‌ی کامل ربات (سیمولاتور + پالیسی + مدل ریموت) هنوز روی
  این مسیر تست نشده؛ تا حالا فقط خود استنتاج مدل از مسیر رله (با
  سؤال‌های ترجمه‌شده به فرمت درست) ثابت شده است.
