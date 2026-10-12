"""``contact-*`` command-line entry points (registered in ``[project.scripts]``).

=================  ===============================  ==========================================
command            module                           purpose
=================  ===============================  ==========================================
contact-train      :mod:`contact_rl.scripts.train`     training with run management
contact-eval       :mod:`contact_rl.scripts.evaluate`  per-episode evaluation (Fig. 6 protocol)
contact-watch      :mod:`contact_rl.scripts.watch`     evaluates new checkpoints of a running job
contact-play       :mod:`contact_rl.scripts.play`      viewer / video for a checkpoint
contact-doctor     :mod:`contact_rl.scripts.doctor`    machine-readiness diagnostics
contact-bench      :mod:`contact_rl.scripts.benchmark` hardware report + env-throughput sweep
=================  ===============================  ==========================================

mjlab's own ``train`` / ``play`` CLIs only import ``mjlab.tasks`` to populate
the shared task registry, so third-party tasks are invisible to them. Every
command here imports :mod:`contact_rl` first, which selects the headless GL
backend before MuJoCo is imported and registers the contact tasks into the
*same* ``mjlab.tasks.registry``. ``contact-train`` / ``contact-play`` then
parse mjlab's config types (all mjlab ``--env.*`` / ``--agent.*`` flags work)
but run the project's own training / viewer code; they are no longer thin
wrappers around mjlab's ``main``. Only a real multi-GPU run
(``--gpu-ids "[0, 1]"`` or ``all`` with several visible GPUs -- note the
quoted Python list syntax) is handed to mjlab's launcher.

Checkpoints live in ``runs/<experiment>/<timestamp>[_name]/checkpoints/``
(``model_<it>.pt``, ``latest.pt``, ``best.pt``, ``index.json``); selectors
accepted by ``--checkpoint`` / ``--resume-from``: a ``.pt`` file, a run dir,
``<run_dir>:best|latest|<it>``, ``best``, ``latest``. mjlab's legacy
``logs/rsl_rl/<experiment>/<run>/model_<it>.pt`` layout is only produced by
the multi-GPU hand-off (and read by ``--agent.resume True``).

On the VPS run every command with ``UV_NO_SYNC=1`` (exported by
``scripts/vps/common.sh`` and the systemd units) or as ``uv run --no-sync``:
a plain ``uv run`` re-syncs the env without ``--extra cu128`` and replaces
the CUDA torch build installed by ``scripts/vps/setup.sh``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

HELP_FLAGS = ("-h", "--help")


def taskless_help(prog: str, tasks: Sequence[str], argv: Sequence[str], usage: str = "") -> None:
  """Handle ``<cmd> --help`` for the two task-positional CLIs.

  ``contact-train`` / ``contact-play`` parse the task id in a first
  :func:`tyro.cli` pass configured with ``add_help=False`` (the help flag must
  reach the *second* pass, which knows the task's options). Without a task id
  that first pass therefore never sees ``--help``, it only sees a missing
  required positional, and the command dies with ``Missing value for argument
  'value'`` -- exit 2, no help. Print a usable top-level help instead; the full
  option list stays behind ``<cmd> <task> --help`` because it is built from the
  selected task's env / agent config.

  Returns normally (so parsing continues) unless help without a task id was
  requested, in which case it raises ``SystemExit(0)``.
  """
  if not any(a in HELP_FLAGS for a in argv) or any(a in tasks for a in argv):
    return
  name = Path(prog).name
  lines = [f"usage: {name} TASK [OPTIONS]", "", "task ids:"]
  lines += [f"  {t}" for t in tasks]
  if usage:
    lines += ["", usage.strip("\n")]
  lines += ["", f"Options depend on the task; see:  {name} {tasks[0] if tasks else 'TASK'} --help"]
  print("\n".join(lines))
  raise SystemExit(0)
