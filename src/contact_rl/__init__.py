"""Contact-explicit multi-task robot learning (mjlab implementation).

Implementation of "Learning to Act Through Contact: A Unified View of Multi-Task
Robot Learning" (Omar & Khadiv, arXiv:2510.03599v2) on top of mjlab.

Importing this package registers the task(s) with mjlab's shared task registry
(``mjlab.tasks.registry``). Because mjlab's ``train`` / ``play`` CLIs only import
``mjlab.tasks`` (not third-party packages), the entry-point scripts in
``contact_rl.scripts`` import this package first so the contact tasks become
visible to those CLIs.

Headless rendering: the task import below pulls in mjlab -> mujoco, and MuJoCo
fixes its OpenGL backend from ``MUJOCO_GL`` at import time. The backend is
therefore selected here, *before* that import (``import contact_rl.<anything>``
always runs this file first). mjlab's own ``train`` sets ``MUJOCO_GL=egl`` only
after mujoco is already imported, which is too late.
"""

from __future__ import annotations

from contact_rl.utils.runtime import configure_headless_rendering as _configure_gl

_configure_gl()

# Importing the tasks package triggers register_mjlab_task for every robot
# config (via import_packages), populating mjlab's shared _REGISTRY.
#
# A broken simulator stack must not make the *diagnostic* tooling unimportable
# (``contact-doctor`` lives in this package), so the error is kept, reported on
# stderr, and re-raised by :func:`require_tasks`, which every train / eval /
# play entry point calls before using the registry.
TASKS_IMPORT_ERROR: BaseException | None = None
_TASKS_TRACEBACK = ""
try:
  from contact_rl import tasks as tasks  # noqa: E402, F401
except Exception as _e:  # noqa: BLE001
  import sys as _sys
  import traceback as _tb

  TASKS_IMPORT_ERROR = _e
  _TASKS_TRACEBACK = _tb.format_exc()
  print(f"[contact_rl] WARNING: task registration failed: {type(_e).__name__}: {_e}", file=_sys.stderr)


def require_tasks() -> None:
  """Raise (with the original traceback) if the contact tasks did not register."""
  if TASKS_IMPORT_ERROR is not None:
    raise RuntimeError(
      "contact_rl tasks failed to import; run `uv run contact-doctor`.\n" + _TASKS_TRACEBACK
    ) from TASKS_IMPORT_ERROR


__all__ = ["tasks", "require_tasks", "TASKS_IMPORT_ERROR"]
