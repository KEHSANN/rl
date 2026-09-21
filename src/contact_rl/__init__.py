"""Contact-explicit multi-task robot learning (mjlab implementation).

Implementation of "Learning to Act Through Contact: A Unified View of Multi-Task
Robot Learning" (Omar & Khadiv, arXiv:2510.03599v2) on top of mjlab.

Importing this package registers the task(s) with mjlab's shared task registry
(``mjlab.tasks.registry``). Because mjlab's ``train`` / ``play`` CLIs only import
``mjlab.tasks`` (not third-party packages), the entry-point scripts in
``contact_rl.scripts`` import this package first so the contact tasks become
visible to those CLIs.
"""

from __future__ import annotations

# Importing the tasks package triggers register_mjlab_task for every robot
# config (via import_packages), populating mjlab's shared _REGISTRY.
from contact_rl import tasks as tasks  # noqa: F401

__all__ = ["tasks"]
