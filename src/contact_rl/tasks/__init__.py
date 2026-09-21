"""Contact-explicit tasks package.

Recursively imports all task sub-packages so that each robot ``config``
sub-package runs its ``register_mjlab_task`` call on import (same mechanism
mjlab uses for its own ``mjlab.tasks``). The ``.mdp`` sub-packages are skipped
here -- they are imported on demand by the env configs.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
