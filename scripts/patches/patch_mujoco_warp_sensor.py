"""Apply the verified mujoco-warp 3.5.0.2 sensor codegen fix after uv sync."""

import hashlib
import importlib.metadata
import os
import tempfile
from pathlib import Path

PACKAGE = "mujoco-warp"
VERSION = "3.5.0.2"
ORIGINAL_SHA256 = "642c858f219d5383d0ee6944a34b323c2b7d5154b763370e5934e95e06453f4c"
PATCHED_SHA256 = "cd9a0010080411f5c0ef3c90599d2bb47a751182fc8c2bd72149923fbe206ffe"
OLD = b"""  else:  # UNKNOWN
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
"""
NEW = b"""  else:  # UNKNOWN
    xmat = wp.identity(3, dtype=wp.float32)
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
"""


def main() -> None:
    distribution = importlib.metadata.distribution(PACKAGE)
    if distribution.version != VERSION:
        raise SystemExit(
            f"{PACKAGE} version {distribution.version} is unsupported; expected {VERSION}"
        )

    path = Path(distribution.locate_file("mujoco_warp/_src/sensor.py"))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    if digest == PATCHED_SHA256:
        print(f"{PACKAGE} sensor fix already applied: {path}")
        return
    if digest != ORIGINAL_SHA256:
        raise SystemExit(f"Unexpected {PACKAGE} sensor.py SHA256 {digest}; refusing to patch")
    if data.count(OLD) != 1:
        raise SystemExit("Expected exactly one UNKNOWN axis block; refusing to patch")

    patched = data.replace(OLD, NEW)
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise SystemExit("Patched file hash mismatch; refusing to write")

    fd, temporary = tempfile.mkstemp(prefix=".sensor.py.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(patched)
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(f"Applied {PACKAGE} {VERSION} sensor fix: {path}")


if __name__ == "__main__":
    main()
