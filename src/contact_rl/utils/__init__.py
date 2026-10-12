"""Runtime / deployment helpers for contact-rl (remote GPU VPS workflow).

Modules here must stay importable without a GPU. ``runtime`` and
``checkpoints`` import neither torch nor mujoco at module level so they can be
used (and unit-tested) before the heavy stack is loaded.
"""
