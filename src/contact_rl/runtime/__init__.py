"""Runtime-only decision layer (Liquid AI d1-3B-w8a8 -> GaitPlanner commands).

Nothing in this package is imported by training, evaluation or the policy
itself. It is only reached through ``contact-play --liquid True``.

Data flow::

  user text (Viser box / stdin)
    -> viewer action queue (sim thread) -> PlannerState snapshot
    -> DecisionLoop worker thread -> LiquidDecider (d1 Decision Index answers)
    -> answers_to_raw -> decision_schema.validate          (deterministic)
    -> viewer action queue (sim thread) -> stale / superseded check
    -> command_override.apply_to_command_term -> GaitPlanner params
    -> contact goals -> unchanged trained policy

The worker thread never touches simulation tensors; every read and write of
planner state happens on the viewer's simulation thread.
"""
