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

Closed-loop mode (``--liquid-mode closed_loop``, default with ``--liquid``):
the instruction is interpreted once into a persistent ``MissionSpec``
(``mission.py``); after every env step ``LiquidRuntime.on_sim_step`` runs the
deterministic ``SafetySupervisor`` and, at a low latency-bounded rate, submits a
``RobotFeedback`` snapshot as a ``tick`` request whose answer is validated and
checked for staleness before it reaches the planner. ``rollout.py`` runs the
same loop headless against the real env and checkpoint.

The worker thread never touches simulation tensors; every read and write of
planner state happens on the viewer's simulation thread.
"""
