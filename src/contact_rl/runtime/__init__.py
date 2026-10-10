"""Runtime-only decision layer (Liquid AI d1-3B-W8A8 -> GaitPlanner commands).

Nothing in this package is imported by the training path. It produces
validated UserCommand-compatible dicts; the simulation thread applies them
through command_override.apply_to_command_term.
"""
