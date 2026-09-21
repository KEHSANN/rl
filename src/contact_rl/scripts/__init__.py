"""Entry-point wrappers for training / playing the contact-explicit task.

mjlab's ``train`` / ``play`` CLIs only import ``mjlab.tasks`` to populate the
shared task registry, so third-party tasks are invisible to them by default.
These thin wrappers import :mod:`contact_rl` first -- registering the contact
tasks into the *same* ``mjlab.tasks.registry._REGISTRY`` -- and then delegate to
mjlab's own ``main`` (whose ``list_tasks()`` call now sees the contact tasks).
"""
