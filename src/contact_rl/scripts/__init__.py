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
