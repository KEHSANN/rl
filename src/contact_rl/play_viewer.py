"""Viser play viewer with contact-command control, checkpoint picker and
correct recurrent-state handling.

Extends mjlab's :class:`ViserPlayViewer` (all its controls stay: pause, step,
reset, speed, env selection, plots) with:

* **Contact control**: gait, direction (heading offset), speed (-> stride via
  the planner), turning (yaw rate), apply to all envs or the selected one,
  restore the trained command sampling. Commands go through
  :func:`contact_rl.tasks.contact.mdp.command_override.apply_to_command_term`
  -> ``GaitPlanner`` -> contact goals -> policy observation (the training
  pathway; no second command system). Out-of-distribution requests are shown
  in the panel and logged, not clamped.
* **Checkpoint picker**: switch between checkpoints of the run without
  restarting (actor weights only; env + GRU state are reset).
* **Restart episode**: resets all envs and the GRU hidden state.
* **GRU reset on termination**: after every step the hidden state of envs
  whose episode ended is zeroed (mjlab's viewer never does this).

All GUI callbacks only enqueue actions; they are executed on the simulation
thread by ``BaseViewer._process_actions``.
"""

from __future__ import annotations

import html
import math
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser import ViserPlayViewer

from contact_rl.tasks.contact.mdp.command_override import (
  KEEP_TRAINED_GAIT,
  apply_to_command_term,
  build_user_command,
  describe_command,
  max_train_speed,
  restore_sampled_commands,
)
from contact_rl.tasks.contact.mdp.planning import GAITS
from contact_rl.utils.policy_state import reset_recurrent_state

TRAINED = KEEP_TRAINED_GAIT


class ContactPlayViewer(ViserPlayViewer):
  def __init__(
    self,
    env,
    policy,
    *,
    viser_server,
    load_policy: Callable[[Path], Any] | None = None,
    checkpoints: Callable[[], list[Path]] | None = None,
    current_checkpoint: Path | None = None,
    **kw,
  ):
    super().__init__(env, policy, viser_server=viser_server, **kw)
    self._load_policy = load_policy
    self._list_ckpts = checkpoints or (lambda: [])
    self._ckpt = current_checkpoint
    self._ckpt_map: dict[str, Path] = {}
    self._warnings: list[str] = []
    self._episode_ends = 0
    self._gru_resets = 0
    self._msg = ""

  # ---------------------------------------------------------------- GUI

  def _term(self):
    return self.env.unwrapped.command_manager.get_term("contact")

  def _ckpt_label(self, p: Path) -> str:
    return f"{p.parent.parent.name}/{p.name}"

  def _refresh_ckpts(self) -> list[str]:
    paths = self._list_ckpts()
    if self._ckpt is not None and self._ckpt not in paths:
      paths = [self._ckpt, *paths]
    self._ckpt_map = {self._ckpt_label(p): p for p in paths}
    return list(self._ckpt_map) or ["(none)"]

  def setup(self) -> None:
    super().setup()
    gui = self._server.gui
    term = self._term()
    s = term.planner.switch_dt
    with gui.add_folder("Contact control"):
      self._ctl_html = gui.add_html("")
      self._gait = gui.add_dropdown("Gait", options=(TRAINED, *GAITS), initial_value=TRAINED)
      self._heading = gui.add_slider("Direction (deg)", min=-180.0, max=180.0, step=5.0, initial_value=0.0)
      vmax = max(max_train_speed(g, s) for g in GAITS)
      self._speed = gui.add_slider("Speed (m/s)", min=0.0, max=round(2.0 * vmax, 2), step=0.02, initial_value=0.2)
      self._yaw = gui.add_slider("Turning (rad/s)", min=-2 * math.pi, max=2 * math.pi, step=0.05, initial_value=0.0)
      self._all_envs = gui.add_checkbox("Apply to all envs", initial_value=True)
      b_apply = gui.add_button("Apply command")
      b_restore = gui.add_button("Restore trained commands")
      b_restart = gui.add_button("Restart episode (reset GRU)")

      @b_apply.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", ("apply", None))

      @b_restore.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", ("restore", None))

      @b_restart.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", ("restart", None))

    with gui.add_folder("Checkpoint"):
      opts = self._refresh_ckpts()
      init = self._ckpt_label(self._ckpt) if self._ckpt is not None else opts[0]
      self._ckpt_dd = gui.add_dropdown("Checkpoint", options=opts, initial_value=init)
      b_load = gui.add_button("Load selected")
      b_refresh = gui.add_button("Refresh list")

      @b_load.on_click
      def _(_) -> None:
        self.request_action("CUSTOM", ("load", self._ckpt_dd.value))

      @b_refresh.on_click
      def _(_) -> None:
        self._ckpt_dd.options = self._refresh_ckpts()

  # -------------------------------------------------------------- actions

  def _ids(self) -> torch.Tensor | None:
    if self._all_envs.value:
      return None
    return torch.tensor([int(self._scene.env_idx)], device=self._term().planner.device)

  def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
    if action != ViewerAction.CUSTOM or not isinstance(payload, tuple):
      return False
    kind, arg = payload
    try:
      with self._sim_lock:
        if kind == "apply":
          cmd = build_user_command(self._gait.value, self._heading.value, self._speed.value, self._yaw.value)
          self._warnings = apply_to_command_term(self._term(), cmd, self._ids())
          self._msg = "command applied" + (" (OUT OF TRAINING DISTRIBUTION)" if self._warnings else "")
        elif kind == "restore":
          self._warnings = []
          self._msg = "trained commands restored" if restore_sampled_commands(self._term(), self._ids()) \
            else "nothing to restore"
        elif kind == "restart":
          self._msg = "episode restarted"
        elif kind == "load":
          path = self._ckpt_map.get(arg)
          if path is None or self._load_policy is None:
            self._msg = f"unknown checkpoint {arg}"
            return True
          self.policy = self._load_policy(path)
          self._ckpt = path
          self._msg = f"loaded {arg}"
        else:
          return False
      if kind in ("restart", "load"):
        self.reset_environment()  # env reset + policy.reset() (GRU state dropped)
        self._gru_resets += 1
    except Exception:
      self._msg = "error: " + traceback.format_exc().strip().splitlines()[-1]
      print(traceback.format_exc())
    self._update_status_display()
    return True

  # ---------------------------------------------------------------- step

  def _execute_step(self) -> bool:
    try:
      with torch.no_grad():
        obs = self.env.get_observations()
        actions = self.policy(obs)
        _, _, dones, _ = self.env.step(actions)
        if reset_recurrent_state(self.policy, dones):
          self._gru_resets += 1
          self._episode_ends += int((dones > 0).sum())
        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(f"[ERROR] Exception during step:\n{self._last_error}", 0)
      self.pause()
      return False

  # -------------------------------------------------------------- status

  def _update_status_display(self) -> None:
    super()._update_status_display()
    if not hasattr(self, "_ctl_html"):
      return
    try:
      i = int(self._scene.env_idx)
      d = describe_command(self._term().planner, i)
      warn = "".join(f'<br/><span style="color:#e67e22;">&#9888; {html.escape(w)}</span>' for w in self._warnings)
      ck = html.escape(self._ckpt_label(self._ckpt)) if self._ckpt else "(none)"
      self._ctl_html.content = f"""
        <div style="font-size:0.85em;line-height:1.3;padding:0 1em 0.5em 1em;">
          <strong>Env {i}:</strong> {d['gait']} | v={d['speed_mps']:.2f} m/s (stride {d['stride_m']:.2f} m)<br/>
          heading {d['heading_offset_deg']:.0f} deg | yaw rate {d['yaw_rate_rps']:.2f} rad/s<br/>
          <strong>Checkpoint:</strong> {ck}<br/>
          <strong>Episode ends:</strong> {self._episode_ends} | <strong>GRU resets:</strong> {self._gru_resets}<br/>
          {html.escape(self._msg)}{warn}
        </div>"""
    except Exception:
      pass
