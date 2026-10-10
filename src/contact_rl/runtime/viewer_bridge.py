"""Glue between :class:`DecisionLoop` and the running ``ContactPlayViewer``.

Every method named ``on_*`` runs on the viewer's simulation thread (it is
called from ``_handle_custom_action``, i.e. from ``BaseViewer._process_actions``).
GUI callbacks, the stdin reader and the decision worker only *enqueue*
viewer actions; they never read or write simulation tensors.

Commands reach the planner exclusively through
``command_override.apply_to_command_term`` (the existing interactive entry
point). The policy, its observations, actions and GRU state are untouched.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import replace
from typing import Any, Callable

from contact_rl.runtime.decision_loop import DecisionLoop, Result
from contact_rl.runtime.decision_schema import Decision, Limits, PlannerState, to_user_command_kwargs

LOG = logging.getLogger("contact_rl.runtime")
MAX_TEXT = 300
ACTION_PREFIX = "liquid_"


def planner_state(term, ids, env_idx: int) -> PlannerState:
  """Snapshot (plain floats) of the command of ``env_idx`` + gaits of ``ids``. Sim thread only."""
  import torch

  from contact_rl.tasks.contact.mdp.command_override import describe_command
  from contact_rl.tasks.contact.mdp.planning import GAITS

  pl = term.planner
  d = describe_command(pl, env_idx)
  sel = pl.gait if ids is None else pl.gait[ids]
  gaits = tuple(GAITS[int(g)] for g in torch.unique(sel).tolist())
  return PlannerState(d["gait"], d["speed_mps"], float(pl.heading_off[env_idx]), d["yaw_rate_rps"], gaits)


class LiquidRuntime:
  def __init__(self, decide_factory: Callable[[Limits], Callable], load: Callable[[], dict] | None,
               limits: Limits = Limits(), timeout_s: float = 5.0):
    self._decide_factory = decide_factory
    self._load = load
    self.limits = limits
    self.timeout_s = timeout_s
    self.loop: DecisionLoop | None = None
    self._request_action: Callable[[str, Any], None] | None = None
    self._term = self._ids = self._env_idx = None
    self.last_text = ""
    self.last_decision: Decision | None = None
    self.last_msg = ""
    self.use_stdin = False

  # ------------------------------------------------------------ lifecycle
  def attach(self, request_action, term_getter, ids_getter, env_idx_getter) -> None:
    self._request_action = request_action
    self._term, self._ids, self._env_idx = term_getter, ids_getter, env_idx_getter
    # Caps use the live command duration of this env, not the nominal constant.
    self.limits = replace(self.limits, switch_dt=float(term_getter().planner.switch_dt))
    self.loop = DecisionLoop(self._decide_factory(self.limits), self._deliver, load=self._load,
                             timeout_s=self.timeout_s)

  def start(self) -> None:
    assert self.loop is not None, "attach() first"
    self.loop.start()
    if self.use_stdin:
      self.start_stdin_reader()

  def stop(self) -> None:
    if self.loop is not None:
      self.loop.stop()

  def start_stdin_reader(self, stream=None) -> threading.Thread:
    """Read instructions line by line (``stop`` = immediate stop without the model)."""
    stream = stream or sys.stdin

    def _read() -> None:
      for line in stream:
        text = line.strip()
        if not text:
          continue
        if text.lower() in ("stop", "/stop"):
          self.send_stop()
        else:
          self.send_text(text)

    t = threading.Thread(target=_read, name="liquid-stdin", daemon=True)
    t.start()
    print("[liquid] type instructions + Enter (e.g. 'walk forward slowly'); 'stop' halts without the model.",
          flush=True)
    return t

  # ------------------------------------------------- any thread: enqueue only
  def send_text(self, text: str) -> None:
    self._request_action("CUSTOM", ("liquid_cmd", str(text)))

  def send_stop(self) -> None:
    self._request_action("CUSTOM", ("liquid_stop", None))

  def _deliver(self, res: Result) -> None:  # worker thread
    self._request_action("CUSTOM", ("liquid_result", res))

  # ------------------------------------------------------- sim thread only
  def handle(self, kind: str, payload) -> str:
    if kind == "liquid_cmd":
      return self.on_command(payload)
    if kind == "liquid_result":
      return self.on_result(payload)
    if kind == "liquid_stop":
      return self.on_stop()
    return f"unknown liquid action {kind}"

  def on_command(self, text) -> str:
    text = " ".join(str(text).split())[:MAX_TEXT]
    if not text:
      return self._set("empty instruction ignored")
    st = planner_state(self._term(), self._ids(), self._env_idx())
    req = self.loop.submit(text, st)
    if req is None:
      return self._set(f"Liquid AI unavailable ({self.loop.last_error}); command not sent")
    self.last_text = text
    return self._set(f"#{req.request_id} sent: {text!r}")

  def on_result(self, res: Result) -> str:
    if not self.loop.is_current(res):
      self.loop.reject_stale()
      return self._set(f"#{res.request.request_id} discarded (superseded by a newer command)")
    dec = res.decision
    self.loop.mark_applied(res)
    self.last_decision = dec
    if dec.is_noop:
      return self._set(f"#{dec.request_id} no change ({dec.reason})")
    warnings = self._apply(dec)
    note = "; ".join(dec.notes + tuple(warnings))
    return self._set(f"#{dec.request_id} applied in {res.latency_s * 1e3:.0f} ms: {self.describe(dec)}"
                     + (f" [{note}]" if note else ""))

  def on_stop(self) -> str:
    self.loop.supersede()  # drop any in-flight model answer
    dec = Decision(request_id=0, generation=0, speed=0.0, yaw_rate=0.0, stop=True, reason="manual stop")
    self._apply(dec)
    self.last_decision = dec
    return self._set("STOP applied (no model): speed 0, yaw rate 0")

  def _apply(self, dec: Decision) -> list[str]:
    from contact_rl.tasks.contact.mdp.command_override import UserCommand, apply_to_command_term

    cmd = UserCommand(**to_user_command_kwargs(dec))
    # Gait changes need a re-anchored plan; other parameters take effect at the
    # next contact switch (no plan discontinuity).
    return apply_to_command_term(self._term(), cmd, self._ids(), reanchor_now=dec.gait is not None)

  # --------------------------------------------------------------- status
  @staticmethod
  def describe(d: Decision) -> str:
    if d.stop:
      return "STOP"
    parts = []
    if d.gait is not None:
      parts.append(d.gait)
    if d.speed is not None:
      parts.append(f"v={d.speed:.2f} m/s")
    if d.heading_offset is not None:
      parts.append(f"dir={d.heading_offset:.2f} rad")
    if d.yaw_rate is not None:
      parts.append(f"yaw={d.yaw_rate:.2f} rad/s")
    return ", ".join(parts) or "no change"

  def _set(self, msg: str) -> str:
    self.last_msg = msg
    LOG.info("[liquid] %s", msg)
    return msg

  def status_line(self) -> str:
    if self.loop is None:
      return "not attached"
    s = self.loop.summary()
    if self.loop.last_error:
      s += f" | last error: {self.loop.last_error}"
    return s


def build_runtime(*, model_id: str, device: str, compile: bool, limits: Limits, yaw_rate: float,
                  timeout_s: float) -> LiquidRuntime:
  """Real d1 runtime. Nothing heavy is imported until the worker loads the model."""
  if abs(yaw_rate) > limits.yaw_rate_max:
    raise ValueError(f"--liquid-yaw-rate {yaw_rate} exceeds --liquid-yaw-max {limits.yaw_rate_max}")
  from contact_rl.runtime.liquid_decider import LiquidDecider

  dec = LiquidDecider(model_id, device, compile)
  return LiquidRuntime(lambda lim: dec.make_decide(lim, yaw_rate), dec.load, limits, timeout_s)
