"""Non-blocking decision scheduler: latest-wins, stale-rejecting, fail-safe.

The worker thread runs the (slow) model call. The simulation thread only calls
``latest()`` / ``poll_and_apply()``, which never waits on inference.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from contact_rl.runtime.decision_schema import Decision, DecisionError, Limits, validate

LOG = logging.getLogger("contact_rl.runtime")


@dataclass
class DecisionRecord:
  command_id: int
  decision: Decision
  latency_s: float
  finished_at: float


class DecisionLoop:
  def __init__(self, infer: Callable[[int], dict], *, period_s: float = 2.0,
               limits_kwargs: dict | None = None, clock: Callable[[], float] = time.monotonic):
    self._infer = infer  # infer(command_id) -> raw dict; may raise
    self._period = period_s
    self._clock = clock
    self._lock = threading.Lock()
    self._next_id = 1
    self._pending_id: int | None = None
    self._current: Decision | None = None
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._limits_kwargs = limits_kwargs or {}
    self._applied_marker: int | None = None
    self.last_error: str | None = None
    self.applied: list[DecisionRecord] = []
    self.rejected_stale = 0
    self.rejected_invalid = 0

  # ------------------------------------------------------------ control
  def start(self) -> None:
    self._thread = threading.Thread(target=self._run, name="decision-loop", daemon=True)
    self._thread.start()

  def stop(self, timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout)

  def _run(self) -> None:
    while not self._stop.is_set():
      self.step_once()
      self._stop.wait(self._period)

  # -------------------------------------------------------------- core
  def step_once(self) -> None:
    """One inference round (called by the worker thread; also used in tests)."""
    with self._lock:
      cid = self._next_id
      self._next_id += 1
      self._pending_id = cid
    t0 = self._clock()
    try:
      raw = self._infer(cid)
      latency = self._clock() - t0
      with self._lock:
        prev = self._current
      dec = validate(raw, cid, prev, Limits(**self._limits_kwargs))
    except DecisionError as e:  # invalid output: keep the last valid command
      self.rejected_invalid += 1
      self.last_error = f"invalid decision: {e}"
      LOG.warning("%s; keeping last valid command", self.last_error)
      return
    except Exception as e:  # model failure: never crash the simulation
      self.rejected_invalid += 1
      self.last_error = f"inference failed: {type(e).__name__}: {e}"
      LOG.warning("%s; keeping last valid command", self.last_error)
      return
    self._offer(DecisionRecord(cid, dec, latency, self._clock()))

  def _offer(self, rec: DecisionRecord) -> None:
    with self._lock:
      # Stale: a newer id has already been applied. Never overwrite newer with older.
      if self.applied and rec.command_id <= self.applied[-1].command_id:
        self.rejected_stale += 1
        return
      self._current = rec.decision
      self.applied.append(rec)
      self.last_error = None

  # ---------------------------------------------------- sim-thread API
  def latest(self) -> Decision | None:
    with self._lock:
      return self._current

  def poll_and_apply(self, apply: Callable[[Decision], None]) -> bool:
    """Called from the simulation thread between steps. Applies a new decision
    exactly once; never blocks on inference."""
    with self._lock:
      if not self.applied:
        return False
      rec = self.applied[-1]
      if self._applied_marker == rec.command_id:
        return False
      self._applied_marker = rec.command_id
    apply(rec.decision)
    return True
