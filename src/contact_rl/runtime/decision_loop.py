"""Non-blocking decision worker: latest-wins, bounded, stale-rejecting.

* The simulation thread calls :meth:`DecisionLoop.submit` (never blocks): the
  request goes into a single-slot queue, replacing any request that has not
  started yet (superseded).
* One worker thread loads the model once (lazily, on :meth:`start`) and runs
  ``decide(request)``. The worker never touches simulation state.
* Results are handed to ``deliver`` (the viewer's action queue). Results that
  are late (``> timeout_s``), or whose generation is older than the latest user
  command, are dropped. The sim thread re-checks with :meth:`is_current`
  before applying, so a slow answer can never overwrite a newer command.
* One-shot mode: decisions are only requested when a user command arrives.
* Closed-loop mode additionally submits low-rate ``tick`` requests carrying a
  feedback snapshot (``snapshot_seq``) and the ``mission_id``. A tick never
  bumps the generation (only user commands, stop, pause and reset do) and is
  refused while another request is pending or in flight, so there is never
  more than one inference outstanding and no queue can build up. Relative
  instructions still cannot compound: the instruction is interpreted once
  (``kind="mission"``) and ticks only adjust the current action inside the
  mission's bounds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from contact_rl.runtime.decision_schema import (
  Decision,
  DecisionError,
  MissionSpec,
  PlannerState,
)

LOG = logging.getLogger("contact_rl.runtime")


@dataclass(frozen=True)
class Request:
  request_id: int
  generation: int
  text: str
  state: PlannerState
  submitted_at: float
  kind: str = "instruction"  # instruction (one-shot) | mission (interpret once) | tick (closed loop)
  mission_id: int = 0
  snapshot_seq: int = -1
  context: dict | None = None  # tick: decision context built from the feedback snapshot (plain python)


@dataclass(frozen=True)
class Result:
  request: Request
  decision: Decision | MissionSpec
  latency_s: float


class DecisionLoop:
  def __init__(
    self,
    decide: Callable[[Request], Decision],
    deliver: Callable[[Result], None],
    *,
    load: Callable[[], dict] | None = None,
    timeout_s: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
  ):
    self._decide = decide
    self._deliver = deliver
    self._load = load
    self.timeout_s = timeout_s
    self._clock = clock
    self._lock = threading.Lock()
    self._wake = threading.Event()
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._pending: Request | None = None
    self._inflight: tuple[Request, float] | None = None
    self._deadline_flagged: set[int] = set()
    self._next_id = 1
    self._generation = 0
    self._last_applied_id = 0
    self._last_applied_seq = -1
    self.status = "idle"  # idle | loading | ready | busy | unavailable | stopped
    self.last_error: str | None = None
    self.load_report: dict = {}
    self.latencies: deque[float] = deque(maxlen=200)
    self.counts = {"submitted": 0, "delivered": 0, "applied": 0, "superseded": 0, "stale": 0,
                   "timeout": 0, "invalid": 0, "failed": 0, "ticks": 0, "skipped_busy": 0}
    self.consecutive_failures = 0  # failed / invalid / late answers since the last delivered one

  # ------------------------------------------------------------ lifecycle
  def start(self) -> None:
    if self._thread is not None:
      return
    self._stop.clear()
    self.status = "loading" if self._load else "ready"
    self._thread = threading.Thread(target=self._run, name="liquid-decision", daemon=True)
    self._thread.start()

  def stop(self, timeout: float = 5.0) -> None:
    self._stop.set()
    self._wake.set()
    if self._thread is not None:
      self._thread.join(timeout)
    self._thread = None
    self.status = "stopped"

  @property
  def running(self) -> bool:
    return self._thread is not None and self._thread.is_alive()

  # ------------------------------------------------- any-thread, non-blocking
  def submit(self, text: str, state: PlannerState, *, kind: str = "instruction",
             mission_id: int = 0) -> Request | None:
    """Queue a decision for ``text`` (newest user command wins). Never blocks."""
    with self._lock:
      if self.status in ("unavailable", "stopped"):
        return None
      self._generation += 1
      req = Request(self._next_id, self._generation, text, state, self._clock(), kind=kind,
                    mission_id=mission_id if mission_id else self._generation)
      self._next_id += 1
      if self._pending is not None:
        self.counts["superseded"] += 1
      self._pending = req
      self.counts["submitted"] += 1
    self._wake.set()
    return req

  def submit_tick(self, text: str, state: PlannerState, *, mission_id: int, snapshot_seq: int,
                  context: dict) -> Request | None:
    """Queue one closed-loop decision. Refused (``None``) while busy: bounded, never queues."""
    with self._lock:
      if self.status in ("loading", "unavailable", "stopped") or self._pending is not None \
          or self._inflight is not None:
        self.counts["skipped_busy"] += 1
        return None
      req = Request(self._next_id, self._generation, text, state, self._clock(), kind="tick",
                    mission_id=mission_id, snapshot_seq=snapshot_seq, context=context)
      self._next_id += 1
      self._pending = req
      self.counts["submitted"] += 1
      self.counts["ticks"] += 1
    self._wake.set()
    return req

  @property
  def generation(self) -> int:
    with self._lock:
      return self._generation

  @property
  def busy(self) -> bool:
    with self._lock:
      return self._pending is not None or self._inflight is not None

  def inflight_age(self) -> float | None:
    """Wall seconds the current inference has been running (None if idle)."""
    with self._lock:
      return None if self._inflight is None else self._clock() - self._inflight[1]

  def flag_deadline(self) -> bool:
    """Sim side: the in-flight inference passed its deadline. Counted once per request as a
    failure (feeds the controlled-stop fallback); its late answer is discarded on arrival."""
    with self._lock:
      if self._inflight is None or self._inflight[0].request_id in self._deadline_flagged:
        return False
      self._deadline_flagged.add(self._inflight[0].request_id)
      self.counts["timeout"] += 1
      self.consecutive_failures += 1
      self.last_error = f"decision #{self._inflight[0].request_id} exceeded the {self.timeout_s:.2f}s deadline"
    LOG.warning("%s; keeping the last valid command", self.last_error)
    return True

  def median_latency(self) -> float | None:
    lat = sorted(self.latencies)
    return lat[len(lat) // 2] if lat else None

  def supersede(self) -> int:
    """Invalidate every queued or in-flight decision (e.g. a manual stop)."""
    with self._lock:
      self._generation += 1
      if self._pending is not None:
        self.counts["superseded"] += 1
        self._pending = None
      return self._generation

  def is_current(self, res: Result) -> bool:
    """Same generation, newer than anything applied, and (ticks) from a newer snapshot."""
    with self._lock:
      r = res.request
      if r.generation != self._generation or r.request_id <= self._last_applied_id:
        return False
      return r.snapshot_seq < 0 or r.snapshot_seq > self._last_applied_seq

  def mark_applied(self, res: Result) -> None:
    with self._lock:
      self._last_applied_id = max(self._last_applied_id, res.request.request_id)
      self._last_applied_seq = max(self._last_applied_seq, res.request.snapshot_seq)
      self.counts["applied"] += 1

  def reject_stale(self, kind: str = "stale") -> None:
    with self._lock:
      self.counts[kind] = self.counts.get(kind, 0) + 1

  # ---------------------------------------------------------------- worker
  def _run(self) -> None:
    if self._load is not None:
      try:
        self.load_report = self._load() or {}
        self.status = "ready"
        LOG.info("liquid model ready: %s", self.load_report)
      except Exception as e:  # noqa: BLE001
        self.status = "unavailable"
        self.last_error = f"{type(e).__name__}: {e}"
        LOG.error("Liquid AI unavailable, policy keeps its current command: %s", self.last_error)
        return
    while not self._stop.is_set():
      self._wake.wait(0.1)
      self._wake.clear()
      with self._lock:
        req, self._pending = self._pending, None
      if req is not None and not self._stop.is_set():
        self.process(req)

  def process(self, req: Request) -> Result | None:
    """Run one request (worker thread; called directly by tests)."""
    with self._lock:
      if self._pending is req:
        self._pending = None
      if req.generation != self._generation:
        self.counts["superseded"] += 1
        return None
      t0 = self._clock()
      self._inflight = (req, t0)
    self.status = "busy"
    try:
      dec = self._decide(req)
    except DecisionError as e:
      self._fail("invalid", f"invalid decision: {e}")
      return None
    except Exception as e:  # noqa: BLE001  model failure must never reach the sim
      self._fail("failed", f"inference failed: {type(e).__name__}: {e}")
      return None
    finally:
      with self._lock:
        self._inflight = None
      self.status = "ready"
    latency = self._clock() - t0
    self.latencies.append(latency)
    if latency > self.timeout_s:
      with self._lock:
        flagged = req.request_id in self._deadline_flagged
        self._deadline_flagged.discard(req.request_id)
      if flagged:  # already counted by flag_deadline()
        LOG.warning("late decision #%d (%.2fs) discarded", req.request_id, latency)
      else:
        self._fail("timeout", f"decision took {latency:.2f}s > timeout {self.timeout_s:.2f}s; discarded")
      return None
    res = Result(req, dec, latency)
    with self._lock:
      if req.generation != self._generation:
        self.counts["superseded"] += 1
        return None
      self.counts["delivered"] += 1
      self.consecutive_failures = 0
      self.last_error = None
    self._deliver(res)
    return res

  def _fail(self, kind: str, msg: str) -> None:
    with self._lock:
      self.counts[kind] += 1
      self.consecutive_failures += 1
      self.last_error = msg
    LOG.warning("%s; keeping the last valid command", msg)

  def summary(self) -> str:
    lat = sorted(self.latencies)
    p50 = f"{1e3 * lat[len(lat) // 2]:.0f} ms" if lat else "n/a"
    c = self.counts
    return (f"{self.status} | latency p50 {p50} | applied {c['applied']} | superseded {c['superseded']} "
            f"| stale {c['stale']} | invalid {c['invalid']} | failed {c['failed']} | timeout {c['timeout']}"
            + (f" | ticks {c['ticks']} | busy-skips {c['skipped_busy']}" if c["ticks"] else ""))
