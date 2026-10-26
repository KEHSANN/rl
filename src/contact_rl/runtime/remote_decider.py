"""Remote drop-in replacement for ``LiquidDecider.ask``.

The robot/sim side keeps everything it already had -- ``make_decide``,
``decision_schema.validate`` / ``validate_mission``, ``SafetySupervisor``,
stale-answer rejection, the GRU policy and the simulator -- and only the
single ``system_one`` call moves to another machine:

    robot server:  make_decide(remote.ask, limits, yaw_rate, cfg)
    model server:  POST /system_one  ->  model.system_one(state, questions)

Contract (identical to ``LiquidDecider.ask`` in ``liquid_decider.py``):

* ``ask(state, questions)`` takes the same ``state`` (a string for a
  fresh command / mission, a JSON dict for a running-mission tick) and
  the same ``questions`` dict (``type`` / ``instructions`` / ``criteria``),
  and returns only the ``answers`` dict::

      {"speed": {"choice": "increase", "confidence": 0.9, ...}, ...}

  No image is ever sent, no tensor, no model weight. The payload is a
  few kilobytes at ~1 Hz, which is why a remote call is cheap here.
* Any transport, auth, timeout or schema failure raises
  ``RemoteUnavailable`` (a ``RuntimeError``, like ``LiquidUnavailable``).
  Callers must treat it exactly like a local model failure: keep the
  last valid command; never guess a decision. The runtime's safety
  logic then stops the mission after repeated failures, as before.
* The HTTP timeout must stay below the runtime's tick deadline
  (default ``ClosedLoopConfig.deadline_s = 2.0`` in the rollout CLI,
  ``DecisionLoop.timeout_s = 5.0`` for one-shot requests). The default
  here, 1.8 s, is deliberately under the 2.0 s tick deadline so a slow
  answer is dropped by the client before the runtime has to discard it.

Authentication: a Bearer token, read from the ``api_key`` argument or
from the ``LIQUID_API_KEY`` environment variable. The token is never
logged and must never be committed to git; the model server reads it
the same way (see ``model_server.py``).

This module uses only the standard library, on purpose: the robot
server must not need torch / transformers / torchao just to ask a
remote model. That is the whole point of splitting the two machines.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_TIMEOUT_S = 1.8
DEFAULT_API_KEY_ENV = "LIQUID_API_KEY"
MAX_BODY_BYTES = 1_000_000  # state + questions are a few KB; anything larger is a bug


class RemoteUnavailable(RuntimeError):
    """The remote model cannot be reached, refused, timed out, or misbehaved."""


class RemoteDecider:
    """``ask``-compatible client for a ``model_server.py`` instance.

    Usage on the robot server (mirrors the ``mock`` branch of
    ``rollout.py``)::

        from contact_rl.runtime.liquid_decider import make_decide
        from contact_rl.runtime.remote_decider import RemoteDecider

        dec = RemoteDecider("https://model.example.com")  # token from LIQUID_API_KEY
        rt = LiquidRuntime(lambda lim: make_decide(dec.ask, lim, cl.turn_rate, cl),
                           dec.load, limits, mode="closed_loop", closed_loop=cl)

    ``load()`` is a readiness probe, not a model load: it GETs /health
    and raises ``RemoteUnavailable`` unless the server reports
    ``model_loaded`` (or ``mock``). The real model is loaded once on
    the server before it starts serving, so a mission never waits on a
    cold model load mid-run.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL, e.g. https://model.example.com")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"no API key: pass api_key= or set {api_key_env} in the environment"
            )
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s

    # -- readiness probe (fits the ``load`` slot of LiquidRuntime) ----

    def load(self) -> dict:
        """GET /health; raise unless the server reports a ready model."""
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{self.base_url}/health", method="GET"),
                timeout=self.timeout_s,
            ) as resp:
                report = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RemoteUnavailable(f"model server /health returned HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise RemoteUnavailable(f"model server /health failed: {type(e).__name__}: {e}") from e
        if not isinstance(report, dict) or not report.get("ok"):
            raise RemoteUnavailable(f"model server /health not ok: {report!r}"[:200])
        if not (report.get("model_loaded") or report.get("mock")):
            raise RemoteUnavailable("model server reports model_loaded=false (still loading?)")
        return report

    # -- the ask contract ----------------------------------------------

    def ask(self, state, questions: dict) -> dict:
        """POST /system_one and return the ``answers`` dict, like local ``ask``."""
        if not isinstance(questions, dict) or not questions:
            raise RemoteUnavailable("questions must be a non-empty dict")
        payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        if len(payload) > MAX_BODY_BYTES:
            raise RemoteUnavailable(f"request payload {len(payload)} bytes exceeds limit")
        req = urllib.request.Request(
            f"{self.base_url}/system_one",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RemoteUnavailable("model server rejected the API key (HTTP 401)") from e
            detail = e.read(200).decode("utf-8", "replace")
            raise RemoteUnavailable(f"model server HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RemoteUnavailable(
                f"model server unreachable/timed out after {self.timeout_s}s: "
                f"{type(e).__name__}: {e}"
            ) from e
        if len(body) > MAX_BODY_BYTES:
            raise RemoteUnavailable("model server response exceeds limit")
        try:
            out = json.loads(body.decode("utf-8"))
        except ValueError as e:
            raise RemoteUnavailable(f"model server returned non-JSON: {e}") from e
        if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
            raise RemoteUnavailable(f"unexpected model server output: {str(out)[:200]!r}")
        answers = out["answers"]
        for name, ans in answers.items():
            if not isinstance(ans, dict) or not isinstance(ans.get("choice"), str):
                raise RemoteUnavailable(f"answer {name!r} has no string 'choice': {str(ans)[:100]!r}")
        return answers
