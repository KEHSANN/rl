"""Model-side HTTP server for LiquidAI/d1-3B-w8a8 (stdlib only, no FastAPI).

Why stdlib: the model machine already needs torch / transformers /
torchao for the model itself; the server adds no further dependency,
so it runs unchanged in a molab notebook, on a plain GPU VPS, or
locally for testing. If you prefer FastAPI, the contract below is the
same -- only the transport changes.

Endpoints
---------
GET  /health
    {"ok": true, "model_id": ..., "model_loaded": bool, "mock": bool}
    No auth: it reveals nothing but readiness. The robot server's
    ``RemoteDecider.load()`` polls this before a mission starts.

POST /system_one
    Request:  {"state": <str | JSON>, "questions": {name: {type, instructions, criteria}}}
    Response: {"answers": {name: {"choice", "confidence", "probabilities"}}, "usage": {...}}
    Auth: ``Authorization: Bearer <token>`` is required unless the
    server was started with ``--allow-no-auth`` (local testing only).
    The token is compared in constant time and is never logged.

    Calls are serialised with one lock: d1 inference is a single
    forward pass on one GPU, and the runtime already guarantees at
    most one outstanding request per robot. Concurrent robots queue
    behind the lock instead of corrupting a shared model call.

Security notes (read before exposing this to the internet)
------------------------------------------------------------
* This server speaks plain HTTP. A Bearer token over HTTP is visible
  to anyone on the path, so in production put it behind a TLS reverse
  proxy (nginx / Caddy) and never publish the raw port. Bind
  ``127.0.0.1`` when the proxy runs on the same machine.
* The token must come from the environment (``LIQUID_API_KEY``) or
  ``--api-key`` at process start, never from a file committed to git
  and never from a notebook synced from GitHub.
* molab caveat, verified from its docs, not from a test here: molab
  notebooks are public-but-undiscoverable and sessions shut down
  after 12 hours / 90 minutes idle. That makes molab a prototype host
  for this server, not a reliable production one, and it does not
  advertise a custom public port. Test port reachability from the
  robot server with curl before relying on molab; if the port is not
  reachable, run this same file on a GPU VPS instead.

Real-model loading mirrors ``LiquidDecider.load`` exactly: CUDA +
sm_80+ (Ampere, INT8 tensor cores), ``AutoModel.from_pretrained(id,
trust_remote_code=True).to("cuda")``, ``model.eval()``, a
``system_one`` check, then ``model.compile(mode="reduce-overhead")``
unless ``--no-compile``. The model is loaded BEFORE the server starts
listening, so the first real request never pays the load time (which
can be minutes) inside the robot's 2-second tick deadline.

``--mock`` replaces ``system_one`` with a small deterministic stub
(same shape as ``rollout.mock_ask``, simplified): mission questions
get a walk/reach answer from the command text, tick questions get
keep/straight/keep/continue, one-shot questions get "keep" for every
question. It is for testing the transport and the client only --
never a statement about what the real model would decide.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "LiquidAI/d1-3B-w8a8"
INSTALL_HINT = 'uv run --with "transformers>=5.19" --with "torchao>=0.18" --with pillow ...'
MAX_BODY_BYTES = 1_000_000


class ServerError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- model


def mock_system_one(state, questions: dict) -> dict:
    """Deterministic stub with the same output shape as the real model.

    NOT the Liquid model. Only for transport/client tests.
    """
    def ans(name: str, choice: str) -> dict:
        criteria = questions[name].get("criteria", {})
        if criteria and choice not in criteria:
            choice = next(iter(criteria))
        return {"choice": choice, "confidence": 1.0}

    out: dict = {}
    if "objective" in questions:  # mission interpretation over the text state
        text = str(state).split("\n")[0].lower()
        out["objective"] = ans(
            "objective",
            "stop" if re.search(r"\b(stop|halt)\b", text) and "target" not in text
            else "reach_target" if "target" in text else "walk",
        )
        out["speed"] = ans("speed", "fast" if "fast" in text else "medium")
        out["direction"] = ans("direction", "forward" if "forward" in text else "keep")
        out["turn"] = ans("turn", "keep")
        out["after_arrival"] = ans(
            "after_arrival",
            "turn_left" if "turn left" in text else "turn_right" if "turn right" in text else "stop",
        )
        out["gait"] = ans("gait", "keep")
        out["caution"] = ans("caution", "normal")
    elif "steer" in questions:  # closed-loop tick over the JSON context
        out = {
            "speed": ans("speed", "keep"),
            "steer": ans("steer", "straight"),
            "gait": ans("gait", "keep"),
            "status": ans("status", "continue"),
        }
    else:  # one-shot instruction
        out = {name: ans(name, "keep") for name in questions}
    for name in questions:
        out.setdefault(name, ans(name, "keep"))
    return {"answers": out, "usage": {"mock": True}}


class ModelService:
    """Owns one loaded model (or the mock) and one inference lock."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 compile: bool = True, mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.compile = compile
        self.mock = mock
        self.model = None
        self.lock = threading.Lock()
        self.load_report: dict = {}

    def load(self) -> dict:
        if self.mock or self.model is not None:
            return self.load_report
        t0 = time.perf_counter()
        try:
            import torch
        except ImportError as e:
            raise ServerError(503, f"torch is not importable ({e})")
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ServerError(503, "model needs CUDA (INT8 tensor cores); torch.cuda.is_available() is False")
            cap = torch.cuda.get_device_capability(torch.device(self.device))
            if cap < (8, 0):
                raise ServerError(
                    503,
                    f"model needs Ampere+ GPU (sm_80+); this GPU is sm_{cap[0]}{cap[1]}. "
                    "No model is substituted automatically.",
                )
        try:
            import torchao  # noqa: F401  (needed by the w8a8 checkpoint)
            from transformers import AutoModel
        except ImportError as e:
            raise ServerError(503, f"missing optional dependency ({e}). Run with: {INSTALL_HINT}")
        try:
            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
            model.eval()
        except Exception as e:  # noqa: BLE001
            raise ServerError(503, f"loading {self.model_id} failed: {type(e).__name__}: {e}")
        if not callable(getattr(model, "system_one", None)):
            raise ServerError(503, f"{self.model_id} has no system_one(); not a d1 decision model")
        if self.compile:
            try:
                model.compile(mode="reduce-overhead")
            except Exception as e:  # noqa: BLE001
                raise ServerError(503, f"model.compile failed: {type(e).__name__}: {e}")
        self.model = model
        self.load_report = {"model_id": self.model_id, "device": self.device,
                            "load_s": time.perf_counter() - t0}
        return self.load_report

    def system_one(self, state, questions: dict) -> dict:
        """One serialised inference; returns the full model output dict."""
        if not isinstance(questions, dict) or not questions:
            raise ServerError(400, "questions must be a non-empty object")
        with self.lock:
            if self.mock:
                return mock_system_one(state, questions)
            if self.model is None:
                raise ServerError(503, "model not loaded")
            import torch

            with torch.inference_mode():
                out = self.model.system_one(state, questions)
        if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
            raise ServerError(502, f"unexpected system_one output: {type(out).__name__}")
        return out


# ---------------------------------------------------------------- HTTP


def _make_handler(service: ModelService, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep stdout clean; no headers logged
            sys.stderr.write(f"[model-server] {self.address_string()} {fmt % args}\n")

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ServerError(400, "missing or oversized request body")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError as e:
                raise ServerError(400, f"invalid JSON: {e}")
            if not isinstance(data, dict):
                raise ServerError(400, "request body must be a JSON object")
            return data

        def _check_auth(self) -> None:
            if api_key is None:
                return
            header = self.headers.get("Authorization") or ""
            presented = header[7:] if header.startswith("Bearer ") else ""
            # Constant-time compare on hashes, so the raw key never sits
            # in a timing-comparable string comparison.
            ok = hmac.compare_digest(
                hashlib.sha256(presented.encode()).hexdigest(),
                hashlib.sha256(api_key.encode()).hexdigest(),
            )
            if not ok:
                raise ServerError(401, "unauthorized")

        def do_GET(self):  # noqa: N802
            try:
                if self.path.rstrip("/") in ("", "/health"):
                    self._send_json(200, {
                        "ok": True,
                        "model_id": service.model_id,
                        "model_loaded": service.model is not None,
                        "mock": service.mock,
                        "load": service.load_report,
                    })
                else:
                    raise ServerError(404, "not found")
            except ServerError as e:
                self._send_json(e.status, {"error": e.message})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            try:
                if self.path.rstrip("/") != "/system_one":
                    raise ServerError(404, "not found")
                self._check_auth()
                data = self._read_json()
                if "state" not in data or "questions" not in data:
                    raise ServerError(400, "body must contain 'state' and 'questions'")
                out = service.system_one(data["state"], data["questions"])
                self._send_json(200, out)
            except ServerError as e:
                self._send_json(e.status, {"error": e.message})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LiquidAI d1 model server (see module docstring).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 127.0.0.1 for local/proxied use, 0.0.0.0 only behind TLS+firewall")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--mock", action="store_true", help="serve the deterministic stub, not the real model")
    ap.add_argument("--api-key", default=None, help="Bearer token (prefer LIQUID_API_KEY env var)")
    ap.add_argument("--api-key-env", default="LIQUID_API_KEY")
    ap.add_argument("--allow-no-auth", action="store_true",
                    help="serve without a Bearer token; local testing only, never on the internet")
    a = ap.parse_args(argv)

    api_key = a.api_key if a.api_key is not None else os.environ.get(a.api_key_env)
    if api_key is None and not a.allow_no_auth:
        print(f"error: no API key: pass --api-key or set {a.api_key_env}, "
              "or pass --allow-no-auth for local testing only", file=sys.stderr)
        return 2

    service = ModelService(a.model, a.model_device, compile=not a.no_compile, mock=a.mock)
    try:
        report = service.load()
    except ServerError as e:
        print(f"error: model load failed ({e.status}): {e.message}", file=sys.stderr)
        return 1
    if report:
        print(f"[model-server] model loaded in {report['load_s']:.1f}s: {report}", file=sys.stderr)

    server = ThreadingHTTPServer((a.host, a.port), _make_handler(service, api_key))
    print(f"[model-server] listening on http://{a.host}:{a.port} "
          f"(mock={a.mock}, auth={'on' if api_key else 'OFF'})", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
