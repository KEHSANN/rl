"""End-to-end transport test: model_server (mock) + RemoteDecider client.

Run from the remote-liquid folder:  python3 tests/test_remote_e2e.py

What it proves (without a GPU, torch, or the simulator):
  1. the server starts, loads nothing real in --mock mode, and /health is ok;
  2. a correct Bearer token gets well-formed answers for mission, tick
     and one-shot question sets, with the same ask() contract as
     LiquidDecider.ask (state + questions in, answers dict out);
  3. a wrong / missing token is refused with 401 -> RemoteUnavailable,
     and the token itself never appears in any error message;
  4. the round trip is fast on localhost (well under the 2.0 s tick deadline).

What it does NOT prove: anything about the real d1 model, the GRU
policy, or the simulator. Those need a GPU machine (model) and the
full repo (robot), respectively; see README.md.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from contact_rl.runtime.remote_decider import RemoteDecider, RemoteUnavailable  # noqa: E402

PORT = 18001
BASE = f"http://127.0.0.1:{PORT}"
KEY = "test-only-local-key"  # dummy value, used only on localhost for this test

MISSION_Q = {
    "objective": {"type": "choice", "instructions": "?", "criteria": {"walk": "", "reach_target": "", "stop": ""}},
    "speed": {"type": "choice", "instructions": "?", "criteria": {"slow": "", "medium": "", "fast": "", "unspecified": ""}},
    "direction": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "forward": ""}},
    "turn": {"type": "choice", "instructions": "?", "criteria": {"keep": ""}},
    "after_arrival": {"type": "choice", "instructions": "?", "criteria": {"stop": "", "turn_left": "", "turn_right": ""}},
    "gait": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "trot": ""}},
    "caution": {"type": "choice", "instructions": "?", "criteria": {"normal": "", "cautious": ""}},
}
TICK_Q = {
    "speed": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "increase": "", "decrease": "", "stop": ""}},
    "steer": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "straight": "", "left": "", "right": ""}},
    "gait": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "trot": ""}},
    "status": {"type": "choice", "instructions": "?", "criteria": {"continue": "", "complete": "", "abort": ""}},
}
ONE_Q = {
    "speed": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "fast": ""}},
    "turn": {"type": "choice", "instructions": "?", "criteria": {"keep": "", "left": ""}},
}


def main() -> int:
    env = dict(os.environ, LIQUID_API_KEY=KEY, PYTHONPATH=SRC)
    proc = subprocess.Popen(
        [sys.executable, "-m", "contact_rl.runtime.model_server",
         "--host", "127.0.0.1", "--port", str(PORT), "--mock"],
        cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # wait for /health
        deadline = time.time() + 15
        while True:
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                    health = json.loads(r.read())
                break
            except Exception:
                if time.time() > deadline:
                    out = proc.stdout.read() if proc.stdout else ""
                    raise SystemExit(f"server did not come up:\n{out}")
                time.sleep(0.2)
        assert health["ok"] and health["mock"], health
        print(f"health ok: {health}")

        dec = RemoteDecider(BASE, api_key=KEY)
        assert dec.load()["ok"]
        t0 = time.time()
        mission = dec.ask("User command: walk forward fast\nCurrent gait: trot\nCurrent speed: 0.5 m/s\n"
                          "Current walking direction: 0 degrees\nCurrent turning rate: 0 rad/s", MISSION_Q)
        tick = dec.ask({"mission": {"x": 1}, "current command": {}, "measured state": {}}, TICK_Q)
        one = dec.ask("User command: go fast", ONE_Q)
        dt = time.time() - t0
        print(f"mission answers: {mission}")
        print(f"tick answers: {tick}")
        print(f"one-shot answers: {one}")
        assert mission["objective"]["choice"] == "walk", mission
        assert mission["speed"]["choice"] == "fast", mission
        assert tick["status"]["choice"] == "continue", tick
        assert set(one) == {"speed", "turn"}, one
        assert dt < 2.0, f"round trip {dt:.2f}s exceeds tick deadline"
        print(f"3 asks in {dt:.3f}s (tick deadline is 2.0s)")

        # wrong token -> 401 -> RemoteUnavailable, key never in the message
        try:
            RemoteDecider(BASE, api_key="wrong-key").ask("x", ONE_Q)
            raise SystemExit("FAIL: wrong key was accepted")
        except RemoteUnavailable as e:
            assert "wrong-key" not in str(e), e
            print(f"wrong key refused as expected: {e}")

        # no token at all -> server 401 (raw HTTP check, bypassing the client guard)
        req = urllib.request.Request(f"{BASE}/system_one", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2)
            raise SystemExit("FAIL: unauthenticated request was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 401, e.code
            print("unauthenticated request refused with HTTP 401")
        print("ALL REMOTE E2E TESTS PASSED")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
