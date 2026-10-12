"""Headless closed-loop rollout: real env + trained checkpoint + LiquidRuntime, no viewer.

Exercises exactly the path the viewer uses (action queue drained on the sim
thread, ``on_sim_step`` after every ``env.step``, ``apply_to_command_term`` ->
GaitPlanner -> contact goals -> unchanged GRU policy -> MuJoCo-Warp), and
prints a JSON report of the measured outcome.

``--decider liquid`` (default) loads the real local d1 model. ``--decider mock``
replaces ONLY ``system_one`` with a transparent keyword/threshold stub so the
integration can be checked on machines that cannot run the model (e.g. no
Ampere GPU). A mock rollout is NOT evidence that the Liquid model works; the
report says which decider ran.

  uv run python -m contact_rl.runtime.rollout --checkpoint best --device cpu --decider mock \\
      --instruction "Move toward the target and turn left." --target 1.5 0.5 --seconds 40
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque


# ------------------------------------------------------------------ mock model
def mock_ask(state, questions: dict) -> dict:
  """Stand-in for ``model.system_one(state, questions)['answers']`` (NOT the Liquid model)."""
  def ans(name, choice):
    assert choice in questions[name]["criteria"], (name, choice)
    return {"choice": choice, "confidence": 1.0}

  out = {}
  if "objective" in questions:  # mission interpretation, over the text state
    t = str(state).split("\n")[0].lower()
    out["objective"] = ans("objective", "stop" if re.search(r"\b(stop|halt)\b", t) and "target" not in t
                           else "reach_target" if "target" in t else "walk")
    out["speed"] = ans("speed", "slow" if "slow" in t or "caut" in t else "medium" if "moderate" in t
                       else "fast" if "fast" in t else "unspecified")
    out["direction"] = ans("direction", "forward" if "forward" in t else "keep")
    out["turn"] = ans("turn", "keep")
    out["after_arrival"] = ans("after_arrival", "turn_left" if "turn left" in t else
                               "turn_right" if "turn right" in t else "stop")
    out["gait"] = ans("gait", "suitable" if "suitable" in t else "keep")
    out["caution"] = ans("caution", "cautious" if "caut" in t or "unstable" in t else "normal")
    for k in questions:
      out.setdefault(k, ans(k, next(iter(questions[k]["criteria"]))))
    return out
  # closed-loop tick, over the JSON context
  def num(s, default=0.0):
    m = re.search(r"[-+]?\d+(\.\d+)?", str(s))
    return float(m.group()) if m else default
  ms, cmd, tgt = state["mission"], state["current command"], state.get("target")
  limit, speed = num(ms["speed limit now"]), num(cmd["speed"])
  unstable = state["measured state"]["stability"] != "stable"
  steer, sp = "keep", "keep"
  if ms["phase"] == "final_turn":
    steer = "left" if state.get("final turn", {}).get("direction") == "left" else "right"
  elif isinstance(tgt, dict):
    bearing, dist = num(tgt["direction"]), num(tgt["distance"])
    steer = "left" if bearing > 15 else "right" if bearing < -15 else "straight"
    if unstable or (dist < 0.6 and speed > 0.1):
      sp = "decrease"
    elif speed < limit - 1e-3:
      sp = "increase"
  else:
    sp = "decrease" if unstable else "increase" if speed < limit - 1e-3 else "keep"
  out = {"speed": ans("speed", sp), "steer": ans("steer", steer), "gait": ans("gait", "keep"),
         "status": ans("status", "continue")}
  return out


# ---------------------------------------------------------------------- rollout
def _shapes(obs) -> dict:
  try:
    return {k: list(v.shape) for k, v in obs.items()}
  except AttributeError:
    return {"obs": list(obs.shape)}


def _hidden_shapes(policy) -> dict:
  """Shapes of recurrent hidden-state tensors found on the policy (GRU state), for the report."""
  import torch

  out, seen = {}, set()
  mods = list(policy.modules()) if hasattr(policy, "modules") else []
  owner = getattr(policy, "__self__", None)  # bound method -> its module
  if owner is not None and hasattr(owner, "modules"):
    mods += list(owner.modules())
  for mod in mods:
    for k, v in vars(mod).items():
      if "hidden" in k.lower() and id(v) not in seen:
        seen.add(id(v))
        vs = v if isinstance(v, (tuple, list)) else (v,)
        shapes = [list(t.shape) for t in vs if isinstance(t, torch.Tensor)]
        if shapes:
          out[f"{type(mod).__name__}.{k}"] = shapes
  return out


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--task", default="Mjlab-Contact-Flat-Unitree-Go2")
  ap.add_argument("--checkpoint", default="best")
  ap.add_argument("--run-root", default="runs")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--decider", choices=("liquid", "mock"), default="liquid")
  ap.add_argument("--model", default="LiquidAI/d1-3B-w8a8")
  ap.add_argument("--model-device", default="cuda")
  ap.add_argument("--instruction", default="Move toward the target and turn left.")
  ap.add_argument("--target", type=float, nargs=2, default=None, metavar=("FWD", "LEFT"))
  ap.add_argument("--seconds", type=float, default=30.0, help="sim seconds")
  ap.add_argument("--decision-hz", type=float, default=1.0)
  ap.add_argument("--deadline-s", type=float, default=2.0)
  ap.add_argument("--realtime", action="store_true", help="pace the sim at wall-clock speed")
  ap.add_argument("--json", default=None, help="also write the report here")
  a = ap.parse_args(argv)

  import torch

  import contact_rl

  contact_rl.require_tasks()
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  from contact_rl.runtime.decision_schema import ClosedLoopConfig, Limits
  from contact_rl.runtime.liquid_decider import LiquidDecider, make_decide
  from contact_rl.runtime.viewer_bridge import LiquidRuntime
  from contact_rl.utils import checkpoints as ck
  from contact_rl.utils.policy_state import reset_recurrent_state

  env_cfg = load_env_cfg(a.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(a.task)
  ckpt = ck.resolve_checkpoint(a.checkpoint, a.run_root)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=a.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=a.device)
  policy = runner.get_inference_policy(device=a.device)
  reset_recurrent_state(policy)
  term = env.command_manager.get_term("contact")

  limits = Limits()
  cl = ClosedLoopConfig(decision_hz=a.decision_hz, deadline_s=a.deadline_s)
  if a.decider == "liquid":
    dec = LiquidDecider(a.model, a.model_device, True)
    rt = LiquidRuntime(lambda lim: dec.make_decide(lim, cl.turn_rate, cl), dec.load, limits, mode="closed_loop",
                       closed_loop=cl)
  else:
    rt = LiquidRuntime(lambda lim: make_decide(mock_ask, lim, cl.turn_rate, cl), None, limits, mode="closed_loop",
                       closed_loop=cl)
  actions: deque = deque()
  rt.attach(lambda name, payload: actions.append(payload), lambda: term, lambda: None, lambda: 0,
            step_dt=float(env.step_dt))
  if a.target is not None:
    rt.on_target(tuple(a.target))
  rt.start()
  t_load = time.monotonic()
  while rt.loop.status == "loading":
    time.sleep(0.1)
  if rt.loop.status == "unavailable":
    raise SystemExit(f"[rollout] decider unavailable: {rt.loop.last_error}")
  load_s = time.monotonic() - t_load

  obs = venv.get_observations()
  obs_shapes0 = _shapes(obs)
  with torch.no_grad():
    act_shape0 = list(policy(obs).shape)
  hidden0 = _hidden_shapes(policy)  # after one call (rsl-rl creates the GRU state lazily)
  reset_recurrent_state(policy)
  traj = []
  rt.send_text(a.instruction)
  steps = int(a.seconds / env.step_dt)
  t0 = time.monotonic()
  policy_steps = 0
  try:
    for k in range(steps):
      while actions:
        kind, arg = actions.popleft()
        rt.handle(kind, arg)
      with torch.no_grad():
        obs = venv.get_observations()
        act = policy(obs)
        _, _, dones, _ = venv.step(act)
        reset_recurrent_state(policy, dones)
      policy_steps += 1
      rt.on_sim_step(dones)
      if k % 25 == 0 and rt.last_feedback is not None:
        fb = rt.last_feedback
        traj.append([round(fb.sim_time, 2), round(fb.x, 3), round(fb.y, 3), round(math.degrees(fb.yaw), 1),
                     round(fb.vx, 3), round(math.degrees(fb.tilt), 1), round(fb.command.speed, 3),
                     round(fb.command.yaw_rate, 2)])
      if a.realtime:
        lag = (k + 1) * env.step_dt - (time.monotonic() - t0)
        if lag > 0:
          time.sleep(lag)
      m = rt.mission
      if m is not None and m.state in ("complete", "failed", "stopped") and not rt.loop.busy:
        break
  finally:
    rt.stop()
  wall = time.monotonic() - t0
  m = rt.mission
  fb = rt.last_feedback
  app = sorted(x for x in rt.app_latencies if math.isfinite(x))
  lat = sorted(rt.loop.latencies)
  with torch.no_grad():
    obs = venv.get_observations()
    interfaces = {"obs_shapes_start": obs_shapes0, "obs_shapes_end": _shapes(obs), "action_shape_start": act_shape0,
                  "action_shape_end": list(policy(obs).shape), "policy_hidden_start": hidden0,
                  "policy_hidden_end": _hidden_shapes(policy), "checkpoint_strict_load": True,
                  "contact_goal_dim": int(term.command.shape[-1])}
  interfaces["unchanged"] = (interfaces["obs_shapes_start"] == interfaces["obs_shapes_end"]
                             and interfaces["action_shape_start"] == interfaces["action_shape_end"]
                             and interfaces["policy_hidden_start"] == interfaces["policy_hidden_end"])
  rep = {
    "decider": a.decider if a.decider == "liquid" else "MOCK system_one stub (not the Liquid model)",
    "model": a.model if a.decider == "liquid" else None,
    "load_report": rt.loop.load_report, "load_wait_s": round(load_s, 2),
    "checkpoint": str(ckpt), "sim_device": a.device, "instruction": a.instruction, "target_rel": a.target,
    "sim_seconds": round(rt.sim_time, 2), "wall_seconds": round(wall, 2),
    "policy_steps": policy_steps, "policy_hz_sim": round(1.0 / env.step_dt, 1),
    "mission": None if m is None else {
      "objective": m.spec.objective, "after_arrival": m.spec.after_arrival, "state": m.state, "phase": m.phase,
      "outcome": m.outcome, "preferred_speed": round(m.spec.preferred_speed, 3), "decisions": m.decisions,
      "kept": m.kept, "changed": m.changed, "target_xy": m.spec.target_xy,
      "final_target_error": None if fb is None or m.target_error(fb) is None
      else [round(m.target_error(fb)[0], 3), round(math.degrees(m.target_error(fb)[1]), 1)],
      "history": m.history[-12:],
    },
    "interfaces": interfaces,
    "loop_counts": rt.loop.counts,
    "inference_ms": {"n": len(lat), "p50": round(1e3 * lat[len(lat) // 2], 2) if lat else None,
                     "max": round(1e3 * lat[-1], 2) if lat else None},
    "application_latency_ms_p50": round(1e3 * app[len(app) // 2], 1) if app else None,
    "events": list(rt.events)[-25:],
    "trajectory[t,x,y,yaw_deg,vx,tilt_deg,cmd_v,cmd_yaw]": traj[::2],
  }
  s = json.dumps(rep, indent=1, default=str)
  print(s)
  if a.json:
    with open(a.json, "w") as f:
      f.write(s)
  venv.close()


if __name__ == "__main__":
  main()
