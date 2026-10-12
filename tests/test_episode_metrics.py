"""Episode accounting: exclusive termination reasons, no reset-step leakage."""

from __future__ import annotations

import numpy as np

from contact_rl.utils import episode_metrics as em

B = lambda *x: np.array(x, dtype=bool)  # noqa: E731
F = lambda *x: np.array(x, dtype=float)  # noqa: E731


def test_classify_timeout_is_not_a_fall():
  # env0 timeout only, env1 fell, env2 fell on the time-out step, env3 other term, env4 alive
  truncated = B(1, 0, 1, 0, 0)
  terminated = B(0, 1, 1, 1, 0)
  fell = B(0, 1, 1, 0, 0)
  assert em.classify(truncated, terminated, fell).tolist() == [em.TIMEOUT, em.FALL, em.FALL, em.OTHER, em.RUNNING]


def test_reset_step_excluded_from_state_metrics_but_reward_counted():
  acc = em.EpisodeAccumulator(np.zeros(2))
  alive = F(0, 0)
  # step 1: both alive, hamming 1 each
  acc.step(F(1, 1), alive, hamming=F(1, 1), loc_err_sum=F(0.1, 0.1), loc_cnt=F(1, 1))
  # step 2: env0 falls; its post-step state is post-reset garbage (hamming 4)
  acc.step(F(1, 1), F(em.FALL, 0), hamming=F(4, 1), loc_err_sum=F(9, 0.1), loc_cnt=F(1, 1))
  # step 3: env0 already done (next episode) must not be counted at all
  acc.step(F(100, 1), F(0, em.TIMEOUT), hamming=F(4, 4), loc_err_sum=F(9, 9), loc_cnt=F(1, 1))
  rows = acc.rows(step_dt=0.02)
  r0, r1 = rows
  assert r0["termination"] == "fall" and r0["fell"] and not r0["success"]
  assert r0["reward"] == 2.0 and r0["length_steps"] == 2
  assert r0["contact_plan_hamming"] == 1.0  # only step 1
  assert abs(r0["contact_location_error_cm"] - 10.0) < 1e-9
  assert r1["termination"] == "timeout" and r1["success"] and not r1["fell"]
  assert r1["reward"] == 3.0 and r1["length_steps"] == 3
  assert r1["contact_plan_hamming"] == 1.0  # steps 1-2, not the reset step 3
  assert acc.all_done()


def test_running_episodes_are_reported_as_running():
  acc = em.EpisodeAccumulator(np.zeros(1))
  acc.step(F(1), F(0), hamming=F(0))
  (r,) = acc.rows(0.02)
  assert r["termination"] == "running" and r["success"] and not acc.all_done()


def test_summary_rates():
  rows = [
    {"termination": t, "success": t in ("timeout", "running"), "reward": float(i), "length_s": 1.0}
    for i, t in enumerate(["fall", "timeout", "timeout", "other_termination"])
  ]
  s = em.summarize(rows)
  assert s["episodes"] == 4
  assert s["fall_rate"] == 0.25 and s["timeout_rate"] == 0.5 and s["other_termination_rate"] == 0.25
  assert s["success_rate"] == 0.5
  assert s["reward_mean"] == 1.5


def test_pooled_fig6_aggregates_are_step_weighted():
  """The legacy Fig. 6 table uses pooled (step-weighted) numbers, as before."""
  acc = em.EpisodeAccumulator(np.zeros(2))
  # env0: 1 valid step with hamming 2; env1: 3 valid steps with hamming 0
  acc.step(F(0, 0), F(0, 0), hamming=F(2, 0), loc_err_sum=F(0.3, 0.1), loc_cnt=F(1, 1))
  acc.step(F(0, 0), F(em.FALL, 0), hamming=F(9, 0), loc_err_sum=F(9, 0.1), loc_cnt=F(1, 1))
  acc.step(F(0, 0), F(0, 0), hamming=F(9, 0), loc_err_sum=F(9, 0.1), loc_cnt=F(1, 1))
  s = em.summarize(acc.rows(0.02))
  assert s["contact_plan_hamming_mean"] == 1.0  # (2 + 0) / 2 episodes
  assert s["contact_plan_hamming_pooled"] == 0.5  # 2 / 4 valid steps
  assert abs(s["contact_location_error_cm_pooled"] - 100.0 * 0.6 / 4) < 1e-9
  assert s["state_steps_total"] == 4 and s["contact_samples_total"] == 4
