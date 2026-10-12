"""Posture accounting must not mistake survival or reset poses for healthy gait."""

import math

import pytest
import torch

from contact_rl.utils.episode_metrics import FALL, TIMEOUT, EpisodeAccumulator
from contact_rl.utils.posture_metrics import PostureAccumulator, summarize


def test_legacy_columns_unchanged_and_terminal_reset_excluded():
  old = EpisodeAccumulator(torch.zeros(2))
  acc = PostureAccumulator(torch.zeros(2), min_height=0.08)
  heights = torch.tensor([[0.15, 0.15, 0.015, 0.015], [0.15] * 4])
  for reason, z, contacts in [
    (torch.zeros(2), heights, torch.tensor([[0, 2, 0], [0, 0, 0]])),
    (torch.tensor([FALL, TIMEOUT]), torch.zeros(2, 4), torch.ones(2, 3)),
    (torch.zeros(2), torch.zeros(2, 4), torch.ones(2, 3)),
  ]:
    old.step(torch.ones(2), reason)
    acc.step(torch.ones(2), reason, knee_heights=z, illegal_found=contacts)
  rows = acc.rows(0.02)
  for before, after in zip(old.rows(0.02), rows, strict=True):
    assert all(after[k] == v for k, v in before.items())
  assert rows[0]["knee_RL_height_min_m"] == pytest.approx(0.015)
  assert rows[0]["knee_RL_below_fraction"] == 1.0
  assert rows[1]["knee_RL_below_fraction"] == 0.0
  assert rows[0]["illegal_contact_count_mean"] == 1.0
  assert rows[1]["illegal_contact_fraction"] == 0.0
  assert rows[0]["posture_samples"] == 1
  assert rows[1]["success"] is True


def test_surviving_kneeling_episode_is_visible_and_summary_is_pooled():
  acc = PostureAccumulator(torch.zeros(2), min_height=0.08)
  z = torch.tensor([[0.04] * 4, [0.15] * 4])
  for reason in [torch.zeros(2), torch.tensor([TIMEOUT, 0]), torch.zeros(2)]:
    acc.step(torch.zeros(2), reason, knee_heights=z, illegal_found=torch.zeros(2, 19))
  rows = acc.rows(0.02)
  summary = summarize(rows)
  assert summary["success_rate"] == 1.0  # historical survival semantics unchanged
  assert summary["knee_below_fraction_mean"] == 0.5
  assert summary["knee_below_fraction_pooled"] == 0.25
  assert rows[0]["illegal_contact_fraction"] == 0.0  # hovering loophole
  assert rows[0]["knee_below_fraction"] == 1.0


def test_no_valid_samples_are_missing_not_healthy():
  acc = PostureAccumulator(torch.zeros(1))
  acc.step(torch.zeros(1), torch.tensor([FALL]), knee_heights=torch.zeros(1, 4), illegal_found=torch.ones(1, 19))
  row = acc.rows(0.02)[0]
  assert row["posture_samples"] == 0
  assert math.isnan(row["knee_RL_height_min_m"])
  assert math.isnan(row["illegal_contact_fraction"])
  assert math.isnan(summarize([row])["knee_below_fraction_pooled"])


@pytest.mark.parametrize("height", [0, -0.1, float("nan"), float("inf")])
def test_invalid_threshold_rejected(height):
  with pytest.raises(ValueError):
    PostureAccumulator(torch.zeros(1), min_height=height)


def test_shape_validation_and_boundary():
  acc = PostureAccumulator(torch.zeros(1))
  with pytest.raises(ValueError):
    acc.step(torch.zeros(1), torch.zeros(1), knee_heights=torch.ones(1, 3), illegal_found=torch.zeros(1, 19))
  acc.step(torch.zeros(1), torch.zeros(1), knee_heights=torch.full((1, 4), 0.08), illegal_found=torch.zeros(1, 19))
  assert acc.rows(0.02)[0]["knee_below_fraction"] == 0.0


def test_legacy_summary_without_posture_is_unchanged():
  from contact_rl.utils.episode_metrics import summarize as legacy

  rows = EpisodeAccumulator(torch.zeros(1)).rows(0.02)
  # Compare finite fields only; legacy reports NaN for empty pooled counts.
  actual, expected = summarize(rows), legacy(rows)
  assert actual.keys() == expected.keys()
  for key, value in expected.items():
    assert actual[key] == value or (isinstance(value, float) and math.isnan(value) and math.isnan(actual[key]))
