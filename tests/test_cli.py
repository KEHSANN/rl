"""CLI parsing of the new commands (tyro-based ones need tyro installed)."""

from __future__ import annotations

import pytest


def test_eval_cli_defaults_and_flags():
  tyro = pytest.importorskip("tyro")
  from contact_rl.scripts.evaluate import EvalConfig

  cfg = tyro.cli(EvalConfig, args=["--checkpoint", "runs/x:best", "--mode", "episodes", "--video", "True",
                                   "--num-envs", "64", "--gaits", "trot", "bound"],
                 config=(tyro.conf.FlagConversionOff,))
  assert cfg.checkpoint == "runs/x:best" and cfg.mode == "episodes" and cfg.video and cfg.num_envs == 64
  assert cfg.gaits == ("trot", "bound")
  legacy = tyro.cli(EvalConfig, args=["--checkpoint-file", "m.pt"], config=(tyro.conf.FlagConversionOff,))
  assert legacy.mode == "fig6" and legacy.checkpoint_file == "m.pt" and legacy.out_csv == "eval_fig6.csv"


def test_watch_cli():
  tyro = pytest.importorskip("tyro")
  from contact_rl.scripts.watch import WatchConfig

  cfg = tyro.cli(WatchConfig, args=["--run", "runs/a/b", "--interval-s", "5", "--every", "100"],
                 config=(tyro.conf.FlagConversionOff,))
  assert cfg.run == "runs/a/b" and cfg.interval_s == 5.0 and cfg.every == 100 and cfg.video


def test_train_and_play_cli_keep_mjlab_flags():
  pytest.importorskip("mjlab")
  from contact_rl.scripts import play, train

  task, args = train.parse_args(["Mjlab-Contact-Flat-Unitree-Go2", "--env.scene.num-envs", "256",
                                 "--resume-from", "runs/go2_contact/x:100", "--keep-last", "3", "--video", "True"])
  assert task == "Mjlab-Contact-Flat-Unitree-Go2" and args.env.scene.num_envs == 256
  assert args.resume_from.endswith(":100") and args.keep_last == 3 and args.video and args.keep_best
  task, pargs = play.parse_args(["Mjlab-Contact-Flat-Unitree-Go2", "--checkpoint", "best"])
  assert pargs.host == "127.0.0.1" and pargs.port == 8080 and not pargs.allow_public
