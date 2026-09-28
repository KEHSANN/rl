"""RL configuration for the Unitree Go2 contact-explicit locomotion task.

The paper (Section 4, "Training") uses **PPO with a recurrent (GRU) policy and
entropy decay** (8192 envs in IsaacLab). On the pinned stack
(``rsl-rl-lib==5.0.1``) recurrence is rsl-rl's ``class_name="RNNModel"``;
mjlab forwards the ``actor`` / ``critic`` dicts unchanged, so the
:class:`RslRlRnnModelCfg` subclass below flows through and PPO switches to the
recurrent mini-batch generator automatically. Entropy decay is applied by
:class:`~.runner.ContactOnPolicyRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Entropy-decay schedule (read by ContactOnPolicyRunner). Linear START -> END
# over DECAY_ITERS PPO iterations, constant afterwards.
ENTROPY_START = 0.01
ENTROPY_END = 0.001
ENTROPY_DECAY_ITERS = 5000


@dataclass
class RslRlRnnModelCfg(RslRlModelCfg):
  """``RslRlModelCfg`` extended with rsl-rl 5.x ``RNNModel`` fields."""

  class_name: str = "RNNModel"
  rnn_type: str = "gru"  # "gru" (paper) or "lstm".
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1


def unitree_go2_ppo_runner_cfg(improved: bool = False) -> RslRlOnPolicyRunnerCfg:
  """PPO runner cfg. ``improved`` enables running observation normalisation
  (EXPERIMENTAL; the observation mixes metres, rad/s and binary flags)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlRnnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=improved,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlRnnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=improved,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=ENTROPY_START,  # Decayed by ContactOnPolicyRunner.
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    seed=42,
    experiment_name="go2_contact_improved" if improved else "go2_contact",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
