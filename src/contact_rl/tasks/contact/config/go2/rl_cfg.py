"""RL configuration for the Unitree Go2 contact-explicit locomotion task.

The paper (Section 4, "Training") uses **PPO with a recurrent (GRU) actor and
entropy decay**. Both are realised here on the pinned stack (``rsl-rl-lib==5.0.1``):

* **Recurrent actor/critic** -- rsl-rl>=5.0.0 unifies recurrence under the model
  ``class_name="RNNModel"`` (``rsl_rl/models/rnn_model.py``), which inherits from
  ``MLPModel`` and adds ``rnn_type`` / ``rnn_hidden_dim`` / ``rnn_num_layers``.
  mjlab converts the runner cfg with ``dataclasses.asdict`` and forwards the
  ``actor`` / ``critic`` dicts to rsl-rl's model builder without dropping unknown
  keys, so the :class:`RslRlRnnModelCfg` subclass below flows straight through.
  PPO then auto-detects recurrence (``actor_critic.is_recurrent`` ->
  ``recurrent_mini_batch_generator``) and manages hidden states / BPTT masking
  itself -- no custom generator needed.
* **Entropy decay** -- ``PPO.entropy_coef`` is a plain mutable float with no
  internal schedule, so :class:`ContactOnPolicyRunner` anneals it during training
  by writing ``self.alg.entropy_coef`` (see ``runner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlRnnModelCfg(RslRlModelCfg):
  """``RslRlModelCfg`` extended with rsl-rl 5.x recurrent (``RNNModel``) fields.

  All base ``MLPModel`` keys (``hidden_dims``, ``activation``,
  ``distribution_cfg``, ...) are inherited and apply to the post-recurrent MLP
  head; the ``rnn_*`` fields configure the GRU/LSTM core.
  """

  class_name: str = "RNNModel"
  rnn_type: str = "gru"  # "gru" (paper) or "lstm".
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for the Go2 contact-explicit task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlRnnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
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
      obs_normalization=False,
      rnn_type="gru",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Entropy-decay start value; ContactOnPolicyRunner decays it over training.
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_contact",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
