"""RL configuration for Unitree Go1 velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class CtsPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  student_learning_rate: float = 1e-3
  student_epochs: int = 5
  student_batch_size: int = 512


@dataclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  reference_path: str = "logs/amp/mob_reference/trot"
  amp_reward_weight: float = 2.0
  discriminator_learning_rate: float = 1e-4
  discriminator_epochs: int = 1
  discriminator_batch_size: int = 1024
  discriminator_batches_per_iteration: int = 8
  gradient_penalty_weight: float = 10.0
  replay_capacity: int = 100_000


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
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
    experiment_name="go1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
