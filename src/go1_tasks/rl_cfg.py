"""RL configuration for Unitree Go1 velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class PaperAmpModelCfg(RslRlModelCfg):
  bound_action_mean: bool = False


@dataclass
class CtsPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  student_learning_rate: float = 1e-3
  student_epochs: int = 5
  student_batch_size: int = 512


@dataclass
class HimModelCfg(RslRlModelCfg):
  """HIM actor：历史编码器 + 速度回归 + SwAV latent。"""

  num_one_step_obs: int = 45
  history_steps: int = 6
  term_dims: tuple[int, ...] = (3, 3, 3, 12, 12, 12)
  latent_dim: int = 16
  velocity_dim: int = 3
  command_dim: int = 3
  num_prototype: int = 32
  temperature: float = 3.0
  estimator_learning_rate: float = 1e-3
  estimator_max_grad_norm: float = 10.0


@dataclass
class HimPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """HIM 估计器的独立优化设置。"""

  estimator_epochs: int = 4
  estimator_batch_size: int = 512
  estimator_lr_follows_policy: bool = True


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
  discriminator_hidden_dims: tuple[int, ...] = (256, 128)
  discriminator_activation: str = "relu"
  discriminator_state_mode: str = "extended"
  discriminator_loss_margin: float = 0.0
  discriminator_trunk_weight_decay: float = 0.0
  discriminator_head_weight_decay: float = 0.0
  reset_discriminator_on_resume: bool = False
  zero_command_amp_threshold: float = 0.0


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
