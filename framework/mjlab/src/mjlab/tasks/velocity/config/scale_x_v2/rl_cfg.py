"""RL configuration for Scale_X_v2 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def scale_x_v2_ppo_runner_cfg(
  max_iterations: int = 15_000,
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Scale_X_v2 velocity task.

  Args:
    max_iterations: Total training iterations. Flat terrain converges faster
      with the reduced ±1 m/s command range, so it uses a smaller value;
      rough terrain keeps the default.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
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
    experiment_name="scale_x_v2_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
