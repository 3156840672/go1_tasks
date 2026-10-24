from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  scale_x_v2_flat_env_cfg,
  scale_x_v2_rough_env_cfg,
)
from .rl_cfg import scale_x_v2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Scale-X-v2",
  env_cfg=scale_x_v2_rough_env_cfg(),
  play_env_cfg=scale_x_v2_rough_env_cfg(play=True),
  rl_cfg=scale_x_v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Scale-X-v2",
  env_cfg=scale_x_v2_flat_env_cfg(),
  play_env_cfg=scale_x_v2_flat_env_cfg(play=True),
  # Flat terrain converges faster with the reduced ±1 m/s command range.
  rl_cfg=scale_x_v2_ppo_runner_cfg(max_iterations=12_000),
  runner_cls=VelocityOnPolicyRunner,
)
