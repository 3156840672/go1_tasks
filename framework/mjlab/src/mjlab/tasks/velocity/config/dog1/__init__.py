from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  dog1_flat_env_cfg,
  dog1_rough_env_cfg,
)
from .rl_cfg import dog1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Dog1",
  env_cfg=dog1_rough_env_cfg(),
  play_env_cfg=dog1_rough_env_cfg(play=True),
  rl_cfg=dog1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Dog1",
  env_cfg=dog1_flat_env_cfg(),
  play_env_cfg=dog1_flat_env_cfg(play=True),
  rl_cfg=dog1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
