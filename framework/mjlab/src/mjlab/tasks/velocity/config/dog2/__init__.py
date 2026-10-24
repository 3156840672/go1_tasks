from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  dog2_flat_env_cfg,
  dog2_rough_env_cfg,
)
from .rl_cfg import dog2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Dog2",
  env_cfg=dog2_rough_env_cfg(),
  play_env_cfg=dog2_rough_env_cfg(play=True),
  rl_cfg=dog2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Dog2",
  env_cfg=dog2_flat_env_cfg(),
  play_env_cfg=dog2_flat_env_cfg(play=True),
  rl_cfg=dog2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
