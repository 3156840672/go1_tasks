from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import armv5_reach_env_cfg
from .rl_cfg import armv5_reach_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Reach-ArmV5",
  env_cfg=armv5_reach_env_cfg(),
  play_env_cfg=armv5_reach_env_cfg(play=True),
  rl_cfg=armv5_reach_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
