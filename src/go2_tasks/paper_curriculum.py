"""Performance-based terrain curriculum for the paper-style Go2 teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_from_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  *,
  command_name: str,
  reward_name: str = "track_linear_velocity",
  promote_threshold: float = 0.85,
  demote_threshold: float = 0.50,
  minimum_command_speed: float = 0.10,
) -> dict[str, torch.Tensor]:
  """Change terrain levels using normalized linear-velocity reward.

  A reward value of one means perfect tracking regardless of the configured
  reward weight or control timestep. Standing and pure-yaw commands do not
  affect the terrain level because their planar tracking task is trivial.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward_cfg = env.reward_manager.get_term_cfg(reward_name)
  if reward_cfg.weight <= 0:
    raise ValueError(f"Curriculum reward {reward_name!r} must have positive weight")

  episode_steps = env.episode_length_buf[env_ids]
  duration = (episode_steps * env.step_dt).clamp_min(env.step_dt)
  weighted_sum = env.reward_manager._episode_sums[reward_name][env_ids]
  tracking_quality = weighted_sum / (reward_cfg.weight * duration)
  tracking_quality = tracking_quality.clamp(0.0, 1.0)

  planar_speed = torch.linalg.vector_norm(command[env_ids, :2], dim=1)
  eligible = (planar_speed > minimum_command_speed) & (episode_steps > 0)
  robot = env.scene["robot"]
  distance = torch.linalg.vector_norm(
    robot.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  # Preserve the original paper's spatial completion requirement. Do not add
  # a yaw-reward gate: heading tracking already steers the robot off the tile.
  exited = distance > terrain_generator.size[0] / 2
  alive = ~env.termination_manager.terminated[env_ids]
  move_up = eligible & alive & exited & (tracking_quality >= promote_threshold)
  move_down = eligible & (distance < planar_speed * duration * demote_threshold)
  move_down &= ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": levels.mean(),
    "max": levels.max(),
    "tracking_quality": tracking_quality.mean(),
    "promote_fraction": move_up.float().mean(),
    "demote_fraction": move_down.float().mean(),
  }
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  if terrain_origins.shape[1] == len(sub_terrain_names):
    for index, name in enumerate(sub_terrain_names):
      mask = terrain.terrain_types == index
      if mask.any():
        result[name] = levels[mask].mean()
  return result


__all__ = ["terrain_levels_from_tracking"]

