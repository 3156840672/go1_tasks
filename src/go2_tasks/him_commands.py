"""Velocity commands with explicit in-place turning samples for Go2 HIM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


@dataclass(kw_only=True)
class HimVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turning_envs: float = 0.1
  min_turning_yaw: float = 0.2

  def build(self, env) -> HimVelocityCommand:
    return HimVelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 <= self.rel_turning_envs <= 1.0:
      raise ValueError("rel_turning_envs must be in [0, 1]")
    low, high = self.ranges.ang_vel_z
    if self.min_turning_yaw <= 0 or low > -self.min_turning_yaw or high < self.min_turning_yaw:
      raise ValueError("turning yaw range must cover both signs of min_turning_yaw")


class HimVelocityCommand(UniformVelocityCommand):
  cfg: HimVelocityCommandCfg

  def __init__(self, cfg: HimVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.is_turning_env[env_ids] = False
    turn_ids = env_ids[
      torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turning_envs
    ]
    if len(turn_ids) == 0:
      return

    self.is_turning_env[turn_ids] = True
    # The inherited sampler may also have flagged these worlds as standing,
    # forward-only, heading-controlled, or world-frame.  ``_update_command``
    # would overwrite the yaw sample in every one of those cases, so clear
    # them: a turning world is exactly "zero linear velocity, fixed yaw rate".
    for flag in (
      self.is_standing_env,
      self.is_forward_env,
      self.is_heading_env,
      self.is_world_env,
    ):
      flag[turn_ids] = False
    self.vel_command_b[turn_ids, :2] = 0.0

    low, high = self.cfg.ranges.ang_vel_z
    positive = torch.rand(len(turn_ids), device=self.device) < 0.5
    magnitude = torch.empty(len(turn_ids), device=self.device)
    # ``magnitude[mask].uniform_()`` fills a temporary copy and leaves the real
    # entries uninitialised (which reads back as ~0 commands), so each sign is
    # drawn separately and written back by index.
    for side, bounds in (
      (positive, (self.cfg.min_turning_yaw, high)),
      (~positive, (self.cfg.min_turning_yaw, -low)),
    ):
      side_ids = side.nonzero(as_tuple=False).flatten()
      if len(side_ids) > 0:
        magnitude[side_ids] = torch.empty(
          len(side_ids), device=self.device
        ).uniform_(*bounds)
    self.vel_command_b[turn_ids, 2] = torch.where(
      positive, magnitude, -magnitude
    )
    self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]
