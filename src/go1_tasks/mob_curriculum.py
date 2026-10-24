"""Reward-threshold velocity sampling for the Go1 MoB task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GridAdaptiveVelocityCommand(UniformVelocityCommand):
  """Grow a weighted velocity grid when tracking succeeds in sampled cells."""

  def __init__(self, cfg: GridAdaptiveVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if (
      cfg.heading_command
      or cfg.rel_world_envs
      or cfg.rel_forward_envs
      or cfg.init_velocity_prob
    ):
      raise ValueError("Grid curriculum requires body-frame velocity commands only.")
    if any(n < 2 for n in cfg.num_bins):
      raise ValueError("Each velocity grid axis needs at least two bins.")

    limits = (
      cfg.ranges.lin_vel_x,
      cfg.ranges.lin_vel_y,
      cfg.ranges.ang_vel_z,
    )
    axes = [
      torch.linspace(low, high, n + 1, device=self.device)
      for (low, high), n in zip(limits, cfg.num_bins, strict=True)
    ]
    self.cell_width = torch.tensor(
      [(high - low) / n for (low, high), n in zip(limits, cfg.num_bins, strict=True)],
      device=self.device,
    )
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in axes]
    self.centers = torch.stack(torch.meshgrid(*centers, indexing="ij"), dim=-1).reshape(
      -1, 3
    )
    initial_limits = torch.tensor(cfg.initial_limits, device=self.device)
    initial_cells = (self.centers.abs() <= initial_limits).all(dim=-1)
    if not initial_cells.any():
      raise ValueError("Initial velocity limits must include at least one grid cell.")
    self.weights = initial_cells.float()
    self.neighbor_offsets = torch.cartesian_prod(
      *(torch.tensor((-1, 0, 1), device=self.device) for _ in range(3))
    )
    self.active_bin = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.lin_score_sum = torch.zeros(self.num_envs, device=self.device)
    self.yaw_score_sum = torch.zeros(self.num_envs, device=self.device)
    self.sample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.metrics["curriculum_coverage"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    valid = (self.active_bin >= 0) & (self._env.episode_length_buf > 0)
    lin_error = torch.sum(
      (self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]) ** 2,
      dim=-1,
    )
    yaw_error = (
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) ** 2
    self.lin_score_sum += torch.exp(-lin_error / 0.25) * valid
    self.yaw_score_sum += torch.exp(-yaw_error / 0.25) * valid
    self.sample_count += valid.long()

  def _update_grid(self, env_ids: torch.Tensor) -> None:
    cfg = cast(GridAdaptiveVelocityCommandCfg, self.cfg)
    eligible = (self.active_bin[env_ids] >= 0) & (
      self.sample_count[env_ids] >= cfg.min_interval_s / self._env.step_dt
    )
    ids = env_ids[eligible]
    if len(ids) == 0:
      return
    successful = (
      self.lin_score_sum[ids] / self.sample_count[ids] >= cfg.lin_threshold
    ) & (self.yaw_score_sum[ids] / self.sample_count[ids] >= cfg.yaw_threshold)
    winning_bins = self.active_bin[ids[successful]]
    if len(winning_bins) == 0:
      return

    nx, ny, nz = cfg.num_bins
    coords = torch.stack(
      (winning_bins // (ny * nz), (winning_bins // nz) % ny, winning_bins % nz),
      dim=-1,
    )
    neighbors = coords[:, None, :] + self.neighbor_offsets[None, :, :]
    valid = ((neighbors >= 0) & (neighbors < neighbors.new_tensor((nx, ny, nz)))).all(
      dim=-1
    )
    neighbors = neighbors[valid]
    flat_ids = torch.unique(
      (neighbors[:, 0] * ny + neighbors[:, 1]) * nz + neighbors[:, 2]
    )
    self.weights[flat_ids] = (self.weights[flat_ids] + cfg.weight_increment).clamp(
      max=1.0
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._update_grid(env_ids)
    super()._resample_command(env_ids)
    moving = env_ids[~self.is_standing_env[env_ids]]
    standing = env_ids[self.is_standing_env[env_ids]]
    self.active_bin[standing] = -1
    self.vel_command_b[standing] = 0.0
    self.vel_command_w[standing] = 0.0
    if len(moving) > 0:
      bins = torch.multinomial(self.weights, len(moving), replacement=True)
      sampled = (
        self.centers[bins]
        + (torch.rand(len(moving), 3, device=self.device) - 0.5) * self.cell_width
      )
      self.active_bin[moving] = bins
      self.vel_command_b[moving] = sampled
      self.vel_command_w[moving] = sampled
    self.lin_score_sum[env_ids] = 0.0
    self.yaw_score_sum[env_ids] = 0.0
    self.sample_count[env_ids] = 0
    self.metrics["curriculum_coverage"].fill_(
      torch.count_nonzero(self.weights) / self.weights.numel()
    )

  def curriculum_state_dict(self) -> dict:
    cfg = cast(GridAdaptiveVelocityCommandCfg, self.cfg)
    return {"weights": self.weights.cpu().clone(), "num_bins": cfg.num_bins}

  def load_curriculum_state_dict(self, state: dict) -> None:
    cfg = cast(GridAdaptiveVelocityCommandCfg, self.cfg)
    if tuple(state["num_bins"]) != cfg.num_bins:
      raise ValueError("Checkpoint velocity grid size does not match the task config.")
    weights = state["weights"]
    if weights.shape != self.weights.shape:
      raise ValueError("Checkpoint velocity grid weights have the wrong shape.")
    self.weights.copy_(weights.to(device=self.device))


@dataclass(kw_only=True)
class GridAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
  num_bins: tuple[int, int, int] = (11, 7, 11)
  initial_limits: tuple[float, float, float] = (0.4, 0.3, 0.4)
  lin_threshold: float = 0.8
  yaw_threshold: float = 0.5
  min_interval_s: float = 1.0
  weight_increment: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> GridAdaptiveVelocityCommand:
    return GridAdaptiveVelocityCommand(self, env)
