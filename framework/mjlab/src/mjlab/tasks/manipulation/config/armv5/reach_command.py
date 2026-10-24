"""Command term for reaching a target 3D coordinate with the end-effector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ReachTargetCommand(CommandTerm):
  """Command specifying a 3D target for the end-effector to reach."""

  cfg: ReachTargetCommandCfg

  def __init__(self, cfg: ReachTargetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self._robot: Entity = env.scene[cfg.entity_name]
    ee_site_ids, _ = self._robot.find_sites(cfg.site_names[0])
    self._ee_site_ids = torch.tensor(ee_site_ids, device=self.device, dtype=torch.long)

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    ee_pos_w = self._robot.data.site_pos_w[:, self._ee_site_ids].squeeze(1)
    position_error = torch.norm(self.target_pos - ee_pos_w, dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["position_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.3, 0.0, 0.2], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )


@dataclass(kw_only=True)
class ReachTargetCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  site_names: tuple[str, ...] = ("grasp_site",)
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (0.1, 0.45)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.05, 0.35)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReachTargetCommand:
    return ReachTargetCommand(self, env)
