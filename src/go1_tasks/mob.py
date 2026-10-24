"""Behavior-conditioned Go1 locomotion primitives inspired by Walk These Ways."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv


# Offset order: phase, bound, pace. Foot order: FR, FL, RR, RL.
GAITS = {
  "Pronk": (0.0, 0.0, 0.0),
  "Trot": (0.5, 0.0, 0.0),
  "Bound": (0.0, 0.5, 0.0),
  "Pace": (0.0, 0.0, 0.5),
}


def foot_phase(cycle: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
  """Return normalized phase for [FR, FL, RR, RL]."""
  phase, bound, pace = offsets.unbind(-1)
  shifts = torch.stack(
    (bound + pace, phase + bound, phase + pace, torch.zeros_like(phase)), dim=-1
  )
  return torch.remainder(cycle[:, None] + shifts, 1.0)


def raibert_foot_targets(
  phases: torch.Tensor,
  twist: torch.Tensor,
  frequency: torch.Tensor,
  stance_width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return desired x/y foot positions in FR, FL, RR, RL order."""
  nominal_x = phases.new_tensor((0.225, 0.225, -0.225, -0.225))
  side = phases.new_tensor((-1.0, 1.0, -1.0, 1.0))
  front_rear = phases.new_tensor((1.0, 1.0, -1.0, -1.0))
  phase_offset = torch.abs(1.0 - 2.0 * phases) - 0.5
  half_period = 0.5 / frequency.clamp_min(0.1)
  desired_x = nominal_x + phase_offset * twist[:, 0:1] * half_period
  desired_y = (
    side * stance_width / 2.0
    + front_rear * phase_offset * twist[:, 2:3] * 0.225 * half_period
  )
  return desired_x, desired_y


class BehaviorCommand(CommandTerm):
  cfg: BehaviorCommandCfg

  def __init__(self, cfg: BehaviorCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.values = torch.zeros(self.num_envs, 8, device=self.device)
    self.cycle = torch.zeros(self.num_envs, device=self.device)
    self._gui_controls: tuple | None = None
    self._gui_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.values

  @property
  def phases(self) -> torch.Tensor:
    return foot_phase(self.cycle, self.values[:, :3])

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    gait_ids = torch.randint(0, len(GAITS), (count,), device=self.device)
    gaits = torch.tensor(tuple(GAITS.values()), device=self.device)
    self.values[env_ids, :3] = gaits[gait_ids]
    for col, bounds in enumerate(
      (
        self.cfg.frequency,
        self.cfg.body_height,
        self.cfg.body_pitch,
        self.cfg.stance_width,
        self.cfg.swing_height,
      ),
      start=3,
    ):
      self.values[env_ids, col] = torch.empty(count, device=self.device).uniform_(
        *bounds
      )

  def _update_command(self) -> None:
    self.cycle = torch.remainder(
      self.cycle + self._env.step_dt * self.values[:, 3], 1.0
    )
    if self._gui_controls is not None:
      gait, sliders = self._gui_controls
      assert self._gui_env_idx is not None
      idx = self._gui_env_idx()
      self.values[idx, :3] = torch.tensor(GAITS[gait.value], device=self.device)
      self.values[idx, 3:] = torch.tensor(
        [slider.value for slider in sliders], device=self.device
      )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self.cycle[env_ids] = 0.0
    return super().reset(env_ids)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action=None,
  ) -> None:
    del on_change, request_action
    with server.gui.add_folder(name.capitalize()):
      gait = server.gui.add_dropdown("Gait", options=tuple(GAITS), initial_value="Trot")
      sliders = [
        server.gui.add_slider(
          label, min=limits[0], max=limits[1], step=step, initial_value=initial
        )
        for label, limits, step, initial in (
          ("Frequency (Hz)", self.cfg.frequency, 0.1, 2.5),
          ("Body height (m)", self.cfg.body_height, 0.01, 0.30),
          ("Body pitch (rad)", self.cfg.body_pitch, 0.01, 0.0),
          ("Stance width (m)", self.cfg.stance_width, 0.01, 0.30),
          ("Swing height (m)", self.cfg.swing_height, 0.01, 0.10),
        )
      ]
    self._gui_controls = (gait, sliders)
    self._gui_env_idx = get_env_idx


@dataclass(kw_only=True)
class BehaviorCommandCfg(CommandTermCfg):
  frequency: tuple[float, float] = (1.5, 4.0)
  body_height: tuple[float, float] = (0.20, 0.38)
  body_pitch: tuple[float, float] = (-0.15, 0.15)
  stance_width: tuple[float, float] = (0.22, 0.40)
  swing_height: tuple[float, float] = (0.04, 0.18)

  def build(self, env: ManagerBasedRlEnv) -> BehaviorCommand:
    return BehaviorCommand(self, env)


def phase_clock(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BehaviorCommand)
  return torch.sin(2.0 * math.pi * term.phases)


def desired_contact_states(phases: torch.Tensor) -> torch.Tensor:
  """Smoothed half-cycle stance target from the reference implementation."""
  scale = 0.07 * math.sqrt(2.0)

  def cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / scale))

  return cdf(phases) * (1.0 - cdf(phases - 0.5)) + cdf(phases - 1.0) * (
    1.0 - cdf(phases - 1.5)
  )


def body_height_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  behavior = env.command_manager.get_term(command_name)
  assert isinstance(behavior, BehaviorCommand)
  return (env.scene["robot"].data.root_link_pos_w[:, 2] - behavior.command[:, 4]).abs()


def mob_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  velocity_command_name: str,
  contact_sensor_name: str,
  limb_contact_sensor_name: str,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reference-scale task reward times exponential auxiliary costs."""
  robot = env.scene[asset_cfg.name]
  twist = env.command_manager.get_command(velocity_command_name)
  assert twist is not None

  lin_vel = robot.data.root_link_lin_vel_b
  ang_vel = robot.data.root_link_ang_vel_b
  xy_error = torch.sum((twist[:, :2] - lin_vel[:, :2]) ** 2, dim=-1)
  yaw_error = (twist[:, 2] - ang_vel[:, 2]) ** 2
  # RewardManager applies dt; the reference's negative-reward scale cancels that dt.
  task = torch.exp(-xy_error / 0.25) + 0.5 * torch.exp(-yaw_error / 0.25)

  feet_vel_w = robot.data.site_lin_vel_w[:, asset_cfg.site_ids]
  contact = env.scene[contact_sensor_name]
  assert contact.data.found is not None
  foot_vel_sq = torch.sum(feet_vel_w**2, dim=-1)
  foot_vel_xy_sq = torch.sum(feet_vel_w[..., :2] ** 2, dim=-1)
  limb_contact = env.scene[limb_contact_sensor_name]
  assert limb_contact.data.force is not None
  limb_collisions = (
    torch.linalg.vector_norm(limb_contact.data.force, dim=-1) > 0.1
  ).sum(-1)

  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  action_scale = joint_action.scale
  action_delta = (
    env.action_manager.action - env.action_manager.prev_action
  ) * action_scale
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  ) * action_scale

  joint_vel = robot.data.joint_vel
  auxiliary = (
    -0.02 * lin_vel[:, 2] ** 2
    - 0.001 * torch.sum(ang_vel[:, :2] ** 2, dim=-1)
    - 0.04 * (foot_vel_xy_sq * (contact.data.found > 0).float()).sum(-1)
    - 0.0001 * torch.sum(robot.data.actuator_force**2, dim=-1)
    - 0.0001 * torch.sum(joint_vel**2, dim=-1)
    - 2.5e-7 * torch.sum(robot.data.joint_acc**2, dim=-1)
    - 10.0
    * torch.sum(
      (robot.data.soft_joint_pos_limits[..., 0] - robot.data.joint_pos).clamp_min(0)
      + (robot.data.joint_pos - robot.data.soft_joint_pos_limits[..., 1]).clamp_min(0),
      dim=-1,
    )
    - 0.1 * torch.sum(action_delta**2, dim=-1) * (env.episode_length_buf > 1)
    - 0.1 * torch.sum(action_acc**2, dim=-1) * (env.episode_length_buf > 2)
    - 5.0 * limb_collisions
  )

  behavior = env.command_manager.get_term(command_name)
  assert isinstance(behavior, BehaviorCommand)
  cmd = behavior.command
  phases = behavior.phases
  desired_contact = desired_contact_states(phases)
  assert contact.data.force is not None
  force_sq = torch.sum(contact.data.force**2, dim=-1)
  swing_force = ((1 - desired_contact) * (1 - torch.exp(-force_sq / 100.0))).mean(-1)
  stance_motion = (desired_contact * (1 - torch.exp(-foot_vel_sq / 10.0))).mean(-1)

  root_pos = robot.data.root_link_pos_w
  _, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
  height_error = (root_pos[:, 2] - cmd[:, 4]) ** 2
  pitch_error = (pitch - cmd[:, 5]) ** 2

  # Yaw-only body frame keeps the foot target level during body pitch and roll.
  feet_relative = robot.data.site_pos_w[:, asset_cfg.site_ids] - root_pos[:, None]
  foot_pos_b = quat_apply_inverse(
    yaw_quat(robot.data.root_link_quat_w)[:, None, :].expand(-1, 4, -1),
    feet_relative,
  )
  swing_progress = torch.where(phases < 0.5, 0.0, 2.0 * phases - 1.0)
  desired_x, desired_y = raibert_foot_targets(phases, twist, cmd[:, 3:4], cmd[:, 6:7])
  foot_error = (foot_pos_b[..., 0] - desired_x) ** 2 + (
    foot_pos_b[..., 1] - desired_y
  ) ** 2
  raibert_cost = foot_error.sum(-1)

  foot_height = env.scene[height_sensor_name].data.heights
  swing_profile = 1.0 - torch.abs(2.0 * swing_progress - 1.0)
  swing_target = 0.02 + cmd[:, 7:8] * swing_profile
  swing_cost = (((foot_height - swing_target) ** 2) * (1 - desired_contact)).sum(-1)
  auxiliary = (
    auxiliary
    - 4.0 * swing_force
    - 4.0 * stance_motion
    - 80.0 * height_error
    - 5.0 * pitch_error
    - 10.0 * raibert_cost
    - 30.0 * swing_cost
  )
  return task * torch.exp(auxiliary)
