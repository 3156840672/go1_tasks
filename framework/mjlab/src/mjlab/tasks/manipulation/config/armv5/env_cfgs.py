"""ArmV5 reaching task: move end-effector to a target 3D coordinate.

Train with:
  uv run scripts/train.py task=Mjlab-Reach-ArmV5
Play with:
  uv run scripts/play.py task=Mjlab-Reach-ArmV5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots import ARMV5_ACTION_SCALE, get_armv5_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .reach_command import ReachTargetCommand, ReachTargetCommandCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _ee_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end-effector to target position, in base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReachTargetCommand):
    raise TypeError(
      f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  target_pos_w = command.target_pos
  distance_vec_w = target_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  return quat_apply(quat_inv(base_quat_w), distance_vec_w)


def _reach_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gaussian reward for EE distance to target: exp(-d² / std²)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReachTargetCommand):
    raise TypeError(
      f"Command '{command_name}' must be a ReachTargetCommand, got {type(command)}"
    )
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(command.target_pos - ee_pos_w, dim=-1)
  return torch.exp(-(distance**2) / std**2)


def armv5_reach_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """ArmV5 reaching task environment configuration."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "ee_to_target": ObservationTermCfg(
      func=_ee_to_target,
      params={
        "command_name": "reach_target",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=ARMV5_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "reach_target": ReachTargetCommandCfg(
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
      target_position_range=ReachTargetCommandCfg.TargetPositionRangeCfg(
        x=(0.1, 0.45),
        y=(-0.2, 0.2),
        z=(0.05, 0.35),
      ),
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  rewards = {
    "reach": RewardTermCfg(
      func=_reach_target_reward,
      weight=1.0,
      params={
        "command_name": "reach_target",
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  curriculum = {
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 500 * 24, "weight": -0.1},
          {"step": 1000 * 24, "weight": -1.0},
        ],
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4096,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="Link1",
      distance=1.5,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  cfg.scene.entities = {
    "robot": get_armv5_robot_cfg(),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["reach_target"].resampling_time_range = (4.0, 4.0)

  return cfg
