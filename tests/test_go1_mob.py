"""Checks for the behavior-conditioned Go1 training task."""

import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import STIFFNESS_HIP
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

from go1_tasks import TASK_IDS
from go1_tasks.mob import (
  GAITS,
  BehaviorCommand,
  BehaviorCommandCfg,
  desired_contact_states,
  foot_phase,
  raibert_foot_targets,
)
from go1_tasks.mob_curriculum import (
  GridAdaptiveVelocityCommand,
  GridAdaptiveVelocityCommandCfg,
)


def test_gait_phase_offsets() -> None:
  cycle = torch.full((4,), 0.1)
  offsets = torch.tensor(list(GAITS.values()))
  actual = foot_phase(cycle, offsets)
  expected = torch.tensor(
    [
      [0.1, 0.1, 0.1, 0.1],
      [0.1, 0.6, 0.6, 0.1],
      [0.6, 0.6, 0.1, 0.1],
      [0.6, 0.1, 0.6, 0.1],
    ]
  )
  torch.testing.assert_close(actual, expected)


def test_contact_target_is_smooth_at_swing_boundary() -> None:
  phases = torch.tensor([[0.25, 0.49, 0.50, 0.51, 0.75]])
  contact = desired_contact_states(phases)
  assert contact[0, 0] > 0.99
  assert contact[0, 4] < 0.01
  assert torch.max(torch.abs(contact[0, 1:4] - 0.5)) < 0.06


def test_raibert_foot_targets_use_front_rear_yaw_sign() -> None:
  phases = torch.full((1, 4), 0.1)
  twist = torch.tensor([[1.0, 0.0, 1.0]])
  frequency = torch.tensor([[2.0]])
  stance_width = torch.tensor([[0.3]])
  desired_x, desired_y = raibert_foot_targets(phases, twist, frequency, stance_width)
  torch.testing.assert_close(desired_x, torch.tensor([[0.3, 0.3, -0.15, -0.15]]))
  torch.testing.assert_close(
    desired_y, torch.tensor([[-0.133125, 0.166875, -0.166875, 0.133125]])
  )


def test_mob_config_does_not_change_existing_go1() -> None:
  assert "Mjlab-Velocity-GaitFree-Unitree-Go1" not in list_tasks()
  cfg = load_env_cfg(TASK_IDS["mob"])
  flat = load_env_cfg(TASK_IDS["flat"])
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert any(s.name == "limb_ground_contact" for s in cfg.scene.sensors or ())
  assert cfg.observations["actor"].history_length == 30
  assert "height_scan" not in cfg.observations["actor"].terms
  behavior_cfg = cfg.commands["behavior"]
  assert isinstance(behavior_cfg, BehaviorCommandCfg)
  assert behavior_cfg.body_height == (0.20, 0.38)
  assert "body_height_error" in cfg.metrics
  assert isinstance(cfg.commands["twist"], GridAdaptiveVelocityCommandCfg)
  assert not cfg.curriculum
  assert not isinstance(
    load_env_cfg(TASK_IDS["mob"], play=True).commands["twist"],
    GridAdaptiveVelocityCommandCfg,
  )
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  assert action_cfg.scale == {
    ".*_hip_joint": 0.125,
    ".*_thigh_joint": 0.25,
    ".*_calf_joint": 0.25,
  }
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  actuator = articulation.actuators[0]
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.stiffness == 20
  assert cfg.events["body_inertia_dr"].params["alpha_range"] == (-0.05, 0.05)
  assert cfg.events["pd_gains_dr"].params["kp_range"] == (0.9, 1.1)
  assert cfg.events["joint_friction_dr"].params["ranges"] == (0.0, 0.05)
  assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.1, 0.1)
  assert cfg.events["push_robot"].interval_range_s == (4.0, 8.0)
  play_cfg = load_env_cfg(TASK_IDS["mob"], play=True)
  assert "push_robot" not in play_cfg.events
  assert "body_inertia_dr" not in play_cfg.events
  flat_articulation = flat.scene.entities["robot"].articulation
  assert flat_articulation is not None
  flat_actuator = flat_articulation.actuators[0]
  assert isinstance(flat_actuator, BuiltinPositionActuatorCfg)
  assert flat_actuator.stiffness == STIFFNESS_HIP


def test_mob_environment_step() -> None:
  cfg = load_env_cfg(TASK_IDS["mob"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (1, 1710)
    twist = env.command_manager.get_term("twist")
    assert isinstance(twist, GridAdaptiveVelocityCommand)
    if twist.active_bin[0] >= 0:
      assert twist.weights[twist.active_bin[0]] > 0
    behavior = env.command_manager.get_term("behavior")
    assert isinstance(behavior, BehaviorCommand)
    assert behavior.command.shape == (1, 8)
    for column, bounds in enumerate(
      (
        behavior.cfg.frequency,
        behavior.cfg.body_height,
        behavior.cfg.body_pitch,
        behavior.cfg.stance_width,
        behavior.cfg.swing_height,
      ),
      start=3,
    ):
      assert bounds[0] <= behavior.command[0, column] <= bounds[1]
    phase_before = behavior.phases.clone()
    observations, reward, *_ = env.step(torch.zeros(1, 12))
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (1, 1710)
    assert torch.isfinite(reward).all()
    assert not torch.equal(behavior.phases, phase_before)
  finally:
    env.close()


def test_height_command_changes_reward() -> None:
  cfg = load_env_cfg(TASK_IDS["mob"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    behavior = env.command_manager.get_term("behavior")
    assert isinstance(behavior, BehaviorCommand)
    reward_cfg = env.reward_manager.get_term_cfg("mob")
    height = env.scene["robot"].data.root_link_pos_w[0, 2].item()
    matched_height = min(max(height, 0.20), 0.38)
    distant_height = 0.20 if height > 0.29 else 0.38
    behavior.values[0, 4] = matched_height
    matched = reward_cfg.func(env, **reward_cfg.params)
    behavior.values[0, 4] = distant_height
    mismatched = reward_cfg.func(env, **reward_cfg.params)
    assert matched.item() > 1.5 * mismatched.item()
  finally:
    env.close()


def test_limb_ground_contact_reduces_reward() -> None:
  cfg = load_env_cfg(TASK_IDS["mob"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    reward_cfg = env.reward_manager.get_term_cfg("mob")
    force = env.scene["limb_ground_contact"].data.force
    assert force is not None
    original = force.clone()
    force.zero_()
    without_collision = reward_cfg.func(env, **reward_cfg.params)
    force[0, 0, 2] = 1.0
    with_collision = reward_cfg.func(env, **reward_cfg.params)
    assert with_collision.item() < without_collision.item()
    force.copy_(original)
  finally:
    env.close()


def test_velocity_grid_expands_only_after_success() -> None:
  cfg = load_env_cfg(TASK_IDS["mob"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    twist = env.command_manager.get_term("twist")
    assert isinstance(twist, GridAdaptiveVelocityCommand)
    center_bin = torch.argmin(torch.sum(twist.centers**2, dim=-1))
    twist.weights.zero_()
    twist.weights[center_bin] = 1.0
    twist.active_bin[0] = center_bin
    twist.sample_count[0] = 49
    twist.lin_score_sum[0] = 49.0
    twist.yaw_score_sum[0] = 49.0
    twist._update_grid(torch.tensor([0]))
    assert torch.count_nonzero(twist.weights) == 1

    twist.sample_count[0] = 50
    twist.lin_score_sum[0] = 0.0
    twist.yaw_score_sum[0] = 50.0
    twist._update_grid(torch.tensor([0]))
    assert torch.count_nonzero(twist.weights) == 1

    twist.lin_score_sum[0] = 50.0
    twist._update_grid(torch.tensor([0]))
    assert torch.count_nonzero(twist.weights) > 1
    state = twist.curriculum_state_dict()
    twist.weights.zero_()
    twist.load_curriculum_state_dict(state)
    torch.testing.assert_close(twist.weights, state["weights"])
  finally:
    env.close()
