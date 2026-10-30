"""CTS Go1 observation and student-policy contract."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import BoxPyramidStairsTerrainCfg, HfPyramidSlopedTerrainCfg

from go1_tasks import TASK_IDS
from go1_tasks.cts import (
  CTSActor,
  feet_regulation,
  linear_velocity_error_l2,
  standing_missing_foot_contacts,
  student_group_mask,
  track_yaw_velocity,
)
from go1_tasks.rl_cfg import CtsPpoAlgorithmCfg


def test_cts_dense_velocity_cost() -> None:
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_command=lambda name: torch.tensor([[0.5, 0.0, 0.0]])
    ),
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_b=torch.tensor([[0.2, 0.0, 0.0]]))
      )
    },
  )
  torch.testing.assert_close(
    linear_velocity_error_l2(env, "twist"), torch.tensor([0.09])
  )


def test_cts_yaw_reward_ignores_roll_pitch() -> None:
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_command=lambda name: torch.tensor([[0.0, 0.0, 0.15]])
    ),
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_ang_vel_b=torch.tensor([[1.0, -1.0, 0.0]]))
      )
    },
  )
  torch.testing.assert_close(
    track_yaw_velocity(env, "twist", 0.25), torch.tensor([math.exp(-0.36)])
  )
  with pytest.raises(ValueError, match="std must be positive"):
    track_yaw_velocity(env, "twist", 0.0)


def test_feet_regulation_matches_paper_formula() -> None:
  height_scale = 0.025 * 0.278
  env = SimpleNamespace(
    scene={
      "heights": SimpleNamespace(
        data=SimpleNamespace(
          heights=torch.tensor([[0.023, 0.023 + height_scale, 0.123, 0.0]])
        )
      ),
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          site_lin_vel_w=torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
          )
        )
      ),
    }
  )
  asset_cfg = SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])
  result = feet_regulation(env, "heights", 0.278, 0.023, asset_cfg)
  expected = 1.0 + 4.0 * math.exp(-1.0) + math.exp(-0.1 / height_scale)
  torch.testing.assert_close(result, torch.tensor([expected]))
  with pytest.raises(ValueError, match="desired_body_height"):
    feet_regulation(env, "heights", 0.0, 0.023, asset_cfg)
  with pytest.raises(ValueError, match="foot_radius"):
    feet_regulation(env, "heights", 0.278, -0.023, asset_cfg)


def test_standing_missing_foot_contacts() -> None:
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_command=lambda name: torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
      )
    ),
    scene={
      "feet": SimpleNamespace(
        data=SimpleNamespace(
          found=torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
        )
      )
    },
  )
  torch.testing.assert_close(
    standing_missing_foot_contacts(env, "feet", "twist"),
    torch.tensor([1.0, 0.0, 0.0]),
  )


def test_cts_task_configuration() -> None:
  cfg = load_env_cfg(TASK_IDS["cts"])
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "height_scan" in cfg.observations["critic"].terms
  assert cfg.rewards["track_linear_velocity"].weight == 6.0
  assert cfg.rewards["track_angular_velocity"].weight == 3.0
  assert cfg.rewards["track_yaw_velocity"].weight == 2.0
  assert cfg.rewards["track_yaw_velocity"].params["std"] == 0.25
  assert cfg.rewards["upright"].weight == 0.1
  assert cfg.rewards["action_rate_l2"].weight == -0.1
  assert cfg.rewards["action_acc_l2"].weight == -0.02
  assert not {"air_time", "foot_clearance", "foot_swing_height"} & cfg.rewards.keys()
  assert cfg.rewards["feet_regulation"].func is feet_regulation
  assert cfg.rewards["feet_regulation"].weight == -0.05
  assert cfg.rewards["feet_regulation"].params["desired_body_height"] == 0.278
  assert cfg.rewards["feet_regulation"].params["foot_radius"] == 0.023
  assert cfg.rewards["feet_regulation"].params["asset_cfg"].preserve_order is True
  assert cfg.rewards["foot_slip"].weight == -0.3
  assert cfg.rewards["standing_missing_foot_contacts"].weight == -0.3
  assert cfg.rewards["soft_landing"].weight == -1e-5
  assert cfg.rewards["shank_collision"].weight == -0.3
  assert cfg.rewards["linear_velocity_error_l2"].weight == -1.0
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.rel_standing_envs == 0.1
  assert twist.rel_heading_envs == 0.1
  assert twist.rel_forward_envs == 0.1
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.max_init_terrain_level == 0
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert terrain_generator.sub_terrains["pyramid_stairs"].proportion == 0.275
  assert terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion == 0.275
  stairs = terrain_generator.sub_terrains["pyramid_stairs"]
  stairs_inv = terrain_generator.sub_terrains["pyramid_stairs_inv"]
  slope = terrain_generator.sub_terrains["hf_pyramid_slope"]
  assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
  assert isinstance(stairs_inv, BoxPyramidStairsTerrainCfg)
  assert isinstance(slope, HfPyramidSlopedTerrainCfg)
  assert stairs.step_height_range == (0.0, 0.08)
  assert stairs_inv.step_width == 0.35
  assert slope.slope_range == (0.0, 0.6)
  assert sum(t.proportion for t in terrain_generator.sub_terrains.values()) == 1.0
  assert cfg.curriculum["command_vel"].params["velocity_stages"][0]["lin_vel_x"] == (
    -0.3,
    0.6,
  )
  assert cfg.curriculum["command_vel"].params["velocity_stages"][1]["lin_vel_x"] == (
    -0.6,
    0.8,
  )
  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert [stage["step"] for stage in stages] == [
    0,
    1200 * 24,
    3600 * 24,
    5600 * 24,
  ]
  assert stages[0]["lin_vel_y"] == (-0.2, 0.2)
  assert stages[2]["lin_vel_y"] == (-0.7, 0.7)
  assert stages[3]["lin_vel_y"] == (-1.0, 1.0)
  assert cfg.observations["cts_history"].history_length == 5
  assert cfg.observations["cts_student"].terms["mask"].params["student_only"] is False
  play = load_env_cfg(TASK_IDS["cts"], play=True)
  assert cfg.sim.nconmax == 128
  assert play.sim.nconmax == 128
  assert play.scene.terrain is not None
  assert play.scene.terrain.terrain_generator is not None
  assert (
    play.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion
    == 0.275
  )
  play_stairs = play.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"]
  assert isinstance(play_stairs, BoxPyramidStairsTerrainCfg)
  assert play_stairs.step_height_range == (0.0, 0.08)
  assert play.observations["cts_student"].terms["mask"].params["student_only"] is True
  rl_cfg = load_rl_cfg(TASK_IDS["cts"])
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.algorithm.class_name.endswith(".CTSPPO")
  assert isinstance(rl_cfg.algorithm, CtsPpoAlgorithmCfg)
  assert rl_cfg.algorithm.student_learning_rate == 1e-3
  assert rl_cfg.algorithm.student_epochs == 5
  assert rl_cfg.algorithm.student_batch_size == 512
  assert rl_cfg.save_interval == 100
  for key in ("rough", "flat"):
    other = load_env_cfg(TASK_IDS[key])
    assert other.rewards["upright"].weight == 1.0
    assert other.rewards["foot_clearance"].weight == -0.1
    assert other.rewards["soft_landing"].weight == -1e-5
    assert load_rl_cfg(TASK_IDS[key]).save_interval == 50
  assert "foot_clearance" not in load_env_cfg(TASK_IDS["mob"]).rewards
  assert load_rl_cfg(TASK_IDS["mob"]).save_interval == 50
  assert cfg.events["cts_link_inertia"].mode == "startup"
  assert cfg.events["cts_link_inertia"].params["alpha_range"] == (
    0.5 * math.log(0.9),
    0.5 * math.log(1.1),
  )
  assert cfg.events["cts_payload"].params["mass_range"] == (-0.5, 2.0)
  assert cfg.events["foot_friction_slide"].params["ranges"] == (0.3, 1.5)
  assert cfg.events["foot_friction_slide"].params["shared_random"] is True
  assert cfg.events["base_com"].params["ranges"] == {
    0: (-0.05, 0.05),
    1: (-0.035, 0.035),
    2: (-0.035, 0.035),
  }
  assert cfg.events["cts_pd_gains"].params["kp_range"] == (0.8, 1.2)
  assert cfg.events["cts_effort_limits"].params["effort_limit_range"] == (0.8, 1.2)
  assert list(cfg.events).index("cts_link_inertia") < list(cfg.events).index("base_com")
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  assert all(actuator.delay_max_lag == 4 for actuator in articulation.actuators)
  assert "cts_link_inertia" not in play.events
  play_articulation = play.scene.entities["robot"].articulation
  assert play_articulation is not None
  assert all(actuator.delay_max_lag == 0 for actuator in play_articulation.actuators)


def test_cts_group_split_requires_teacher_worlds() -> None:
  env = SimpleNamespace(num_envs=4, device="cpu")
  torch.testing.assert_close(
    student_group_mask(env), torch.tensor([[0.0], [0.0], [0.0], [1.0]])
  )
  for count in (1, 2, 3, 5):
    env.num_envs = count
    with pytest.raises(ValueError, match="positive multiple of four"):
      student_group_mask(env)
  env.num_envs = 1
  torch.testing.assert_close(
    student_group_mask(env, student_only=True), torch.ones(1, 1)
  )
  torch.testing.assert_close(
    student_group_mask(env, teacher_only=True), torch.zeros(1, 1)
  )


def test_student_action_does_not_depend_on_privileged_observation() -> None:
  obs = TensorDict(
    {
      "actor": torch.randn(2, 45),
      "critic": torch.randn(2, 259),
      "cts_history": torch.randn(2, 225),
      "cts_student": torch.tensor([[0.0], [1.0]]),
    },
    batch_size=[2],
  )
  actor = CTSActor(
    obs,
    {"actor": ["actor"]},
    "actor",
    12,
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  baseline = actor(obs)
  changed = obs.clone()
  critic_obs = obs["critic"]
  assert isinstance(critic_obs, torch.Tensor)
  changed["critic"] = torch.randn_like(critic_obs)
  updated = actor(changed)
  torch.testing.assert_close(baseline[1], updated[1])
  assert not torch.equal(baseline[0], updated[0])
  student_export = actor.as_onnx()
  torch.testing.assert_close(
    student_export(obs["actor"][1:2], obs["cts_history"][1:2]),
    baseline[1:2],
  )
  actor.update_normalization(obs)
  normalized = actor(obs)
  assert not torch.equal(normalized, baseline)
  normalized_export = actor.as_onnx()
  torch.testing.assert_close(
    normalized_export(obs["actor"][1:2], obs["cts_history"][1:2]),
    normalized[1:2],
  )
