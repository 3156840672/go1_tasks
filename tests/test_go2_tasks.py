"""Go2 asset, registration, and environment smoke tests."""

import torch

from mjlab.actuator import DcMotorActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

from go2_tasks import TASK_IDS
from go2_tasks.env_cfgs import (
  GO2_AMP_REFERENCE_PATH,
  GO2_DR_BODY_NAMES,
  unitree_go2_amp_env_cfg,
)
from go2_tasks.deployment import (
  ACTION_SCALE,
  DEFAULT_JOINT_POS,
  JOINT_LOWER,
  JOINT_UPPER,
  TORQUE_LIMIT,
  policy_action_to_joint_targets,
)
from go2_tasks.robots.go2 import (
  FOOT_GEOM_NAMES,
  GO2_MOTOR_ACTUATOR_CFG,
  JOINT_NAMES,
  get_go2_robot_cfg,
)


def test_go2_asset_contract() -> None:
  entity = Entity(get_go2_robot_cfg())
  model = entity.spec.compile()
  assert entity.num_joints == 12
  assert entity.num_actuators == 12
  assert tuple(entity.site_names[-4:]) == ("FL", "FR", "RL", "RR")
  assert all(model.geom(name).id >= 0 for name in FOOT_GEOM_NAMES)
  for index, name in enumerate(JOINT_NAMES):
    lower, upper = model.joint(name).range
    assert abs(lower - JOINT_LOWER[index]) < 1e-6
    assert abs(upper - JOINT_UPPER[index]) < 1e-6
  assert GO2_MOTOR_ACTUATOR_CFG.stiffness == 25.0
  assert GO2_MOTOR_ACTUATOR_CFG.damping == 0.5
  assert GO2_MOTOR_ACTUATOR_CFG.saturation_effort == 23.5
  assert GO2_MOTOR_ACTUATOR_CFG.velocity_limit == 30.0
  for index in range(model.nu):
    actuator = model.actuator(index)
    assert tuple(actuator.forcerange) == (-23.5, 23.5)


def test_go2_task_family_registered() -> None:
  registered = set(list_tasks())
  assert set(TASK_IDS.values()) <= registered
  for key, task_id in TASK_IDS.items():
    cfg = load_env_cfg(task_id)
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    assert cfg.scene.entities["robot"].init_state.pos[2] == 0.38
    if key == "amp":
      twist = cfg.commands["twist"]
      assert twist.ranges.lin_vel_x == (-1.0, 1.0)
      assert twist.ranges.lin_vel_y == (-0.6, 0.6)
      assert twist.ranges.ang_vel_z == (-1.0, 1.0)
      assert twist.rel_standing_envs == 0.05
      assert cfg.rewards["track_angular_velocity"].weight == 2.0
      assert cfg.rewards["track_angular_velocity"].params["std"] == 0.25
      assert (
        cfg.events["amp_reference_state_init"].params["reference_path"]
        == GO2_AMP_REFERENCE_PATH
      )
    if key == "cts":
      articulation = cfg.scene.entities["robot"].articulation
      assert articulation is not None
      assert all(a.delay_max_lag == 4 for a in articulation.actuators)


def test_go2_randomization_skips_massless_feet() -> None:
  from mjlab.envs.mdp.dr import pseudo_inertia

  for key in ("mob", "cts"):
    cfg = load_env_cfg(TASK_IDS[key])
    matching = [e for e in cfg.events.values() if e.func is pseudo_inertia]
    assert matching
    for event in matching:
      assert tuple(event.params["asset_cfg"].body_names) == GO2_DR_BODY_NAMES


def test_go2_amp_keeps_domain_randomization_and_pushes() -> None:
  cfg = unitree_go2_amp_env_cfg(play=False)
  expected = {
    "encoder_bias",
    "base_com",
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "body_inertia_dr",
    "pd_gains_dr",
    "joint_friction_dr",
    "push_robot",
  }
  assert expected <= set(cfg.events)
  assert cfg.events["pd_gains_dr"].params["kp_range"] == (0.9, 1.1)
  assert cfg.events["pd_gains_dr"].params["kd_range"] == (0.9, 1.1)
  assert cfg.events["joint_friction_dr"].params["ranges"] == (0.0, 0.05)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.015, 0.015)
  push = cfg.events["push_robot"]
  assert push.interval_range_s == (4.0, 8.0)
  assert push.params["velocity_range"] == {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "yaw": (-0.3, 0.3),
  }
  reset = cfg.events["reset_robot_joints"]
  assert reset.params["position_range"] == (-0.1, 0.1)
  assert reset.params["velocity_range"] == (-0.1, 0.1)


def test_go2_mob_environment_step_is_finite() -> None:
  cfg = load_env_cfg(TASK_IDS["mob"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    assert isinstance(env.scene["robot"].actuators[0], DcMotorActuator)
    action = torch.zeros(1, env.action_manager.total_action_dim)
    for _ in range(5):
      observations, reward, *_ = env.step(action)
      assert torch.isfinite(observations["actor"]).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(env.scene["robot"].data.root_link_pos_w).all()
  finally:
    env.close()


def test_go2_deployment_action_contract() -> None:
  action = torch.arange(12, dtype=torch.float32).numpy() / 10
  targets = policy_action_to_joint_targets(action, clamp_to_limits=False)
  expected = DEFAULT_JOINT_POS + ACTION_SCALE * action
  assert tuple(TORQUE_LIMIT) == (23.5,) * 12
  assert tuple(targets) == tuple(expected)


def test_go2_flat_environment_step() -> None:
  cfg = load_env_cfg(TASK_IDS["flat"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 48)
    action_term = env.action_manager.get_term("joint_pos")
    assert tuple(action_term.target_names) == JOINT_NAMES
    observations, reward, *_ = env.step(torch.zeros(1, 12))
    assert observations["critic"].shape == (1, 72)
    assert torch.isfinite(reward).all()
  finally:
    env.close()
