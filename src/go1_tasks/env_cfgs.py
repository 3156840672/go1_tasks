"""Unitree Go1 velocity environment configurations."""

import math
from copy import deepcopy
from dataclasses import fields
from typing import Literal

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import BoxPyramidStairsTerrainCfg, HfPyramidSlopedTerrainCfg

from .cts import (
  feet_regulation,
  linear_velocity_error_l2,
  randomize_trunk_payload,
  standing_missing_foot_contacts,
  student_group_mask,
  track_yaw_velocity,
)
from .mob import BehaviorCommandCfg, body_height_error, mob_reward, phase_clock
from .mob_curriculum import GridAdaptiveVelocityCommandCfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go1_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  # Set raycast sensor frame to Go1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("trunk_collision", "head_collision"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace the base foot_friction with per-axis friction events for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_clearance"].weight = -0.1
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_go1_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough Go1 task with separate deployable and privileged observations."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 1024
  # Rough-terrain contacts can exceed the base task's 35-contact allocation.
  cfg.sim.nconmax = 128
  if not play:
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.max_init_terrain_level = 0
  actor = cfg.observations["actor"]
  actor.terms.pop("base_lin_vel")
  actor.terms.pop("height_scan")

  # Favor commanded translation without removing contact and stability costs.
  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_yaw_velocity"] = RewardTermCfg(
    func=track_yaw_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": 0.25},
  )
  cfg.rewards["upright"].weight = 0.1
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2, weight=-0.02
  )
  cfg.rewards["shank_collision"].weight = -0.3
  for reward_name in ("air_time", "foot_clearance", "foot_swing_height"):
    cfg.rewards.pop(reward_name)
  cfg.rewards["feet_regulation"] = RewardTermCfg(
    func=feet_regulation,
    weight=-0.05,
    params={
      "height_sensor_name": "foot_height_scan",
      "desired_body_height": cfg.scene.entities["robot"].init_state.pos[2],
      "foot_radius": 0.023,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("FR", "FL", "RR", "RL"), preserve_order=True
      ),
    },
  )
  cfg.rewards["foot_slip"].weight = -0.3
  cfg.rewards["standing_missing_foot_contacts"] = RewardTermCfg(
    func=standing_missing_foot_contacts,
    weight=-0.3,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  cfg.rewards["linear_velocity_error_l2"] = RewardTermCfg(
    func=linear_velocity_error_l2,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  terrain_generator = deepcopy(cfg.scene.terrain.terrain_generator)
  for name, proportion in {
    "flat": 0.15,
    "pyramid_stairs": 0.275,
    "pyramid_stairs_inv": 0.275,
    "hf_pyramid_slope": 0.075,
    "hf_pyramid_slope_inv": 0.075,
    "random_rough": 0.075,
    "wave_terrain": 0.075,
  }.items():
    terrain_generator.sub_terrains[name].proportion = proportion
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stairs = terrain_generator.sub_terrains[name]
    assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
    stairs.step_height_range = (0.0, 0.08)
    stairs.step_width = 0.35
  for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
    slope = terrain_generator.sub_terrains[name]
    assert isinstance(slope, HfPyramidSlopedTerrainCfg)
    slope.slope_range = (0.0, 0.6)
  cfg.scene.terrain.terrain_generator = terrain_generator
  if not play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.rel_heading_envs = 0.1
    twist_cmd.rel_forward_envs = 0.1
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.3, 0.6),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.15, 0.15),
      },
      {
        "step": 1200 * 24,
        "lin_vel_x": (-0.6, 0.8),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.3, 0.3),
      },
      {
        "step": 3600 * 24,
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-0.7, 0.7),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 5600 * 24,
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-0.5, 0.5),
      },
    ]

  history = deepcopy(actor)
  history.history_length = 5
  cfg.observations["cts_history"] = history
  cfg.observations["cts_student"] = deepcopy(actor)
  cfg.observations["cts_student"].terms = {
    "mask": ObservationTermCfg(func=student_group_mask, params={"student_only": play})
  }
  cfg.observations["cts_student"].enable_corruption = False

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = deepcopy(robot_cfg.articulation)
  for actuator in robot_cfg.articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    actuator.stiffness = 20.0
    actuator.damping = 0.5
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = {
    ".*_hip_joint": 0.125,
    ".*_thigh_joint": 0.25,
    ".*_calf_joint": 0.25,
  }
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  if not play:
    cfg.events["cts_link_inertia"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "alpha_range": (0.5 * math.log(0.9), 0.5 * math.log(1.1)),
      },
    )
    cfg.events["cts_payload"] = EventTermCfg(
      mode="startup",
      func=randomize_trunk_payload,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "mass_range": (-0.5, 2.0),
      },
    )
    # Pseudo-inertia writes body_ipos; apply the COM offset afterwards.
    base_com = cfg.events.pop("base_com")
    base_com.params["ranges"] = {
      0: (-0.05, 0.05),
      1: (-0.035, 0.035),
      2: (-0.035, 0.035),
    }
    cfg.events["base_com"] = base_com
    cfg.events["foot_friction_slide"].params["ranges"] = (0.3, 1.5)
    cfg.events["cts_pd_gains"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    )
    cfg.events["cts_effort_limits"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.8, 1.2),
      },
    )
    for actuator in robot_cfg.articulation.actuators:
      actuator.delay_max_lag = 4
  return cfg


def unitree_go1_mob_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-ground Go1 with behavior-conditioned locomotion commands."""
  cfg = unitree_go1_flat_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 1024
  limb_geoms = tuple(
    f"{leg}_{part}_collision{i}"
    for leg in ("FR", "FL", "RR", "RL")
    for part, indices in (("thigh", (1, 2, 3)), ("calf", (1, 2)))
    for i in indices
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="limb_ground_contact",
      primary=ContactMatch(mode="geom", pattern=limb_geoms, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("force",),
      reduce="none",
      num_slots=1,
    ),
  )

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = deepcopy(robot_cfg.articulation)
  for actuator in robot_cfg.articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    actuator.stiffness = 20.0
    actuator.damping = 0.5
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = {
    ".*_hip_joint": 0.125,
    ".*_thigh_joint": 0.25,
    ".*_calf_joint": 0.25,
  }

  if not play:
    cfg.events["body_inertia_dr"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "alpha_range": (-0.05, 0.05),
      },
    )
    cfg.events["pd_gains_dr"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    )
    cfg.events["joint_friction_dr"] = EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "abs",
        "ranges": (0.0, 0.05),
      },
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
    cfg.events["push_robot"] = EventTermCfg(
      mode="interval",
      func=mdp.push_by_setting_velocity,
      interval_range_s=(4.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "yaw": (-0.3, 0.3),
        },
      },
    )

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.ranges.lin_vel_x = (-1.0, 1.0)
  twist.ranges.lin_vel_y = (-0.6, 0.6)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  if not play:
    cfg.commands["twist"] = GridAdaptiveVelocityCommandCfg(
      **{field.name: deepcopy(getattr(twist, field.name)) for field in fields(twist)}
    )
  cfg.commands["behavior"] = BehaviorCommandCfg(
    resampling_time_range=(1e8, 1e8) if play else (3.0, 8.0),
  )

  actor = cfg.observations["actor"]
  actor.terms.pop("base_lin_vel")
  actor.terms["behavior"] = ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "behavior"}
  )
  actor.terms["clock"] = ObservationTermCfg(
    func=phase_clock, params={"command_name": "behavior"}
  )
  actor.history_length = 30

  critic = cfg.observations["critic"]
  critic.terms["behavior"] = ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "behavior"}
  )
  critic.terms["clock"] = ObservationTermCfg(
    func=phase_clock, params={"command_name": "behavior"}
  )

  cfg.rewards = {
    "mob": RewardTermCfg(
      func=mob_reward,
      weight=1.0,
      params={
        "command_name": "behavior",
        "velocity_command_name": "twist",
        "contact_sensor_name": "feet_ground_contact",
        "limb_contact_sensor_name": "limb_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("FR", "FL", "RR", "RL"), preserve_order=True
        ),
      },
    )
  }
  cfg.metrics["body_height_error"] = MetricsTermCfg(
    func=body_height_error, params={"command_name": "behavior"}
  )
  cfg.curriculum.pop("command_vel", None)
  return cfg


def unitree_go1_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat forward-locomotion task using MoB Trot clips only as AMP references."""
  cfg = unitree_go1_mob_env_cfg(play=play)
  # AMP learns Trot style from motion data, not from MoB behavior commands.
  cfg.commands.pop("behavior")
  actor = cfg.observations["actor"]
  actor.terms.pop("behavior")
  actor.terms.pop("clock")
  actor.history_length = 1
  critic = cfg.observations["critic"]
  critic.terms.pop("behavior")
  critic.terms.pop("clock")
  cfg.metrics.pop("body_height_error")
  # The reference clips only contain positive forward speeds.
  twist = unitree_go1_mob_env_cfg(play=True).commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.2, 0.8)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.resampling_time_range = (4.0, 8.0)
  cfg.commands["twist"] = twist
  cfg.curriculum.pop("command_vel", None)

  flat_rewards = unitree_go1_flat_env_cfg(play=play).rewards
  cfg.rewards = {
    name: deepcopy(flat_rewards[name])
    for name in (
      "track_linear_velocity",
      "upright",
      "dof_pos_limits",
      "action_rate_l2",
      "foot_slip",
    )
  }
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["upright"].weight = 0.2
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["foot_slip"].weight = -0.05
  return cfg
