"""Dog2 velocity environment configurations.

Sim-to-real setup: the actor policy only sees IMU (gyro, accelerometer,
velocimeter) and motor feedback (position, velocity). The critic (and
rewards) have access to privileged simulation sensors (terrain height,
contacts) for better training — these are NOT needed at deployment.
"""

import math
from typing import Literal

from mjlab.asset_zoo.robots import (
  DOG2_ACTION_SCALE,
  get_dog2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def dog2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Dog2 rough terrain velocity configuration (sim-to-real).

  Actor: IMU + proprioception only (deployable on real robot).
  Critic + rewards: privileged terrain/contact sensors for better training.
  """
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = None  # Auto-size for large batch.

  cfg.scene.entities = {"robot": get_dog2_robot_cfg()}

  # Wire terrain sensors to Dog2 body/sites.
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  # Remove terrain_scan (not needed: critic height_scan already removed).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  # Privileged contact sensors (critic + rewards only, not actor).
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
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_geom_names = tuple(f"{leg}_thigh_collision" for leg in foot_names)
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
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2, 3)
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
  trunk_ground_cfg = ContactSensorCfg(
    name="base_link_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("base_front_collision", "base_rear_collision"),
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
    trunk_ground_cfg,
  )

  ##
  # Actor observations: IMU + proprioception only (deployable).
  ##

  del cfg.observations["actor"].terms["height_scan"]
  # Real robot: no reliable linear velocity from IMU alone.
  del cfg.observations["actor"].terms["base_lin_vel"]

  ##
  # Critic observations: + privileged terrain/contact info (training only).
  ##

  del cfg.observations["critic"].terms["height_scan"]

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DOG2_ACTION_SCALE

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Per-axis foot friction domain randomization.
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
      "ranges": (5e-3, 3e-2),
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
      "ranges": (1e-3, 8e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Stronger push disturbance for robustness.
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (-0.8, 0.8),
    "roll": (-1.0, 1.0),
    "pitch": (-1.0, 1.0),
    "yaw": (-1.5, 1.5),
  }

  # Additional domain randomization for robustness.
  cfg.events["randomize_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.85, 1.15),
    },
  )
  cfg.events["randomize_joint_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.dof_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": (0.0, 0.1),
    },
  )
  cfg.events["randomize_pd_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.8, 1.2),
    },
  )

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

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["foot_clearance"].weight = 0.3
  # Improve angular velocity tracking.
  cfg.rewards["track_angular_velocity"].weight = 6.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.15)
  # Dog2 has lower stance than Go1; push feet to lift higher.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.15
  cfg.rewards["foot_swing_height"].weight = -0.02
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.3
  cfg.rewards["action_rate_l2"].weight = -0.15

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
    params={"sensor_name": trunk_ground_cfg.name},
  )

  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  if play:
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


def dog2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Dog2 flat terrain velocity configuration (sim-to-real)."""
  cfg = dog2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove sensors and privileged obs not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "base_link_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  for key in ("self_collisions", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
