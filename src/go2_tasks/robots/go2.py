"""Unitree Go2 MJCF and actuator configuration for mjlab."""

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_XML = Path(__file__).parents[1] / "assets" / "go2" / "go2.xml"

FOOT_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint"
  for leg in FOOT_NAMES
  for joint in ("hip", "thigh", "calf")
)
BASE_BODY_NAME = "base_link"
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in FOOT_NAMES)
THIGH_GEOM_NAMES = tuple(f"{leg}_thigh_collision" for leg in FOOT_NAMES)
CALF_GEOM_NAMES = tuple(
  f"{leg}_calf_collision{index}" for leg in FOOT_NAMES for index in (1, 2)
)
BASE_GEOM_NAMES = tuple(f"base_collision{index}" for index in (1, 2, 3))


def get_spec() -> mujoco.MjSpec:
  """Load Go2 and add stable semantic names needed by mjlab managers."""
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.delete(spec.key("home"))
  # Scene-level solver settings come from ManagerBasedRlEnvCfg.
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = 1.0
  # mjlab creates position actuators from the explicit configs below. Remove
  # source MJCF motors to avoid a second, parallel actuator set.
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  spec.sensor("imu_gyro").name = "imu_ang_vel"
  spec.sensor("imu_acc").name = "imu_lin_acc"
  linear_velocity = spec.sensor("frame_vel")
  linear_velocity.name = "imu_lin_vel"
  linear_velocity.type = mujoco.mjtSensor.mjSENS_VELOCIMETER

  base = spec.body(BASE_BODY_NAME)
  base_collisions = [geom for geom in base.geoms if geom.group == 3]
  for index, geom in enumerate(base_collisions, start=1):
    geom.name = f"base_collision{index}"
    geom.margin = 0.0

  for leg in ("FL", "FR", "RL", "RR"):
    hip_collisions = [g for g in spec.body(f"{leg}_hip").geoms if g.group == 3]
    thigh_collisions = [
      g for g in spec.body(f"{leg}_thigh").geoms if g.group == 3
    ]
    calf = spec.body(f"{leg}_calf")
    calf_collisions = [g for g in calf.geoms if g.group == 3]
    hip_collisions[0].name = f"{leg}_hip_collision"
    thigh_collisions[0].name = f"{leg}_thigh_collision"
    hip_collisions[0].margin = 0.0
    thigh_collisions[0].margin = 0.0
    for index, geom in enumerate(calf_collisions[:-1], start=1):
      geom.name = f"{leg}_calf_collision{index}"
      geom.margin = 0.0
    calf_collisions[-1].name = f"{leg}_foot_collision"
    calf_collisions[-1].margin = 0.0
    calf.add_site(name=leg, pos=[-0.002, 0.0, -0.213], size=[0.005])
  return spec


# Match robot_lab's UNITREE_GO2_CFG DC motor training profile. The source MJCF
# keeps its own physical joint ranges, inertia, damping and armature.
GO2_MOTOR_ACTUATOR_CFG = DcMotorActuatorCfg(
  target_names_expr=(".*_(hip|thigh|calf)_joint",),
  stiffness=25.0,
  damping=0.5,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_MOTOR_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)

GO2_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  solref=(0.01, 1),
  condim={r".*_foot_collision": 6, r".*_collision.*": 1},
  priority={r".*_foot_collision": 1},
  friction={r".*_foot_collision": (1.0, 5e-3, 5e-4)},
)

GO2_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.38),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

GO2_ACTION_SCALE = {
  ".*_hip_joint": 0.125,
  ".*_thigh_joint": 0.25,
  ".*_calf_joint": 0.25,
}


def get_go2_robot_cfg() -> EntityCfg:
  """Return a fresh Unitree Go2 entity configuration."""
  return EntityCfg(
    init_state=GO2_INIT_STATE,
    collisions=(GO2_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


__all__ = [
  "BASE_BODY_NAME",
  "BASE_GEOM_NAMES",
  "CALF_GEOM_NAMES",
  "FOOT_GEOM_NAMES",
  "FOOT_NAMES",
  "GO2_ACTION_SCALE",
  "GO2_MOTOR_ACTUATOR_CFG",
  "JOINT_NAMES",
  "THIGH_GEOM_NAMES",
  "get_go2_robot_cfg",
]
