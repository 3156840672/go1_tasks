"""Unitree DOG1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

DOG1_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "dog1" / "xmls" / "dog.xml"
assert DOG1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DOG1_XML))


##
# Actuator config.
##

# Rotor inertia.
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/DOG1_description/urdf/DOG1.urdf#L515
# Extracted Ixx (rotation along x-axis).
ROTOR_INERTIA = 0.000111842

# Gearbox.
HIP_GEAR_RATIO = 1
KNEE_GEAR_RATIO = 1

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=21.0,
  effort_limit=33.5,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=21.0,
  effort_limit=33.5,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
# DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

# STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
# DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_HIP = 40.0
STIFFNESS_KNEE = 40.0

DAMPING_HIP = 1.0
DAMPING_KNEE = 1.0

DOG1_HIP_ACTUATOR_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
DOG1_KNEE_ACTUATOR_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.378),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions.
# Foot collisions are given custom condim, friction.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # Harden all collision geoms.
  solref=(0.01, 1),
  # Configure feet colliders. Other colliders are frictionless (condim=1).
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

DOG1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DOG1_HIP_ACTUATOR_CFG,
    DOG1_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dog1_robot_cfg() -> EntityCfg:
  """Get a fresh DOG1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DOG1_ARTICULATION,
  )


DOG1_ACTION_SCALE: dict[str, float] = {}
for a in DOG1_ARTICULATION.actuators:
  assert isinstance(a, IdealPdActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    DOG1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_dog1_robot_cfg())

  viewer.launch(robot.spec.compile())
