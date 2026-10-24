"""Dog2 quadruped constants."""

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

DOG2_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "dog2" / "xmls" / "dog2.xml"
assert DOG2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DOG2_XML))


##
# Actuator config.
##

# Dog2 uses direct-drive motors (no gearbox).
# Armature = motor rotor inertia; PD gains are hardcoded (same as dog1).

# Rotor inertia (from Unitree Go1 spec).
ROTOR_INERTIA = 0.000111842

# Gearbox (direct drive).
HIP_GEAR_RATIO = 1
KNEE_GEAR_RATIO = 1

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=15.0,
  effort_limit=35.0,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=15.0,
  effort_limit=35.0,
)

STIFFNESS = 40.0
DAMPING = 1.0

DOG2_HIP_ACTUATOR_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
DOG2_KNEE_ACTUATOR_CFG = IdealPdActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    ".*thigh_joint": 0.7,
    ".*calf_joint": -1.6,
    ".*R_hip_joint": -0.1,
    ".*L_hip_joint": 0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot$"

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

DOG2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DOG2_HIP_ACTUATOR_CFG,
    DOG2_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dog2_robot_cfg() -> EntityCfg:
  """Get a fresh Dog2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DOG2_ARTICULATION,
  )


DOG2_ACTION_SCALE: dict[str, float] = {}
for a in DOG2_ARTICULATION.actuators:
  assert isinstance(a, IdealPdActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    DOG2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_dog2_robot_cfg())

  viewer.launch(robot.spec.compile())
