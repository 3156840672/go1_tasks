"""ArmV5 constants.

A 5-DOF robotic arm driven by bus servos.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ARMV5_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "armv5" / "xmls" / "armv5.xml"
)
assert ARMV5_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(ARMV5_XML))


##
# Actuator config.
##

# Estimated effective inertias at each joint [kg⋅m²].
# These represent the total inertia of all downstream links reflected to the joint.
EFFECTIVE_INERTIAS = {
  "joint1": 0.1,
  "joint2": 0.08,
  "joint3": 0.05,
  "joint4": 0.01,
  "joint5": 0.002,
}

# Bus servo: generic position-controlled servo with 5 Nm peak torque.
ARMATURE = 0.001
BUS_SERVO = ElectricActuator(
  reflected_inertia=ARMATURE,
  velocity_limit=5.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 2 * 2.0 * math.pi  # 2 Hz bandwidth
DAMPING_RATIO = 2.0

# All joints use the same bus servo type.
_ARM_JOINTS: dict[str, ElectricActuator] = {
  "joint1": BUS_SERVO,
  "joint2": BUS_SERVO,
  "joint3": BUS_SERVO,
  "joint4": BUS_SERVO,
  "joint5": BUS_SERVO,
}
ARM_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=EFFECTIVE_INERTIAS[name] * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS[name] * NATURAL_FREQ,
    effort_limit=motor.effort_limit,
    armature=motor.reflected_inertia,
  )
  for name, motor in _ARM_JOINTS.items()
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Use all geoms for self-collision.
SELF_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim=3,
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=ARM_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_armv5_robot_cfg() -> EntityCfg:
  """Get a fresh ArmV5 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


ARMV5_ACTION_SCALE: dict[str, float] = {}
_MAX_ACTION_SCALE = 0.5  # rad per step max
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    ARMV5_ACTION_SCALE[n] = min(0.25 * e / s, _MAX_ACTION_SCALE)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_armv5_robot_cfg())

  viewer.launch(robot.spec.compile())
