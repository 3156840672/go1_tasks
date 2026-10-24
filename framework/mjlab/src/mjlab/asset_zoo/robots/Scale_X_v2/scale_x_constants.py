"""Scale_X_v2 constants."""

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

SCALE_X_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "Scale_X_v2" / "xmls" / "scale_x.xml"
)
assert SCALE_X_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SCALE_X_XML))


##
# Actuator config.
##

# DM-J4340p (48V) motor specs.
# Rated torque: 9 Nm, Peak torque: 27 Nm.
# Rated speed: 36 rpm (used as the velocity limit).
# Gear ratio: 1:40.
# Rotor inertia not specified; use XML armature (0.01) as the effective
# reflected inertia for stiffness/damping computation.
EFFECTIVE_ARMATURE = 0.01
EFFORT_LIMIT = 20.0  # Configured torque limit (peak is 27 Nm).
VELOCITY_LIMIT = 36.0 * 2.0 * 3.1415926535 / 60.0  # Rated 36 rpm -> rad/s.

SCALE_X_ACTUATOR = ElectricActuator(
  reflected_inertia=EFFECTIVE_ARMATURE,
  velocity_limit=VELOCITY_LIMIT,
  effort_limit=EFFORT_LIMIT,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS = EFFECTIVE_ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * EFFECTIVE_ARMATURE * NATURAL_FREQ

# All 12 leg joints use the same DM-J4340-2EC motor.
SCALE_X_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_pitch_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=SCALE_X_ACTUATOR.effort_limit,
  armature=SCALE_X_ACTUATOR.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.7),
  joint_pos={
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Enable all collisions including self-collisions.
# Foot collisions use condim=3 for proper ground contact;
# other collisions use condim=1 (frictionless).
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

# Enable base/feet vs terrain collisions.
# Feet also collide with each other (contype=1) so foot-foot contacts are
# simulated and the self_collision penalty can discourage foot scissoring.
# The base keeps contype=0: it receives contacts from the feet but does not
# generate them, which avoids unwanted base-vs-base pairs.
BASE_AND_FEET_COLLISION = CollisionCfg(
  geom_names_expr=("base_link_collision", r"^(left|right)_foot_collision$"),
  contype={
    r"^(left|right)_foot_collision$": 1,
    "base_link_collision": 0,
  },
  conaffinity=1,
  condim={r"^(left|right)_foot_collision$": 3, "base_link_collision": 3},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

##
# Final config.
##

SCALE_X_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SCALE_X_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_scale_x_robot_cfg() -> EntityCfg:
  """Get a fresh Scale_X_v2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(BASE_AND_FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=SCALE_X_ARTICULATION,
  )


SCALE_X_ACTION_SCALE: dict[str, float] = {}
for a in SCALE_X_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    SCALE_X_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_scale_x_robot_cfg())

  viewer.launch(robot.spec.compile())
