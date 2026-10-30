"""Go2 variants of the shared quadruped locomotion tasks."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import pseudo_inertia
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ObjRef, RayCastSensorCfg

from go1_tasks.env_cfgs import (
  unitree_go1_amp_env_cfg,
  unitree_go1_cts_env_cfg,
  unitree_go1_flat_env_cfg,
  unitree_go1_mob_env_cfg,
  unitree_go1_rough_env_cfg,
)

from .robots.go2 import (
  BASE_BODY_NAME,
  BASE_GEOM_NAMES,
  CALF_GEOM_NAMES,
  FOOT_GEOM_NAMES,
  GO2_ACTION_SCALE,
  JOINT_NAMES,
  THIGH_GEOM_NAMES,
  get_go2_robot_cfg,
)


GO2_DR_BODY_NAMES = (BASE_BODY_NAME, ".*_hip", ".*_thigh", ".*_calf")
# Every OCS2 gait category lives below this directory; the AMP reader walks it
# recursively, so Trot, Flying Trot, the walks, and both turns share one dataset.
GO2_AMP_REFERENCE_PATH = "logs/amp/go2_reference"


def _replace_contact_match(sensor, *, primary=None, secondary=None) -> None:
  if primary is not None:
    sensor.primary = primary
  if secondary is not None:
    sensor.secondary = secondary


def _apply_go2_profile(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Replace Go1-specific entity and semantic names with the Go2 profile."""
  cfg.scene.entities["robot"] = get_go2_robot_cfg()
  cfg.viewer.body_name = BASE_BODY_NAME

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY_NAME
    elif sensor.name == "feet_ground_contact":
      _replace_contact_match(
        sensor,
        primary=ContactMatch(
          mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"
        ),
      )
    elif sensor.name == "self_collision":
      match = ContactMatch(mode="subtree", pattern=BASE_BODY_NAME, entity="robot")
      _replace_contact_match(sensor, primary=match, secondary=match)
    elif sensor.name == "thigh_ground_touch":
      _replace_contact_match(
        sensor,
        primary=ContactMatch(
          mode="geom", pattern=THIGH_GEOM_NAMES, entity="robot"
        ),
      )
    elif sensor.name == "shank_ground_touch":
      _replace_contact_match(
        sensor,
        primary=ContactMatch(
          mode="geom", pattern=CALF_GEOM_NAMES, entity="robot"
        ),
      )
    elif sensor.name == "trunk_ground_touch":
      _replace_contact_match(
        sensor,
        primary=ContactMatch(
          mode="geom", pattern=BASE_GEOM_NAMES, entity="robot"
        ),
      )
    elif sensor.name == "limb_ground_contact":
      _replace_contact_match(
        sensor,
        primary=ContactMatch(
          mode="geom",
          pattern=THIGH_GEOM_NAMES + CALF_GEOM_NAMES,
          entity="robot",
        ),
      )

  for name in ("foot_friction_slide", "foot_friction_spin", "foot_friction_roll"):
    if name in cfg.events:
      cfg.events[name].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOM_NAMES
      )
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)
  if "cts_payload" in cfg.events:
    cfg.events["cts_payload"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)

  # Go2 keeps the foot links massless (their mass is folded into the calf
  # inertial). Pseudo-inertia randomization on a zero-mass body is singular and
  # yields NaNs, so randomize only bodies with positive mass.
  for event in cfg.events.values():
    if event.func is pseudo_inertia:
      event.params["asset_cfg"].body_names = GO2_DR_BODY_NAMES

  for name in ("upright", "body_ang_vel"):
    if name in cfg.rewards and "asset_cfg" in cfg.rewards[name].params:
      cfg.rewards[name].params["asset_cfg"].body_names = (BASE_BODY_NAME,)
  if "feet_regulation" in cfg.rewards:
    cfg.rewards["feet_regulation"].params["desired_body_height"] = 0.38
    cfg.rewards["feet_regulation"].params["foot_radius"] = 0.022

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = JOINT_NAMES
  action.preserve_order = True
  action.scale = GO2_ACTION_SCALE

  # Policy I/O follows the Unitree/RobotLab deployment order FR, FL, RR, RL,
  # independent of the body declaration order in the source MJCF.
  for group in cfg.observations.values():
    for term_name in ("joint_pos", "joint_vel"):
      if term_name in group.terms:
        group.terms[term_name].params["asset_cfg"] = SceneEntityCfg(
          "robot", joint_names=JOINT_NAMES, preserve_order=True
        )
  return cfg


def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 rough-terrain velocity task."""
  return _apply_go2_profile(unitree_go1_rough_env_cfg(play=play))


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree Go2 flat-terrain velocity task."""
  return _apply_go2_profile(unitree_go1_flat_env_cfg(play=play))


def unitree_go2_mob_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the behavior-conditioned Go2 locomotion task."""
  return _apply_go2_profile(unitree_go1_mob_env_cfg(play=play))


def unitree_go2_cts_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the concurrent teacher-student Go2 task."""
  cfg = _apply_go2_profile(unitree_go1_cts_env_cfg(play=play))
  if not play:
    # The Go1 config sets this on its actuators, but the Go2 profile replaces
    # the articulation, so re-apply the CTS actuator delay to the Go2 motors.
    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    for actuator in robot_cfg.articulation.actuators:
      actuator.delay_max_lag = 4
  return cfg


def unitree_go2_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Go2 AMP with a conservative first-stage command distribution."""
  cfg = _apply_go2_profile(unitree_go1_amp_env_cfg(play=play))
  # Use the normal deployable velocity envelope from the MoB task. AMP is a
  # style prior and does not require commands to remain inside demonstration
  # velocities; task rewards drive motions beyond the short OCS2 clips.
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (-1.0, 1.0)
  twist.ranges.lin_vel_y = (-0.6, 0.6)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  # Uniform continuous sampling almost never produces an exact zero command.
  # Reserve training environments for strict zero commands so the policy learns
  # to hold still instead of satisfying AMP with an in-place stepping gait.
  twist.rel_standing_envs = 0.0 if play else 0.05
  flat_rewards = unitree_go2_flat_env_cfg(play=play).rewards
  cfg.rewards["track_angular_velocity"] = deepcopy(
    flat_rewards["track_angular_velocity"]
  )
  # The default 0.707 rad/s tolerance lets a non-turning policy earn a large
  # reward throughout [-1, 1]. Use the tighter yaw-tracking tolerance already
  # proven by the CTS task and balance it with the linear tracking reward.
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.25
  if "amp_reference_state_init" in cfg.events:
    cfg.events["amp_reference_state_init"].params["reference_path"] = (
      GO2_AMP_REFERENCE_PATH
    )
  return cfg


__all__ = [
  "GO2_AMP_REFERENCE_PATH",
  "unitree_go2_amp_env_cfg",
  "unitree_go2_cts_env_cfg",
  "unitree_go2_flat_env_cfg",
  "unitree_go2_mob_env_cfg",
  "unitree_go2_rough_env_cfg",
]
