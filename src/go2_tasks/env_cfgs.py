"""Go2 variants of the shared quadruped locomotion tasks."""

from __future__ import annotations

import math
from dataclasses import fields
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import pseudo_inertia
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ObjRef, RayCastSensorCfg

from go1_tasks.cts import linear_velocity_error_l2, track_yaw_velocity
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
from .him import (
  COMMAND_DIM,
  HISTORY_STEPS,
  PRIVILEGED_DIM,
  PROPRIO_DIM,
)
from .him_commands import HimVelocityCommandCfg
from .paper_amp import paper_privileged_state
from . import paper_rewards
from .paper_curriculum import terrain_levels_from_tracking
from .terrains import CurriculumRandomRoughTerrainCfg


GO2_DR_BODY_NAMES = (BASE_BODY_NAME, ".*_hip", ".*_thigh", ".*_calf")
# Every OCS2 gait category lives below this directory; the AMP reader walks it
# recursively, so Trot, Flying Trot, the walks, and both turns share one dataset.
GO2_AMP_REFERENCE_PATH = "logs/amp/go2_reference"
GO2_PAPER_AMP_REFERENCE_PATH = "logs/amp/go2_paper_reference"

# HIMLoco 的策略输入：3 命令 + 3 角速度 + 3 投影重力 + 12 关节角 + 12 关节速度
# + 12 上一动作。顺序必须与部署端 him_actor_observation() 一致。
HIM_PROPRIO_TERMS = (
  "command",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


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
  # Match the paper's exp(-yaw_error^2 / 0.15).  Using std=0.25 made the
  # reward and its gradient nearly zero before the policy learned to turn.
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.15)
  if "amp_reference_state_init" in cfg.events:
    cfg.events["amp_reference_state_init"].params["reference_path"] = (
      GO2_AMP_REFERENCE_PATH
    )
  return cfg


def unitree_go2_rough_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create blind single-stage Go2 AMP on curriculum rough terrain."""
  cfg = unitree_go2_rough_env_cfg(play=play)
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  noise = generator.sub_terrains["random_rough"]
  generator.sub_terrains["random_rough"] = CurriculumRandomRoughTerrainCfg(
    **{field.name: getattr(noise, field.name) for field in fields(noise)}
  )
  cfg.scene.num_envs = 1 if play else 1024
  if not play:
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.max_init_terrain_level = 0

  # Keep the deployable 45-D actor blind. The critic may use the 187-D terrain
  # scan during training, which is asymmetric PPO rather than a teacher policy.
  actor = cfg.observations["actor"]
  actor.terms.pop("base_lin_vel")
  actor.terms.pop("height_scan")
  actor.history_length = 1

  flat_amp = unitree_go2_amp_env_cfg(play=play)
  cfg.commands["twist"] = deepcopy(flat_amp.commands["twist"])
  if play:
    cfg.curriculum.pop("command_vel", None)
  else:
    # The recorded OCS2 clips actually cover about 0.25 m/s longitudinal,
    # 0.19 m/s lateral, and 0.52 rad/s yaw, regardless of the larger values in
    # their filenames.  Begin inside that support so the discriminator cannot
    # classify policy samples by speed alone, then expand to the deployable
    # envelope after the gait has been learned.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.3, 0.3)
    twist.ranges.lin_vel_y = (-0.2, 0.2)
    twist.ranges.ang_vel_z = (-0.55, 0.55)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.3, 0.3),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.55, 0.55),
      },
      {
        "step": 3500 * 24,
        "lin_vel_x": (-0.6, 0.6),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.75, 0.75),
      },
      {
        "step": 6000 * 24,
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-0.6, 0.6),
        "ang_vel_z": (-1.0, 1.0),
      },
    ]

  # Use the sparse AMP task objective, while retaining rough-terrain-relative
  # upright/slip terms and safety penalties for non-foot contacts.
  rough_upright = deepcopy(cfg.rewards["upright"])
  rough_upright.weight = flat_amp.rewards["upright"].weight
  rough_slip = deepcopy(cfg.rewards["foot_slip"])
  rough_slip.weight = flat_amp.rewards["foot_slip"].weight
  safety_rewards = {
    name: deepcopy(cfg.rewards[name])
    for name in ("self_collisions", "shank_collision", "trunk_head_collision")
  }
  cfg.rewards = deepcopy(flat_amp.rewards)
  cfg.rewards["upright"] = rough_upright
  cfg.rewards["foot_slip"] = rough_slip
  cfg.rewards.update(safety_rewards)

  # Rough already supplies encoder/COM/friction randomization. Copy the extra
  # AMP dynamics randomization, gentler periodic push, and reference reset.
  for name in (
    "body_inertia_dr",
    "pd_gains_dr",
    "joint_friction_dr",
    "push_robot",
    "amp_reference_state_init",
  ):
    if name in flat_amp.events:
      cfg.events[name] = deepcopy(flat_amp.events[name])
  if not play:
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  return cfg


def unitree_go2_him_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough-terrain Go2 task for Hybrid Internal Model (HIMLoco).

  The deployable actor only sees a ``HISTORY_STEPS``-step history of the 45-D
  proprioceptive vector.  The critic is the paper's single-frame privileged
  vector ``[proprio(45), base_lin_vel(3), base_lin_acc(3), height_scan(187)]``:
  the estimator reads the privileged velocity from indices 45:48 and the
  successor state from indices 3:48, so the first 45 entries must keep exactly
  the actor's term order.  mjlab has no "external force" observation, so the
  paper's 3-D disturbance block carries the simulated base linear
  acceleration, which reacts to pushes, impacts and terrain variation.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 1024

  if not play:
    # The inherited sampler almost never produces exactly zero vx/vy with a
    # nonzero yaw command. Reserve worlds for in-place turns in both directions.
    twist = cfg.commands["twist"]
    cfg.commands["twist"] = HimVelocityCommandCfg(
      **{field.name: deepcopy(getattr(twist, field.name)) for field in fields(twist)}
    )

  source_actor = cfg.observations["actor"].terms
  source_critic = cfg.observations["critic"].terms
  proprio = {name: deepcopy(source_actor[name]) for name in HIM_PROPRIO_TERMS}
  # HIMLoco scales these observations before appending them to history.
  proprio["command"].scale = (2.0, 2.0, 0.25)
  proprio["base_ang_vel"].scale = 0.25
  proprio["joint_vel"].scale = 0.05
  privileged = {
    **{name: deepcopy(proprio[name]) for name in HIM_PROPRIO_TERMS},
    "base_lin_vel": deepcopy(source_critic["base_lin_vel"]),
    "base_lin_acc": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_acc"},
    ),
    "height_scan": deepcopy(source_critic["height_scan"]),
  }
  privileged["base_lin_vel"].scale = 2.0

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=proprio,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=HISTORY_STEPS,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=privileged,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # The paper rewards yaw tracking and penalizes roll/pitch rates separately.
  # mjlab's default couples all three rates inside one exponential, so rough
  # terrain roll/pitch motion suppresses the yaw signal even when yaw tracks.
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_yaw_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )
  cfg.rewards["body_ang_vel"].weight = -0.2
  # Scale the paper's -0.01 action-rate cost by the same factor of four as
  # linear tracking. The inherited -0.2 cost dominated the yaw reward.
  cfg.rewards["action_rate_l2"].weight = -0.04
  # exp(-e^2/sigma^2) is saturated while the robot stands (e ~ 0.7 m/s), so it
  # cannot pull the policy out of that basin.  The dense L2 cost keeps a
  # gradient at large errors; CTS uses the same term on the same Go2 commands.
  cfg.rewards["linear_velocity_error_l2"] = RewardTermCfg(
    func=linear_velocity_error_l2,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  return cfg


def unitree_go2_paper_amp_env_cfg(
  play: bool = False, *, baseline: bool = False
) -> ManagerBasedRlEnvCfg:
  """Rough AMP teacher environment with the paper's 30/187 split inputs.

  ``baseline`` keeps the earlier, proven reward and command-curriculum setup
  (narrow commands expanding over training, no heading control, flat-AMP
  regularizers) while retaining the paper teacher observation groups. It is a
  control experiment for isolating AMP and reward changes from the paper's
  wide fixed-command envelope.
  """
  cfg = unitree_go2_rough_amp_env_cfg(play=play)
  # Start episodes from the same motion distribution used by the discriminator.
  if "amp_reference_state_init" in cfg.events:
    cfg.events["amp_reference_state_init"].params["reference_path"] = (
      GO2_PAPER_AMP_REFERENCE_PATH
    )
  if not baseline:
    # Table I terms use physical quantities, not normalized policy actions.
    safety = {name: deepcopy(cfg.rewards[name]) for name in
              ("self_collisions", "shank_collision", "trunk_head_collision", "dof_pos_limits")}
    cfg.rewards = {
      "track_linear_velocity": RewardTermCfg(func=paper_rewards.planar_tracking, weight=1.,
        params={"command_name":"twist", "std":math.sqrt(.15)}),
      "track_angular_velocity": RewardTermCfg(func=paper_rewards.yaw_tracking, weight=.5,
        params={"command_name":"twist", "std":math.sqrt(.15)}),
      # The per-step four-foot stride term has a -2 standing baseline.
      # Offset it for alive episodes so shortening an episode is not rewarded.
      "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=2.),
      "failure": RewardTermCfg(func=envs_mdp.is_terminated, weight=-2.),
      "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-4),
      "joint_acceleration": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
      "joint_position_change": RewardTermCfg(func=paper_rewards.joint_position_change, weight=-.1),
      "stride_duration": RewardTermCfg(func=paper_rewards.stride_duration, weight=1.,
        params={"command_name":"twist", "sensor_name":"feet_ground_contact"}),
      **safety,
    }
  # The deployable observation remains exactly the existing 45-D AMP vector.
  cfg.observations["privileged"] = ObservationGroupCfg(
    terms={
      "state": ObservationTermCfg(func=paper_privileged_state),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["terrain"] = ObservationGroupCfg(
    terms={
      "height_scan": deepcopy(cfg.observations["critic"].terms["height_scan"]),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  if not play:
    # Wu et al. advance terrain difficulty from locomotion performance. The
    # generic mjlab curriculum uses displacement, which penalizes turning and
    # lateral commands and can advance after poor velocity tracking.
    cfg.curriculum["terrain_levels"].func = terrain_levels_from_tracking
    cfg.curriculum["terrain_levels"].params = {
      "command_name": "twist",
      "reward_name": "track_linear_velocity",
      "promote_threshold": 0.85,
      "demote_threshold": 0.50,
      "minimum_command_speed": 0.10,
    }
    if not baseline:
      # Isolate terrain learning from time-driven command expansion.
      cfg.curriculum.pop("command_vel", None)
      twist = cfg.commands["twist"]
      twist.ranges.lin_vel_x = (-1.0, 1.0)
      twist.ranges.lin_vel_y = (-0.6, 0.6)
      twist.ranges.ang_vel_z = (-1.0, 1.0)
      twist.heading_command = True
      twist.ranges.heading = (-math.pi, math.pi)
      twist.rel_heading_envs = 1.0
      twist.resampling_time_range = (20.0, 20.0)
  return cfg


def unitree_go2_paper_amp_baseline_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Paper teacher observations with the earlier narrow-command baseline."""
  return unitree_go2_paper_amp_env_cfg(play=play, baseline=True)


__all__ = [
  "GO2_AMP_REFERENCE_PATH",
  "HIM_PROPRIO_TERMS",
  "GO2_PAPER_AMP_REFERENCE_PATH",
  "unitree_go2_amp_env_cfg",
  "unitree_go2_cts_env_cfg",
  "unitree_go2_flat_env_cfg",
  "unitree_go2_him_env_cfg",
  "unitree_go2_mob_env_cfg",
  "unitree_go2_paper_amp_env_cfg",
  "unitree_go2_paper_amp_baseline_env_cfg",
  "unitree_go2_rough_amp_env_cfg",
  "unitree_go2_rough_env_cfg",
]
