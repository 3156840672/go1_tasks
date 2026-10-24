"""Scale_X_v2 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  SCALE_X_ACTION_SCALE,
  get_scale_x_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import body as dr_body
from mjlab.envs.mdp.dr import geom as dr_geom
from mjlab.envs.mdp.dr import joint as dr_joint
from mjlab.managers.event_manager import EventTermCfg
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


def scale_x_v2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Scale_X_v2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_scale_x_robot_cfg()}

  # Set raycast sensor frame to base_link (no pelvis on Scale_X_v2).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # Remove linear velocity from the actor observation: the real robot has
  # no velocity estimator. The critic keeps it as privileged information.
  cfg.observations["actor"].terms.pop("base_lin_vel", None)

  # Foot sites for foot height sensor and rewards.
  site_names = ("right_foot", "left_foot")
  geom_names = ("right_foot_collision", "left_foot_collision")

  # Wire foot height scan to foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll|right_ankle_roll)$",
      entity="robot",
    ),
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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Scale velocity curriculum to half-length training (15k iterations).
  # Final stage caps linear speed at 1 m/s (both directions and lateral) and
  # turning at 1 rad/s per deployment requirements.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.3, 0.3),
    },
    {
      "step": 2500 * 24,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.0, 1.0),
    },
  ]

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SCALE_X_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.9
  # Cap play-mode command ranges to match the training curriculum.
  twist_cmd.ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    heading=twist_cmd.ranges.heading,
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # ── Enhanced robustness: wider DR ranges ──

  # Push robot: more aggressive disturbances.
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (-0.6, 0.6),
    "roll": (-1.0, 1.0),
    "pitch": (-1.0, 1.0),
    "yaw": (-1.5, 1.5),
  }

  # Foot friction: standard range (matches common sim-to-real practice).
  # A lower bound of 0.1 (ice-like) previously encouraged slip-heavy gaits
  # with reduced foot clearance, which aggravated foot-foot collisions.
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.25)

  # Encoder bias: enlarged.
  cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)

  # Base COM offset: enlarged.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }

  # Joint position initialization noise.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.2, 0.2)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.3, 0.3)

  # Body mass + inertia randomization (uniform density scaling ±20%).
  cfg.events["body_mass_dr"] = EventTermCfg(
    mode="startup",
    func=dr_body.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "alpha_range": (-0.2, 0.2),
    },
  )

  # Joint stiffness & damping randomization (±20%).
  cfg.events["joint_stiffness_dr"] = EventTermCfg(
    mode="startup",
    func=dr_joint.joint_stiffness,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.8, 1.2),
      "operation": "scale",
    },
  )
  cfg.events["joint_damping_dr"] = EventTermCfg(
    mode="startup",
    func=dr_joint.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.8, 1.2),
      "operation": "scale",
    },
  )

  # Foot collision geometry randomization: simulates manufacturing/assembly
  # tolerances on the feet so the policy is robust to slightly misaligned
  # collision geometry. geom_size is NOT used because the foot collision
  # geoms are meshes (mjlab's geom_size only supports primitive types).
  # The foot sites are untouched, so foot_height/clearance observations and
  # rewards stay consistent while only the collision volume moves.
  cfg.events["foot_collision_pos_dr"] = EventTermCfg(
    mode="startup",
    func=dr_geom.geom_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "ranges": {
        0: (-0.01, 0.01),
        1: (-0.01, 0.01),
        2: (-0.005, 0.005),
      },
      "operation": "add",
    },
  )
  cfg.events["foot_collision_quat_dr"] = EventTermCfg(
    mode="startup",
    func=dr_geom.geom_quat,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "roll_range": (-0.03, 0.03),
      "pitch_range": (-0.03, 0.03),
      "yaw_range": (-0.03, 0.03),
    },
  )

  # Rationale for std values:
  # Scale_X_v2 has only leg joints (12 DOF), no upper body.
  # - Knees/hip_pitch get the loosest std to allow natural leg bending.
  # - Hip roll/yaw stay tight: 0.2 rad hip roll adduction already brings the
  #   feet within ~7 cm, so running stds are capped below that.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # Running values are ~1.5-2x walking values.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.1,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.12,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Stronger anti-oscillation penalties: the torso was swaying heavily
  # in sim2sim (unconverged policy + weak damping terms).
  cfg.rewards["body_ang_vel"].weight = -0.1
  cfg.rewards["angular_momentum"].weight = -0.05
  # Encourage swing phase: feet were barely lifting off the ground.
  cfg.rewards["air_time"].weight = 1.0
  # feet_clearance returns a deviation cost (|h - target| * foot_xy_speed).
  # A POSITIVE weight rewards deviating from the 0.1 m target, so a fast,
  # low foot scores high — this pulled the swing peak back down. Use a
  # NEGATIVE weight so the swing phase (fast feet) is pushed toward the
  # 0.1 m clearance target instead.
  cfg.rewards["foot_clearance"].weight = -0.3
  # Pull swing-peak height toward the 0.1 m target at landing (doubled).
  cfg.rewards["foot_swing_height"].weight = -2.0

  # Strong penalty with a low force threshold so even light foot-foot
  # contact is discouraged (foot scissoring caused sim2sim falls).
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 2.0},
  )

  # Discourage asymmetric left/right gaits.
  cfg.rewards["leg_symmetry"] = RewardTermCfg(
    func=mdp.leg_symmetry_penalty,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  # Observation delay DR: models real-world control-loop latency
  # (CAN bus + controller). 1 control step = 20 ms at 50 Hz.
  # The actor observes delayed sensor values; the command and last action
  # stay current (they originate on the controller side).
  for term_name in (
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "height_scan",
  ):
    term = cfg.observations["actor"].terms.get(term_name)
    if term is not None:
      term.delay_min_lag = 0
      term.delay_max_lag = 1
      term.delay_hold_prob = 0.5

  # Apply play mode overrides.
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


def scale_x_v2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Scale_X_v2 flat terrain velocity configuration."""
  cfg = scale_x_v2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Remove terrain curriculum (no terrain generator on flat).
  cfg.curriculum.pop("terrain_levels", None)

  # Remove terrain grid scan (187-dim, useless on flat ground).
  # Keep foot_height_scan and its dependent critic observations + rewards.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  return cfg
