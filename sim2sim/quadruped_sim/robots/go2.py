"""Go2 scene assembly and policy simulation.

The 45/48-D actors cover the flat, AMP, blind rough, rough-AMP and HIM
checkpoints; older 235-D rough checkpoints need the 187-D terrain scan and are
not supported.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..policies.actor import ACTION_DIM, OBS_DIM, load_standard_policy
from ..policies.cts import CtsStudentPolicy, load_cts_policy
from ..policies.dreamwaq import DreamWaQPolicy, load_dreamwaq_policy
from ..policies.go2 import (
  AMP_OBS_DIM,
  HimActor,
  load_him_policy,
  load_paper_amp_student,
)
from ..scene import (
  EXPORTED_TERRAIN,
  SHADOW_RADIUS,
  VISUAL_MEANSIZE,
  finish_exported_scene,
  fit_view_range,
  follow_shadow_light,
  move_terrain_light_to_mocap,
  shadow_light_mocap,
  spawn_position,
)
from ..terrain import add_terrain

__all__ = [
  "ACTION_SCALE",
  "AMP_OBS_DIM",
  "BASE_HEIGHT",
  "CtsStudentPolicy",
  "COMMAND_LIMITS",
  "DEFAULT",
  "EFFORT",
  "Go2Simulator",
  "JOINTS",
  "KD",
  "KP",
  "OBS_DIM",
  "SATURATION_EFFORT",
  "SHADOW_RADIUS",
  "VELOCITY_LIMIT",
  "VISUAL_MEANSIZE",
  "fit_view_range",
  "make_model",
  "move_terrain_light_to_mocap",
]

#: <repo>/src/go2_tasks/assets/go2/go2.xml, shared with the training task.
GO2_ASSETS = Path(__file__).resolve().parents[3] / "src" / "go2_tasks" / "assets"
GO2_XML = GO2_ASSETS / "go2" / "go2.xml"

JOINTS = tuple(
  f"{leg}_{part}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for part in ("hip", "thigh", "calf")
)
DEFAULT = np.array(
  [value for _ in range(4) for value in (0.0, 0.8, -1.5)], dtype=np.float64
)
ACTION_SCALE = np.array(
  [value for _ in range(4) for value in (0.125, 0.25, 0.25)], dtype=np.float64
)
KP = np.full(ACTION_DIM, 25.0)
KD = np.full(ACTION_DIM, 0.5)
EFFORT = np.full(ACTION_DIM, 23.5)
SATURATION_EFFORT = np.full(ACTION_DIM, 23.5)
VELOCITY_LIMIT = np.full(ACTION_DIM, 30.0)
#: Match the training task's initial base_link height.
BASE_HEIGHT = 0.38
COMMAND_LIMITS = np.array([1.0, 0.6, 1.0], dtype=np.float32)


def make_model(
  terrain: str = "flat",
  terrain_seed: int = 42,
  terrain_type: str | None = None,
  terrain_level: int = 0,
) -> mujoco.MjModel:
  """Load the Go2 MJCF with the deployment actuators and the chosen terrain."""
  tree = ET.parse(GO2_XML)
  root = tree.getroot()
  compiler = root.find("compiler")
  assert compiler is not None
  compiler.set("meshdir", str(GO2_XML.parent / "assets"))
  option = root.find("option")
  if option is None:
    option = ET.SubElement(root, "option")
  option.attrib.update(
    timestep="0.005",
    iterations="10",
    ls_iterations="20",
    ccd_iterations="50",
    impratio="10",
    cone="elliptic",
    integrator="implicitfast",
  )

  # Match the training model: body-frame IMU linear velocity.
  sensor = root.find("sensor")
  assert sensor is not None
  for item in sensor.findall("framelinvel"):
    if item.get("name") == "frame_vel":
      item.tag = "velocimeter"
      item.attrib.pop("objtype", None)
      item.attrib.pop("objname", None)
      item.set("site", "imu")

  # Match the training collision settings: hard contacts, condim-6 feet, and
  # the same foot friction values used by the robot profile.
  for geom in root.iter("geom"):
    geom_class = geom.get("class")
    if geom_class in ("collision", "foot"):
      geom.set("solref", "0.01 1")
    if geom_class == "foot":
      geom.set("condim", "6")
      geom.set("friction", "1 0.005 0.0005")

  actuators = root.find("actuator")
  assert actuators is not None
  for item in list(actuators):
    actuators.remove(item)
  for index, name in enumerate(JOINTS):
    ET.SubElement(
      actuators,
      "motor",
      name=name + "_motor",
      joint=name,
      ctrlrange=f"{-EFFORT[index]} {EFFORT[index]}",
    )

  with tempfile.TemporaryDirectory() as directory:
    add_terrain(
      root, Path(directory), terrain, terrain_seed, terrain_type, terrain_level
    )
    path = Path(directory) / "scene.xml"
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    # The training scene fixes this statistic; without it the terrain size
    # inflates every diagnostic glyph by up to 140x.
    model.stat.meansize = VISUAL_MEANSIZE
    if terrain == EXPORTED_TERRAIN:
      finish_exported_scene(model)
    else:
      fit_view_range(model)
    return model


class Go2Simulator:
  """Go2 with one of the flat/AMP/student/HIM policies attached."""

  def __init__(
    self,
    no_shadows: bool = False,
    policy: Path | None = None,
    terrain: str = "flat",
    terrain_seed: int = 42,
    student: Path | None = None,
    him: Path | None = None,
    cts: Path | None = None,
    terrain_type: str | None = None,
    terrain_level: int = 0,
    terrain_column: str | None = None,
    dreamwaq: Path | None = None,
  ) -> None:
    self.model = make_model(terrain, terrain_seed, terrain_type, terrain_level)
    self.shadows_enabled = not no_shadows
    if not self.shadows_enabled:
      self.model.vis.quality.shadowsize = 0
    self.spawn = spawn_position(
      self.model, terrain, terrain_type, terrain_level, terrain_column
    )
    self.light_mocap_id = shadow_light_mocap(self.model)
    if terrain == EXPORTED_TERRAIN:
      assert self.light_mocap_id >= 0, "exported terrain lost its shadow light"
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINTS
      ]
    )
    self.vadr = np.array(
      [
        self.model.jnt_dofadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINTS
      ]
    )
    self.aid = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor")
        for name in JOINTS
      ]
    )
    self.lin_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "frame_vel"
    )
    self.ang_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
    )
    assert all(
      i >= 0 for i in (*self.qadr, *self.vadr, *self.aid, self.lin_id, self.ang_id)
    )
    sources = [path for path in (policy, student, him, cts, dreamwaq) if path is not None]
    if len(sources) > 1:
      raise ValueError("Use one of --policy, --student, --him, --cts or --dreamwaq")
    self.policy = None
    if policy is not None:
      self.policy = load_standard_policy(
        policy, supported_obs_dims=(AMP_OBS_DIM, OBS_DIM)
      )
    elif student is not None:
      self.policy = load_paper_amp_student(student)
    elif him is not None:
      self.policy = load_him_policy(him)
    elif cts is not None:
      self.policy = load_cts_policy(cts)
    elif dreamwaq is not None:
      self.policy = load_dreamwaq_policy(dreamwaq)
    self.command = np.zeros(3, dtype=np.float32)
    self.active = False
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.target = DEFAULT.copy()
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:7] = [
      self.spawn[0],
      self.spawn[1],
      self.spawn[2] + BASE_HEIGHT,
      1,
      0,
      0,
      0,
    ]
    self.data.qpos[self.qadr] = DEFAULT
    self._follow_shadow_light()
    self.data.ctrl[self.aid] = 0
    self.last_action.fill(0)
    self.target = DEFAULT.copy()
    if self.policy is not None and hasattr(self.policy, "reset"):
      self.policy.reset()
    mujoco.mj_forward(self.model, self.data)

  def _follow_shadow_light(self) -> None:
    follow_shadow_light(self.data, self.light_mocap_id)

  def set_active(self, active: bool) -> None:
    if active and self.policy is None:
      print("No checkpoint loaded; staying in passive stand")
      return
    if isinstance(self.policy, DreamWaQPolicy) and self.active != active:
      self.policy.reset()
    self.active = active
    self.last_action.fill(0)
    if not active:
      print("Passive stand")
    elif isinstance(self.policy, CtsStudentPolicy):
      print("CTS student active")
    elif isinstance(self.policy, DreamWaQPolicy):
      print("DreamWaQ policy active")
    else:
      print("Standard policy active")

  def observation(self) -> np.ndarray:
    if isinstance(self.policy, DreamWaQPolicy):
      # mj_step advances qpos/qvel past the sensor values. mjlab refreshes
      # sensors before recording a frame, so do the same at 50 Hz here.
      mujoco.mj_forward(self.model, self.data)
    lin, ang = self.base_velocity()
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    # Flat uses privileged base linear velocity in its 48-D actor observation.
    # AMP and the blind Rough/CTS actors omit it and keep the same 45-D order.
    obs_dim = getattr(self.policy, "obs_dim", OBS_DIM)
    components = (
      ang,
      gravity,
      self.data.qpos[self.qadr] - DEFAULT,
      self.data.qvel[self.vadr],
      self.last_action,
      self.command,
    )
    if isinstance(self.policy, HimActor):
      # HIM keeps the command first and expects the caller to feed the
      # current step only; the ring buffer lives inside the policy module.
      components = (
        self.command * np.array((2.0, 2.0, 0.25), dtype=np.float32),
        ang * 0.25,
        gravity,
        self.data.qpos[self.qadr] - DEFAULT,
        self.data.qvel[self.vadr] * 0.05,
        self.last_action,
      )
    elif obs_dim == OBS_DIM:
      components = (lin, *components)
    elif obs_dim != AMP_OBS_DIM:
      raise ValueError(f"Unsupported Go2 policy observation dimension: {obs_dim}")
    obs = np.concatenate(components).astype(np.float32)
    assert obs.shape == (obs_dim,)
    return obs

  def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
    """Return body-frame linear and angular velocity sensor values."""
    lin_adr = self.model.sensor_adr[self.lin_id]
    ang_adr = self.model.sensor_adr[self.ang_id]
    return (
      self.data.sensordata[lin_adr : lin_adr + 3].copy(),
      self.data.sensordata[ang_adr : ang_adr + 3].copy(),
    )

  def policy_step(self) -> None:
    if not self.active:
      self.target = DEFAULT.copy()
      self.last_action.fill(0)
      return
    obs = self.observation()
    with torch.inference_mode():
      action = self.policy(torch.from_numpy(obs[None]))[0].numpy()
    if not np.isfinite(action).all():
      raise FloatingPointError("Policy returned non-finite actions")
    self.target = DEFAULT + ACTION_SCALE * action
    self.last_action = action

  def physics_step(self) -> None:
    """PD control plus mjlab's DC-motor torque-speed clipping."""
    vel = self.data.qvel[self.vadr]
    effort = KP * (self.target - self.data.qpos[self.qadr]) - KD * vel
    corner_vel = VELOCITY_LIMIT * (1.0 + EFFORT / SATURATION_EFFORT)
    clipped_vel = np.clip(vel, -corner_vel, corner_vel)
    upper = np.minimum(
      SATURATION_EFFORT * (1.0 - clipped_vel / VELOCITY_LIMIT), EFFORT
    )
    lower = np.maximum(
      SATURATION_EFFORT * (-1.0 - clipped_vel / VELOCITY_LIMIT), -EFFORT
    )
    self.data.ctrl[self.aid] = np.clip(effort, lower, upper)
    self._follow_shadow_light()
    mujoco.mj_step(self.model, self.data)
