"""Go1 scene assembly and the RMA / flat-velocity simulator."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..policies.actor import ACTION_DIM, OBS_DIM, load_standard_policy
from ..policies.cts import CTS_OBS_DIM, CtsStudentPolicy, load_cts_policy
from ..policies.rma import HISTORY_LENGTH, load_policy
from ..scene import (
  EXPORTED_TERRAIN,
  finish_exported_scene,
  fit_view_range,
  follow_shadow_light,
  shadow_light_mocap,
  spawn_position,
)
from ..terrain import add_terrain

#: sim2sim/ — holds the Go1 MJCF and its meshes.
ROOT = Path(__file__).resolve().parents[2]
JOINTS = tuple(
  f"{leg}_{part}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for part in ("hip", "thigh", "calf")
)
DEFAULT = np.array(
  [
    v
    for leg in ("FR", "FL", "RR", "RL")
    for v in ((0.1 if leg.endswith("R") else -0.1), 0.9, -1.8)
  ],
  dtype=np.float64,
)
HIP_ARMATURE = 0.000111842 * 6**2
KNEE_ARMATURE = 0.000111842 * 9**2
HIP_KP = HIP_ARMATURE * (20 * np.pi) ** 2
KNEE_KP = KNEE_ARMATURE * (20 * np.pi) ** 2
HIP_KD = 4 * HIP_ARMATURE * 20 * np.pi
KNEE_KD = 4 * KNEE_ARMATURE * 20 * np.pi
KP = np.array([KNEE_KP if j.endswith("calf_joint") else HIP_KP for j in JOINTS])
KD = np.array([KNEE_KD if j.endswith("calf_joint") else HIP_KD for j in JOINTS])
EFFORT = np.array([35.55 if j.endswith("calf_joint") else 23.7 for j in JOINTS])
ACTION_SCALE = 0.25 * EFFORT / KP
#: Standing base height of the stock Go1 MJCF.
BASE_HEIGHT = 0.278


def make_model(
  terrain: str = "rough",
  terrain_seed: int = 42,
  terrain_type: str | None = None,
  terrain_level: int = 0,
  *,
  mob: bool = False,
) -> mujoco.MjModel:
  """Load the Go1 MJCF, add the terrain and the deployment actuators."""
  tree = ET.parse(ROOT / "model" / "go1.xml")
  root = tree.getroot()
  root.find("compiler").set("meshdir", str(ROOT / "model" / "assets"))
  option = dict(timestep="0.005", iterations="10", ls_iterations="20")
  if mob:
    option.update(
      ccd_iterations="50", impratio="1", cone="pyramidal", integrator="implicitfast"
    )
  else:
    option.update(ccd_iterations="500", impratio="10", cone="elliptic")
  ET.SubElement(root, "option", **option)
  if mob:
    # MoB was trained with soft single-point contacts and sticky feet.
    for geom in root.findall(".//geom"):
      name = geom.get("name", "")
      if "_collision" not in name:
        continue
      geom.set("solref", "0.01 1")
      if name.endswith("_foot_collision"):
        geom.set("condim", "6")
        geom.set("priority", "1")
        geom.set("friction", "1 0.005 0.0005")
      else:
        geom.set("condim", "1")
  actuators = ET.SubElement(root, "actuator")
  for index, name in enumerate(JOINTS):
    joint = root.find(f".//joint[@name='{name}']")
    if joint is None:
      raise ValueError(f"Missing joint {name}")
    joint.set(
      "armature", str(KNEE_ARMATURE if name.endswith("calf_joint") else HIP_ARMATURE)
    )
    kp = 20.0 if mob else KP[index]
    kd = 0.5 if mob else KD[index]
    ET.SubElement(
      actuators,
      "general",
      name=name + "_motor",
      joint=name,
      gainprm=str(kp),
      biastype="affine",
      biasprm=f"0 {-kp} {-kd}",
      forcelimited="true",
      forcerange=f"{-EFFORT[index]} {EFFORT[index]}",
    )
  with tempfile.TemporaryDirectory() as directory:
    add_terrain(
      root, Path(directory), terrain, terrain_seed, terrain_type, terrain_level
    )
    path = Path(directory) / "scene.xml"
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
  if terrain == EXPORTED_TERRAIN:
    finish_exported_scene(model)
  else:
    fit_view_range(model)
  return model


class Go1Simulator:
  """Go1 with either the RMA teacher/adapter pair or a 48-D flat actor."""

  def __init__(
    self,
    teacher: Path | None,
    adapter: Path | None,
    terrain: str = "rough",
    terrain_seed: int = 42,
    policy: Path | None = None,
    cts: Path | None = None,
    terrain_type: str | None = None,
    terrain_level: int = 0,
    terrain_column: str | None = None,
    no_shadows: bool = False,
  ) -> None:
    self.model = make_model(terrain, terrain_seed, terrain_type, terrain_level)
    self.shadows_enabled = not no_shadows
    if no_shadows:
      self.model.vis.quality.shadowsize = 0
    self.spawn = spawn_position(
      self.model, terrain, terrain_type, terrain_level, terrain_column
    )
    self.light_mocap_id = shadow_light_mocap(self.model)
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        ]
        for j in JOINTS
      ]
    )
    self.vadr = np.array(
      [
        self.model.jnt_dofadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        ]
        for j in JOINTS
      ]
    )
    self.aid = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j + "_motor")
        for j in JOINTS
      ]
    )
    self.lin_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_vel"
    )
    self.ang_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
    )
    assert all(
      i >= 0 for i in (*self.qadr, *self.vadr, *self.aid, self.lin_id, self.ang_id)
    )
    self.actor = self.adapter = self.standard_policy = self.cts = None
    sources = [path for path in (policy, cts) if path is not None]
    if sources and teacher is not None:
      raise ValueError("Use either --policy, --cts or the --teacher/--adapter pair")
    if len(sources) > 1:
      raise ValueError("Use only one of --policy or --cts")
    if (teacher is None) != (adapter is None):
      raise ValueError("Both --teacher and --adapter are required together")
    if policy is not None:
      self.standard_policy = load_standard_policy(policy)
    elif cts is not None:
      self.cts = load_cts_policy(cts)
    elif teacher is not None:
      self.actor, self.adapter = load_policy(teacher, adapter)
    self.command = np.zeros(3, dtype=np.float32)
    self.active = False
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
    follow_shadow_light(self.data, self.light_mocap_id)
    self.data.qpos[self.qadr] = DEFAULT
    self.data.ctrl[self.aid] = DEFAULT
    self.history = np.zeros((HISTORY_LENGTH, OBS_DIM), dtype=np.float32)
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.target = DEFAULT.copy()
    mujoco.mj_forward(self.model, self.data)

  def set_active(self, active: bool) -> None:
    if (
      active
      and self.actor is None
      and self.standard_policy is None
      and self.cts is None
    ):
      print("No checkpoints loaded; staying in passive stand")
      return
    self.active = active
    self.history.fill(0)
    self.last_action.fill(0)
    if not active:
      print("Passive stand")
    elif self.cts is not None:
      print("CTS student active")
    elif self.standard_policy is not None:
      print("Standard policy active")
    else:
      print("RMA active")

  def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame base linear and angular velocity from the IMU sensors."""
    lin = self.data.sensordata[self.model.sensor_adr[self.lin_id] :][:3]
    ang = self.data.sensordata[self.model.sensor_adr[self.ang_id] :][:3]
    return lin, ang

  def observation(self) -> np.ndarray:
    lin, ang = self.base_velocity()
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    components = (
      ang,
      gravity,
      self.data.qpos[self.qadr] - DEFAULT,
      self.data.qvel[self.vadr],
      self.last_action,
      self.command,
    )
    # The flat/RMA actor keeps the privileged base linear velocity in front;
    # the blind CTS actor drops it and uses the 45-D layout.
    if self.cts is None:
      components = (lin, *components)
    obs = np.concatenate(components).astype(np.float32)
    expected = CTS_OBS_DIM if self.cts is not None else OBS_DIM
    assert obs.shape == (expected,)
    return obs

  def policy_step(self) -> None:
    if not self.active:
      self.target = DEFAULT.copy()
      self.last_action.fill(0)
      return
    obs = self.observation()
    with torch.inference_mode():
      obs_tensor = torch.from_numpy(obs[None])
      if self.cts is not None:
        action = self.cts(obs_tensor)[0].numpy()
      elif self.standard_policy is not None:
        action = self.standard_policy(obs_tensor)[0].numpy()
      else:
        history = torch.from_numpy(self.history[None])
        latent = self.adapter(history)
        action = self.actor(obs_tensor, latent)[0].numpy()
    if not np.isfinite(action).all():
      raise FloatingPointError("Policy returned non-finite actions")
    self.target = DEFAULT + ACTION_SCALE * action
    if self.cts is None:
      # The RMA adapter consumes this 48-D history; CTS keeps its own 45-D
      # per-term buffer inside the policy module.
      self.history[:-1] = self.history[1:].copy()
      self.history[-1] = obs
    self.last_action = action

  def physics_step(self) -> None:
    self.data.ctrl[self.aid] = self.target
    mujoco.mj_step(self.model, self.data)
