"""Independent MuJoCo Go1 RMA sim2sim runner."""

from __future__ import annotations

import argparse
import os
import select
import struct
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from rma_policy import (
  ACTION_DIM,
  HISTORY_LENGTH,
  OBS_DIM,
  load_policy,
  load_standard_policy,
)
from terrain import TERRAIN_MODES, add_terrain

ROOT = Path(__file__).resolve().parent
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


def make_model(
  terrain: str = "rough", terrain_seed: int = 42, *, mob: bool = False
) -> mujoco.MjModel:
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
  for i, name in enumerate(JOINTS):
    joint = root.find(f".//joint[@name='{name}']")
    if joint is None:
      raise ValueError(f"Missing joint {name}")
    joint.set(
      "armature", str(KNEE_ARMATURE if name.endswith("calf_joint") else HIP_ARMATURE)
    )
    kp = 20.0 if mob else KP[i]
    kd = 0.5 if mob else KD[i]
    ET.SubElement(
      actuators,
      "general",
      name=name + "_motor",
      joint=name,
      gainprm=str(kp),
      biastype="affine",
      biasprm=f"0 {-kp} {-kd}",
      forcelimited="true",
      forcerange=f"{-EFFORT[i]} {EFFORT[i]}",
    )
  with tempfile.TemporaryDirectory() as directory:
    add_terrain(root, Path(directory), terrain, terrain_seed)
    path = Path(directory) / "scene.xml"
    tree.write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path))


class Joystick:
  """Linux joystick API, with no extra package or required device."""

  def __init__(self, path: str) -> None:
    self.fd = None
    self.axes = [0.0] * 8
    self.buttons = set()
    self.hat_pressed: set[str] = set()
    self._hat_held: set[str] = set()
    if path:
      try:
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"Joystick: {path}")
      except OSError as exc:
        print(f"Joystick unavailable ({exc}); keyboard remains active")

  def poll(self) -> set[int]:
    pressed = set()
    self.hat_pressed.clear()
    if self.fd is None:
      return pressed
    while select.select([self.fd], [], [], 0)[0]:
      try:
        event = os.read(self.fd, 8)
      except OSError:
        break
      if len(event) != 8:
        break
      _, value, kind, number = struct.unpack("IhBB", event)
      initial = bool(kind & 0x80)
      kind &= ~0x80
      if kind == 2 and number < len(self.axes):
        self.axes[number] = float(value) / 32768.0
        if number in (6, 7):
          held = self._hat_directions()
          if not initial:
            self.hat_pressed.update(held - self._hat_held)
          self._hat_held = held
      elif kind == 1:
        if value:
          self.buttons.add(number)
          if not initial:
            pressed.add(number)
        else:
          self.buttons.discard(number)
    return pressed

  def _hat_directions(self) -> set[str]:
    directions = set()
    if self.axes[6] < -0.5:
      directions.add("left")
    elif self.axes[6] > 0.5:
      directions.add("right")
    if self.axes[7] < -0.5:
      directions.add("up")
    elif self.axes[7] > 0.5:
      directions.add("down")
    return directions

  def command(self) -> np.ndarray | None:
    if self.fd is None:
      return None
    values = np.array([-self.axes[1], -self.axes[0], -self.axes[3]], dtype=np.float32)
    values[np.abs(values) < 0.05] = 0
    return values * np.array([1.0, 1.0, 0.5], dtype=np.float32)

  def close(self) -> None:
    if self.fd is not None:
      os.close(self.fd)


class Simulator:
  def __init__(
    self,
    teacher: Path | None,
    adapter: Path | None,
    terrain: str = "rough",
    terrain_seed: int = 42,
    policy: Path | None = None,
  ) -> None:
    self.model = make_model(terrain, terrain_seed)
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
    self.actor = self.adapter = self.standard_policy = None
    if policy is not None and (teacher is not None or adapter is not None):
      raise ValueError("Use either --policy or the --teacher/--adapter pair")
    if (teacher is None) != (adapter is None):
      raise ValueError("Both --teacher and --adapter are required together")
    if policy is not None:
      self.standard_policy = load_standard_policy(policy)
    elif teacher is not None:
      self.actor, self.adapter = load_policy(teacher, adapter)
    self.command = np.zeros(3, dtype=np.float32)
    self.active = False
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:7] = [0, 0, 0.278, 1, 0, 0, 0]
    self.data.qpos[self.qadr] = DEFAULT
    self.data.ctrl[self.aid] = DEFAULT
    self.history = np.zeros((HISTORY_LENGTH, OBS_DIM), dtype=np.float32)
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.target = DEFAULT.copy()
    mujoco.mj_forward(self.model, self.data)

  def set_active(self, active: bool) -> None:
    if active and self.actor is None and self.standard_policy is None:
      print("No checkpoints loaded; staying in passive stand")
      return
    self.active = active
    self.history.fill(0)
    self.last_action.fill(0)
    if not active:
      print("Passive stand")
    elif self.standard_policy is not None:
      print("Standard policy active")
    else:
      print("RMA active")

  def observation(self) -> np.ndarray:
    lin = self.data.sensordata[self.model.sensor_adr[self.lin_id] :][:3]
    ang = self.data.sensordata[self.model.sensor_adr[self.ang_id] :][:3]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate(
      (
        lin,
        ang,
        gravity,
        self.data.qpos[self.qadr] - DEFAULT,
        self.data.qvel[self.vadr],
        self.last_action,
        self.command,
      )
    ).astype(np.float32)
    assert obs.shape == (OBS_DIM,)
    return obs

  def policy_step(self) -> None:
    if not self.active:
      self.target = DEFAULT.copy()
      self.last_action.fill(0)
      return
    obs = self.observation()
    with torch.inference_mode():
      obs_tensor = torch.from_numpy(obs[None])
      if self.standard_policy is not None:
        action = self.standard_policy(obs_tensor)[0].numpy()
      else:
        history = torch.from_numpy(self.history[None])
        latent = self.adapter(history)
        action = self.actor(obs_tensor, latent)[0].numpy()
    if not np.isfinite(action).all():
      raise FloatingPointError("Policy returned non-finite actions")
    self.target = DEFAULT + ACTION_SCALE * action
    self.history[:-1] = self.history[1:].copy()
    self.history[-1] = obs
    self.last_action = action

  def physics_step(self) -> None:
    self.data.ctrl[self.aid] = self.target
    mujoco.mj_step(self.model, self.data)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy", type=Path, help="48-observation model_*.pt from Go1-Velocity-Flat"
  )
  parser.add_argument("--teacher", type=Path, help="model_*.pt from stage 1")
  parser.add_argument("--adapter", type=Path, help="adapter_*.pt from stage 2")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--steps", type=int, default=1000, help="Physics steps in headless mode"
  )
  parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    metavar=("VX", "VY", "WZ"),
    default=(0.0, 0.0, 0.0),
  )
  parser.add_argument("--terrain", choices=TERRAIN_MODES, default="rough")
  parser.add_argument("--terrain-seed", type=int, default=42)
  parser.add_argument(
    "--joystick", default="/dev/input/js0", help="Linux joystick device; empty disables"
  )
  args = parser.parse_args()
  if args.steps <= 0:
    parser.error("--steps must be positive")
  if any(abs(v) > limit for v, limit in zip(args.command, (1.0, 1.0, 0.5))):
    parser.error("--command limits: |vx|, |vy| <= 1.0; |wz| <= 0.5")
  sim = Simulator(
    args.teacher,
    args.adapter,
    args.terrain,
    args.terrain_seed,
    policy=args.policy,
  )
  keyboard_command = np.array(args.command, dtype=np.float32)
  sim.command[:] = keyboard_command
  joystick = Joystick(args.joystick if not args.headless else "")
  if args.headless:
    if sim.actor is not None or sim.standard_policy is not None:
      sim.set_active(True)
    for step in range(args.steps):
      if step % 4 == 0:
        sim.policy_step()
      sim.physics_step()
    print(f"Completed {args.steps} steps; height={sim.data.qpos[2]:.3f} m")
    return

  import glfw
  import mujoco.viewer

  def on_key(key: int) -> None:
    if key == glfw.KEY_1:
      sim.set_active(True)
    elif key == glfw.KEY_0:
      sim.set_active(False)
    elif key == glfw.KEY_R:
      sim.reset()
    elif key == glfw.KEY_SPACE:
      keyboard_command[:] = 0
    elif key in (glfw.KEY_W, glfw.KEY_UP):
      keyboard_command[0] = np.clip(keyboard_command[0] + 0.1, -1, 1)
    elif key in (glfw.KEY_S, glfw.KEY_DOWN):
      keyboard_command[0] = np.clip(keyboard_command[0] - 0.1, -1, 1)
    elif key == glfw.KEY_A:
      keyboard_command[1] = np.clip(keyboard_command[1] + 0.1, -1, 1)
    elif key == glfw.KEY_D:
      keyboard_command[1] = np.clip(keyboard_command[1] - 0.1, -1, 1)
    elif key == glfw.KEY_Q:
      keyboard_command[2] = np.clip(keyboard_command[2] + 0.1, -0.5, 0.5)
    elif key == glfw.KEY_E:
      keyboard_command[2] = np.clip(keyboard_command[2] - 0.1, -0.5, 0.5)
    print(
      f"keyboard vx={keyboard_command[0]:+.1f} vy={keyboard_command[1]:+.1f} wz={keyboard_command[2]:+.1f}"
    )

  print("Keys: 1 RMA, 0 passive, R reset, WASD/QE velocity, Space zero")
  print("Gamepad: A RMA, B passive, X reset; left stick velocity, right stick X yaw")
  try:
    with mujoco.viewer.launch_passive(
      sim.model, sim.data, key_callback=on_key
    ) as viewer:
      viewer.cam.lookat[:] = [0, 0, 0.25]
      viewer.cam.distance = 1.8
      step = 0
      while viewer.is_running():
        start = time.monotonic()
        buttons = joystick.poll()
        if 0 in buttons:
          sim.set_active(True)
        if 1 in buttons:
          sim.set_active(False)
        if 2 in buttons:
          sim.reset()
          step = 0
        joy_cmd = joystick.command()
        sim.command[:] = (
          joy_cmd if joy_cmd is not None and np.any(joy_cmd) else keyboard_command
        )
        if step % 4 == 0:
          sim.policy_step()
        sim.physics_step()
        viewer.sync()
        step += 1
        time.sleep(max(0.0, sim.model.opt.timestep - (time.monotonic() - start)))
  finally:
    joystick.close()


if __name__ == "__main__":
  main()
