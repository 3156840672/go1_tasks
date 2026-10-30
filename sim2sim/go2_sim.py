"""Independent MuJoCo Go2 policy sim2sim runner.

Uses the same self-contained Go2 MJCF that the training task loads and the
fixed deployment contract in :mod:`go2_tasks.deployment`. It imports neither
mjlab nor rsl_rl.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from go1_sim import Joystick
from rma_policy import ACTION_DIM, OBS_DIM, load_standard_policy
from terrain import TERRAIN_MODES, add_terrain

ROOT = Path(__file__).resolve().parent
GO2_XML = ROOT.parent / "src" / "go2_tasks" / "assets" / "go2" / "go2.xml"

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
BASE_HEIGHT = 0.38
AMP_OBS_DIM = 45
COMMAND_LIMITS = np.array([1.0, 0.6, 1.0], dtype=np.float32)


def make_model(
  terrain: str = "flat", terrain_seed: int = 42
) -> mujoco.MjModel:
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
    add_terrain(root, Path(directory), terrain, terrain_seed)
    path = Path(directory) / "scene.xml"
    tree.write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path))


class Simulator:
  def __init__(
    self,
    policy: Path | None = None,
    terrain: str = "flat",
    terrain_seed: int = 42,
  ) -> None:
    self.model = make_model(terrain, terrain_seed)
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
    self.policy = (
      load_standard_policy(policy, supported_obs_dims=(AMP_OBS_DIM, OBS_DIM))
      if policy is not None
      else None
    )
    self.command = np.zeros(3, dtype=np.float32)
    self.active = False
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.target = DEFAULT.copy()
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:7] = [0, 0, BASE_HEIGHT, 1, 0, 0, 0]
    self.data.qpos[self.qadr] = DEFAULT
    self.data.ctrl[self.aid] = 0
    self.last_action.fill(0)
    self.target = DEFAULT.copy()
    mujoco.mj_forward(self.model, self.data)

  def set_active(self, active: bool) -> None:
    if active and self.policy is None:
      print("No checkpoint loaded; staying in passive stand")
      return
    self.active = active
    self.last_action.fill(0)
    print("Standard policy active" if active else "Passive stand")

  def observation(self) -> np.ndarray:
    lin, ang = self.base_velocity()
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    # Flat uses privileged base linear velocity in its 48-D actor observation.
    # AMP deliberately omits it and uses the remaining 45 terms in the same order.
    obs_dim = getattr(self.policy, "obs_dim", OBS_DIM)
    components = (
      ang,
      gravity,
      self.data.qpos[self.qadr] - DEFAULT,
      self.data.qvel[self.vadr],
      self.last_action,
      self.command,
    )
    if obs_dim == OBS_DIM:
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
    # Same PD and DC motor torque-speed clipping as mjlab's DcMotorActuator.
    vel = self.data.qvel[self.vadr]
    effort = KP * (self.target - self.data.qpos[self.qadr]) - KD * vel
    corner_vel = VELOCITY_LIMIT * (1.0 + EFFORT / SATURATION_EFFORT)
    clipped_vel = np.clip(vel, -corner_vel, corner_vel)
    upper = np.minimum(SATURATION_EFFORT * (1.0 - clipped_vel / VELOCITY_LIMIT), EFFORT)
    lower = np.maximum(SATURATION_EFFORT * (-1.0 - clipped_vel / VELOCITY_LIMIT), -EFFORT)
    self.data.ctrl[self.aid] = np.clip(effort, lower, upper)
    mujoco.mj_step(self.model, self.data)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--policy",
    type=Path,
    help="45-D AMP or 48-D Flat model_*.pt checkpoint",
  )
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
  parser.add_argument("--terrain", choices=TERRAIN_MODES, default="flat")
  parser.add_argument("--terrain-seed", type=int, default=42)
  parser.add_argument(
    "--joystick", default="/dev/input/js0", help="Linux joystick device; empty disables"
  )
  args = parser.parse_args()
  if args.steps <= 0:
    parser.error("--steps must be positive")
  if any(abs(v) > limit for v, limit in zip(args.command, COMMAND_LIMITS)):
    parser.error("--command limits: |vx| <= 1.0; |vy| <= 0.6; |wz| <= 1.0")
  sim = Simulator(args.policy, args.terrain, args.terrain_seed)
  keyboard_command = np.array(args.command, dtype=np.float32)
  sim.command[:] = keyboard_command
  joystick = Joystick(args.joystick if not args.headless else "")
  if args.headless:
    if sim.policy is not None:
      sim.set_active(True)
    measured_velocities = []
    for step in range(args.steps):
      if step % 4 == 0:
        sim.policy_step()
      sim.physics_step()
      if step >= args.steps // 2:
        lin_vel, ang_vel = sim.base_velocity()
        measured_velocities.append((lin_vel[0], lin_vel[1], ang_vel[2]))
    mean_velocity = np.mean(measured_velocities, axis=0)
    error = np.abs(mean_velocity - sim.command)
    print(
      f"Completed {args.steps} steps; height={sim.data.qpos[2]:.3f} m; "
      f"mean vx/vy/wz={mean_velocity[0]:+.3f}/{mean_velocity[1]:+.3f}/"
      f"{mean_velocity[2]:+.3f}; abs error={error[0]:.3f}/{error[1]:.3f}/"
      f"{error[2]:.3f}"
    )
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
      keyboard_command[1] = np.clip(keyboard_command[1] + 0.1, -0.6, 0.6)
    elif key == glfw.KEY_D:
      keyboard_command[1] = np.clip(keyboard_command[1] - 0.1, -0.6, 0.6)
    elif key == glfw.KEY_Q:
      keyboard_command[2] = np.clip(keyboard_command[2] + 0.1, -1, 1)
    elif key == glfw.KEY_E:
      keyboard_command[2] = np.clip(keyboard_command[2] - 0.1, -1, 1)
    print(
      f"keyboard vx={keyboard_command[0]:+.1f} vy={keyboard_command[1]:+.1f} wz={keyboard_command[2]:+.1f}"
    )

  print("Keys: 1 policy, 0 passive, R reset, WASD/QE velocity, Space zero")
  print("Gamepad: A policy, B passive, X reset; left stick velocity, right stick X yaw")
  try:
    with mujoco.viewer.launch_passive(
      sim.model, sim.data, key_callback=on_key
    ) as viewer:
      viewer.cam.lookat[:] = [0, 0, 0.3]
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
        joy_cmd = joystick.command(tuple(COMMAND_LIMITS))
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
