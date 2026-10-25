"""Standalone MuJoCo sim2sim runner for the Go1 MoB policy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from go1_sim import DEFAULT, JOINTS, Joystick, make_model
from mob_policy import ACTION_DIM, OBS_DIM, load_mob_policy
from terrain import TERRAIN_MODES

HISTORY_LENGTH = 30
CONTROL_DECIMATION = 4
ACTION_SCALE = np.tile((0.125, 0.25, 0.25), 4)
GAITS = {
  "Pronk": (0.0, 0.0, 0.0),
  "Trot": (0.5, 0.0, 0.0),
  "Bound": (0.0, 0.5, 0.0),
  "Pace": (0.0, 0.0, 0.5),
}
TERM_DIMS = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "joint_pos": 12,
  "joint_vel": 12,
  "actions": 12,
  "command": 3,
  "behavior": 8,
  "clock": 4,
}
assert sum(TERM_DIMS.values()) * HISTORY_LENGTH == OBS_DIM


class MobGamepadControls:
  PARAMETERS = (
    ("body_height", 4, 0.01),
    ("body_pitch", 5, 0.01),
    ("frequency", 3, 0.1),
    ("stance_width", 6, 0.01),
    ("swing_height", 7, 0.01),
  )

  def __init__(self, bounds: dict[str, tuple[float, float]]) -> None:
    self.bounds = bounds
    self.selected = 0

  @property
  def selected_name(self) -> str:
    return self.PARAMETERS[self.selected][0]

  def handle(self, sim: MobSimulator, buttons: set[int], dpad: set[str]) -> bool:
    if 0 in buttons:
      sim.set_active(True)
    if 1 in buttons:
      sim.set_active(False)
    reset = 2 in buttons
    if reset:
      sim.reset()

    gait_names = tuple(GAITS)
    if 3 in buttons or "right" in dpad:
      sim.set_gait(gait_names[(gait_names.index(sim.gait) + 1) % len(gait_names)])
    elif "left" in dpad:
      sim.set_gait(gait_names[(gait_names.index(sim.gait) - 1) % len(gait_names)])

    if "up" in dpad:
      self.selected = (self.selected - 1) % len(self.PARAMETERS)
    elif "down" in dpad:
      self.selected = (self.selected + 1) % len(self.PARAMETERS)

    name, index, step = self.PARAMETERS[self.selected]
    if 4 in buttons:
      sim.adjust_behavior(index, -step, self.bounds[name])
    if 5 in buttons:
      sim.adjust_behavior(index, step, self.bounds[name])
    return reset


class MobSimulator:
  def __init__(
    self,
    checkpoint: Path | None,
    terrain: str = "flat",
    terrain_seed: int = 42,
  ) -> None:
    self.model = make_model(terrain, terrain_seed, mob=True)
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
    self.ang_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
    )
    if min(*self.qadr, *self.vadr, *self.aid, self.ang_id) < 0:
      raise ValueError("Go1 model is missing a joint, actuator, or IMU gyro")

    self.policy = load_mob_policy(checkpoint) if checkpoint is not None else None
    self.command = np.zeros(3, dtype=np.float32)
    self.behavior = np.array(
      (*GAITS["Trot"], 2.5, 0.30, 0.0, 0.30, 0.10), dtype=np.float32
    )
    self.gait = "Trot"
    self.active = False
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:7] = [0.0, 0.0, 0.278, 1.0, 0.0, 0.0, 0.0]
    self.data.qpos[self.qadr] = DEFAULT
    self.data.ctrl[self.aid] = DEFAULT
    self.target = DEFAULT.copy()
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.history = {
      name: np.zeros((HISTORY_LENGTH, dim), dtype=np.float32)
      for name, dim in TERM_DIMS.items()
    }
    self.history_ready = False
    self.cycle = 0.0
    mujoco.mj_forward(self.model, self.data)

  def set_active(self, active: bool) -> None:
    if active and self.policy is None:
      print("No MoB checkpoint loaded; remaining in passive stand")
      return
    self.active = active
    self.last_action.fill(0.0)
    for values in self.history.values():
      values.fill(0.0)
    self.history_ready = False
    self.target = DEFAULT.copy()
    print("MoB active" if active else "Passive stand")

  def set_gait(self, name: str) -> None:
    self.gait = name
    self.behavior[:3] = GAITS[name]

  def adjust_behavior(
    self, index: int, amount: float, limits: tuple[float, float]
  ) -> None:
    self.behavior[index] = np.clip(self.behavior[index] + amount, *limits)

  def raw_observation_terms(self) -> dict[str, np.ndarray]:
    gyro_adr = self.model.sensor_adr[self.ang_id]
    ang_vel = self.data.sensordata[gyro_adr : gyro_adr + 3]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    phase, bound, pace = self.behavior[:3]
    shifts = np.array((bound + pace, phase + bound, phase + pace, 0.0))
    clock = np.sin(2.0 * np.pi * ((self.cycle + shifts) % 1.0))
    return {
      "base_ang_vel": ang_vel.astype(np.float32),
      "projected_gravity": gravity.astype(np.float32),
      "joint_pos": (self.data.qpos[self.qadr] - DEFAULT).astype(np.float32),
      "joint_vel": self.data.qvel[self.vadr].astype(np.float32),
      "actions": self.last_action.copy(),
      "command": self.command.copy(),
      "behavior": self.behavior.copy(),
      "clock": clock.astype(np.float32),
    }

  def observation(self) -> np.ndarray:
    terms = self.raw_observation_terms()
    for name, current in terms.items():
      buffer = self.history[name]
      if self.history_ready:
        buffer[:-1] = buffer[1:].copy()
        buffer[-1] = current
      else:
        buffer[:] = current
    self.history_ready = True
    obs = np.concatenate([values.reshape(-1) for values in self.history.values()])
    assert obs.shape == (OBS_DIM,)
    return obs

  def policy_step(self) -> None:
    if not self.active:
      self.target = DEFAULT.copy()
      self.last_action.fill(0.0)
      return
    obs = self.observation()
    assert self.policy is not None
    with torch.inference_mode():
      action = self.policy(torch.from_numpy(obs[None]))[0].numpy()
    if not np.isfinite(action).all():
      raise FloatingPointError("MoB policy returned non-finite actions")
    self.target = DEFAULT + ACTION_SCALE * action
    self.last_action[:] = action
    self.cycle = (
      self.cycle + CONTROL_DECIMATION * self.model.opt.timestep * self.behavior[3]
    ) % 1.0

  def physics_step(self) -> None:
    self.data.ctrl[self.aid] = self.target
    mujoco.mj_step(self.model, self.data)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--terrain", choices=TERRAIN_MODES, default="flat")
  parser.add_argument("--terrain-seed", type=int, default=42)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--steps", type=int, default=1000, help="Physics steps in headless mode"
  )
  parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.0),
    metavar=("VX", "VY", "WZ"),
  )
  parser.add_argument("--gait", choices=tuple(GAITS), default="Trot")
  parser.add_argument("--frequency", type=float, default=2.5)
  parser.add_argument("--body-height", type=float, default=0.30)
  parser.add_argument("--body-pitch", type=float, default=0.0)
  parser.add_argument("--stance-width", type=float, default=0.30)
  parser.add_argument("--swing-height", type=float, default=0.10)
  parser.add_argument(
    "--joystick", default="/dev/input/js0", help="Linux joystick; empty disables"
  )
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.steps <= 0:
    parser.error("--steps must be positive")
  bounds = {
    "frequency": (1.5, 4.0),
    "body_height": (0.20, 0.38),
    "body_pitch": (-0.15, 0.15),
    "stance_width": (0.22, 0.40),
    "swing_height": (0.04, 0.18),
  }
  for name, (low, high) in bounds.items():
    value = getattr(args, name)
    if not low <= value <= high:
      parser.error(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
  if any(abs(v) > limit for v, limit in zip(args.command, (1.0, 0.6, 1.0))):
    parser.error("--command limits: |vx|<=1.0, |vy|<=0.6, |wz|<=1.0")
  if args.terrain != "flat":
    print(
      "Warning: MoB was trained on flat terrain; non-flat terrain is out of distribution"
    )

  sim = MobSimulator(args.checkpoint, args.terrain, args.terrain_seed)
  sim.command[:] = args.command
  sim.set_gait(args.gait)
  sim.behavior[3:] = [
    args.frequency,
    args.body_height,
    args.body_pitch,
    args.stance_width,
    args.swing_height,
  ]
  sim.set_active(True)
  if args.headless:
    for step in range(args.steps):
      if step % CONTROL_DECIMATION == 0:
        sim.policy_step()
      sim.physics_step()
    print(f"Completed {args.steps} physics steps; height={sim.data.qpos[2]:.3f} m")
    return

  import glfw
  import mujoco.viewer

  keyboard_command = sim.command.copy()

  def show_commands() -> None:
    b = sim.behavior
    print(
      f"vx={sim.command[0]:+.2f} vy={sim.command[1]:+.2f} wz={sim.command[2]:+.2f} "
      f"{sim.gait} freq={b[3]:.1f} height={b[4]:.2f} pitch={b[5]:+.2f} "
      f"width={b[6]:.2f} swing={b[7]:.2f}"
    )

  def on_key(key: int) -> None:
    if key == glfw.KEY_0:
      sim.set_active(False)
    elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
      sim.set_gait(tuple(GAITS)[key - glfw.KEY_1])
      sim.set_active(True)
    elif key == glfw.KEY_R:
      sim.reset()
    elif key == glfw.KEY_SPACE:
      keyboard_command[:] = 0.0
    elif key in (glfw.KEY_W, glfw.KEY_UP, glfw.KEY_S, glfw.KEY_DOWN):
      keyboard_command[0] = np.clip(
        keyboard_command[0] + (0.1 if key in (glfw.KEY_W, glfw.KEY_UP) else -0.1),
        -1.0,
        1.0,
      )
    elif key in (glfw.KEY_A, glfw.KEY_D):
      keyboard_command[1] = np.clip(
        keyboard_command[1] + (0.1 if key == glfw.KEY_A else -0.1), -0.6, 0.6
      )
    elif key in (glfw.KEY_Q, glfw.KEY_E):
      keyboard_command[2] = np.clip(
        keyboard_command[2] + (0.1 if key == glfw.KEY_Q else -0.1), -1.0, 1.0
      )
    else:
      adjustments = {
        glfw.KEY_Z: (4, -0.01, bounds["body_height"]),
        glfw.KEY_X: (4, 0.01, bounds["body_height"]),
        glfw.KEY_C: (5, -0.01, bounds["body_pitch"]),
        glfw.KEY_V: (5, 0.01, bounds["body_pitch"]),
        glfw.KEY_F: (3, -0.1, bounds["frequency"]),
        glfw.KEY_G: (3, 0.1, bounds["frequency"]),
        glfw.KEY_B: (6, -0.01, bounds["stance_width"]),
        glfw.KEY_N: (6, 0.01, bounds["stance_width"]),
        glfw.KEY_H: (7, -0.01, bounds["swing_height"]),
        glfw.KEY_J: (7, 0.01, bounds["swing_height"]),
      }
      if key in adjustments:
        sim.adjust_behavior(*adjustments[key])
    sim.command[:] = keyboard_command
    show_commands()

  print(
    "Keys: 1-4 Pronk/Trot/Bound/Pace, 0 passive, R reset, WASD/QE velocity, Space stop"
  )
  print("Z/X height, C/V pitch, F/G frequency, B/N stance width, H/J swing height")
  print(
    "Gamepad: A active, B passive, X reset, Y or D-pad right next gait, D-pad left previous gait"
  )
  print(
    "D-pad up/down selects height, pitch, frequency, stance width or swing height; LB/RB adjusts it"
  )
  joystick = Joystick(args.joystick)
  gamepad = MobGamepadControls(bounds)
  try:
    with mujoco.viewer.launch_passive(
      sim.model, sim.data, key_callback=on_key
    ) as viewer:
      viewer.cam.lookat[:] = [0.0, 0.0, 0.25]
      viewer.cam.distance = 1.8
      step = 0
      while viewer.is_running():
        start = time.monotonic()
        buttons = joystick.poll()
        if gamepad.handle(sim, buttons, joystick.hat_pressed):
          step = 0
        if buttons or joystick.hat_pressed:
          print(f"Gamepad setting: {gamepad.selected_name}")
          show_commands()
        if joystick.fd is not None:
          axes = np.array([-joystick.axes[1], -joystick.axes[0], -joystick.axes[3]])
          axes[np.abs(axes) < 0.05] = 0.0
          joy_command = (axes * (1.0, 0.6, 1.0)).astype(np.float32)
          sim.command[:] = joy_command if np.any(joy_command) else keyboard_command
        else:
          sim.command[:] = keyboard_command
        if step % CONTROL_DECIMATION == 0:
          sim.policy_step()
        sim.physics_step()
        viewer.sync()
        step += 1
        time.sleep(max(0.0, sim.model.opt.timestep - (time.monotonic() - start)))
  finally:
    joystick.close()


if __name__ == "__main__":
  main()
