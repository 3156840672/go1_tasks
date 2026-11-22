"""Shared viewer, keyboard command and simulation loops for the runners.

The interactive paths import ``glfw``/``mujoco.viewer`` lazily so that
``--headless`` runs keep working without a display server.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from .joystick import Joystick

#: Physics steps per policy step (50 Hz policy, 200 Hz physics).
DECIMATION = 4


def chase_azimuth(quat: np.ndarray) -> float:
  """Camera azimuth that keeps the camera behind the robot's heading."""
  w, x, y, z = quat
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return float(np.degrees(yaw)) + 180.0


def configure_viewer_rendering(viewer, shadows_enabled: bool) -> None:
  """Apply the requested shadow mode; leave diagnostic displays usable."""
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows_enabled)


class KeyboardCommand:
  """WASD/QE velocity command shared by every runner.

  ``limits`` carries the task's command range, so the same keys saturate at
  ``|vy| <= 1.0`` for Go1 and ``|vy| <= 0.6`` for Go2 and MoB.
  """

  #: glfw key name -> (axis, delta); resolved lazily so headless runs work.
  VELOCITY_KEYS = {
    "KEY_W": (0, +0.1),
    "KEY_UP": (0, +0.1),
    "KEY_S": (0, -0.1),
    "KEY_DOWN": (0, -0.1),
    "KEY_A": (1, +0.1),
    "KEY_D": (1, -0.1),
    "KEY_Q": (2, +0.1),
    "KEY_E": (2, -0.1),
  }

  def __init__(self, command: Sequence[float], limits: Sequence[float]) -> None:
    self.values = np.array(command, dtype=np.float32)
    self.limits = np.asarray(limits, dtype=np.float32)

  def zero(self) -> None:
    self.values[:] = 0.0

  def on_key(self, key: int) -> bool:
    """Apply a velocity key; return True when the key belonged to the command."""
    import glfw

    if key == glfw.KEY_SPACE:
      self.zero()
      return True
    for name, (axis, delta) in self.VELOCITY_KEYS.items():
      if key == getattr(glfw, name):
        limit = float(self.limits[axis])
        self.values[axis] = float(
          np.clip(self.values[axis] + delta, -limit, limit)
        )
        return True
    return False

  def describe(self) -> str:
    vx, vy, wz = (float(value) for value in self.values)
    return f"keyboard vx={vx:+.1f} vy={vy:+.1f} wz={wz:+.1f}"


def _place_camera(
  viewer,
  sim,
  *,
  camera: str,
  follow_distance: float,
  lookat: Sequence[float],
  distance: float,
) -> None:
  if camera == "free":
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = distance
    return
  # Tracking keeps base_link centered while the world scrolls past.
  base_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  viewer.cam.trackbodyid = base_id
  viewer.cam.distance = follow_distance
  viewer.cam.elevation = -22.0
  viewer.cam.azimuth = 135.0


def run_viewer(
  sim,
  *,
  command: KeyboardCommand,
  on_key: Callable[[int], bool | None] | None = None,
  gamepad: Callable[[set[int], set[str]], bool] | None = None,
  camera: str = "free",
  follow_distance: float = 2.2,
  lookat: Sequence[float] = (0.0, 0.0, 0.25),
  distance: float = 1.8,
  joystick_path: str = "/dev/input/js0",
  command_scale: Sequence[float] | None = None,
  decimation: int = DECIMATION,
  banner: Sequence[str] = (),
) -> None:
  """Run the interactive viewer loop until it is closed.

  Buttons ``0/1/2`` enable, pause and reset the policy unless ``gamepad`` takes
  over the device (MoB adds gait and behaviour controls).  ``on_key`` handles
  runner-specific keys before the shared velocity keys are applied and
  returns True when it consumed the key.
  """
  import glfw
  import mujoco.viewer  # noqa: F401  (registers the passive viewer)

  for line in banner:
    print(line)

  def key_callback(key: int) -> None:
    if on_key is not None and on_key(key):
      return
    if command.on_key(key):
      print(command.describe())

  scale = command_scale if command_scale is not None else command.limits
  joystick = Joystick(joystick_path)
  step = 0
  try:
    with mujoco.viewer.launch_passive(
      sim.model, sim.data, key_callback=key_callback
    ) as viewer:
      configure_viewer_rendering(viewer, getattr(sim, "shadows_enabled", True))
      _place_camera(
        viewer,
        sim,
        camera=camera,
        follow_distance=follow_distance,
        lookat=lookat,
        distance=distance,
      )
      while viewer.is_running():
        start = time.monotonic()
        buttons = joystick.poll()
        if gamepad is not None:
          if gamepad(buttons, joystick.hat_pressed):
            step = 0
        else:
          if 0 in buttons:
            sim.set_active(True)
          if 1 in buttons:
            sim.set_active(False)
          if 2 in buttons:
            sim.reset()
            step = 0
        joystick_command = joystick.command(tuple(scale))
        sim.command[:] = (
          joystick_command
          if joystick_command is not None and np.any(joystick_command)
          else command.values
        )
        if step % decimation == 0:
          sim.policy_step()
        sim.physics_step()
        if camera == "chase":
          viewer.cam.azimuth = chase_azimuth(sim.data.qpos[3:7])
        viewer.sync()
        step += 1
        time.sleep(max(0.0, sim.model.opt.timestep - (time.monotonic() - start)))
  finally:
    joystick.close()