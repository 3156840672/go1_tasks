"""Shared viewer plumbing: keyboard command, camera and loop wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from quadruped_sim.policies.actor import StandardActor
from quadruped_sim.robots.go2 import Go2Simulator
from quadruped_sim.viewer import KeyboardCommand, chase_azimuth, run_viewer


class _FakeViewer:
  """Minimal stand-in for the passive viewer handle."""

  def __init__(self, frames: int = 3) -> None:
    self.cam = SimpleNamespace(
      lookat=np.zeros(3),
      distance=1.8,
      azimuth=135.0,
      elevation=0.0,
      type=mujoco.mjtCamera.mjCAMERA_FREE,
      trackbodyid=-1,
    )
    self.user_scn = SimpleNamespace(flags={})
    self.frames = frames
    self.rendered = 0

  def is_running(self) -> bool:
    return self.rendered < self.frames

  def sync(self) -> None:
    self.rendered += 1

  def __enter__(self) -> _FakeViewer:
    return self

  def __exit__(self, *exc_info: object) -> bool:
    return False


class KeyboardCommandTests(unittest.TestCase):
  def test_command_saturates_at_the_task_limits(self):
    import glfw

    go1 = KeyboardCommand((0.0, 0.0, 0.0), (1.0, 1.0, 0.5))
    for _ in range(20):
      self.assertTrue(go1.on_key(glfw.KEY_A))
      self.assertTrue(go1.on_key(glfw.KEY_Q))
    np.testing.assert_allclose(go1.values, [0.0, 1.0, 0.5], atol=1e-6)

    go2 = KeyboardCommand((0.0, 0.0, 0.0), (1.0, 0.6, 1.0))
    for _ in range(20):
      self.assertTrue(go2.on_key(glfw.KEY_D))
      self.assertTrue(go2.on_key(glfw.KEY_E))
    np.testing.assert_allclose(go2.values, [0.0, -0.6, -1.0], atol=1e-6)

    self.assertTrue(go2.on_key(glfw.KEY_SPACE))
    np.testing.assert_allclose(go2.values, 0.0)

  def test_every_velocity_key_maps_to_a_real_glfw_name(self):
    import glfw

    for name, (axis, delta) in KeyboardCommand.VELOCITY_KEYS.items():
      command = KeyboardCommand((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
      self.assertTrue(hasattr(glfw, name), name)
      self.assertTrue(command.on_key(getattr(glfw, name)), name)
      self.assertAlmostEqual(float(command.values[axis]), delta, places=6)

  def test_unrelated_keys_are_left_to_the_runner(self):
    import glfw

    command = KeyboardCommand((0.0, 0.0, 0.0), (1.0, 0.6, 1.0))
    self.assertFalse(command.on_key(glfw.KEY_1))
    self.assertFalse(command.on_key(glfw.KEY_R))
    np.testing.assert_allclose(command.values, 0.0)


class ViewerLoopTests(unittest.TestCase):
  def test_loop_runs_the_policy_and_chases_the_heading(self):
    import glfw

    sim = Go2Simulator()
    sim.policy = StandardActor((16,))
    sim.set_active(True)
    command = KeyboardCommand((0.3, 0.0, 0.0), (1.0, 0.6, 1.0))
    viewer = _FakeViewer()
    captured: dict[str, object] = {}

    def fake_launch(model, data, key_callback=None):
      captured["callback"] = key_callback
      return viewer

    with mock.patch(
      "mujoco.viewer.launch_passive", side_effect=fake_launch
    ):
      run_viewer(
        sim,
        command=command,
        camera="chase",
        joystick_path="",
      )

    self.assertEqual(viewer.rendered, 3)
    # The policy ran on the first frame and the robot settled onto its feet.
    self.assertTrue(np.any(sim.last_action))
    self.assertLess(sim.data.qpos[2], 0.38)
    # The chase camera turned with the (still upright) base heading.
    self.assertNotEqual(viewer.cam.azimuth, 135.0)
    self.assertGreater(viewer.cam.distance, 0.0)
    self.assertEqual(
      viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW], int(True)
    )

    callback = captured["callback"]
    assert callback is not None
    callback(glfw.KEY_W)
    self.assertAlmostEqual(float(command.values[0]), 0.4, places=6)


class ChaseAzimuthTests(unittest.TestCase):
  def test_heading_offset_follows_the_yaw(self):
    self.assertAlmostEqual(chase_azimuth(np.array([1.0, 0.0, 0.0, 0.0])), 180.0)
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    self.assertAlmostEqual(chase_azimuth(yaw90) % 360.0, 270.0)


if __name__ == "__main__":
  unittest.main()