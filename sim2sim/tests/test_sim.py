import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import mujoco

from go1_sim import ACTION_SCALE, DEFAULT, JOINTS, Joystick, Simulator, make_model
from rma_policy import (
  ACTION_DIM,
  HISTORY_LENGTH,
  LATENT_DIM,
  OBS_DIM,
  Actor,
  Adapter,
  StandardActor,
  load_standard_policy,
)


class SimTests(unittest.TestCase):
  def test_standard_policy_checkpoint(self):
    actor = StandardActor((512, 256, 128))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.pt"
      state = {"actor_state_dict": actor.state_dict()}
      state["actor_state_dict"]["distribution.std_param"] = torch.ones(12)
      torch.save(state, path)
      loaded = load_standard_policy(path)
      observation = torch.randn(1, OBS_DIM)
      with torch.inference_mode():
        torch.testing.assert_close(loaded(observation), actor(observation))

  def test_terrain_collision_types(self):
    for terrain, expected_type in (
      ("flat", mujoco.mjtGeom.mjGEOM_PLANE),
      ("rough", mujoco.mjtGeom.mjGEOM_HFIELD),
      ("stairs", mujoco.mjtGeom.mjGEOM_HFIELD),
    ):
      model = make_model(terrain, terrain_seed=7)
      ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
      self.assertGreaterEqual(ground_id, 0)
      self.assertEqual(model.geom_type[ground_id], expected_type)
      if terrain != "flat":
        self.assertGreater(np.ptp(model.hfield_data), 0.0)

  def test_rl_sar_obstacle_course(self):
    model = make_model("rl_sar")
    self.assertEqual(model.nhfield, 2)
    self.assertGreaterEqual(model.ngeom, 100)
    geom_types = set(model.geom_type.tolist())
    self.assertIn(mujoco.mjtGeom.mjGEOM_PLANE, geom_types)
    self.assertIn(mujoco.mjtGeom.mjGEOM_BOX, geom_types)
    self.assertIn(mujoco.mjtGeom.mjGEOM_CYLINDER, geom_types)
    self.assertIn(mujoco.mjtGeom.mjGEOM_HFIELD, geom_types)

  def test_inference_shapes(self):
    actor, adapter = Actor(), Adapter()
    with torch.inference_mode():
      latent = adapter(torch.zeros(1, HISTORY_LENGTH, OBS_DIM))
      action = actor(torch.zeros(1, OBS_DIM), latent)
    self.assertEqual(tuple(latent.shape), (1, LATENT_DIM))
    self.assertEqual(tuple(action.shape), (1, ACTION_DIM))

  def test_scene_observations_and_history(self):
    sim = Simulator(None, None)
    self.assertEqual(len(JOINTS), ACTION_DIM)
    self.assertEqual(sim.model.nu, ACTION_DIM)
    np.testing.assert_allclose(sim.data.qpos[sim.qadr], DEFAULT)
    expected = np.zeros(OBS_DIM)
    expected[6:9] = [0, 0, -1]
    np.testing.assert_allclose(sim.observation(), expected, atol=1e-6)
    sim.actor, sim.adapter = Actor().eval(), Adapter().eval()
    sim.set_active(True)
    sim.command[:] = [0.2, 0.0, 0.1]
    sim.policy_step()
    np.testing.assert_allclose(sim.history[-1, -3:], sim.command)
    np.testing.assert_allclose(sim.target, DEFAULT + ACTION_SCALE * sim.last_action)
    for _ in range(4):
      sim.physics_step()
    sim.policy_step()
    np.testing.assert_allclose(sim.history[-2, -3:], sim.command)
    sim.reset()
    self.assertFalse(sim.history.any())

  def test_joystick_events(self):
    import struct

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "js0"
      path.write_bytes(
        struct.pack("IhBB", 0, -16384, 2, 1) + struct.pack("IhBB", 1, 1, 1, 0)
      )
      js = Joystick(str(path))
      try:
        self.assertEqual(js.poll(), {0})
        np.testing.assert_allclose(js.command(), [0.5, 0.0, 0.0])
      finally:
        js.close()

  def test_joystick_dpad_edges_and_initial_events(self):
    import os
    import struct

    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    js = Joystick("")
    js.fd = read_fd

    def event(value, kind, number):
      return struct.pack("IhBB", 0, value, kind, number)

    try:
      os.write(write_fd, event(1, 0x81, 0) + event(-32767, 0x82, 7))
      self.assertEqual(js.poll(), set())
      self.assertEqual(js.hat_pressed, set())
      os.write(write_fd, event(0, 2, 7) + event(-32767, 2, 7))
      js.poll()
      self.assertEqual(js.hat_pressed, {"up"})
      js.poll()
      self.assertEqual(js.hat_pressed, set())
      os.write(write_fd, event(0, 2, 7) + event(32767, 2, 6))
      js.poll()
      self.assertEqual(js.hat_pressed, {"right"})
    finally:
      js.close()
      os.close(write_fd)


if __name__ == "__main__":
  unittest.main()
