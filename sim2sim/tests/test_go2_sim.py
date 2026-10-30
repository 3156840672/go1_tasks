"""Go2 sim2sim model and inference contract tests."""

import unittest

import mujoco
import numpy as np
import torch

from go1_sim import Joystick
from go2_sim import (
  ACTION_SCALE,
  AMP_OBS_DIM,
  BASE_HEIGHT,
  DEFAULT,
  EFFORT,
  JOINTS,
  KD,
  KP,
  SATURATION_EFFORT,
  Simulator,
  VELOCITY_LIMIT,
  make_model,
)
from rma_policy import ACTION_DIM, OBS_DIM, StandardActor


class Go2SimTests(unittest.TestCase):
  def test_model_actuator_contract(self):
    model = make_model("flat")
    self.assertEqual(model.nu, ACTION_DIM)
    self.assertEqual(len(JOINTS), ACTION_DIM)
    for index, name in enumerate(JOINTS):
      actuator = model.actuator(name + "_motor")
      np.testing.assert_allclose(actuator.ctrlrange, [-EFFORT[index], EFFORT[index]])

  def test_dc_motor_torque_speed_limit(self):
    sim = Simulator(None)
    sim.target = DEFAULT + 10.0
    sim.physics_step()
    np.testing.assert_allclose(sim.data.ctrl[sim.aid], EFFORT)

    sim.reset()
    sim.target = DEFAULT + 10.0
    sim.data.qvel[sim.vadr] = 30.0
    sim.physics_step()
    np.testing.assert_allclose(sim.data.ctrl[sim.aid], 0.0, atol=1e-6)

  def test_terrain_modes(self):
    for terrain, expected_type in (
      ("flat", mujoco.mjtGeom.mjGEOM_PLANE),
      ("rough", mujoco.mjtGeom.mjGEOM_HFIELD),
      ("stairs", mujoco.mjtGeom.mjGEOM_HFIELD),
    ):
      model = make_model(terrain, terrain_seed=7)
      ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
      self.assertGreaterEqual(ground_id, 0)
      self.assertEqual(model.geom_type[ground_id], expected_type)

  def test_scene_observation_and_passive_stand(self):
    sim = Simulator(None)
    self.assertEqual(tuple(sim.data.qpos[sim.qadr]), tuple(DEFAULT))
    self.assertAlmostEqual(sim.data.qpos[2], BASE_HEIGHT)
    expected = np.zeros(OBS_DIM)
    expected[6:9] = [0, 0, -1]
    np.testing.assert_allclose(sim.observation(), expected, atol=1e-6)
    for _ in range(800):
      sim.physics_step()
    self.assertTrue(np.isfinite(sim.data.qpos).all())
    self.assertGreater(sim.data.qpos[2], 0.15)
    self.assertGreater(sim.data.qpos[3], 0.98)
    self.assertLess(abs(sim.data.qpos[4]), 0.2)
    self.assertLess(abs(sim.data.qpos[5]), 0.2)

  def test_policy_action_contract(self):
    sim = Simulator(None)
    sim.policy = StandardActor((64, 64)).eval()
    sim.set_active(True)
    sim.command[:] = [0.3, 0.0, 0.1]
    sim.policy_step()
    self.assertEqual(sim.observation().shape, (OBS_DIM,))
    np.testing.assert_allclose(
      sim.target, DEFAULT + ACTION_SCALE * sim.last_action
    )
    np.testing.assert_allclose(sim.observation()[-3:], sim.command)
    sim.set_active(False)
    sim.policy_step()
    np.testing.assert_allclose(sim.target, DEFAULT)
    self.assertFalse(sim.last_action.any())

  def test_amp_policy_observation_contract(self):
    sim = Simulator(None)
    sim.policy = StandardActor((512, 256, 128), obs_dim=AMP_OBS_DIM).eval()
    sim.set_active(True)
    sim.command[:] = [0.3, -0.2, 0.7]
    obs = sim.observation()
    self.assertEqual(obs.shape, (AMP_OBS_DIM,))
    np.testing.assert_allclose(obs[:3], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[-3:], sim.command)
    sim.policy_step()
    self.assertEqual(sim.last_action.shape, (ACTION_DIM,))

  def test_non_finite_policy_is_rejected(self):
    class BadPolicy:
      def __call__(self, obs):
        return torch.full((obs.shape[0], ACTION_DIM), float("nan"))

    sim = Simulator(None)
    sim.policy = BadPolicy()
    sim.set_active(True)
    with self.assertRaises(FloatingPointError):
      sim.policy_step()

  def test_joystick_is_shared_with_go1(self):
    joystick = Joystick("")
    try:
      self.assertIsNone(joystick.command())
    finally:
      joystick.close()

  def test_constants_match_training_contract(self):
    from go2_tasks.deployment import (
      ACTION_SCALE as DEPLOY_ACTION_SCALE,
      DEFAULT_JOINT_POS,
      JOINT_LOWER,
      JOINT_NAMES,
      JOINT_UPPER,
      KD as DEPLOY_KD,
      KP as DEPLOY_KP,
      SATURATION_EFFORT as DEPLOY_SATURATION_EFFORT,
      TORQUE_LIMIT,
      VELOCITY_LIMIT as DEPLOY_VELOCITY_LIMIT,
    )

    self.assertEqual(JOINTS, JOINT_NAMES)
    np.testing.assert_allclose(ACTION_SCALE, DEPLOY_ACTION_SCALE)
    np.testing.assert_allclose(DEFAULT, DEFAULT_JOINT_POS)
    np.testing.assert_allclose(EFFORT, TORQUE_LIMIT)
    np.testing.assert_allclose(KP, DEPLOY_KP)
    np.testing.assert_allclose(KD, DEPLOY_KD)
    np.testing.assert_allclose(SATURATION_EFFORT, DEPLOY_SATURATION_EFFORT)
    np.testing.assert_allclose(VELOCITY_LIMIT, DEPLOY_VELOCITY_LIMIT)
    model = make_model("flat")
    for name, lower, upper in zip(JOINT_NAMES, JOINT_LOWER, JOINT_UPPER):
      np.testing.assert_allclose(model.joint(name).range, [lower, upper], atol=1e-5)


if __name__ == "__main__":
  unittest.main()
