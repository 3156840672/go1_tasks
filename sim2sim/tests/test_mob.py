import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from go1_mob_sim import (
  ACTION_SCALE,
  GAITS,
  HISTORY_LENGTH,
  TERM_DIMS,
  MobGamepadControls,
  MobSimulator,
)
from go1_sim import DEFAULT, make_model
from mob_policy import ACTION_DIM, OBS_DIM, MobActor, load_mob_policy


class MobSimTests(unittest.TestCase):
  def test_checkpoint_contract(self):
    actor = MobActor((512, 256, 128))
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.pt"
      torch.save({"actor_state_dict": actor.state_dict()}, path)
      loaded = load_mob_policy(path)
      obs = torch.randn(1, OBS_DIM)
      with torch.inference_mode():
        torch.testing.assert_close(loaded(obs), actor(obs))

      wrong = MobActor((128,))
      wrong.mlp[0] = torch.nn.Linear(48, 128)
      torch.save({"actor_state_dict": wrong.state_dict()}, path)
      with self.assertRaisesRegex(ValueError, "1710-input"):
        load_mob_policy(path)

  def test_mob_motor_and_collision_settings(self):
    model = make_model("flat", mob=True)
    self.assertEqual(model.nu, ACTION_DIM)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 20.0)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -20.0)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -0.5)
    self.assertEqual(model.opt.ccd_iterations, 50)
    self.assertEqual(model.opt.impratio, 1.0)
    self.assertEqual(model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)
    self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FR_foot_collision")
    calf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FR_calf_collision1")
    self.assertEqual(model.geom_condim[foot], 6)
    self.assertEqual(model.geom_condim[calf], 1)

  def test_term_major_oldest_first_history(self):
    sim = MobSimulator(None)
    self.assertEqual(sum(TERM_DIMS.values()) * HISTORY_LENGTH, OBS_DIM)
    self.assertEqual(sim.gait, "Trot")
    sim.set_gait("Pace")
    np.testing.assert_allclose(sim.behavior[:3], GAITS["Pace"])
    first = sim.observation()
    self.assertEqual(first.shape, (OBS_DIM,))
    command_start = sum(list(TERM_DIMS.values())[:5]) * HISTORY_LENGTH
    np.testing.assert_allclose(first[command_start : command_start + 90], 0.0)

    sim.command[:] = [0.2, -0.1, 0.3]
    second = sim.observation()
    command_history = second[command_start : command_start + 90].reshape(30, 3)
    np.testing.assert_allclose(command_history[:-1], 0.0)
    np.testing.assert_allclose(command_history[-1], sim.command)
    sim.cycle = 0.25
    np.testing.assert_allclose(
      sim.raw_observation_terms()["clock"], [-1, 1, -1, 1], atol=1e-6
    )

  def test_action_offsets(self):
    sim = MobSimulator(None)
    sim.policy = MobActor((32,)).eval()
    for parameter in sim.policy.parameters():
      parameter.data.zero_()
    sim.policy.mlp[-1].bias.data.fill_(1.0)
    sim.set_active(True)
    sim.policy_step()
    np.testing.assert_allclose(sim.target, DEFAULT + ACTION_SCALE)
    np.testing.assert_allclose(sim.last_action, 1.0)
    self.assertEqual(sim.observation().shape, (OBS_DIM,))

  def test_gamepad_reaches_all_behavior_commands(self):
    bounds = {
      "body_height": (0.20, 0.38),
      "body_pitch": (-0.15, 0.15),
      "frequency": (1.5, 4.0),
      "stance_width": (0.22, 0.40),
      "swing_height": (0.04, 0.18),
    }
    sim = MobSimulator(None)
    gamepad = MobGamepadControls(bounds)
    for name, index, step in gamepad.PARAMETERS:
      self.assertEqual(gamepad.selected_name, name)
      before = float(sim.behavior[index])
      gamepad.handle(sim, {5}, set())
      self.assertAlmostEqual(float(sim.behavior[index]), before + step, places=5)
      gamepad.handle(sim, {4}, set())
      self.assertAlmostEqual(float(sim.behavior[index]), before, places=5)
      gamepad.handle(sim, set(), {"down"})
    self.assertEqual(gamepad.selected_name, "body_height")
    gamepad.handle(sim, set(), {"up"})
    self.assertEqual(gamepad.selected_name, "swing_height")
    gamepad.handle(sim, set(), {"right"})
    self.assertEqual(sim.gait, "Bound")
    gamepad.handle(sim, set(), {"left"})
    self.assertEqual(sim.gait, "Trot")
    gamepad.handle(sim, {3}, set())
    self.assertEqual(sim.gait, "Bound")
    self.assertTrue(gamepad.handle(sim, {2}, set()))


if __name__ == "__main__":
  unittest.main()
