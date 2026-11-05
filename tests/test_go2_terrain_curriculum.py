"""Verify actual heightfields become easier without reducing peak difficulty."""

import mujoco
import numpy as np

from mjlab.terrains import HfRandomUniformTerrainCfg
from go2_tasks.terrains import CurriculumRandomRoughTerrainCfg
from go2_tasks.train_terrain_amp import training_config


def _generate(cfg, difficulty):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty, spec, np.random.default_rng(42))
  field = output.geometries[0].hfield
  return float(field.size[2]), np.asarray(field.userdata)


def test_rough_heightfield_curriculum_preserves_hardest_terrain():
  values = dict(size=(2.,2.), noise_range=(.02,.1), noise_step=.02,
                vertical_scale=.005, horizontal_scale=.1, border_width=.25)
  cfg = CurriculumRandomRoughTerrainCfg(**values)
  low, _ = _generate(cfg, 0.)
  middle, _ = _generate(cfg, .5)
  high, high_data = _generate(cfg, 1.)
  old_high, old_data = _generate(HfRandomUniformTerrainCfg(**values), 1.)
  assert 0 < low < middle < high
  assert high == old_high
  np.testing.assert_array_equal(high_data, old_data)


def test_fresh_terrain_training_keeps_deployment_and_standing_contract():
  cfg = training_config()
  assert not cfg.agent.resume
  assert cfg.agent.max_iterations == 8000
  command = cfg.env.commands["twist"]
  assert command.resampling_time_range == (20., 20.)
  assert command.heading_command and command.rel_heading_envs == 1.
  assert command.rel_standing_envs == .05
  assert cfg.agent.algorithm.zero_command_amp_threshold == 0.
  assert cfg.agent.algorithm.discriminator_hidden_dims == (1024,512)
  assert "push_robot" in cfg.env.events
  assert isinstance(cfg.env.scene.terrain.terrain_generator.sub_terrains["random_rough"],
                    CurriculumRandomRoughTerrainCfg)
  final_range = cfg.env.curriculum["command_vel"].params["velocity_stages"][-1]
  assert final_range["lin_vel_x"] == (-1.,1.)
  assert final_range["lin_vel_y"] == (-.6,.6)
  assert final_range["ang_vel_z"] == (-1.,1.)
