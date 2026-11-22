"""Verify the blind-friendly rough terrain mix used by the Go2 tasks."""

import mujoco
import numpy as np
import pytest

from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

import go1_tasks  # noqa: F401  (registers the Go1 tasks)
import go2_tasks  # noqa: F401  (registers the Go2 tasks)
from go2_tasks.terrains import (
  BLIND_ROUGH_PROPORTIONS,
  MAX_BLIND_OBSTACLE_HEIGHT,
  blind_rough_terrains,
)
from tools.eval_go2_task import pin_terrain_level

ADDED_TERRAINS = (
  "discrete_obstacles",
  "box_random_grid",
  "random_spread_boxes",
  "open_stairs",
  "random_stairs",
  "tilted_grid",
  "perlin_noise",
)

HEIGHT_FIELDS = (
  "obstacle_height_range",
  "grid_height_range",
  "box_height_range",
  "step_height_range",
  "height_range",
)


def test_rough_terrain_mix_covers_every_column():
  generator = load_env_cfg("Go2-Velocity-HIM").scene.terrain.terrain_generator
  assert generator is not None
  assert set(generator.sub_terrains) == set(BLIND_ROUGH_PROPORTIONS)
  # In curriculum mode the generator emits one column per terrain type and uses
  # the proportions only to weight spawns, so only the weights need to be sane.
  assert all(sub.proportion > 0 for sub in generator.sub_terrains.values())
  assert sum(sub.proportion for sub in generator.sub_terrains.values()) == 1.0


@pytest.mark.parametrize(
  "name", ("discrete_obstacles", "open_stairs", "random_stairs", "random_spread_boxes")
)
def test_step_terrains_use_the_full_curriculum_budget(name):
  """The bound is the level-9 rung, so it must actually reach the budget."""
  sub_terrain = blind_rough_terrains()[name]
  bounds = [
    getattr(sub_terrain, field)
    for field in HEIGHT_FIELDS
    if getattr(sub_terrain, field, None) is not None
  ]
  assert any(
    max(bound) == pytest.approx(MAX_BLIND_OBSTACLE_HEIGHT) for bound in bounds
  )


def test_go1_and_the_mjlab_preset_are_unchanged():
  go1 = load_env_cfg("Go1-Velocity-Rough").scene.terrain.terrain_generator
  assert go1 is not None
  assert set(go1.sub_terrains) == set(ROUGH_TERRAINS_CFG.sub_terrains)


def test_eval_level_keeps_every_column_at_the_requested_difficulty():
  cfg = load_env_cfg("Go2-Velocity-HIM", play=True)
  training_rows = load_env_cfg("Go2-Velocity-HIM").scene.terrain.terrain_generator.num_rows
  pin_terrain_level(cfg, 9, training_rows)
  generator = cfg.scene.terrain.terrain_generator
  assert generator.curriculum
  assert generator.num_rows == 1
  assert generator.difficulty_range == (0.95, 0.95)
  assert cfg.scene.terrain.max_init_terrain_level == 0
  assert set(generator.sub_terrains) == set(BLIND_ROUGH_PROPORTIONS)
  assert "randomize_terrain" not in cfg.events
  assert "terrain_levels" not in cfg.curriculum


@pytest.mark.parametrize("name", ADDED_TERRAINS)
def test_added_terrain_heights_stay_within_the_blind_budget(name):
  sub_terrain = blind_rough_terrains()[name]
  for field in HEIGHT_FIELDS:
    value = getattr(sub_terrain, field, None)
    if value is None:
      continue
    upper = value if isinstance(value, float) else max(value)
    assert upper <= MAX_BLIND_OBSTACLE_HEIGHT + 1e-9


@pytest.mark.parametrize("name", ADDED_TERRAINS)
def test_added_terrain_generates_at_both_difficulty_extremes(name):
  sub_terrain = blind_rough_terrains()[name]
  sub_terrain.size = (8.0, 8.0)
  for difficulty in (0.0, 1.0):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = sub_terrain.function(difficulty, spec, np.random.default_rng(0))
    assert output.geometries
