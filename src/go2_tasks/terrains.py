"""Go2 terrain curricula without changing the robot's physical parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from mjlab.terrains import HfRandomUniformTerrainCfg, SubTerrainCfg
from mjlab.terrains.config import (
  box_random_grid,
  discrete_obstacles,
  open_stairs,
  perlin_noise,
  random_spread_boxes,
  random_stairs,
  tilted_grid,
)

# Column share of every terrain in the Go2 rough curriculum.  The seven entries
# inherited from mjlab's rough preset are re-weighted to make room for the
# obstacle terrains; the total must stay at 1.0 because the generator allocates
# ``num_cols`` terrain columns by proportion.
BLIND_ROUGH_PROPORTIONS: dict[str, float] = {
  "flat": 0.10,
  "pyramid_stairs": 0.10,
  "pyramid_stairs_inv": 0.10,
  "hf_pyramid_slope": 0.07,
  "hf_pyramid_slope_inv": 0.07,
  "random_rough": 0.07,
  "wave_terrain": 0.07,
  "discrete_obstacles": 0.06,
  "box_random_grid": 0.06,
  "random_spread_boxes": 0.06,
  "open_stairs": 0.06,
  "random_stairs": 0.06,
  "tilted_grid": 0.06,
  "perlin_noise": 0.06,
}

# Difficulty maps a row linearly onto each parameter range
# (``difficulty = (row + U[0,1)) / num_rows``), so the *upper* bound is the
# hardest rung the curriculum can ever reach, not the starting difficulty.
# Training worlds start at level 0-5 and ``terrain_levels_vel`` promotes a world
# after it walks 4 m and demotes it when it covers less than half the commanded
# distance, so a high ceiling costs nothing early and only the worlds that
# already walk well ever see it.  A Go2 shank is ~0.22 m, so 0.14 m (~0.64 shank)
# is the tallest feature a proprioception-only policy can clear from foot
# contact alone; taller obstacles need a terrain scan to place a foot.
MAX_BLIND_OBSTACLE_HEIGHT = 0.14


def blind_rough_terrains() -> dict[str, SubTerrainCfg]:
  """Extra rough terrains that a proprioception-only Go2 can actually cross.

  These add obstacle diversity without the exteroception-only terrains (stepping
  stones, narrow beams, tall discrete obstacles): every range tops out at the
  hardest difficulty (level 9) the blind policy can still be expected to clear,
  and every gap is wide enough to stumble through, so the 45-D history policy can
  learn them from foot contact alone.  Difficulty still scales the heights (and
  narrows the open-stair treads), so each terrain keeps a usable curriculum from
  level 0 to level 9.
  """
  return {
    # Random bumps and pits, half of them negative (``choice`` mode).
    "discrete_obstacles": discrete_obstacles(
      obstacle_height_range=(0.02, MAX_BLIND_OBSTACLE_HEIGHT),
      obstacle_width_range=(0.3, 0.8),
      num_obstacles=30,
      platform_width=1.5,
    ),
    # Blocky grid whose cell heights span [-h, +h] about the nominal surface, so
    # the range is half the obstacle budget to keep peak-to-peak relief in line.
    # The lower bound must stay positive: the merged-box colour ramp divides by
    # the interpolated height, which is zero at difficulty 0.
    "box_random_grid": box_random_grid(
      grid_width=0.4,
      grid_height_range=(0.02, MAX_BLIND_OBSTACLE_HEIGHT / 2),
      platform_width=1.5,
      merge_similar_heights=True,
    ),
    # Loose boxes on a flat floor: trips and stumbles rather than walls.
    "random_spread_boxes": random_spread_boxes(
      num_boxes=40,
      box_width_range=(0.3, 0.7),
      box_length_range=(0.3, 0.7),
      box_height_range=(0.02, MAX_BLIND_OBSTACLE_HEIGHT),
      platform_width=1.5,
    ),
    # Stairs without risers.  The class narrows the tread from the upper bound
    # down to the lower one as difficulty rises, hence (hard, easy).
    "open_stairs": open_stairs(
      step_height_range=(0.02, MAX_BLIND_OBSTACLE_HEIGHT),
      step_width_range=(0.40, 0.75),
      platform_width=1.5,
      inverted=False,
    ),
    # Pyramid of randomly sized steps; the generator halves the height at
    # difficulty 0, so the bounds are the full-difficulty heights.
    "random_stairs": random_stairs(
      step_width=0.5,
      step_height_range=(0.03, MAX_BLIND_OBSTACLE_HEIGHT),
      platform_width=1.5,
    ),
    # Tiles that tilt and step; both scale with difficulty.
    # The preset frames the tiles with a ring whose top sits 0.2 m above the
    # patch floor; entering from a neighbouring cell would mean climbing a wall
    # taller than anything else in this set, so the frame is dropped to floor
    # level and the tiles themselves keep the +/- height_range wobble.
    "tilted_grid": tilted_grid(
      grid_width=0.5,
      tilt_range_deg=15.0,
      height_range=0.12,
      platform_width=1.5,
      border_height=0.0,
    ),
    # Smooth rolling heightfield, the hilly counterpart of random_rough.  Smooth
    # slopes are easier per unit height than a step, so it stops slightly lower.
    "perlin_noise": perlin_noise(
      height_range=(0.0, 0.12),
      octaves=4,
      persistence=0.3,
      scale=10.0,
      resolution=0.1,
      border_width=0.25,
    ),
  }


def add_blind_rough_terrains(cfg: Any) -> Any:
  """Install the blind-friendly terrain mix into a rough-terrain config."""
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("This task has no terrain generator")
  # mjlab hands out a shallow copy of the module-level rough preset, so the
  # sub-terrain dict must be copied before it is mutated.
  generator = deepcopy(terrain.terrain_generator)
  generator.sub_terrains.update(blind_rough_terrains())

  missing = set(BLIND_ROUGH_PROPORTIONS) - set(generator.sub_terrains)
  if missing:
    raise ValueError(f"Terrain config is missing {sorted(missing)}")
  for name, proportion in BLIND_ROUGH_PROPORTIONS.items():
    generator.sub_terrains[name] = replace(
      generator.sub_terrains[name], proportion=proportion
    )

  total = sum(sub.proportion for sub in generator.sub_terrains.values())
  if abs(total - 1.0) > 1e-6:
    raise ValueError(f"Terrain proportions must sum to 1.0, got {total}")
  terrain.terrain_generator = generator
  return cfg


@dataclass(kw_only=True)
class CurriculumRandomRoughTerrainCfg(HfRandomUniformTerrainCfg):
  """Scale noise about its midpoint; difficulty=1 preserves the source terrain.

  mjlab's uniform-noise generator deliberately ignores difficulty. Scale its
  input range explicitly so the curriculum can actually offer easier terrain.
  """

  minimum_amplitude_fraction: float = 0.125

  def noise_cfg(self, difficulty: float) -> HfRandomUniformTerrainCfg:
    if not 0 < self.minimum_amplitude_fraction <= 1:
      raise ValueError("minimum_amplitude_fraction must be in (0, 1]")
    difficulty = max(0.0, min(1.0, float(difficulty)))
    factor = self.minimum_amplitude_fraction + (
      1 - self.minimum_amplitude_fraction
    ) * difficulty
    midpoint = sum(self.noise_range) / 2
    half_range = (self.noise_range[1] - self.noise_range[0]) / 2 * factor
    scale = self.vertical_scale
    low = round((midpoint - half_range) / scale) * scale
    high = round((midpoint + half_range) / scale) * scale
    step = max(1, round(self.noise_step * factor / scale)) * scale
    return replace(self, noise_range=(low, high), noise_step=step)

  def function(self, difficulty, spec, rng):
    return HfRandomUniformTerrainCfg.function(
      self.noise_cfg(difficulty), difficulty, spec, rng
    )
