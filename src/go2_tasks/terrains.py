"""Go2 terrain curricula without changing the robot's physical parameters."""

from dataclasses import dataclass, replace

from mjlab.terrains import HfRandomUniformTerrainCfg


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
