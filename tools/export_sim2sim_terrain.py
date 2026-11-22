"""Export the Go2 training terrain columns as standalone MJCF for sim2sim.

The sim2sim viewer (``sim2sim/go2_sim.py``) only needs mujoco + numpy, so it
must not import the training stack.  This tool does the importing once and
writes, per terrain column, a self-contained MJCF plus the texture PNGs it
references (heightfield elevations are inlined by MuJoCo's ``to_xml``).

Each exported file also carries one invisible site per difficulty level,
``spawn_level_0`` .. ``spawn_level_9``, holding that patch's spawn position, so
the runtime can place the robot on a chosen level without re-deriving the grid.

Usage:
  .venv/bin/python tools/export_sim2sim_terrain.py                  # all 14
  .venv/bin/python tools/export_sim2sim_terrain.py --types flat open_stairs
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from importlib.metadata import version

import mujoco
import numpy as np
from PIL import Image

import go2_tasks  # noqa: F401  (registers the Go2 tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator

TASK = "Go2-Velocity-HIM"
DEFAULT_OUT = "sim2sim/terrain_assets/go2_rough"
LEVEL_PREFIX = "level_"


def _dump_textures(spec: mujoco.MjSpec, texture_dir: str) -> None:
  """Write in-memory textures to PNG; MuJoCo cannot serialize buffers."""
  os.makedirs(texture_dir, exist_ok=True)
  for texture in spec.textures:
    if not texture.data:
      continue
    pixels = np.frombuffer(texture.data, dtype=np.uint8).reshape(
      texture.height, texture.width, texture.nchannel
    )
    path = os.path.abspath(os.path.join(texture_dir, f"{texture.name}.png"))
    Image.fromarray(pixels).save(path)
    texture.file = path
    texture.data = b""


def _write_spec(spec: mujoco.MjSpec, path: str) -> str:
  path = os.path.abspath(path)
  # MuJoCo needs real paths while serializing the spec, but the saved bundle
  # should resolve textures relative to its XML file after it is moved.
  xml = spec.to_xml().replace(f'file="{os.path.dirname(path)}/', 'file="')
  with open(path, "w", encoding="utf-8") as handle:
    handle.write(xml)
  return path


def export_column(name: str, sub_cfg, out_dir: os.PathLike, seed: int) -> str:
  """Write ``<out_dir>/<name>.xml`` and its textures; return the XML path."""
  env_cfg = load_env_cfg(TASK)
  generator_cfg = copy.deepcopy(env_cfg.scene.terrain.terrain_generator)
  generator_cfg.sub_terrains = {name: copy.deepcopy(sub_cfg)}
  generator_cfg.sub_terrains[name].proportion = 1.0
  generator_cfg.seed = seed
  # One column at a time keeps each sim2sim scene small (~1/14 of the grid).
  generator = TerrainGenerator(generator_cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  _dump_textures(spec, os.path.join(out_dir, name))

  # Invisible spawn marker per difficulty level.
  origins = generator.terrain_origins[:, 0]
  for row in range(generator_cfg.num_rows):
    spec.worldbody.add_site(
      name=f"spawn_level_{row}",
      pos=origins[row].tolist(),
      size=[0.01, 0.01, 0.01],
      rgba=[0.0, 0.0, 0.0, 0.0],
    )
  return _write_spec(spec, os.path.join(out_dir, f"{name}.xml"))


def _repack_rectangle(
  spec: mujoco.MjSpec, origins: np.ndarray, shape: tuple[int, int], size: float
) -> tuple[np.ndarray, list]:
  """Fold the single-row strip of patches into a ``shape`` rectangle.

  The generator lays every column out along one axis; shifting each patch's
  geoms by a constant offset turns the 112 m strip into a compact block
  without touching the terrain data itself.
  """
  rows, cols = shape
  count = len(origins)
  offsets = np.zeros((count, 2))
  for index in range(count):
    row, col = divmod(index, cols)
    target = np.array(
      [(row - (rows - 1) / 2.0) * size, (col - (cols - 1) / 2.0) * size]
    )
    offsets[index] = target - origins[index, :2]

  body = next((item for item in spec.worldbody.bodies if item.name == "terrain"), None)
  if body is None:
    raise ValueError("Exported terrain has no `terrain` body")
  # The generator's rim frame sits around the un-packed strip; drop it and lay a
  # fresh one around the rectangle so nothing sticks out of the block.
  rim = [
    geom
    for geom in body.geoms
    if max(float(geom.size[0]), float(geom.size[1])) > size
  ]
  for geom in rim:
    spec.delete(geom)
  return offsets, rim


def _shift_patches(spec: mujoco.MjSpec, offsets: np.ndarray, count: int, size: float) -> None:
  body = spec.body("terrain")
  for geom in body.geoms:
    position = np.asarray(geom.pos, dtype=float)
    index = int(round(position[1] / size + (count - 1) / 2.0))
    index = max(0, min(count - 1, index))
    geom.pos = (position + [*offsets[index], 0.0]).tolist()


def _level_seams(
  spec: mujoco.MjSpec,
  origins: np.ndarray,
  offsets: np.ndarray,
  size: float,
  tolerance: float = 0.02,
) -> np.ndarray:
  """Sink a patch whose whole rim is a wall, so a neighbour can walk in.

  Some presets put a raised frame around a patch (``tilted_grid`` tops its
  border at 0.2 m).  Entering from the next cell then means climbing a vertical
  wall that is taller than anything else in the set.  Patches whose rim is
  uniformly above the floor get dropped by that height; a patch that already
  reaches 0 somewhere is left alone, since that is just its shape.
  """
  centers = np.asarray(origins)[:, :2] + np.asarray(offsets)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = spec.body("terrain")
  radius = size / 2.0 - 0.05
  sinks = np.zeros(len(centers))
  for index, center in enumerate(centers):
    heights = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
      start = np.array(
        [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), 4.0]
      )
      geomid = np.zeros(1, dtype=np.int32)
      distance = mujoco.mj_ray(
        model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid
      )
      heights.append(4.0 - distance if distance >= 0.0 else 0.0)
    heights = np.asarray(heights)
    step = float(np.median(heights))
    if step <= tolerance or float(heights.min()) <= tolerance:
      continue
    sinks[index] = step
    for geom in body.geoms:
      position = np.asarray(geom.pos, dtype=float)
      if np.abs(position[:2] - center).max() <= size / 2.0 + 1e-6:
        geom.pos = (position - [0.0, 0.0, step]).tolist()
  return sinks


def export_level(
  level: int,
  out_dir: os.PathLike,
  seed: int,
  shape: tuple[int, int] | None = None,
) -> str:
  """Export every terrain column at one curriculum level, packed in a block.

  Training draws a row's difficulty uniformly from ``[L/10, (L+1)/10)``; we pin
  the generator to the row's midpoint so the map is representative of that
  level.  The 14 patches are folded into a ``shape`` rectangle (2 x 7 by
  default) so the scene stays compact instead of one 112 m strip.
  """
  env_cfg = load_env_cfg(TASK)
  generator_cfg = copy.deepcopy(env_cfg.scene.terrain.terrain_generator)
  rows = generator_cfg.num_rows
  midpoint = (level + 0.5) / rows
  generator_cfg.num_rows = 1
  generator_cfg.difficulty_range = (midpoint, midpoint)
  columns = list(generator_cfg.sub_terrains)
  if shape is None:
    cols = math.ceil(math.sqrt(len(columns)))
    shape = (math.ceil(len(columns) / cols), cols)
  # Fill the leftover cells with flat patches so the block has no holes.
  for index in range(shape[0] * shape[1] - len(columns)):
    generator_cfg.sub_terrains[f"flat_fill_{index + 1}"] = copy.deepcopy(
      generator_cfg.sub_terrains["flat"]
    )
  generator_cfg.seed = seed
  generator = TerrainGenerator(generator_cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  columns = list(generator_cfg.sub_terrains)
  origins = generator.terrain_origins[0]
  offsets, _ = _repack_rectangle(spec, origins, shape, generator_cfg.size[0])
  _shift_patches(spec, offsets, len(columns), generator_cfg.size[0])
  sinks = _level_seams(spec, origins, offsets, generator_cfg.size[0])
  _dump_textures(spec, os.path.join(out_dir, f"level_{level}"))
  for index, name in enumerate(columns):
    position = origins[index].copy()
    position[:2] += offsets[index]
    position[2] -= sinks[index]
    spec.worldbody.add_site(
      name=f"spawn_{name}",
      pos=position.tolist(),
      size=[0.01, 0.01, 0.01],
      rgba=[0.0, 0.0, 0.0, 0.0],
    )
  return _write_spec(spec, os.path.join(out_dir, f"level_{level}.xml"))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", default=DEFAULT_OUT)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--types",
    nargs="*",
    default=None,
    help="terrain columns to export (default: every column of the task)",
  )
  parser.add_argument(
    "--shape",
    default=None,
    help="rectangle for the all-terrain maps, e.g. 4x4 (default: square-ish)",
  )
  parser.add_argument(
    "--levels",
    nargs="*",
    type=int,
    default=None,
    help="curriculum levels for the all-terrain maps (default: 0-9)",
  )
  args = parser.parse_args()

  env_cfg = load_env_cfg(TASK)
  sub_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
  names = args.types or list(sub_terrains)
  os.makedirs(args.out, exist_ok=True)

  for name in names:
    if name not in sub_terrains:
      raise SystemExit(f"Unknown terrain {name!r}; choose from {list(sub_terrains)}")
    path = export_column(name, sub_terrains[name], args.out, args.seed)
    size_mb = os.path.getsize(path) / 1e6
    print(f"{name:22s} -> {path} ({size_mb:.2f} MB)")

  levels = (
    range(env_cfg.scene.terrain.terrain_generator.num_rows)
    if args.levels is None
    else args.levels
  )
  shape = None
  if args.shape:
    rows_text, cols_text = args.shape.lower().split("x")
    shape = (int(rows_text), int(cols_text))
  for level in levels:
    path = export_level(level, args.out, args.seed, shape)
    print(f"level_{level:<16} -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)")

  manifest = {
    "mjlab_version": version("mjlab"),
    "seed": args.seed,
    "columns": list(sub_terrains),
    "rows": env_cfg.scene.terrain.terrain_generator.num_rows,
    "levels": list(levels),
    "shape": list(
      shape
      or (
        math.ceil(len(sub_terrains) / math.ceil(math.sqrt(len(sub_terrains)))),
        math.ceil(math.sqrt(len(sub_terrains))),
      )
    ),
  }
  with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")


if __name__ == "__main__":
  main()
