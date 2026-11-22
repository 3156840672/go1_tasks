"""Render the Go2 rough-terrain curriculum to PNGs (headless, OSMesa).

Produces two images under ``docs/images/`` (created on demand):

* ``go2_rough_terrain_grid.png``   - the whole 14 x 10 grid in one view.
* ``go2_rough_terrain_level9.png`` - one tile per terrain at difficulty 9, so the
  hardest version of every column can be compared side by side.

Usage:
  MUJOCO_GL=osmesa .venv/bin/python tools/render_go2_terrain.py
"""

from __future__ import annotations

import copy
import math
import os

import matplotlib
import mujoco
import numpy as np
from PIL import Image

import go2_tasks  # noqa: F401  (registers the Go2 tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator

TASK = "Go2-Velocity-HIM"
OUT_DIR = "docs/images"
NEW_TERRAINS = (
  "discrete_obstacles",
  "box_random_grid",
  "random_spread_boxes",
  "open_stairs",
  "random_stairs",
  "tilted_grid",
  "perlin_noise",
)
FOVY = 45.0
IMAGE_WIDTH, IMAGE_HEIGHT = 1800, 1100


def add_camera(spec, name, center, distance, elevation_deg, azimuth_deg):
  """Camera ``distance`` from ``center``, pitched down and rotated by azimuth."""
  elevation = math.radians(elevation_deg)
  azimuth = math.radians(azimuth_deg)
  # Unit vector pointing from the target towards the camera.
  offset = np.array(
    [
      math.cos(elevation) * math.cos(azimuth),
      math.cos(elevation) * math.sin(azimuth),
      math.sin(elevation),
    ]
  )
  x_axis = np.cross(np.array([0.0, 0.0, 1.0]), offset)
  x_axis /= np.linalg.norm(x_axis)
  y_axis = np.cross(offset, x_axis)
  camera = spec.worldbody.add_camera(
    name=name,
    pos=(np.asarray(center, dtype=float) + distance * offset).tolist(),
    xyaxes=x_axis.tolist() + y_axis.tolist(),
  )
  camera.fovy = FOVY
  return np.asarray(center, dtype=float) + distance * offset, x_axis, y_axis, offset


def project(point, camera):
  """Project a world point into pixel coordinates of a rendered image."""
  position, x_axis, y_axis, z_axis = camera
  vector = np.asarray(point, dtype=float) - position
  depth = -float(vector @ z_axis)
  if depth <= 0.0:
    return None
  tan_y = math.tan(math.radians(FOVY) / 2.0)
  tan_x = tan_y * (IMAGE_WIDTH / IMAGE_HEIGHT)
  return (
    (float(vector @ x_axis) / depth / tan_x + 1.0) * 0.5 * IMAGE_WIDTH,
    (1.0 - float(vector @ y_axis) / depth / tan_y) * 0.5 * IMAGE_HEIGHT,
  )


def fit_distance(span_x: float, span_y: float, aspect: float, margin=1.08) -> float:
  half = math.tan(math.radians(FOVY) / 2.0)
  return max(span_x / (2 * half * aspect), span_y / (2 * half)) * margin


def compile_generator(generator_cfg):
  generator = TerrainGenerator(generator_cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  return generator, spec


def make_renderer(spec, width, height):
  model = spec.compile()
  # OSMesa defaults to a 640x480 offscreen framebuffer.
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = height
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  renderer = mujoco.Renderer(model, height, width)
  return model, data, renderer


def render(renderer, data, camera):
  renderer.update_scene(data, camera=camera)
  return renderer.render()


def main() -> None:
  os.makedirs(OUT_DIR, exist_ok=True)
  env_cfg = load_env_cfg(TASK)
  base_cfg = env_cfg.scene.terrain.terrain_generator

  # --- Whole grid, straight down -------------------------------------------
  generator, spec = compile_generator(base_cfg)
  origins = generator.terrain_origins
  patch = base_cfg.size[0]
  center = origins.mean(axis=(0, 1))
  span_x = origins[:, :, 0].max() - origins[:, :, 0].min() + patch
  span_y = origins[:, :, 1].max() - origins[:, :, 1].min() + patch
  camera = add_camera(
    spec,
    "overview",
    (center[0], center[1], 0.0),
    distance=fit_distance(span_x, span_y, IMAGE_WIDTH / IMAGE_HEIGHT),
    elevation_deg=89.0,
    azimuth_deg=0.0,
  )
  _, data, renderer = make_renderer(spec, IMAGE_WIDTH, IMAGE_HEIGHT)
  plan = render(renderer, data, "overview")
  renderer.close()

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  # Crop the figure to the grid itself (the render includes the outer border).
  half = patch / 2.0
  corners = [
    (origins[:, :, 0].min() - half, origins[:, :, 1].min() - half),
    (origins[:, :, 0].max() + half, origins[:, :, 1].min() - half),
    (origins[:, :, 0].min() - half, origins[:, :, 1].max() + half),
    (origins[:, :, 0].max() + half, origins[:, :, 1].max() + half),
  ]
  projected = [project((x, y, 0.0), camera) for x, y in corners]
  xs = [point[0] for point in projected if point]
  ys = [point[1] for point in projected if point]
  pad = 8.0
  x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

  figure, axis = plt.subplots(
    figsize=((x1 - x0) / 140, (y1 - y0) / 140 + 0.5)
  )
  axis.imshow(plan)
  axis.set_xlim(x0, x1)
  axis.set_ylim(y1, y0)
  axis.axis("off")
  names = list(base_cfg.sub_terrains)
  for column, name in enumerate(names):
    label_at = project(origins[0, column] + (-patch * 0.45, 0.0, 0.0), camera)
    if label_at is not None:
      axis.text(
        label_at[0], y0 + 4.0, name, rotation=90, fontsize=10,
        ha="right", va="top", color="black",
      )
  for row in range(base_cfg.num_rows):
    label_at = project(origins[row, 0] + (0.0, -patch * 0.5, 0.0), camera)
    if label_at is not None:
      axis.text(
        x0 + 4.0, label_at[1], f"{row}", fontsize=10,
        ha="left", va="center", color="black",
      )
  axis.set_title(
    f"Go2 rough curriculum — {len(names)} terrains x "
    f"{base_cfg.num_rows} levels (0 = easiest, at the top)",
    fontsize=12,
  )
  figure.tight_layout()
  figure.savefig(
    f"{OUT_DIR}/go2_rough_terrain_grid.png", dpi=150, facecolor="white"
  )
  print(f"grid: {len(names)} columns x {base_cfg.num_rows} rows")

  # --- Every terrain at difficulty 9, one cleanly framed tile each ----------
  tiles = []
  for name in base_cfg.sub_terrains:
    column_cfg = copy.deepcopy(base_cfg)
    column_cfg.sub_terrains = {name: copy.deepcopy(base_cfg.sub_terrains[name])}
    column_cfg.sub_terrains[name].proportion = 1.0
    column_generator, column_spec = compile_generator(column_cfg)
    patch_center = column_generator.terrain_origins[base_cfg.num_rows - 1, 0]
    add_camera(
      column_spec,
      "tile",
      (patch_center[0], patch_center[1], 0.0),
      distance=fit_distance(patch * 1.25, patch * 1.25, 520 / 380),
      elevation_deg=36.0,
      azimuth_deg=210.0,
    )
    _, column_data, column_renderer = make_renderer(column_spec, 520, 380)
    tiles.append((name, render(column_renderer, column_data, "tile")))
    column_renderer.close()
    print(f"  {name}: level-9 tile")

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(3, 5, figsize=(16, 8.4))
  for axis in axes.ravel():
    axis.axis("off")
  for axis, (name, image) in zip(axes.ravel(), tiles):
    axis.imshow(image)
    axis.set_title(name, fontsize=12)
  fig.suptitle("Go2 rough curriculum — every terrain at difficulty 9", fontsize=14)
  fig.tight_layout()
  fig.savefig(
    f"{OUT_DIR}/go2_rough_terrain_level9.png", dpi=120, facecolor="white"
  )
  print("level-9 montage saved")


if __name__ == "__main__":
  main()