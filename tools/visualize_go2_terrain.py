"""Interactive view of the terrain curriculum the Go2 tasks actually train on.

Unlike ``mjlab.scripts.visualize_terrain`` (which renders mjlab's preset gallery),
this builds the real generator from the task config: 14 columns, one per terrain
type, and 10 rows of increasing difficulty, in the exact mix the policy sees.

Usage:
  .venv/bin/python tools/visualize_go2_terrain.py                 # :8081
  .venv/bin/python tools/visualize_go2_terrain.py --port 8082
"""

from __future__ import annotations

import argparse
import time

import mujoco
import viser
from mjviser.conversions import merge_geoms

import go2_tasks  # noqa: F401  (registers the Go2 tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator


def build_terrain_mesh(task: str):
  """Compile the task's terrain grid and return it as one merged mesh."""
  env_cfg = load_env_cfg(task)
  terrain = env_cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise SystemExit(f"{task} has no terrain generator")

  generator = TerrainGenerator(terrain.terrain_generator)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
  if not geom_ids:
    raise SystemExit("No terrain geoms found")
  return generator, merge_geoms(model, geom_ids), len(geom_ids)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Go2-Velocity-HIM")
  parser.add_argument("--port", type=int, default=8081)
  args = parser.parse_args()

  start = time.time()
  generator, mesh, num_geoms = build_terrain_mesh(args.task)
  print(f"{num_geoms} terrain geoms -> {len(mesh.faces):,} faces "
        f"({time.time() - start:.1f}s)")

  server = viser.ViserServer(port=args.port)
  server.scene.add_mesh_trimesh("/terrain", mesh)

  sub_terrains = list(generator.cfg.sub_terrains)
  origins = generator.terrain_origins
  for col, name in enumerate(sub_terrains):
    position = origins[0, col]
    server.scene.add_label(
      f"/labels/col_{col}",
      text=name,
      position=(position[0], position[1], position[2] + 1.0),
    )
  for row in range(generator.cfg.num_rows):
    position = origins[row, 0]
    server.scene.add_label(
      f"/labels/row_{row}",
      text=f"level {row}",
      position=(position[0], position[1], position[2] + 1.6),
    )

  server.gui.add_markdown(
    f"**{args.task}**: {len(sub_terrains)} 种地形 × "
    f"{generator.cfg.num_rows} 级难度\n\n"
    "沿 X 轴（行）是难度递增，沿 Y 轴（列）是地形种类。"
  )
  print(f"Viser terrain viewer on http://127.0.0.1:{args.port}/")
  while True:
    time.sleep(1.0)


if __name__ == "__main__":
  main()