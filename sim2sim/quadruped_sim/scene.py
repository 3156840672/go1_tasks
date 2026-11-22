"""Scene fix-ups shared by every robot that loads the exported terrain map.

The exported map is a 4x4 curriculum grid carrying its own light; the stock
Go1/Go2 MJCFs assume a small, evenly lit scene.  These helpers keep the map
looking and behaving the same whichever robot is dropped on it, so the terrain
flags of the three runners are interchangeable.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

#: Terrain mode that loads the exported training map instead of generating one.
EXPORTED_TERRAIN = "go2_rough"
#: The training scene's ``stat.meansize``; the stock value inflates every marker.
VISUAL_MEANSIZE = 0.03
#: Shadow map half-size in metres; the map's extent would spread it over 250 m.
SHADOW_RADIUS = 4.0
#: Clip planes in metres, converted to MuJoCo's extent-relative units.
VIEW_NEAR = 0.05
VIEW_FAR = 400.0
#: The exported grid is ~250 m wide, so it needs its own near/far pair.
EXPORTED_VIEW_NEAR = 0.2
EXPORTED_VIEW_FAR = 80.0
#: Bodies added by the export; they must not count towards the robot's averages.
TERRAIN_BODIES = frozenset({"terrain", "shadow_light"})


def move_terrain_light_to_mocap(root: ET.Element) -> None:
  """Give the exported map's light a local, movable shadow projection.

  Called on the terrain XML before it is merged into a robot MJCF, so a robot
  that ships its own light keeps it.
  """
  world = root.find("worldbody")
  if world is None:
    raise ValueError("Terrain MJCF has no worldbody")
  lights = [
    (parent, light) for parent in world.iter() for light in parent.findall("light")
  ]
  if len(lights) != 1:
    raise ValueError(f"Expected one exported terrain light, found {len(lights)}")
  parent, light = lights[0]
  parent.remove(light)
  light.set("pos", "0 0 8")
  follow = ET.SubElement(world, "body", name="shadow_light", mocap="true")
  follow.append(light)


def fit_view_range(
  model: mujoco.MjModel,
  near: float = VIEW_NEAR,
  far: float = VIEW_FAR,
  shadow_radius: float | None = None,
) -> None:
  """Keep the camera clip planes in metres, not tied to the scene size.

  MuJoCo scales ``frustum_near``/``frustum_far`` by ``stat.extent``, which a
  wide terrain map inflates to ~250 m.  The near plane then sits ~2.5 m from
  the camera and slices the robot in half as soon as the follow camera gets
  close.  ``stat.extent`` itself stays untouched; the optional shadow radius
  changes only the light's render projection.
  """
  model.vis.map.znear = near / model.stat.extent
  model.vis.map.zfar = far / model.stat.extent
  if shadow_radius is not None:
    model.vis.map.shadowclip = shadow_radius / model.stat.extent


def finish_exported_scene(model: mujoco.MjModel) -> None:
  """Restore the per-robot statistics that the whole exported grid distorts."""
  model.stat.meansize = VISUAL_MEANSIZE
  # The map adds a massless terrain body; leaving it in meanmass changes
  # force-arrow length and perturbation strength.
  robot_masses = [
    model.body_mass[index]
    for index in range(1, model.nbody)
    if model.body(index).name not in TERRAIN_BODIES
  ]
  model.stat.meanmass = float(np.mean(robot_masses))
  fit_view_range(model, EXPORTED_VIEW_NEAR, EXPORTED_VIEW_FAR, SHADOW_RADIUS)


def spawn_position(
  model: mujoco.MjModel,
  terrain: str,
  terrain_type: str | None = None,
  terrain_level: int = 0,
  terrain_column: str | None = None,
) -> np.ndarray:
  """Start position of the robot: the map's spawn site, else the origin."""
  if terrain != EXPORTED_TERRAIN:
    return np.zeros(3)
  if terrain_type == "all":
    site_name = f"spawn_{terrain_column or 'flat'}"
  else:
    site_name = f"spawn_level_{terrain_level}"
  site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  if site < 0:
    raise ValueError(f"Terrain has no spawn site {site_name!r}")
  return np.array(model.site_pos[site], dtype=np.float64)


def shadow_light_mocap(model: mujoco.MjModel) -> int:
  """Mocap id of the exported map's light body, or -1 when absent."""
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shadow_light")
  return -1 if body < 0 else int(model.body_mocapid[body])


def follow_shadow_light(data: mujoco.MjData, mocap: int) -> None:
  """Keep the map's light above the robot instead of the map centre."""
  if mocap >= 0:
    data.mocap_pos[mocap] = [data.qpos[0], data.qpos[1], 0.0]
