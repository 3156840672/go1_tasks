"""Small standalone MuJoCo terrains for Go1 sim2sim."""

from __future__ import annotations

import copy
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

TERRAIN_SIZE = 8.0
GRID_SIZE = 161
TERRAIN_MODES = ("flat", "rough", "stairs", "rl_sar")


def _chunk(name: bytes, payload: bytes) -> bytes:
  return (
    struct.pack(">I", len(payload))
    + name
    + payload
    + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
  )


def _write_height_png(path: Path, heights: np.ndarray) -> float:
  scale = float(heights.max())
  if scale <= 0:
    raise ValueError("Heightfield must contain positive heights")
  pixels = np.rint(heights / scale * 255).astype(np.uint8)
  rows, cols = pixels.shape
  raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
  ihdr = struct.pack(">IIBBBBB", cols, rows, 8, 0, 0, 0, 0)
  path.write_bytes(
    b"\x89PNG\r\n\x1a\n"
    + _chunk(b"IHDR", ihdr)
    + _chunk(b"IDAT", zlib.compress(raw))
    + _chunk(b"IEND", b"")
  )
  return scale


def generate_heights(kind: str, seed: int) -> np.ndarray:
  if kind not in ("rough", "stairs"):
    raise ValueError("Expected generated terrain: rough or stairs")
  coords = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, GRID_SIZE)
  x, y = np.meshgrid(coords, coords)
  if kind == "rough":
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 6, size=(41, 41)) * 0.02
    heights = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)[:GRID_SIZE, :GRID_SIZE]
  else:
    step = np.floor(np.maximum(np.abs(x) - 0.8, 0.0) / 0.4)
    heights = np.broadcast_to(0.05 * np.minimum(step, 4), x.shape).copy()
  heights[(np.abs(x) < 0.7) & (np.abs(y) < 0.7)] = 0.0
  return heights.astype(np.float32)


def _add_rl_sar_terrain(root: ET.Element) -> None:
  source_dir = Path(__file__).resolve().parent / "terrain_assets"
  source = ET.parse(source_dir / "rl_sar_scene_terrain.xml").getroot()
  target_asset = root.find("asset")
  target_world = root.find("worldbody")
  source_asset = source.find("asset")
  source_world = source.find("worldbody")
  if None in (target_asset, target_world, source_asset, source_world):
    raise ValueError("Invalid Go1 or rl_sar terrain MJCF")

  for item in source_asset:
    item = copy.deepcopy(item)
    if item.tag == "hfield":
      file_name = Path(item.get("file", "")).name
      item.set("file", str(source_dir / file_name))
    target_asset.append(item)
  for item in source_world:
    target_world.append(copy.deepcopy(item))


def add_terrain(root: ET.Element, directory: Path, kind: str, seed: int) -> None:
  if kind not in TERRAIN_MODES:
    raise ValueError(f"Unknown terrain {kind!r}; choose from {TERRAIN_MODES}")
  world = root.find("worldbody")
  if world is None:
    raise ValueError("Go1 MJCF has no worldbody")
  common = dict(
    name="ground", friction="1 0.005 0.0005", condim="6", rgba="0.38 0.48 0.38 1"
  )
  if kind == "flat":
    ET.SubElement(world, "geom", type="plane", size="0 0 0.1", **common)
    return
  if kind == "rl_sar":
    _add_rl_sar_terrain(root)
    return
  heights = generate_heights(kind, seed)
  png_path = directory / "terrain.png"
  height_scale = _write_height_png(png_path, heights)
  asset = root.find("asset")
  if asset is None:
    raise ValueError("Go1 MJCF has no asset section")
  ET.SubElement(
    asset,
    "hfield",
    name="ground_heightfield",
    file=str(png_path),
    size=f"{TERRAIN_SIZE / 2} {TERRAIN_SIZE / 2} {height_scale} 0.1",
  )
  ET.SubElement(world, "geom", type="hfield", hfield="ground_heightfield", **common)
