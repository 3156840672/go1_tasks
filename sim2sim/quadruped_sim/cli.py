"""Shared command-line surface of the three standalone sim2sim runners.

Every runner accepts the same terrain, run and view flags, so a command line
that works for one robot works for the others; only the policy flags differ,
because the checkpoint formats differ.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .scene import EXPORTED_TERRAIN
from .terrain import (
  TERRAIN_MODES,
  curriculum_columns,
  curriculum_terrain_types,
)
from .viewer import DECIMATION

#: Curriculum levels per column in the exported training map.
TERRAIN_LEVELS = tuple(range(10))
CAMERA_MODES = ("free", "track", "chase")


@dataclass(frozen=True)
class HeadlessResult:
  """One-line outcome of a headless run, identical for every runner."""

  steps: int
  height: float
  mean: tuple[float, float, float] | None

  def format(self, command: Sequence[float]) -> str:
    message = f"Completed {self.steps} steps; height={self.height:.3f} m"
    if self.mean is None:
      return message
    errors = [abs(mean - target) for mean, target in zip(self.mean, command)]
    return (
      f"{message}; mean vx/vy/wz="
      f"{self.mean[0]:+.3f}/{self.mean[1]:+.3f}/{self.mean[2]:+.3f}; "
      f"abs error={errors[0]:.3f}/{errors[1]:.3f}/{errors[2]:.3f}"
    )


def add_terrain_args(parser: argparse.ArgumentParser, *, default: str) -> None:
  """Terrain selectors shared by every runner, exported map included."""
  parser.add_argument(
    "--terrain",
    choices=TERRAIN_MODES,
    default=default,
    help=f"generated terrain, or {EXPORTED_TERRAIN!r} for the exported training map",
  )
  parser.add_argument(
    "--terrain-seed",
    type=int,
    default=42,
    help=f"layout seed of the generated terrains (ignored by {EXPORTED_TERRAIN})",
  )
  parser.add_argument(
    "--terrain-type",
    choices=curriculum_terrain_types() or None,
    default=None,
    help=(
      "training terrain column, or 'all' for one level of the full grid "
      f"(only with --terrain {EXPORTED_TERRAIN})"
    ),
  )
  parser.add_argument(
    "--terrain-column",
    choices=curriculum_columns() or None,
    default=None,
    help="spawn column of the full grid (--terrain-type all)",
  )
  parser.add_argument(
    "--terrain-level",
    type=int,
    default=0,
    choices=TERRAIN_LEVELS,
    metavar="0-9",
    help="curriculum level to spawn on",
  )


def add_run_args(parser: argparse.ArgumentParser) -> None:
  """Headless mode, physics step count and the initial velocity command."""
  parser.add_argument("--headless", action="store_true", help="run without a window")
  parser.add_argument(
    "--steps",
    type=int,
    default=1000,
    help=f"physics steps in headless mode ({DECIMATION} per policy step)",
  )
  parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    metavar=("VX", "VY", "WZ"),
    default=(0.0, 0.0, 0.0),
    help="initial command in m/s and rad/s",
  )


def add_view_args(parser: argparse.ArgumentParser, *, camera: str = "free") -> None:
  """Camera, shadow and joystick flags shared by the interactive runners."""
  parser.add_argument(
    "--camera",
    choices=CAMERA_MODES,
    default=camera,
    help=(
      "track: follow base_link; chase: also turn with the robot's heading; "
      "free: leave the camera where the user puts it"
    ),
  )
  parser.add_argument("--follow-distance", type=float, default=2.2)
  shadows = parser.add_mutually_exclusive_group()
  shadows.add_argument(
    "--no-shadows",
    dest="no_shadows",
    action="store_true",
    help="disable shadows",
  )
  shadows.add_argument(
    "--shadows",
    dest="no_shadows",
    action="store_false",
    help="enable shadows (the default)",
  )
  parser.add_argument(
    "--joystick",
    default="/dev/input/js0",
    help="Linux joystick device; empty disables",
  )


def check_common(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
  """Validate the flags every runner shares."""
  if args.steps <= 0:
    parser.error("--steps must be positive")


def check_command(
  parser: argparse.ArgumentParser,
  command: Sequence[float],
  limits: Sequence[float],
) -> None:
  if any(abs(value) > limit for value, limit in zip(command, limits)):
    parser.error(
      f"--command limits: |vx| <= {limits[0]}; |vy| <= {limits[1]}; "
      f"|wz| <= {limits[2]}"
    )


def simulate(sim, steps: int, decimation: int = DECIMATION) -> HeadlessResult:
  """Advance without a window, sampling the base velocity over the second half."""
  base_velocity = getattr(sim, "base_velocity", None)
  samples: list[tuple[float, float, float]] = []
  for step in range(steps):
    if step % decimation == 0:
      sim.policy_step()
    sim.physics_step()
    if base_velocity is not None and step >= steps // 2:
      linear, angular = base_velocity()
      samples.append((float(linear[0]), float(linear[1]), float(angular[2])))
  mean = None
  if samples:
    mean = tuple(float(value) for value in np.mean(samples, axis=0))
  return HeadlessResult(steps, float(sim.data.qpos[2]), mean)
