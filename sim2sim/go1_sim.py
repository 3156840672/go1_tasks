"""Independent MuJoCo Go1 RMA sim2sim runner.

Runs the RMA teacher/adapter pair or a 48-D flat-velocity policy on any of the
shared sim2sim terrains, including the exported Go2 rough map.  Only mujoco,
numpy and torch are required; neither mjlab nor rsl_rl is imported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quadruped_sim.cli import (
  add_run_args,
  add_terrain_args,
  add_view_args,
  check_command,
  check_common,
  simulate,
)
from quadruped_sim.robots.go1 import Go1Simulator
from quadruped_sim.viewer import KeyboardCommand, run_viewer

COMMAND_LIMITS = (1.0, 1.0, 0.5)
BANNER = (
  "Keys: 1 RMA, 0 passive, R reset, WASD/QE velocity, Space zero",
  "Gamepad: A RMA, B passive, X reset; left stick velocity, right stick X yaw",
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--policy", type=Path, help="48-observation model_*.pt from Go1-Velocity-Flat"
  )
  parser.add_argument("--teacher", type=Path, help="model_*.pt from stage 1")
  parser.add_argument("--adapter", type=Path, help="adapter_*.pt from stage 2")
  parser.add_argument(
    "--cts",
    type=Path,
    help=(
      "CTS student: the '<run>.onnx' export or the run's model_*.pt "
      "(the checkpoint stores the student encoder, so both run the same policy)"
    ),
  )
  add_run_args(parser)
  add_terrain_args(parser, default="rough")
  add_view_args(parser)
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  check_common(parser, args)
  check_command(parser, args.command, COMMAND_LIMITS)
  if (args.policy is not None or args.cts is not None) and (
    args.teacher is not None or args.adapter is not None
  ):
    parser.error("Use either --policy/--cts or the --teacher/--adapter pair")
  if args.policy is not None and args.cts is not None:
    parser.error("Use only one of --policy or --cts")
  if (args.teacher is None) != (args.adapter is None):
    parser.error("Both --teacher and --adapter are required together")
  sim = Go1Simulator(
    args.teacher,
    args.adapter,
    args.terrain,
    args.terrain_seed,
    policy=args.policy,
    cts=args.cts,
    terrain_type=args.terrain_type,
    terrain_level=args.terrain_level,
    terrain_column=args.terrain_column,
    no_shadows=args.no_shadows,
  )
  keyboard = KeyboardCommand(args.command, COMMAND_LIMITS)
  sim.command[:] = keyboard.values
  if args.headless:
    if (
      sim.actor is not None
      or sim.standard_policy is not None
      or sim.cts is not None
    ):
      sim.set_active(True)
    print(simulate(sim, args.steps).format(sim.command))
    return

  import glfw

  def on_key(key: int) -> None:
    if key == glfw.KEY_1:
      sim.set_active(True)
    elif key == glfw.KEY_0:
      sim.set_active(False)
    elif key == glfw.KEY_R:
      sim.reset()

  run_viewer(
    sim,
    command=keyboard,
    on_key=on_key,
    camera=args.camera,
    follow_distance=args.follow_distance,
    joystick_path=args.joystick,
    banner=BANNER,
  )


if __name__ == "__main__":
  main()
