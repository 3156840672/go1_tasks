"""Independent MuJoCo Go2 policy sim2sim runner.

Runs Go2 Flat, AMP, CTS, HIM and DreamWaQ policies on the
generated terrains and on the exact training-terrain export.  The scene comes
from the same Go2 MJCF the training task loads; neither mjlab nor rsl_rl is
imported.
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
from quadruped_sim.robots.go2 import COMMAND_LIMITS, Go2Simulator
from quadruped_sim.viewer import KeyboardCommand, run_viewer

BANNER = (
  "Keys: 1 policy, 0 passive, R reset, WASD/QE velocity, Space zero",
  "Gamepad: A policy, B passive, X reset; left stick velocity, right stick X yaw",
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--policy",
    type=Path,
    help=(
      "45-D AMP or blind Rough (Rough/Rough AMP) or 48-D Flat model_*.pt checkpoint"
    ),
  )
  parser.add_argument(
    "--student",
    type=Path,
    help="Paper AMP LSTM student_*.pt checkpoint",
  )
  parser.add_argument(
    "--him",
    type=Path,
    help="HIM (Go2-Velocity-HIM) model_*.pt checkpoint, or `latest`",
  )
  parser.add_argument(
    "--dreamwaq",
    type=Path,
    help="DreamWaQ model_*.pt checkpoint or ONNX policy export",
  )
  parser.add_argument(
    "--cts",
    type=Path,
    help=(
      "CTS student: the '<run>.onnx' export or the run's model_*.pt "
      "(the checkpoint stores the student encoder, so both run the same policy)"
    ),
  )
  add_run_args(parser)
  add_terrain_args(parser, default="flat")
  add_view_args(parser, camera="track")
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  check_common(parser, args)
  check_command(parser, args.command, COMMAND_LIMITS)
  sources = (args.policy, args.student, args.him, args.cts, args.dreamwaq)
  if sum(path is not None for path in sources) > 1:
    parser.error("Use one of --policy, --student, --him, --cts or --dreamwaq")
  sim = Go2Simulator(
    no_shadows=args.no_shadows,
    policy=args.policy,
    terrain=args.terrain,
    terrain_seed=args.terrain_seed,
    terrain_type=args.terrain_type,
    terrain_level=args.terrain_level,
    terrain_column=args.terrain_column,
    student=args.student,
    him=args.him,
    dreamwaq=args.dreamwaq,
    cts=args.cts,
  )
  keyboard = KeyboardCommand(args.command, COMMAND_LIMITS)
  sim.command[:] = keyboard.values
  if args.headless:
    if sim.policy is not None:
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
    lookat=(0.0, 0.0, 0.3),
    joystick_path=args.joystick,
    banner=BANNER,
  )


if __name__ == "__main__":
  main()
