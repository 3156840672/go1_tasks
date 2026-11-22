"""Standalone MuJoCo sim2sim runner for the Go1 MoB policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from quadruped_sim.robots.go1_mob import (
  BEHAVIOUR_BOUNDS,
  BEHAVIOUR_DEFAULTS,
  GAITS,
  MobGamepadControls,
  MobSimulator,
)
from quadruped_sim.cli import (
  add_run_args,
  add_terrain_args,
  add_view_args,
  check_command,
  check_common,
  simulate,
)
from quadruped_sim.viewer import KeyboardCommand, run_viewer

COMMAND_LIMITS = (1.0, 0.6, 1.0)
#: Key -> behaviour slot adjustments, applied after the shared velocity keys.
BEHAVIOUR_KEYS = {
  "KEY_Z": (4, -0.01, "body_height"),
  "KEY_X": (4, +0.01, "body_height"),
  "KEY_C": (5, -0.01, "body_pitch"),
  "KEY_V": (5, +0.01, "body_pitch"),
  "KEY_F": (3, -0.1, "frequency"),
  "KEY_G": (3, +0.1, "frequency"),
  "KEY_B": (6, -0.01, "stance_width"),
  "KEY_N": (6, +0.01, "stance_width"),
  "KEY_H": (7, -0.01, "swing_height"),
  "KEY_J": (7, +0.01, "swing_height"),
}
BANNER = (
  "Keys: 1-4 Pronk/Trot/Bound/Pace, 0 passive, R reset, WASD/QE velocity, Space stop",
  "Z/X height, C/V pitch, F/G frequency, B/N stance width, H/J swing height",
  "Gamepad: A active, B passive, X reset, Y or D-pad right next gait, "
  "D-pad left previous gait",
  "D-pad up/down selects height, pitch, frequency, stance width or swing height; "
  "LB/RB adjusts it",
)


def _show_state(sim: MobSimulator) -> None:
  behaviour = sim.behavior
  print(
    f"vx={sim.command[0]:+.2f} vy={sim.command[1]:+.2f} wz={sim.command[2]:+.2f} "
    f"{sim.gait} freq={behaviour[3]:.1f} height={behaviour[4]:.2f} "
    f"pitch={behaviour[5]:+.2f} width={behaviour[6]:.2f} swing={behaviour[7]:.2f}"
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--policy",
    "--checkpoint",
    dest="policy",
    type=Path,
    help="Optional MoB model_*.pt checkpoint (--checkpoint is a legacy alias)",
  )
  add_run_args(parser)
  add_terrain_args(parser, default="flat")
  parser.add_argument("--gait", choices=tuple(GAITS), default="Trot")
  for name in BEHAVIOUR_BOUNDS:
    parser.add_argument(
      f"--{name.replace('_', '-')}", type=float, default=BEHAVIOUR_DEFAULTS[name]
    )
  add_view_args(parser)
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  check_common(parser, args)
  check_command(parser, args.command, COMMAND_LIMITS)
  if args.policy is not None and not args.policy.is_file():
    parser.error(f"Checkpoint not found: {args.policy}")
  for name, (low, high) in BEHAVIOUR_BOUNDS.items():
    value = getattr(args, name)
    if not low <= value <= high:
      parser.error(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
  if args.terrain != "flat":
    print(
      "Warning: MoB was trained on flat terrain; "
      "non-flat terrain is out of distribution"
    )

  sim = MobSimulator(
    args.policy,
    args.terrain,
    args.terrain_seed,
    terrain_type=args.terrain_type,
    terrain_level=args.terrain_level,
    terrain_column=args.terrain_column,
    no_shadows=args.no_shadows,
  )
  sim.command[:] = args.command
  sim.set_gait(args.gait)
  sim.behavior[3:] = [getattr(args, name) for name in BEHAVIOUR_BOUNDS]
  if args.policy is not None:
    sim.set_active(True)
  if args.headless:
    print(simulate(sim, args.steps).format(sim.command))
    return

  import glfw

  keyboard = KeyboardCommand(args.command, COMMAND_LIMITS)
  sim.command[:] = keyboard.values

  def on_key(key: int) -> bool:
    if key == glfw.KEY_0:
      sim.set_active(False)
    elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
      sim.set_gait(tuple(GAITS)[key - glfw.KEY_1])
      sim.set_active(True)
    elif key == glfw.KEY_R:
      sim.reset()
    elif not keyboard.on_key(key):
      for name, (index, delta, bound) in BEHAVIOUR_KEYS.items():
        if key == getattr(glfw, name):
          sim.adjust_behavior(index, delta, BEHAVIOUR_BOUNDS[bound])
          break
    sim.command[:] = keyboard.values
    _show_state(sim)
    return True

  gamepad = MobGamepadControls(BEHAVIOUR_BOUNDS)

  def handle_gamepad(buttons: set[int], dpad: set[str]) -> bool:
    reset = gamepad.handle(sim, buttons, dpad)
    if buttons or dpad:
      print(f"Gamepad setting: {gamepad.selected_name}")
      _show_state(sim)
    return reset

  run_viewer(
    sim,
    command=keyboard,
    on_key=on_key,
    gamepad=handle_gamepad,
    camera=args.camera,
    follow_distance=args.follow_distance,
    command_scale=COMMAND_LIMITS,
    joystick_path=args.joystick,
    banner=BANNER,
  )


if __name__ == "__main__":
  main()
