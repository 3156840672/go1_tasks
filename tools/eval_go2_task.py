"""Headless evaluation of a Go2 velocity checkpoint in its own task env.

Unlike ``sim2sim/go2_sim.py`` this uses the training environment (same MJCF,
terrain generator, observation pipeline and command distribution), so it
reports in-distribution tracking quality. Commands are sampled by the env
exactly as during training; reset steps are excluded from the averages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def pin_terrain_level(env_cfg, level: int, num_levels: int) -> None:
  """Build one representative curriculum row and keep resets on that row."""
  terrain = env_cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("This task has no terrain generator")
  generator = terrain.terrain_generator
  if not 0 <= level < num_levels:
    raise ValueError(f"--terrain-level must be in 0..{num_levels - 1}")
  midpoint = (level + 0.5) / num_levels
  generator.num_rows = 1
  generator.difficulty_range = (midpoint, midpoint)
  generator.curriculum = True
  terrain.max_init_terrain_level = 0
  env_cfg.events.pop("randomize_terrain", None)
  env_cfg.curriculum.pop("terrain_levels", None)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Go2-Velocity-HIM")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=None, help="Fix terrain and reset sampling.")
  parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    metavar=("VX", "VY", "WZ"),
    default=None,
    help="Hold a fixed command instead of sampling from the task distribution.",
  )
  parser.add_argument(
    "--terrain-level",
    type=int,
    default=None,
    help="Pin every env to this terrain level (disables the terrain curriculum).",
  )
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  if args.seed is not None:
    env_cfg.seed = args.seed
    terrain = env_cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
      terrain.terrain_generator.seed = args.seed
  # Play configs keep the training contact allocation; evaluation on a fresh
  # random terrain can exceed it.
  if getattr(env_cfg.sim, "nconmax", None) is not None:
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax), 256)
  if args.terrain_level is not None:
    try:
      training_terrain = load_env_cfg(args.task).scene.terrain
      if training_terrain is None or training_terrain.terrain_generator is None:
        raise ValueError("This task has no terrain generator")
      pin_terrain_level(
        env_cfg, args.terrain_level, training_terrain.terrain_generator.num_rows
      )
    except ValueError as exc:
      parser.error(str(exc))

  if args.command is not None:
    # Set the sampler before initialisation so every observed frame, including
    # the first one after an auto-reset, contains the chosen command.
    vx, vy, wz = args.command
    command_cfg = env_cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.ranges.lin_vel_x = (vx, vx)
    command_cfg.ranges.lin_vel_y = (vy, vy)
    command_cfg.ranges.ang_vel_z = (wz, wz)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    command_cfg.rel_world_envs = 0.0
    if hasattr(command_cfg, "rel_turning_envs"):
      command_cfg.rel_turning_envs = 0.0
    command_cfg.resampling_time_range = (1e9, 1e9)

  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=args.device),
    clip_actions=agent_cfg.clip_actions,
  )
  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), None, args.device)
  try:
    runner.load(args.checkpoint, map_location=args.device)
  except Exception as exc:  # AMP tasks add their own optimiser groups
    print(f"[WARN] full checkpoint load failed ({exc}); loading weights only")
    runner.load(
      args.checkpoint,
      map_location=args.device,
      load_cfg={
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
    )
  policy = runner.get_inference_policy(device=args.device)

  obs = env.get_observations().to(args.device)
  errors_xy, errors_yaw, speeds, heights, falls, steps = [], [], [], [], 0, 0
  command_levels = []
  with torch.inference_mode():
    for _ in range(args.steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions.to(env.device))
      obs = obs.to(args.device)
      done = dones.reshape(-1) > 0
      command = env.unwrapped.command_manager.get_command("twist")
      velocity = env.unwrapped.scene["robot"].data.root_link_lin_vel_b
      angular_velocity = env.unwrapped.scene["robot"].data.root_link_ang_vel_b
      levels = getattr(env.unwrapped.scene.terrain, "terrain_levels", None)
      keep = ~done
      if keep.any():
        errors_xy.append(
          torch.linalg.vector_norm(command[keep, :2] - velocity[keep, :2], dim=-1)
        )
        errors_yaw.append((command[keep, 2] - angular_velocity[keep, 2]).abs())
        speeds.append(command[keep, :2].norm(dim=-1))
        heights.append(env.unwrapped.scene["robot"].data.root_link_pos_w[keep, 2])
        if levels is not None:
          command_levels.append(levels[keep].float())
      falls += int(done.sum())
      steps += len(done)

  def mean(items) -> float:
    if not items:
      return float("nan")
    return float(torch.cat(items).mean())

  print(f"task            : {args.task}")
  print(f"checkpoint      : {args.checkpoint}")
  print(f"envs x steps    : {args.num_envs} x {args.steps}")
  print(f"command         : {'fixed ' + str(args.command) if args.command else 'sampled'}")
  print(f"mean |cmd|      : {mean(speeds):.3f} m/s")
  print(f"error_vel_xy    : {mean(errors_xy):.3f} m/s")
  print(f"error_vel_yaw   : {mean(errors_yaw):.3f} rad/s")
  print(f"mean height     : {mean(heights):.3f} m")
  if command_levels:
    level = args.terrain_level if args.terrain_level is not None else mean(command_levels)
    print(f"terrain level   : {level:.2f}")
  print(f"terminated      : {100.0 * falls / max(steps, 1):.2f} % of env-steps")


if __name__ == "__main__":
  main()
