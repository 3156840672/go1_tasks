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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Go2-Velocity-HIM")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--device", default="cuda:0")
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
    help="Override the initial terrain level (rough tasks only).",
  )
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  # Play configs keep the training contact allocation; evaluation on a fresh
  # random terrain can exceed it.
  if getattr(env_cfg.sim, "nconmax", None) is not None:
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax), 256)
  if args.terrain_level is not None:
    terrain = env_cfg.scene.terrain
    if terrain is None or terrain.terrain_generator is None:
      raise SystemExit("This task has no terrain generator")
    terrain.terrain_generator.max_init_terrain_level = args.terrain_level
    terrain.terrain_generator.curriculum = True

  fixed_command = (
    torch.tensor(args.command, dtype=torch.float32) if args.command else None
  )
  if fixed_command is not None:
    # Constant command: no resampling, no heading controller rewriting w_z.
    command_cfg = env_cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
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
  if fixed_command is not None:
    term = env.unwrapped.command_manager.get_term("twist")
    term.vel_command_b[:] = fixed_command.to(term.vel_command_b.device)
  errors_xy, errors_yaw, speeds, heights, falls, steps = [], [], [], [], 0, 0
  command_levels = []
  with torch.inference_mode():
    for _ in range(args.steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions.to(env.device))
      obs = obs.to(args.device)
      done = dones.reshape(-1) > 0
      if fixed_command is not None:
        term = env.unwrapped.command_manager.get_term("twist")
        term.vel_command_b[:] = fixed_command.to(term.vel_command_b.device)
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
    print(f"terrain level   : {mean(command_levels):.2f}")
  print(f"terminated      : {100.0 * falls / max(steps, 1):.2f} % of env-steps")


if __name__ == "__main__":
  main()
