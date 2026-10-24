"""Evaluate a Go1 CTS student checkpoint on flat ground at a fixed command."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from go1_tasks import task_id


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--vx", type=float, default=0.5)
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--seconds", type=float, default=5.0)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--branch", choices=("student", "teacher"), default="student")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--stochastic", action="store_true")
  parser.add_argument("--randomized", action="store_true")
  args = parser.parse_args()

  torch.manual_seed(args.seed)

  task = task_id("cts")
  cfg = load_env_cfg(task, play=True)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.num_envs = 1
  cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
  if not args.randomized:
    cfg.events = {
      name: term
      for name, term in cfg.events.items()
      if term.mode != "startup" and name != "randomize_terrain"
    }

  command_cfg = cfg.commands["twist"]
  assert isinstance(command_cfg, UniformVelocityCommandCfg)
  command_cfg.heading_command = False
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_standing_envs = 0.0
  command_cfg.rel_forward_envs = 0.0
  command_cfg.resampling_time_range = (1000.0, 1000.0)
  command_cfg.ranges.lin_vel_x = (args.vx, args.vx)
  command_cfg.ranges.lin_vel_y = (args.vy, args.vy)
  command_cfg.ranges.ang_vel_z = (args.yaw, args.yaw)
  command_cfg.ranges.heading = None
  if args.branch == "teacher":
    cfg.observations["cts_student"].terms["mask"].params = {"teacher_only": True}

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
      str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device
    )
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
    command = env.command_manager.get_command("twist")
    assert command is not None
    actor_terms = env.observation_manager.active_terms["actor"]
    command_index = actor_terms.index("command")
    command_offset = sum(
      math.prod(shape)
      for shape in env.observation_manager.group_obs_term_dim["actor"][:command_index]
    )
    steps = round(args.seconds / env.step_dt)
    warmup = min(round(0.5 / env.step_dt), steps // 4)
    actual_x: list[float] = []
    actual_y: list[float] = []
    actual_yaw: list[float] = []
    command_x: list[float] = []
    observed_x: list[float] = []
    observed_y: list[float] = []
    observed_yaw: list[float] = []
    action_magnitude: list[float] = []
    abs_error: list[float] = []
    abs_y_error: list[float] = []
    abs_yaw_error: list[float] = []
    action_changes: list[float] = []
    shank_contact_steps = 0
    shank_force_over_10_steps = 0
    foot_contact_counts = torch.zeros(4, device=args.device)
    foot_height_sums = torch.zeros(4, device=args.device)
    previous_action: torch.Tensor | None = None
    resets = 0
    with torch.inference_mode():
      for step in range(steps):
        actor_obs = obs["actor"]
        assert isinstance(actor_obs, torch.Tensor)
        observed = actor_obs[0, command_offset : command_offset + 3]
        actions = policy(obs, stochastic_output=args.stochastic)
        obs, _, dones, _ = wrapped.step(actions)
        resets += int(dones[0].item())
        if step >= warmup and not dones[0]:
          shank_contact = env.scene["shank_ground_touch"].data
          assert shank_contact.found is not None
          assert shank_contact.force_history is not None
          shank_contact_steps += int((shank_contact.found[0] > 0).any().item())
          shank_force_over_10_steps += int(
            (torch.linalg.vector_norm(shank_contact.force_history[0], dim=-1) > 10.0)
            .any()
            .item()
          )
          foot_contact = env.scene["feet_ground_contact"].data.found
          assert foot_contact is not None
          foot_contact_counts += (foot_contact[0] > 0).float()
          foot_height_sums += env.scene["foot_height_scan"].data.heights[0]
          vx = env.scene["robot"].data.root_link_lin_vel_b[0, 0].item()
          vy = env.scene["robot"].data.root_link_lin_vel_b[0, 1].item()
          yaw = env.scene["robot"].data.root_link_ang_vel_b[0, 2].item()
          command_x.append(command[0, 0].item())
          observed_x.append(observed[0].item())
          observed_y.append(observed[1].item())
          observed_yaw.append(observed[2].item())
          action_magnitude.append(actions[0].square().mean().sqrt().item())
          actual_x.append(vx)
          actual_y.append(vy)
          actual_yaw.append(yaw)
          abs_error.append(abs(args.vx - vx))
          abs_y_error.append(abs(args.vy - vy))
          abs_yaw_error.append(abs(args.yaw - yaw))
          if previous_action is not None:
            action_changes.append(
              (actions[0] - previous_action).square().mean().sqrt().item()
            )
        previous_action = actions[0].clone()
    if not actual_x:
      raise RuntimeError("No valid samples: robot reset on every measured step")
    print(
      f"checkpoint={args.checkpoint} branch={args.branch} "
      f"stochastic={args.stochastic} "
      f"randomized={args.randomized} "
      f"command_vx={sum(command_x) / len(command_x):.3f} "
      f"command_vy={args.vy:.3f} command_yaw={args.yaw:.3f} "
      f"observed_vx_cmd={sum(observed_x) / len(observed_x):.3f} "
      f"observed_vy_cmd={sum(observed_y) / len(observed_y):.3f} "
      f"observed_yaw_cmd={sum(observed_yaw) / len(observed_yaw):.3f} "
      f"actual_vx={sum(actual_x) / len(actual_x):.3f} "
      f"actual_vy={sum(actual_y) / len(actual_y):.3f} "
      f"actual_yaw={sum(actual_yaw) / len(actual_yaw):.3f} "
      f"mae_vx={sum(abs_error) / len(abs_error):.3f} "
      f"mae_vy={sum(abs_y_error) / len(abs_y_error):.3f} "
      f"mae_yaw={sum(abs_yaw_error) / len(abs_yaw_error):.3f} "
      f"action_rms={sum(action_magnitude) / len(action_magnitude):.3f} "
      f"action_delta_rms={sum(action_changes) / max(len(action_changes), 1):.3f} "
      f"shank_contact_fraction={shank_contact_steps / len(actual_x):.3f} "
      f"shank_force_over_10_fraction={shank_force_over_10_steps / len(actual_x):.3f} "
      f"foot_contact_fraction={[(round(v, 3)) for v in (foot_contact_counts / len(actual_x)).tolist()]} "
      f"foot_site_height_mean={[(round(v, 3)) for v in (foot_height_sums / len(actual_x)).tolist()]} "
      f"resets={resets} samples={len(actual_x)}"
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
