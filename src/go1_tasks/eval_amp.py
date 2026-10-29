"""Compare an AMP policy's foot timing with its MoB Trot reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import go1_tasks  # noqa: F401  Register Go1 tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "Go1-Velocity-AMP"
FOOT_NAMES = ("FR", "FL", "RR", "RL")


def _foot_summary(heights: np.ndarray) -> tuple[np.ndarray, float, float]:
  if heights.ndim != 2 or heights.shape[1] != 4 or len(heights) < 3:
    raise ValueError("At least three samples of four foot heights are required")
  lift = np.ptp(heights, axis=0)
  if np.any(np.std(heights, axis=0) < 1e-5):
    return lift, float("nan"), float("nan")
  corr = np.corrcoef(heights.T)
  diagonal = float((corr[0, 3] + corr[1, 2]) / 2)
  adjacent = float((corr[0, 1] + corr[2, 3]) / 2)
  return lift, diagonal, adjacent


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--vx", type=float, default=0.6)
  parser.add_argument("--seconds", type=float, default=5.0)
  parser.add_argument("--warmup", type=float, default=0.5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--reference-dir", type=Path,
                      default=Path("logs/amp/mob_reference/trot"))
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if (
    not 0.0 < args.vx <= 1.0
    or args.seconds <= 0
    or not 0 <= args.warmup < args.seconds
  ):
    parser.error("Use positive vx and seconds, with 0 <= warmup < seconds")

  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.seed = args.seed
  cfg.scene.num_envs = 1
  cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1)
  cfg.events = {
    name: term for name, term in cfg.events.items()
    if term.mode != "startup" and name != "randomize_terrain"
  }
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (args.vx, args.vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.resampling_time_range = (1.0e9, 1.0e9)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK_ID).clip_actions)
  try:
    agent_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID)
    assert runner_cls is not None
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    foot_ids, names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    assert tuple(names) == FOOT_NAMES
    heights: list[np.ndarray] = []
    speeds: list[float] = []
    steps = round(args.seconds / env.step_dt)
    warmup_steps = round(args.warmup / env.step_dt)
    fell = False
    with torch.no_grad():
      for step in range(steps):
        obs, _, dones, _ = wrapped.step(policy(obs))
        if bool(dones[0]):
          fell = True
          break
        if step >= warmup_steps:
          heights.append(robot.data.site_pos_w[0, foot_ids, 2].cpu().numpy().copy())
          speeds.append(float(robot.data.root_link_lin_vel_b[0, 0]))
    if len(heights) < 3:
      raise RuntimeError("Policy did not remain upright long enough to evaluate gait")
    lift, diagonal, adjacent = _foot_summary(np.stack(heights))
    print(f"policy: vx={np.mean(speeds):.3f} m/s, samples={len(heights)}, fell={fell}")
    print(f"policy: foot lift FR FL RR RL={[round(float(v), 3) for v in lift]} m")
    print(
      f"policy: diagonal correlation={diagonal:.3f}, "
      f"adjacent correlation={adjacent:.3f}"
    )

    speed_dir = args.reference_dir / f"vx_{args.vx:+.2f}".replace(".", "p")
    clip_path = speed_dir / "clip_000.npz"
    if clip_path.is_file():
      with np.load(clip_path, allow_pickle=False) as clip:
        ref_heights = clip["foot_pos_w"][:, :, 2]
        ref_vx = float(clip["root_lin_vel_b"][:, 0].mean())
      ref_lift, ref_diagonal, ref_adjacent = _foot_summary(ref_heights)
      print(f"reference: vx={ref_vx:.3f} m/s, clip={clip_path}")
      print(
        f"reference: foot lift FR FL RR RL="
        f"{[round(float(v), 3) for v in ref_lift]} m"
      )
      print(
        f"reference: diagonal correlation={ref_diagonal:.3f}, "
        f"adjacent correlation={ref_adjacent:.3f}"
      )
    else:
      print(f"reference: no clip at {clip_path}")
  finally:
    wrapped.close()


if __name__ == "__main__":
  main()
