"""Collect continuous Go1 motion clips from a trained MoB policy for AMP."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import go1_tasks  # noqa: F401  Register the Go1 tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from .mob import BehaviorCommand, GAITS

TASK_ID = "Go1-Velocity-MoB"
FOOT_NAMES = ("FR", "FL", "RR", "RL")
FIELDS = (
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "root_lin_vel_b",
  "root_ang_vel_b",
  "joint_pos",
  "joint_vel",
  "foot_pos_w",
)


def _snapshot(robot, foot_ids: list[int]) -> dict[str, np.ndarray]:
  data = robot.data
  values = {
    "root_pos_w": data.root_link_pos_w[0],
    "root_quat_w": data.root_link_quat_w[0],
    "root_lin_vel_w": data.root_link_lin_vel_w[0],
    "root_ang_vel_w": data.root_link_ang_vel_w[0],
    "root_lin_vel_b": data.root_link_lin_vel_b[0],
    "root_ang_vel_b": data.root_link_ang_vel_b[0],
    "joint_pos": data.joint_pos[0],
    "joint_vel": data.joint_vel[0],
    "foot_pos_w": data.site_pos_w[0, foot_ids],
  }
  return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


def _set_commands(env: ManagerBasedRlEnv, velocity: float, gait: str) -> None:
  twist = env.command_manager.get_term("twist")
  behavior = env.command_manager.get_term("behavior")
  assert isinstance(behavior, BehaviorCommand)
  command = torch.tensor((velocity, 0.0, 0.0), device=env.device)
  twist.vel_command_b[:] = command
  twist.vel_command_w[:] = command
  twist.is_heading_env[:] = False
  twist.is_world_env[:] = False
  twist.is_forward_env[:] = False
  twist.is_standing_env[:] = False
  behavior.values[:] = torch.tensor(
    (*GAITS[gait], 2.5, 0.30, 0.0, 0.30, 0.10), device=env.device
  )
  behavior.cycle.zero_()


def _collect_clip(
  wrapped: RslRlVecEnvWrapper,
  policy,
  robot,
  foot_ids: list[int],
  velocity: float,
  gait: str,
  warmup_steps: int,
  clip_steps: int,
) -> dict[str, np.ndarray] | None:
  env = wrapped.unwrapped
  wrapped.reset()
  _set_commands(env, velocity, gait)
  # Reset created one observation using random commands. The warmup flushes all
  # 30 history slots before any state is written to the reference clip.
  env.observation_manager.compute(update_history=True)
  obs = wrapped.get_observations()

  # The environment mutates command and history tensors across clip resets;
  # inference_mode would turn newly created tensors into immutable inference
  # tensors and make the next reset fail.
  with torch.no_grad():
    for _ in range(warmup_steps):
      obs, _, dones, _ = wrapped.step(policy(obs))
      if bool(dones[0].item()):
        return None

    frames = [_snapshot(robot, foot_ids)]
    for _ in range(clip_steps):
      obs, _, dones, _ = wrapped.step(policy(obs))
      if bool(dones[0].item()):
        return None
      frames.append(_snapshot(robot, foot_ids))

  return {key: np.stack([frame[key] for frame in frames]) for key in FIELDS}


def _validate_clip(clip: dict[str, np.ndarray], expected_frames: int) -> None:
  for key, values in clip.items():
    if values.shape[0] != expected_frames or not np.isfinite(values).all():
      raise ValueError(f"Invalid {key} in collected motion clip")
  quat_norm = np.linalg.norm(clip["root_quat_w"], axis=-1)
  if not np.allclose(quat_norm, 1.0, atol=1e-3):
    raise ValueError("Collected root quaternions are not normalized")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=Path("logs/amp/mob_reference"))
  parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8))
  parser.add_argument("--gait", choices=tuple(GAITS), default="Trot")
  parser.add_argument("--clips-per-speed", type=int, default=3)
  parser.add_argument("--seconds", type=float, default=5.0)
  parser.add_argument("--warmup-seconds", type=float, default=1.0)
  parser.add_argument("--max-attempts-per-clip", type=int, default=5)
  parser.add_argument(
    "--resume", action="store_true", help="Validate and skip existing clips"
  )
  parser.add_argument("--device", choices=("cpu", "cuda:0"), default=None)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.clips_per_speed < 1 or args.max_attempts_per_clip < 1:
    parser.error("Clip counts and maximum attempts must be positive")
  if args.seconds <= 0 or args.warmup_seconds < 0:
    parser.error("--seconds must be positive and --warmup-seconds nonnegative")
  if any(not math.isfinite(speed) or abs(speed) > 1.0 for speed in args.speeds):
    parser.error("Each speed must be finite and within [-1.0, 1.0] m/s")
  if len(set(args.speeds)) != len(args.speeds):
    parser.error("--speeds must not contain duplicates")

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.auto_reset = False
  # Keep one command for the entire clip, including its warmup period.
  cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
  cfg.commands["behavior"].resampling_time_range = (1.0e9, 1.0e9)

  agent_cfg = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(TASK_ID)
    assert runner_cls is not None
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    robot = env.scene["robot"]
    foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != FOOT_NAMES:
      raise ValueError(f"Unexpected foot site order: {foot_names}")

    dt = env.step_dt
    warmup_steps = max(30, math.ceil(args.warmup_seconds / dt))
    clip_steps = math.ceil(args.seconds / dt)
    for speed in args.speeds:
      speed_dir = args.output / args.gait.lower() / f"vx_{speed:+.2f}".replace(".", "p")
      speed_dir.mkdir(parents=True, exist_ok=True)
      for clip_idx in range(args.clips_per_speed):
        file_path = speed_dir / f"clip_{clip_idx:03d}.npz"
        if file_path.exists():
          if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {file_path}; pass --resume")
          with np.load(file_path) as saved:
            _validate_clip({key: saved[key] for key in FIELDS}, clip_steps + 1)
            if (
              not np.isclose(float(saved["fps"]), 1.0 / dt)
              or not np.isclose(float(saved["velocity_command"][0, 0]), speed)
              or str(saved["source_checkpoint"]) != str(args.checkpoint.resolve())
              or not np.allclose(saved["behavior_command"][0, :3], GAITS[args.gait])
            ):
              raise ValueError(
                f"Existing clip does not match this request: {file_path}"
              )
          print(f"[SKIP] {file_path}")
          continue
        for attempt in range(1, args.max_attempts_per_clip + 1):
          clip = _collect_clip(
            wrapped, policy, robot, foot_ids, speed, args.gait, warmup_steps, clip_steps
          )
          if clip is not None:
            break
          print(f"[WARN] {speed:+.2f} m/s clip {clip_idx}: fell, retry {attempt}")
        else:
          raise RuntimeError(
            f"Could not collect stable clip {clip_idx} at {speed:+.2f} m/s"
          )

        _validate_clip(clip, clip_steps + 1)
        np.savez_compressed(
          file_path,
          **clip,
          fps=np.array(1.0 / dt, dtype=np.float32),
          joint_names=np.array(robot.joint_names),
          foot_site_names=np.array(foot_names),
          velocity_command=np.tile((speed, 0.0, 0.0), (clip_steps + 1, 1)),
          behavior_command=np.tile(
            (*GAITS[args.gait], 2.5, 0.30, 0.0, 0.30, 0.10),
            (clip_steps + 1, 1),
          ),
          source_checkpoint=np.array(str(args.checkpoint.resolve())),
          seed=np.array(args.seed),
        )
        mean_vx = float(clip["root_lin_vel_b"][:, 0].mean())
        print(f"[OK] {file_path}: {clip_steps + 1} frames, mean body vx={mean_vx:.3f}")
  finally:
    wrapped.close()


if __name__ == "__main__":
  main()
