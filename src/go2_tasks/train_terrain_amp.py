"""Fresh Go2 AMP teacher training with terrain curriculum and sustained commands."""

import argparse

from mjlab.scripts.train import TrainConfig, launch_training

from . import TASK_IDS


def training_config(iterations=8000, num_envs=1024, seed=42, run_name="terrain-curriculum-fresh"):
  """Keep the proven baseline rewards and train longer at the full speed range."""
  if iterations < 1 or num_envs < 1:
    raise ValueError("iterations and num_envs must be positive")
  cfg = TrainConfig.from_task(TASK_IDS["paper_amp_teacher"])
  cfg.env.scene.num_envs = num_envs
  cfg.agent.max_iterations = iterations
  cfg.agent.seed = seed
  cfg.agent.run_name = run_name
  cfg.agent.resume = False
  cfg.agent.logger = "tensorboard"
  cfg.agent.upload_model = False
  cfg.agent.save_interval = 250
  command = cfg.env.commands["twist"]
  # Resampling every 4-8 s makes even accurate tracking reverse direction
  # before the robot exits an 8 m tile. Allow the curriculum a full episode
  # to observe sustained progress, with a stable target heading.
  command.resampling_time_range = (20., 20.)
  command.heading_command = True
  command.rel_heading_envs = 1.
  command.ranges.heading = (-3.141592653589793, 3.141592653589793)
  return cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--iterations", type=int, default=8000)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--run-name", default="terrain-curriculum-fresh")
  args = parser.parse_args()
  cfg = training_config(args.iterations, args.num_envs, args.seed, args.run_name)
  launch_training(TASK_IDS["paper_amp_teacher"], cfg)


if __name__ == "__main__":
  main()
