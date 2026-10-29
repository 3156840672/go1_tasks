"""Train the two Go1 RMA stages in simulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import get_checkpoint_path

from .rma import HISTORY_LENGTH, AdaptationModule
from .rma_cfg import unitree_go1_rma_env_cfg, unitree_go1_rma_runner_cfg


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("stage", choices=("teacher", "adaptation"))
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--steps", type=int, default=32)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume teacher training from the latest checkpoint in --output.",
  )
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  return parser.parse_args()


def make_env(num_envs: int, device: str) -> RslRlVecEnvWrapper:
  cfg = unitree_go1_rma_env_cfg()
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return RslRlVecEnvWrapper(env)


def train_teacher(args: argparse.Namespace, env: RslRlVecEnvWrapper) -> None:
  cfg = unitree_go1_rma_runner_cfg()
  if args.iterations is not None:
    cfg.max_iterations = args.iterations
  cfg.num_steps_per_env = args.steps

  if args.resume:
    if args.output is None:
      raise ValueError(
        "--resume requires --output pointing to an existing run directory"
      )
    run_dir = args.output
    if not run_dir.is_dir():
      raise ValueError(f"Run directory does not exist: {run_dir}")
    resume_path = get_checkpoint_path(run_dir.parent, run_dir.name, "model_.*.pt")
  else:
    run_dir = args.output or Path("logs/rsl_rl/go1_rma") / datetime.now().strftime(
      "%Y-%m-%d_%H-%M-%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = None

  runner = MjlabOnPolicyRunner(env, asdict(cfg), str(run_dir), args.device)
  if resume_path is not None:
    print(f"Resuming teacher training from checkpoint: {resume_path}")
    runner.load(str(resume_path), map_location=args.device)
  runner.learn(cfg.max_iterations, init_at_random_ep_len=True)
  print(f"Teacher checkpoints: {run_dir}")


def train_adaptation(args: argparse.Namespace, env: RslRlVecEnvWrapper) -> None:
  if args.checkpoint is None or not args.checkpoint.is_file():
    raise ValueError(
      "adaptation requires --checkpoint pointing to a teacher model_*.pt"
    )
  cfg = unitree_go1_rma_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(cfg), None, args.device)
  runner.load(str(args.checkpoint), map_location=args.device)
  teacher = runner.alg.actor.eval()
  for param in teacher.parameters():
    param.requires_grad_(False)

  obs = env.get_observations()
  obs_dim = obs["actor"].shape[-1]
  history = torch.zeros(env.num_envs, HISTORY_LENGTH, obs_dim, device=args.device)
  adapter = AdaptationModule(obs_dim).to(args.device)
  optimizer = torch.optim.Adam(adapter.parameters(), lr=5e-4)
  iterations = args.iterations or 1000
  out_dir = args.output or args.checkpoint.parent / "adaptation"
  out_dir.mkdir(parents=True, exist_ok=True)

  for iteration in range(iterations):
    histories = []
    targets = []
    for _ in range(args.steps):
      with torch.no_grad():
        estimate = adapter(history)
        action = teacher.act_with_latent(obs["actor"], estimate)
        target = teacher.encoder(obs["privileged"])
        histories.append(history.clone())
        targets.append(target)
        next_obs, _, dones, _ = env.step(action)
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = obs["actor"]
        history[dones.bool()] = 0
        obs = next_obs

    inputs = torch.stack(histories).flatten(0, 1)
    target_latents = torch.stack(targets).flatten(0, 1)
    losses = []
    for _ in range(4):
      for indices in torch.randperm(inputs.shape[0], device=args.device).split(2048):
        loss = F.mse_loss(adapter(inputs[indices]), target_latents[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    if iteration % 10 == 0:
      mean_loss = torch.stack(losses).mean().item()
      print(f"iteration={iteration} latent_mse={mean_loss:.6f}")
    if (iteration + 1) % 50 == 0 or iteration + 1 == iterations:
      torch.save(
        {
          "adapter_state_dict": adapter.state_dict(),
          "teacher_checkpoint": str(args.checkpoint.resolve()),
          "obs_dim": obs_dim,
          "history_length": HISTORY_LENGTH,
          "iteration": iteration + 1,
        },
        out_dir / f"adapter_{iteration + 1}.pt",
      )
  print(f"Adaptation checkpoints: {out_dir}")


def main() -> None:
  args = parse_args()
  if (
    args.num_envs <= 0
    or args.steps <= 0
    or (args.iterations is not None and args.iterations <= 0)
  ):
    raise ValueError("--num-envs, --steps and --iterations must be positive")
  torch.manual_seed(42)
  env = make_env(args.num_envs, args.device)
  try:
    if args.stage == "teacher":
      train_teacher(args, env)
    else:
      train_adaptation(args, env)
  finally:
    env.close()


if __name__ == "__main__":
  main()
