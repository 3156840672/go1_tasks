"""Train the LSTM student from a completed Go2 paper-AMP teacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from . import unitree_go2_paper_amp_runner_cfg
from .env_cfgs import unitree_go2_paper_amp_env_cfg
from .paper_amp import (
  SEQUENCE_LENGTH,
  PaperAmpStudent,
  PaperAmpTeacherActor,
  export_student_onnx,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--teacher", type=Path, required=True, help="Teacher model_*.pt checkpoint."
  )
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--iterations", type=int, default=4000)
  parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--latent-loss-weight", type=float, default=1.0)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", type=Path, help="Student checkpoint to resume.")
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  return parser.parse_args()


def _save(
  student: PaperAmpStudent,
  optimizer: torch.optim.Optimizer,
  teacher_path: Path,
  output: Path,
  iteration: int,
  sequence_length: int,
) -> None:
  checkpoint = output / f"student_{iteration}.pt"
  torch.save(
    {
      "student_state_dict": student.state_dict(),
      "bound_action_mean": getattr(student, "bound_action_mean", True),
      "optimizer_state_dict": optimizer.state_dict(),
      "teacher_checkpoint": str(teacher_path.resolve()),
      "iteration": iteration,
      "proprio_dim": 45,
      "latent_dim": 24,
      "lstm_hidden_dim": 256,
      "sequence_length": sequence_length,
    },
    checkpoint,
  )
  export_student_onnx(student, output / f"student_{iteration}.onnx")


def train(args: argparse.Namespace) -> None:
  if not args.teacher.is_file():
    raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher}")
  positive = (
    args.num_envs,
    args.iterations,
    args.sequence_length,
    args.epochs,
    args.batch_size,
    args.save_interval,
  )
  if any(value <= 0 for value in positive):
    raise ValueError(
      "Environment, iteration, sequence and batch counts must be positive"
    )
  if args.learning_rate <= 0 or args.latent_loss_weight < 0:
    raise ValueError(
      "Learning rate must be positive and latent loss weight nonnegative"
    )

  cfg = unitree_go2_paper_amp_env_cfg(play=False)
  cfg.scene.num_envs = args.num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
  output = args.output or (
    Path("logs/rsl_rl/go2_paper_amp_student")
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  output.mkdir(parents=True, exist_ok=True)

  try:
    runner_cfg = unitree_go2_paper_amp_runner_cfg()
    runner_cfg.logger = "tensorboard"
    runner_cfg.upload_model = False
    teacher_saved = torch.load(args.teacher, map_location="cpu", weights_only=True)
    runner_cfg.actor.bound_action_mean = teacher_saved.get("amp_bound_action_mean", True)
    teacher_runner = MjlabOnPolicyRunner(
      env, asdict(runner_cfg), log_dir=None, device=args.device
    )
    teacher_runner.load(str(args.teacher), map_location=args.device)
    teacher = teacher_runner.alg.actor.eval()
    if not isinstance(teacher, PaperAmpTeacherActor):
      raise TypeError("Checkpoint is not a Go2 paper-AMP teacher")
    for parameter in teacher.parameters():
      parameter.requires_grad_(False)

    student = PaperAmpStudent().to(args.device)
    student.copy_low_level_from(teacher)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    start_iteration = 0
    if args.resume is not None:
      saved = torch.load(args.resume, map_location=args.device, weights_only=True)
      if Path(saved["teacher_checkpoint"]).name != args.teacher.name:
        raise ValueError("Student checkpoint was distilled from another teacher")
      if saved.get("bound_action_mean", True) != teacher.bound_action_mean:
        raise ValueError("Student action activation differs from teacher")
      student.load_state_dict(saved["student_state_dict"])
      optimizer.load_state_dict(saved["optimizer_state_dict"])
      start_iteration = int(saved["iteration"])

    obs = env.get_observations()
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
      hidden, cell = student.initial_state(args.num_envs, args.device)
      observations, teacher_actions, teacher_latents, dones = [], [], [], []
      student.eval()
      for _ in range(args.sequence_length):
        with torch.no_grad():
          target_action = teacher(obs)
          target_latent = teacher.encode_privileged(obs)
          action, _, hidden, cell = student.step(obs["actor"], hidden, cell)
        observations.append(obs["actor"].clone())
        teacher_actions.append(target_action.clone())
        teacher_latents.append(target_latent.clone())
        obs, _, done, _ = env.step(action)
        done_bool = done.bool()
        dones.append(done_bool)
        keep = (~done_bool).float().view(1, args.num_envs, 1)
        hidden, cell = hidden * keep, cell * keep

      # Each environment contributes one 1-second DAgger sequence. The student
      # generated the visited states; labels always come from the frozen teacher.
      x = torch.stack(observations, dim=1)
      action_target = torch.stack(teacher_actions, dim=1)
      latent_target = torch.stack(teacher_latents, dim=1)
      done_tensor = torch.stack(dones, dim=1)
      student.train()
      action_total = latent_total = 0.0
      updates = 0
      for _ in range(args.epochs):
        for indices in torch.randperm(args.num_envs, device=args.device).split(
          args.batch_size
        ):
          predicted_action, predicted_latent = student.forward_sequence(
            x[indices], done_tensor[indices]
          )
          action_loss = F.mse_loss(predicted_action, action_target[indices])
          latent_loss = F.mse_loss(predicted_latent, latent_target[indices])
          loss = action_loss + args.latent_loss_weight * latent_loss
          optimizer.zero_grad(set_to_none=True)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
          optimizer.step()
          action_total += action_loss.item()
          latent_total += latent_loss.item()
          updates += 1

      if iteration == 1 or iteration % 10 == 0:
        print(
          f"iteration={iteration} action_mse={action_total / updates:.6f} "
          f"latent_mse={latent_total / updates:.6f}"
        )
      final_iteration = start_iteration + args.iterations
      if iteration % args.save_interval == 0 or iteration == final_iteration:
        _save(
          student,
          optimizer,
          args.teacher,
          output,
          iteration,
          args.sequence_length,
        )
    print(f"Student checkpoints and ONNX: {output}")
  finally:
    env.close()


def main() -> None:
  torch.manual_seed(42)
  train(parse_args())


if __name__ == "__main__":
  main()
