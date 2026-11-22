"""Go1 RMA teacher/adapter: 48-D observation, 50-frame history, 8-D latent."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .actor import ACTION_DIM, OBS_DIM

HISTORY_LENGTH = 50
LATENT_DIM = 8


class Actor(nn.Module):
  """RMA actor: current observation concatenated with the latent code."""

  def __init__(self) -> None:
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(OBS_DIM + LATENT_DIM, 128),
      nn.ELU(),
      nn.Linear(128, 128),
      nn.ELU(),
      nn.Linear(128, 128),
      nn.ELU(),
      nn.Linear(128, ACTION_DIM),
    )

  def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return self.mlp(torch.cat((obs, latent), dim=-1))


class Adapter(nn.Module):
  """Temporal encoder that turns the observation history into the latent code."""

  def __init__(self) -> None:
    super().__init__()
    self.embed = nn.Sequential(nn.Linear(OBS_DIM, 32), nn.ELU(), nn.Linear(32, 32))
    self.temporal = nn.Sequential(
      nn.Conv1d(32, 32, 8, stride=4),
      nn.ELU(),
      nn.Conv1d(32, 32, 5),
      nn.ELU(),
      nn.Conv1d(32, 32, 5),
      nn.ELU(),
      nn.Flatten(),
      nn.Linear(96, LATENT_DIM),
    )

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    if history.shape[-2:] != (HISTORY_LENGTH, OBS_DIM):
      raise ValueError(f"Expected [B, {HISTORY_LENGTH}, {OBS_DIM}] history")
    return self.temporal(self.embed(history).transpose(1, 2))


def load_policy(teacher_path: Path, adapter_path: Path) -> tuple[Actor, Adapter]:
  """Load the stage-1 actor and the stage-2 adapter trained against it."""
  teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
  student = torch.load(adapter_path, map_location="cpu", weights_only=True)
  if (
    student.get("obs_dim") != OBS_DIM or student.get("history_length") != HISTORY_LENGTH
  ):
    raise ValueError("Adapter checkpoint observation/history shape mismatch")
  if Path(student["teacher_checkpoint"]).name != teacher_path.name:
    raise ValueError("Adapter was trained with a different teacher checkpoint")
  actor = Actor()
  actor.load_state_dict(
    {k: v for k, v in teacher["actor_state_dict"].items() if k.startswith("mlp.")}
  )
  adapter = Adapter()
  adapter.load_state_dict(student["adapter_state_dict"])
  return actor.eval(), adapter.eval()