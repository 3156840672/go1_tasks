"""Inference-only Go1 RMA networks; no mjlab or rsl_rl dependency."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

OBS_DIM = 48
HISTORY_LENGTH = 50
LATENT_DIM = 8
ACTION_DIM = 12


class Actor(nn.Module):
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


class StandardActor(nn.Module):
  """Proprioceptive actor used by the Go1 flat velocity task."""

  def __init__(self, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    dims = (OBS_DIM, *hidden_dims, ACTION_DIM)
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
      layers.append(nn.Linear(input_dim, output_dim))
      if index < len(dims) - 2:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


class Adapter(nn.Module):
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


def load_standard_policy(checkpoint_path: Path) -> StandardActor:
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  state = checkpoint["actor_state_dict"]
  weight_keys = sorted(
    (key for key in state if key.startswith("mlp.") and key.endswith(".weight")),
    key=lambda key: int(key.split(".")[1]),
  )
  if len(weight_keys) < 2:
    raise ValueError("Checkpoint does not contain a feed-forward actor MLP")
  input_dim = state[weight_keys[0]].shape[1]
  output_dim = state[weight_keys[-1]].shape[0]
  if input_dim != OBS_DIM:
    raise ValueError(
      f"Policy expects {input_dim} observations; this runner supports the "
      f"new {OBS_DIM}-observation policy without height scan"
    )
  if output_dim != ACTION_DIM:
    raise ValueError(f"Policy outputs {output_dim} actions, expected {ACTION_DIM}")
  hidden_dims = tuple(state[key].shape[0] for key in weight_keys[:-1])
  actor = StandardActor(hidden_dims)
  actor.load_state_dict(
    {key: value for key, value in state.items() if key.startswith("mlp.")}
  )
  return actor.eval()
