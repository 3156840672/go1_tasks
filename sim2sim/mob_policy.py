"""Inference-only actor for the Go1 MoB task; no mjlab dependency."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

OBS_DIM = 1710
ACTION_DIM = 12


class MobActor(nn.Module):
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
    if obs.shape[-1] != OBS_DIM:
      raise ValueError(f"Expected {OBS_DIM} observations, got {obs.shape[-1]}")
    return self.mlp(obs)


def load_mob_policy(path: Path) -> MobActor:
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  state = checkpoint["actor_state_dict"]
  weights = sorted(
    (key for key in state if key.startswith("mlp.") and key.endswith(".weight")),
    key=lambda key: int(key.split(".")[1]),
  )
  if len(weights) < 2:
    raise ValueError("Checkpoint has no feed-forward actor")
  input_dim = state[weights[0]].shape[1]
  output_dim = state[weights[-1]].shape[0]
  if (input_dim, output_dim) != (OBS_DIM, ACTION_DIM):
    raise ValueError(
      f"Expected a {OBS_DIM}-input/{ACTION_DIM}-output MoB actor, "
      f"got {input_dim}/{output_dim}"
    )
  actor = MobActor(tuple(state[key].shape[0] for key in weights[:-1]))
  actor.load_state_dict(
    {key: value for key, value in state.items() if key.startswith("mlp.")}
  )
  return actor.eval()
