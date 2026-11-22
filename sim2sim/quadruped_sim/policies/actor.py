"""Feed-forward actor shared by the flat-terrain velocity policies."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

#: Every runner in this directory drives the same 12 leg joints.
ACTION_DIM = 12
#: Flat-velocity observation: angular velocity, gravity, joint state, action, command.
OBS_DIM = 48


class StandardActor(nn.Module):
  """Proprioceptive MLP actor, sized from the checkpoint at load time."""

  def __init__(self, hidden_dims: tuple[int, ...], obs_dim: int = OBS_DIM) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    dims = (obs_dim, *hidden_dims, ACTION_DIM)
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
      layers.append(nn.Linear(input_dim, output_dim))
      if index < len(dims) - 2:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


def linear_dims(
  state: dict[str, torch.Tensor], prefix: str = "mlp."
) -> tuple[int, ...]:
  """Output width of every ``<prefix>N.weight`` layer, in depth order."""
  keys = sorted(
    (key for key in state if key.startswith(prefix) and key.endswith(".weight")),
    key=lambda key: int(key[len(prefix) :].split(".")[0]),
  )
  if not keys:
    raise ValueError(f"Checkpoint has no {prefix} layers")
  return tuple(state[key].shape[0] for key in keys)


def load_standard_policy(
  checkpoint_path: Path, supported_obs_dims: tuple[int, ...] = (OBS_DIM,)
) -> StandardActor:
  """Load an ``mlp.``-only actor and check its interface dimensions."""
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
  if input_dim not in supported_obs_dims:
    hint = ""
    if input_dim in {dim + 32 for dim in supported_obs_dims}:
      hint = (
        f". A {input_dim}-D actor is a CTS policy head: it consumes a 32-D "
        "latent from the history encoder, so pass the run's model_*.pt or its "
        "'<run>.onnx' export through --cts instead of --policy"
      )
    raise ValueError(
      f"Policy expects {input_dim} observations; this runner supports the "
      f"following observation dimensions: {supported_obs_dims}{hint}"
    )
  if output_dim != ACTION_DIM:
    raise ValueError(f"Policy outputs {output_dim} actions, expected {ACTION_DIM}")
  hidden_dims = tuple(state[key].shape[0] for key in weight_keys[:-1])
  actor = StandardActor(hidden_dims, obs_dim=input_dim)
  actor.load_state_dict(
    {key: value for key, value in state.items() if key.startswith("mlp.")}
  )
  return actor.eval()