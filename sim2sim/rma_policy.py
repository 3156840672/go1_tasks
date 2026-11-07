"""Inference-only flat velocity networks; no mjlab or rsl_rl dependency."""

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
  """Proprioceptive actor used by the flat velocity tasks."""

  def __init__(
    self, hidden_dims: tuple[int, ...], obs_dim: int = OBS_DIM
  ) -> None:
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


class PaperAmpStudentActor(nn.Module):
  """Stateful inference copy of the paper AMP LSTM student."""

  def __init__(self) -> None:
    super().__init__()
    self.obs_dim = 45
    self.lstm = nn.LSTM(self.obs_dim, 256, batch_first=True)
    self.memory_encoder = nn.Sequential(
      nn.Linear(256, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 24),
    )
    self.low_level = nn.Sequential(
      nn.Linear(self.obs_dim + 24, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 64),
      nn.ELU(),
      nn.Linear(64, ACTION_DIM),
      nn.Tanh(),
    )
    self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

  def reset(self, batch_size: int = 1) -> None:
    self.hidden = (
      torch.zeros(1, batch_size, 256),
      torch.zeros(1, batch_size, 256),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if self.hidden is None or self.hidden[0].shape[1] != obs.shape[0]:
      self.reset(obs.shape[0])
    output, self.hidden = self.lstm(obs.unsqueeze(1), self.hidden)
    latent = self.memory_encoder(output[:, 0])
    return self.low_level(torch.cat((obs, latent), dim=-1))


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


def load_standard_policy(
  checkpoint_path: Path, supported_obs_dims: tuple[int, ...] = (OBS_DIM,)
) -> StandardActor:
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
    raise ValueError(
      f"Policy expects {input_dim} observations; this runner supports the "
      f"following observation dimensions: {supported_obs_dims}"
    )
  if output_dim != ACTION_DIM:
    raise ValueError(f"Policy outputs {output_dim} actions, expected {ACTION_DIM}")
  hidden_dims = tuple(state[key].shape[0] for key in weight_keys[:-1])
  actor = StandardActor(hidden_dims, obs_dim=input_dim)
  actor.load_state_dict(
    {key: value for key, value in state.items() if key.startswith("mlp.")}
  )
  return actor.eval()


def load_paper_amp_student(checkpoint_path: Path) -> PaperAmpStudentActor:
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  required = {
    "student_state_dict",
    "proprio_dim",
    "latent_dim",
    "lstm_hidden_dim",
  }
  if not required <= checkpoint.keys():
    raise ValueError("Checkpoint is not a paper AMP student")
  if (
    checkpoint["proprio_dim"] != 45
    or checkpoint["latent_dim"] != 24
    or checkpoint["lstm_hidden_dim"] != 256
  ):
    raise ValueError("Paper AMP student dimensions do not match deployment contract")
  actor = PaperAmpStudentActor()
  if not checkpoint.get("bound_action_mean", True):
    actor.low_level[-1] = nn.Identity()
  actor.load_state_dict(checkpoint["student_state_dict"])
  actor.reset()
  return actor.eval()


HIM_OBS_DIM = 45
HIM_HISTORY_STEPS = 6
HIM_VELOCITY_DIM = 3
HIM_CONTRACT_VERSION = 2


class HimActor(nn.Module):
  """Inference copy of the HIM policy (history encoder + policy MLP).

  HIMLoco stacks the history newest-first.  This module keeps the same
  ordering internally, so the caller only feeds the current 45-D
  proprioceptive observation and the ring buffer is maintained here.
  """

  def __init__(
    self,
    obs_dim: int = HIM_OBS_DIM,
    history_steps: int = HIM_HISTORY_STEPS,
    velocity_dim: int = HIM_VELOCITY_DIM,
    latent_dim: int = 16,
    encoder_hidden_dims: tuple[int, ...] = (128, 64),
    hidden_dims: tuple[int, ...] = (512, 256, 128),
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.history_steps = history_steps
    self.velocity_dim = velocity_dim
    self.encoder = _mlp(
      history_steps * obs_dim, velocity_dim + latent_dim, encoder_hidden_dims
    )
    self.mlp = _mlp(obs_dim + velocity_dim + latent_dim, ACTION_DIM, hidden_dims)
    self._history: torch.Tensor | None = None

  def reset(self, batch_size: int = 1) -> None:
    self._history = torch.zeros(batch_size, self.history_steps, self.obs_dim)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if self._history is None or self._history.shape[0] != obs.shape[0]:
      self.reset(obs.shape[0])
    assert self._history is not None
    # Newest sample first, matching HIMLoco's observation buffer.
    self._history = torch.roll(self._history, 1, dims=1)
    self._history[:, 0] = obs
    parts = self.encoder(self._history.flatten(1))
    velocity = parts[..., : self.velocity_dim]
    latent = torch.nn.functional.normalize(
      parts[..., self.velocity_dim :], dim=-1, p=2
    )
    return self.mlp(torch.cat((obs, velocity, latent), dim=-1))


def _mlp(
  input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
) -> nn.Sequential:
  layers: list[nn.Module] = []
  width = input_dim
  for hidden in hidden_dims:
    layers += [nn.Linear(width, hidden), nn.ELU()]
    width = hidden
  layers.append(nn.Linear(width, output_dim))
  return nn.Sequential(*layers)


def _linear_dims(state: dict[str, torch.Tensor], prefix: str) -> tuple[int, ...]:
  keys = sorted(
    (key for key in state if key.startswith(prefix) and key.endswith(".weight")),
    key=lambda key: int(key[len(prefix):].split(".")[0]),
  )
  if not keys:
    raise ValueError(f"Checkpoint has no {prefix} layers")
  return tuple(state[key].shape[0] for key in keys)


def load_him_policy(checkpoint_path: Path) -> HimActor:
  """Load a HIM checkpoint produced by ``Go2-Velocity-HIM``."""
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  state = checkpoint.get("actor_state_dict")
  if not isinstance(state, dict) or "estimator.encoder.0.weight" not in state:
    raise ValueError("Checkpoint is not a HIM policy")
  if checkpoint.get("him_contract_version") != HIM_CONTRACT_VERSION:
    raise ValueError(
      "Incompatible HIM checkpoint: train a new policy with the corrected "
      "history, SwAV gradient, and observation scales"
    )
  encoder_dims = _linear_dims(state, "estimator.encoder.")
  mlp_dims = _linear_dims(state, "mlp.")
  flat_dim = state["estimator.encoder.0.weight"].shape[1]
  output_dim = encoder_dims[-1]
  latent_dim = output_dim - HIM_VELOCITY_DIM
  if flat_dim % HIM_OBS_DIM != 0:
    raise ValueError(f"Unexpected HIM encoder input: {flat_dim}")
  history_steps = flat_dim // HIM_OBS_DIM
  first_mlp_key = min(
    (key for key in state if key.startswith("mlp.") and key.endswith(".weight")),
    key=lambda key: int(key[len("mlp."):].split(".")[0]),
  )
  if state[first_mlp_key].shape[1] != HIM_OBS_DIM + HIM_VELOCITY_DIM + latent_dim:
    raise ValueError("HIM encoder/MLP latent dimensions disagree")
  if mlp_dims[-1] != ACTION_DIM:
    raise ValueError(f"Policy outputs {mlp_dims[-1]} actions, expected {ACTION_DIM}")
  actor = HimActor(
    history_steps=history_steps,
    latent_dim=latent_dim,
    encoder_hidden_dims=encoder_dims[:-1],
    hidden_dims=mlp_dims[:-1],
  )
  actor.encoder.load_state_dict(
    {key[len("estimator.encoder."):]: value for key, value in state.items()
     if key.startswith("estimator.encoder.")}
  )
  actor.mlp.load_state_dict(
    {key[len("mlp."):]: value for key, value in state.items() if key.startswith("mlp.")}
  )
  actor.reset()
  return actor.eval()
