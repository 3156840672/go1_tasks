"""Go2 policy heads: paper AMP LSTM student and HIM history encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .actor import ACTION_DIM, linear_dims

#: AMP and the LSTM student drop the privileged base linear velocity.
AMP_OBS_DIM = 45
HIM_OBS_DIM = 45
HIM_HISTORY_STEPS = 6
HIM_VELOCITY_DIM = 3
HIM_CONTRACT_VERSION = 2

#: <repo>/logs/rsl_rl, shared by every task in the workspace.
LOG_ROOT = Path(__file__).resolve().parents[3] / "logs" / "rsl_rl"


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


class PaperAmpStudentActor(nn.Module):
  """Stateful inference copy of the paper AMP LSTM student."""

  def __init__(self) -> None:
    super().__init__()
    self.obs_dim = AMP_OBS_DIM
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
    checkpoint["proprio_dim"] != AMP_OBS_DIM
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


def resolve_checkpoint(path: str | Path) -> Path:
  """Accept ``latest`` or a missing path and point at a real checkpoint."""
  text = str(path)
  checkpoints = sorted(
    LOG_ROOT.glob("*/*/model_*.pt"),
    key=lambda item: item.stat().st_mtime,
    reverse=True,
  )
  if text in {"latest", "last"}:
    # Other tasks (rough, AMP, ...) also write model_*.pt; only HIM loads here.
    him = [item for item in checkpoints if "him" in item.parts[-3].lower()]
    if not him and not checkpoints:
      raise ValueError(f"No checkpoints under {LOG_ROOT}")
    return (him or checkpoints)[0]
  candidate = Path(text).expanduser()
  if candidate.exists():
    return candidate
  pool = [item for item in checkpoints if "him" in item.parts[-3].lower()]
  recent = "\n".join(f"  {item}" for item in (pool or checkpoints)[:3]) or "  (none)"
  raise ValueError(
    f"Checkpoint not found: {candidate}\n"
    f"Newest checkpoints under {LOG_ROOT}:\n{recent}\n"
    "Pass a real path, or use --him latest."
  )


def load_him_policy(checkpoint_path: str | Path) -> HimActor:
  """Load a HIM checkpoint produced by ``Go2-Velocity-HIM``."""
  checkpoint = torch.load(
    resolve_checkpoint(checkpoint_path), map_location="cpu", weights_only=True
  )
  state = checkpoint.get("actor_state_dict")
  if not isinstance(state, dict) or "estimator.encoder.0.weight" not in state:
    raise ValueError("Checkpoint is not a HIM policy")
  if checkpoint.get("him_contract_version") != HIM_CONTRACT_VERSION:
    raise ValueError(
      "Incompatible HIM checkpoint: train a new policy with the corrected "
      "history, SwAV gradient, and observation scales"
    )
  encoder_dims = linear_dims(state, "estimator.encoder.")
  mlp_dims = linear_dims(state, "mlp.")
  flat_dim = state["estimator.encoder.0.weight"].shape[1]
  latent_dim = encoder_dims[-1] - HIM_VELOCITY_DIM
  if flat_dim % HIM_OBS_DIM != 0:
    raise ValueError(f"Unexpected HIM encoder input: {flat_dim}")
  history_steps = flat_dim // HIM_OBS_DIM
  first_mlp_key = min(
    (key for key in state if key.startswith("mlp.") and key.endswith(".weight")),
    key=lambda key: int(key[len("mlp.") :].split(".")[0]),
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
    {
      key[len("estimator.encoder.") :]: value
      for key, value in state.items()
      if key.startswith("estimator.encoder.")
    }
  )
  actor.mlp.load_state_dict(
    {
      key[len("mlp.") :]: value
      for key, value in state.items()
      if key.startswith("mlp.")
    }
  )
  actor.reset()
  return actor.eval()