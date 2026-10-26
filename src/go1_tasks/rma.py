"""Go1 RMA privileged observations and policy networks."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

# Mass (1), COM (3), motor gain (12), foot friction (1), terrain (1).
PRIVILEGED_DIM = 18
LATENT_DIM = 8
HISTORY_LENGTH = 50


def privileged_factors(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read randomized physics parameters from the batched simulation model."""
  robot = env.scene["robot"]
  model = env.sim.model
  defaults = env.sim.get_default_field
  body_id = robot.data.indexing.body_ids[robot.find_bodies("trunk")[0][0]]
  foot_id = robot.data.indexing.geom_ids[robot.find_geoms("FR_foot_collision")[0][0]]
  ctrl_ids = robot.data.indexing.ctrl_ids
  height_sensor = env.scene["foot_height_scan"]
  assert isinstance(height_sensor, TerrainHeightSensor)

  mass = (model.body_mass[:, body_id] - defaults("body_mass")[body_id]).unsqueeze(-1)
  com = model.body_ipos[:, body_id] - defaults("body_ipos")[body_id]
  gains = (
    model.actuator_gainprm[:, ctrl_ids, 0] / defaults("actuator_gainprm")[ctrl_ids, 0]
  )
  friction = model.geom_friction[:, foot_id, 0:1]
  terrain = height_sensor.data.heights.amax(dim=1, keepdim=True)
  factors = torch.cat((mass, com, gains, friction, terrain), dim=-1)
  assert factors.shape[-1] == PRIVILEGED_DIM
  return factors


class RmaActor(MLPModel):
  """PPO actor with a jointly trained privileged environment encoder."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.encoder = MLP(PRIVILEGED_DIM, LATENT_DIM, (256, 128), "elu")

  def _get_latent_dim(self) -> int:
    return self.obs_dim - PRIVILEGED_DIM + LATENT_DIM

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    return torch.cat((obs["actor"], self.encoder(obs["privileged"])), dim=-1)

  def act_with_latent(self, base: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    output = self.mlp(torch.cat((base, latent), dim=-1))
    assert self.distribution is not None
    return self.distribution.deterministic_output(output)


class AdaptationModule(nn.Module):
  """Predict the teacher latent from 50 prior actor observations."""

  def __init__(self, obs_dim: int) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.embed = MLP(obs_dim, 32, (32,), "elu")
    self.temporal = nn.Sequential(
      nn.Conv1d(32, 32, kernel_size=8, stride=4),
      nn.ELU(),
      nn.Conv1d(32, 32, kernel_size=5),
      nn.ELU(),
      nn.Conv1d(32, 32, kernel_size=5),
      nn.ELU(),
      nn.Flatten(),
      nn.Linear(96, LATENT_DIM),
    )

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    if history.shape[-2:] != (HISTORY_LENGTH, self.obs_dim):
      raise ValueError(f"Expected history [B, {HISTORY_LENGTH}, {self.obs_dim}]")
    embedded = self.embed(history).transpose(1, 2)
    return self.temporal(embedded)
