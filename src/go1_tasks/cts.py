"""Concurrent teacher-student policy for Go1 rough-terrain locomotion."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def randomize_trunk_payload(
  env,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Add a point-mass payload at the trunk COM after link inertial scaling."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  assert isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) == 1
  body_id = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids][0]
  delta = torch.empty(len(env_ids), device=env.device).uniform_(*mass_range)
  mass = env.sim.model.body_mass[env_ids, body_id]
  env.sim.model.body_mass[env_ids, body_id] = mass + delta


def student_group_mask(
  env, student_only: bool = False, teacher_only: bool = False
) -> torch.Tensor:
  """Reserve the final quarter of training worlds for the student."""
  if student_only and teacher_only:
    raise ValueError("student_only and teacher_only cannot both be true")
  count = env.num_envs
  mask = torch.zeros(count, 1, device=env.device)
  if student_only:
    mask.fill_(1.0)
  elif not teacher_only:
    if count < 4 or count % 4 != 0:
      raise ValueError(
        "CTS training requires num_envs to be a positive multiple of four"
      )
    mask[(3 * count) // 4 :] = 1.0
  return mask


def linear_velocity_error_l2(env, command_name: str) -> torch.Tensor:
  """Dense tracking cost to retain a learning signal for large speed errors."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene["robot"].data.root_link_lin_vel_b
  return torch.sum((command[:, :2] - actual[:, :2]) ** 2, dim=-1)


def track_yaw_velocity(env, command_name: str, std: float) -> torch.Tensor:
  """Track commanded yaw without penalizing roll/pitch motion on uneven ground."""
  if std <= 0:
    raise ValueError("std must be positive")
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual_yaw = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  return torch.exp(-((command[:, 2] - actual_yaw) / std).square())


def feet_regulation(
  env,
  height_sensor_name: str,
  desired_body_height: float,
  foot_radius: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """CTS paper r_fr: penalize horizontal foot motion close to the terrain."""
  if desired_body_height <= 0:
    raise ValueError("desired_body_height must be positive")
  if foot_radius < 0:
    raise ValueError("foot_radius must be nonnegative")
  foot_heights = (env.scene[height_sensor_name].data.heights - foot_radius).clamp_min(
    0.0
  )
  foot_vel_xy = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return (
    foot_vel_xy.square().sum(dim=-1)
    * torch.exp(-foot_heights / (0.025 * desired_body_height))
  ).sum(dim=-1)


def standing_missing_foot_contacts(
  env,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Count ungrounded feet only when the commanded motion is near zero."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  standing = (
    torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
    < command_threshold
  )
  return (sensor.data.found == 0).sum(dim=-1).float() * standing


class CTSActor(MLPModel):
  """Shared policy head with privileged and history-based latent encoders."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
    self.proprio_dim = obs["actor"].shape[-1]
    self.privileged_dim = obs["critic"].shape[-1]
    self.history_dim = obs["cts_history"].shape[-1]
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.teacher_encoder = MLP(self.privileged_dim, 32, (512, 256), "elu")
    self.student_encoder = MLP(self.history_dim, 32, (512, 256), "elu")

  def _get_latent_dim(self) -> int:
    return self.proprio_dim + 32

  def embeddings(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    teacher = F.normalize(self.teacher_encoder(obs["critic"]), dim=-1)
    student = F.normalize(self.student_encoder(obs["cts_history"]), dim=-1)
    return teacher, student

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    teacher, student = self.embeddings(obs)
    # Student PPO updates the shared head; its encoder is trained by regression.
    student_mask = obs["cts_student"]
    latent = teacher * (1.0 - student_mask) + student.detach() * student_mask
    return torch.cat((self.obs_normalizer(obs["actor"]), latent), dim=-1)

  def as_onnx(self, verbose=False):
    del verbose
    return _CTSStudentExport(self)


class _CTSStudentExport(nn.Module):
  is_recurrent = False

  def __init__(self, actor: CTSActor):
    super().__init__()
    self.encoder = copy.deepcopy(actor.student_encoder)
    self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
    self.mlp = copy.deepcopy(actor.mlp)
    assert actor.distribution is not None
    self.deterministic = actor.distribution.as_deterministic_output_module()
    self.proprio_dim = actor.proprio_dim
    self.history_dim = actor.history_dim

  def forward(self, proprio: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    latent = F.normalize(self.encoder(history), dim=-1)
    normalized = self.obs_normalizer(proprio)
    return self.deterministic(self.mlp(torch.cat((normalized, latent), dim=-1)))

  def get_dummy_inputs(self):
    return torch.zeros(1, self.proprio_dim), torch.zeros(1, self.history_dim)

  @property
  def input_names(self):
    return ["proprio", "history"]

  @property
  def output_names(self):
    return ["actions"]


class CTSPPO(PPO):
  """Standard PPO on both groups, followed by student latent regression."""

  def __init__(
    self,
    *args,
    student_learning_rate=1e-3,
    student_epochs=5,
    student_batch_size=512,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    assert isinstance(self.actor, CTSActor)
    if student_learning_rate <= 0 or student_epochs < 1 or student_batch_size < 1:
      raise ValueError(
        "CTS student learning rate, epochs and batch size must be positive"
      )
    self.student_epochs = student_epochs
    self.student_batch_size = student_batch_size
    self.student_optimizer = torch.optim.Adam(
      self.actor.student_encoder.parameters(), lr=student_learning_rate
    )

  def update(self) -> dict[str, float]:
    actor = self.actor
    assert isinstance(actor, CTSActor)
    obs = self.storage.observations
    mask = obs["cts_student"].reshape(-1) > 0.5
    student_history = obs["cts_history"].reshape(-1, actor.history_dim)[mask].clone()
    privileged = obs["critic"].reshape(-1, actor.privileged_dim)[mask].clone()
    losses = super().update()
    if student_history.numel() == 0:
      return losses

    with torch.no_grad():
      targets = F.normalize(actor.teacher_encoder(privileged), dim=-1)
    total_loss = 0.0
    for _ in range(self.student_epochs):
      indices = torch.randperm(len(student_history), device=student_history.device)
      for batch_ids in indices.split(self.student_batch_size):
        prediction = F.normalize(
          actor.student_encoder(student_history[batch_ids]), dim=-1
        )
        loss = F.mse_loss(prediction, targets[batch_ids])
        self.student_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.student_encoder.parameters(), 1.0)
        self.student_optimizer.step()
        total_loss += loss.item()
    batches_per_epoch = (
      len(student_history) + self.student_batch_size - 1
    ) // self.student_batch_size
    losses["student_reconstruction"] = total_loss / (
      self.student_epochs * batches_per_epoch
    )
    return losses

  def save(self) -> dict:
    result = super().save()
    result["student_optimizer_state_dict"] = self.student_optimizer.state_dict()
    return result

  def load(self, loaded_dict, load_cfg=None, strict=True):
    restored = super().load(loaded_dict, load_cfg, strict)
    if (
      load_cfg is None or load_cfg.get("optimizer")
    ) and "student_optimizer_state_dict" in loaded_dict:
      self.student_optimizer.load_state_dict(
        loaded_dict["student_optimizer_state_dict"]
      )
    return restored
