"""Teacher/student networks from the robust and agile AMP paper for Go2."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor

from .robots.go2 import BASE_BODY_NAME, FOOT_GEOM_NAMES

PROPRIO_DIM = 45
PRIVILEGED_DIM = 30
TERRAIN_DIM = 187
PRIVILEGED_LATENT_DIM = 8
TERRAIN_LATENT_DIM = 16
TEACHER_LATENT_DIM = PRIVILEGED_LATENT_DIM + TERRAIN_LATENT_DIM
LSTM_HIDDEN_DIM = 256
ACTION_DIM = 12
SEQUENCE_LENGTH = 50


def _contact_indicator(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[name]
  assert sensor.data.found is not None
  return (
    sensor.data.found.reshape(env.num_envs, -1).amax(dim=1, keepdim=True) > 0
  ).float()


def paper_privileged_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return 30 simulation-only values useful for dynamics/terrain inference.

  The paper names the categories but does not publish their per-field packing.
  This Go2 packing keeps the stated 30-D contract and exposes quantities that
  are actually randomized or sensed in this environment: velocity, contact
  wrench/contact state, friction, inertial parameters and collision state.
  """
  robot = env.scene["robot"]
  model = env.sim.model
  defaults = env.sim.get_default_field

  foot_ids, foot_names = robot.find_geoms(FOOT_GEOM_NAMES, preserve_order=True)
  if tuple(foot_names) != FOOT_GEOM_NAMES:
    raise ValueError(f"Unexpected Go2 foot order: {foot_names}")
  foot_ids_t = robot.data.indexing.geom_ids[foot_ids]
  friction = model.geom_friction[:, foot_ids_t[0], :3]

  feet: ContactSensor = env.scene["feet_ground_contact"]
  assert feet.data.force is not None and feet.data.found is not None
  forces = torch.sign(feet.data.force.flatten(1)) * torch.log1p(
    feet.data.force.flatten(1).abs()
  )
  contacts = (feet.data.found.reshape(env.num_envs, -1) > 0).float()

  body_ids, _ = robot.find_bodies(BASE_BODY_NAME, preserve_order=True)
  body_id = robot.data.indexing.body_ids[body_ids[0]]
  mass_delta = (
    model.body_mass[:, body_id] - defaults("body_mass")[body_id]
  ).unsqueeze(-1)
  com_delta = model.body_ipos[:, body_id] - defaults("body_ipos")[body_id]
  ctrl_ids = robot.data.indexing.ctrl_ids
  motor_strength = (
    model.actuator_gainprm[:, ctrl_ids, 0]
    / defaults("actuator_gainprm")[ctrl_ids, 0]
  ).mean(dim=1, keepdim=True)
  collisions = torch.cat(
    (
      _contact_indicator(env, "trunk_ground_touch"),
      _contact_indicator(env, "thigh_ground_touch"),
      _contact_indicator(env, "shank_ground_touch"),
    ),
    dim=-1,
  )
  state = torch.cat(
    (
      robot.data.root_link_lin_vel_b,
      forces,
      contacts,
      friction,
      mass_delta,
      com_delta,
      motor_strength,
      collisions,
    ),
    dim=-1,
  )
  if state.shape[-1] != PRIVILEGED_DIM:
    raise RuntimeError(
      f"Expected {PRIVILEGED_DIM} privileged values, got {state.shape[-1]}"
    )
  return state


class PaperAmpTeacherActor(MLPModel):
  """Paper teacher: 45 proprio + encoded 30 privileged + encoded 187 terrain."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, bound_action_mean=True, **kwargs) -> None:
    self.bound_action_mean = bound_action_mean
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    if obs["actor"].shape[-1] != PROPRIO_DIM:
      raise ValueError("Paper AMP teacher requires a 45-D proprioceptive observation")
    if obs["privileged"].shape[-1] != PRIVILEGED_DIM:
      raise ValueError("Paper AMP teacher requires a 30-D privileged state")
    if obs["terrain"].shape[-1] != TERRAIN_DIM:
      raise ValueError("Paper AMP teacher requires a 187-D terrain scan")
    self.privileged_encoder = MLP(
      PRIVILEGED_DIM, PRIVILEGED_LATENT_DIM, (64, 32), "elu"
    )
    self.terrain_encoder = MLP(TERRAIN_DIM, TERRAIN_LATENT_DIM, (256, 128), "elu")
    # Table III specifies a tanh action mean after the 256/128/64 low-level MLP.
    assert self.distribution is not None
    self.mlp = MLP(
      PROPRIO_DIM + TEACHER_LATENT_DIM,
      self.distribution.input_dim,
      (256, 128, 64),
      "elu",
      last_activation="tanh" if self.bound_action_mean else None,
    )
    self.distribution.init_mlp_weights(self.mlp)

  def _get_latent_dim(self) -> int:
    return PROPRIO_DIM + TEACHER_LATENT_DIM

  def encode_privileged(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat(
      (
        self.privileged_encoder(obs["privileged"]),
        self.terrain_encoder(obs["terrain"]),
      ),
      dim=-1,
    )

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    del masks, hidden_state
    return torch.cat((obs["actor"], self.encode_privileged(obs)), dim=-1)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _TeacherExport(self)


class _TeacherExport(nn.Module):
  is_recurrent = False

  def __init__(self, teacher: PaperAmpTeacherActor) -> None:
    super().__init__()
    self.privileged_encoder = copy.deepcopy(teacher.privileged_encoder)
    self.terrain_encoder = copy.deepcopy(teacher.terrain_encoder)
    self.bound_action_mean = teacher.bound_action_mean
    self.low_level = copy.deepcopy(teacher.mlp)
    assert teacher.distribution is not None
    self.output = teacher.distribution.as_deterministic_output_module()

  def forward(self, proprio, privileged, terrain):
    latent = torch.cat(
      (self.privileged_encoder(privileged), self.terrain_encoder(terrain)), dim=-1
    )
    return self.output(self.low_level(torch.cat((proprio, latent), dim=-1)))

  def get_dummy_inputs(self):
    return (
      torch.zeros(1, PROPRIO_DIM),
      torch.zeros(1, PRIVILEGED_DIM),
      torch.zeros(1, TERRAIN_DIM),
    )

  @property
  def input_names(self):
    return ["proprio", "privileged", "terrain"]

  @property
  def output_names(self):
    return ["actions"]


class PaperAmpStudent(nn.Module):
  """Deployable student with explicit recurrent state inputs and outputs."""

  def __init__(self) -> None:
    super().__init__()
    self.lstm = nn.LSTM(PROPRIO_DIM, LSTM_HIDDEN_DIM, batch_first=True)
    self.memory_encoder = MLP(
      LSTM_HIDDEN_DIM, TEACHER_LATENT_DIM, (256, 128), "elu"
    )
    self.low_level = MLP(
      PROPRIO_DIM + TEACHER_LATENT_DIM,
      ACTION_DIM,
      (256, 128, 64),
      "elu",
      last_activation="tanh",
    )

  def copy_low_level_from(self, teacher: PaperAmpTeacherActor) -> None:
    self.bound_action_mean = teacher.bound_action_mean
    self.low_level = copy.deepcopy(teacher.mlp)

  @staticmethod
  def initial_state(batch_size: int, device: torch.device | str):
    shape = (1, batch_size, LSTM_HIDDEN_DIM)
    return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

  def step(self, proprio: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
    output, (next_hidden, next_cell) = self.lstm(
      proprio.unsqueeze(1), (hidden, cell)
    )
    latent = self.memory_encoder(output[:, 0])
    action = self.low_level(torch.cat((proprio, latent), dim=-1))
    return action, latent, next_hidden, next_cell

  def forward_sequence(
    self, proprio: torch.Tensor, dones: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Process [B,T,45] sequences, resetting state after terminal steps."""
    batch, steps, dim = proprio.shape
    if dim != PROPRIO_DIM or dones.shape != (batch, steps):
      raise ValueError("Invalid student sequence shape")
    hidden, cell = self.initial_state(batch, proprio.device)
    actions, latents = [], []
    for index in range(steps):
      if index:
        keep = (~dones[:, index - 1]).float().view(1, batch, 1)
        hidden, cell = hidden * keep, cell * keep
      action, latent, hidden, cell = self.step(proprio[:, index], hidden, cell)
      actions.append(action)
      latents.append(latent)
    return torch.stack(actions, dim=1), torch.stack(latents, dim=1)

  def as_onnx(self) -> nn.Module:
    return _StudentExport(self)


class _StudentExport(nn.Module):
  """Stateless ONNX wrapper; the controller owns and feeds back h/c."""

  def __init__(self, student: PaperAmpStudent) -> None:
    super().__init__()
    self.student = copy.deepcopy(student)

  def forward(self, proprio, hidden_state, cell_state):
    action, _, next_hidden, next_cell = self.student.step(
      proprio, hidden_state, cell_state
    )
    return action, next_hidden, next_cell

  def get_dummy_inputs(self):
    return (
      torch.zeros(1, PROPRIO_DIM),
      torch.zeros(1, 1, LSTM_HIDDEN_DIM),
      torch.zeros(1, 1, LSTM_HIDDEN_DIM),
    )

  @property
  def input_names(self):
    return ["proprio", "hidden_state", "cell_state"]

  @property
  def output_names(self):
    return ["actions", "next_hidden_state", "next_cell_state"]


def export_student_onnx(student: PaperAmpStudent, path: Path) -> None:
  wrapper = student.as_onnx().cpu().eval()
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    wrapper,
    wrapper.get_dummy_inputs(),
    str(path),
    opset_version=18,
    input_names=wrapper.input_names,
    output_names=wrapper.output_names,
    dynamic_axes={
      "proprio": {0: "batch"},
      "hidden_state": {1: "batch"},
      "cell_state": {1: "batch"},
      "actions": {0: "batch"},
      "next_hidden_state": {1: "batch"},
      "next_cell_state": {1: "batch"},
    },
    dynamo=False,
  )


__all__ = [
  "ACTION_DIM",
  "LSTM_HIDDEN_DIM",
  "PRIVILEGED_DIM",
  "PROPRIO_DIM",
  "PaperAmpStudent",
  "PaperAmpTeacherActor",
  "SEQUENCE_LENGTH",
  "TEACHER_LATENT_DIM",
  "TERRAIN_DIM",
  "export_student_onnx",
  "paper_privileged_state",
]
