"""Hybrid Internal Model (HIMLoco, ICLR 2024) for Unitree Go2.

The upstream implementation lives in ``OpenRobotLab/HIMLoco`` and is built on
Isaac Gym + the legacy rsl_rl.  This module ports the parts that define the
method onto mjlab's rsl-rl 5.x models/PPO:

* a history encoder that regresses the body linear velocity and produces a
  16-D SwAV embedding (the "hybrid internal model"),
* a policy MLP that consumes ``[current proprio, estimated velocity, latent]``,
* a PPO variant that owns the estimator's separate optimizer.

Deviations from the reference code are listed in ``docs/go2_him.md``.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

PROPRIO_DIM = 45
"""Observation sent to the policy at one control step."""
VELOCITY_DIM = 3
"""Body-frame linear velocity regressed by the estimator."""
COMMAND_DIM = 3
"""Leading dimensions of ``PROPRIO_DIM`` that hold the velocity command."""
LATENT_DIM = 16
"""Dimension of the SwAV embedding."""
HISTORY_STEPS = 6
"""Number of stacked observation steps fed to the estimator."""
NUM_PROTOTYPE = 32
"""Number of SwAV prototypes (clusters) in the estimator."""
HIM_CONTRACT_VERSION = 2
"""Checkpoint contract after history, SwAV gradient, and observation-scale fixes."""
TEMPERATURE = 3.0
"""Softmax temperature used by the swapped-prediction loss."""
TERM_DIMS = (3, 3, 3, 12, 12, 12)
"""Per-term sizes of one proprio step, in the env's observation-group order."""
PRIVILEGED_DIM = PROPRIO_DIM + VELOCITY_DIM + 3 + 187
"""Critic observation: proprio + lin. vel. + disturbance proxy + height scan."""


def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  """SwAV's doubly-stochastic normalisation of the prototype assignment.

  Each prototype must carry the same total mass across the batch and each
  sample must put the same total mass on the prototypes.  Running this on a
  batch (rather than per sample) is what makes the loss "contrastive": the
  encoder has to spread samples over the prototypes instead of collapsing.
  """
  q = torch.exp(out / eps).T
  num_prototypes, batch = q.shape
  q /= q.sum()
  for _ in range(iters):
    q /= torch.sum(q, dim=1, keepdim=True)
    q /= num_prototypes
    q /= torch.sum(q, dim=0, keepdim=True)
    q /= batch
  return (q * batch).T


class HIMEstimator(nn.Module):
  """History encoder + successor-state target network + SwAV prototypes."""

  def __init__(
    self,
    history_steps: int = HISTORY_STEPS,
    one_step_dim: int = PROPRIO_DIM,
    latent_dim: int = LATENT_DIM,
    velocity_dim: int = VELOCITY_DIM,
    command_dim: int = COMMAND_DIM,
    encoder_hidden_dims: tuple[int, ...] = (128, 64),
    target_hidden_dims: tuple[int, ...] = (128, 64),
    num_prototype: int = NUM_PROTOTYPE,
    temperature: float = TEMPERATURE,
    activation: str = "elu",
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
  ) -> None:
    super().__init__()
    if command_dim != velocity_dim:
      raise ValueError(
        "HIM's successor-state target drops the command and appends the "
        "privileged velocity, so both blocks must have the same size"
      )
    self.history_steps = history_steps
    self.one_step_dim = one_step_dim
    self.latent_dim = latent_dim
    self.velocity_dim = velocity_dim
    self.command_dim = command_dim
    self.target_input_dim = one_step_dim - command_dim + velocity_dim
    self.num_prototype = num_prototype
    self.temperature = temperature
    self.max_grad_norm = max_grad_norm

    act = _resolve_activation(activation)

    encoder_layers: list[nn.Module] = []
    width = history_steps * one_step_dim
    for hidden in encoder_hidden_dims:
      encoder_layers += [nn.Linear(width, hidden), act()]
      width = hidden
    encoder_layers.append(nn.Linear(width, velocity_dim + latent_dim))
    self.encoder = nn.Sequential(*encoder_layers)

    target_layers: list[nn.Module] = []
    width = self.target_input_dim
    for hidden in target_hidden_dims:
      target_layers += [nn.Linear(width, hidden), act()]
      width = hidden
    target_layers.append(nn.Linear(width, latent_dim))
    self.target = nn.Sequential(*target_layers)

    self.proto = nn.Embedding(num_prototype, latent_dim)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

  def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the encoder output into estimated velocity and unit-norm latent."""
    parts = self.encoder(obs_history.detach())
    velocity = parts[..., : self.velocity_dim]
    latent = F.normalize(parts[..., self.velocity_dim :], dim=-1, p=2)
    return velocity, latent

  def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference path: the estimator is never differentiated through."""
    with torch.no_grad():
      return self.encode(obs_history)

  @staticmethod
  def successor_state(
    next_privileged: torch.Tensor,
    one_step_dim: int = PROPRIO_DIM,
    command_dim: int = COMMAND_DIM,
    velocity_dim: int = VELOCITY_DIM,
  ) -> torch.Tensor:
    """Build the target-network input from one privileged observation.

    The upstream code slices ``next_critic_obs[:, 3:48]``, i.e. the next
    proprioceptive observation *without the command* plus the next body
    linear velocity.  The command is excluded because the successor state of
    the robot does not depend on what was commanded; the privileged velocity
    is included because it is part of the response the estimator must infer.
    """
    return torch.cat(
      (
        next_privileged[..., command_dim:one_step_dim],
        next_privileged[..., one_step_dim : one_step_dim + velocity_dim],
      ),
      dim=-1,
    )

  def update(
    self,
    obs_history: torch.Tensor,
    next_privileged: torch.Tensor,
    lr: float | None = None,
  ) -> tuple[float, float]:
    """One estimator step: velocity regression + SwAV swapped prediction."""
    if lr is not None:
      for group in self.optimizer.param_groups:
        group["lr"] = lr

    with torch.no_grad():
      velocity_target = next_privileged[
        ..., self.one_step_dim : self.one_step_dim + self.velocity_dim
      ]
      success = self.successor_state(
        next_privileged, self.one_step_dim, self.command_dim, self.velocity_dim
      )

    prediction = self.encoder(obs_history)
    predicted_velocity = prediction[..., : self.velocity_dim]
    source = F.normalize(prediction[..., self.velocity_dim :], dim=-1, p=2)
    target = F.normalize(self.target(success), dim=-1, p=2)

    with torch.no_grad():
      self.proto.weight.copy_(F.normalize(self.proto.weight.data, dim=-1, p=2))
    score_source = source @ self.proto.weight.T
    score_target = target @ self.proto.weight.T
    with torch.no_grad():
      code_source = sinkhorn(score_source)
      code_target = sinkhorn(score_target)

    log_p_source = F.log_softmax(score_source / self.temperature, dim=-1)
    log_p_target = F.log_softmax(score_target / self.temperature, dim=-1)
    swap_loss = -0.5 * (
      code_source * log_p_target + code_target * log_p_source
    ).mean()
    estimation_loss = F.mse_loss(predicted_velocity, velocity_target)

    total = estimation_loss + swap_loss
    # A single non-finite sample would poison the encoder for the rest of the
    # run, so skip the step instead of letting Adam ingest NaN/Inf gradients.
    if not torch.isfinite(total):
      self.optimizer.zero_grad()
      return float("nan"), float("nan")

    self.optimizer.zero_grad()
    total.backward()
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return estimation_loss.item(), swap_loss.item()


class HIMActor(MLPModel):
  """Policy that consumes the current observation plus the HIM embedding.

  The actor observation group is a flattened history with the *oldest* step
  first (mjlab's convention).  HIMLoco stores the newest step first, so the
  history is flipped before it reaches the estimator; deployments therefore
  feed a newest-first ring buffer, exactly like the reference code.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    num_one_step_obs: int = PROPRIO_DIM,
    history_steps: int = HISTORY_STEPS,
    term_dims: tuple[int, ...] = TERM_DIMS,
    latent_dim: int = LATENT_DIM,
    velocity_dim: int = VELOCITY_DIM,
    command_dim: int = COMMAND_DIM,
    num_prototype: int = NUM_PROTOTYPE,
    temperature: float = TEMPERATURE,
    estimator_learning_rate: float = 1e-3,
    estimator_max_grad_norm: float = 10.0,
    obs_normalization: bool = False,
    **kwargs,
  ) -> None:
    if obs_normalization:
      raise ValueError("HIMActor does not support observation normalisation")
    self.num_one_step_obs = num_one_step_obs
    self.history_steps = history_steps
    self.term_dims = tuple(term_dims)
    if sum(self.term_dims) != num_one_step_obs:
      raise ValueError(
        f"term_dims {self.term_dims} do not add up to {num_one_step_obs}"
      )
    self.latent_dim = latent_dim
    self.velocity_dim = velocity_dim
    self.command_dim = command_dim
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      obs_normalization=False,
      **kwargs,
    )
    expected = history_steps * num_one_step_obs
    if self.obs_dim != expected:
      raise ValueError(
        f"HIMActor expects a {expected}-D observation history, got {self.obs_dim}"
      )
    self.estimator = HIMEstimator(
      history_steps=history_steps,
      one_step_dim=num_one_step_obs,
      latent_dim=latent_dim,
      velocity_dim=velocity_dim,
      command_dim=command_dim,
      num_prototype=num_prototype,
      temperature=temperature,
      learning_rate=estimator_learning_rate,
      max_grad_norm=estimator_max_grad_norm,
    )

  def _get_latent_dim(self) -> int:
    return self.num_one_step_obs + self.velocity_dim + self.latent_dim

  def frame_major_history(self, obs: TensorDict) -> torch.Tensor:
    """Rebuild ``[batch, history, proprio]`` from mjlab's term-major layout.

    A group with ``concatenate_terms=True`` and ``flatten_history_dim=True``
    stores each term's history contiguously, i.e.
    ``[command_t-H..t, ang_vel_t-H..t, ...]``.  The encoder and the policy need
    one observation per time step, so the blocks are split, reshaped and
    interleaved back into ``[t-H .. t]`` frame order.
    """
    flat = obs["actor"]
    dims = self.term_dims
    chunks = flat.split([dim * self.history_steps for dim in dims], dim=-1)
    frames = [
      chunk.reshape(flat.shape[0], self.history_steps, dim)
      for chunk, dim in zip(chunks, dims)
    ]
    return torch.cat(frames, dim=-1)

  def newest_first_history(self, obs: TensorDict) -> torch.Tensor:
    """Return the observation history in HIMLoco order (newest step first)."""
    return self.frame_major_history(obs).flip(1).flatten(1)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    history = self.newest_first_history(obs)
    velocity, latent = self.estimator(history)
    current = history[:, : self.num_one_step_obs]
    return torch.cat((current, velocity, latent), dim=-1)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _HIMExport(self)

  def as_jit(self) -> nn.Module:
    return _HIMExport(self)


class _HIMExport(nn.Module):
  """Deployable ``history -> action`` wrapper around estimator + policy head.

  Input: ``[batch, history_steps * 45]``, frame-major, newest step first.
  """

  is_recurrent = False

  def __init__(self, actor: HIMActor) -> None:
    super().__init__()
    self.encoder = copy.deepcopy(actor.estimator.encoder)
    self.mlp = copy.deepcopy(actor.mlp)
    assert actor.distribution is not None
    self.deterministic_output = actor.distribution.as_deterministic_output_module()
    self.history_steps = actor.history_steps
    self.num_one_step_obs = actor.num_one_step_obs
    self.velocity_dim = actor.velocity_dim

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    # Deployment convention (same as HIMLoco): the flat history is frame-major
    # with the **newest** observation first.
    flat = history.reshape(
      history.shape[0], self.history_steps * self.num_one_step_obs
    )
    parts = self.encoder(flat)
    velocity = parts[..., : self.velocity_dim]
    latent = F.normalize(parts[..., self.velocity_dim :], dim=-1, p=2)
    current = flat[:, : self.num_one_step_obs]
    return self.deterministic_output(
      self.mlp(torch.cat((current, velocity, latent), dim=-1))
    )

  def reset(self) -> None:
    """No hidden state to reset; kept for parity with rsl-rl exports."""

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (
      torch.zeros(1, self.history_steps * self.num_one_step_obs),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class HIMPPO(PPO):
  """PPO plus the HIM estimator's own optimiser and successor-state buffer."""

  def __init__(
    self,
    *args,
    estimator_epochs: int = 4,
    estimator_batch_size: int = 512,
    estimator_lr_follows_policy: bool = True,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    assert isinstance(self.actor, HIMActor)
    if estimator_epochs < 1 or estimator_batch_size < 1:
      raise ValueError("estimator_epochs and estimator_batch_size must be positive")
    self.estimator_epochs = estimator_epochs
    self.estimator_batch_size = estimator_batch_size
    self.estimator_lr_follows_policy = estimator_lr_follows_policy

    steps = self.storage.num_transitions_per_env
    envs = self.storage.num_envs
    self._history_buffer = torch.zeros(
      steps, envs, self.actor.obs_dim, device=self.device
    )
    self._next_privileged_buffer = torch.zeros(
      steps, envs, self.critic.obs_dim, device=self.device
    )
    self._done_buffer = torch.zeros(steps, envs, 1, device=self.device)
    self._rollout_index = 0

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample actions and remember the history that produced them."""
    if self._rollout_index >= self._history_buffer.shape[0]:
      raise OverflowError("Estimator buffer overflow; call update() first")
    actions = super().act(obs)
    self._history_buffer[self._rollout_index].copy_(
      self.actor.newest_first_history(obs)
    )
    return actions

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    """Record the successor privileged observation before the storage step."""
    index = self._rollout_index
    self._next_privileged_buffer[index].copy_(obs["critic"])
    self._done_buffer[index].copy_(dones.reshape(-1, 1).float())
    self._rollout_index += 1
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    """Run PPO, then train the estimator on the collected transitions.

    HIMLoco interleaves one estimator step with every PPO mini-batch.  Doing
    the estimator epochs after the PPO update (the same pattern this repo uses
    for the CTS student encoder) keeps PPO's implementation untouched while
    optimising the same objective on the same data.
    """
    steps = self._rollout_index
    if steps == 0:
      raise RuntimeError("update() called before any transition was collected")
    history = self._history_buffer[:steps].flatten(0, 1)
    next_privileged = self._next_privileged_buffer[:steps].flatten(0, 1)
    dones = self._done_buffer[:steps].flatten(0, 1)
    self._rollout_index = 0

    losses = super().update()
    losses.update(self._update_estimator(history, next_privileged, dones))
    return losses

  def _update_estimator(
    self,
    history: torch.Tensor,
    next_privileged: torch.Tensor,
    dones: torch.Tensor,
  ) -> dict[str, float]:
    # mjlab resets terminated worlds inside ``step``, so the privileged
    # observation recorded for the step *after* a done flag is already the
    # reset state.  HIMLoco instead stores the terminal observation; dropping
    # these few samples is the closest faithful alternative.
    keep = dones.reshape(-1) < 0.5
    keep &= torch.isfinite(history).all(dim=-1)
    keep &= torch.isfinite(next_privileged).all(dim=-1)
    count = int(keep.sum())
    if count == 0:
      return {"estimator_velocity": 0.0, "estimator_swap": 0.0}
    history = history[keep]
    next_privileged = next_privileged[keep]

    estimator = self.actor.estimator
    lr = self.learning_rate if self.estimator_lr_follows_policy else None
    total_velocity = 0.0
    total_swap = 0.0
    batches = 0
    for _ in range(self.estimator_epochs):
      order = torch.randperm(count, device=history.device)
      for chunk in order.split(min(self.estimator_batch_size, count)):
        velocity_loss, swap_loss = estimator.update(
          history[chunk], next_privileged[chunk], lr=lr
        )
        if math.isnan(velocity_loss):
          continue
        total_velocity += velocity_loss
        total_swap += swap_loss
        batches += 1
    if batches == 0:
      return {"estimator_velocity": 0.0, "estimator_swap": 0.0}
    return {
      "estimator_velocity": total_velocity / batches,
      "estimator_swap": total_swap / batches,
    }

  def save(self) -> dict:
    saved = super().save()
    saved["him_contract_version"] = HIM_CONTRACT_VERSION
    saved["estimator_optimizer_state_dict"] = (
      self.actor.estimator.optimizer.state_dict()
    )
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if loaded_dict.get("him_contract_version") != HIM_CONTRACT_VERSION:
      raise ValueError(
        "Incompatible HIM checkpoint: pre-fix observation history/scales or "
        "SwAV training. Train a new Go2-Velocity-HIM policy."
      )
    restored = super().load(loaded_dict, load_cfg, strict)
    if (
      (load_cfg is None or load_cfg.get("optimizer"))
      and "estimator_optimizer_state_dict" in loaded_dict
    ):
      self.actor.estimator.optimizer.load_state_dict(
        loaded_dict["estimator_optimizer_state_dict"]
      )
    return restored


def _resolve_activation(name: str) -> type[nn.Module]:
  activations = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
  }
  try:
    return activations[name]
  except KeyError as exc:
    raise ValueError(f"Unknown activation: {name}") from exc


__all__ = [
  "COMMAND_DIM",
  "HIMActor",
  "HIMEstimator",
  "HIMPPO",
  "HISTORY_STEPS",
  "HIM_CONTRACT_VERSION",
  "LATENT_DIM",
  "NUM_PROTOTYPE",
  "PRIVILEGED_DIM",
  "PROPRIO_DIM",
  "TEMPERATURE",
  "TERM_DIMS",
  "VELOCITY_DIM",
  "sinkhorn",
]
