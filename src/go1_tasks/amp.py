"""Adversarial motion prior for the Go1 MoB reference clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from torch import nn

from .amp_data import TRANSITION_DIM, AmpMotionDataset, amp_frame_from_robot


class AmpDiscriminator(nn.Module):
  """Scalar logit for each pair of adjacent 46-dimensional motion states."""

  def __init__(self) -> None:
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(TRANSITION_DIM, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    )

  def forward(self, transitions: torch.Tensor) -> torch.Tensor:
    if transitions.shape[-1] != TRANSITION_DIM:
      raise ValueError(f"AMP transition must have {TRANSITION_DIM} features")
    return self.network(transitions).squeeze(-1)


class AmpReplayBuffer:
  """Fixed-size circular store for past policy transitions."""

  def __init__(self, capacity: int, device: str) -> None:
    if capacity < 1:
      raise ValueError("Replay capacity must be positive")
    self.data = torch.empty(capacity, TRANSITION_DIM, device=device)
    self.size = 0
    self.cursor = 0

  def add(self, values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != TRANSITION_DIM:
      raise ValueError("Invalid AMP replay batch shape")
    values = values.detach().to(self.data.device)
    if len(values) >= len(self.data):
      values = values[-len(self.data) :]
    count = len(values)
    first = min(count, len(self.data) - self.cursor)
    self.data[self.cursor : self.cursor + first] = values[:first]
    if count > first:
      self.data[: count - first] = values[first:]
    self.cursor = (self.cursor + count) % len(self.data)
    self.size = min(len(self.data), self.size + count)

  def sample(self, count: int) -> torch.Tensor:
    if self.size == 0:
      raise ValueError("AMP replay buffer is empty")
    indices = torch.randint(self.size, (count,), device=self.data.device)
    return self.data[indices]


class AmpPPO(PPO):
  """PPO with online AMP reward and a separately optimized discriminator."""

  def __init__(
    self,
    *args,
    reference_path: str = "logs/amp/mob_reference/trot",
    amp_reward_weight: float = 0.5,
    discriminator_learning_rate: float = 1e-4,
    discriminator_epochs: int = 2,
    discriminator_batch_size: int = 256,
    gradient_penalty_weight: float = 5.0,
    replay_capacity: int = 100_000,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if amp_reward_weight <= 0 or discriminator_learning_rate <= 0:
      raise ValueError(
        "AMP reward weight and discriminator learning rate must be positive"
      )
    if discriminator_epochs < 1 or discriminator_batch_size < 1:
      raise ValueError("AMP discriminator epochs and batch size must be positive")
    if gradient_penalty_weight < 0:
      raise ValueError("AMP gradient penalty must be nonnegative")
    if self.is_multi_gpu:
      raise ValueError("Go1 AMP training currently supports one GPU only")
    self.reference_path = reference_path
    self.amp_reward_weight = amp_reward_weight
    self.discriminator_epochs = discriminator_epochs
    self.discriminator_batch_size = discriminator_batch_size
    self.gradient_penalty_weight = gradient_penalty_weight
    self.discriminator = AmpDiscriminator().to(self.device)
    self.discriminator_optimizer = torch.optim.Adam(
      self.discriminator.parameters(), lr=discriminator_learning_rate
    )
    self.replay = AmpReplayBuffer(replay_capacity, self.device)
    self._policy_transitions: list[torch.Tensor] = []
    self._previous_frame: torch.Tensor | None = None
    self._robot = None
    self._step_dt: float | None = None
    self.motions: AmpMotionDataset | None = None

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    assert isinstance(algorithm, AmpPPO)
    algorithm._robot = env.unwrapped.scene["robot"]
    algorithm._step_dt = env.unwrapped.step_dt
    return algorithm

  def train_mode(self) -> None:
    super().train_mode()
    if self.motions is None:
      assert self._robot is not None and self._step_dt is not None
      self.motions = AmpMotionDataset(
        self.reference_path,
        joint_names=tuple(self._robot.joint_names),
        fps=1.0 / self._step_dt,
      )
      print(
        f"[AMP] {self.motions.num_clips} clips, {len(self.motions)} transitions, "
        f"{self.motions.fps:g} Hz"
      )

  def act(self, obs):
    assert self._robot is not None and self.motions is not None
    self._previous_frame = amp_frame_from_robot(
      self._robot, joint_names=self.motions.joint_names
    ).detach()
    return super().act(obs)

  def process_env_step(self, obs, rewards, dones, extras):
    assert self._previous_frame is not None and self._robot is not None
    assert self._step_dt is not None
    next_frame = amp_frame_from_robot(self._robot)
    transitions = torch.cat((self._previous_frame, next_frame), dim=-1)
    valid = ~dones.bool().reshape(-1)
    with torch.no_grad():
      imitation_reward = F.softplus(self.discriminator(transitions))
      rewards += self.amp_reward_weight * self._step_dt * imitation_reward * valid
    if bool(valid.any()):
      self._policy_transitions.append(transitions[valid].detach())
    self._previous_frame = None
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    losses = super().update()
    if not self._policy_transitions:
      losses["amp_discriminator"] = 0.0
      return losses
    # Rollout tensors were created under torch.inference_mode; clone here so
    # autograd may use replay samples as discriminator inputs.
    policy = torch.cat(self._policy_transitions).clone()
    self._policy_transitions.clear()
    self.replay.add(policy)
    assert self.motions is not None
    total = 0.0
    updates = 0
    batches_per_epoch = max(
      1,
      (len(policy) + self.discriminator_batch_size - 1)
      // self.discriminator_batch_size,
    )
    for _ in range(self.discriminator_epochs * batches_per_epoch):
      expert = self.motions.sample(
        self.discriminator_batch_size, device=self.device
      ).requires_grad_(self.gradient_penalty_weight > 0)
      generated = self.replay.sample(self.discriminator_batch_size).clone()
      expert_logits = self.discriminator(expert)
      policy_logits = self.discriminator(generated)
      loss = F.binary_cross_entropy_with_logits(
        expert_logits, torch.ones_like(expert_logits)
      ) + F.binary_cross_entropy_with_logits(
        policy_logits, torch.zeros_like(policy_logits)
      )
      if self.gradient_penalty_weight > 0:
        gradients = torch.autograd.grad(
          expert_logits.sum(), expert, create_graph=True
        )[0]
        loss = loss + self.gradient_penalty_weight * gradients.square().sum(-1).mean()
      self.discriminator_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
      self.discriminator_optimizer.step()
      total += float(loss.detach())
      updates += 1
    losses["amp_discriminator"] = total / updates
    return losses

  def save(self) -> dict:
    saved = super().save()
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
    return saved

  def load(self, loaded_dict, load_cfg=None, strict=True):
    restored = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("amp_discriminator"):
      self.discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"], strict=strict
      )
    if load_cfg is None or load_cfg.get("optimizer"):
      self.discriminator_optimizer.load_state_dict(
        loaded_dict["amp_optimizer_state_dict"]
      )
    return restored
