"""Adversarial motion prior for the Go1 MoB reference clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from torch import autograd, nn

from .amp_data import (
  FOOT_NAMES,
  FRAME_DIM,
  TRANSITION_DIM,
  AmpMotionDataset,
  amp_frame_from_robot,
)


class AmpDiscriminator(nn.Module):
  """Scalar logit for each pair of adjacent 46-dimensional motion states.

  The output is a raw logit in the least-squares GAN formulation used by AMP:
  expert transitions target +1 and policy transitions target -1. Finite targets
  keep the discriminator from becoming infinitely confident, which would make
  the style reward vanish.
  """

  def __init__(self, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    in_dim = TRANSITION_DIM
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
      in_dim = hidden_dim
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(in_dim, 1)

  def forward(self, transitions: torch.Tensor) -> torch.Tensor:
    if transitions.shape[-1] != TRANSITION_DIM:
      raise ValueError(f"AMP transition must have {TRANSITION_DIM} features")
    return self.head(self.trunk(transitions)).squeeze(-1)

  def compute_grad_pen(
    self, transitions: torch.Tensor, weight: float
  ) -> torch.Tensor:
    """Penalize the squared input gradient on expert transitions (AMP paper)."""
    transitions = transitions.detach().requires_grad_(True)
    logits = self.forward(transitions)
    gradients = autograd.grad(
      logits.sum(), transitions, create_graph=True, retain_graph=True
    )[0]
    return weight * gradients.square().sum(-1).mean()


class AmpFeatureNormalizer:
  """Running per-feature mean/std for AMP states.

  The per-frame features have very different scales (body height ~0.3 m vs
  joint velocity ~8 rad/s). Normalizing keeps the discriminator from relying
  only on the largest dimensions. Statistics are initialized from the expert
  motion dataset and then updated with both expert and policy samples.
  """

  def __init__(self, dim: int, device: str) -> None:
    self.mean = torch.zeros(dim, device=device)
    self.var = torch.ones(dim, device=device)
    self.count = 0

  @torch.no_grad()
  def initialize(self, frames: torch.Tensor) -> None:
    if frames.shape[0] < 1:
      raise ValueError("Cannot initialize the AMP normalizer from no samples")
    self.mean.copy_(frames.mean(0))
    self.var.copy_(frames.var(0, unbiased=False))
    self.count = int(frames.shape[0])

  @torch.no_grad()
  def update(self, frames: torch.Tensor) -> None:
    if self.count == 0:
      self.initialize(frames)
      return
    batch_mean = frames.mean(0)
    batch_var = frames.var(0, unbiased=False)
    batch_count = frames.shape[0]
    delta = batch_mean - self.mean
    total = self.count + batch_count
    self.mean += delta * batch_count / total
    m2 = (
      self.var * self.count
      + batch_var * batch_count
      + delta.square() * self.count * batch_count / total
    )
    self.var = m2 / total
    self.count = total

  def normalize(self, frames: torch.Tensor) -> torch.Tensor:
    # Floor the variance so constant features (common in short test clips) do
    # not blow up the normalized values.
    std = torch.sqrt(self.var.clamp_min(1e-4))
    return ((frames - self.mean) / (std + 1e-8)).clamp(-10.0, 10.0)

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {
      "mean": self.mean,
      "var": self.var,
      "count": torch.tensor(self.count, device=self.mean.device),
    }

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.mean.copy_(state["mean"])
    self.var.copy_(state["var"])
    self.count = int(state["count"])


def amp_style_reward(logits: torch.Tensor) -> torch.Tensor:
  """Bounded AMP reward in [0, 1], maximal at a discriminator logit of +1."""
  return torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)


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
    amp_reward_weight: float = 2.0,
    discriminator_learning_rate: float = 1e-4,
    discriminator_epochs: int = 1,
    discriminator_batch_size: int = 1024,
    discriminator_batches_per_iteration: int = 8,
    gradient_penalty_weight: float = 10.0,
    replay_capacity: int = 100_000,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if amp_reward_weight <= 0 or discriminator_learning_rate <= 0:
      raise ValueError(
        "AMP reward weight and discriminator learning rate must be positive"
      )
    if (
      discriminator_epochs < 1
      or discriminator_batch_size < 1
      or discriminator_batches_per_iteration < 1
    ):
      raise ValueError(
        "AMP discriminator update counts and batch size must be positive"
      )
    if gradient_penalty_weight < 0:
      raise ValueError("AMP gradient penalty must be nonnegative")
    if self.is_multi_gpu:
      raise ValueError("Go1 AMP training currently supports one GPU only")
    self.reference_path = reference_path
    self.amp_reward_weight = amp_reward_weight
    self.discriminator_epochs = discriminator_epochs
    self.discriminator_batch_size = discriminator_batch_size
    self.discriminator_batches_per_iteration = discriminator_batches_per_iteration
    self.gradient_penalty_weight = gradient_penalty_weight
    self.discriminator = AmpDiscriminator().to(self.device)
    self.discriminator.eval()
    self.discriminator_optimizer = torch.optim.Adam(
      self.discriminator.parameters(), lr=discriminator_learning_rate
    )
    self.replay = AmpReplayBuffer(replay_capacity, self.device)
    self._policy_transitions: list[torch.Tensor] = []
    self._previous_frame: torch.Tensor | None = None
    self._robot = None
    self._step_dt: float | None = None
    self._foot_ids: list[int] | None = None
    self.motions: AmpMotionDataset | None = None
    self.amp_normalizer: AmpFeatureNormalizer | None = None
    self._style_reward_values: list[torch.Tensor] = []

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    assert isinstance(algorithm, AmpPPO)
    robot = env.unwrapped.scene["robot"]
    algorithm._robot = robot
    algorithm._step_dt = env.unwrapped.step_dt
    foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != FOOT_NAMES:
      raise ValueError(f"Unexpected foot order: {foot_names}")
    algorithm._foot_ids = foot_ids
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
    if self.amp_normalizer is None:
      frames = torch.cat(
        (
          self.motions.transitions[:, :FRAME_DIM],
          self.motions.transitions[:, FRAME_DIM:],
        ),
        dim=0,
      )
      self.amp_normalizer = AmpFeatureNormalizer(FRAME_DIM, self.device)
      self.amp_normalizer.initialize(frames)

  def _normalize_transitions(self, transitions: torch.Tensor) -> torch.Tensor:
    assert self.amp_normalizer is not None
    return torch.cat(
      (
        self.amp_normalizer.normalize(transitions[..., :FRAME_DIM]),
        self.amp_normalizer.normalize(transitions[..., FRAME_DIM:]),
      ),
      dim=-1,
    )

  def _update_normalizer(self, transitions: torch.Tensor) -> None:
    assert self.amp_normalizer is not None
    frames = torch.cat(
      (transitions[:, :FRAME_DIM], transitions[:, FRAME_DIM:]), dim=0
    )
    self.amp_normalizer.update(frames.detach())

  def act(self, obs):
    assert self._robot is not None and self.motions is not None
    self._previous_frame = amp_frame_from_robot(
      self._robot,
      joint_names=self.motions.joint_names,
      foot_ids=self._foot_ids,
    ).detach()
    return super().act(obs)

  def process_env_step(self, obs, rewards, dones, extras):
    assert self._previous_frame is not None and self._robot is not None
    assert self._step_dt is not None
    next_frame = amp_frame_from_robot(self._robot, foot_ids=self._foot_ids)
    transitions = torch.cat((self._previous_frame, next_frame), dim=-1)
    valid = ~dones.bool().reshape(-1)
    with torch.no_grad():
      logits = self.discriminator(self._normalize_transitions(transitions))
      imitation_reward = amp_style_reward(logits)
      rewards += self.amp_reward_weight * self._step_dt * imitation_reward * valid
    if bool(valid.any()):
      self._policy_transitions.append(transitions[valid].detach())
      self._style_reward_values.append(imitation_reward[valid].detach())
    self._previous_frame = None
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    losses = super().update()
    losses["amp_style_reward_raw"] = (
      float(torch.cat(self._style_reward_values).mean())
      if self._style_reward_values
      else 0.0
    )
    assert self._step_dt is not None
    losses["amp_style_reward_per_step"] = (
      self.amp_reward_weight * self._step_dt * losses["amp_style_reward_raw"]
    )
    self._style_reward_values.clear()
    if not self._policy_transitions:
      losses["amp_discriminator"] = 0.0
      losses["amp_expert_logit"] = 0.0
      losses["amp_policy_logit"] = 0.0
      return losses
    # Rollout tensors were created under torch.inference_mode; clone here so
    # autograd may use replay samples as discriminator inputs.
    policy = torch.cat(self._policy_transitions).clone()
    self._policy_transitions.clear()
    self.replay.add(policy)
    assert self.motions is not None
    assert self.amp_normalizer is not None
    total = 0.0
    updates = 0
    expert_logit_sum = 0.0
    policy_logit_sum = 0.0
    self.discriminator.train()
    ones = torch.ones(
      self.discriminator_batch_size, device=self.device
    )
    for _ in range(
      self.discriminator_epochs * self.discriminator_batches_per_iteration
    ):
      expert_raw = self.motions.sample(
        self.discriminator_batch_size, device=self.device
      )
      policy_raw = self.replay.sample(self.discriminator_batch_size).clone()
      expert = self._normalize_transitions(expert_raw)
      policy = self._normalize_transitions(policy_raw)
      expert_logits = self.discriminator(expert)
      policy_logits = self.discriminator(policy)
      # Least-squares GAN targets. Unlike BCE these are finite, so the
      # discriminator stops pushing its logits towards infinity.
      loss = 0.5 * (
        F.mse_loss(expert_logits, ones)
        + F.mse_loss(policy_logits, -ones)
      )
      if self.gradient_penalty_weight > 0:
        loss = loss + self.discriminator.compute_grad_pen(
          expert_raw, self.gradient_penalty_weight
        )
      self.discriminator_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
      self.discriminator_optimizer.step()
      self._update_normalizer(expert_raw)
      self._update_normalizer(policy_raw)
      total += float(loss.detach())
      expert_logit_sum += float(expert_logits.detach().mean())
      policy_logit_sum += float(policy_logits.detach().mean())
      updates += 1
    self.discriminator.eval()
    losses["amp_discriminator"] = total / updates
    losses["amp_expert_logit"] = expert_logit_sum / updates
    losses["amp_policy_logit"] = policy_logit_sum / updates
    return losses

  def save(self) -> dict:
    saved = super().save()
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
    if self.amp_normalizer is not None:
      saved["amp_normalizer_state"] = self.amp_normalizer.state_dict()
    return saved

  def load(self, loaded_dict, load_cfg=None, strict=True):
    restored = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("amp_discriminator"):
      self.discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"], strict=strict
      )
      if "amp_normalizer_state" in loaded_dict:
        if self.amp_normalizer is None:
          self.amp_normalizer = AmpFeatureNormalizer(FRAME_DIM, self.device)
        self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state"])
    if load_cfg is None or load_cfg.get("optimizer"):
      self.discriminator_optimizer.load_state_dict(
        loaded_dict["amp_optimizer_state_dict"]
      )
    return restored
