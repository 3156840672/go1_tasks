"""Adversarial motion prior for the Go1 MoB reference clips."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from torch import autograd, nn

from .amp_data import (
  AMP_STATE_MODES,
  FOOT_NAMES,
  FRAME_DIM,
  PAPER_FRAME_DIM,
  TRANSITION_DIM,
  AmpMotionDataset,
  amp_frame_from_robot,
)


class AmpDiscriminator(nn.Module):
  """Scalar logit for each adjacent-state transition used by AMP.

  The output is a raw logit in the least-squares GAN formulation used by AMP:
  expert transitions target +1 and policy transitions target -1. Finite targets
  keep the discriminator from becoming infinitely confident, which would make
  the style reward vanish.
  """

  def __init__(
    self,
    hidden_dims: tuple[int, ...] = (256, 128),
    *,
    input_dim: int = TRANSITION_DIM,
    activation: str = "relu",
  ) -> None:
    super().__init__()
    activations: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "elu": nn.ELU}
    try:
      activation_cls = activations[activation.lower()]
    except KeyError as exc:
      raise ValueError(
        f"Unknown AMP discriminator activation {activation!r}; "
        f"expected one of {tuple(activations)}"
      ) from exc
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(in_dim, hidden_dim), activation_cls()))
      in_dim = hidden_dim
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(in_dim, 1)
    self.input_dim = input_dim

  def forward(self, transitions: torch.Tensor) -> torch.Tensor:
    if transitions.shape[-1] != self.input_dim:
      raise ValueError(f"AMP transition must have {self.input_dim} features")
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


def amp_discriminator_loss(expert_logits, policy_logits, target_magnitude=0.0):
  """LSGAN regression; zero selects the paper targets +/-1.

  A positive legacy margin selects symmetric smoothed targets, never masks
  samples. Overshooting either target must retain a restoring gradient.
  """
  target = target_magnitude if target_magnitude > 0 else 1.0
  return 0.5 * (
    F.mse_loss(expert_logits, torch.full_like(expert_logits, target))
    + F.mse_loss(policy_logits, torch.full_like(policy_logits, -target))
  )


class AmpReplayBuffer:
  """Fixed-size circular store for past policy transitions."""

  def __init__(self, capacity: int, transition_dim: int, device: str) -> None:
    if capacity < 1:
      raise ValueError("Replay capacity must be positive")
    self.data = torch.empty(capacity, transition_dim, device=device)
    self.transition_dim = transition_dim
    self.size = 0
    self.cursor = 0

  def add(self, values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != self.transition_dim:
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
    discriminator_hidden_dims: tuple[int, ...] = (256, 128),
    discriminator_activation: str = "relu",
    discriminator_state_mode: str = "extended",
    discriminator_loss_margin: float = 0.0,
    discriminator_trunk_weight_decay: float = 0.0,
    discriminator_head_weight_decay: float = 0.0,
    reset_discriminator_on_resume: bool = False,
    zero_command_amp_threshold: float = 0.0,
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
    if discriminator_trunk_weight_decay < 0 or discriminator_head_weight_decay < 0:
      raise ValueError("AMP discriminator weight decay must be nonnegative")
    if not 0 <= discriminator_loss_margin < 1:
      raise ValueError("Discriminator loss margin must be in [0, 1)")
    self.discriminator_loss_margin = discriminator_loss_margin
    if zero_command_amp_threshold < 0:
      raise ValueError("AMP zero-command threshold must be nonnegative")
    if discriminator_state_mode not in AMP_STATE_MODES:
      raise ValueError(
        f"Unknown AMP state mode {discriminator_state_mode!r}; "
        f"expected one of {AMP_STATE_MODES}"
      )
    if self.is_multi_gpu:
      raise ValueError("Go1 AMP training currently supports one GPU only")
    self.reference_path = reference_path
    self.amp_reward_weight = amp_reward_weight
    self.discriminator_epochs = discriminator_epochs
    self.discriminator_batch_size = discriminator_batch_size
    self.discriminator_batches_per_iteration = discriminator_batches_per_iteration
    self.gradient_penalty_weight = gradient_penalty_weight
    self.reset_discriminator_on_resume = reset_discriminator_on_resume
    self.zero_command_amp_threshold = zero_command_amp_threshold
    self.discriminator_state_mode = discriminator_state_mode
    self.frame_dim = (
      PAPER_FRAME_DIM if discriminator_state_mode == "paper" else FRAME_DIM
    )
    self.transition_dim = 2 * self.frame_dim
    if not discriminator_hidden_dims or any(dim <= 0 for dim in discriminator_hidden_dims):
      raise ValueError("AMP discriminator hidden dimensions must be positive")
    self.discriminator = AmpDiscriminator(
      discriminator_hidden_dims,
      input_dim=self.transition_dim,
      activation=discriminator_activation,
    ).to(self.device)
    self.discriminator.eval()
    # Use separate regularization for the feature network and final classifier.
    # The stronger head decay follows the reference AMP implementation and
    # prevents a large final-layer weight from trivially saturating the LSGAN.
    self.discriminator_optimizer = torch.optim.Adam(
      (
        {
          "params": self.discriminator.trunk.parameters(),
          "weight_decay": discriminator_trunk_weight_decay,
        },
        {
          "params": self.discriminator.head.parameters(),
          "weight_decay": discriminator_head_weight_decay,
        },
      ),
      lr=discriminator_learning_rate,
    )
    self.replay = AmpReplayBuffer(replay_capacity, self.transition_dim, self.device)
    self._policy_transitions: list[torch.Tensor] = []
    self._previous_frame: torch.Tensor | None = None
    self._robot = None
    self._step_dt: float | None = None
    self._foot_ids: list[int] | None = None
    self._terrain_sensor = None
    self._command_manager = None
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
    try:
      algorithm._terrain_sensor = env.unwrapped.scene["terrain_scan"]
    except KeyError:
      algorithm._terrain_sensor = None
    algorithm._command_manager = env.unwrapped.command_manager
    return algorithm

  def _root_height_above_terrain(self) -> torch.Tensor | None:
    """Measure base height at the closest terrain ray for rough-terrain AMP."""
    if self._terrain_sensor is None:
      return None
    data = self._terrain_sensor.data
    valid = data.distances >= 0
    frame_xy = data.frame_pos_w[:, :1, :2]
    distance_xy = (data.hit_pos_w[..., :2] - frame_xy).square().sum(-1)
    distance_xy = torch.where(
      valid, distance_xy, torch.full_like(distance_xy, float("inf"))
    )
    closest = distance_xy.argmin(dim=1, keepdim=True)
    hit_z = data.hit_pos_w[..., 2].gather(1, closest)
    height = data.frame_pos_w[:, 0, 2:3] - hit_z
    # A terrain scan can miss every ray near a map boundary. Preserve the flat
    # behavior as a finite fallback until the out-of-bounds termination resets it.
    all_missed = ~valid.any(dim=1, keepdim=True)
    assert self._robot is not None
    return torch.where(all_missed, self._robot.data.root_link_pos_w[:, 2:3], height)

  def train_mode(self) -> None:
    super().train_mode()
    if self.motions is None:
      assert self._robot is not None and self._step_dt is not None
      self.motions = AmpMotionDataset(
        self.reference_path,
        joint_names=tuple(self._robot.joint_names),
        fps=1.0 / self._step_dt,
        state_mode=self.discriminator_state_mode,
      )
      print(
        f"[AMP] {self.motions.num_clips} clips, {len(self.motions)} transitions, "
        f"{self.motions.fps:g} Hz"
      )
    if self.amp_normalizer is None:
      frames = torch.cat(
        (
          self.motions.transitions[:, : self.frame_dim],
          self.motions.transitions[:, self.frame_dim :],
        ),
        dim=0,
      )
      self.amp_normalizer = AmpFeatureNormalizer(self.frame_dim, self.device)
      self.amp_normalizer.initialize(frames)

  def _normalize_transitions(self, transitions: torch.Tensor) -> torch.Tensor:
    assert self.amp_normalizer is not None
    return torch.cat(
      (
        self.amp_normalizer.normalize(transitions[..., : self.frame_dim]),
        self.amp_normalizer.normalize(transitions[..., self.frame_dim :]),
      ),
      dim=-1,
    )

  def _update_normalizer(self, transitions: torch.Tensor) -> None:
    assert self.amp_normalizer is not None
    frames = torch.cat(
      (transitions[:, : self.frame_dim], transitions[:, self.frame_dim :]), dim=0
    )
    self.amp_normalizer.update(frames.detach())

  def act(self, obs):
    assert self._robot is not None and self.motions is not None
    self._previous_frame = amp_frame_from_robot(
      self._robot,
      joint_names=self.motions.joint_names,
      foot_ids=self._foot_ids,
      root_height=self._root_height_above_terrain(),
      state_mode=self.discriminator_state_mode,
    ).detach()
    return super().act(obs)

  def process_env_step(self, obs, rewards, dones, extras):
    assert self._previous_frame is not None and self._robot is not None
    assert self._step_dt is not None
    next_frame = amp_frame_from_robot(
      self._robot,
      foot_ids=self._foot_ids,
      root_height=self._root_height_above_terrain(),
      state_mode=self.discriminator_state_mode,
    )
    transitions = torch.cat((self._previous_frame, next_frame), dim=-1)
    valid = ~dones.bool().reshape(-1)
    style_valid = valid
    if self.zero_command_amp_threshold > 0:
      assert self._command_manager is not None
      command = self._command_manager.get_command("twist")
      assert command is not None
      moving = torch.linalg.vector_norm(command, dim=-1) > self.zero_command_amp_threshold
      style_valid = valid & moving
    with torch.no_grad():
      logits = self.discriminator(self._normalize_transitions(transitions))
      imitation_reward = amp_style_reward(logits)
      rewards += (
        self.amp_reward_weight * self._step_dt * imitation_reward * style_valid
      )
    if bool(style_valid.any()):
      self._policy_transitions.append(transitions[style_valid].detach())
      self._style_reward_values.append(imitation_reward[style_valid].detach())
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
      # Always regress to finite targets, including samples beyond either
      # target. A one-sided mask removes the restoring gradient on overshoot.
      loss = amp_discriminator_loss(
        expert_logits, policy_logits, self.discriminator_loss_margin
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
    if hasattr(self.actor, "bound_action_mean"):
      saved["amp_bound_action_mean"] = self.actor.bound_action_mean
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
    if self.amp_normalizer is not None:
      saved["amp_normalizer_state"] = self.amp_normalizer.state_dict()
    return saved

  def load(self, loaded_dict, load_cfg=None, strict=True):
    if hasattr(self.actor, "bound_action_mean") and (load_cfg is None or load_cfg.get("actor")):
      if loaded_dict.get("amp_bound_action_mean", True) != self.actor.bound_action_mean:
        raise ValueError("Teacher action activation differs from checkpoint; use its saved configuration")
    restored = super().load(loaded_dict, load_cfg, strict)
    reset_discriminator = self.reset_discriminator_on_resume and load_cfg is None
    if reset_discriminator:
      print(
        "[AMP] Keeping the resumed actor/critic and resetting the discriminator, "
        "optimizer, normalizer, and replay buffer"
      )
    if not reset_discriminator and (
      load_cfg is None or load_cfg.get("amp_discriminator")
    ):
      self.discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"], strict=strict
      )
      if "amp_normalizer_state" in loaded_dict:
        if self.amp_normalizer is None:
          self.amp_normalizer = AmpFeatureNormalizer(self.frame_dim, self.device)
        self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state"])
    if not reset_discriminator and (
      load_cfg is None or load_cfg.get("optimizer")
    ):
      self.discriminator_optimizer.load_state_dict(
        loaded_dict["amp_optimizer_state_dict"]
      )
    return restored
