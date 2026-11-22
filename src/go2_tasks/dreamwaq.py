"""DreamWaQ CENet, blind actor and PPO integration for Unitree Go2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

HISTORY_STEPS = 5
TERM_DIMS = (3, 3, 12, 12, 12, 3)
PROPRIO_DIM = sum(TERM_DIMS)
VELOCITY_DIM = 3
CONTEXT_DIM = 16
FEATURE_DIM = PROPRIO_DIM + VELOCITY_DIM + CONTEXT_DIM
_FEATURES = "dreamwaq_policy_features"
_VELOCITY = slice(PROPRIO_DIM, PROPRIO_DIM + VELOCITY_DIM)


@dataclass
class DreamWaQPpoCfg(RslRlPpoAlgorithmCfg):
  cenet_beta: float = 0.5
  cenet_learning_rate: float = 1e-3
  cenet_epochs: int = 4
  cenet_batch_size: int = 1024
  ada_boot: bool = True


class DreamWaQRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None) -> None:
    super().save(path, {**(infos or {}), "action_clip": self.env.clip_actions})

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    # The parent seeds the first timeout with random elapsed time. Use only
    # returns collected from an actual episode start for AdaBoot.
    if init_at_random_ep_len:
      self.alg.partial_episode.fill_(True)
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    super().export_policy_to_onnx(path, filename, verbose)
    clip = self.env.clip_actions
    attach_metadata_to_onnx(
      str(Path(path) / filename), {"action_clip": "none" if clip is None else clip}
    )


class CENet(nn.Module):
  """Regress current body velocity and predict the next proprioceptive frame."""

  def __init__(self) -> None:
    super().__init__()
    self.encoder = MLP(HISTORY_STEPS * PROPRIO_DIM, 64, (128,), "elu")
    self.velocity = nn.Linear(64, VELOCITY_DIM)
    self.mean = nn.Linear(64, CONTEXT_DIM)
    self.log_variance = nn.Linear(64, CONTEXT_DIM)
    # Fig. 2 labels the decoder as 48-D, while Eq. (1) defines 45 proprioceptive
    # values. Use Eq. (1) for the next-frame reconstruction target.
    self.decoder = MLP(CONTEXT_DIM, PROPRIO_DIM, (64, 128), "elu")

  def forward(
    self, history: torch.Tensor, sample_context: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = F.elu(self.encoder(history))
    context = self.mean(hidden)
    if sample_context:
      context = self._sample(context, self.log_variance(hidden).clamp(-10.0, 5.0))
    return self.velocity(hidden), context

  @staticmethod
  def _sample(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return mean + torch.randn_like(mean) * (0.5 * log_variance).exp()

  def losses(self, history, velocity, next_frame) -> tuple[torch.Tensor, ...]:
    hidden = F.elu(self.encoder(history))
    predicted_velocity = self.velocity(hidden)
    mean = self.mean(hidden)
    log_variance = self.log_variance(hidden).clamp(-10.0, 5.0)
    context = self._sample(mean, log_variance)
    kl = 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance)
    return (
      F.mse_loss(predicted_velocity, velocity),
      F.mse_loss(self.decoder(context), next_frame),
      kl.sum(dim=-1).mean(),
    )


def _frame_major(history: torch.Tensor, widths: list[int], steps: int) -> torch.Tensor:
  """Convert mjlab term-major history to oldest-first frames."""
  chunks = history.split([steps * dim for dim in widths], dim=-1)
  return torch.cat(
    [chunk.reshape(-1, steps, dim) for chunk, dim in zip(chunks, widths)],
    dim=-1,
  )


def frame_major(history: torch.Tensor) -> torch.Tensor:
  return _frame_major(history, list(TERM_DIMS), HISTORY_STEPS)


class DreamWaQActor(MLPModel):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.obs_dim != HISTORY_STEPS * PROPRIO_DIM:
      raise ValueError(f"Unexpected DreamWaQ history width: {self.obs_dim}")
    self.cenet = CENet()
    self.frame_norm = EmpiricalNormalization(PROPRIO_DIM)

  def _get_latent_dim(self) -> int:
    return FEATURE_DIM

  @torch.no_grad()
  def features(self, history: torch.Tensor, sample_context: bool = False) -> torch.Tensor:
    frames = self.frame_norm(frame_major(history))
    velocity, context = self.cenet(frames.flatten(1), sample_context)
    return torch.cat((frames[:, -1], velocity, context), dim=-1)

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    if _FEATURES in obs:
      return obs[_FEATURES]
    return self.features(obs["actor"])

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _Export(self)

  def as_jit(self) -> nn.Module:
    return _Export(self)


class _Export(nn.Module):
  is_recurrent = False
  input_names = ["obs_history"]
  output_names = ["actions"]

  def __init__(self, actor: DreamWaQActor) -> None:
    super().__init__()
    self.frame_norm = deepcopy(actor.frame_norm)
    self.encoder = deepcopy(actor.cenet.encoder)
    self.velocity = deepcopy(actor.cenet.velocity)
    self.mean = deepcopy(actor.cenet.mean)
    self.policy = deepcopy(actor.mlp)
    self.term_dims = list(TERM_DIMS)
    self.history_steps = HISTORY_STEPS
    assert actor.distribution is not None
    self.output = actor.distribution.as_deterministic_output_module()

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    frames = self.frame_norm(_frame_major(history, self.term_dims, self.history_steps))
    hidden = F.elu(self.encoder(frames.flatten(1)))
    features = torch.cat(
      (frames[:, -1], self.velocity(hidden), self.mean(hidden)), dim=-1
    )
    return self.output(self.policy(features))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (torch.zeros(1, HISTORY_STEPS * PROPRIO_DIM),)


class DreamWaQPPO(PPO):
  """Cache acting-time policy features and fit CENet after each PPO update."""

  def __init__(
    self,
    *args,
    cenet_beta: float = 0.5,
    cenet_learning_rate: float = 1e-3,
    cenet_epochs: int = 4,
    cenet_batch_size: int = 1024,
    ada_boot: bool = True,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if self.is_multi_gpu:
      raise ValueError("DreamWaQ currently supports single-device training")
    if self.symmetry is not None:
      raise ValueError("DreamWaQ caches policy features and needs task-specific symmetry")
    if cenet_beta < 0 or min(cenet_learning_rate, cenet_epochs, cenet_batch_size) <= 0:
      raise ValueError("Invalid DreamWaQ CENet hyperparameters")
    self.cenet_beta = cenet_beta
    self.cenet_epochs = cenet_epochs
    self.cenet_batch_size = cenet_batch_size
    self.ada_boot = ada_boot
    self.boot_probability = 0.0 if ada_boot else 1.0
    self.cenet_optimizer = torch.optim.Adam(
      self.actor.cenet.parameters(), lr=cenet_learning_rate
    )
    shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
    self.storage.observations[_FEATURES] = torch.zeros(
      *shape, FEATURE_DIM, device=self.device
    )
    self.next_frames = torch.zeros(*shape, PROPRIO_DIM, device=self.device)
    self.episode_returns = torch.zeros(shape[1], device=self.device)
    self.completed_returns = torch.zeros(*shape, device=self.device)
    self.partial_episode = torch.zeros(shape[1], dtype=torch.bool, device=self.device)
    self.completed_episodes = torch.zeros(*shape, dtype=torch.bool, device=self.device)

  def act(self, obs: TensorDict) -> torch.Tensor:
    features = self.actor.features(obs["actor"], sample_context=True)
    if self.ada_boot:
      draws = torch.rand(features.shape[0], 1, device=self.device)
      use_estimate = draws < self.boot_probability
      features[:, _VELOCITY] = torch.where(
        use_estimate, features[:, _VELOCITY], obs["critic"][:, _VELOCITY],
      )
    return super().act(obs.copy().set(_FEATURES, features))

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    step = self.storage.step
    self.next_frames[step].copy_(frame_major(obs["actor"])[:, -1])
    done = dones.reshape(-1).bool()
    self.episode_returns.add_(rewards.reshape(-1))
    self.completed_returns[step].copy_(self.episode_returns)
    self.completed_episodes[step].copy_(done & ~self.partial_episode)
    self.episode_returns.masked_fill_(done, 0.0)
    self.partial_episode.masked_fill_(done, False)
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    steps = self.storage.step
    obs = self.storage.observations[:steps].flatten(0, 1)
    dones = self.storage.dones[:steps].flatten().bool()
    finished = self.completed_returns[:steps][self.completed_episodes[:steps]]
    losses = super().update()
    if self.ada_boot and finished.numel() >= 2:
      cv = finished.std(unbiased=False) / finished.mean().abs().clamp_min(1e-8)
      self.boot_probability = (1.0 - cv.tanh()).item()
    losses["adaboot_probability"] = self.boot_probability

    frames = frame_major(obs["actor"][~dones])
    velocity = obs["critic"][~dones, _VELOCITY]
    successor = self.next_frames[:steps].flatten(0, 1)[~dones]
    sums = torch.zeros(3, device=self.device)
    count = frames.shape[0]
    if count:
      normalizer = self.actor.frame_norm
      with torch.no_grad():
        normalizer.update(frames[:, -1])
        history = normalizer(frames).flatten(1)
        next_frame = normalizer(successor)
      for _ in range(self.cenet_epochs):
        order = torch.randperm(count, device=self.device)
        for batch in order.split(self.cenet_batch_size):
          terms = self.actor.cenet.losses(
            history[batch], velocity[batch], next_frame[batch]
          )
          velocity_mse, reconstruction_mse, kl = terms
          self.cenet_optimizer.zero_grad(set_to_none=True)
          (velocity_mse + reconstruction_mse + self.cenet_beta * kl).backward()
          nn.utils.clip_grad_norm_(self.actor.cenet.parameters(), 1.0)
          self.cenet_optimizer.step()
          sums += torch.stack(terms).detach() * batch.numel()
      sums /= count * self.cenet_epochs
    velocity_mse, reconstruction_mse, kl = sums.tolist()
    losses["cenet_velocity_mse"] = velocity_mse
    losses["cenet_next_obs_mse"] = reconstruction_mse
    losses["cenet_kl"] = kl
    losses["cenet_valid_transitions"] = float(count)
    return losses

  def save(self) -> dict:
    return {
      **super().save(),
      "cenet_optimizer_state_dict": self.cenet_optimizer.state_dict(),
      "adaboot_probability": self.boot_probability,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    restored = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer"):
      self.cenet_optimizer.load_state_dict(loaded_dict["cenet_optimizer_state_dict"])
      if self.ada_boot:
        self.boot_probability = loaded_dict["adaboot_probability"]
    return restored
