"""Go2 Hybrid Internal Model task, model and estimator contracts."""

import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from go2_tasks import TASK_IDS
from go1_tasks.cts import track_yaw_velocity
from go2_tasks.env_cfgs import unitree_go2_him_env_cfg
from go2_tasks.him_commands import HimVelocityCommand, HimVelocityCommandCfg
from go2_tasks.him import (
  COMMAND_DIM,
  HIMActor,
  HIMEstimator,
  HIMPPO,
  HISTORY_STEPS,
  LATENT_DIM,
  PRIVILEGED_DIM,
  PROPRIO_DIM,
  TERM_DIMS,
  VELOCITY_DIM,
  sinkhorn,
)


def _distribution_cfg() -> dict:
  return {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }


def _actor(obs: TensorDict) -> HIMActor:
  return HIMActor(
    obs,
    {"actor": ["actor"], "critic": ["critic"]},
    "actor",
    12,
    hidden_dims=(64, 32),
    distribution_cfg=_distribution_cfg(),
  )


def test_him_task_and_runner_contract() -> None:
  assert TASK_IDS["him"] == "Go2-Velocity-HIM"
  cfg = load_env_cfg(TASK_IDS["him"])
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  assert tuple(actor.terms) == (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert actor.history_length == HISTORY_STEPS
  # The estimator slices the privileged vector at the same offsets as HIMLoco.
  assert tuple(critic.terms)[:6] == tuple(actor.terms)
  assert PROPRIO_DIM == 45 and PRIVILEGED_DIM == 238
  assert actor.terms["command"].scale == (2.0, 2.0, 0.25)
  assert actor.terms["base_ang_vel"].scale == 0.25
  assert actor.terms["joint_vel"].scale == 0.05
  assert critic.terms["base_lin_vel"].scale == 2.0

  # Tracking weights follow the paper's 1.0:0.5 ratio, and the dense L2 cost
  # keeps a gradient while the robot is still standing.
  assert cfg.rewards["track_linear_velocity"].weight == 4.0
  assert cfg.rewards["track_angular_velocity"].weight == 2.0
  assert cfg.rewards["track_angular_velocity"].func is track_yaw_velocity
  assert math.isclose(
    cfg.rewards["track_angular_velocity"].params["std"], math.sqrt(0.25)
  )
  assert cfg.rewards["body_ang_vel"].weight == -0.2
  assert cfg.rewards["action_rate_l2"].weight == -0.04
  assert cfg.rewards["linear_velocity_error_l2"].weight == -1.0

  agent = load_rl_cfg(TASK_IDS["him"])
  assert agent.actor.class_name == "go2_tasks.him.HIMActor"
  assert agent.algorithm.class_name == "go2_tasks.him.HIMPPO"
  assert agent.obs_groups == {"actor": ("actor",), "critic": ("critic",)}
  assert agent.algorithm.estimator_epochs >= 1
  assert HIMPPO.__mro__[1].__name__ == "PPO"


def test_him_environment_dimensions() -> None:
  cfg = load_env_cfg(TASK_IDS["him"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (1, HISTORY_STEPS * PROPRIO_DIM)
    assert obs["critic"].shape == (1, PRIVILEGED_DIM)
    for _ in range(3):
      obs, reward, *_ = env.step(torch.zeros(1, 12))
      assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()
  finally:
    env.close()


def test_him_env_history_is_term_major() -> None:
  """mjlab stores each term's history contiguously, not whole frames."""
  cfg = load_env_cfg(TASK_IDS["him"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    command = env.command_manager.get_command("twist")
    # The history backfills on reset, so the command block repeats HISTORY_STEPS
    # times at the front of the group.
    block = obs["actor"][:, : COMMAND_DIM * HISTORY_STEPS].reshape(
      1, HISTORY_STEPS, COMMAND_DIM
    )
    scale = command.new_tensor((2.0, 2.0, 0.25))
    expected = (command * scale).unsqueeze(1).expand(-1, HISTORY_STEPS, -1)
    torch.testing.assert_close(block, expected, atol=1e-5, rtol=0)
  finally:
    env.close()


def test_him_estimator_learns_successor_state() -> None:
  estimator = HIMEstimator(history_steps=HISTORY_STEPS, latent_dim=LATENT_DIM)
  history = torch.randn(8, HISTORY_STEPS * PROPRIO_DIM)
  privileged = torch.randn(8, PRIVILEGED_DIM)
  velocity, latent = estimator(history)
  assert velocity.shape == (8, VELOCITY_DIM)
  assert latent.shape == (8, LATENT_DIM)
  assert torch.allclose(latent.norm(dim=-1), torch.ones(8), atol=1e-5)
  # Successor state = next proprio without command + privileged velocity.
  successor = estimator.successor_state(privileged)
  assert successor.shape == (8, PROPRIO_DIM)
  assert torch.allclose(successor[:, PROPRIO_DIM - 3 :], privileged[:, 45:48])

  velocity_loss, swap_loss = estimator.update(history, privileged, lr=1e-3)
  assert torch.isfinite(torch.tensor(velocity_loss))
  assert torch.isfinite(torch.tensor(swap_loss))
  assert all(
    group["lr"] == 1e-3 for group in estimator.optimizer.param_groups
  )
  # Swapped prediction must update the latent outputs, target, and prototypes.
  assert estimator.encoder[-1].weight.grad[VELOCITY_DIM:].abs().sum() > 0
  assert all(
    param.grad is not None and param.grad.abs().sum() > 0
    for param in estimator.target.parameters()
  )
  assert estimator.proto.weight.grad is not None
  assert estimator.proto.weight.grad.abs().sum() > 0


def test_sinkhorn_is_doubly_stochastic() -> None:
  batch, prototypes = 10, 32
  codes = sinkhorn(torch.randn(batch, prototypes))
  # Every sample distributes unit mass over the prototypes (the final
  # normalisation is per sample); prototype mass is equalised approximately
  # and therefore carries the full batch mass.
  assert torch.allclose(codes.sum(dim=1), torch.ones(batch), atol=1e-4)
  assert torch.allclose(codes.sum(), torch.tensor(float(batch)), atol=1e-3)
  prototype_mass = codes.sum(dim=0)
  assert torch.all(prototype_mass > 0.05 * batch / prototypes)


def _term_major(frames: torch.Tensor) -> torch.Tensor:
  """Pack ``[batch, history, 45]`` frames into mjlab's term-major layout."""
  batch, steps, _ = frames.shape
  start = 0
  blocks = []
  for dim in TERM_DIMS:
    blocks.append(frames[:, :, start : start + dim].reshape(batch, steps * dim))
    start += dim
  return torch.cat(blocks, dim=-1)


def test_him_actor_rebuilds_frame_order_from_term_major() -> None:
  frames = torch.zeros(2, HISTORY_STEPS, PROPRIO_DIM)
  for step in range(HISTORY_STEPS):
    frames[:, step] = torch.arange(PROPRIO_DIM).float() + 100 * step
  obs = TensorDict(
    {
      "actor": _term_major(frames),
      "critic": torch.zeros(2, PRIVILEGED_DIM),
    },
    batch_size=[2],
  )
  actor = _actor(obs)
  assert actor._get_latent_dim() == PROPRIO_DIM + VELOCITY_DIM + LATENT_DIM
  # mjlab's group layout is term-major; the model must restore frame order and
  # put the newest step first for the estimator.
  torch.testing.assert_close(actor.frame_major_history(obs), frames)
  assert torch.allclose(
    actor.newest_first_history(obs), frames.flip(1).flatten(1)
  )
  # The policy's direct branch consumes the newest observation.
  current = actor.get_latent(obs)[:, :PROPRIO_DIM]
  torch.testing.assert_close(current, frames[:, -1])

  actions = actor(obs, stochastic_output=True)
  assert actions.shape == (2, 12)
  # The export takes a frame-major newest-first history and must agree.
  exported = actor.as_onnx()
  assert torch.allclose(
    exported(frames.flip(1).flatten(1)), actor(obs), atol=1e-6
  ), "ONNX wrapper and HIMActor must agree"

  # PPO must train the estimator on the same order used during inference.
  algorithm = object.__new__(HIMPPO)
  algorithm.actor = actor
  algorithm._history_buffer = torch.zeros(1, 2, HISTORY_STEPS * PROPRIO_DIM)
  algorithm._rollout_index = 0
  with patch.object(HIMPPO.__mro__[1], "act", return_value=torch.zeros(2, 12)):
    algorithm.act(obs)
  torch.testing.assert_close(
    algorithm._history_buffer[0], actor.newest_first_history(obs)
  )


def test_him_command_block_length() -> None:
  assert COMMAND_DIM == 3
  assert PRIVILEGED_DIM == PROPRIO_DIM + VELOCITY_DIM + 3 + 187


def test_old_him_checkpoint_is_rejected() -> None:
  algorithm = object.__new__(HIMPPO)
  with pytest.raises(ValueError, match="Incompatible HIM checkpoint"):
    algorithm.load({}, None, strict=True)


def test_him_deployment_observation_layout() -> None:
  import numpy as np

  from go2_tasks.deployment import (
    DEFAULT_JOINT_POS,
    HIM_HISTORY_STEPS,
    HIM_OBS_DIM,
    him_actor_observation,
  )

  command = np.array([0.3, -0.2, 0.7], dtype=np.float32)
  obs = him_actor_observation(
    command=command,
    base_ang_vel=np.array([1.0, -2.0, 3.0], dtype=np.float32),
    projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
    joint_pos=DEFAULT_JOINT_POS.copy(),
    joint_vel=np.ones(12, dtype=np.float32),
    previous_action=np.zeros(12, dtype=np.float32),
  )
  assert obs.shape == (HIM_OBS_DIM,)
  assert HIM_OBS_DIM == PROPRIO_DIM and HIM_HISTORY_STEPS == HISTORY_STEPS
  # HIMLoco puts the command first; the rest starts with angular velocity.
  np.testing.assert_allclose(obs[:3], command * [2.0, 2.0, 0.25])
  np.testing.assert_allclose(obs[3:6], [0.25, -0.5, 0.75])
  np.testing.assert_allclose(obs[6:9], [0.0, 0.0, -1.0], atol=1e-6)
  np.testing.assert_allclose(obs[9:21], np.zeros(12), atol=1e-6)
  np.testing.assert_allclose(obs[21:33], np.full(12, 0.05))


def test_him_turning_command_cfg() -> None:
  """Training reserves in-place turns; playback keeps the inherited sampler."""
  twist = load_env_cfg(TASK_IDS["him"]).commands["twist"]
  assert isinstance(twist, HimVelocityCommandCfg)
  assert twist.rel_turning_envs == pytest.approx(0.1)
  assert twist.min_turning_yaw == pytest.approx(0.2)
  # Heading control is what makes the reference sampler produce turns; the
  # turning worlds must override the yaw sample on top of it.
  assert twist.heading_command is True
  low, high = twist.ranges.ang_vel_z
  assert twist.min_turning_yaw <= high and -twist.min_turning_yaw >= low

  with pytest.raises(ValueError, match="rel_turning_envs"):
    replace(twist, rel_turning_envs=1.5)
  with pytest.raises(ValueError, match="turning yaw range"):
    replace(twist, min_turning_yaw=0.0)

  play_twist = unitree_go2_him_env_cfg(play=True).commands["twist"]
  assert not isinstance(play_twist, HimVelocityCommandCfg), (
    "play mode must not reserve turning worlds; it is driven by fixed commands"
  )


def test_him_turning_command_keeps_zero_linear_yaw_samples() -> None:
  """Turning worlds must survive ``_update_command`` instead of being zeroed."""
  cfg = load_env_cfg(TASK_IDS["him"])
  cfg.scene.num_envs = 32
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    term = env.command_manager.get_term("twist")
    assert isinstance(term, HimVelocityCommand)
    env_ids = torch.arange(env.num_envs, device=env.device)

    term.cfg.rel_turning_envs = 0.0
    term._resample_command(env_ids)
    assert not term.is_turning_env.any(), "0% must leave every world to the sampler"

    term.cfg.rel_turning_envs = 1.0
    term._resample_command(env_ids)
    # ``_update_command`` zeroes standing/heading/world envs; a turning world
    # that kept any of those flags silently loses its yaw command.
    term._update_command()

    assert term.is_turning_env.all()
    for flag in (
      term.is_standing_env,
      term.is_heading_env,
      term.is_world_env,
      term.is_forward_env,
    ):
      assert not flag.any()
    command = env.command_manager.get_command("twist")
    assert torch.count_nonzero(command[:, :2]) == 0
    yaw = command[:, 2]
    max_yaw = max(abs(bound) for bound in term.cfg.ranges.ang_vel_z)
    assert torch.all(yaw.abs() >= term.cfg.min_turning_yaw - 1e-6)
    assert torch.all(yaw.abs() <= max_yaw + 1e-6)
    assert (yaw > 0).any() and (yaw < 0).any(), "both turn directions must appear"
  finally:
    env.close()
