"""AMP task and one short PPO/discriminator update."""

from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.algorithms import PPO

from go1_tasks import TASK_IDS
from go1_tasks.amp import AmpPPO
from go1_tasks.amp_data import TRANSITION_DIM, amp_frame_from_robot


def _write_reference_clip(path, robot) -> None:
  foot_ids, foot_names = robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)
  data = robot.data
  fields = {
    "root_pos_w": data.root_link_pos_w[0],
    "root_quat_w": data.root_link_quat_w[0],
    "root_lin_vel_b": data.root_link_lin_vel_b[0],
    "root_ang_vel_b": data.root_link_ang_vel_b[0],
    "joint_pos": data.joint_pos[0],
    "joint_vel": data.joint_vel[0],
    "foot_pos_w": data.site_pos_w[0, foot_ids],
  }
  np.savez_compressed(
    path,
    **{
      name: np.repeat(value.cpu().numpy()[None], 3, axis=0)
      for name, value in fields.items()
    },
    fps=np.array(50.0),
    joint_names=np.array(robot.joint_names),
    foot_site_names=np.array(foot_names),
  )


def test_amp_task_and_update(tmp_path) -> None:
  cfg = load_env_cfg(TASK_IDS["amp"])
  cfg.scene.num_envs = 2
  cfg.events.pop("push_robot", None)
  agent_cfg = load_rl_cfg(TASK_IDS["amp"])
  agent_cfg.num_steps_per_env = 2
  agent_cfg.algorithm.num_learning_epochs = 1
  agent_cfg.algorithm.num_mini_batches = 1
  agent_cfg.algorithm.discriminator_epochs = 1
  agent_cfg.algorithm.discriminator_batch_size = 4
  agent_cfg.algorithm.replay_capacity = 16
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    obs = wrapped.get_observations()
    _write_reference_clip(tmp_path / "reference.npz", env.scene["robot"])
    agent_cfg.algorithm.reference_path = str(tmp_path)
    assert obs["actor"].shape == (2, 45)
    assert obs["critic"].shape == (2, 72)
    assert "behavior" not in cfg.commands
    train_cfg = asdict(agent_cfg)
    train_cfg["multi_gpu"] = None
    for model in (train_cfg["actor"], train_cfg["critic"]):
      for key in ("cnn_cfg", "distribution_cfg"):
        if model.get(key) is None:
          model.pop(key, None)
      if model.get("rnn_type") is None:
        for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          model.pop(key, None)
    algorithm = AmpPPO.construct_algorithm(obs, wrapped, train_cfg, "cpu")
    assert isinstance(algorithm, PPO)
    assert algorithm.motions is None
    algorithm.train_mode()
    assert algorithm.motions is not None
    assert algorithm.motions.fps == 50.0
    assert algorithm.motions.sample(4).shape == (4, TRANSITION_DIM)
    with torch.inference_mode():
      for _ in range(agent_cfg.num_steps_per_env):
        actions = algorithm.act(obs)
        obs, rewards, dones, extras = wrapped.step(actions)
        assert algorithm._previous_frame is not None
        transitions = torch.cat(
          (algorithm._previous_frame, amp_frame_from_robot(env.scene["robot"])),
          dim=-1,
        )
        expected = rewards.clone() + (
          algorithm.amp_reward_weight
          * env.step_dt
          * F.softplus(algorithm.discriminator(transitions))
          * ~dones.bool().reshape(-1)
        )
        algorithm.process_env_step(obs, rewards, dones, extras)
        torch.testing.assert_close(rewards, expected)
      algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert torch.isfinite(torch.tensor(list(losses.values()))).all()
    assert "amp_discriminator" in losses
    checkpoint = algorithm.save()
    assert "amp_discriminator_state_dict" in checkpoint
    assert algorithm.load(checkpoint, load_cfg=None, strict=True)
    inference_cfg = asdict(agent_cfg)
    inference_cfg["multi_gpu"] = None
    inference_cfg["algorithm"]["reference_path"] = str(tmp_path / "missing")
    for model in (inference_cfg["actor"], inference_cfg["critic"]):
      for key in ("cnn_cfg", "distribution_cfg"):
        if model.get(key) is None:
          model.pop(key, None)
      if model.get("rnn_type") is None:
        for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          model.pop(key, None)
    inference = AmpPPO.construct_algorithm(obs, wrapped, inference_cfg, "cpu")
    assert inference.motions is None
    assert not inference.load(checkpoint, load_cfg={"actor": True}, strict=True)
    assert inference.get_policy()(obs).shape == (2, 12)
  finally:
    wrapped.close()
