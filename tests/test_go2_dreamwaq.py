"""DreamWaQ training-transition alignment and deployment parity."""

from dataclasses import asdict
from unittest.mock import patch

import onnx
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from go2_tasks import TASK_IDS
from go2_tasks.dreamwaq import (
  DreamWaQActor, HISTORY_STEPS, PROPRIO_DIM, TERM_DIMS, frame_major,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from sim2sim.quadruped_sim.robots.go2 import Go2Simulator


def _history(frames):
  return torch.cat([term.flatten(1) for term in frames.split(TERM_DIMS, dim=-1)], dim=-1)


def _observations(start):
  frames = (start + torch.arange(2 * HISTORY_STEPS * PROPRIO_DIM)).reshape(
    2, HISTORY_STEPS, PROPRIO_DIM
  ).float() / 1000
  critic = torch.zeros(2, 238)
  critic[:, 45:48] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) + start
  return frames, TensorDict({"actor": _history(frames), "critic": critic}, [2])


def test_dreamwaq_actor_uses_true_velocity_only_at_rollout_when_scheduled():
  _, obs = _observations(0)
  cfg = asdict(load_rl_cfg(TASK_IDS["dreamwaq"]))
  cfg.update(num_steps_per_env=2, multi_gpu=None)
  cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, cenet_epochs=1)
  for key in ("actor", "critic"):
    for name in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg[key].pop(name, None)

  class Env:
    num_envs = 2
    num_actions = 12

  algorithm = PPO.construct_algorithm(obs, Env(), cfg, "cpu")
  actor = algorithm.actor
  estimated = actor.features(obs["actor"])
  with torch.inference_mode():
    algorithm.act(obs)
    cached = actor.get_latent(algorithm.transition.observations)
    torch.testing.assert_close(cached[:, 45:48], obs["critic"][:, 45:48])
    assert not torch.allclose(cached[:, 45:48], estimated[:, 45:48])
    assert "dreamwaq_policy_features" not in obs
    torch.testing.assert_close(algorithm.transition.observations["critic"], obs["critic"])

    next_frames, successor = _observations(10)
    algorithm.process_env_step(successor, torch.tensor([2.0, 3.0]), torch.tensor([0, 1]), {})
    following_frames, following = _observations(20)
    algorithm.act(successor)
    algorithm.process_env_step(following, torch.tensor([4.0, 1.0]), torch.tensor([1, 0]), {})
    algorithm.compute_returns(following)

  torch.testing.assert_close(algorithm.next_frames[0], next_frames[:, -1])
  torch.testing.assert_close(algorithm.next_frames[1], following_frames[:, -1])
  with patch.object(actor.cenet, "losses", wraps=actor.cenet.losses) as cenet_loss:
    loss = algorithm.update()
  _, velocity_targets, reconstruction_targets = cenet_loss.call_args.args
  expected_velocities = torch.stack((obs["critic"][0, 45:48], successor["critic"][1, 45:48]))
  torch.testing.assert_close(velocity_targets.sort(dim=0).values, expected_velocities)
  expected_successors = actor.frame_norm(torch.stack((next_frames[0, -1], following_frames[1, -1])))
  torch.testing.assert_close(reconstruction_targets.sort(dim=0).values, expected_successors)
  assert loss["cenet_valid_transitions"] == 2
  expected_probability = 1.0 - torch.tensor(1.5 / 4.5).tanh().item()
  assert abs(loss["adaboot_probability"] - expected_probability) < 1e-6
  assert actor.frame_norm.count.item() == 2
  assert algorithm.cenet_optimizer.state

  algorithm.partial_episode.fill_(True)
  with torch.inference_mode():
    algorithm.act(obs)
    algorithm.process_env_step(successor, torch.tensor([9.0, 1.0]), torch.ones(2), {})
    algorithm.act(successor)
    algorithm.process_env_step(following, torch.tensor([2.0, 2.0]), torch.ones(2), {})
    algorithm.compute_returns(following)
  assert algorithm.update()["adaboot_probability"] == 1.0


def test_dreamwaq_history_and_exports_match_actor(tmp_path):
  task = TASK_IDS["dreamwaq"]
  env = load_env_cfg(task)
  assert tuple(env.observations["actor"].terms) == (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"
  )
  assert env.observations["actor"].history_length == 5
  frames, obs = _observations(0)
  torch.testing.assert_close(frame_major(obs["actor"]), frames)
  cfg = asdict(load_rl_cfg(task).actor)
  actor = DreamWaQActor(
    obs, {"actor": ["actor"]}, "actor", 12,
    **{name: cfg[name] for name in ("hidden_dims", "activation", "distribution_cfg")},
  )
  actor.frame_norm.update(frames[:, -1])
  expected = actor.eval()(obs[:1]).detach()
  scripted = torch.jit.script(actor.as_jit())
  torch.testing.assert_close(scripted(obs["actor"][:1]), expected)

  path = tmp_path / "dreamwaq.onnx"
  export = actor.as_onnx().eval()
  torch.onnx.export(
    export, export.get_dummy_inputs(), str(path), input_names=export.input_names,
    output_names=export.output_names, opset_version=18, dynamo=False,
  )
  model = onnx.load(path)
  onnx.checker.check_model(model)
  result, = ReferenceEvaluator(model).run(None, {"obs_history": obs["actor"][:1].numpy()})
  torch.testing.assert_close(torch.from_numpy(result), expected, atol=1e-6, rtol=1e-5)

  attach_metadata_to_onnx(str(path), {"action_clip": 0.001})
  sim = Go2Simulator(dreamwaq=path)
  sim.data.qvel[3:6] = [0.2, -0.3, 0.4]
  frame = torch.from_numpy(sim.observation())
  torch.testing.assert_close(frame[:3], torch.tensor([0.2, -0.3, 0.4]))
  history = _history(frame.reshape(1, 1, -1).expand(-1, HISTORY_STEPS, -1))
  expected_action = scripted(history).detach().clamp(-0.001, 0.001)
  sim.set_active(True)
  sim.policy_step()
  torch.testing.assert_close(torch.from_numpy(sim.last_action), expected_action[0])

  checkpoint = tmp_path / "model_0.pt"
  torch.save(
    {"actor_state_dict": actor.state_dict(), "infos": {"action_clip": 0.001}},
    checkpoint,
  )
  checkpoint_sim = Go2Simulator(dreamwaq=checkpoint)
  checkpoint_sim.data.qvel[3:6] = [0.2, -0.3, 0.4]
  checkpoint_sim.set_active(True)
  checkpoint_sim.policy_step()
  torch.testing.assert_close(torch.from_numpy(checkpoint_sim.last_action), expected_action[0])
  torch.testing.assert_close(
    torch.from_numpy(checkpoint_sim.policy._predict(obs["actor"][:1].numpy())), expected
  )
