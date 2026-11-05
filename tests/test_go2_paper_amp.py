"""Go2 paper-style AMP teacher/student contracts."""

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from go1_tasks.amp import AmpDiscriminator
from go1_tasks.amp_data import PAPER_TRANSITION_DIM
from go2_tasks import TASK_IDS, unitree_go2_paper_amp_runner_cfg
from go2_tasks.env_cfgs import GO2_PAPER_AMP_REFERENCE_PATH
from go2_tasks.paper_amp import (
  LSTM_HIDDEN_DIM,
  PaperAmpStudent,
  PaperAmpTeacherActor,
)
from go2_tasks.paper_curriculum import terrain_levels_from_tracking


def test_paper_amp_task_and_runner_contract() -> None:
  assert TASK_IDS["paper_amp_teacher"] == "Go2-Velocity-Rough-AMP-Teacher"
  cfg = load_env_cfg(TASK_IDS["paper_amp_teacher"])
  assert tuple(cfg.observations["privileged"].terms) == ("state",)
  assert tuple(cfg.observations["terrain"].terms) == ("height_scan",)
  runner = unitree_go2_paper_amp_runner_cfg()
  assert runner.obs_groups["actor"] == ("actor", "privileged", "terrain")
  assert runner.actor.class_name == "go2_tasks.paper_amp.PaperAmpTeacherActor"
  assert runner.algorithm.reference_path == GO2_PAPER_AMP_REFERENCE_PATH
  assert (
    cfg.events["amp_reference_state_init"].params["reference_path"]
    == runner.algorithm.reference_path
  )
  assert runner.algorithm.discriminator_hidden_dims == (1024, 512)
  assert runner.algorithm.discriminator_activation == "elu"
  assert runner.algorithm.discriminator_state_mode == "paper"
  assert runner.algorithm.discriminator_trunk_weight_decay == 1e-3
  assert runner.algorithm.discriminator_head_weight_decay == 1e-1
  assert runner.algorithm.zero_command_amp_threshold == 0.0
  assert cfg.curriculum["terrain_levels"].func is terrain_levels_from_tracking
  assert cfg.curriculum["terrain_levels"].params["promote_threshold"] == 0.85

  discriminator = AmpDiscriminator(
    runner.algorithm.discriminator_hidden_dims,
    input_dim=PAPER_TRANSITION_DIM,
    activation=runner.algorithm.discriminator_activation,
  )
  assert discriminator.trunk[0].in_features == 62
  assert discriminator.trunk[0].out_features == 1024
  assert isinstance(discriminator.trunk[1], torch.nn.ELU)
  assert discriminator.trunk[2].in_features == 1024
  assert discriminator.trunk[2].out_features == 512
  assert isinstance(discriminator.trunk[3], torch.nn.ELU)
  assert discriminator.head.in_features == 512
  assert discriminator.head.out_features == 1


def test_paper_amp_environment_dimensions() -> None:
  cfg = load_env_cfg(TASK_IDS["paper_amp_teacher"])
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (1, 45)
    assert obs["privileged"].shape == (1, 30)
    assert obs["terrain"].shape == (1, 187)
    assert torch.isfinite(obs["privileged"]).all()
    for _ in range(3):
      obs, reward, _, _, _ = env.step(torch.zeros(1, 12))
      assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_teacher_and_lstm_student_dimensions() -> None:
  obs = TensorDict(
    {
      "actor": torch.zeros(2, 45),
      "privileged": torch.zeros(2, 30),
      "terrain": torch.zeros(2, 187),
    },
    batch_size=[2],
  )
  teacher = PaperAmpTeacherActor(
    obs,
    {"actor": ["actor", "privileged", "terrain"]},
    "actor",
    12,
    hidden_dims=(256, 128, 64),
    activation="elu",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  assert teacher(obs).shape == (2, 12)
  assert teacher.encode_privileged(obs).shape == (2, 24)

  student = PaperAmpStudent()
  student.copy_low_level_from(teacher)
  sequence = torch.zeros(2, 50, 45)
  actions, latents = student.forward_sequence(
    sequence, torch.zeros(2, 50, dtype=torch.bool)
  )
  assert actions.shape == (2, 50, 12)
  assert latents.shape == (2, 50, 24)
  hidden, cell = student.initial_state(2, "cpu")
  assert hidden.shape == cell.shape == (1, 2, LSTM_HIDDEN_DIM)


def test_unbounded_teacher_and_student_preserve_action_authority():
  obs = TensorDict({'actor':torch.zeros(1,45), 'privileged':torch.zeros(1,30),
                   'terrain':torch.zeros(1,187)}, batch_size=[1])
  teacher = PaperAmpTeacherActor(
    obs, {'actor':['actor','privileged','terrain']}, 'actor', 12,
    bound_action_mean=False,
    distribution_cfg={'class_name':'GaussianDistribution','init_std':1.0,'std_type':'scalar'})
  with torch.no_grad():
    teacher.mlp[-1].weight.zero_()
    teacher.mlp[-1].bias.fill_(2.)
  torch.testing.assert_close(teacher(obs), torch.full((1,12),2.))
  student = PaperAmpStudent()
  student.copy_low_level_from(teacher)
  action, _, _, _ = student.step(obs['actor'], *student.initial_state(1,'cpu'))
  torch.testing.assert_close(action, torch.full((1,12),2.))
