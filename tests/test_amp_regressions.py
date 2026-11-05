"""Regression tests for discriminator overshoot and terrain progression."""
from types import SimpleNamespace as NS

import torch

from go1_tasks.amp import amp_discriminator_loss
from go2_tasks.paper_curriculum import terrain_levels_from_tracking


def test_discriminator_overshoot_has_restoring_gradient():
  for target in (0.0, 0.8):
    expert = torch.tensor([3.5], requires_grad=True)
    policy = torch.tensor([-3.5], requires_grad=True)
    amp_discriminator_loss(expert, policy, target).backward()
    assert expert.grad.item() > 0
    assert policy.grad.item() < 0


def test_terrain_requires_distance_and_tracking_not_yaw():
  # exited + good tracking, exited + poor tracking, stationary + good,
  # short travel + poor tracking, strict standing, initial reset, fallen.
  n = 7
  moves = []
  terrain = NS(
    cfg=NS(terrain_generator=NS(size=(8., 8.), sub_terrains={})),
    terrain_levels=torch.zeros(n), terrain_origins=torch.zeros(1, 0, 3),
    update_env_origins=lambda ids, up, down: moves.append((up, down)),
  )
  class Scene:
    env_origins = torch.zeros(n, 3)
    def __getitem__(self, key):
      return NS(data=NS(root_link_pos_w=torch.tensor(
        [[5.,0,0],[5.,0,0],[0.,0,0],[1.,0,0],[0.,0,0],[0.,0,0],[5.,0,0]])))
  scene = Scene()
  scene.terrain = terrain
  command = torch.tensor([[.5,0,0]] * n)
  command[4] = 0
  env = NS(scene=scene, step_dt=.02,
    episode_length_buf=torch.tensor([1000]*5+[0,1000]),
    command_manager=NS(get_command=lambda name: command),
    termination_manager=NS(terminated=torch.tensor([False]*6+[True])),
    reward_manager=NS(get_term_cfg=lambda name: NS(weight=2.),
      _episode_sums={'track_linear_velocity':torch.tensor([36.,12.,36.,12.,36.,0.,36.])}))
  terrain_levels_from_tracking(env, torch.arange(n), command_name='twist')
  up, down = moves[0]
  assert up.tolist() == [True,False,False,False,False,False,False]
  assert down.tolist() == [False,False,True,True,False,False,False]


def test_physical_joint_change_reset_and_tracking():
  from go2_tasks.paper_rewards import joint_position_change, planar_tracking, yaw_tracking
  q = torch.zeros(2,12)
  robot = NS(data=NS(joint_pos=q,root_link_lin_vel_b=torch.tensor([[.3,0,2.],[0,0,0.]]),
                    root_link_ang_vel_b=torch.tensor([[3.,2.,.4],[0,0,0.]])))
  env = NS(scene={'robot':robot}, command_manager=NS(get_command=lambda name:torch.tensor([[.3,0,.4],[0,0,0.]])))
  term = joint_position_change(None,env)
  q[0,0] = .1
  torch.testing.assert_close(term(env),torch.tensor([.01,0.]))
  q[0,0] = .2
  term.reset(torch.tensor([0]))
  torch.testing.assert_close(term(env),torch.zeros(2))
  torch.testing.assert_close(planar_tracking(env,'twist',.4),torch.ones(2))
  torch.testing.assert_close(yaw_tracking(env,'twist',.4),torch.ones(2))
