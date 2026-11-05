"""Explicit reward definitions for the paper teacher's next controlled trial."""
import torch


def planar_tracking(env, command_name, std):
  command = env.command_manager.get_command(command_name)
  error = (command[:, :2] - env.scene['robot'].data.root_link_lin_vel_b[:, :2]).square().sum(-1)
  return torch.exp(-error / std**2)


def yaw_tracking(env, command_name, std):
  command = env.command_manager.get_command(command_name)
  error = (command[:, 2] - env.scene['robot'].data.root_link_ang_vel_b[:, 2]).square()
  return torch.exp(-error / std**2)


class joint_position_change:
  """Actual joint displacement over one control step, not normalized actions."""
  def __init__(self, cfg, env):
    self.robot = env.scene['robot']
    self.previous = self.robot.data.joint_pos.clone()

  def __call__(self, env):
    current = self.robot.data.joint_pos
    penalty = (current - self.previous).square().sum(-1)
    self.previous.copy_(current)
    return penalty

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.previous[ids] = self.robot.data.joint_pos[ids]


def stride_duration(env, command_name, sensor_name):
  """Four-foot interpretation of Table I, gated off for strict standing.

  The paper's printed six-foot index and unspecified touchdown timing are
  ambiguous. This implementation uses current air duration every control step.
  """
  command = env.command_manager.get_command(command_name)
  air = env.scene[sensor_name].data.current_air_time
  value = (air - .5).clamp(max=0).sum(-1)
  return value * (command.norm(dim=-1) > .05)
