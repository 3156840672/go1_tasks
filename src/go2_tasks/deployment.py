"""Stable Go2 policy I/O contract shared with real-robot deployment code."""

from __future__ import annotations

import numpy as np

from .robots.go2 import JOINT_NAMES

CONTROL_DT = 0.02
DEFAULT_JOINT_POS = np.array(
  [value for _ in range(4) for value in (0.0, 0.8, -1.5)], dtype=np.float32
)
ACTION_SCALE = np.array(
  [value for _ in range(4) for value in (0.125, 0.25, 0.25)], dtype=np.float32
)

JOINT_LOWER = np.array(
  [
    -1.0472,
    -1.5708,
    -2.7227,
    -1.0472,
    -1.5708,
    -2.7227,
    -1.0472,
    -0.5236,
    -2.7227,
    -1.0472,
    -0.5236,
    -2.7227,
  ],
  dtype=np.float32,
)
JOINT_UPPER = np.array(
  [
    1.0472,
    3.4907,
    -0.83776,
    1.0472,
    3.4907,
    -0.83776,
    1.0472,
    4.5379,
    -0.83776,
    1.0472,
    4.5379,
    -0.83776,
  ],
  dtype=np.float32,
)
KP = np.full(12, 25.0, dtype=np.float32)
KD = np.full(12, 0.5, dtype=np.float32)
# robot_lab UNITREE_GO2_CFG motor model; separate from the source MJCF limits.
TORQUE_LIMIT = np.full(12, 23.5, dtype=np.float32)
SATURATION_EFFORT = np.full(12, 23.5, dtype=np.float32)
VELOCITY_LIMIT = np.full(12, 30.0, dtype=np.float32)
SOURCE_MJCF_TORQUE_LIMIT = np.array(
  [value for _ in range(4) for value in (23.7, 23.7, 45.43)], dtype=np.float32
)


def policy_action_to_joint_targets(
  action: np.ndarray, *, clamp_to_limits: bool = True
) -> np.ndarray:
  """Convert canonical 12-D policy actions into Go2 position targets."""
  action = np.asarray(action, dtype=np.float32)
  if action.shape != (12,):
    raise ValueError(f"Expected a 12-D Go2 action, got {action.shape}")
  targets = DEFAULT_JOINT_POS + ACTION_SCALE * action
  if clamp_to_limits:
    targets = np.clip(targets, JOINT_LOWER, JOINT_UPPER)
  return targets


def flat_actor_observation(
  *,
  base_lin_vel: np.ndarray,
  base_ang_vel: np.ndarray,
  projected_gravity: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  previous_action: np.ndarray,
  command: np.ndarray,
) -> np.ndarray:
  """Build the 48-D Go2 Flat/Rough proprioceptive observation prefix."""
  values = tuple(
    np.asarray(value, dtype=np.float32).reshape(-1)
    for value in (
      base_lin_vel,
      base_ang_vel,
      projected_gravity,
      joint_pos - DEFAULT_JOINT_POS,
      joint_vel,
      previous_action,
      command,
    )
  )
  expected = (3, 3, 3, 12, 12, 12, 3)
  if tuple(len(value) for value in values) != expected:
    raise ValueError("Invalid Go2 observation component dimensions")
  return np.concatenate(values)


__all__ = [
  "ACTION_SCALE",
  "CONTROL_DT",
  "DEFAULT_JOINT_POS",
  "JOINT_LOWER",
  "JOINT_NAMES",
  "JOINT_UPPER",
  "KD",
  "KP",
  "SATURATION_EFFORT",
  "SOURCE_MJCF_TORQUE_LIMIT",
  "TORQUE_LIMIT",
  "VELOCITY_LIMIT",
  "flat_actor_observation",
  "policy_action_to_joint_targets",
]
