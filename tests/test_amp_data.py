"""AMP motion features and clip boundary checks."""

import numpy as np
import pytest
import torch

from go1_tasks.amp_data import (
  FRAME_DIM,
  TRANSITION_DIM,
  AmpMotionDataset,
  amp_frame,
)


def _write_clip(path, height: float, *, joint_names=None) -> None:
  n = 3
  root_pos = np.zeros((n, 3), dtype=np.float32)
  root_pos[:, 2] = height
  root_quat = np.zeros((n, 4), dtype=np.float32)
  root_quat[:, 0] = 1.0
  joint_pos = np.full((n, 12), height, dtype=np.float32)
  joint_pos[:, 0] += np.arange(n)
  np.savez_compressed(
    path,
    root_pos_w=root_pos,
    root_quat_w=root_quat,
    root_lin_vel_b=np.zeros((n, 3), dtype=np.float32),
    root_ang_vel_b=np.zeros((n, 3), dtype=np.float32),
    joint_pos=joint_pos,
    joint_vel=np.zeros((n, 12), dtype=np.float32),
    foot_pos_w=np.zeros((n, 4, 3), dtype=np.float32),
    fps=np.array(50.0),
    joint_names=np.array(joint_names or [f"joint_{i}" for i in range(12)]),
    foot_site_names=np.array(["FR", "FL", "RR", "RL"]),
  )


def test_amp_frame_features() -> None:
  pos = torch.tensor([[1.0, 2.0, 0.3]])
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  linear = torch.tensor([[0.5, 0.0, 0.0]])
  angular = torch.zeros(1, 3)
  joints = torch.zeros(1, 12)
  foot = pos[:, None, :].expand(-1, 4, -1).clone()
  foot[:, 0, 0] += 0.2
  result = amp_frame(pos, quat, linear, angular, joints, joints, foot)
  assert result.shape == (1, FRAME_DIM)
  torch.testing.assert_close(result[0, :7], torch.tensor([0.3, 0, 0, -1, 0.5, 0, 0]))
  torch.testing.assert_close(result[0, -12:-9], torch.tensor([0.2, 0, 0]))


def test_dataset_only_pairs_within_each_clip(tmp_path) -> None:
  _write_clip(tmp_path / "a.npz", 0.3)
  _write_clip(tmp_path / "b.npz", 0.8)
  dataset = AmpMotionDataset(tmp_path, fps=50.0)
  assert dataset.num_clips == 2
  assert len(dataset) == 4
  assert dataset.transitions.shape == (4, TRANSITION_DIM)
  torch.testing.assert_close(
    dataset.transitions[:, 0], dataset.transitions[:, FRAME_DIM]
  )
  assert dataset.sample(7).shape == (7, TRANSITION_DIM)


def test_dataset_rejects_inconsistent_joint_order(tmp_path) -> None:
  _write_clip(tmp_path / "a.npz", 0.3)
  _write_clip(
    tmp_path / "b.npz",
    0.8,
    joint_names=[f"joint_{i}" for i in reversed(range(12))],
  )
  with pytest.raises(ValueError, match="joint order"):
    AmpMotionDataset(tmp_path)
