"""Read MoB motion clips and build matching AMP features for live robot states."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

FOOT_NAMES = ("FR", "FL", "RR", "RL")
FRAME_DIM = 46
TRANSITION_DIM = 2 * FRAME_DIM
REQUIRED_FIELDS = (
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_b",
  "root_ang_vel_b",
  "joint_pos",
  "joint_vel",
  "foot_pos_w",
)
# The reset event additionally needs the world-frame root velocities that the
# discriminator feature does not use. Keep the two contracts in one place.
RESET_FIELDS = (
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
)


def amp_frame(
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
  root_lin_vel_b: torch.Tensor,
  root_ang_vel_b: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  foot_pos_w: torch.Tensor,
) -> torch.Tensor:
  """Build (N, 46) AMP states from (N, ...) tensors in a fixed joint order.

  The height assumes flat ground at world z=0. The four feet must be FR, FL,
  RR, RL. Commands and absolute x/y/yaw are intentionally excluded.
  """
  n = root_pos_w.shape[0]
  expected = (
    (root_pos_w, (n, 3)),
    (root_quat_w, (n, 4)),
    (root_lin_vel_b, (n, 3)),
    (root_ang_vel_b, (n, 3)),
    (joint_pos, (n, 12)),
    (joint_vel, (n, 12)),
    (foot_pos_w, (n, 4, 3)),
  )
  for value, shape in expected:
    if value.shape != shape:
      raise ValueError(f"Expected shape {shape}, got {tuple(value.shape)}")
  gravity_w = torch.zeros_like(root_pos_w)
  gravity_w[:, 2] = -1.0
  gravity_b = quat_apply_inverse(root_quat_w, gravity_w)
  foot_rel_w = foot_pos_w - root_pos_w[:, None, :]
  foot_rel_b = quat_apply_inverse(
    root_quat_w[:, None, :].expand(-1, 4, -1), foot_rel_w
  ).reshape(n, 12)
  return torch.cat(
    (
      root_pos_w[:, 2:3],
      gravity_b,
      root_lin_vel_b,
      root_ang_vel_b,
      joint_pos,
      joint_vel,
      foot_rel_b,
    ),
    dim=-1,
  )


def amp_frame_from_robot(
  robot,
  *,
  joint_names: tuple[str, ...] | None = None,
  foot_ids: Sequence[int] | None = None,
) -> torch.Tensor:
  """Extract the same AMP frame from every environment of a live Go1 robot.

  ``foot_ids`` may be cached once at startup to avoid resolving site names on
  every control step.
  """
  if joint_names is not None and tuple(robot.joint_names) != joint_names:
    raise ValueError("Live robot joint order differs from the reference motions")
  if foot_ids is None:
    foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(foot_names) != FOOT_NAMES:
      raise ValueError(f"Unexpected foot order: {foot_names}")
  data = robot.data
  return amp_frame(
    data.root_link_pos_w,
    data.root_link_quat_w,
    data.root_link_lin_vel_b,
    data.root_link_ang_vel_b,
    data.joint_pos,
    data.joint_vel,
    data.site_pos_w[:, foot_ids],
  )


class AmpMotionDataset:
  """Validated adjacent-frame samples from MoB .npz clips.

  Each transition is concatenated as [frame_t, frame_t+1]. Uniform random
  indices therefore sample transitions uniformly, including across clips.
  """

  def __init__(
    self,
    path: str | Path,
    *,
    joint_names: tuple[str, ...] | None = None,
    fps: float | None = None,
  ) -> None:
    paths = sorted(Path(path).rglob("*.npz"))
    if not paths:
      raise ValueError(f"No .npz motion clips found under {path}")
    transitions: list[torch.Tensor] = []
    canonical_joints = joint_names
    canonical_fps = fps
    for file_path in paths:
      with np.load(file_path, allow_pickle=False) as clip:
        required = set((*REQUIRED_FIELDS, "fps", "joint_names", "foot_site_names"))
        missing = required - set(clip.files)
        if missing:
          raise ValueError(f"{file_path}: missing fields {sorted(missing)}")
        n = len(clip["root_pos_w"])
        if n < 2:
          raise ValueError(f"{file_path}: at least two frames are required")
        shapes = {
          "root_pos_w": (n, 3),
          "root_quat_w": (n, 4),
          "root_lin_vel_b": (n, 3),
          "root_ang_vel_b": (n, 3),
          "joint_pos": (n, 12),
          "joint_vel": (n, 12),
          "foot_pos_w": (n, 4, 3),
        }
        for field, shape in shapes.items():
          values = clip[field]
          if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{file_path}: invalid {field}, expected {shape}")
        norms = np.linalg.norm(clip["root_quat_w"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
          raise ValueError(f"{file_path}: root quaternions are not normalized")
        names = tuple(str(name) for name in clip["joint_names"])
        feet = tuple(str(name) for name in clip["foot_site_names"])
        rate = float(clip["fps"])
        if len(names) != 12 or len(set(names)) != 12:
          raise ValueError(f"{file_path}: invalid joint names")
        if feet != FOOT_NAMES:
          raise ValueError(f"{file_path}: unexpected foot order {feet}")
        if not np.isfinite(rate) or rate <= 0:
          raise ValueError(f"{file_path}: invalid fps")
        if canonical_joints is None:
          canonical_joints = names
        if canonical_fps is None:
          canonical_fps = rate
        if names != canonical_joints or not np.isclose(rate, canonical_fps):
          raise ValueError(f"{file_path}: joint order or fps differs from the dataset")
        frame = amp_frame(
          *(torch.from_numpy(clip[key].astype(np.float32)) for key in REQUIRED_FIELDS)
        )
        transitions.append(torch.cat((frame[:-1], frame[1:]), dim=-1))
    self.transitions = torch.cat(transitions, dim=0)
    self.joint_names = canonical_joints
    self.fps = canonical_fps
    self.num_clips = len(paths)

  def __len__(self) -> int:
    return self.transitions.shape[0]

  def sample(
    self, batch_size: int, *, device: torch.device | str = "cpu"
  ) -> torch.Tensor:
    """Return uniformly sampled (batch_size, 92) transitions."""
    if batch_size < 1:
      raise ValueError("batch_size must be positive")
    indices = torch.randint(len(self), (batch_size,))
    return self.transitions[indices].to(device)
