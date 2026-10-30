"""Validate Go2 AMP clips, kinematics, frame transforms, and distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from go1_tasks.amp_data import AmpMotionDataset, REQUIRED_FIELDS, RESET_FIELDS
from go2_tasks.robots.go2 import get_spec


EXPECTED_CATEGORIES = {
  "trot", "flying_trot", "pace", "dynamic_walk", "static_walk",
  "turn_left", "turn_right", "side_left", "side_right", "backward_trot",
  "stand",
}
FEET = ("FR", "FL", "RR", "RL")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("path", type=Path, nargs="?",
                      default=Path("logs/amp/go2_reference"))
  parser.add_argument("--allow-partial", action="store_true")
  args = parser.parse_args()
  paths = sorted(args.path.rglob("*.npz"))
  if not paths:
    parser.error(f"No NPZ clips under {args.path}")

  model = get_spec().compile()
  data = mujoco.MjData(model)
  model_joints = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(1, model.njnt)
  )
  joint_qpos = [int(model.joint(name).qposadr[0]) for name in model_joints]
  required = set(REQUIRED_FIELDS) | set(RESET_FIELDS) | {
    "fps", "joint_names", "foot_site_names",
  }
  transitions_by_category: dict[str, int] = {}

  for path in paths:
    with np.load(path, allow_pickle=False) as clip:
      missing = required - set(clip.files)
      if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
      names = tuple(str(v) for v in clip["joint_names"])
      feet = tuple(str(v) for v in clip["foot_site_names"])
      fps = float(clip["fps"])
      n = len(clip["root_pos_w"])
      if names != model_joints or feet != FEET or not np.isclose(fps, 50.0):
        raise ValueError(f"{path}: robot names or FPS differ from Go2 AMP")
      if n != 226:
        raise ValueError(f"{path}: expected 226 frames (4.5 s), got {n}")
      if any(not np.isfinite(clip[key]).all() for key in required if key in clip.files
             and np.issubdtype(clip[key].dtype, np.number)):
        raise ValueError(f"{path}: nonfinite values")
      quat = clip["root_quat_w"]
      if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{path}: non-unit quaternion")
      rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
      if not np.allclose(rotation.apply(clip["root_lin_vel_b"]),
                         clip["root_lin_vel_w"], atol=2e-4):
        raise ValueError(f"{path}: linear velocity frame mismatch")
      if not np.allclose(rotation.apply(clip["root_ang_vel_b"]),
                         clip["root_ang_vel_w"], atol=2e-4):
        raise ValueError(f"{path}: angular velocity frame mismatch")
      for frame in range(n):
        data.qpos[:3] = clip["root_pos_w"][frame]
        data.qpos[3:7] = quat[frame]
        data.qpos[joint_qpos] = clip["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        actual_feet = np.stack([data.site(name).xpos for name in FEET])
        if not np.allclose(actual_feet, clip["foot_pos_w"][frame], atol=1e-4):
          raise ValueError(f"{path}: foot FK mismatch at frame {frame}")
      category = path.parent.name
      transitions_by_category[category] = transitions_by_category.get(category, 0) + n - 1
      lin = clip["root_lin_vel_b"].mean(axis=0)
      yaw = float(clip["root_ang_vel_b"][:, 2].mean())
      if category == "turn_left" and yaw < 0.08:
        raise ValueError(f"{path}: left-turn yaw rate is too small or has wrong sign")
      if category == "turn_right" and yaw > -0.08:
        raise ValueError(f"{path}: right-turn yaw rate is too small or has wrong sign")
      if category == "side_left" and lin[1] < 0.04:
        raise ValueError(f"{path}: leftward velocity is too small or has wrong sign")
      if category == "side_right" and lin[1] > -0.04:
        raise ValueError(f"{path}: rightward velocity is too small or has wrong sign")
      if category.startswith("side_") and (abs(lin[0]) > 0.04 or abs(yaw) > 0.05):
        raise ValueError(f"{path}: side motion has excessive forward or yaw drift")
      if category == "backward_trot" and lin[0] > -0.08:
        raise ValueError(f"{path}: backward velocity is too small or has wrong sign")
      if category == "backward_trot" and (abs(lin[1]) > 0.04 or abs(yaw) > 0.05):
        raise ValueError(f"{path}: backward motion has excessive lateral or yaw drift")
      if category == "stand":
        lin_rms = float(np.sqrt(np.mean(clip["root_lin_vel_b"] ** 2)))
        ang_rms = float(np.sqrt(np.mean(clip["root_ang_vel_b"] ** 2)))
        joint_rms = float(np.sqrt(np.mean(clip["joint_vel"] ** 2)))
        if lin_rms > 0.02 or ang_rms > 0.02 or joint_rms > 0.05:
          raise ValueError(
            f"{path}: stand clip is moving: lin={lin_rms:.3f}, "
            f"ang={ang_rms:.3f}, joint={joint_rms:.3f}"
          )
      print(f"{category:13} {path.name:31} {(n-1)/fps:4.2f}s  "
            f"vx={lin[0]:+.3f} vy={lin[1]:+.3f} yaw={yaw:+.3f}")

  categories = set(transitions_by_category)
  if not args.allow_partial and categories != EXPECTED_CATEGORIES:
    raise ValueError(
      f"Categories differ: missing={sorted(EXPECTED_CATEGORIES-categories)}, "
      f"extra={sorted(categories-EXPECTED_CATEGORIES)}"
    )
  dataset = AmpMotionDataset(args.path, joint_names=model_joints, fps=50.0)
  total = len(dataset)
  shares = ", ".join(
    f"{name}={count/total:.1%}"
    for name, count in sorted(transitions_by_category.items())
  )
  print(f"OK: {dataset.num_clips} clips, {total} transitions, shape "
        f"{tuple(dataset.transitions.shape)}")
  print(f"Transition distribution: {shares}")


if __name__ == "__main__":
  main()
