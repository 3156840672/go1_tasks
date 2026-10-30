"""Render a collected AMP reference clip as an animated Go1 skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_XML

LEGS = ("FR", "FL", "RR", "RL")
COLORS = ("#e45756", "#4c78a8", "#f2a541", "#54a24b")


def _load_clip(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as saved:
    required = (
      "fps", "root_pos_w", "root_quat_w", "joint_names", "joint_pos", "foot_pos_w"
    )
    missing = set(required) - set(saved.files)
    if missing:
      raise ValueError(f"{path}: missing fields {sorted(missing)}")
    clip = {name: saved[name] for name in saved.files}
  if len(clip["root_pos_w"]) < 2 or float(clip["fps"]) <= 0:
    raise ValueError(f"{path}: invalid frame count or fps")
  return clip


def _skeleton_frames(
  clip: dict[str, np.ndarray], indices: np.ndarray, model: mujoco.MjModel
) -> np.ndarray:
  """Return (frames, 4 legs, 4 points, xyz) in root-centered world coordinates."""
  data = mujoco.MjData(model)
  names = [str(name) for name in clip["joint_names"]]
  joint_addresses = [model.joint(name).qposadr[0] for name in names]
  frames = np.empty((len(indices), 4, 4, 3), dtype=np.float64)
  for out_idx, frame_idx in enumerate(indices):
    data.qpos[:3] = clip["root_pos_w"][frame_idx]
    data.qpos[3:7] = clip["root_quat_w"][frame_idx]
    data.qpos[joint_addresses] = clip["joint_pos"][frame_idx]
    mujoco.mj_forward(model, data)
    for leg_idx, leg in enumerate(LEGS):
      for point_idx, body in enumerate(("hip", "thigh", "calf")):
        frames[out_idx, leg_idx, point_idx] = data.body(f"{leg}_{body}").xpos
      frames[out_idx, leg_idx, 3] = data.site(leg).xpos
    # The reference stores site positions too. A mismatch means the saved joint
    # order, root convention, or robot model differs from this visualizer.
    if not np.allclose(frames[out_idx, :, 3], clip["foot_pos_w"][frame_idx], atol=1e-4):
      raise ValueError(f"Frame {frame_idx}: FK feet differ from stored reference feet")
    frames[out_idx, :, :, :2] -= clip["root_pos_w"][frame_idx, :2]
  return frames


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("clip", type=Path, help="Collected clip_*.npz file")
  parser.add_argument(
    "--robot", choices=("go1", "go2"), default="go1",
    help="Robot model used for forward kinematics",
  )
  parser.add_argument(
    "--output", type=Path, default=Path("logs/amp/reference_preview.gif")
  )
  parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
  parser.add_argument("--seconds", type=float, default=3.0, help="Duration in seconds")
  parser.add_argument("--fps", type=float, default=15.0, help="Output animation FPS")
  args = parser.parse_args()
  if args.start < 0 or args.seconds <= 0 or args.fps <= 0:
    parser.error("--start must be nonnegative; --seconds and --fps must be positive")
  clip = _load_clip(args.clip)
  if args.robot == "go1":
    model = mujoco.MjModel.from_xml_path(str(GO1_XML))
  else:
    from go2_tasks.robots.go2 import get_spec

    model = get_spec().compile()
  source_fps = float(clip["fps"])
  start = round(args.start * source_fps)
  stop = min(
    len(clip["root_pos_w"]) - 1,
    round((args.start + args.seconds) * source_fps),
  )
  if start >= stop:
    parser.error("Requested time range contains fewer than two reference frames")
  count = max(2, round((stop - start) / source_fps * args.fps) + 1)
  indices = np.unique(np.rint(np.linspace(start, stop, count)).astype(int))
  frames = _skeleton_frames(clip, indices, model)
  times = np.arange(len(clip["root_pos_w"])) / source_fps
  foot_z = clip["foot_pos_w"][:, :, 2]

  fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), width_ratios=(1, 1, 1.5),
                           layout="constrained")
  side, top, height = axes
  side.set(xlabel="Forward x relative to trunk (m)", ylabel="Height (m)",
           xlim=(-0.43, 0.43), ylim=(-0.04, 0.47), title="Side view")
  side.axhline(0, color="0.7", lw=1)
  top.set(
    xlabel="Forward x relative to trunk (m)",
    ylabel="Left y relative to trunk (m)",
    xlim=(-0.43, 0.43), ylim=(-0.33, 0.33), title="Top view"
  )
  for ax in (side, top):
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
  top.plot([-0.19, 0.19], [0, 0], color="black", lw=4)
  top.plot([0], [0], "ks", markersize=9)
  height.set(xlabel="Time (s)", ylabel="Foot height (m)",
             xlim=(times[start], times[stop]), title="Foot lift (diagonal pairs)")
  height.grid(alpha=0.2)
  for leg_idx, leg in enumerate(LEGS):
    height.plot(times[start:stop + 1], foot_z[start:stop + 1, leg_idx],
                color=COLORS[leg_idx], label=leg, lw=1.4)
  height.legend(ncol=2, loc="upper right")
  cursor = height.axvline(times[start], color="black", ls="--", lw=1)
  trunk = side.plot([], [], color="black", lw=4)[0]
  side_lines = [
    side.plot([], [], color=color, marker="o", ms=3, lw=2)[0] for color in COLORS
  ]
  top_lines = [
    top.plot([], [], color=color, marker="o", ms=3, lw=2)[0] for color in COLORS
  ]
  fig.suptitle(f"AMP reference: {args.clip.parent.name}/{args.clip.name}")

  def update(index: int):
    trunk_z = float(clip["root_pos_w"][indices[index], 2])
    trunk.set_data([-0.19, 0.19], [trunk_z, trunk_z])
    for leg_idx in range(4):
      points = frames[index, leg_idx]
      side_lines[leg_idx].set_data(points[:, 0], points[:, 2])
      top_lines[leg_idx].set_data(points[:, 0], points[:, 1])
    cursor.set_xdata([times[indices[index]], times[indices[index]]])
    return [trunk, *side_lines, *top_lines, cursor]

  animation = FuncAnimation(fig, update, frames=len(indices), interval=1000 / args.fps,
                            blit=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  animation.save(args.output, writer=PillowWriter(fps=args.fps))
  plt.close(fig)
  print(f"Saved {len(indices)} frames to {args.output}")


if __name__ == "__main__":
  main()
