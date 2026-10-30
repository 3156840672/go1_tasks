"""Convert extracted Go2 OCS2 states into a 50 Hz Go2 AMP reference clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from go2_tasks.robots.go2 import get_spec


FEET = ("FR", "FL", "RR", "RL")


def interpolate(t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(grid, t, values[:, i]) for i in range(values.shape[1])], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Keep this much of the stable recorded interval")
    args = parser.parse_args()
    if args.fps != 50.0:
        parser.error("Current Go2 AMP environment requires 50 Hz clips")
    with np.load(args.raw, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    model = get_spec().compile()
    sim = mujoco.MjData(model)
    names = tuple(str(name) for name in raw["joint_names"])
    model_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(1, model.njnt))
    if names != model_names:
        raise ValueError(f"Raw joint order {names} differs from model order {model_names}")
    start = max(raw["odom_t"][0], raw["joint_t"][0]) + 0.02
    stop = min(raw["odom_t"][-1], raw["joint_t"][-1]) - 0.02
    grid = start + np.arange(int((stop - start) * args.fps) + 1) / args.fps
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            parser.error("--max-seconds must be positive")
        grid = grid[:int(args.max_seconds * args.fps) + 1]
    if len(grid) < 2:
        raise ValueError("No complete 50 Hz interval in bag")
    pos = interpolate(raw["odom_t"], raw["root_pos_w"], grid)
    quat = raw["root_quat_w"]
    xyzw = quat[:, [1, 2, 3, 0]]
    rot = Slerp(raw["odom_t"] - start,
                Rotation.from_quat(xyzw))(grid - start)
    quat = rot.as_quat()[:, [3, 0, 1, 2]]
    lin_b = interpolate(raw["odom_t"], raw["root_lin_vel_b"], grid)
    ang_b = interpolate(raw["odom_t"], raw["root_ang_vel_b"], grid)
    lin_w = rot.apply(lin_b)
    ang_w = rot.apply(ang_b)
    joints = interpolate(raw["joint_t"], raw["joint_pos"], grid)
    joint_vel = interpolate(raw["joint_t"], raw["joint_vel"], grid)
    addresses = [int(model.joint(name).qposadr[0]) for name in names]
    feet = np.empty((len(grid), 4, 3), dtype=np.float32)
    for i in range(len(grid)):
        sim.qpos[:3] = pos[i]
        sim.qpos[3:7] = quat[i]
        sim.qpos[addresses] = joints[i]
        mujoco.mj_forward(model, sim)
        feet[i] = [sim.site(name).xpos.copy() for name in FEET]
    fields = {
        "root_pos_w": pos,
        "root_quat_w": quat,
        "root_lin_vel_w": lin_w,
        "root_ang_vel_w": ang_w,
        "root_lin_vel_b": lin_b,
        "root_ang_vel_b": ang_b,
        "joint_pos": joints,
        "joint_vel": joint_vel,
        "foot_pos_w": feet,
    }
    if any(not np.isfinite(values).all() for values in fields.values()):
        raise ValueError("Converted clip contains nonfinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{key: values.astype(np.float32) for key, values in fields.items()},
        fps=np.array(args.fps, np.float32),
        joint_names=np.asarray(names),
        foot_site_names=np.asarray(FEET),
    )
    print(f"Saved {len(grid)} frames ({(len(grid)-1)/args.fps:.2f} s) to {args.output}")
    print(f"Mean body velocity: {lin_b.mean(axis=0).round(3)} m/s; "
          f"mean yaw rate: {ang_b[:, 2].mean():.3f} rad/s")


if __name__ == "__main__":
    main()
