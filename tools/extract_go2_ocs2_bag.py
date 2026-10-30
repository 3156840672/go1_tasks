"""Extract actual Go2 base and joint states from a ROS 2 bag to a raw NPZ.

Use /usr/bin/python3 with the OCS2 workspace sourced. The resulting raw NPZ
is an intermediate file; convert it with convert_go2_ocs2_amp.py afterward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


def timestamp(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    odom_t, root_pos, root_quat, lin_b, ang_b = [], [], [], [], []
    joint_t, joint_pos, joint_vel = [], [], []
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == "/go2/base_odom":
            msg = deserialize_message(raw, Odometry)
            p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
            odom_t.append(timestamp(msg.header.stamp))
            root_pos.append((p.x, p.y, p.z))
            root_quat.append((q.w, q.x, q.y, q.z))
            lin_b.append((v.linear.x, v.linear.y, v.linear.z))
            ang_b.append((v.angular.x, v.angular.y, v.angular.z))
        elif topic == "/joint_states":
            msg = deserialize_message(raw, JointState)
            names = {name: i for i, name in enumerate(msg.name)}
            if any(name not in names for name in JOINT_NAMES):
                raise ValueError("JointState is missing Go2 joints")
            if len(msg.position) != len(msg.name) or len(msg.velocity) != len(msg.name):
                raise ValueError("JointState position or velocity is incomplete")
            joint_t.append(timestamp(msg.header.stamp))
            joint_pos.append([msg.position[names[name]] for name in JOINT_NAMES])
            joint_vel.append([msg.velocity[names[name]] for name in JOINT_NAMES])
    if len(odom_t) < 2 or len(joint_t) < 2:
        raise ValueError("Bag has insufficient base or joint samples")
    for label, stamps in (("odometry", odom_t), ("joints", joint_t)):
        if np.any(np.diff(stamps) <= 0):
            raise ValueError(f"{label} timestamps are not strictly increasing")
        if max(np.diff(stamps)) > 0.05:
            raise ValueError(f"{label} has a gap longer than 50 ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        odom_t=np.asarray(odom_t, np.float64),
        root_pos_w=np.asarray(root_pos, np.float64),
        root_quat_w=np.asarray(root_quat, np.float64),
        root_lin_vel_b=np.asarray(lin_b, np.float64),
        root_ang_vel_b=np.asarray(ang_b, np.float64),
        joint_t=np.asarray(joint_t, np.float64),
        joint_pos=np.asarray(joint_pos, np.float64),
        joint_vel=np.asarray(joint_vel, np.float64),
        joint_names=np.asarray(JOINT_NAMES),
    )
    print(f"Extracted {len(odom_t)} odometry and {len(joint_t)} joint frames: {args.output}")


if __name__ == "__main__":
    main()
