"""Record a Go2 OCS2 gait only after the simulated motion is stable.

Run with ROS Humble's /usr/bin/python3 after sourcing the OCS2 workspace env.sh.
Do not run the keyboard or joystick publisher at the same time.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import rclpy
from control_input_msgs.msg import Inputs
from go2_interfaces.msg import FootContacts
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


TOPICS = (
    "/clock",
    "/go2/base_odom",
    "/joint_states",
    "/go2/foot_contacts",
    "/control_input",
)


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def tilt_radians(quat) -> float:
    # Angle between body +Z and world +Z; yaw does not affect it.
    up_z = 1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    return math.acos(max(-1.0, min(1.0, up_z)))


class StableTrotRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("stable_go2_ocs2_trot_recorder")
        self.args = args
        self.publisher = self.create_publisher(Inputs, "/control_input", 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(Odometry, "/go2/base_odom", self.on_odom, 100)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 100)
        self.create_subscription(FootContacts, "/go2/foot_contacts", self.on_contacts, 100)
        self.clock_time: float | None = None
        self.last_odom: float | None = None
        self.last_joints: float | None = None
        self.last_contacts: float | None = None
        self.odom_samples: deque[tuple[float, float, float, float, float, float]] = deque()
        self.contact_samples: deque[tuple[float, bool, bool, bool, bool, int]] = deque()
        self.state = "waiting"
        self.state_since = time.monotonic()
        self.stance_command_at = 0.0
        self.last_velocity_command_at = 0.0
        self.trot_command_at = 0.0
        self.bag: subprocess.Popen | None = None
        self.record_start: float | None = None
        self.failed = False
        self.done = False
        self.create_timer(0.1, self.tick)

    def on_clock(self, msg: Clock) -> None:
        self.clock_time = stamp_seconds(msg.clock)

    def on_odom(self, msg: Odometry) -> None:
        t = stamp_seconds(msg.header.stamp)
        if self.last_odom is not None and t < self.last_odom:
            self.odom_samples.clear()
            self.contact_samples.clear()
        self.last_odom = t
        pose, twist = msg.pose.pose, msg.twist.twist
        self.odom_samples.append((
            t,
            pose.position.z,
            tilt_radians(pose.orientation),
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        ))
        self.trim(self.odom_samples, t, 4.0)

    def on_joints(self, msg: JointState) -> None:
        if len(msg.name) >= 12 and len(msg.position) >= 12 and len(msg.velocity) >= 12:
            self.last_joints = stamp_seconds(msg.header.stamp)

    def on_contacts(self, msg: FootContacts) -> None:
        t = stamp_seconds(msg.header.stamp)
        self.last_contacts = t
        contact = dict(zip(msg.foot_names, msg.in_contact))
        fr_rl = bool(contact.get("FR") and contact.get("RL") and
                     not contact.get("FL") and not contact.get("RR"))
        fl_rr = bool(contact.get("FL") and contact.get("RR") and
                     not contact.get("FR") and not contact.get("RL"))
        right = bool(contact.get("FR") and contact.get("RR") and
                     not contact.get("FL") and not contact.get("RL"))
        left = bool(contact.get("FL") and contact.get("RL") and
                    not contact.get("FR") and not contact.get("RR"))
        self.contact_samples.append((t, fr_rl, fl_rr, right, left,
                                     sum(bool(v) for v in contact.values())))
        self.trim(self.contact_samples, t, 4.0)

    @staticmethod
    def trim(samples: deque, now: float, window: float) -> None:
        while samples and samples[0][0] < now - window:
            samples.popleft()

    def send(
        self,
        command: int = 0,
        lx: float = 0.0,
        ly: float = 0.0,
        rx: float = 0.0,
    ) -> None:
        msg = Inputs()
        msg.command = command
        msg.lx = lx
        msg.ly = ly
        msg.rx = rx
        self.publisher.publish(msg)

    def fresh(self) -> bool:
        if self.clock_time is None:
            return False
        return all(t is not None and abs(self.clock_time - t) < 0.5 for t in (
            self.last_odom, self.last_joints, self.last_contacts
        ))

    def stable_stance(self) -> bool:
        if not self.fresh() or not self.odom_samples:
            return False
        samples = [s for s in self.odom_samples if s[0] >= self.clock_time - 2.0]
        if len(samples) < 20 or samples[-1][0] - samples[0][0] < 1.8:
            return False
        return all(0.28 < s[1] < 0.43 and s[2] < 0.18 for s in samples) and all(
            abs(s[3]) < 0.08 and abs(s[4]) < 0.08 for s in samples
        )

    def stable_motion(self) -> bool:
        if not self.fresh() or not self.odom_samples or not self.contact_samples:
            return False
        window_start = self.clock_time - 3.0
        odom = [s for s in self.odom_samples if s[0] >= window_start]
        contacts = [s for s in self.contact_samples if s[0] >= window_start]
        if (len(odom) < 30 or len(contacts) < 20 or
                odom[-1][0] - odom[0][0] < 2.8 or
                contacts[-1][0] - contacts[0][0] < 2.8):
            return False
        forward_speeds = [s[3] for s in odom]
        lateral_speeds = [s[4] for s in odom]
        turns = [s[5] for s in odom]
        if self.args.motion == "turn_left":
            progress = statistics.mean(turns) > self.args.min_yaw_rate
            motion_speeds = turns
        elif self.args.motion == "turn_right":
            progress = statistics.mean(turns) < -self.args.min_yaw_rate
            motion_speeds = turns
        elif self.args.motion == "side_left":
            progress = statistics.mean(lateral_speeds) > self.args.min_speed
            motion_speeds = lateral_speeds
        elif self.args.motion == "side_right":
            progress = statistics.mean(lateral_speeds) < -self.args.min_speed
            motion_speeds = lateral_speeds
        elif self.args.motion == "backward_trot":
            progress = statistics.mean(forward_speeds) < -self.args.min_speed
            motion_speeds = forward_speeds
        else:
            progress = statistics.mean(forward_speeds) > self.args.min_speed
            motion_speeds = forward_speeds
        if self.args.motion in ("trot", "backward_trot"):
            gait = sum(s[1] for s in contacts) >= 5 and sum(s[2] for s in contacts) >= 5
        elif self.args.motion == "pace":
            gait = sum(s[3] for s in contacts) >= 5 and sum(s[4] for s in contacts) >= 5
        else:
            # Walking and turning can have three-foot support; require changing
            # support sets rather than assuming a particular diagonal pattern.
            gait = len(set(s[5] for s in contacts)) >= 2
        # Pace is a lateral-sequence gait: the trunk sways sideways by design, so
        # its lateral velocity is much larger than a trot's and must not reject it.
        is_side_motion = self.args.motion.startswith("side_")
        lateral_limit = 0.5 if self.args.motion == "pace" else 0.16
        return (
            all(0.26 < s[1] < 0.43 and s[2] < 0.28 for s in odom)
            and (is_side_motion or all(abs(s[4]) < lateral_limit for s in odom))
            and (not is_side_motion or all(abs(s[3]) < 0.16 for s in odom))
            and (self.args.motion.startswith("turn_") or all(abs(s[5]) < 0.3 for s in odom))
            and progress
            and statistics.pstdev(motion_speeds) < 0.12
            and gait
        )

    def transition(self, state: str) -> None:
        self.state = state
        self.state_since = time.monotonic()
        self.get_logger().info(f"State: {state}")

    def start_bag(self) -> None:
        output = self.args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Bag path already exists: {output}")
        cmd = [sys.executable, "/opt/ros/humble/bin/ros2", "bag", "record",
               "-o", str(output), *TOPICS]
        self.bag = subprocess.Popen(cmd, start_new_session=True)
        self.record_start = self.clock_time
        self.get_logger().info(f"Stable {self.args.motion} confirmed; recording to {output}")
        self.transition("recording")

    def stop_bag(self) -> None:
        if self.bag is not None and self.bag.poll() is None:
            self.bag.send_signal(signal.SIGINT)
            try:
                self.bag.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.bag.terminate()
                self.bag.wait(timeout=5)

    def tick(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.state == "waiting":
            if self.fresh():
                self.send(command=2)
                self.stance_command_at = now
                self.transition("standing")
        elif self.state == "standing":
            if self.stable_stance():
                if self.args.motion == "stand":
                    self.start_bag()
                else:
                    self.send(command={"trot": 3, "pace": 6,
                                       "dynamic_walk": 8, "static_walk": 9,
                                       "flying_trot": 5, "turn_left": 3,
                                       "turn_right": 3, "side_left": 3,
                                       "side_right": 3,
                                       "backward_trot": 3}[self.args.motion])
                    self.trot_command_at = now
                    self.transition("trot_warmup")
            elif now - self.stance_command_at > 2.0:
                self.send(command=2)
                self.stance_command_at = now
        elif self.state == "trot_warmup":
            if now - self.trot_command_at > 0.5 and now - self.last_velocity_command_at > 0.5:
                self.send(lx=self.args.lx, ly=self.args.ly, rx=self.args.rx)
                self.last_velocity_command_at = now
            if self.stable_motion():
                self.start_bag()
        elif self.state == "recording":
            if now - self.last_velocity_command_at > 0.5:
                self.send(lx=self.args.lx, ly=self.args.ly, rx=self.args.rx)
                self.last_velocity_command_at = now
            if not self.fresh() or not self.odom_samples:
                self.failed = True
                self.done = True
                self.get_logger().error("Sensor stream stopped; rejecting this recording")
            elif (self.odom_samples[-1][1] < 0.24 or
                  self.odom_samples[-1][2] > 0.4):
                self.failed = True
                self.done = True
                self.get_logger().error("Robot became unstable; rejecting this recording")
            elif self.bag is not None and self.bag.poll() is not None:
                self.failed = True
                self.done = True
                self.get_logger().error("ros2 bag recorder exited unexpectedly")
            elif self.clock_time - self.record_start >= self.args.seconds:
                self.done = True
                self.get_logger().info("Recording complete")
            if self.done:
                self.stop_bag()
                self.send()
                self.send(command=2)
        if not self.done and now - self.args.started_at > self.args.timeout:
            self.failed = True
            self.done = True
            self.get_logger().error("Timed out waiting for stable motion; no bag was accepted")
            self.stop_bag()
            self.send()
            self.send(command=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ly", type=float, default=0.3,
                        help="OCS2 forward input, not measured m/s")
    parser.add_argument("--lx", type=float, default=0.0,
                        help="OCS2 lateral-input magnitude for side_left/right")
    parser.add_argument("--motion", choices=("trot", "pace", "dynamic_walk",
                                             "static_walk", "flying_trot",
                                             "turn_left", "turn_right",
                                             "side_left", "side_right",
                                             "backward_trot", "stand"),
                        default="trot")
    parser.add_argument("--rx", type=float, default=0.0,
                        help="OCS2 yaw-input magnitude for turn_left/right")
    parser.add_argument("--min-speed", type=float, default=0.04)
    parser.add_argument("--min-yaw-rate", type=float, default=0.1)
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="Accepted recording duration in simulation seconds")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Wall-clock timeout waiting for stable motion")
    args = parser.parse_args()
    if (args.seconds <= 0 or args.timeout <= 0 or
            not 0.0 <= args.ly <= 1.0 or not 0.0 <= args.lx <= 1.0 or
            not 0.0 <= args.rx <= 1.0):
        parser.error("ly/lx/rx must be in [0, 1], seconds and timeout must be positive")
    if args.motion == "turn_left":
        args.ly, args.rx = 0.0, -(args.rx or 0.3)
    elif args.motion == "turn_right":
        args.ly, args.rx = 0.0, args.rx or 0.3
    elif args.motion == "side_left":
        args.ly, args.lx = 0.0, -(args.lx or 0.3)
    elif args.motion == "side_right":
        args.ly, args.lx = 0.0, args.lx or 0.3
    elif args.motion == "backward_trot":
        args.ly = -(args.ly or 0.3)
    elif args.motion == "stand":
        args.ly = 0.0
    elif args.rx != 0.0:
        parser.error("rx is only used with turn_left and turn_right")
    if not args.motion.startswith("side_") and args.lx != 0.0:
        parser.error("lx is only used with side_left and side_right")
    args.started_at = time.monotonic()
    rclpy.init()
    node = StableTrotRecorder(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.failed = True
    finally:
        node.stop_bag()
        node.send()
        node.send(command=2)
        node.destroy_node()
        rclpy.shutdown()
    return 1 if node.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
