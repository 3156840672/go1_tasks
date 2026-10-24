"""Standalone sim2sim deployment script for Dog2 policies.

Loads a trained ONNX policy and runs it in a standalone MuJoCo simulation.
The observation computation matches the training actor (IMU + proprioception),
and PD control reproduces the training actuator configuration.

Usage:
    uv run sim2sim-deploy <onnx_path>
    uv run sim2sim-deploy <onnx_path> --play-speed 1.0
"""

from __future__ import annotations

import fcntl
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import tyro

# ── Dog2 Constants (mirrored from dog2_constants.py) ──────────────────────────

DOG2_XML = (
  Path(__file__).resolve().parents[1]
  / "asset_zoo"
  / "robots"
  / "dog2"
  / "xmls"
  / "dog2.xml"
)

SCENE_TERRAIN_XML = (
  Path(__file__).resolve().parents[1]
  / "asset_zoo"
  / "robots"
  / "dog2"
  / "xmls"
  / "scene_terrain.xml"
)

# PD gains (from dog2_constants.py).
STIFFNESS = 40.0  # Nm/rad
DAMPING = 1.0  # Nm/(rad/s)
EFFORT_LIMIT = 35.0  # Nm (from ElectricActuator)
ARMature = 0.000111842  # Rotor inertia (from Unitree Go1 spec).

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_csv_list(value: str) -> list[float]:
  """Parse a comma-separated list of floats from ONNX metadata."""
  return [float(x.strip()) for x in value.split(",") if x.strip()]


def _quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Apply inverse quaternion rotation: q^-1 * v * q."""
  q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
  uv_x = q_y * v[2] - q_z * v[1]
  uv_y = q_z * v[0] - q_x * v[2]
  uv_z = q_x * v[1] - q_y * v[0]
  uuv_x = q_y * uv_z - q_z * uv_y
  uuv_y = q_z * uv_x - q_x * uv_z
  uuv_z = q_x * uv_y - q_y * uv_x
  uv_x *= 2.0
  uv_y *= 2.0
  uv_z *= 2.0
  uuv_x *= 2.0
  uuv_y *= 2.0
  uuv_z *= 2.0
  return np.array(
    [
      v[0] + q_w * uv_x + uuv_x,
      v[1] + q_w * uv_y + uuv_y,
      v[2] + q_w * uv_z + uuv_z,
    ]
  )


# ── Joystick Reader ────────────────────────────────────────────────────────────

# Linux js_event struct (from linux/joystick.h).
_JS_EVENT_FMT = "IhBB"
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02

# Default axis mapping for a typical gamepad (Xbox, PS, etc.).
DEFAULT_AXIS_MAP: dict[str, tuple[int, float, float]] = {
  "lin_vel_x": (1, -1.0, 3.0),  # Left stick Y -> forward speed (up = positive).
  "lin_vel_y": (0, 1.0, 1.0),  # Left stick X -> lateral speed.
  "ang_vel_z": (3, 1.0, 0.7),  # Right stick X -> turning speed.
}


class JoystickReader:
  """Non-blocking Linux joystick reader via /dev/input/js*."""

  def __init__(self, device_path: str = "/dev/input/js0"):
    self._fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
    fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    self._axes: dict[int, float] = {}
    self._buttons: dict[int, bool] = {}
    self._pressed_this_frame: set[int] = set()
    self._connected = True

  def close(self) -> None:
    if self._fd >= 0:
      os.close(self._fd)
      self._fd = -1

  def poll(self) -> None:
    """Read all pending joystick events (non-blocking)."""
    self._pressed_this_frame.clear()
    while True:
      try:
        data = os.read(self._fd, _JS_EVENT_SIZE)
      except BlockingIOError:
        break
      except OSError:
        self._connected = False
        break
      if len(data) < _JS_EVENT_SIZE:
        break
      _time_ms, value, ev_type, number = struct.unpack(_JS_EVENT_FMT, data)
      if ev_type & _JS_EVENT_AXIS:
        self._axes[number] = value / 32767.0
      elif ev_type & _JS_EVENT_BUTTON:
        pressed = value == 1
        was_pressed = self._buttons.get(number, False)
        self._buttons[number] = pressed
        if pressed and not was_pressed:
          self._pressed_this_frame.add(number)

  def was_pressed(self, button: int) -> bool:
    """True if button was just pressed (rising edge) this poll cycle."""
    return button in self._pressed_this_frame

  def is_held(self, button: int) -> bool:
    """True if button is currently held."""
    return self._buttons.get(button, False)

  def get_command(
    self,
    axis_map: dict[str, tuple[int, float, float]] | None = None,
  ) -> tuple[float, float, float]:
    """Get (lin_vel_x, lin_vel_y, ang_vel_z) from current axis state."""
    if axis_map is None:
      axis_map = DEFAULT_AXIS_MAP
    cmd = [0.0, 0.0, 0.0]
    keys = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
    for i, key in enumerate(keys):
      axis, sign, scale = axis_map[key]
      raw = self._axes.get(axis, 0.0)
      cmd[i] = sign * raw * scale
    return (cmd[0], cmd[1], cmd[2])

  @property
  def connected(self) -> bool:
    return self._connected


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DeployConfig:
  onnx_path: tyro.conf.Positional[str]
  """Path to the ONNX policy file."""

  play_speed: float = 1.0
  """Simulation speed multiplier (1.0 = real-time)."""

  fixed_command: tuple[float, float, float] = (0.5, 0.0, 0.0)
  """Fixed velocity command (lin_vel_x, lin_vel_y, ang_vel_z) in body frame."""

  record_video: bool = False
  """Record a video of the simulation."""

  video_path: str = "sim2sim_deploy.mp4"
  """Output path for the recorded video."""

  duration: float = 60.0
  """Simulation duration in seconds."""

  no_terrain: bool = False
  """Load bare dog2.xml without terrain (robot will fall if no floor)."""

  joystick: bool = False
  """Use joystick/gamepad for velocity commands instead of fixed_command."""

  joystick_device: str = "/dev/input/js0"
  """Joystick device path (Linux only)."""

  start_in_rl: bool = False
  """Start directly in RL mode (default: stand mode)."""


# ── Observation Builder ───────────────────────────────────────────────────────


class ObservationBuilder:
  """Builds the actor observation vector matching the training config.

  Observation order (from dog2 env_cfgs):
    1. base_ang_vel          (3): gyro reading from IMU
    2. projected_gravity     (3): gravity vector in body frame
    3. joint_pos             (12): joint pos relative to default
    4. joint_vel             (12): joint vel (default is zero)
    5. actions               (12): last processed action (target joint position)
    6. command               (3): velocity command in body frame
  Total: 45 dims
  """

  def __init__(
    self,
    joint_names: list[str],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
  ):
    self._joint_names = joint_names
    self._default_joint_pos = default_joint_pos
    self._action_scale = action_scale
    self._num_joints = len(joint_names)
    self._last_action: np.ndarray = np.zeros(self._num_joints, dtype=np.float32)
    # total dim: 3 + 3 + 12 + 12 + 12 + 3 = 45
    self.obs_dim = 3 + 3 + self._num_joints * 3 + 3

  def build(
    self,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    command: tuple[float, float, float],
  ) -> np.ndarray:
    """Build the observation vector from current simulation state."""
    # 1. base_ang_vel: gyro reading (sensor adr 0, dim 3).
    base_ang_vel = np.array(d.sensordata[0:3].copy())

    # 2. projected_gravity: gravity in body frame.
    # Body quaternion is qpos[3:7] of the free joint.
    body_quat = d.qpos[3:7].copy()  # w, x, y, z
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    projected_gravity = _quat_apply_inverse(body_quat, gravity_w)

    # 3. joint_pos_rel: current joint pos - default joint pos.
    joint_pos = d.qpos[7 : 7 + self._num_joints].copy()
    joint_pos_rel = joint_pos - self._default_joint_pos

    # 4. joint_vel_rel: current joint vel (default is 0).
    joint_vel = d.qvel[6 : 6 + self._num_joints].copy()

    # 5. last action — stored as processed action (target joint position),
    #    matching the training's JointPositionAction behavior.
    last_action = self._last_action.copy()

    # 6. command: velocity command in body frame.
    cmd = np.array(command, dtype=np.float32)

    obs = np.concatenate(
      [
        base_ang_vel,
        projected_gravity,
        joint_pos_rel,
        joint_vel,
        last_action,
        cmd,
      ]
    ).astype(np.float32)

    assert obs.shape == (self.obs_dim,), (
      f"Expected obs dim {self.obs_dim}, got {obs.shape}"
    )
    return obs

  def update_last_action(self, action_norm: np.ndarray) -> None:
    """Store the last processed action (target joint position).

    The training stores ``processed = action_norm * scale + default_joint_pos``,
    not the raw normalized action.
    """
    self._last_action = (
      action_norm * self._action_scale + self._default_joint_pos
    ).astype(np.float32)


# ── Main Deployment Loop ──────────────────────────────────────────────────────


def _load_metadata(onnx_path: str) -> dict:
  """Load metadata from ONNX model."""
  model = onnx.load(onnx_path)
  metadata = {}
  for prop in model.metadata_props:
    metadata[prop.key] = prop.value
  return metadata


def _get_actuated_joint_info(m: mujoco.MjModel) -> list[str]:
  """Get the joint names for all non-free joints."""
  joint_names: list[str] = []
  for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    joint_names.append(name)
  return joint_names


def _load_model_with_armature(xml_path: Path, joint_names: list[str]) -> mujoco.MjModel:
  """Load MJCF and set armature on joints (matching training).

  The training uses IdealPdActuator which computes PD torque in software
  and writes to ctrl (gaintype=FIXED, gain=1 → ctrl IS torque).
  Our manual PD via qfrc_applied is equivalent. We only need to match
  armature (reflected rotor inertia) from the training config.
  """
  spec = mujoco.MjSpec.from_file(str(xml_path))
  for jname in joint_names:
    spec.joint(jname).armature = ARMature
  return spec.compile()


def run_deploy(cfg: DeployConfig) -> None:
  onnx_path = Path(cfg.onnx_path)
  if not onnx_path.exists():
    print(f"Error: ONNX file not found: {onnx_path}")
    sys.exit(1)

  # ── Load ONNX model and metadata ──────────────────────────────────────
  metadata = _load_metadata(str(onnx_path))
  print("ONNX metadata:")
  for k, v in metadata.items():
    val_str = v[:80] + "..." if len(v) > 80 else v
    print(f"  {k}: {val_str}")

  # Use metadata if available, otherwise fall back to code constants.
  joint_names_meta: list[str] | None = None
  action_scale_meta: np.ndarray | None = None
  default_joint_pos_meta: np.ndarray | None = None

  if "joint_names" in metadata:
    joint_names_meta = metadata["joint_names"].split(",")
  if "action_scale" in metadata:
    action_scale_meta = np.array(_parse_csv_list(metadata["action_scale"]))
  if "default_joint_pos" in metadata:
    default_joint_pos_meta = np.array(_parse_csv_list(metadata["default_joint_pos"]))

  # ── Load MuJoCo model ─────────────────────────────────────────────────
  xml_path = DOG2_XML if cfg.no_terrain else SCENE_TERRAIN_XML
  print(f"\nLoading MJCF and adding actuators: {xml_path}")

  # First compile without actuators to get joint names.
  m_raw = mujoco.MjModel.from_xml_path(str(xml_path))
  joint_names = _get_actuated_joint_info(m_raw)
  num_joints = len(joint_names)
  print(f"Actuated joints ({num_joints}): {joint_names}")

  # Build joint order mapping: metadata → MJCF.
  if joint_names_meta is not None:
    missing = set(joint_names_meta) - set(joint_names)
    if missing:
      print(f"Warning: metadata joints not in MJCF: {missing}")
    else:
      print("Joint order: meta == MJCF (verified)")
  else:
    joint_names_meta = joint_names

  # Recompile with armature set on joints.
  m = _load_model_with_armature(xml_path, joint_names)
  d = mujoco.MjData(m)

  # Resolve default joint positions.
  if default_joint_pos_meta is not None:
    default_joint_pos = default_joint_pos_meta
  else:
    # Fallback: all zeros (robot will collapse — ONNX metadata preferred).
    default_joint_pos = np.zeros(num_joints)
    print("Warning: no default_joint_pos in metadata, using zeros.")

  # Resolve action scale.
  if action_scale_meta is not None:
    action_scale = action_scale_meta
  else:
    action_scale = np.full(num_joints, 0.25 * 35.0 / 40.0)
  print(f"Action scale: {np.round(action_scale, 3)}")

  # ── Initialize simulation state ───────────────────────────────────────
  # Use training INIT_STATE (from dog2_constants: pos=(0, 0, 0.278)).
  d.qpos[0:3] = [0.0, 0.0, 0.278]
  d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion.
  # Set initial joint positions to defaults.
  d.qpos[7 : 7 + num_joints] = default_joint_pos
  mujoco.mj_forward(m, d)

  # ── Set up observation builder ────────────────────────────────────────
  obs_builder = ObservationBuilder(
    joint_names=joint_names_meta,
    default_joint_pos=default_joint_pos,
    action_scale=action_scale,
  )

  # ── Load ONNX runtime session ─────────────────────────────────────────
  ort_session = ort.InferenceSession(str(onnx_path))
  input_name = ort_session.get_inputs()[0].name
  print(f"\nONNX input: {input_name}, dims: {ort_session.get_inputs()[0].shape}")
  print(f"Observation dim: {obs_builder.obs_dim}")

  # ── Viewer ────────────────────────────────────────────────────────────
  frames: list[Any] = []
  if cfg.record_video:
    renderer = mujoco.Renderer(m, 480, 640)
  else:
    renderer = None

  with mujoco.viewer.launch_passive(m, d) as viewer:  # type: ignore[attr-defined]
    if viewer is None:
      print(
        "Warning: No viewer available (headless). Use --record-video to capture output."
      )
      if not cfg.record_video:
        print("Running headless simulation...")

    viewer_running = viewer is not None and viewer.is_running()

    # ── Joystick setup ─────────────────────────────────────────────────
    joystick: JoystickReader | None = None
    if cfg.joystick:
      try:
        joystick = JoystickReader(cfg.joystick_device)
        print(f"Joystick connected: {cfg.joystick_device}")
      except OSError as e:
        print(f"Warning: Cannot open joystick {cfg.joystick_device}: {e}")
        print("Falling back to fixed_command.")
        joystick = None

    sim_time = 0.0
    model_timestep = m.opt.timestep
    # Decimation: policy runs at 50 Hz (matching training decimation=4, dt=0.005).
    # Training: sim_dt=0.005, decimation=4 => policy_dt=0.02 => 50 Hz.
    steps_per_policy = 4

    step_count = 0
    real_start = time.time()
    sim_start = sim_time

    # Initialize target_positions to default joint pos.
    target_positions: np.ndarray = default_joint_pos.copy()
    mode: str = "rl" if cfg.start_in_rl else "stand"
    print(f"Starting in {'RL' if mode == 'rl' else 'STAND'} mode")

    while (
      sim_time < cfg.duration
      and (viewer is None or viewer.is_running())
      and viewer_running
    ):
      # ── Poll joystick and handle mode switches ────────────────────
      if joystick is not None:
        joystick.poll()
        if joystick.was_pressed(0):  # A button → stand.
          mode = "stand"
          print("[Joystick] A pressed — entering STAND mode")
        elif joystick.was_pressed(1):  # B button → RL.
          mode = "rl"
          print("[Joystick] B pressed — entering RL mode")

      # ── Compute observation and run policy ───────────────────────
      if step_count % steps_per_policy == 0:
        # Read joystick or use fixed command.
        if joystick is not None:
          command: tuple[float, float, float] = joystick.get_command()
        else:
          command = cfg.fixed_command

        if mode == "stand":
          # Stand: hold default joint positions via PD.
          target_positions = default_joint_pos.copy()
          # Last action for stand = default_joint_pos (action_norm=0 -> processed=default).
          obs_builder.update_last_action(np.zeros(num_joints, dtype=np.float32))
        else:
          obs = obs_builder.build(m, d, command=command)
          ort_inputs = {input_name: obs[np.newaxis, :]}
          ort_outputs = ort_session.run(None, ort_inputs)
          action_norm: np.ndarray = np.asarray(ort_outputs[0][0])  # type: ignore[index]
          # NOTE: training uses clip_actions=None — no clipping.
          # Clipping here would cut off the policy's intended output range.
          obs_builder.update_last_action(action_norm)
          target_positions = default_joint_pos + action_norm * action_scale

      # ── Apply PD control (matching training IdealPdActuator) ──
      # Training: IdealPdActuator computes torque in software →
      #   writes to gaintype=FIXED ctrl. Our qfrc_applied is equivalent.
      current_pos = d.qpos[7 : 7 + num_joints]
      current_vel = d.qvel[6 : 6 + num_joints]
      torque = STIFFNESS * (target_positions - current_pos) - DAMPING * current_vel
      torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
      d.qfrc_applied[6 : 6 + num_joints] = torque

      # ── Step simulation ─────────────────────────────────────────
      mujoco.mj_step(m, d)

      # ── Render ──────────────────────────────────────────────────
      if viewer is not None and viewer.is_running():
        viewer.sync()

      if renderer is not None and step_count % 2 == 0:
        renderer.update_scene(d)
        frames.append(renderer.render().copy())

      # ── Timing ──────────────────────────────────────────────────
      step_count += 1
      sim_time += model_timestep
      if cfg.play_speed != 1.0:
        elapsed = time.time() - real_start
        target_elapsed = (sim_time - sim_start) / cfg.play_speed
        sleep_time = target_elapsed - elapsed
        if sleep_time > 0:
          time.sleep(sleep_time)

  # ── Save video ────────────────────────────────────────────────────────────
  if joystick is not None:
    joystick.close()

  if renderer is not None and frames:
    print(f"\nSaving video ({len(frames)} frames) to {cfg.video_path}...")
    import imageio

    imageio.mimsave(cfg.video_path, frames, fps=int(1.0 / model_timestep / 2))  # type: ignore[arg-type]
    print("Done.")


def main():
  import mjlab.tasks  # noqa: F401 — populate task registry

  cfg = tyro.cli(DeployConfig, config=None)
  run_deploy(cfg)


if __name__ == "__main__":
  main()
