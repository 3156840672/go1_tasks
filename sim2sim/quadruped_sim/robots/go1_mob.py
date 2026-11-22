"""Go1 MoB: gait/behaviour-tunable policy simulator."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from ..policies.actor import ACTION_DIM
from ..policies.mob import OBS_DIM, load_mob_policy
from ..scene import follow_shadow_light, shadow_light_mocap, spawn_position
from .go1 import BASE_HEIGHT, DEFAULT, JOINTS, make_model

HISTORY_LENGTH = 30
CONTROL_DECIMATION = 4
ACTION_SCALE = np.tile((0.125, 0.25, 0.25), 4)
GAITS = {
  "Pronk": (0.0, 0.0, 0.0),
  "Trot": (0.5, 0.0, 0.0),
  "Bound": (0.0, 0.5, 0.0),
  "Pace": (0.0, 0.0, 0.5),
}
TERM_DIMS = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "joint_pos": 12,
  "joint_vel": 12,
  "actions": 12,
  "command": 3,
  "behavior": 8,
  "clock": 4,
}
assert sum(TERM_DIMS.values()) * HISTORY_LENGTH == OBS_DIM

#: Behaviour slots in ``MobSimulator.behavior`` after the three gait terms.
BEHAVIOUR_BOUNDS = {
  "frequency": (1.5, 4.0),
  "body_height": (0.20, 0.38),
  "body_pitch": (-0.15, 0.15),
  "stance_width": (0.22, 0.40),
  "swing_height": (0.04, 0.18),
}
#: Neutral behaviour values, in the same order as ``BEHAVIOUR_BOUNDS``.
BEHAVIOUR_DEFAULTS = {
  "frequency": 2.5,
  "body_height": 0.30,
  "body_pitch": 0.0,
  "stance_width": 0.30,
  "swing_height": 0.10,
}
assert tuple(BEHAVIOUR_DEFAULTS) == tuple(BEHAVIOUR_BOUNDS)


class MobGamepadControls:
  """Gamepad mapping for gait switching and behaviour tuning."""

  PARAMETERS = (
    ("body_height", 4, 0.01),
    ("body_pitch", 5, 0.01),
    ("frequency", 3, 0.1),
    ("stance_width", 6, 0.01),
    ("swing_height", 7, 0.01),
  )

  def __init__(self, bounds: dict[str, tuple[float, float]]) -> None:
    self.bounds = bounds
    self.selected = 0

  @property
  def selected_name(self) -> str:
    return self.PARAMETERS[self.selected][0]

  def handle(self, sim: MobSimulator, buttons: set[int], dpad: set[str]) -> bool:
    if 0 in buttons:
      sim.set_active(True)
    if 1 in buttons:
      sim.set_active(False)
    reset = 2 in buttons
    if reset:
      sim.reset()

    gait_names = tuple(GAITS)
    if 3 in buttons or "right" in dpad:
      sim.set_gait(gait_names[(gait_names.index(sim.gait) + 1) % len(gait_names)])
    elif "left" in dpad:
      sim.set_gait(gait_names[(gait_names.index(sim.gait) - 1) % len(gait_names)])

    if "up" in dpad:
      self.selected = (self.selected - 1) % len(self.PARAMETERS)
    elif "down" in dpad:
      self.selected = (self.selected + 1) % len(self.PARAMETERS)

    name, index, step = self.PARAMETERS[self.selected]
    if 4 in buttons:
      sim.adjust_behavior(index, -step, self.bounds[name])
    if 5 in buttons:
      sim.adjust_behavior(index, step, self.bounds[name])
    return reset


class MobSimulator:
  """Go1 MoB policy with the 1710-D observation assembled from 30 control steps."""

  def __init__(
    self,
    checkpoint: Path | None,
    terrain: str = "flat",
    terrain_seed: int = 42,
    terrain_type: str | None = None,
    terrain_level: int = 0,
    terrain_column: str | None = None,
    no_shadows: bool = False,
  ) -> None:
    self.model = make_model(
      terrain, terrain_seed, terrain_type, terrain_level, mob=True
    )
    self.shadows_enabled = not no_shadows
    if no_shadows:
      self.model.vis.quality.shadowsize = 0
    self.spawn = spawn_position(
      self.model, terrain, terrain_type, terrain_level, terrain_column
    )
    self.light_mocap_id = shadow_light_mocap(self.model)
    self.data = mujoco.MjData(self.model)
    self.qadr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINTS
      ]
    )
    self.vadr = np.array(
      [
        self.model.jnt_dofadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINTS
      ]
    )
    self.aid = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor")
        for name in JOINTS
      ]
    )
    self.ang_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
    )
    if min(*self.qadr, *self.vadr, *self.aid, self.ang_id) < 0:
      raise ValueError("Go1 model is missing a joint, actuator, or IMU gyro")

    self.policy = load_mob_policy(checkpoint) if checkpoint is not None else None
    self.command = np.zeros(3, dtype=np.float32)
    self.behavior = np.array(
      (*GAITS["Trot"], *BEHAVIOUR_DEFAULTS.values()), dtype=np.float32
    )
    self.gait = "Trot"
    self.active = False
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:7] = [
      self.spawn[0],
      self.spawn[1],
      self.spawn[2] + BASE_HEIGHT,
      1.0,
      0.0,
      0.0,
      0.0,
    ]
    follow_shadow_light(self.data, self.light_mocap_id)
    self.data.qpos[self.qadr] = DEFAULT
    self.data.ctrl[self.aid] = DEFAULT
    self.target = DEFAULT.copy()
    self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    self.history = {
      name: np.zeros((HISTORY_LENGTH, dim), dtype=np.float32)
      for name, dim in TERM_DIMS.items()
    }
    self.history_ready = False
    self.cycle = 0.0
    mujoco.mj_forward(self.model, self.data)

  def set_active(self, active: bool) -> None:
    if active and self.policy is None:
      print("No MoB checkpoint loaded; remaining in passive stand")
      return
    self.active = active
    self.last_action.fill(0.0)
    for values in self.history.values():
      values.fill(0.0)
    self.history_ready = False
    self.target = DEFAULT.copy()
    print("MoB active" if active else "Passive stand")

  def set_gait(self, name: str) -> None:
    self.gait = name
    self.behavior[:3] = GAITS[name]

  def adjust_behavior(
    self, index: int, amount: float, limits: tuple[float, float]
  ) -> None:
    self.behavior[index] = np.clip(self.behavior[index] + amount, *limits)

  def base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame base linear velocity from the free joint plus the gyro."""
    gyro_adr = self.model.sensor_adr[self.ang_id]
    angular = self.data.sensordata[gyro_adr : gyro_adr + 3]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    linear = rotation.reshape(3, 3).T @ self.data.qvel[0:3]
    return linear, angular

  def raw_observation_terms(self) -> dict[str, np.ndarray]:
    gyro_adr = self.model.sensor_adr[self.ang_id]
    ang_vel = self.data.sensordata[gyro_adr : gyro_adr + 3]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
    gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    phase, bound, pace = self.behavior[:3]
    shifts = np.array((bound + pace, phase + bound, phase + pace, 0.0))
    clock = np.sin(2.0 * np.pi * ((self.cycle + shifts) % 1.0))
    return {
      "base_ang_vel": ang_vel.astype(np.float32),
      "projected_gravity": gravity.astype(np.float32),
      "joint_pos": (self.data.qpos[self.qadr] - DEFAULT).astype(np.float32),
      "joint_vel": self.data.qvel[self.vadr].astype(np.float32),
      "actions": self.last_action.copy(),
      "command": self.command.copy(),
      "behavior": self.behavior.copy(),
      "clock": clock.astype(np.float32),
    }

  def observation(self) -> np.ndarray:
    terms = self.raw_observation_terms()
    for name, current in terms.items():
      buffer = self.history[name]
      if self.history_ready:
        buffer[:-1] = buffer[1:].copy()
        buffer[-1] = current
      else:
        buffer[:] = current
    self.history_ready = True
    obs = np.concatenate([values.reshape(-1) for values in self.history.values()])
    assert obs.shape == (OBS_DIM,)
    return obs

  def policy_step(self) -> None:
    if not self.active:
      self.target = DEFAULT.copy()
      self.last_action.fill(0.0)
      return
    obs = self.observation()
    assert self.policy is not None
    with torch.inference_mode():
      action = self.policy(torch.from_numpy(obs[None]))[0].numpy()
    if not np.isfinite(action).all():
      raise FloatingPointError("MoB policy returned non-finite actions")
    self.target = DEFAULT + ACTION_SCALE * action
    self.last_action[:] = action
    self.cycle = (
      self.cycle + CONTROL_DECIMATION * self.model.opt.timestep * self.behavior[3]
    ) % 1.0

  def physics_step(self) -> None:
    self.data.ctrl[self.aid] = self.target
    mujoco.mj_step(self.model, self.data)
