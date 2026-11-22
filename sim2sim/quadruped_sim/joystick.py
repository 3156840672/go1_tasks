"""Linux joystick polling; no extra package and no required device."""

from __future__ import annotations

import os
import select
import struct

import numpy as np


class Joystick:
  """Read ``/dev/input/js*`` events non-blockingly.

  Buttons ``0/1/2`` map to enable/passive/reset in every runner, the left stick
  drives the planar velocity command and the right stick's horizontal axis the
  yaw rate.  Direcional hats show up as ``hat_pressed``.
  """

  def __init__(self, path: str) -> None:
    self.fd = None
    self.axes = [0.0] * 8
    self.buttons = set()
    self.hat_pressed: set[str] = set()
    self._hat_held: set[str] = set()
    if path:
      try:
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        print(f"Joystick: {path}")
      except OSError as exc:
        print(f"Joystick unavailable ({exc}); keyboard remains active")

  def poll(self) -> set[int]:
    pressed = set()
    self.hat_pressed.clear()
    if self.fd is None:
      return pressed
    while select.select([self.fd], [], [], 0)[0]:
      try:
        event = os.read(self.fd, 8)
      except OSError:
        break
      if len(event) != 8:
        break
      _, value, kind, number = struct.unpack("IhBB", event)
      initial = bool(kind & 0x80)
      kind &= ~0x80
      if kind == 2 and number < len(self.axes):
        self.axes[number] = float(value) / 32768.0
        if number in (6, 7):
          held = self._hat_directions()
          if not initial:
            self.hat_pressed.update(held - self._hat_held)
          self._hat_held = held
      elif kind == 1:
        if value:
          self.buttons.add(number)
          if not initial:
            pressed.add(number)
        else:
          self.buttons.discard(number)
    return pressed

  def _hat_directions(self) -> set[str]:
    directions = set()
    if self.axes[6] < -0.5:
      directions.add("left")
    elif self.axes[6] > 0.5:
      directions.add("right")
    if self.axes[7] < -0.5:
      directions.add("up")
    elif self.axes[7] > 0.5:
      directions.add("down")
    return directions

  def command(
    self, scale: tuple[float, float, float] = (1.0, 1.0, 0.5)
  ) -> np.ndarray | None:
    """Planar velocity command from the sticks, dead-zoned at 5 %."""
    if self.fd is None:
      return None
    values = np.array([-self.axes[1], -self.axes[0], -self.axes[3]], dtype=np.float32)
    values[np.abs(values) < 0.05] = 0
    return values * np.asarray(scale, dtype=np.float32)

  def close(self) -> None:
    if self.fd is not None:
      os.close(self.fd)