"""Reset events used to bootstrap AMP training from reference motion states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .amp_data import RESET_FIELDS


class reset_from_amp_reference:
  """Reset a fraction of environments to random frames from AMP clips."""

  def __init__(self, cfg, env) -> None:
    path = Path(cfg.params["reference_path"])
    paths = sorted(path.rglob("*.npz"))
    if not paths:
      raise ValueError(f"No AMP reference clips found under {path}")

    fields: dict[str, list[np.ndarray]] = {name: [] for name in RESET_FIELDS}
    joint_names: tuple[str, ...] | None = None
    for file_path in paths:
      with np.load(file_path, allow_pickle=False) as clip:
        missing = set(fields) - set(clip.files)
        if missing:
          raise ValueError(f"{file_path}: missing fields {sorted(missing)}")
        names = tuple(str(name) for name in clip["joint_names"])
        if joint_names is None:
          joint_names = names
        elif names != joint_names:
          raise ValueError(f"{file_path}: inconsistent joint order")
        for name in fields:
          values = clip[name]
          if not np.isfinite(values).all():
            raise ValueError(f"{file_path}: non-finite values in {name}")
          fields[name].append(values.astype(np.float32))
        norms = np.linalg.norm(clip["root_quat_w"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
          raise ValueError(f"{file_path}: root quaternions are not normalized")

    robot = env.scene["robot"]
    if joint_names != tuple(robot.joint_names):
      raise ValueError("AMP reference joint order differs from the live robot")
    self._robot = robot
    self._env = env
    self._probability = float(cfg.params["probability"])
    if not 0.0 <= self._probability <= 1.0:
      raise ValueError("AMP reference reset probability must be in [0, 1]")
    self._frames = {
      name: torch.from_numpy(np.concatenate(values)).to(env.device)
      for name, values in fields.items()
    }

  def __call__(self, env, env_ids, **_kwargs) -> None:
    del env
    if isinstance(env_ids, slice):
      ids = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
    else:
      ids = env_ids
    selected = ids[torch.rand(len(ids), device=self._env.device) < self._probability]
    if len(selected) == 0:
      return

    frame_ids = torch.randint(
      len(self._frames["joint_pos"]),
      (len(selected),),
      device=self._env.device,
    )
    root_pos = self._env.scene.env_origins[selected].clone()
    root_pos[:, 2] += self._frames["root_pos_w"][frame_ids, 2]
    root_state = torch.cat(
      (
        root_pos,
        self._frames["root_quat_w"][frame_ids],
        self._frames["root_lin_vel_w"][frame_ids],
        self._frames["root_ang_vel_w"][frame_ids],
      ),
      dim=-1,
    )
    self._robot.write_root_state_to_sim(root_state, selected)
    # Clamp into the soft joint limits so a reference frame can never inject an
    # out-of-range pose into the solver.
    limits = self._robot.data.soft_joint_pos_limits[selected]
    joint_pos = self._frames["joint_pos"][frame_ids].clamp(
      limits[..., 0], limits[..., 1]
    )
    self._robot.write_joint_state_to_sim(
      joint_pos,
      self._frames["joint_vel"][frame_ids],
      env_ids=selected,
    )