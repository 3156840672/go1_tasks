"""Unitree Go1 velocity tasks for mjlab.

Importing this package registers the Go1 tasks in mjlab's task registry::

    import go1_tasks

Each task is registered under both a package-owned ID (``Go1-Velocity-*``) and
the historical ``Mjlab-Velocity-*-Unitree-Go1`` name. The historical ID is
skipped with a warning if another package has already registered it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import mjlab.tasks  # noqa: F401  Register bundled tasks before this package.
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go1_cts_env_cfg,
  unitree_go1_flat_env_cfg,
  unitree_go1_mob_env_cfg,
  unitree_go1_rough_env_cfg,
)
from .mob_runner import MobOnPolicyRunner
from .rl_cfg import CtsPpoAlgorithmCfg, unitree_go1_ppo_runner_cfg


def mob_rl_cfg() -> RslRlBaseRunnerCfg:
  """RL configuration for the behavior-conditioned Go1 task."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "go1_mob"
  return cfg


def cts_rl_cfg() -> RslRlBaseRunnerCfg:
  """RL configuration for the concurrent teacher-student Go1 task."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = "go1_cts"
  cfg.save_interval = 100
  cfg.actor.class_name = "go1_tasks.cts.CTSActor"
  cfg.algorithm = CtsPpoAlgorithmCfg(**asdict(cfg.algorithm))
  cfg.algorithm.class_name = "go1_tasks.cts.CTSPPO"
  return cfg


@dataclass(frozen=True)
class TaskSpec:
  """A Go1 task and the IDs it can be registered under."""

  key: str
  ids: tuple[str, ...]
  env_cfg: Callable[..., ManagerBasedRlEnvCfg]
  rl_cfg: Callable[[], RslRlBaseRunnerCfg]
  runner_cls: type | None


LEGACY_PREFIX = "Mjlab-Velocity"
"""Historical task IDs originally shipped inside the mjlab source tree."""

_SPECS: tuple[TaskSpec, ...] = (
  TaskSpec(
    key="rough",
    ids=("Mjlab-Velocity-Rough-Unitree-Go1", "Go1-Velocity-Rough"),
    env_cfg=unitree_go1_rough_env_cfg,
    rl_cfg=unitree_go1_ppo_runner_cfg,
    runner_cls=VelocityOnPolicyRunner,
  ),
  TaskSpec(
    key="flat",
    ids=("Mjlab-Velocity-Flat-Unitree-Go1", "Go1-Velocity-Flat"),
    env_cfg=unitree_go1_flat_env_cfg,
    rl_cfg=unitree_go1_ppo_runner_cfg,
    runner_cls=VelocityOnPolicyRunner,
  ),
  TaskSpec(
    key="mob",
    ids=("Mjlab-Velocity-MoB-Unitree-Go1", "Go1-Velocity-MoB"),
    env_cfg=unitree_go1_mob_env_cfg,
    rl_cfg=mob_rl_cfg,
    runner_cls=MobOnPolicyRunner,
  ),
  TaskSpec(
    key="cts",
    ids=("Mjlab-Velocity-CTS-Unitree-Go1", "Go1-Velocity-CTS"),
    env_cfg=unitree_go1_cts_env_cfg,
    rl_cfg=cts_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
  ),
)

#: Package-owned task IDs. These never collide with mjlab's bundled tasks.
TASK_IDS: dict[str, str] = {spec.key: spec.ids[-1] for spec in _SPECS}

#: Task ID this package actually registered for each key.
_REGISTERED: dict[str, str] = {}

__all__ = [
  "TASK_IDS",
  "TaskSpec",
  "cts_rl_cfg",
  "mob_rl_cfg",
  "register_tasks",
  "task_id",
]


def register_tasks(*, warn: bool = True) -> list[str]:
  """Register every Go1 task that is not already present.

  Returns the task IDs registered by this call. IDs owned by another package
  (for example mjlab's bundled Go1 tasks) are left untouched.
  """
  registered: list[str] = []
  taken = set(list_tasks())
  shadowed: list[str] = []
  for spec in _SPECS:
    for index, task_id_ in enumerate(spec.ids):
      if task_id_ in taken:
        if index == 0:
          shadowed.append(task_id_)
        continue
      register_mjlab_task(
        task_id=task_id_,
        env_cfg=spec.env_cfg(),
        play_env_cfg=spec.env_cfg(play=True),
        rl_cfg=spec.rl_cfg(),
        runner_cls=spec.runner_cls,
      )
      taken.add(task_id_)
      registered.append(task_id_)
      _REGISTERED.setdefault(spec.key, task_id_)
  if warn and shadowed:
    warnings.warn(
      "Task IDs already registered by mjlab's bundled Go1 tasks: "
      + ", ".join(shadowed)
      + ". Use the package-owned Go1-Velocity-* IDs instead.",
      UserWarning,
      stacklevel=2,
    )
  return registered


def task_id(key: str) -> str:
  """Return the task ID this package registered for ``key``.

  When mjlab's bundled Go1 tasks occupy the historical IDs, this is the
  package-owned ``Go1-Velocity-*`` ID; otherwise it is the historical ID.
  """
  if key not in TASK_IDS:
    raise KeyError(f"Unknown Go1 task {key!r}; expected one of {sorted(TASK_IDS)}")
  return _REGISTERED.get(key, TASK_IDS[key])


register_tasks()
