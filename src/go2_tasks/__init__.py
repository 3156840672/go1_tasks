"""Unitree Go2 locomotion tasks for mjlab."""

from __future__ import annotations

from dataclasses import asdict

from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from go1_tasks.mob_runner import MobOnPolicyRunner
from go1_tasks.rl_cfg import (
  AmpPpoAlgorithmCfg,
  CtsPpoAlgorithmCfg,
  unitree_go1_ppo_runner_cfg,
)

from .env_cfgs import (
  GO2_AMP_REFERENCE_PATH,
  unitree_go2_amp_env_cfg,
  unitree_go2_cts_env_cfg,
  unitree_go2_flat_env_cfg,
  unitree_go2_mob_env_cfg,
  unitree_go2_rough_env_cfg,
)


def _rl_cfg(kind: str) -> RslRlBaseRunnerCfg:
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.experiment_name = f"go2_{kind}"
  if kind == "cts":
    cfg.save_interval = 100
    cfg.actor.class_name = "go1_tasks.cts.CTSActor"
    cfg.algorithm = CtsPpoAlgorithmCfg(**asdict(cfg.algorithm))
    cfg.algorithm.class_name = "go1_tasks.cts.CTSPPO"
  elif kind == "amp":
    cfg.algorithm = AmpPpoAlgorithmCfg(**asdict(cfg.algorithm))
    cfg.algorithm.reference_path = GO2_AMP_REFERENCE_PATH
    cfg.algorithm.class_name = "go1_tasks.amp.AmpPPO"
  return cfg


_SPECS = {
  "rough": (unitree_go2_rough_env_cfg, VelocityOnPolicyRunner),
  "flat": (unitree_go2_flat_env_cfg, VelocityOnPolicyRunner),
  "mob": (unitree_go2_mob_env_cfg, MobOnPolicyRunner),
  "amp": (unitree_go2_amp_env_cfg, VelocityOnPolicyRunner),
  "cts": (unitree_go2_cts_env_cfg, VelocityOnPolicyRunner),
}

TASK_IDS = {
  "rough": "Go2-Velocity-Rough",
  "flat": "Go2-Velocity-Flat",
  "mob": "Go2-Velocity-MoB",
  "amp": "Go2-Velocity-AMP",
  "cts": "Go2-Velocity-CTS",
}


def register_tasks() -> list[str]:
  """Register every Go2 task not already present and return new task IDs."""
  taken = set(list_tasks())
  registered: list[str] = []
  for key, (env_cfg, runner_cls) in _SPECS.items():
    task_id = TASK_IDS[key]
    if task_id in taken:
      continue
    register_mjlab_task(
      task_id=task_id,
      env_cfg=env_cfg(),
      play_env_cfg=env_cfg(play=True),
      rl_cfg=_rl_cfg(key),
      runner_cls=runner_cls,
    )
    registered.append(task_id)
    taken.add(task_id)
  return registered


def task_id(task: str) -> str:
  try:
    return TASK_IDS[task.lower()]
  except KeyError as exc:
    raise KeyError(f"Unknown Go2 task: {task}") from exc


register_tasks()

__all__ = ["TASK_IDS", "register_tasks", "task_id"]
