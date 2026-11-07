"""Unitree Go2 locomotion tasks for mjlab."""

from __future__ import annotations

from dataclasses import asdict

from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from go1_tasks.mob_runner import MobOnPolicyRunner
from go1_tasks.rl_cfg import (
  AmpPpoAlgorithmCfg,
  HimModelCfg,
  HimPpoAlgorithmCfg,
  PaperAmpModelCfg,
  CtsPpoAlgorithmCfg,
  unitree_go1_ppo_runner_cfg,
)

from .env_cfgs import (
  GO2_AMP_REFERENCE_PATH,
  GO2_PAPER_AMP_REFERENCE_PATH,
  unitree_go2_amp_env_cfg,
  unitree_go2_cts_env_cfg,
  unitree_go2_flat_env_cfg,
  unitree_go2_him_env_cfg,
  unitree_go2_mob_env_cfg,
  unitree_go2_paper_amp_env_cfg,
  unitree_go2_paper_amp_baseline_env_cfg,
  unitree_go2_rough_amp_env_cfg,
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
  elif kind == "him":
    # HIMLoco: deployable actor reads only a 6-step proprio history; the
    # estimator is trained inside HIMPPO against the privileged critic obs.
    cfg.actor = HimModelCfg(**asdict(cfg.actor))
    cfg.actor.class_name = "go2_tasks.him.HIMActor"
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.algorithm = HimPpoAlgorithmCfg(**asdict(cfg.algorithm))
    cfg.algorithm.class_name = "go2_tasks.him.HIMPPO"
    cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",)}
    # HIMLoco clips actions at 100 in the environment; without it a single
    # oversized action is stored in the next observation and feeds back into
    # the history encoder, which is exactly how the first run diverged.
    cfg.clip_actions = 100.0
  elif kind in ("amp", "rough_amp", "paper_amp_teacher"):
    cfg.algorithm = AmpPpoAlgorithmCfg(**asdict(cfg.algorithm))
    cfg.algorithm.reference_path = GO2_AMP_REFERENCE_PATH
    cfg.algorithm.class_name = "go1_tasks.amp.AmpPPO"
    if kind == "paper_amp_teacher":
      cfg.actor = PaperAmpModelCfg(**asdict(cfg.actor))
      cfg.actor.class_name = "go2_tasks.paper_amp.PaperAmpTeacherActor"
      cfg.actor.hidden_dims = (256, 128, 64)
      # Match Table III of Wu et al. for the paper-style teacher.  This task
      # uses 30 seconds of directional Trot plus a short standing reference.
      cfg.algorithm.reference_path = GO2_PAPER_AMP_REFERENCE_PATH
      cfg.algorithm.discriminator_hidden_dims = (1024, 512)
      cfg.algorithm.discriminator_activation = "elu"
      cfg.algorithm.discriminator_state_mode = "paper"
      # Same discriminator regularization split used by the complete AMP
      # reference implementation: light decay on features, strong decay on
      # the scalar classification head. The network dimensions remain exactly
      # those from Table III of the paper.
      cfg.algorithm.amp_reward_weight = 2.0
      cfg.algorithm.discriminator_loss_margin = 0.0
      cfg.algorithm.discriminator_batches_per_iteration = 2
      cfg.algorithm.discriminator_trunk_weight_decay = 1e-3
      cfg.algorithm.discriminator_head_weight_decay = 1e-1
      # Standing references now cover strict zero commands. Include these
      # transitions in style rewards and replay so the discriminator can teach
      # stationary motion instead of merely ignoring in-place stepping.
      cfg.algorithm.zero_command_amp_threshold = 0.0
      cfg.obs_groups = {
        "actor": ("actor", "privileged", "terrain"),
        "critic": ("critic",),
      }
  return cfg


def unitree_go2_paper_amp_runner_cfg() -> RslRlBaseRunnerCfg:
  """Public runner config used by teacher training and student distillation."""
  return _rl_cfg("paper_amp_teacher")


_SPECS = {
  "rough": (unitree_go2_rough_env_cfg, VelocityOnPolicyRunner),
  "flat": (unitree_go2_flat_env_cfg, VelocityOnPolicyRunner),
  "mob": (unitree_go2_mob_env_cfg, MobOnPolicyRunner),
  "amp": (unitree_go2_amp_env_cfg, VelocityOnPolicyRunner),
  "rough_amp": (unitree_go2_rough_amp_env_cfg, VelocityOnPolicyRunner),
  "paper_amp_teacher": (unitree_go2_paper_amp_baseline_env_cfg, VelocityOnPolicyRunner),
  "cts": (unitree_go2_cts_env_cfg, VelocityOnPolicyRunner),
  "him": (unitree_go2_him_env_cfg, VelocityOnPolicyRunner),
}

TASK_IDS = {
  "rough": "Go2-Velocity-Rough",
  "flat": "Go2-Velocity-Flat",
  "mob": "Go2-Velocity-MoB",
  "amp": "Go2-Velocity-AMP",
  "rough_amp": "Go2-Velocity-Rough-AMP",
  "paper_amp_teacher": "Go2-Velocity-Rough-AMP-Teacher",
  "cts": "Go2-Velocity-CTS",
  "him": "Go2-Velocity-HIM",
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

__all__ = [
  "TASK_IDS",
  "register_tasks",
  "task_id",
  "unitree_go2_paper_amp_runner_cfg",
]
