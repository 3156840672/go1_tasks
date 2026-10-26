"""Go1 RMA configuration derived from the rough-terrain velocity task."""

from mjlab.envs import mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .env_cfgs import unitree_go1_rough_env_cfg
from .rl_cfg import unitree_go1_ppo_runner_cfg
from .rma import privileged_factors


def unitree_go1_rma_env_cfg():
  """Build the RMA environment with proprioceptive and privileged groups."""
  cfg = unitree_go1_rough_env_cfg()
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["privileged"] = ObservationGroupCfg(
    terms={"factors": ObservationTermCfg(func=privileged_factors)},
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.events["rma_payload"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "operation": "add",
      "ranges": (0.0, 2.0),
    },
  )
  cfg.events["rma_motor_gains"] = EventTermCfg(
    mode="startup",
    func=mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  return cfg


def unitree_go1_rma_runner_cfg():
  """Build the stage-one RMA teacher runner configuration."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.actor.class_name = "go1_tasks.rma:RmaActor"
  cfg.actor.hidden_dims = (128, 128, 128)
  cfg.algorithm.learning_rate = 5e-4
  cfg.obs_groups = {
    "actor": ("actor", "privileged"),
    "critic": ("critic", "privileged"),
  }
  cfg.experiment_name = "go1_rma"
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg
