"""Checks for the two-stage Go1 RMA implementation."""

import pytest
import torch
from tensordict import TensorDict

from go1_tasks.rma import (
  HISTORY_LENGTH,
  LATENT_DIM,
  PRIVILEGED_DIM,
  AdaptationModule,
  RmaActor,
)
from go1_tasks.rma_cfg import unitree_go1_rma_env_cfg, unitree_go1_rma_runner_cfg


def test_rma_environment_and_runner_configuration() -> None:
  env_cfg = unitree_go1_rma_env_cfg()
  assert "height_scan" not in env_cfg.observations["actor"].terms
  assert tuple(env_cfg.observations["privileged"].terms) == ("factors",)
  assert env_cfg.observations["privileged"].enable_corruption is False
  assert env_cfg.events["rma_payload"].params["ranges"] == (0.0, 2.0)
  assert env_cfg.events["rma_motor_gains"].params["kp_range"] == (0.9, 1.1)

  runner_cfg = unitree_go1_rma_runner_cfg()
  assert runner_cfg.actor.class_name == "go1_tasks.rma:RmaActor"
  assert runner_cfg.actor.hidden_dims == (128, 128, 128)
  assert runner_cfg.obs_groups["actor"] == ("actor", "privileged")
  assert runner_cfg.experiment_name == "go1_rma"
  assert runner_cfg.logger == "tensorboard"


def test_rma_network_contract_matches_sim2sim() -> None:
  obs_dim = 48
  adapter = AdaptationModule(obs_dim)
  latent = adapter(torch.zeros(2, HISTORY_LENGTH, obs_dim))
  assert latent.shape == (2, LATENT_DIM)
  with pytest.raises(ValueError, match="Expected history"):
    adapter(torch.zeros(2, HISTORY_LENGTH - 1, obs_dim))

  observations = TensorDict(
    {
      "actor": torch.zeros(2, obs_dim),
      "privileged": torch.zeros(2, PRIVILEGED_DIM),
    },
    batch_size=[2],
  )
  actor = RmaActor(
    observations,
    {"actor": ["actor", "privileged"]},
    "actor",
    12,
    hidden_dims=(128, 128, 128),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  assert actor(observations).shape == (2, 12)
  assert actor.act_with_latent(observations["actor"], latent).shape == (2, 12)
