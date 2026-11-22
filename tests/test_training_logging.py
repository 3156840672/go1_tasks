"""Smoke test for the training logger dependency combination."""

from rsl_rl.utils.wandb_log_writer import WandbLogWriter


def test_wandb_logger_initializes_offline(monkeypatch, tmp_path) -> None:
  """RSL-RL must be able to start its configured W&B writer."""
  monkeypatch.setenv("WANDB_MODE", "disabled")
  writer = WandbLogWriter(log_dir=str(tmp_path), project_name="mjlab")
  try:
    assert writer.log_dir == str(tmp_path)
  finally:
    writer.stop()
    writer.close()
