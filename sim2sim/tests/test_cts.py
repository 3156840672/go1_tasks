"""CTS student deployment: ONNX loading, history layout and simulator wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from quadruped_sim.policies.actor import ACTION_DIM, OBS_DIM
from quadruped_sim.policies.cts import (
  CTS_HISTORY_LENGTH,
  CTS_LATENT_DIM,
  CTS_OBS_DIM,
  CTS_TERM_DIMS,
  CtsCheckpointStudent,
  CtsOnnxStudent,
  CtsStudentPolicy,
  load_cts_policy,
)
from quadruped_sim.robots.go1 import Go1Simulator
from quadruped_sim.robots.go2 import Go2Simulator

#: The real artifacts live under logs/rsl_rl/<task>/<run>/.
CTS_ROOT = Path(__file__).resolve().parents[2] / "logs" / "rsl_rl"
CTS_DIR = CTS_ROOT / "go2_cts" / "2026-10-31_10-43-13"
CTS_RUN = CTS_DIR / "2026-10-31_10-43-13.onnx"
CTS_CHECKPOINT = CTS_DIR / "model_9999.pt"
#: The Go1 run stopped at 6300, which is the iteration its ONNX was exported at.
GO1_DIR = CTS_ROOT / "go1_cts" / "2026-10-29_14-14-46"
GO1_RUN = GO1_DIR / "2026-10-29_14-14-46.onnx"
GO1_CHECKPOINT = GO1_DIR / "model_6300.pt"


def write_checkpoint(
  path: Path, proprio_dim: int = CTS_OBS_DIM, latent_dim: int = CTS_LATENT_DIM
) -> Path:
  """Write a CTS-shaped ``model_*.pt`` with deterministic, tiny weights."""
  generator = torch.Generator().manual_seed(0)

  def mlp(widths: tuple[int, ...], prefix: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index, (fan_in, fan_out) in enumerate(
      zip(widths[:-1], widths[1:], strict=True)
    ):
      key = f"{prefix}.{2 * index}"
      state[f"{key}.weight"] = 0.1 * torch.randn(
        fan_out, fan_in, generator=generator
      )
      state[f"{key}.bias"] = 0.1 * torch.randn(fan_out, generator=generator)
    return state

  history_dim = proprio_dim * CTS_HISTORY_LENGTH
  state = {
    **mlp(
      (history_dim, 512, 256, latent_dim), "student_encoder"
    ),
    **mlp((proprio_dim + latent_dim, 512, 256, 128, ACTION_DIM), "mlp"),
    "distribution.std_param": torch.zeros(ACTION_DIM),
  }
  torch.save({"actor_state_dict": state, "iter": 9999}, path)
  return path


def write_passthrough(path: Path, proprio_dim: int = CTS_OBS_DIM) -> Path:
  """Write a CTS-shaped ONNX whose action is the first 12 history values."""
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  history_dim = proprio_dim * CTS_HISTORY_LENGTH
  initializers = [
    numpy_helper.from_array(np.array([0], np.int64), "starts"),
    numpy_helper.from_array(np.array([ACTION_DIM], np.int64), "ends"),
    numpy_helper.from_array(np.array([1], np.int64), "axes"),
  ]
  graph = helper.make_graph(
    [helper.make_node("Slice", ["history", "starts", "ends", "axes"], ["actions"])],
    "cts_passthrough",
    [
      helper.make_tensor_value_info("proprio", TensorProto.FLOAT, [1, proprio_dim]),
      helper.make_tensor_value_info("history", TensorProto.FLOAT, [1, history_dim]),
    ],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
    initializer=initializers,
  )
  model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="sim2sim-test"
  )
  onnx.save(model, str(path))
  return path


class HistoryLayoutTests(unittest.TestCase):
  """mjlab keeps one ring buffer per actor term, so history is term-major."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = write_passthrough(Path(self.directory.name) / "student.onnx")
    self.policy = load_cts_policy(self.path)

  def frame(self, step: float) -> torch.Tensor:
    values = np.full(CTS_OBS_DIM, step, dtype=np.float32)
    values[0] = 100.0 + step
    return torch.from_numpy(values[None])

  def test_history_is_ordered_by_term_then_time(self):
    for step in (1.0, 2.0, 3.0):
      action = self.policy(self.frame(step))
    # Backfill makes the first push fill all five slots: [f1, f1, f1, f2, f3].
    expected = [
      101.0, 1.0, 1.0,
      101.0, 1.0, 1.0,
      101.0, 1.0, 1.0,
      102.0, 2.0, 2.0,
    ]
    self.assertEqual(action.shape, (1, ACTION_DIM))
    np.testing.assert_allclose(action[0].numpy(), expected)

  def test_reset_starts_a_fresh_buffer(self):
    for step in (1.0, 2.0):
      self.policy(self.frame(step))
    self.policy.reset()
    action = self.policy(self.frame(5.0))
    np.testing.assert_allclose(
      action[0].numpy(), [105.0, 5.0, 5.0] * 4
    )

  def test_term_widths_cover_the_actor_observation(self):
    self.assertEqual(sum(CTS_TERM_DIMS), CTS_OBS_DIM)
    self.assertNotEqual(CTS_OBS_DIM, OBS_DIM)


class LoaderTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)

  def test_missing_file_is_reported(self):
    with self.assertRaises(FileNotFoundError):
      load_cts_policy(Path(self.directory.name) / "absent.onnx")

  def test_unknown_suffix_is_rejected(self):
    path = Path(self.directory.name) / "model_9999.bin"
    path.write_bytes(b"")
    with self.assertRaisesRegex(ValueError, "neither a CTS ONNX"):
      load_cts_policy(path)

  def test_checkpoint_without_an_actor_is_rejected(self):
    path = Path(self.directory.name) / "model_9999.pt"
    torch.save({"iter": 1}, path)
    with self.assertRaisesRegex(ValueError, "no actor_state_dict"):
      load_cts_policy(path)

  def test_checkpoint_without_the_student_encoder_is_rejected(self):
    path = write_checkpoint(Path(self.directory.name) / "head_only.pt")
    state = torch.load(path, weights_only=True)
    state["actor_state_dict"] = {
      key: value
      for key, value in state["actor_state_dict"].items()
      if not key.startswith("student_encoder")
    }
    torch.save(state, path)
    with self.assertRaisesRegex(ValueError, "student_encoder"):
      load_cts_policy(path)

  def test_checkpoint_with_a_48d_layout_is_rejected(self):
    path = write_checkpoint(
      Path(self.directory.name) / "wide_head.pt", proprio_dim=48
    )
    with self.assertRaisesRegex(ValueError, "45-D frames"):
      load_cts_policy(path)

  def test_wrong_proprio_width_is_rejected(self):
    path = write_passthrough(Path(self.directory.name) / "wide.onnx", 48)
    with self.assertRaisesRegex(ValueError, "blind 45-D layout"):
      load_cts_policy(path)

  def test_unrelated_graph_is_rejected(self):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
      [helper.make_node("Relu", ["proprio"], ["actions"])],
      "unrelated",
      [helper.make_tensor_value_info("proprio", TensorProto.FLOAT, [1, 45])],
      [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
    )
    path = Path(self.directory.name) / "unrelated.onnx"
    onnx.save(helper.make_model(graph), str(path))
    with self.assertRaisesRegex(ValueError, "not a CTS student export"):
      load_cts_policy(path)


class CheckpointLoaderTests(unittest.TestCase):
  """model_*.pt holds student_encoder + mlp, so it can run without ONNX."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = write_checkpoint(Path(self.directory.name) / "model_9999.pt")
    self.policy = load_cts_policy(self.path)

  def test_checkpoint_builds_the_student_branch(self):
    self.assertIsInstance(self.policy, CtsCheckpointStudent)
    self.assertIsInstance(self.policy, CtsStudentPolicy)
    self.assertEqual(self.policy.frames, CTS_HISTORY_LENGTH)
    self.assertEqual(self.policy.obs_dim, CTS_OBS_DIM)

  def test_history_layout_matches_the_onnx_path(self):
    onnx_policy = load_cts_policy(
      write_passthrough(Path(self.directory.name) / "passthrough.onnx")
    )
    for step in (1.0, 2.0, 3.0):
      frame = torch.full((1, CTS_OBS_DIM), step)
      self.policy(frame)
      onnx_policy(frame)
    np.testing.assert_allclose(
      self.policy._history, onnx_policy._history, rtol=0, atol=0
    )

  def test_output_is_finite_and_the_right_shape(self):
    action = self.policy(torch.zeros(1, CTS_OBS_DIM))
    self.assertEqual(action.shape, (1, ACTION_DIM))
    self.assertTrue(np.isfinite(action.numpy()).all())


class SimulatorWiringTests(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = write_passthrough(Path(self.directory.name) / "student.onnx")

  def test_go2_feeds_the_blind_45d_observation(self):
    sim = Go2Simulator(cts=self.path)
    self.assertIsInstance(sim.policy, CtsStudentPolicy)
    self.assertEqual(sim.observation().shape, (CTS_OBS_DIM,))
    sim.set_active(True)
    for _ in range(5):
      sim.policy_step()
      sim.physics_step()
    self.assertTrue(np.isfinite(sim.target).all())

  def test_go1_feeds_the_blind_45d_observation(self):
    sim = Go1Simulator(None, None, cts=self.path)
    self.assertEqual(sim.observation().shape, (CTS_OBS_DIM,))
    sim.set_active(True)
    for _ in range(5):
      sim.policy_step()
      sim.physics_step()
    self.assertTrue(np.isfinite(sim.target).all())

  def test_go2_accepts_a_plain_checkpoint(self):
    checkpoint = write_checkpoint(Path(self.directory.name) / "model_9999.pt")
    sim = Go2Simulator(cts=checkpoint)
    self.assertIsInstance(sim.policy, CtsCheckpointStudent)
    sim.set_active(True)
    for _ in range(5):
      sim.policy_step()
      sim.physics_step()
    self.assertTrue(np.isfinite(sim.target).all())

  def test_only_one_policy_source_is_allowed(self):
    with self.assertRaises(ValueError):
      Go2Simulator(policy=Path("model.pt"), cts=self.path)
    with self.assertRaises(ValueError):
      Go1Simulator(None, None, policy=Path("model.pt"), cts=self.path)


@unittest.skipUnless(CTS_RUN.is_file(), "no exported CTS run in logs/")
class ExportedRunTests(unittest.TestCase):
  def test_pt_checkpoint_matches_the_onnx_export(self):
    for onnx_path, checkpoint_path in (
      (CTS_RUN, CTS_CHECKPOINT),
      (GO1_RUN, GO1_CHECKPOINT),
    ):
      if not onnx_path.is_file() or not checkpoint_path.is_file():
        continue
      with self.subTest(run=onnx_path.parent.name):
        self.assert_matches(onnx_path, checkpoint_path)

  def assert_matches(self, onnx_path: Path, checkpoint_path: Path) -> None:
    onnx_policy = load_cts_policy(onnx_path)
    checkpoint_policy = load_cts_policy(checkpoint_path)
    self.assertIsInstance(onnx_policy, CtsOnnxStudent)
    self.assertIsInstance(checkpoint_policy, CtsCheckpointStudent)
    self.assertEqual(onnx_policy.frames, checkpoint_policy.frames)
    generator = np.random.default_rng(0)
    differences = []
    for _ in range(200):
      obs = torch.from_numpy(
        generator.normal(size=(1, CTS_OBS_DIM)).astype(np.float32)
      )
      difference = (onnx_policy(obs) - checkpoint_policy(obs)).abs().max()
      differences.append(float(difference))
    self.assertLess(max(differences), 1e-4)

  def test_real_student_tracks_a_forward_command(self):
    sim = Go2Simulator(
      cts=CTS_RUN,
      terrain="go2_rough",
      terrain_type="all",
      terrain_level=0,
      terrain_column="flat",
    )
    sim.command[:] = (0.5, 0.0, 0.0)
    sim.set_active(True)
    for _ in range(400):
      sim.policy_step()
      for _ in range(4):
        sim.physics_step()
    vx = float(np.mean(sim.base_velocity()[0][0]))
    self.assertGreater(vx, 0.3)


if __name__ == "__main__":
  unittest.main()
