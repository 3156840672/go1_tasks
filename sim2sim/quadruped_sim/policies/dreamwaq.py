"""Run a Go2 DreamWaQ checkpoint or ONNX policy in independent MuJoCo."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .actor import ACTION_DIM
from .cts import CTS_TERM_DIMS, NORMALIZER_EPS, Go2HistoryPolicy, _mlp_from_state


class DreamWaQPolicy(Go2HistoryPolicy):
  def __init__(self, action_clip: float | None) -> None:
    super().__init__(frames=5)
    if action_clip is not None and (
      not math.isfinite(action_clip) or action_clip <= 0
    ):
      raise ValueError(f"Invalid DreamWaQ action clip: {action_clip}")
    self.action_clip = action_clip

  def _act(self, proprio: np.ndarray, history: np.ndarray) -> np.ndarray:
    action = self._predict(history)
    if action.shape != (1, ACTION_DIM):
      raise ValueError(f"Unexpected DreamWaQ action shape: {action.shape}")
    if not np.isfinite(action).all():
      raise FloatingPointError("DreamWaQ returned non-finite actions")
    if self.action_clip is not None:
      action = np.clip(action, -self.action_clip, self.action_clip)
    return action

  def _predict(self, history: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class DreamWaQOnnxPolicy(DreamWaQPolicy):
  def __init__(self, path: Path) -> None:
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(str(path))
    metadata = {item.key: item.value for item in model.metadata_props}
    clip = metadata.get("action_clip", "100")  # Default for earlier exports.
    super().__init__(None if clip == "none" else float(clip))
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {"obs_history"} or set(outputs) != {"actions"}:
      raise ValueError(f"Expected a Go2 DreamWaQ ONNX export: {path}")
    dimensions = inputs["obs_history"].type.tensor_type.shape.dim
    if len(dimensions) != 2 or dimensions[1].dim_value != self.frames * self.obs_dim:
      raise ValueError("DreamWaQ ONNX input must have shape [batch, 225]")
    self.evaluator = ReferenceEvaluator(model)

  def _predict(self, history: np.ndarray) -> np.ndarray:
    action, = self.evaluator.run(["actions"], {"obs_history": history})
    return action


class DreamWaQCheckpointPolicy(DreamWaQPolicy):
  def __init__(self, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("actor_state_dict", {})
    if "cenet.velocity.weight" not in state:
      raise ValueError(f"Expected a Go2 DreamWaQ checkpoint: {path}")
    super().__init__((checkpoint.get("infos") or {}).get("action_clip", 100.0))
    self.encoder, history_dim, hidden_dim = _mlp_from_state(state, "cenet.encoder")
    self.policy, policy_in, policy_out = _mlp_from_state(state, "mlp")
    if history_dim != self.frames * self.obs_dim or (
      policy_in, policy_out
    ) != (64, ACTION_DIM):
      raise ValueError(f"Unexpected DreamWaQ checkpoint dimensions: {path}")
    self.velocity = torch.nn.Linear(hidden_dim, 3)
    self.context = torch.nn.Linear(hidden_dim, 16)
    for name, layer in (("velocity", self.velocity), ("mean", self.context)):
      layer.load_state_dict(
        {key: state[f"cenet.{name}.{key}"] for key in ("weight", "bias")}
      )
    self.mean = state["frame_norm._mean"]
    self.std = state["frame_norm._std"]

  @torch.inference_mode()
  def _predict(self, history: np.ndarray) -> np.ndarray:
    widths = [self.frames * n for n in CTS_TERM_DIMS]
    parts = torch.from_numpy(history).split(widths, dim=-1)
    frames = torch.cat(
      [part.reshape(1, self.frames, n) for part, n in zip(parts, CTS_TERM_DIMS)], dim=-1
    )
    frames = (frames - self.mean) / (self.std + NORMALIZER_EPS)
    hidden = F.elu(self.encoder(frames.flatten(1)))
    features = torch.cat((frames[:, -1], self.velocity(hidden), self.context(hidden)), dim=-1)
    return self.policy(features).numpy()


def load_dreamwaq_policy(path: Path) -> DreamWaQPolicy:
  if path.suffix.lower() == ".pt":
    return DreamWaQCheckpointPolicy(path)
  if path.suffix.lower() == ".onnx":
    return DreamWaQOnnxPolicy(path)
  raise ValueError(f"DreamWaQ expects a model_*.pt checkpoint or .onnx export: {path}")
