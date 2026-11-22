"""CTS (concurrent teacher-student) student policy for deployment.

CTS trains a shared policy head on ``[proprio, latent]``: the 45-D blind actor
observation concatenated with a 32-D latent that comes either from the
privileged teacher encoder (259-D critic observation) or from the student
encoder (``cts_history``, 5 frames of 45 D).  Only the student path is
deployable, and both artifacts of a run can run it:

* ``<run>.onnx`` - the graph the trainer exports next to the checkpoint; it
  bundles the student encoder, the L2 normalisation and the policy head, so it
  is what a real robot should load.
* ``model_*.pt`` - ``actor_state_dict`` holds ``student_encoder.*``,
  ``mlp.*`` and ``distribution.std_param``, which is enough to rebuild the same
  computation in NumPy/Torch.  Verified equal to the ONNX export to ~1e-6 on
  every exported run in ``logs/``, and it works for intermediate checkpoints
  that predate the export.

The two differences from the training graph that a checkpoint cannot reveal
are the activation (ELU everywhere in every CTS task in this repo) and the
observation normaliser (absent here; when a run normalises, ``obs_normalizer.*``
is applied as rsl-rl's ``EmpiricalNormalization`` does).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .actor import ACTION_DIM

#: Blind CTS actor: the rough actor without base_lin_vel and height_scan.
CTS_OBS_DIM = 45
#: ``cts_history`` is the actor group with ``history_length = 5``.
CTS_HISTORY_LENGTH = 5
#: Actor term widths in concatenation order (ang, gravity, q, dq, action, cmd).
#: mjlab keeps one ring buffer *per term*, so the flattened history is
#: term-major: five frames of the first term, then five of the second, ...
CTS_TERM_DIMS = (3, 3, 12, 12, 12, 3)
#: Width of the latent the encoders produce and the head consumes.
CTS_LATENT_DIM = 32
#: ``rsl_rl.modules.EmpiricalNormalization`` divides by ``std + eps``.
NORMALIZER_EPS = 0.01


def _dims(value_info) -> tuple[int, ...]:
  return tuple(
    dim.dim_value for dim in value_info.type.tensor_type.shape.dim
  )


class Go2HistoryPolicy:
  """Feed a Go2 policy the same term-major history used in mjlab."""

  obs_dim = CTS_OBS_DIM

  def __init__(self, frames: int = CTS_HISTORY_LENGTH) -> None:
    if sum(CTS_TERM_DIMS) != self.obs_dim:
      raise ValueError(
        f"Go2 term layout covers {sum(CTS_TERM_DIMS)} of {self.obs_dim} values"
      )
    self.frames = frames
    self._history = np.zeros((frames, self.obs_dim), dtype=np.float32)
    self._fresh = True
    offsets = np.cumsum((0, *CTS_TERM_DIMS))
    self._term_slices = [
      slice(start, stop) for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]

  def reset(self) -> None:
    self._history.fill(0)
    self._fresh = True

  def _push(self, frame: np.ndarray) -> None:
    if self._fresh:
      # mjlab backfills a fresh circular buffer with its first frame.
      self._history[:] = frame
      self._fresh = False
    else:
      self._history[:-1] = self._history[1:]
      self._history[-1] = frame

  def _act(self, proprio: np.ndarray, history: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    proprio = obs.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
    if proprio.shape[1] != self.obs_dim:
      raise ValueError(
        f"Go2 policy expects {self.obs_dim} observations, got "
        f"{proprio.shape[1]}"
      )
    self._push(proprio[0])
    history = np.concatenate(
      [self._history[:, term].reshape(-1) for term in self._term_slices]
    )
    action = np.array(
      self._act(proprio, history.reshape(1, -1)), dtype=np.float32
    )
    return torch.from_numpy(action)


class CtsStudentPolicy(Go2HistoryPolicy):
  """Base type shared by the CTS deployment formats."""


class CtsOnnxStudent(CtsStudentPolicy):
  """The ``<run>.onnx`` graph the trainer exports next to the checkpoint."""

  def __init__(self, path: Path) -> None:
    try:
      import onnx
      from onnx.reference import ReferenceEvaluator
    except ImportError as error:  # pragma: no cover - mjlab already needs it
      raise SystemExit(
        "reading a CTS '.onnx' export needs the 'onnx' package "
        "(uv sync pulls it in through mjlab); 'model_*.pt' works without it"
      ) from error
    model = onnx.load(str(path))
    inputs = {item.name: _dims(item) for item in model.graph.input}
    outputs = {item.name: _dims(item) for item in model.graph.output}
    if set(inputs) != {"proprio", "history"} or set(outputs) != {"actions"}:
      raise ValueError(
        f"{path.name} is not a CTS student export: inputs={sorted(inputs)}, "
        f"outputs={sorted(outputs)}"
      )
    proprio_dim = inputs["proprio"][-1]
    history_dim = inputs["history"][-1]
    action_dim = outputs["actions"][-1]
    if proprio_dim != CTS_OBS_DIM:
      raise ValueError(
        f"{path.name} expects {proprio_dim} proprio observations; CTS actors "
        f"use the blind {CTS_OBS_DIM}-D layout"
      )
    if history_dim % proprio_dim != 0:
      raise ValueError(
        f"{path.name} feeds {history_dim} history values, which is not a whole "
        f"number of {proprio_dim}-D frames"
      )
    if action_dim != ACTION_DIM:
      raise ValueError(
        f"{path.name} outputs {action_dim} actions, expected {ACTION_DIM}"
      )
    super().__init__(frames=history_dim // proprio_dim)
    self._session = ReferenceEvaluator(model)

  def _act(self, proprio: np.ndarray, history: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
      # The exported ELU overflows expm1 on large negative inputs; the result
      # is a saturated activation, not a NaN.
      (action,) = self._session.run(
        ["actions"], {"proprio": proprio, "history": history}
      )
    return action


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
  checkpoint = torch.load(path, map_location="cpu", weights_only=True)
  state = checkpoint.get("actor_state_dict")
  if state is None:
    raise ValueError(
      f"{path.name} has no actor_state_dict; expected a model_*.pt written by "
      "the velocity runner"
    )
  return state


def _mlp_from_state(
  state: dict[str, torch.Tensor], prefix: str
) -> tuple[torch.nn.Sequential, int, int]:
  """Rebuild ``MLP(in, out, hidden, 'elu')`` from ``<prefix>.*`` weights."""
  indices = sorted(
    {
      int(key[len(prefix) + 1 :].split(".")[0])
      for key in state
      if key.startswith(f"{prefix}.") and key.endswith(".weight")
    }
  )
  if not indices:
    raise ValueError(f"checkpoint has no '{prefix}.*' weights")
  layers: list[torch.nn.Module] = []
  for position, index in enumerate(indices):
    weight = state[f"{prefix}.{index}.weight"]
    bias = state[f"{prefix}.{index}.bias"]
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
    linear.weight.data = weight.clone()
    linear.bias.data = bias.clone()
    layers.append(linear)
    if position != len(indices) - 1:
      layers.append(torch.nn.ELU())
  module = torch.nn.Sequential(*layers).eval()
  first = state[f"{prefix}.{indices[0]}.weight"]
  last = state[f"{prefix}.{indices[-1]}.weight"]
  return module, int(first.shape[1]), int(last.shape[0])


class CtsCheckpointStudent(CtsStudentPolicy):
  """The same student, rebuilt from ``model_*.pt`` (no ONNX needed)."""

  def __init__(self, path: Path) -> None:
    state = _load_state_dict(path)
    encoder, history_dim, latent_dim = _mlp_from_state(state, "student_encoder")
    head, head_in, head_out = _mlp_from_state(state, "mlp")
    if history_dim % CTS_OBS_DIM != 0:
      raise ValueError(
        f"{path.name} feeds {history_dim} history values, which is not a whole "
        f"number of {CTS_OBS_DIM}-D frames"
      )
    if latent_dim != CTS_LATENT_DIM:
      raise ValueError(
        f"{path.name} encodes a {latent_dim}-D latent, expected "
        f"{CTS_LATENT_DIM}"
      )
    if head_in != CTS_OBS_DIM + latent_dim:
      raise ValueError(
        f"{path.name} has a {head_in}-D policy head; a CTS head takes "
        f"{CTS_OBS_DIM} proprio + {latent_dim} latent values"
      )
    if head_out != ACTION_DIM:
      raise ValueError(
        f"{path.name} outputs {head_out} actions, expected {ACTION_DIM}"
      )
    super().__init__(frames=history_dim // CTS_OBS_DIM)
    self._encoder = encoder
    self._head = head
    self._mean = state.get("obs_normalizer._mean")
    self._std = state.get("obs_normalizer._std")
    if self._std is None and "obs_normalizer._var" in state:
      self._std = torch.sqrt(state["obs_normalizer._var"])

  def _act(self, proprio: np.ndarray, history: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      proprio_t = torch.from_numpy(proprio)
      if self._mean is not None and self._std is not None:
        proprio_t = (proprio_t - self._mean) / (self._std + NORMALIZER_EPS)
      latent = torch.nn.functional.normalize(
        self._encoder(torch.from_numpy(history)), dim=-1
      )
      return self._head(torch.cat((proprio_t, latent), dim=-1)).numpy()


def load_cts_policy(path: Path) -> CtsStudentPolicy:
  """Load a CTS student from ``<run>.onnx`` or ``model_*.pt``."""
  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f"CTS policy not found: {path}")
  if path.suffix == ".onnx":
    return CtsOnnxStudent(path)
  if path.suffix == ".pt":
    return CtsCheckpointStudent(path)
  raise ValueError(
    f"{path.name} is neither a CTS ONNX export nor a model_*.pt checkpoint"
  )
