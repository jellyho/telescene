# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""onnxruntime wrapper that checks the released SONIC model signatures on load.

Signatures (names, shapes, dtypes) are taken from:

* encoder/decoder: gear_sonic ``utils/inference_helpers.py:318-324,451-457`` (input ``obs_dict``, outputs
  ``encoded_tokens`` / ``action``, exported with batch 1 and no dynamic axes) and IsaacLab-demos
  ``sonic_policy.py:82,156,159`` (``(1, 1762)`` encoder input, ``(1, 994)`` decoder input).
* planner: ``sonic_policy.py:59-71,93`` and gear_sonic_deploy ``localmotion_kplanner_onnx.hpp`` /
  ``localmotion_kplanner_tensorrt.hpp:75-112,188-207`` (11 inputs for planner version >= 1, outputs
  ``mujoco_qpos`` ``[1, 64, 36]`` float32 and ``num_pred_frames`` int32).

Models exported with a fixed batch of 1 are run row by row when more than one environment is fed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

_ORT_TYPES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(bool)": "bool",
}


@dataclass(frozen=True)
class TensorSpec:
    """Expected ONNX tensor: ``None`` in ``shape`` accepts any size (and any symbolic dim); ``dtype`` may list
    several accepted types."""

    name: str
    shape: tuple[int | None, ...] | None
    dtype: str | tuple[str, ...]
    optional: bool = False

    @property
    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype,) if isinstance(self.dtype, str) else tuple(self.dtype)


ENCODER_INPUTS = (TensorSpec("obs_dict", (1, None), "float32"),)
ENCODER_OUTPUTS = (TensorSpec("encoded_tokens", (1, None), "float32"),)
DECODER_INPUTS = (TensorSpec("obs_dict", (1, None), "float32"),)
DECODER_OUTPUTS = (TensorSpec("action", (1, 29), "float32"),)
PLANNER_INPUTS = (
    TensorSpec("context_mujoco_qpos", (1, 4, 36), "float32"),
    TensorSpec("target_vel", (1,), "float32"),
    TensorSpec("mode", (1,), "int64"),
    TensorSpec("movement_direction", (1, 3), "float32"),
    TensorSpec("facing_direction", (1, 3), "float32"),
    TensorSpec("random_seed", (1,), "int64"),
    # planner version >= 1
    TensorSpec("height", (1,), "float32", optional=True),
    TensorSpec("has_specific_target", (1, 1), "int64", optional=True),
    TensorSpec("specific_target_positions", (1, 4, 3), "float32", optional=True),
    TensorSpec("specific_target_headings", (1, 4), "float32", optional=True),
    TensorSpec("allowed_pred_num_tokens", (1, 11), "int64", optional=True),
)
PLANNER_OUTPUTS = (
    TensorSpec("mujoco_qpos", (1, None, 36), "float32"),
    # the deploy reads int32 (localmotion_kplanner_onnx.hpp:218); int64 is accepted as well because the released
    # file is unverified and sonic_policy.py:94 only calls .item()
    TensorSpec("num_pred_frames", None, ("int32", "int64")),
)


def default_providers(device: str | torch.device | None = None) -> list:
    """CUDA execution provider (TF32 off, as in sonic_policy.py:52-55) when available, else CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list = []
    if "CUDAExecutionProvider" in available and (device is None or str(device).startswith("cuda")):
        device_id = 0
        if device is not None and ":" in str(device):
            device_id = int(str(device).split(":")[1])
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        providers.append(("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}))
    providers.append("CPUExecutionProvider")
    return providers


class OnnxModel:
    """One ONNX session with signature checks and batch-1 fallback.

    Args:
        path: Path of the ``.onnx`` file.
        inputs: Expected inputs. Required ones must exist; optional ones are fed only if present.
        outputs: Expected outputs.
        providers: onnxruntime providers. Defaults to :func:`default_providers`.
        device: Torch device used to pick the CUDA device id.
        num_threads: intra/inter-op thread count (1 matches sonic_policy.py:49-50).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        inputs: Sequence[TensorSpec],
        outputs: Sequence[TensorSpec],
        providers: list | None = None,
        device: str | torch.device | None = None,
        num_threads: int = 1,
    ):
        import onnxruntime as ort

        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"ONNX model not found: {self.path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            self.path, options, providers=providers if providers is not None else default_providers(device)
        )
        self._model_inputs = {i.name: i for i in self.session.get_inputs()}
        self._model_outputs = {o.name: o for o in self.session.get_outputs()}
        self.input_specs = self._check(inputs, self._model_inputs, "input")
        self.output_specs = self._check(outputs, self._model_outputs, "output")
        self.output_names = [s.name for s in self.output_specs]
        first = self._model_inputs[self.input_specs[0].name].shape
        self.fixed_batch = isinstance(first[0], int) if first else False

    def _check(self, specs: Sequence[TensorSpec], actual: Mapping, kind: str) -> list[TensorSpec]:
        found = []
        for spec in specs:
            if spec.name not in actual:
                if spec.optional:
                    continue
                raise ValueError(
                    f"{self.path}: missing {kind} '{spec.name}'. Model {kind}s: "
                    + ", ".join(f"{k}{list(v.shape)}:{v.type}" for k, v in actual.items())
                )
            node = actual[spec.name]
            dtype = _ORT_TYPES.get(node.type, node.type)
            if dtype not in spec.dtypes:
                raise ValueError(f"{self.path}: {kind} '{spec.name}' has dtype {dtype}, expected {spec.dtype}")
            if spec.shape is not None:
                shape = list(node.shape)
                if len(shape) != len(spec.shape):
                    raise ValueError(f"{self.path}: {kind} '{spec.name}' has shape {shape}, expected {spec.shape}")
                for got, want in zip(shape, spec.shape):
                    if want is not None and isinstance(got, int) and got != want:
                        raise ValueError(
                            f"{self.path}: {kind} '{spec.name}' has shape {shape}, expected {spec.shape}"
                        )
            found.append(spec)
        return found

    def input_shape(self, name: str) -> list:
        return list(self._model_inputs[name].shape)

    def output_shape(self, name: str) -> list:
        return list(self._model_outputs[name].shape)

    def has_input(self, name: str) -> bool:
        return name in self._model_inputs

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run on numpy feeds whose first axis is the batch; returns outputs with the same batch."""
        feeds = {k: np.ascontiguousarray(v, dtype=_np_dtype(self._model_inputs[k].type)) for k, v in feeds.items()}
        batch = next(iter(feeds.values())).shape[0]
        if not self.fixed_batch or batch == 1:
            values = self.session.run(self.output_names, feeds)
            return dict(zip(self.output_names, values))
        rows = [self.session.run(self.output_names, {k: v[b : b + 1] for k, v in feeds.items()}) for b in range(batch)]
        return {
            name: np.concatenate([np.atleast_1d(np.asarray(r[i])) for r in rows], axis=0)
            for i, name in enumerate(self.output_names)
        }

    def run_torch(self, feeds: Mapping[str, torch.Tensor], device: str | torch.device | None = None) -> dict:
        """Torch convenience: inputs are moved to CPU numpy, outputs come back as tensors on ``device``."""
        out = self.run({k: v.detach().cpu().numpy() for k, v in feeds.items()})
        dev = device if device is not None else next(iter(feeds.values())).device
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in out.items()}


def _np_dtype(ort_type: str):
    return np.dtype(_ORT_TYPES.get(ort_type, "float32"))
