# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Tiny stand-in ONNX models with the released SONIC signatures (no weights, no licence needed).

They let the whole pipeline, including the Isaac Lab action term, run before the real checkpoints are
downloaded. Outputs are deterministic functions of the inputs and meaningless as control.

* ``model_encoder.onnx``: ``obs_dict [B, enc_dim] -> encoded_tokens [B, 64]`` (tanh of a random linear map).
* ``model_decoder.onnx``: ``obs_dict [B, 994] -> action [B, 29]``.
* ``planner_sonic.onnx``: the 11 planner inputs -> ``mujoco_qpos [B, 64, 36]`` float32 and
  ``num_pred_frames [B]`` int32. The root moves along ``movement_direction`` at ``max(target_vel, 0)``, the
  joints relax from the last context frame towards SONIC's default pose and wiggle when ``mode > 0``.

``batch=1`` reproduces the fixed batch of the released export; ``batch=None`` makes the batch dim symbolic.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .joint_order import DEFAULT_ANGLES_MUJOCO, DEFAULT_ROOT_HEIGHT


def _onnx():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    return onnx, TensorProto, helper, numpy_helper


def _save(model, path: str):
    onnx, *_ = _onnx()
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx.save(model, path)


def make_linear_model(
    path: str,
    input_name: str,
    output_name: str,
    in_dim: int,
    out_dim: int,
    seed: int,
    gain: float = 1.0,
    out_scale: float = 1.0,
    batch: int | None = 1,
):
    """``out = out_scale * tanh(x @ W + b)`` with ``W ~ N(0, gain^2 / in_dim)``."""
    onnx, TensorProto, helper, numpy_helper = _onnx()
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((in_dim, out_dim)) * gain / math.sqrt(in_dim)).astype(np.float32)
    b = (rng.standard_normal(out_dim) * 0.1).astype(np.float32)
    bdim = batch if batch is not None else "batch"
    x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [bdim, in_dim])
    y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [bdim, out_dim])
    nodes = [
        helper.make_node("MatMul", [input_name, "W"], ["xw"]),
        helper.make_node("Add", ["xw", "B"], ["pre"]),
        helper.make_node("Tanh", ["pre"], ["act"]),
        helper.make_node("Mul", ["act", "S"], [output_name]),
    ]
    inits = [
        numpy_helper.from_array(w, "W"),
        numpy_helper.from_array(b, "B"),
        numpy_helper.from_array(np.array(out_scale, dtype=np.float32), "S"),
    ]
    graph = helper.make_graph(nodes, "sonic_dummy", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="telescene")
    model.ir_version = 8
    _save(model, path)


def make_planner_model(path: str, num_frames: int = 48, batch: int | None = 1):
    """Dummy planner with the version-1 signature (11 inputs)."""
    onnx, TensorProto, helper, numpy_helper = _onnx()
    bdim = batch if batch is not None else "batch"
    f, i64 = TensorProto.FLOAT, TensorProto.INT64
    inputs = [
        helper.make_tensor_value_info("context_mujoco_qpos", f, [bdim, 4, 36]),
        helper.make_tensor_value_info("target_vel", f, [bdim]),
        helper.make_tensor_value_info("mode", i64, [bdim]),
        helper.make_tensor_value_info("movement_direction", f, [bdim, 3]),
        helper.make_tensor_value_info("facing_direction", f, [bdim, 3]),
        helper.make_tensor_value_info("random_seed", i64, [bdim]),
        helper.make_tensor_value_info("height", f, [bdim]),
        helper.make_tensor_value_info("has_specific_target", i64, [bdim, 1]),
        helper.make_tensor_value_info("specific_target_positions", f, [bdim, 4, 3]),
        helper.make_tensor_value_info("specific_target_headings", f, [bdim, 4]),
        helper.make_tensor_value_info("allowed_pred_num_tokens", i64, [bdim, 11]),
    ]
    outputs = [
        helper.make_tensor_value_info("mujoco_qpos", f, [bdim, 64, 36]),
        helper.make_tensor_value_info("num_pred_frames", TensorProto.INT32, [bdim]),
    ]
    t = (np.arange(64, dtype=np.float32) / 30.0).reshape(1, 64, 1)
    joint_mask = np.zeros((1, 1, 36), dtype=np.float32)
    joint_mask[..., 7:] = 1.0
    relax = (1.0 - np.exp(-t)) * joint_mask
    default = np.zeros((1, 1, 36), dtype=np.float32)
    default[0, 0, 2] = DEFAULT_ROOT_HEIGHT
    default[0, 0, 7:] = DEFAULT_ANGLES_MUJOCO
    wiggle = (0.05 * np.sin(2.0 * np.pi * t) * joint_mask).astype(np.float32)
    select = np.zeros((3, 36), dtype=np.float32)
    select[0, 0] = select[1, 1] = select[2, 2] = 1.0
    inits = [
        numpy_helper.from_array(np.array([3], dtype=np.int64), "last_idx"),
        numpy_helper.from_array(t, "T"),
        numpy_helper.from_array(relax.astype(np.float32), "RELAX"),
        numpy_helper.from_array(default, "DEFAULT"),
        numpy_helper.from_array(wiggle, "WIGGLE"),
        numpy_helper.from_array(select, "SELECT"),
        numpy_helper.from_array(np.zeros((1, 64, 36), dtype=np.float32), "ZEROS"),
        numpy_helper.from_array(np.array([-1, 1, 1], dtype=np.int64), "shape_b11"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "ZERO"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "ONE"),
        numpy_helper.from_array(np.array(float(num_frames), dtype=np.float32), "NFRAMES"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes12"),
    ]
    nodes = [
        # last context frame, broadcast over 64 output frames
        helper.make_node("Gather", ["context_mujoco_qpos", "last_idx"], ["last"], axis=1),  # [B,1,36]
        helper.make_node("Add", ["last", "ZEROS"], ["base"]),  # [B,64,36]
        # joints relax towards the default pose
        helper.make_node("Sub", ["DEFAULT", "base"], ["to_default"]),
        helper.make_node("Mul", ["to_default", "RELAX"], ["relaxed"]),
        # root translation along movement_direction at max(target_vel, 0)
        helper.make_node("Relu", ["target_vel"], ["speed"]),
        helper.make_node("Reshape", ["speed", "shape_b11"], ["speed_b11"]),
        helper.make_node("MatMul", ["movement_direction", "SELECT"], ["dir36"]),  # [B,36]
        helper.make_node("Unsqueeze", ["dir36", "last_idx_axis"], ["dir_b136"]),
        helper.make_node("Mul", ["dir_b136", "speed_b11"], ["vel"]),
        helper.make_node("Mul", ["vel", "T"], ["disp"]),
        # wiggle when mode > 0
        helper.make_node("Cast", ["mode"], ["mode_f"], to=f),
        helper.make_node("Min", ["mode_f", "ONE"], ["moving"]),
        helper.make_node("Reshape", ["moving", "shape_b11"], ["moving_b11"]),
        helper.make_node("Mul", ["WIGGLE", "moving_b11"], ["wig"]),
        # every other input contributes 0 so all 11 are consumed
        helper.make_node("ReduceSum", ["facing_direction"], ["s0"], keepdims=0),
        helper.make_node("Cast", ["random_seed"], ["rs_f"], to=f),
        helper.make_node("ReduceSum", ["rs_f"], ["s1"], keepdims=0),
        helper.make_node("ReduceSum", ["height"], ["s2"], keepdims=0),
        helper.make_node("Cast", ["has_specific_target"], ["hst_f"], to=f),
        helper.make_node("ReduceSum", ["hst_f"], ["s3"], keepdims=0),
        helper.make_node("ReduceSum", ["specific_target_positions"], ["s4"], keepdims=0),
        helper.make_node("ReduceSum", ["specific_target_headings"], ["s5"], keepdims=0),
        helper.make_node("Cast", ["allowed_pred_num_tokens"], ["apt_f"], to=f),
        helper.make_node("ReduceSum", ["apt_f"], ["s6"], keepdims=0),
        helper.make_node("Sum", ["s0", "s1", "s2", "s3", "s4", "s5", "s6"], ["s_all"]),
        helper.make_node("Mul", ["s_all", "ZERO"], ["zero_term"]),
        helper.make_node("Sum", ["base", "relaxed", "disp", "wig", "zero_term"], ["mujoco_qpos"]),
        # num_pred_frames = constant per batch row
        helper.make_node("ReduceSum", ["context_mujoco_qpos", "axes12"], ["ctx_sum"], keepdims=0),  # [B]
        helper.make_node("Mul", ["ctx_sum", "ZERO"], ["ctx_zero"]),
        helper.make_node("Add", ["ctx_zero", "NFRAMES"], ["frames_f"]),
        helper.make_node("Cast", ["frames_f"], ["num_pred_frames"], to=TensorProto.INT32),
    ]
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "last_idx_axis"))
    graph = helper.make_graph(nodes, "sonic_dummy_planner", inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="telescene")
    model.ir_version = 8
    _save(model, path)


def make_dummy_sonic_dir(
    out_dir: str,
    encoder_dim: int = 1762,
    decoder_dim: int = 994,
    token_dim: int = 64,
    num_frames: int = 48,
    batch: int | None = 1,
    seed: int = 0,
) -> str:
    """Write the three dummy models into ``out_dir`` with the released file names; returns ``out_dir``."""
    from .policy import DECODER_FILE, ENCODER_FILE, PLANNER_FILE

    make_linear_model(
        os.path.join(out_dir, ENCODER_FILE), "obs_dict", "encoded_tokens", encoder_dim, token_dim, seed, batch=batch
    )
    make_linear_model(
        os.path.join(out_dir, DECODER_FILE), "obs_dict", "action", decoder_dim, 29, seed + 1, gain=0.3,
        out_scale=0.5, batch=batch,
    )
    make_planner_model(os.path.join(out_dir, PLANNER_FILE), num_frames=num_frames, batch=batch)
    return out_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m telescene.sonic.core.dummy OUT_DIR")
    os.makedirs(sys.argv[1], exist_ok=True)
    print(make_dummy_sonic_dir(sys.argv[1]))
