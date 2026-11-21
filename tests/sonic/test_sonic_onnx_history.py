# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""ONNX wrapper (signature checks, batch handling) and history buffers."""

import os

import numpy as np
import pytest
import torch

from telescene.sonic.core import onnx_model as om
from telescene.sonic.core.dummy import make_linear_model
from telescene.sonic.core.history import HistoryBuffer
from telescene.sonic.core.policy import SonicPolicyCore


def _models(d):
    enc = om.OnnxModel(os.path.join(d, "model_encoder.onnx"), om.ENCODER_INPUTS, om.ENCODER_OUTPUTS)
    dec = om.OnnxModel(os.path.join(d, "model_decoder.onnx"), om.DECODER_INPUTS, om.DECODER_OUTPUTS)
    pln = om.OnnxModel(os.path.join(d, "planner_sonic.onnx"), om.PLANNER_INPUTS, om.PLANNER_OUTPUTS)
    return enc, dec, pln


def test_load_and_signature(dummy_dir):
    enc, dec, pln = _models(dummy_dir)
    assert enc.fixed_batch and dec.fixed_batch and pln.fixed_batch
    assert enc.input_shape("obs_dict") == [1, 1762]
    assert enc.output_shape("encoded_tokens") == [1, 64]
    assert dec.input_shape("obs_dict") == [1, 994]
    assert dec.output_shape("action") == [1, 29]
    assert [s.name for s in pln.input_specs] == [s.name for s in om.PLANNER_INPUTS]
    assert "CPUExecutionProvider" in enc.session.get_providers()


def test_planner_outputs(dummy_dir):
    _, _, pln = _models(dummy_dir)
    feeds = {
        "context_mujoco_qpos": np.zeros((1, 4, 36), np.float32),
        "target_vel": np.array([0.5], np.float32),
        "mode": np.array([1], np.int64),
        "movement_direction": np.array([[1, 0, 0]], np.float32),
        "facing_direction": np.array([[1, 0, 0]], np.float32),
        "random_seed": np.array([1234], np.int64),
        "height": np.array([-1], np.float32),
        "has_specific_target": np.zeros((1, 1), np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), np.float32),
        "specific_target_headings": np.zeros((1, 4), np.float32),
        "allowed_pred_num_tokens": np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
    }
    out = pln.run(feeds)
    assert out["mujoco_qpos"].shape == (1, 64, 36) and out["mujoco_qpos"].dtype == np.float32
    assert out["num_pred_frames"].dtype == np.int32 and int(out["num_pred_frames"].item()) == 48
    # root moves forward at 0.5 m/s, 30 Hz frames
    np.testing.assert_allclose(out["mujoco_qpos"][0, 30, 0], 0.5, atol=1e-5)


def test_wrong_signature_raises(tmp_path):
    bad = tmp_path / "bad.onnx"
    make_linear_model(str(bad), "obs", "encoded_tokens", 10, 64, seed=0)
    with pytest.raises(ValueError, match="missing input 'obs_dict'"):
        om.OnnxModel(bad, om.ENCODER_INPUTS, om.ENCODER_OUTPUTS)
    bad2 = tmp_path / "bad2.onnx"
    make_linear_model(str(bad2), "obs_dict", "action", 994, 12, seed=0)
    with pytest.raises(ValueError, match="shape"):
        om.OnnxModel(bad2, om.DECODER_INPUTS, om.DECODER_OUTPUTS)
    with pytest.raises(FileNotFoundError):
        om.OnnxModel(tmp_path / "missing.onnx", om.ENCODER_INPUTS, om.ENCODER_OUTPUTS)


def test_layout_width_mismatch(tmp_path, dummy_dir):
    from telescene.sonic.core.dummy import make_dummy_sonic_dir

    d = make_dummy_sonic_dir(str(tmp_path / "ll"), encoder_dim=1247)
    with pytest.raises(ValueError, match="encoder input width 1247"):
        SonicPolicyCore.from_dir(d, num_envs=1)
    core = SonicPolicyCore.from_dir(d, num_envs=1, observation_config="low_latency")
    assert core.layout.encoder_dim == 1247


def test_fixed_batch_loop_matches_dynamic(dummy_dir, dummy_dir_dynamic):
    enc_fixed, _, pln_fixed = _models(dummy_dir)
    enc_dyn, _, pln_dyn = _models(dummy_dir_dynamic)
    assert not enc_dyn.fixed_batch
    x = np.random.default_rng(0).standard_normal((5, 1762)).astype(np.float32)
    a = enc_fixed.run({"obs_dict": x})["encoded_tokens"]
    b = enc_dyn.run({"obs_dict": x})["encoded_tokens"]
    assert a.shape == (5, 64)
    np.testing.assert_allclose(a, b, atol=1e-6)
    row = enc_fixed.run({"obs_dict": x[2:3]})["encoded_tokens"]
    np.testing.assert_array_equal(a[2:3], row)
    out = enc_fixed.run_torch({"obs_dict": torch.from_numpy(x)})
    assert isinstance(out["encoded_tokens"], torch.Tensor)


def test_history_zero_padding_and_order():
    h = HistoryBuffer(2, 10, 3)
    for k in range(3):
        h.append(torch.full((2, 3), float(k + 1)))
    data = h.data
    assert torch.all(data[:, :7] == 0)
    torch.testing.assert_close(data[0, 7:, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(h.window(2, 2)[0, :, 0], torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError):
        h.window(6, 2)


def test_history_repeat_and_reset():
    h = HistoryBuffer(3, 4, 2, init="repeat")
    h.append(torch.arange(6.0).view(3, 2))
    torch.testing.assert_close(h.data[1], torch.tensor([[2.0, 3.0]] * 4))
    h.append(torch.full((3, 2), 9.0))
    h.reset(torch.tensor([1]))
    h.append(torch.full((3, 2), 5.0))
    torch.testing.assert_close(h.data[1], torch.full((4, 2), 5.0))
    torch.testing.assert_close(h.data[0, -1], torch.tensor([5.0, 5.0]))
    torch.testing.assert_close(h.data[0, -2], torch.tensor([9.0, 9.0]))
    torch.testing.assert_close(h.data[0, 0], torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError):
        HistoryBuffer(1, 2, 2, init="first")
