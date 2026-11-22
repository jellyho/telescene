"""GigaBrain-WBC: the encoder input against an independent computation with scipy, the controller's inputs, and, where
``scripts/download_trackers.sh`` has fetched the graphs, the decoder's KV cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_supermimic import control  # noqa: E402

from telescene.trackers import gigabrain as gb  # noqa: E402
from telescene.trackers import onnx_tracker as ot  # noqa: E402
from telescene.trackers import supermimic as sm  # noqa: E402

SPEC = Path(__file__).parents[2] / "specs" / "trackers" / "gigabrain_wbc.json"


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def random_frame(rng):
    return ot.Frame(rng.normal(size=29), np.zeros(29), rng.normal(size=3), random_quat(rng), np.zeros(3), np.zeros(3))


def test_encoder_input_matches_scipy():
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(3)
    frames = [random_frame(rng) for _ in range(10)]
    base = random_frame(rng)
    robot_quat, robot_pos = random_quat(rng), rng.normal(size=3)
    got = gb.encoder_input(frames, base, robot_quat, robot_pos, mode=0)

    def rot(q):
        return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])

    robot = rot(robot_quat)
    want = np.concatenate([
        [1, 0, 0, 0],
        np.concatenate([f.joint_pos for f in frames]),
        np.concatenate([(robot.inv() * rot(f.root_quat)).as_matrix()[:, :2].reshape(-1) for f in frames]),
        np.concatenate([rot(base.root_quat).inv().apply(f.root_pos - base.root_pos) for f in frames]),
        np.concatenate([robot.inv().apply(f.root_pos - robot_pos) for f in frames]),
        np.concatenate([rot(f.root_quat).inv().apply([0.0, 0.0, -1.0]) for f in frames]),
    ])
    assert got.shape == (444,)
    np.testing.assert_allclose(got, want, atol=1e-12)
    assert np.all(gb.encoder_input(frames, base, robot_quat, None)[384:414] == 0)  # no position: zeros


def standing():
    return sm.RobotState(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), sm.DEFAULT_JOINT_POS.copy(), np.zeros(29),
                         pos=np.array([0.0, 0.0, 0.79]), lin_vel_b=np.zeros(3))


def test_controller_inputs():
    spec = ot.TrackerSpec.from_json(SPEC)
    assert spec.controller == "gigabrain" and spec.encoder.name == "gigabrain_wbc_encoder.onnx"
    seen = []
    controller = ot.make_controller(standing, spec, policy=lambda inputs: seen.append(inputs) or np.zeros(29))
    assert isinstance(controller, gb.GigaBrainController)
    for _ in range(3):
        controller.act(control(tracking=False))
    obs, enc = seen[-1]["obs_dict"], seen[-1]["encoder"]
    assert obs.shape == (93,) and enc.shape == (444,)
    np.testing.assert_allclose(obs[90:93], [0, 0, -1])  # projected gravity, upright
    # standing: every reference frame is the default pose, in the network's order
    np.testing.assert_allclose(enc[4:33], spec.default_joint_pos, atol=1e-9)
    np.testing.assert_allclose(enc[4:294].reshape(10, 29), np.tile(spec.default_joint_pos, (10, 1)), atol=1e-9)
    np.testing.assert_allclose(enc[354:384], 0, atol=1e-9)  # the reference does not move from frame -1


def test_kv_cache():
    pytest.importorskip("onnxruntime")
    spec = ot.TrackerSpec.from_json(SPEC)
    if not (spec.onnx.exists() and spec.encoder.exists()):
        pytest.skip("scripts/download_trackers.sh")
    policy = gb.GigaBrainPolicy(spec)
    controller = ot.make_controller(standing, spec, policy=policy)
    for i in range(1, 4):
        target = controller.act(control(tracking=False))
        assert np.all(np.isfinite(target))
        assert float(policy.cache["abs_position"].reshape(-1)[0]) == i
    controller.reset()
    assert not policy.cache["abs_position"].any() and not policy.cache["past_key_values"].any()
