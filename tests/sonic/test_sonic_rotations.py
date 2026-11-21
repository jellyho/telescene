# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""wxyz quaternion helpers against scipy."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from telescene.sonic.core import rotations as rot


def _rand(n, seed=0):
    r = Rotation.random(n, random_state=seed)
    xyzw = r.as_quat()
    return r, torch.from_numpy(xyzw[:, [3, 0, 1, 2]]).float()


def test_mul_apply_matrix():
    ra, qa = _rand(16, 0)
    rb, qb = _rand(16, 1)
    v = torch.randn(16, 3)
    prod = (ra * rb).as_quat()[:, [3, 0, 1, 2]]
    got = rot.quat_mul(qa, qb).numpy()
    np.testing.assert_allclose(np.abs((got * prod).sum(-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(rot.quat_apply(qa, v).numpy(), ra.apply(v.numpy()), atol=1e-5)
    np.testing.assert_allclose(rot.quat_apply_inverse(qa, v).numpy(), ra.inv().apply(v.numpy()), atol=1e-5)
    np.testing.assert_allclose(rot.quat_to_matrix(qa).numpy(), ra.as_matrix(), atol=1e-5)
    np.testing.assert_allclose(rot.rot6d(qa).numpy(), ra.as_matrix()[..., :2].reshape(16, 6), atol=1e-5)
    back = rot.quat_from_matrix(rot.quat_to_matrix(qa))
    np.testing.assert_allclose((back * qa).sum(-1).abs().numpy(), 1.0, atol=1e-5)


def test_conventions_round_trip():
    _, q = _rand(8, 2)
    torch.testing.assert_close(rot.wxyz_to_xyzw(rot.xyzw_to_wxyz(q)), q)
    got = rot.xyzw_to_wxyz(torch.tensor([[0.1, 0.2, 0.3, 0.9]]))
    torch.testing.assert_close(got, torch.tensor([[0.9, 0.1, 0.2, 0.3]]))


def test_heading():
    yaw = torch.tensor([0.3, -2.0, 3.0])
    q = rot.yaw_to_quat(yaw)
    torch.testing.assert_close(rot.heading_quat(q), q)
    torch.testing.assert_close(rot.quat_yaw(q), yaw)
    tilt = torch.from_numpy(Rotation.from_euler("xyz", [[0.2, -0.1, 0.0]] * 3).as_quat()[:, [3, 0, 1, 2]]).float()
    q_tilted = rot.quat_mul(q, tilt)
    torch.testing.assert_close(rot.quat_yaw(q_tilted), yaw, atol=0.05, rtol=0)
    h = rot.heading_quat(q_tilted)
    torch.testing.assert_close(h[:, 1:3], torch.zeros(3, 2))
    torch.testing.assert_close(rot.quat_mul(h, rot.heading_quat_inv(q_tilted)), torch.tensor([[1.0, 0, 0, 0]] * 3))


def test_slerp():
    ra, qa = _rand(1, 3)
    rb, qb = _rand(1, 4)
    for t in (0.0, 0.25, 0.7, 1.0):
        expected = Slerp([0, 1], Rotation.concatenate([ra, rb]))([t]).as_quat()[:, [3, 0, 1, 2]]
        got = rot.quat_slerp(qa, qb, torch.tensor([t])).numpy()
        np.testing.assert_allclose(np.abs((got * expected).sum(-1)), 1.0, atol=1e-5)
    # nearly equal quaternions take the normalised-lerp branch
    q = rot.quat_slerp(qa, qa, torch.tensor([0.5]))
    torch.testing.assert_close(q, qa)
