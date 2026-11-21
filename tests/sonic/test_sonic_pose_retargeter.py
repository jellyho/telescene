# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""The body-to-SONIC-pose retargeter: layout, loss-of-tracking behaviour, and the tick budget.

This node sits in a 50 Hz loop, so "it works" is not enough: it has to work inside 20 ms shared
with SONIC's own inference, and it must not allocate per tick. Both are asserted here rather than
asserted in a docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from telescene.sonic.core.pico_body import load_smpl_skeleton  # noqa: E402
from telescene.sonic.pose_retargeter import (  # noqa: E402
    ACTION_SLICE,
    JOINTS,
    POSE_ACTIVE,
    ROOT_QUAT,
    SMPL_JOINTS,
    WRISTS,
    SonicPoseRetargeter,
)

from telescene.sonic.core.pico_body import find_human_joints_info  # noqa: E402

SKELETON = find_human_joints_info()

pytestmark = pytest.mark.skipif(not SKELETON.is_file(), reason="SONIC rest skeleton not fetched")


@pytest.fixture
def node():
    return SonicPoseRetargeter(load_smpl_skeleton(SKELETON), device="cpu")


def identity_frame():
    quat = np.zeros((SMPL_JOINTS, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    return quat, np.ones(SMPL_JOINTS, dtype=np.uint8)


def test_the_slice_is_the_documented_layout(node):
    quat, valid = identity_frame()
    out = node.retarget(quat, valid, pose_active=True)
    assert out.shape == (ACTION_SLICE,) == (83,)
    assert out[POSE_ACTIVE] == 1.0
    assert out[JOINTS].shape == (SMPL_JOINTS * 3,)
    assert out[ROOT_QUAT].shape == (4,)
    assert out[WRISTS].shape == (6,)
    assert np.isfinite(out).all(), "a non-finite action would propagate into SONIC's encoder"


def test_the_root_quaternion_is_xyzw_as_the_action_term_reads_it(node):
    """The slot's convention, pinned against both producer and consumer.

    ``PicoBodyRetargeter`` produces wxyz; ``SonicAction`` reads action[77:81] as xyzw and runs
    ``xyzw_to_wxyz`` on it. Writing the producer's quaternion through unconverted sends a wxyz
    identity into an xyzw slot, which the term then reads as a 180-degree roll -- the robot comes
    up upside down and nothing in the shapes or norms complains. So compare against the producer
    rather than against a hard-coded value.
    """
    from telescene.sonic.core.pico_body import PicoBodyRetargeter
    from telescene.sonic.core.rotations import xyzw_to_wxyz

    quat, valid = identity_frame()
    out = node.retarget(quat, valid, pose_active=True)

    body = PicoBodyRetargeter(num_envs=1, skeleton=load_smpl_skeleton(SKELETON), device="cpu")
    frame = body(
        torch.as_tensor(quat).reshape(1, SMPL_JOINTS, 4),
        torch.as_tensor(valid).reshape(1, SMPL_JOINTS).to(torch.bool),
    )
    as_the_term_sees_it = xyzw_to_wxyz(torch.as_tensor(out[ROOT_QUAT]))
    torch.testing.assert_close(as_the_term_sees_it, frame.root_quat[0], atol=1e-6, rtol=0)


def test_the_root_quaternion_is_normalised(node):
    quat, valid = identity_frame()
    out = node.retarget(quat, valid, pose_active=True)
    assert float(np.linalg.norm(out[ROOT_QUAT])) == pytest.approx(1.0, abs=1e-5)


def test_pose_active_follows_the_toggle(node):
    quat, valid = identity_frame()
    assert node.retarget(quat, valid, pose_active=False)[POSE_ACTIVE] == 0.0
    assert node.retarget(quat, valid, pose_active=True)[POSE_ACTIVE] == 1.0


def test_losing_tracking_holds_the_last_posture_and_drops_pose_active(node):
    """Zeros would be a posture SONIC tries to adopt. Holding is the safe failure."""
    quat, valid = identity_frame()
    good = node.retarget(quat, valid, pose_active=True).copy()
    assert node.has_frame

    lost = node.retarget(quat, np.zeros(SMPL_JOINTS, dtype=np.uint8), pose_active=True)
    assert lost[POSE_ACTIVE] == 0.0, "the operator is no longer driving the robot"
    assert np.allclose(lost[JOINTS], good[JOINTS]), "held, not zeroed"


def test_no_body_data_at_all_reports_no_frame(node):
    node.retarget(*identity_frame()[:1], np.zeros(SMPL_JOINTS, dtype=np.uint8), pose_active=True)
    assert not node.has_frame, "a session that never delivered body data must be distinguishable"


def test_reset_forgets_the_held_posture(node):
    quat, valid = identity_frame()
    node.retarget(quat, valid, pose_active=True)
    node.reset()
    assert not node.has_frame
    assert not node.retarget(quat, np.zeros(SMPL_JOINTS, dtype=np.uint8), pose_active=True).any()


def test_it_does_not_allocate_per_tick(node):
    """The returned buffer must be the same object every call, or a 50 Hz loop churns garbage."""
    quat, valid = identity_frame()
    first = node.retarget(quat, valid, pose_active=True)
    second = node.retarget(quat, valid, pose_active=True)
    assert first is second


def test_it_fits_the_tick_budget(node):
    """50 Hz is a 20 ms tick, shared with SONIC's ~8.2 ms of inference, physics and encoding.

    The threshold is deliberately loose -- 4 ms, measured at about 1.4 -- so this fails on a
    regression of several times over, not on a noisy machine.
    """
    import time

    quat, valid = identity_frame()
    for _ in range(20):
        node.retarget(quat, valid, pose_active=True)
    n = 200
    start = time.perf_counter()
    for _ in range(n):
        node.retarget(quat, valid, pose_active=True)
    per_tick_ms = (time.perf_counter() - start) / n * 1000
    assert per_tick_ms < 4.0, f"{per_tick_ms:.2f} ms/tick leaves too little of the 20 ms budget"
