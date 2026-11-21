from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from telescene.hands import (ROOM_TO_SIM_XYZW, WRIST_IN_GRIP_XYZW, head_target, is_valid, operator_yaw, quat_apply,
                             quat_mul, room_position_to_sim, wrist_target)


def body(facing_yaw_room=0.0, at=(0.0, 0.0)):
    """Standing at ``at`` (room x, z), facing -Z turned left by ``facing_yaw_room``."""
    b = np.zeros((24, 7))
    b[:, 6] = 1
    c, s = math.cos(facing_yaw_room), math.sin(facing_yaw_room)
    # The left hip is at -X when facing -Z. Turning left swings it towards +Z.
    b[0, :3] = (at[0], 0.95, at[1])
    b[1, :3] = (at[0] - 0.1 * c, 0.9, at[1] + 0.1 * s)
    b[2, :3] = (at[0] + 0.1 * c, 0.9, at[1] - 0.1 * s)
    return b


def test_constants_are_the_axis_changes_they_claim():
    assert quat_apply(ROOM_TO_SIM_XYZW, (0, 0, -1)) == pytest.approx([1, 0, 0])
    assert quat_apply(ROOM_TO_SIM_XYZW, (-1, 0, 0)) == pytest.approx([0, 1, 0])
    assert quat_apply(ROOM_TO_SIM_XYZW, (0, 1, 0)) == pytest.approx([0, 0, 1])
    assert quat_apply(WRIST_IN_GRIP_XYZW, (1, 0, 0)) == pytest.approx([0, 0, -1])
    assert quat_apply(WRIST_IN_GRIP_XYZW, (0, 1, 0)) == pytest.approx([-1, 0, 0])
    assert quat_apply(WRIST_IN_GRIP_XYZW, (0, 0, 1)) == pytest.approx([0, 1, 0])


def test_quat_helpers_agree_with_scipy():
    a, b = Rotation.from_euler("xyz", (0.3, -0.2, 0.9)), Rotation.from_euler("xyz", (-1.0, 0.4, 0.1))
    assert (Rotation.from_quat(quat_mul(a.as_quat(), b.as_quat())) * (a * b).inv()).magnitude() < 1e-9
    assert quat_apply(a.as_quat(), (1, 2, 3)) == pytest.approx(a.apply((1, 2, 3)))


def test_operator_yaw_follows_the_hips_not_the_head():
    assert operator_yaw(body()) == pytest.approx(0.0, abs=1e-9)
    assert operator_yaw(body(0.5)) == pytest.approx(0.5)
    assert operator_yaw(body(), anchor_yaw=math.pi) == pytest.approx(math.pi, abs=1e-9) or \
        operator_yaw(body(), anchor_yaw=math.pi) == pytest.approx(-math.pi, abs=1e-9)


def grip(x, y, z):
    return np.array((x, y, z, 0, 0, 0, 1.0))


def test_a_hand_ahead_of_the_operator_lands_ahead_of_the_robot_scaled():
    target = wrist_target(grip(0.0, 1.0, -0.4), body(), (2.0, 3.0), 0.0, 0.0, 0.75)
    assert target[:3] == pytest.approx([2.0 + 0.3, 3.0, 0.75])


def test_walking_away_does_not_move_the_robots_hands():
    near = wrist_target(grip(0.0, 1.0, -0.4), body(), (2.0, 3.0), 0.0, 0.0, 0.75)
    far = wrist_target(grip(1.5, 1.0, -2.4), body(at=(1.5, -2.0)), (2.0, 3.0), 0.0, 0.0, 0.75)
    assert far[:3] == pytest.approx(near[:3])


def test_robot_heading_carries_the_hand_round():
    target = wrist_target(grip(0.0, 1.0, -0.4), body(), (0.0, 0.0), math.pi / 2, 0.0, 1.0)
    assert target[:3] == pytest.approx([0.0, 0.4, 1.0])


def test_operator_turning_with_the_robot_changes_nothing():
    target = wrist_target(grip(-0.4 * math.sin(0.5), 1.0, -0.4 * math.cos(0.5)), body(0.5), (0.0, 0.0), 0.5, 0.0, 1.0)
    assert np.hypot(*target[:2]) == pytest.approx(0.4)
    assert math.atan2(target[1], target[0]) == pytest.approx(0.5)


def test_upright_grip_gives_a_wrist_along_the_forearm_with_the_thumb_up():
    q = wrist_target(grip(0.0, 1.0, -0.4), body(), (0.0, 0.0), 0.0, 0.0, 1.0)[3:7]
    assert quat_apply(q, (1, 0, 0)) == pytest.approx([1, 0, 0])  # forearm points ahead
    assert quat_apply(q, (0, 0, 1)) == pytest.approx([0, 0, 1])  # up stays up


def test_tilting_the_controller_up_tilts_the_wrist_up():
    up = Rotation.from_euler("x", 0.5).as_quat()  # about room +X: -Z ahead rises towards +Y
    q = wrist_target(np.array((0.0, 1.0, -0.4, *up)), body(), (0.0, 0.0), 0.0, 0.0, 1.0)[3:7]
    forearm = quat_apply(q, (1, 0, 0))
    assert forearm[2] == pytest.approx(math.sin(0.5))


def test_head_keeps_its_tilt_and_loses_room_yaw():
    q = head_target(np.array((0, 1.6, 0, 0, 0, 0, 1.0)), body(), 1.0, 0.0)[3:7]
    assert quat_apply(q, (0, 0, -1)) == pytest.approx([math.cos(1.0), math.sin(1.0), 0.0])


def test_zero_quaternion_is_invalid():
    assert not is_valid(np.zeros(7)) and is_valid(grip(0, 0, 0))


def test_room_position_turns_with_the_anchor():
    assert room_position_to_sim((0, 0, -1), math.pi / 2) == pytest.approx([0, 1, 0])
