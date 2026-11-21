"""Where the operator's hands should put the robot's hands, for controllers that take wrist targets.

Hands are measured from the operator's own body, not the room, and that offset is placed on the robot. If they were
room positions, a robot still walking to catch up with the operator would see its hands flung ahead of it.
"""

from __future__ import annotations

import math

import numpy as np

# OpenXR room axes to the simulator's: forward -Z is +X, left -X is +Y, up +Y is +Z. xyzw.
ROOM_TO_SIM_XYZW = (0.5, -0.5, -0.5, 0.5)
# The axes of a wrist link (x along the forearm, y left, z up) in an upright OpenXR grip (x right, y up, -z ahead). xyzw.
WRIST_IN_GRIP_XYZW = (-0.5, 0.5, 0.5, 0.5)
LEFT_HIP, RIGHT_HIP = 1, 2


def quat_mul(a, b) -> np.ndarray:
    """xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), np.float64)


def quat_apply(q, v) -> np.ndarray:
    u = np.asarray(q[:3], np.float64)
    v = np.asarray(v, np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), np.float64)


def room_position_to_sim(position, anchor_yaw: float = 0.0) -> np.ndarray:
    """Axes and heading only. No anchor, no scale."""
    x, y, z = position
    return quat_apply(yaw_quat(anchor_yaw), (-z, -x, y))


def operator_yaw(body: np.ndarray, anchor_yaw: float = 0.0) -> float:
    """The way the hips face, in simulator yaw. From the hip joints' positions, so it needs no joint axis convention,
    and unlike the head it stays put when the operator only looks around."""
    left = room_position_to_sim(body[LEFT_HIP, :3] - body[RIGHT_HIP, :3], anchor_yaw)
    return math.atan2(-left[0], left[1])


def is_valid(pose) -> bool:
    return float(np.linalg.norm(pose[3:7])) > 0.5


def wrist_target(grip, body: np.ndarray, root_xy, root_yaw: float, anchor_yaw: float, world_scale: float) -> np.ndarray:
    """``(7,)`` xyz, xyzw of a wrist link in the simulator's world, from an OpenXR controller grip.

    Args:
        grip: ``(7,)`` controller grip in the room.
        body: ``(24, 7)`` tracked body in the room. Only the pelvis and hips are used.
        root_xy: Robot pelvis position.
        root_yaw: Robot heading.
        anchor_yaw: Simulator yaw of the room.
        world_scale: Simulator metres per real metre.
    """
    turn = yaw_quat(root_yaw - operator_yaw(body, anchor_yaw))
    from_pelvis = room_position_to_sim(np.asarray(grip[:3]) - body[0, :3], anchor_yaw) * world_scale
    placed = quat_apply(turn, (from_pelvis[0], from_pelvis[1], 0.0))
    height = float(grip[1]) * world_scale
    orientation = quat_mul(quat_mul(quat_mul(turn, yaw_quat(anchor_yaw)), ROOM_TO_SIM_XYZW), quat_mul(grip[3:7], WRIST_IN_GRIP_XYZW))
    return np.array((root_xy[0] + placed[0], root_xy[1] + placed[1], height, *orientation), np.float64)


def head_target(head, body: np.ndarray, root_yaw: float, anchor_yaw: float) -> np.ndarray:
    """``(7,)`` of the headset with the same turn applied. Position is unused by SONIC, which takes only the tilt."""
    turn = yaw_quat(root_yaw - operator_yaw(body, anchor_yaw))
    orientation = quat_mul(quat_mul(quat_mul(turn, yaw_quat(anchor_yaw)), ROOM_TO_SIM_XYZW), head[3:7])
    return np.array((0.0, 0.0, 0.0, *orientation), np.float64)
