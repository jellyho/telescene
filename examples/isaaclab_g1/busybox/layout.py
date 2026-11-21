# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Poses in the cell frame.

Cell frame = garage_layout world: Z up, metres, yaw 0 along +X. The cell is spawned at the
environment origin, so these are environment-local poses. Quaternions are (x, y, z, w), as in
Isaac Lab 3.0. No Isaac Lab import.
"""

from __future__ import annotations

import math


Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
"""Position [m] and quaternion (x, y, z, w)."""

IDENTITY_POSE: Pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    """Rotation about +Z by ``yaw`` [rad] as (x, y, z, w)."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def quat_mul_xyzw(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Hamilton product ``a * b`` of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_apply_xyzw(q: tuple[float, ...], v: tuple[float, ...]) -> tuple[float, float, float]:
    """Rotate the vector ``v`` by the (x, y, z, w) quaternion ``q``."""
    x, y, z, _ = quat_mul_xyzw(quat_mul_xyzw(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return (x, y, z)


def compose_pose(parent: Pose, local: Pose) -> Pose:
    """``parent * local``: ``local`` (relative to the parent) in the frame ``parent`` is given in."""
    (pp, pq), (lp, lq) = parent, local
    rotated = quat_apply_xyzw(pq, lp)
    return (
        (pp[0] + rotated[0], pp[1] + rotated[1], pp[2] + rotated[2]),
        quat_mul_xyzw(pq, lq),
    )
