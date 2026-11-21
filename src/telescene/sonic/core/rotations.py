# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched quaternion helpers in SONIC's **(w, x, y, z)** convention.

Isaac Lab 3.0 uses (x, y, z, w); convert at the boundary with :func:`xyzw_to_wxyz` / :func:`wxyz_to_xyzw`.
Formulas follow gear_sonic_deploy ``include/math_utils.hpp`` (heading: lines 300-314, slerp: 339-370).
"""

from __future__ import annotations

import torch


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1)


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternions ``q``."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rot6d(q: torch.Tensor) -> torch.Tensor:
    """First two rotation-matrix columns flattened row-wise: ``[m00, m01, m10, m11, m20, m21]``.

    Matches ``as_matrix()[..., :2].reshape(-1)`` (sonic_policy.py:155) and the deploy gatherer
    (g1_deploy_onnx_ref.cpp:676-684).
    """
    return quat_to_matrix(q)[..., :2].reshape(*q.shape[:-1], 6)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """World-Z twist of ``q``: normalised ``(w, 0, 0, z)`` with a 1e-9 norm floor (math_utils.hpp:300)."""
    h = torch.zeros_like(q)
    h[..., 0] = q[..., 0]
    h[..., 3] = q[..., 3]
    return quat_normalize(h, eps=1e-9)


def heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(heading_quat(q))


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw angle of the heading twist of ``q``."""
    h = heading_quat(q)
    return 2.0 * torch.atan2(h[..., 3], h[..., 0])


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path slerp; linear+normalise when ``dot > 0.9995`` (math_utils.hpp:339)."""
    t = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    if t.dim() < q0.dim():
        t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    lerp = quat_normalize(q0 + t * (q1 - q0))
    theta = torch.acos(dot.clamp(-1.0, 1.0))
    sin_theta = torch.sin(theta).clamp_min(1e-12)
    slerp = (torch.sin((1.0 - t) * theta) * q0 + torch.sin(t * theta) * q1) / sin_theta
    return torch.where(dot > 0.9995, lerp, slerp)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to unit quaternion (w >= 0)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    q = torch.stack(
        (
            (1 + m00 + m11 + m22).clamp_min(0).sqrt(),
            (1 + m00 - m11 - m22).clamp_min(0).sqrt(),
            (1 - m00 + m11 - m22).clamp_min(0).sqrt(),
            (1 - m00 - m11 + m22).clamp_min(0).sqrt(),
        ),
        dim=-1,
    ) * 0.5
    q[..., 1] = torch.copysign(q[..., 1], m21 - m12)
    q[..., 2] = torch.copysign(q[..., 2], m02 - m20)
    q[..., 3] = torch.copysign(q[..., 3], m10 - m01)
    return quat_normalize(q)
