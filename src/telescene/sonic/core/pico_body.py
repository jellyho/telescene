# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""PICO full-body tracking to one SONIC POSE-mode (SMPL encoder) frame, batched over environments.

Input is the 24-joint body of ``XR_BD_body_tracking`` as Isaac Capture's ``FullBodySource`` hands it over
(``FullBodyInput``: ``body_joint_positions (24, 3)``, ``body_joint_orientations (24, 4)`` **xyzw**,
``body_joint_valid (24,)``; IsaacCapture v1.4.145 ``tensor_types/standard_types.py:210-250``, joint order
``indices.py:83-110`` ``BodyJointIndex``), in the raw OpenXR / DeviceIO frame. SONIC's own Isaac Teleop
reader passes exactly that through unchanged, as the same ``(24, 7)`` array its XRoboToolkit ``PicoReader``
produces (gear_sonic ``utils/teleop/input_readers.py:214-292`` ``_body_data_to_24x7``, ``422-480``), so no
frame change is applied here. Do not transform the body with the env's ``world_T_anchor`` the way the head
and wrist poses are.

What happens, all in ``scripts/pico_manager_thread_server.py`` unless named otherwise:

1. ``compute_from_body_poses`` (561-586): each joint's global rotation is post-multiplied by Ry(180 deg),
   local rotations are taken against the SMPL parents (``PoseStreamer.parent_indices``, 1270-1296) and
   become the axis-angle ``body_pose`` (21 joints) and ``global_orient``.
2. ``process_smpl_joints`` (450-489): the root goes from Y-up to Z-up (``isaac_utils/rotations.py:711-719``
   ``smpl_root_ytoz_up``); SMPL-X forward kinematics of the 22 body joints plus both thumb tips (indices
   39, 54) with the rest skeleton of ``gear_sonic/data/human/human_joints_info.pkl``
   (``trl/utils/torch_transform.py:595-656`` ``compute_human_joints``); SMPL's rest orientation is removed
   from the root (``rotations.py:704-708`` ``remove_smpl_base_rot``) and the joints are rotated into that
   root frame. As in SONIC, the pelvis rest position ``J[0]`` is not subtracted, so the joints carry a
   constant offset; joint positions relative to joint 0 are what is frame-independent.
3. Wrist joint targets (``PoseStreamer.run_once`` 1420-1478): the elbow's swing (its twist about y removed,
   ``trl/utils/rotation_conversion.py:605-618`` ``decompose_rotation_aa``) is moved into the wrist, both as
   intrinsic XYZ Euler angles.

Differences from SONIC: no 50 Hz time resampling (``run_once`` 1364-1404); the env reads the latest sample
each policy tick. An invalid joint keeps its last valid rotation and a joint never seen yet takes its
parent's rotation (identity local rotation); SONIC writes zeros there. Zero-angle and 180-degree singular
points of the swing-twist split give finite values instead of SONIC's NaN.

The rest skeleton is read at run time from the GR00T-WholeBodyControl checkout with
``torch.load(weights_only=True)`` (tensors and plain containers only, no code). It is SMPL-X-derived data
whose redistribution terms are not stated, so it is not copied into this repository.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import REPO_ROOT
from .rotations import quat_apply, quat_conjugate, quat_mul, quat_normalize, quat_to_matrix, xyzw_to_wxyz

NUM_BODY_JOINTS = 24
"""``XR_BD_body_tracking`` joints, in SMPL order (pelvis 0 ... left hand 22, right hand 23)."""
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
"""Parents of the 22 SMPL body joints (``PoseStreamer.parent_indices``; the same as the SMPL-X tree)."""
THUMB_TIPS = (39, 54)
"""SMPL-X left / right thumb tips appended as joints 22, 23 (``compute_human_joints`` ``use_thumb_joints``)."""
THUMB_PARENTS = (20, 21)
"""The wrists: with zero hand pose a thumb tip is a fixed offset from its wrist."""
SMPL_BODY_POSE_JOINTS = 21
"""``body_pose[..., :63]``: joints 1-21 (``process_smpl_joints`` feeds only these to the FK)."""
ELBOW_WRIST = (17, 18, 19, 20)
"""Left elbow, right elbow, left wrist, right wrist as indices into ``body_pose`` (SMPL joints 18-21)."""

HUMAN_JOINTS_ENV_VAR = "TELESCENE_SMPL_SKELETON"
DEFAULT_HUMAN_JOINTS = (
    REPO_ROOT / "checkpoints" / "human_joints_info.pkl"
)

_RY180 = (0.0, 0.0, 1.0, 0.0)
_RX90 = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
_SMPL_BASE = (0.5, 0.5, 0.5, 0.5)  # 120 deg about (1, 1, 1): SMPL's rest root orientation


# ------------------------------------------------------------------------------------------- rotations
def aa_to_quat(aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Axis-angle ``(..., 3)`` to wxyz; ``w = cos(angle / 2) >= 0`` for angles up to pi."""
    angle = aa.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    k = torch.where(angle > eps, torch.sin(half) / angle.clamp_min(eps), 0.5 - angle * angle / 48.0)
    return torch.cat((torch.cos(half), aa * k), dim=-1)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """The sign with ``w >= 0`` (scipy's ``as_rotvec`` range: angle in [0, pi])."""
    return torch.where(q[..., 0:1] < 0, -q, q)


def quat_to_aa(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Unit wxyz to axis-angle with the angle in [0, pi]."""
    q = quat_canonical(q)
    s = q[..., 1:4].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., 0:1])
    k = torch.where(s > eps, angle / s.clamp_min(eps), torch.full_like(s, 2.0))
    return q[..., 1:4] * k


def quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """Intrinsic XYZ Euler angles (scipy ``as_euler("XYZ")``): ``R = Rx(a) Ry(b) Rz(c)``.

    Away from gimbal lock (``|b| = pi / 2``) this equals scipy; at it scipy picks ``c = 0`` with a warning.
    """
    m = quat_to_matrix(quat_normalize(q))
    b = torch.asin(m[..., 0, 2].clamp(-1.0, 1.0))
    a = torch.atan2(-m[..., 1, 2], m[..., 2, 2])
    c = torch.atan2(-m[..., 0, 1], m[..., 0, 0])
    return torch.stack((a, b, c), dim=-1)


def swing_about(q: torch.Tensor, axis: int) -> torch.Tensor:
    """``q = twist * swing`` with the twist about local ``axis``; returns the swing (``decompose_rotation_aa``)."""
    twist = torch.zeros_like(q)
    twist[..., 0] = q[..., 0]
    twist[..., 1 + axis] = q[..., 1 + axis]
    n = twist.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    twist = torch.where(n > 1e-9, twist / n.clamp_min(1e-9), identity)  # 180 deg about a perpendicular axis
    return quat_mul(quat_conjugate(twist), q)


# -------------------------------------------------------------------------------------------- skeleton
@dataclass(frozen=True)
class SmplSkeleton:
    """Rest positions of the 22 SMPL body joints and both thumb tips, ``(24, 3)``, SMPL frame (Y up)."""

    rest: torch.Tensor

    @classmethod
    def from_joints_info(cls, info: dict) -> SmplSkeleton:
        """From the ``{J (55, 3), parents_list, rot_mats}`` dict of ``human_joints_info.pkl``."""
        joints = torch.as_tensor(info["J"])
        joints = joints if joints.is_floating_point() else joints.float()
        parents = list(info["parents_list"])
        if tuple(joints.shape) != (55, 3):
            raise ValueError(f"expected J of shape (55, 3) (SMPL-X), got {tuple(joints.shape)}")
        if tuple(parents[: len(SMPL_PARENTS)]) != SMPL_PARENTS:
            raise ValueError(f"unexpected SMPL-X parents {parents[:len(SMPL_PARENTS)]}")
        for tip, wrist in zip(THUMB_TIPS, THUMB_PARENTS):     # tip -> thumb2 -> thumb1 -> wrist
            if parents[parents[parents[tip]]] != wrist:
                raise ValueError(f"joint {tip} is not a grandchild of the thumb on wrist {wrist}")
        return cls(joints[list(range(len(SMPL_PARENTS))) + list(THUMB_TIPS)].clone())


def find_human_joints_info(path: str | os.PathLike[str] | None = None) -> Path:
    """An explicit path, else ``$TELESCENE_SMPL_SKELETON``, else the GR00T-WholeBodyControl checkout."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if os.environ.get(HUMAN_JOINTS_ENV_VAR):
        return Path(os.environ[HUMAN_JOINTS_ENV_VAR]).expanduser().resolve()
    return DEFAULT_HUMAN_JOINTS


def load_smpl_skeleton(path: str | os.PathLike[str] | None = None) -> SmplSkeleton:
    """Read ``human_joints_info.pkl`` with ``torch.load(weights_only=True)`` (no code is unpickled)."""
    file = find_human_joints_info(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"SONIC rest skeleton not found at {file}. Run scripts/download_sonic.sh, which fetches it from "
            f"NVlabs/GR00T-WholeBodyControl and checks its hash, or set {HUMAN_JOINTS_ENV_VAR}."
        )
    with open(file, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise FileNotFoundError(
                f"{file} is a Git LFS pointer, not the file. Run scripts/download_sonic.sh."
            )
    return SmplSkeleton.from_joints_info(torch.load(file, map_location="cpu", weights_only=True))


# --------------------------------------------------------------------------------------- conversion
@dataclass
class PicoSmplFrame:
    """One POSE-mode frame per environment, as :meth:`SmplStream.push` takes it."""

    smpl_joints: torch.Tensor
    """``(N, 24, 3)`` joints in the root frame (``smpl_joints_local``)."""
    root_quat: torch.Tensor
    """``(N, 4)`` wxyz root orientation, Z up, SMPL rest orientation removed (``body_quat_w``)."""
    smpl_pose: torch.Tensor
    """``(N, 21, 3)`` local axis-angle of SMPL joints 1-21."""
    wrist_joint_pos: torch.Tensor
    """``(N, 6)`` G1 wrist targets ``[l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]`` (policy 23-28)."""
    ok: torch.Tensor
    """``(N,)`` bool: the pelvis has been tracked at least once; push with ``mask=ok``."""
    pelvis_pos: torch.Tensor | None = None
    """``(N, 3)`` tracked pelvis position, OpenXR frame (SONIC's unused ``transl``)."""


def _const(values: tuple[float, ...], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def body_to_smpl(quat_wxyz: torch.Tensor, skeleton: SmplSkeleton) -> dict[str, torch.Tensor]:
    """The conversion for complete bodies: ``quat_wxyz (N, 24, 4)`` global joint rotations, OpenXR frame."""
    n = quat_wxyz.shape[0]
    g = quat_mul(quat_normalize(quat_wxyz), _const(_RY180, quat_wxyz).expand(n, NUM_BODY_JOINTS, 4))
    nb = len(SMPL_PARENTS)
    parents = torch.tensor(SMPL_PARENTS[1:], device=quat_wxyz.device)
    local = torch.cat((g[:, :1], quat_mul(quat_conjugate(g[:, parents]), g[:, 1:nb])), dim=1)
    local = quat_canonical(local)                                  # (N, 22, 4), as scipy -> rotvec -> quat
    smpl_pose = quat_to_aa(local[:, 1:])

    rx90 = _const(_RX90, quat_wxyz).expand(n, 4)
    root = quat_mul(rx90, local[:, 0])                             # smpl_root_ytoz_up
    rest = skeleton.rest.to(device=quat_wxyz.device, dtype=quat_wxyz.dtype)
    rot = [root]
    pos = [rest[0].expand(n, 3)]                                   # the root keeps J[0] (not rotated)
    for i in range(1, nb):
        p = SMPL_PARENTS[i]
        pos.append(pos[p] + quat_apply(rot[p], (rest[i] - rest[p]).expand(n, 3)))
        rot.append(quat_mul(rot[p], local[:, i]))
    for k, wrist in enumerate(THUMB_PARENTS):
        pos.append(pos[wrist] + quat_apply(rot[wrist], (rest[nb + k] - rest[wrist]).expand(n, 3)))
    joints = torch.stack(pos, dim=1)                               # (N, 24, 3)

    root_quat = quat_mul(root, quat_conjugate(_const(_SMPL_BASE, quat_wxyz)).expand(n, 4))  # remove_smpl_base_rot
    smpl_joints = quat_apply(quat_conjugate(root_quat).unsqueeze(1).expand(n, joints.shape[1], 4), joints)

    elbow_l, elbow_r, wrist_l, wrist_r = (aa_to_quat(smpl_pose[:, j]) for j in ELBOW_WRIST)
    sl, sr = quat_to_euler_xyz(swing_about(elbow_l, 1)), quat_to_euler_xyz(swing_about(elbow_r, 1))
    wl, wr = quat_to_euler_xyz(wrist_l), quat_to_euler_xyz(wrist_r)
    wrist = torch.stack(
        (sl[:, 0] + wl[:, 0], -(sr[:, 0] + wr[:, 0]), wl[:, 1], -wr[:, 1], sl[:, 2] + wl[:, 2], sr[:, 2] + wr[:, 2]),
        dim=-1,
    )
    return {"smpl_joints": smpl_joints, "root_quat": root_quat, "smpl_pose": smpl_pose, "wrist_joint_pos": wrist}


class PicoBodyRetargeter:
    """Stateful wrapper of :func:`body_to_smpl` that fills joints the headset did not track this sample."""

    def __init__(self, num_envs: int, skeleton: SmplSkeleton, device: str | torch.device = "cpu"):
        self.skeleton = skeleton
        self.last = torch.zeros(num_envs, NUM_BODY_JOINTS, 4, device=device)
        self.last[..., 0] = 1.0
        self.seen = torch.zeros(num_envs, NUM_BODY_JOINTS, dtype=torch.bool, device=device)
        self._parents = SMPL_PARENTS + THUMB_PARENTS               # the hands (22, 23) hang off the wrists

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self.seen[idx] = False
        self.last[idx] = 0.0
        self.last[idx, :, 0] = 1.0

    def __call__(
        self,
        quat_xyzw: torch.Tensor,
        valid: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> PicoSmplFrame:
        """``quat_xyzw (N, 24, 4)`` and ``valid (N, 24)`` as in ``FullBodyInput``; zero quaternions are invalid."""
        q = xyzw_to_wxyz(quat_xyzw.to(self.last))
        good = q.norm(dim=-1) > 1e-6
        if valid is not None:
            good &= valid.to(device=good.device).bool()
        self.last = torch.where(good.unsqueeze(-1), quat_normalize(q), self.last)
        self.seen |= good
        filled = self.last.clone()
        for i in range(1, NUM_BODY_JOINTS):                     # never seen: copy the parent (local identity)
            p = self._parents[i]
            filled[:, i] = torch.where(self.seen[:, i : i + 1], filled[:, i], filled[:, p])
        out = body_to_smpl(filled, self.skeleton)
        return PicoSmplFrame(
            ok=self.seen[:, 0].clone(),
            pelvis_pos=None if positions is None else positions[:, 0].to(self.last),
            **out,
        )
