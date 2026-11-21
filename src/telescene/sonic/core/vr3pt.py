# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""VR 3-point (left wrist, right wrist, head) targets for SONIC's teleop encoder.

Target definition (what the encoder was trained on), gear_sonic ``envs/manager_env/mdp/observations.py``
``vr_3point_local_target`` / ``vr_3point_local_orn_target`` (1348-1442) and ``commands.py:2229-2300``, with
bodies and offsets from ``config/manager_env/commands/terms/motion.yaml:47-49``:

* point ``i`` = ``body_pos + R_body * offset_i`` for bodies ``left_wrist_yaw_link`` (offset ``(0.18, -0.025, 0)``),
  ``right_wrist_yaw_link`` (``(0.18, 0.025, 0)``) and ``torso_link`` (``(0, 0, 0.35)``);
* orientation ``i`` = body orientation;
* both expressed in the (reference) **pelvis** frame: ``p = R_pelvis^T (p_w - p_pelvis)``,
  ``q = q_pelvis^-1 * q_w``; quaternions (w, x, y, z); layout ``[lw xyz, rw xyz, head xyz]`` and
  ``[lw wxyz, rw wxyz, head wxyz]``.

From a Quest (head + 2 controllers, no pelvis tracking) we build the same quantities:

* wrists: the wrist-link target poses that Isaac Lab's stock ``Se3AbsRetargeter`` already produces for Pink IK
  (sim world frame), expressed in the robot's measured pelvis frame, plus the trained offsets;
* head: the HMD orientation is mapped to robot head axes, re-zeroed at calibration and used as the torso
  orientation; its position follows the PICO manager's kinematic chain
  ``(0, 0, 0.05) + 0.35 * z_axis(q)`` (pico_manager_thread_server.py:955-957, 1201-1205).

Optional wrist calibration follows ``ThreePointPose._capture_calibration`` (pico_manager_thread_server.py:1098-1207):
on enable, offsets are captured so the target equals the robot's current wrist points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import rotations as rot

WRIST_OFFSETS = ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0))
HEAD_OFFSET = (0.0, 0.0, 0.35)
TORSO_LINK_OFFSET_Z = 0.05
NECK_LINK_LENGTH = 0.35

DEFAULT_POSITIONS = (0.0903, 0.1615, -0.2411, 0.1280, -0.1522, -0.2461, 0.0241, -0.0081, 0.4028)
"""Fallback targets when no VR data is available (input_interface.hpp:177-189)."""
DEFAULT_ORIENTATIONS = (
    0.7295, 0.3145, 0.5533, -0.2506, 0.7320, -0.2639, 0.5395, 0.3217, 0.9991, 0.011, 0.0402, -0.0002,
)  # fmt: skip
"""Fallback orientations, wxyz (input_interface.hpp:198-206)."""

HMD_TO_HEAD_WXYZ = (0.5, -0.5, 0.5, 0.5)
"""Rotation from robot head axes (x fwd, y left, z up) to the OpenXR view axes the HMD pose uses
(x right, y up, -z forward): ``q_head = q_hmd * HMD_TO_HEAD``.

The HMD keeps OpenXR body axes after ``HeadSource.transformed(world_T_anchor)``: Isaac Capture's ``HeadTransform``
left-multiplies (``q' = R * q``, ``utilities/head_transform.py``) and Isaac Lab composes the OpenXR->USD basis
change into that ``R`` (``xr_anchor_manager.py`` ``_OXR_TO_USD_ROTATION``, ``_build_matrix``). Checked in the
sources only, not on a Quest; calibration removes a constant yaw/tilt offset but not a wrong axis mapping."""


@dataclass
class Vr3ptCfg:
    """Options for building VR 3-point targets."""

    root_frame: str = "full"
    """``"full"``: pelvis position + orientation (training definition). ``"heading"``: pelvis position + yaw."""
    head_mode: str = "hmd"
    """``"hmd"``: torso orientation follows the calibrated HMD. ``"neutral"``: upright torso (head = default)."""
    hmd_axes_wxyz: tuple[float, float, float, float] = HMD_TO_HEAD_WXYZ
    calibrate_head: bool = True
    """Re-zero the HMD orientation when VR control is enabled (PICO ``CALIB_FULL`` neck step)."""
    calibrate_wrists: bool = False
    """Capture wrist offsets against the robot's current wrists on enable (PICO ``CALIB``)."""


class Vr3ptBuilder:
    """Batched builder for ``vr_3point_local_target`` (9) and ``vr_3point_local_orn_target`` (12).

    All poses are ``(E, 7)`` ``[x, y, z, qw, qx, qy, qz]`` in one common frame (sim world, or already the pelvis
    frame with an identity root pose).
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", cfg: Vr3ptCfg | None = None):
        self.cfg = cfg or Vr3ptCfg()
        if self.cfg.root_frame not in ("full", "heading"):
            raise ValueError(f"root_frame must be 'full' or 'heading', got {self.cfg.root_frame!r}")
        if self.cfg.head_mode not in ("hmd", "neutral"):
            raise ValueError(f"head_mode must be 'hmd' or 'neutral', got {self.cfg.head_mode!r}")
        self.device = torch.device(device)
        self.wrist_offsets = torch.tensor(WRIST_OFFSETS, device=device)
        self.hmd_axes = torch.tensor(self.cfg.hmd_axes_wxyz, device=device)
        self.head_cal = torch.zeros(num_envs, 4, device=device)
        self.head_cal[:, 0] = 1.0
        self.wrist_pos_offset = torch.zeros(num_envs, 2, 3, device=device)
        self.wrist_rot_offset = torch.zeros(num_envs, 2, 4, device=device)
        self.wrist_rot_offset[..., 0] = 1.0
        self.default_pos = torch.tensor(DEFAULT_POSITIONS, device=device).view(3, 3)
        self.default_orn = torch.tensor(DEFAULT_ORIENTATIONS, device=device).view(3, 4)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self.head_cal[idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.wrist_pos_offset[idx] = 0.0
        self.wrist_rot_offset[idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    def _root(self, root_quat: torch.Tensor) -> torch.Tensor:
        return rot.heading_quat(root_quat) if self.cfg.root_frame == "heading" else root_quat

    def to_root(self, pose: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """World pose(s) ``(E, ..., 7)`` -> position and quaternion in the root frame."""
        q_root = self._root(root_quat)
        shape = pose.shape[:-1]
        q_root_b = q_root.reshape(q_root.shape[0], *([1] * (len(shape) - 1)), 4).expand(*shape, 4)
        p_root_b = root_pos.reshape(root_pos.shape[0], *([1] * (len(shape) - 1)), 3).expand(*shape, 3)
        pos = rot.quat_apply_inverse(q_root_b, pose[..., 0:3] - p_root_b)
        quat = rot.quat_mul(rot.quat_conjugate(q_root_b), rot.quat_normalize(pose[..., 3:7]))
        return pos, quat

    def wrist_points(self, wrist_pose: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """Wrist-link poses ``(E, 2, 7)`` -> 3-point wrist positions/orientations in the root frame."""
        pos, quat = self.to_root(wrist_pose, root_pos, root_quat)
        offsets = self.wrist_offsets.unsqueeze(0).expand_as(pos)
        return pos + rot.quat_apply(quat, offsets), quat

    def head_orientation(self, head_pose: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """HMD pose ``(E, 7)`` -> head (torso) orientation in robot axes and the root frame, uncalibrated."""
        _, q = self.to_root(head_pose, root_pos, root_quat)
        return rot.quat_mul(q, self.hmd_axes.expand_as(q))

    def calibrate(
        self,
        env_ids: torch.Tensor,
        wrist_pose: torch.Tensor,
        head_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        robot_wrist_pose: torch.Tensor | None = None,
    ):
        """Capture head (and optionally wrist) calibration for ``env_ids``. Inputs are full-batch tensors."""
        if env_ids.numel() == 0:
            return
        if self.cfg.calibrate_head:
            q = self.head_orientation(head_pose, root_pos, root_quat)
            self.head_cal[env_ids] = rot.quat_normalize(q[env_ids])
        if self.cfg.calibrate_wrists and robot_wrist_pose is not None:
            p_t, q_t = self.wrist_points(wrist_pose, root_pos, root_quat)
            p_r, q_r = self.wrist_points(robot_wrist_pose, root_pos, root_quat)
            self.wrist_pos_offset[env_ids] = (p_t - p_r)[env_ids]
            self.wrist_rot_offset[env_ids] = rot.quat_mul(q_r, rot.quat_conjugate(q_t))[env_ids]

    def __call__(
        self,
        wrist_pose: torch.Tensor,
        head_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build targets.

        Args:
            wrist_pose: ``(E, 2, 7)`` left/right wrist-link target poses, wxyz.
            head_pose: ``(E, 7)`` HMD pose, wxyz.
            root_pos: ``(E, 3)`` pelvis position in the same frame.
            root_quat: ``(E, 4)`` pelvis orientation, wxyz.

        Returns:
            ``positions (E, 9)`` and ``orientations (E, 12)``; rows with an invalid (zero) quaternion fall back to
            the deploy defaults.
        """
        n = wrist_pose.shape[0]
        p_w, q_w = self.wrist_points(wrist_pose, root_pos, root_quat)
        p_w = p_w - self.wrist_pos_offset
        q_w = rot.quat_mul(self.wrist_rot_offset, q_w)

        if self.cfg.head_mode == "hmd":
            q_h = self.head_orientation(head_pose, root_pos, root_quat)
            q_h = rot.quat_normalize(rot.quat_mul(rot.quat_conjugate(self.head_cal), q_h))
        else:
            q_h = torch.zeros(n, 4, device=wrist_pose.device)
            q_h[:, 0] = 1.0
        z = rot.quat_apply(q_h, torch.tensor([0.0, 0.0, 1.0], device=q_h.device).expand(n, 3))
        p_h = torch.tensor([0.0, 0.0, TORSO_LINK_OFFSET_Z], device=q_h.device) + NECK_LINK_LENGTH * z

        pos = torch.cat((p_w, p_h.unsqueeze(1)), dim=1)
        orn = torch.cat((q_w, q_h.unsqueeze(1)), dim=1)
        valid = torch.cat(
            (wrist_pose[..., 3:7].norm(dim=-1) > 1e-6, (head_pose[..., 3:7].norm(dim=-1) > 1e-6).unsqueeze(1)), dim=1
        )
        if self.cfg.head_mode == "neutral":
            valid[:, 2] = True
        pos = torch.where(valid.unsqueeze(-1), pos, self.default_pos.expand_as(pos))
        orn = torch.where(valid.unsqueeze(-1), orn, self.default_orn.expand_as(orn))
        return pos.reshape(n, 9), orn.reshape(n, 12)
