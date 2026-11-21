# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Streamed SMPL reference for SONIC's POSE (SMPL encoder) mode.

The deploy plays streamed frames as a motion and the SMPL encoder reads ``smpl_joints_10frame_step1`` etc.
from the current frame onwards (g1_deploy_onnx_ref.cpp:910-968, 616-700). With live input there are no
future frames, so the window is the last ``window`` received frames, oldest first (latency
``(window - 1) / 50`` s; 180 ms for the default model, 60 ms for the 4-frame low-latency model).

Frame content (pico_manager_thread_server.py ``process_smpl_joints`` 450-489 and zmq.md protocol v3):

* ``smpl_joints``: ``(24, 3)`` SMPL joint positions in the SMPL root frame (root rotation removed);
* ``root_quat``: ``(4,)`` SMPL root (pelvis) orientation, wxyz, Z-up world (yaw is re-aligned to the robot
  heading when streaming starts);
* ``wrist_joint_pos``: ``(6,)`` wrist joints in policy order ``[l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]``
  (policy indices 23-28).
"""

from __future__ import annotations

import torch

from .joint_order import WRIST_POLICY_INDEX
from .motion import MotionBuffer


class SmplStream:
    """Ring of the last ``window`` SMPL frames stored as a :class:`MotionBuffer` (cursor fixed at 0)."""

    def __init__(self, num_envs: int, window: int, device: str | torch.device = "cpu"):
        self.window = window
        self.motion = MotionBuffer(num_envs, window, device, with_smpl=True)
        self.motion.length[:] = window
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._wrist = torch.as_tensor(WRIST_POLICY_INDEX, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self.count[idx] = 0

    def push(
        self,
        smpl_joints: torch.Tensor,
        root_quat: torch.Tensor,
        wrist_joint_pos: torch.Tensor | None = None,
        smpl_pose: torch.Tensor | None = None,
        root_height: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Append one frame for the environments in ``mask`` (default all)."""
        m = self.motion
        n = smpl_joints.shape[0]
        mask = torch.ones(n, dtype=torch.bool, device=smpl_joints.device) if mask is None else mask
        if not bool(mask.any()):
            return
        joint_pos = torch.zeros(n, m.joint_pos.shape[-1], device=smpl_joints.device)
        if wrist_joint_pos is not None:
            joint_pos[:, self._wrist] = wrist_joint_pos
        root_pos = torch.zeros(n, 3, device=smpl_joints.device)
        if root_height is not None:
            root_pos[:, 2] = root_height
        new = {
            "smpl_joints": smpl_joints.reshape(n, -1, 3),
            "root_quat": root_quat,
            "joint_pos": joint_pos,
            "root_pos": root_pos,
            "smpl_pose": smpl_pose.reshape(n, -1, 3) if smpl_pose is not None else torch.zeros_like(m.smpl_pose[:, 0]),
        }
        first = mask & (self.count == 0)
        rest = mask & ~first
        for name, value in new.items():
            field = getattr(m, name)
            if bool(rest.any()):
                field[rest] = torch.roll(field[rest], shifts=-1, dims=1)
                field[rest, -1] = value[rest]
            if bool(first.any()):
                field[first] = value[first].unsqueeze(1).expand(-1, self.window, *value.shape[1:])
        m.joint_vel[mask] = 0.0
        self.count[mask] += 1

    @property
    def started(self) -> torch.Tensor:
        """Environments that have received at least one frame."""
        return self.count > 0
