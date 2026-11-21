# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched 50 Hz reference motion and the planner's 30 Hz -> 50 Hz resampling and cross-fade.

Ported from gear_sonic_deploy ``localmotion_kplanner.hpp`` (``ResampleGeneratedSequence50Hz`` from line 395,
``UpdateContextFromMotion`` from line 628) and ``g1_deploy_onnx_ref.cpp`` ``CurrentFrameAdvancement`` (3184-3300),
cross-checked with IsaacLab-demos ``sonic_policy.py`` ``_sample_motion``/``_plan``/``_replan`` (30-112).
"""

from __future__ import annotations

import torch

from . import rotations as rot
from .joint_order import ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB, NUM_BODY_JOINTS
from .obs_config import NUM_SMPL_JOINTS, NUM_SMPL_POSES

PLANNER_FPS = 30
CONTROL_FPS = 50
BLEND_FRAMES = 8
"""Cross-fade width (g1_deploy_onnx_ref.cpp:3248)."""


class MotionBuffer:
    """Per-environment reference motion with a playback cursor.

    Joint tensors are in SONIC policy order. Frames past ``length - 1`` are clamped on read.

    Args:
        num_envs: Batch size.
        capacity: Maximum number of 50 Hz frames.
        device: Torch device.
        with_smpl: Also store SMPL joints (24 x 3) and poses (21 x 3).
    """

    def __init__(self, num_envs: int, capacity: int, device: str | torch.device = "cpu", with_smpl: bool = False):
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.device(device)
        self.root_pos = torch.zeros(num_envs, capacity, 3, device=device)
        self.root_quat = torch.zeros(num_envs, capacity, 4, device=device)
        self.root_quat[..., 0] = 1.0
        self.joint_pos = torch.zeros(num_envs, capacity, NUM_BODY_JOINTS, device=device)
        self.joint_vel = torch.zeros(num_envs, capacity, NUM_BODY_JOINTS, device=device)
        self.smpl_joints = torch.zeros(num_envs, capacity, NUM_SMPL_JOINTS, 3, device=device) if with_smpl else None
        self.smpl_pose = torch.zeros(num_envs, capacity, NUM_SMPL_POSES, 3, device=device) if with_smpl else None
        self.length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frame = torch.zeros(num_envs, dtype=torch.long, device=device)

    def frame_indices(self, num_frames: int, step: int, offset: int = 0) -> torch.Tensor:
        """Indices ``frame + offset + k * step`` clamped to the last valid frame, shape ``(num_envs, num_frames)``."""
        k = torch.arange(num_frames, device=self.device) * step
        idx = self.frame.unsqueeze(1) + offset + k.unsqueeze(0)
        return torch.minimum(idx, (self.length - 1).clamp_min(0).unsqueeze(1))

    @staticmethod
    def gather(field: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather ``field[env, idx[env, k]]`` -> ``(num_envs, K, ...)``."""
        expand = idx.reshape(*idx.shape, *([1] * (field.dim() - 2))).expand(*idx.shape, *field.shape[2:])
        return torch.gather(field, 1, expand)

    def current(self, field: torch.Tensor) -> torch.Tensor:
        return self.gather(field, self.frame_indices(1, 1))[:, 0]

    def write(self, env_ids: torch.Tensor, root_pos, root_quat, joint_pos, joint_vel, length: torch.Tensor):
        """Overwrite the motions of ``env_ids`` (inputs are ``(len(env_ids), T, ...)`` with ``T <= capacity``)."""
        t = root_pos.shape[1]
        if t > self.capacity:
            raise ValueError(f"motion of {t} frames exceeds buffer capacity {self.capacity}")
        self.root_pos[env_ids, :t] = root_pos
        self.root_quat[env_ids, :t] = root_quat
        self.joint_pos[env_ids, :t] = joint_pos
        self.joint_vel[env_ids, :t] = joint_vel
        self.length[env_ids] = length


def resample_planner_output(qpos: torch.Tensor, count: torch.Tensor):
    """Planner ``mujoco_qpos`` (B, F30, 36) at 30 Hz -> 50 Hz root pose and policy-order joints.

    ``T50 = floor(count * 50 / 30)`` frames; joint velocities are forward differences x 50 with the last frame
    repeated. Frames past ``T50`` repeat the last valid frame.

    Returns:
        ``(root_pos (B,T,3), root_quat_wxyz (B,T,4), joint_pos (B,T,29), joint_vel (B,T,29), length (B,))``
        with ``T = max(T50)``.
    """
    count = count.to(torch.long).clamp(min=2, max=qpos.shape[1])
    length = (count * CONTROL_FPS) // PLANNER_FPS
    t_max = int(length.max())
    f = torch.arange(t_max, device=qpos.device)
    f = torch.minimum(f.unsqueeze(0), (length - 1).unsqueeze(1))  # (B, T)
    # 30 Hz sample time f * 30/50 in exact integer arithmetic
    f0 = (f * PLANNER_FPS) // CONTROL_FPS
    f1 = torch.minimum(f0 + 1, (count - 1).unsqueeze(1))
    w = (((f * PLANNER_FPS) % CONTROL_FPS).to(qpos.dtype) / CONTROL_FPS).unsqueeze(-1)
    q0 = MotionBuffer.gather(qpos, f0)
    q1 = MotionBuffer.gather(qpos, f1)
    lin = (1.0 - w) * q0 + w * q1
    quat = rot.quat_slerp(q0[..., 3:7], q1[..., 3:7], w)
    joints = lin[..., 7:][..., torch.as_tensor(MUJOCO_TO_ISAACLAB, device=qpos.device)]
    vel = torch.zeros_like(joints)
    vel[:, :-1] = (joints[:, 1:] - joints[:, :-1]) * CONTROL_FPS
    last = (length - 1).clamp_min(1)
    rows = torch.arange(qpos.shape[0], device=qpos.device)
    # last valid frame repeats the previous velocity (ResampleGeneratedSequence50Hz)
    vel[rows, length - 1] = vel[rows, last - 1]
    tail = torch.arange(t_max, device=qpos.device).unsqueeze(0) >= length.unsqueeze(1)
    vel = torch.where(tail.unsqueeze(-1), vel[rows, length - 1].unsqueeze(1).expand_as(vel), vel)
    return lin[..., 0:3], quat, joints, vel, length


def sample_context(motion: MotionBuffer, env_ids: torch.Tensor, look_ahead: int = 2) -> torch.Tensor:
    """Planner context: 4 frames at 30 Hz starting ``look_ahead`` ticks ahead, as MuJoCo qpos ``(n, 4, 36)``."""
    device = motion.device
    # 50 Hz sample time frame + look_ahead + n * 50/30 in exact integer arithmetic
    n50 = torch.arange(4, device=device) * CONTROL_FPS
    base = (motion.frame[env_ids] + look_ahead).unsqueeze(1)
    f0 = base + n50 // PLANNER_FPS
    w = ((n50 % PLANNER_FPS).to(torch.float32) / PLANNER_FPS).expand(f0.shape).clone()
    last = (motion.length[env_ids] - 1).clamp_min(0).unsqueeze(1)
    past_end = f0 >= last
    f0 = torch.minimum(f0, last)
    w[past_end] = 0.0
    f1 = torch.minimum(f0 + 1, last)
    w = w.unsqueeze(-1)

    def lerp(field):
        a = MotionBuffer.gather(field[env_ids], f0)
        b = MotionBuffer.gather(field[env_ids], f1)
        return a, b

    p0, p1 = lerp(motion.root_pos)
    q0, q1 = lerp(motion.root_quat)
    j0, j1 = lerp(motion.joint_pos)
    pos = (1.0 - w) * p0 + w * p1
    quat = rot.quat_slerp(q0, q1, w)
    joints = ((1.0 - w) * j0 + w * j1)[..., torch.as_tensor(ISAACLAB_TO_MUJOCO, device=device)]
    return torch.cat((pos, quat, joints), dim=-1)


def blend_into(motion: MotionBuffer, env_ids: torch.Tensor, new: tuple, gen_offset: int):
    """Cross-fade a new plan into the current motion and rebase so the current frame becomes frame 0.

    ``gen_offset`` is ``gen_frame - current_frame`` (the planner look-ahead, 2 when planning synchronously).
    New length is ``gen_offset + T_new``; the new plan's weight ramps linearly over :data:`BLEND_FRAMES`
    frames starting at ``gen_offset``. A negative ``gen_offset`` is a plan that arrived late: its frames that
    are already past are skipped and the fade starts part way through.
    """
    root_pos, root_quat, joint_pos, joint_vel, new_len = new
    device = motion.device
    n = env_ids.numel()
    out_len = new_len + gen_offset
    t_out = int(out_len.max())
    if t_out > motion.capacity:
        raise ValueError(f"blended motion of {t_out} frames exceeds buffer capacity {motion.capacity}")
    f = torch.arange(t_out, device=device).unsqueeze(0).expand(n, -1)
    old_last = (motion.length[env_ids] - 1).clamp_min(0).unsqueeze(1)
    f_old = torch.minimum(f + motion.frame[env_ids].unsqueeze(1), old_last)
    f_new = (f - gen_offset).clamp(min=0)
    f_new = torch.minimum(f_new, (new_len - 1).unsqueeze(1))
    w = ((f - gen_offset).to(torch.float32) / BLEND_FRAMES).clamp(0.0, 1.0).unsqueeze(-1)

    def mix(old_field, new_field):
        a = MotionBuffer.gather(old_field[env_ids], f_old)
        b = MotionBuffer.gather(new_field, f_new)
        return a, b

    a, b = mix(motion.root_pos, root_pos)
    pos = (1 - w) * a + w * b
    a, b = mix(motion.root_quat, root_quat)
    quat = rot.quat_slerp(a, b, w)
    a, b = mix(motion.joint_pos, joint_pos)
    jp = (1 - w) * a + w * b
    a, b = mix(motion.joint_vel, joint_vel)
    jv = (1 - w) * a + w * b
    motion.write(env_ids, pos, quat, jp, jv, out_len)
    motion.frame[env_ids] = 0
