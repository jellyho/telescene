# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched encoder and decoder observation builders (ports of the deploy gatherers).

Gatherers mirror gear_sonic_deploy ``g1_deploy_onnx_ref.cpp``:

* ``GatherMotionJointPositionsMultiFrame`` / ``...VelocitiesMultiFrame`` (740-907): frames
  ``frame + k * step`` clamped to the motion end; the full-body variant replaces the 17 upper-body joints
  with external targets when given (PLANNER_FROZEN_UPPER);
* ``GatherMotionAnchorOrientationMutiFrame`` (616-700): ``rot6d(conj(left) * delta_heading * ref_root)`` with
  ``left`` the robot base quaternion (``base``), its heading (``heading``) or the first reference frame's
  heading (``refheading``);
* ``GatherMotionRootZPositionMultiFrame`` (427-455), ``GatherMotionSmplJointsMultiFrame`` (910-968),
  VR 3-point (1106-1270), ``GatherEncoderMode`` (1691-1697);
* history terms (1454-1640): oldest first; joint positions relative to the default pose; gravity
  ``R_base^T (0, 0, -1)``.

The reference motion read by the motion gatherers is the planner motion for encoder modes 0 (g1) and
1 (teleop) and the streamed SMPL motion for mode 2 (smpl), as the deploy reads its single current motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import rotations as rot
from .joint_order import (
    DEFAULT_ANGLES_POLICY,
    LOWER_BODY_MUJOCO_ORDER_IN_POLICY_INDEX,
    UPPER_BODY_POLICY_INDEX,
    WRIST_POLICY_INDEX,
)
from .motion import MotionBuffer
from .obs_config import SMPL_ELBOW_WRIST_POSES, SMPL_LOWER_JOINTS, ObsLayout, ObsSpec

SMPL_MODE = 2

PROPRIO_SLICES = {
    "base_angular_velocity": slice(0, 3),
    "body_joint_positions": slice(3, 32),
    "body_joint_velocities": slice(32, 61),
    "last_actions": slice(61, 90),
    "gravity_dir": slice(90, 93),
}
PROPRIO_DIM = 93
"""One proprio sample: gyro (3), joint pos - default (29), joint vel (29), last action (29), gravity (3);
the split used by sonic_policy.py:146-158."""


@dataclass
class EncoderContext:
    """Everything the encoder gatherers read, for one control tick.

    Attributes:
        mode: ``(E,)`` long encoder mode id per environment.
        base_quat: ``(E, 4)`` robot pelvis orientation, wxyz.
        planner_motion: Planner reference motion.
        planner_heading: ``(E, 4)`` heading alignment applied to planner root quaternions.
        smpl_motion: Streamed SMPL motion (or ``None``).
        smpl_heading: ``(E, 4)`` heading alignment applied to SMPL root quaternions.
        vr_position: ``(E, 9)`` VR 3-point positions (pelvis frame).
        vr_orientation: ``(E, 12)`` VR 3-point quaternions (wxyz, pelvis frame).
        vr_compliance: ``(E, 3)``.
        upper_body_joint_pos: ``(E, 17)`` external upper-body joint targets (policy order of
            ``UPPER_BODY_POLICY_INDEX``) or ``None``.
        upper_body_joint_vel: ``(E, 17)`` or ``None`` (zeros, as when only positions are streamed).
        upper_body_mask: ``(E,)`` bool, environments using the external upper body.
    """

    mode: torch.Tensor
    base_quat: torch.Tensor
    planner_motion: MotionBuffer
    planner_heading: torch.Tensor
    smpl_motion: MotionBuffer | None = None
    smpl_heading: torch.Tensor | None = None
    vr_position: torch.Tensor | None = None
    vr_orientation: torch.Tensor | None = None
    vr_compliance: torch.Tensor | None = None
    upper_body_joint_pos: torch.Tensor | None = None
    upper_body_joint_vel: torch.Tensor | None = None
    upper_body_mask: torch.Tensor | None = None


def _per_source(ctx: EncoderContext, fn) -> torch.Tensor:
    """Evaluate ``fn(motion, heading)`` on the planner motion, overriding SMPL-mode rows with the SMPL stream."""
    out = fn(ctx.planner_motion, ctx.planner_heading)
    smpl_rows = ctx.mode == SMPL_MODE
    if ctx.smpl_motion is not None and bool(smpl_rows.any()):
        heading = ctx.smpl_heading if ctx.smpl_heading is not None else ctx.planner_heading
        out = torch.where(smpl_rows.view(-1, *([1] * (out.dim() - 1))), fn(ctx.smpl_motion, heading), out)
    return out


def _joint_subset(spec: ObsSpec) -> torch.Tensor | None:
    if spec.subset == "lowerbody":
        return torch.as_tensor(LOWER_BODY_MUJOCO_ORDER_IN_POLICY_INDEX)
    if spec.subset == "wrists":
        return torch.as_tensor(WRIST_POLICY_INDEX)
    return None


def _motion_joints(ctx: EncoderContext, spec: ObsSpec, velocity: bool) -> torch.Tensor:
    def fn(motion: MotionBuffer, _heading):
        idx = motion.frame_indices(spec.num_frames, spec.step)
        values = MotionBuffer.gather(motion.joint_vel if velocity else motion.joint_pos, idx)
        subset = _joint_subset(spec)
        if subset is not None:
            return values[..., subset.to(values.device)]
        if ctx.upper_body_joint_pos is not None and ctx.upper_body_mask is not None:
            ub = ctx.upper_body_joint_vel if velocity else ctx.upper_body_joint_pos
            if ub is None:
                ub = torch.zeros_like(ctx.upper_body_joint_pos)
            replaced = values.clone()
            cols = torch.as_tensor(UPPER_BODY_POLICY_INDEX, device=values.device)
            replaced[..., cols] = ub.unsqueeze(1).expand(-1, spec.num_frames, -1)
            values = torch.where(ctx.upper_body_mask.view(-1, 1, 1), replaced, values)
        return values

    return _per_source(ctx, fn).flatten(1)


def _anchor_orientation(ctx: EncoderContext, spec: ObsSpec) -> torch.Tensor:
    def fn(motion: MotionBuffer, heading: torch.Tensor):
        idx = motion.frame_indices(spec.num_frames, spec.step)
        ref = MotionBuffer.gather(motion.root_quat, idx)  # (E, F, 4)
        ref = rot.quat_mul(heading.unsqueeze(1).expand_as(ref), ref)
        if spec.variant == "heading":
            left = rot.heading_quat(ctx.base_quat)
        elif spec.variant == "refheading":
            left = rot.heading_quat(ref[:, 0])
        else:
            left = ctx.base_quat
        rel = rot.quat_mul(rot.quat_conjugate(left).unsqueeze(1).expand_as(ref), ref)
        return rot.rot6d(rel)

    return _per_source(ctx, fn).flatten(1)


def _root_z(ctx: EncoderContext, spec: ObsSpec) -> torch.Tensor:
    def fn(motion: MotionBuffer, _heading):
        idx = motion.frame_indices(spec.num_frames, spec.step)
        return MotionBuffer.gather(motion.root_pos, idx)[..., 2]

    return _per_source(ctx, fn).flatten(1)


def _smpl(ctx: EncoderContext, spec: ObsSpec, field: str) -> torch.Tensor:
    num_envs = ctx.mode.shape[0]
    motion = ctx.smpl_motion
    if motion is None:
        return torch.zeros(num_envs, spec.dim, device=ctx.base_quat.device)
    idx = motion.frame_indices(spec.num_frames, spec.step)
    values = MotionBuffer.gather(getattr(motion, field), idx)  # (E, F, J, 3)
    if spec.subset == "lower":
        values = values[:, :, list(SMPL_LOWER_JOINTS)]
    elif spec.subset == "elbow_wrist":
        values = values[:, :, list(SMPL_ELBOW_WRIST_POSES)]
    return values.reshape(num_envs, -1)


def _heading_diff(ctx: EncoderContext) -> torch.Tensor:
    def fn(motion: MotionBuffer, heading: torch.Tensor):
        ref = rot.quat_mul(heading, motion.current(motion.root_quat))
        diff = rot.quat_mul(rot.heading_quat_inv(ctx.base_quat), rot.heading_quat(ref))
        return rot.rot6d(diff)

    return _per_source(ctx, fn)


def gather(ctx: EncoderContext, spec: ObsSpec) -> torch.Tensor:
    """Compute one encoder observation term, shape ``(E, spec.dim)``."""
    num_envs = ctx.mode.shape[0]
    device = ctx.base_quat.device
    kind = spec.kind
    if kind == "composite":
        return torch.cat([gather(ctx, p) for p in spec.parts], dim=-1)
    if kind == "encoder_mode":
        out = torch.zeros(num_envs, spec.dim, device=device)
        out[:, 0] = ctx.mode.to(out.dtype)
        return out
    if kind == "motion_joint_pos":
        return _motion_joints(ctx, spec, velocity=False)
    if kind == "motion_joint_vel":
        return _motion_joints(ctx, spec, velocity=True)
    if kind == "anchor_orientation":
        return _anchor_orientation(ctx, spec)
    if kind == "motion_root_z":
        return _root_z(ctx, spec)
    if kind == "smpl_joints":
        return _smpl(ctx, spec, "smpl_joints")
    if kind == "smpl_pose":
        return _smpl(ctx, spec, "smpl_pose")
    if kind == "heading_diff":
        return _heading_diff(ctx)
    if kind in ("vr_3point_pos", "vr_3point_orn", "vr_3point_compliance"):
        value = {"vr_3point_pos": ctx.vr_position, "vr_3point_orn": ctx.vr_orientation,
                 "vr_3point_compliance": ctx.vr_compliance}[kind]
        if value is None:
            return torch.zeros(num_envs, spec.dim, device=device)
        return value.reshape(num_envs, spec.dim).to(device)
    raise KeyError(f"Observation kind {kind!r} ({spec.name}) is not an encoder observation")


def build_encoder_obs(layout: ObsLayout, ctx: EncoderContext) -> torch.Tensor:
    """Concatenate the encoder input ``(E, layout.encoder_dim)``; terms not required by an env's mode are zero."""
    num_envs = ctx.mode.shape[0]
    out = torch.zeros(num_envs, layout.encoder_dim, device=ctx.base_quat.device)
    mode_ids = sorted(layout.modes)
    for slot in layout.encoder:
        if layout.modes:
            using = [m for m in mode_ids if slot.spec.name in layout.modes[m].required]
            rows = torch.isin(ctx.mode, torch.tensor(using, dtype=ctx.mode.dtype, device=ctx.mode.device))
            if not bool(rows.any()):
                continue
            out[:, slot.offset : slot.end] = gather(ctx, slot.spec) * rows.unsqueeze(-1)
        else:
            out[:, slot.offset : slot.end] = gather(ctx, slot.spec)
    return out


def proprio_sample(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_quat: torch.Tensor,
    base_ang_vel: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """One ``(E, 93)`` history sample from policy-order joint state (absolute rad) and the base IMU."""
    default = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=joint_pos.dtype, device=joint_pos.device)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=base_quat.dtype, device=base_quat.device).expand(base_quat.shape[0], 3)
    gravity = rot.quat_apply_inverse(base_quat, down)
    return torch.cat((base_ang_vel, joint_pos - default, joint_vel, last_action, gravity), dim=-1)


def build_decoder_obs(layout: ObsLayout, tokens: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Decoder input ``(E, layout.decoder_dim)`` from tokens and a ``(E, L, 93)`` oldest-first history."""
    num_envs = tokens.shape[0]
    out = torch.zeros(num_envs, layout.decoder_dim, device=tokens.device)
    length = history.shape[1]
    for slot in layout.decoder:
        spec = slot.spec
        if spec.kind == "token":
            out[:, slot.offset : slot.end] = tokens
        elif spec.kind == "proprio":
            span = (spec.num_frames - 1) * spec.step + 1
            window = history[:, length - span :: spec.step, PROPRIO_SLICES[spec.subset]]
            out[:, slot.offset : slot.end] = window.reshape(num_envs, -1)
        else:
            raise KeyError(f"Decoder observation {spec.name!r} is not supported (token and history terms only)")
    return out
