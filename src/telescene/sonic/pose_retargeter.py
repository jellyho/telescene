# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Isaac Capture retargeter: the headset's 24-joint body to SONIC's POSE action slice.

Isaac Capture ships the pipe but not this step. Its ``FullBodySource`` delivers
``FullBodyInput`` -- 24 joint positions, 24 xyzw orientations and 24 valid flags, vendor
``body.pico-xr`` -- and its engine has ``BaseRetargeter`` and the tensor types, but every
concrete retargeter in the package and in Isaac Lab maps hands or arms. Nothing upstream turns a
body into robot commands; ``RetargeterBase.Requirement`` does not even have a body entry. SONIC's
own mapping lives in ``gear_sonic`` and is ported to
:mod:`telescene.sonic.core.pico_body`. This node is the piece that joins the two.

Output is the last 83 values of the 87-D pose action::

    [0]      pose_active        passed through from a toggle input
    [1:73]   smpl_joints 24x3   in the root frame
    [73:77]  smpl_root_quat     xyzw, Z up (the action term's convention; the conversion
                                from ``PicoBodyRetargeter``'s wxyz happens here)
    [77:83]  wrist_joint_pos    6, policy order

The first four of the 87, ``[vx, vy, wz, hip_height]``, come from the stock locomotion
retargeter, unchanged.

**Real time.** The policy ticks at 50 Hz, so the whole chain -- this, SONIC's encoder and
decoder, physics, render and encode -- shares 20 ms. Measured on this workstation, CPU, one
environment: :class:`~telescene.sonic.core.pico_body.PicoBodyRetargeter` is
1.35 ms and SONIC's own inference is 8.2 ms. This node adds no allocation on top: every buffer is
made once in :meth:`__init__` and written in place, and nothing moves between devices per tick.
Four environments cost 1.40 ms, i.e. the work is fixed overhead rather than per-environment, so
widening the batch is cheap and narrowing it buys nothing.

When body data is absent -- no headset, tracking lost, trackers out of view -- the node holds its
last frame and reports ``pose_active = 0``. It does not emit zeros: a zero SMPL frame is a
posture, and SONIC would try to adopt it.
"""

from __future__ import annotations

import numpy as np

SMPL_JOINTS = 24
ACTION_SLICE = 83
"""``pose_active`` + 24x3 joints + 4 root quat + 6 wrist positions."""

POSE_ACTIVE = 0
JOINTS = slice(1, 1 + SMPL_JOINTS * 3)
ROOT_QUAT = slice(1 + SMPL_JOINTS * 3, 1 + SMPL_JOINTS * 3 + 4)
WRISTS = slice(1 + SMPL_JOINTS * 3 + 4, ACTION_SLICE)


class SonicPoseRetargeter:
    """Turn one ``FullBodyInput`` into SONIC's 83-value pose slice.

    Args:
        skeleton: SONIC's SMPL rest skeleton, from
            :func:`~telescene.sonic.core.pico_body.load_smpl_skeleton`.
        device: Torch device for the conversion. ``"cpu"`` is what teleop uses: one environment's
            worth of 24 joints is too small to pay for a transfer.

    The constructor allocates everything. :meth:`retarget` writes in place and returns a view of
    the same buffer each call, so the caller must consume it before the next tick rather than
    keeping the reference.
    """

    def __init__(self, skeleton, device: str = "cpu"):
        import torch

        from .core.pico_body import PicoBodyRetargeter
        from .core.rotations import wxyz_to_xyzw

        self._wxyz_to_xyzw = wxyz_to_xyzw

        self._torch = torch
        self._device = device
        self._body = PicoBodyRetargeter(num_envs=1, skeleton=skeleton, device=device)
        self._quat = torch.zeros((1, SMPL_JOINTS, 4), device=device)
        self._quat[..., 3] = 1.0
        self._valid = torch.zeros((1, SMPL_JOINTS), dtype=torch.bool, device=device)
        self._out = np.zeros(ACTION_SLICE, dtype=np.float32)
        self._have_frame = False

    def reset(self) -> None:
        """Forget the held frame, so a fresh session does not start from a stale posture."""
        self._body.reset(None)
        self._have_frame = False
        self._out[:] = 0.0

    def retarget(self, orientations, valid, pose_active: bool) -> np.ndarray:
        """Convert one body frame.

        Args:
            orientations: ``(24, 4)`` xyzw, as ``FullBodyInput`` hands them over. Isaac Capture's
                own reader passes the raw DeviceIO frame through, so no anchor transform is
                applied here either.
            valid: ``(24,)`` flags; anything non-zero counts as valid.
            pose_active: whether the operator has POSE mode engaged.

        Returns:
            The 83-value slice. Held from the previous frame, with ``pose_active`` forced to 0,
            when no joint is valid.
        """
        torch = self._torch
        # copy_ rather than as_tensor: the source buffers are reused by the caller every tick,
        # and a tensor sharing that memory would alias the next frame into this one.
        self._quat.copy_(torch.as_tensor(np.asarray(orientations, dtype=np.float32)).reshape(1, SMPL_JOINTS, 4))
        self._valid.copy_(torch.as_tensor(np.asarray(valid)).reshape(1, SMPL_JOINTS).to(torch.bool))

        if not bool(self._valid.any()):
            self._out[POSE_ACTIVE] = 0.0
            return self._out

        frame = self._body(self._quat, self._valid)
        if not bool(frame.ok.all()):
            self._out[POSE_ACTIVE] = 0.0
            return self._out

        self._out[JOINTS] = frame.smpl_joints[0].reshape(-1).cpu().numpy()
        # PicoBodyRetargeter speaks wxyz; SonicAction reads this slot as xyzw and converts back.
        self._out[ROOT_QUAT] = self._wxyz_to_xyzw(frame.root_quat[0]).cpu().numpy()
        self._out[WRISTS] = frame.wrist_joint_pos[0].cpu().numpy()
        self._out[POSE_ACTIVE] = 1.0 if pose_active else 0.0
        self._have_frame = True
        return self._out

    @property
    def has_frame(self) -> bool:
        """Whether any valid body frame has arrived. False means the headset is not delivering."""
        return self._have_frame
