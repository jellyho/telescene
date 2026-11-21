# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Headset controller values to the hand and locomotion columns of the SONIC actions.

The formulas are those of Isaac Lab's stock teleop retargeters
(``TriHandMotionControllerRetargeter._map_to_hand_joints`` and ``LocomotionRootCmdRetargeter._compute_fn``), so a
take means the same thing as one recorded through Isaac Lab's own XR teleop.
"""

from __future__ import annotations

import numpy as np

HAND_JOINT_NAMES = ("thumb_rotation", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal",
                    "middle_proximal", "middle_distal")
"""TriHand's 7-DoF order in Isaac Lab's stock retargeter: generic names, mapped to the robot by HAND_OUTPUT_ORDER."""

HAND_OUTPUT_ORDER = (
    "l_index_proximal", "l_middle_proximal", "l_thumb_rotation", "r_index_proximal", "r_middle_proximal", "r_thumb_rotation",
    "l_index_distal", "l_middle_distal", "l_thumb_proximal", "r_index_distal", "r_middle_distal", "r_thumb_proximal",
    "l_thumb_distal", "r_thumb_distal",
)
"""The 14 hand action columns, in the order of ``TRIHAND_JOINT_NAMES`` (the ``hands`` action term)."""



THUMB_ROTATION_TRIGGER_SCALE = 0.5
THUMB_ROTATION_SQUEEZE_SCALE = 0.5
THUMB_PROXIMAL_SCALE = 0.4
THUMB_DISTAL_SCALE = 0.7

MOVEMENT_SCALE = 0.5
HIP_RATE = 0.35
HIP_LIMITS = (0.4, 1.0)
INITIAL_HIP = 0.72

_COLUMNS = [(name[0], HAND_JOINT_NAMES.index(name[2:])) for name in HAND_OUTPUT_ORDER]


def trihand(trigger: float, squeeze: float, left: bool) -> np.ndarray:
    """Seven hand joints in ``HAND_JOINT_NAMES`` order."""
    thumb = -max(trigger, squeeze)
    rotation = THUMB_ROTATION_TRIGGER_SCALE * trigger - THUMB_ROTATION_SQUEEZE_SCALE * squeeze
    if not left:
        rotation = -rotation
    joints = np.array([rotation, thumb * THUMB_PROXIMAL_SCALE, thumb * THUMB_DISTAL_SCALE,
                       trigger, trigger, squeeze, squeeze], dtype=np.float32)
    return -joints if left else joints


def hand_columns(left_trigger: float, left_grip: float, right_trigger: float, right_grip: float) -> np.ndarray:
    """The 14 hand action columns, in ``HAND_OUTPUT_ORDER``."""
    sides = {"l": trihand(left_trigger, left_grip, True), "r": trihand(right_trigger, right_grip, False)}
    return np.array([sides[side][index] for side, index in _COLUMNS], dtype=np.float32)


class Locomotion:
    """``[vx, vy, wz, hip_height]`` from the sticks. Only the hip height has state."""

    def __init__(self) -> None:
        self.hip = INITIAL_HIP

    def reset(self) -> None:
        self.hip = INITIAL_HIP

    def __call__(self, left_axis, right_axis, dt: float) -> np.ndarray:
        self.hip = float(np.clip(self.hip + right_axis[1] * dt * HIP_RATE, *HIP_LIMITS))
        return np.array([left_axis[1] * MOVEMENT_SCALE, -left_axis[0] * MOVEMENT_SCALE, -right_axis[0], self.hip],
                        dtype=np.float32)


class EdgeToggle:
    """Flip on a rising edge: holding a button must not flicker POSE mode."""

    def __init__(self, on: bool = True) -> None:
        self.on, self._held = on, False

    def __call__(self, pressed: bool) -> bool:
        if pressed and not self._held:
            self.on = not self.on
        self._held = pressed
        return self.on



def body_arrays(poses) -> tuple[np.ndarray, np.ndarray]:
    """``(24, 7)`` x,y,z,qx,qy,qz,qw into ``(24, 4)`` xyzw and validity flags.

    The SDK has no per-joint flag, and an unset joint is all zeros, whose quaternion is not a
    rotation. Validity is therefore a usable quaternion norm.
    """
    array = np.asarray(poses, dtype=np.float32).reshape(24, 7)
    quat = array[:, 3:7]
    return quat, np.abs(np.linalg.norm(quat, axis=1) - 1.0) < 0.1
