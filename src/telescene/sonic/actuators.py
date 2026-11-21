# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Actuator configs with SONIC's training gains for the 29 G1 body joints (Isaac Lab v3.0.0).

SONIC was trained with implicit PD gains ``kp = armature * (2 pi 10 Hz)^2``, ``kd = 4 * armature * 2 pi 10 Hz``
(doubled for ankles and waist roll/pitch), the matching armature and these effort/velocity limits
(gear_sonic ``envs/manager_env/robots/g1.py:239-357``). The stock ``G1_29DOF_CFG`` uses very different gains
(e.g. arms 3000, waist 5000), so the robot must be re-configured before SONIC drives it. Hand actuator groups
are kept as they are.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import replace

from .core.joint_order import MUJOCO_JOINT_NAMES, gains_by_name

_GROUPS = {
    "legs": ("hip_pitch", "hip_roll", "hip_yaw", "knee"),
    "feet": ("ankle_pitch", "ankle_roll"),
    "waist": ("waist_yaw", "waist_roll", "waist_pitch"),
    "arms": ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"),
}

SONIC_INIT_JOINT_POS: dict[str, float] = {n: g["default_angle"] for n, g in gains_by_name().items()}
"""SONIC's default standing pose, usable as ``ArticulationCfg.InitialStateCfg.joint_pos``."""

SONIC_INIT_ROOT_HEIGHT = 0.76
"""Initial pelvis height of SONIC training (g1.py:224)."""


def sonic_body_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Implicit actuator groups ``legs``, ``feet``, ``waist``, ``arms`` with exact per-joint SONIC values."""
    gains = gains_by_name()
    groups: dict[str, ImplicitActuatorCfg] = {}
    for group, keys in _GROUPS.items():
        names = [n for n in MUJOCO_JOINT_NAMES if any(n.endswith(f"{k}_joint") for k in keys)]
        groups[group] = ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness={n: gains[n]["stiffness"] for n in names},
            damping={n: gains[n]["damping"] for n in names},
            armature={n: gains[n]["armature"] for n in names},
            joint_effort_limit={n: gains[n]["effort_limit"] for n in names},
            joint_velocity_limit={n: gains[n]["velocity_limit"] for n in names},
        )
    assert sum(len(g.joint_names_expr) for g in groups.values()) == len(MUJOCO_JOINT_NAMES)
    return groups


def with_sonic_actuators(robot_cfg: ArticulationCfg, keep: tuple[str, ...] = ("hands",)) -> ArticulationCfg:
    """Copy of ``robot_cfg`` whose body actuators are replaced by :func:`sonic_body_actuators`.

    Args:
        robot_cfg: e.g. ``G1_29DOF_CFG`` (groups ``legs``, ``feet``, ``waist``, ``arms``, ``hands``).
        keep: Actuator groups to keep unchanged (the hand groups).
    """
    actuators = {k: v for k, v in robot_cfg.actuators.items() if k in keep}
    actuators.update(sonic_body_actuators())
    return replace(robot_cfg, actuators=actuators)
