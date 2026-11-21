# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""G1 joint orders, motor constants and PD gains of the released SONIC policy.

Two 29-joint orders exist upstream:

* **MuJoCo order** (URDF tree order): planner ``mujoco_qpos[7:]``, reference motions, the real robot SDK.
  Source: ``JOINT_NAMES`` in IsaacLab-demos ``contrib/g1_terrain/sonic_parameters.py:10-30`` and
  ``G1_MUJOCO_ORDER`` in gear_sonic ``envs/manager_env/mdp/observations.py:32-62``.
* **SONIC "IsaacLab" order** (the policy order): every encoder/decoder joint tensor and the decoder
  output. It is the BFS order of SONIC's *training* asset (hands fixed), **not** the order of the
  ``g1.usd`` used in this repo, which interleaves hand joints. Source: ``G1_ISAACLab_ORDER`` in
  gear_sonic ``envs/env_utils/joint_utils.py:10-40`` and ``POLICY_JOINT_NAMES`` in
  ``sonic_parameters.py:33-63``.

Always map to the simulator **by joint name** (:class:`JointMap`); never assume an index order.

Index tables are copied from gear_sonic_deploy ``src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``
(lines 68-104) and checked against the name lists in ``tests/sonic/test_joint_order.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

NUM_BODY_JOINTS = 29

MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""29 body joints in MuJoCo/URDF order."""

POLICY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
"""29 body joints in SONIC policy ("IsaacLab") order."""

# policy_parameters.hpp:100-104. ISAACLAB_TO_MUJOCO[m] = policy index of MuJoCo joint m;
# MUJOCO_TO_ISAACLAB[p] = MuJoCo index of policy joint p.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

# Joint subsets as policy-order indices (policy_parameters.hpp:76-97). Lower body is listed in MuJoCo order
# (left leg then right leg): this is the order of the teleop encoder's lower-body inputs.
LOWER_BODY_MUJOCO_ORDER_IN_POLICY_INDEX = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
LOWER_BODY_POLICY_INDEX = np.array([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], dtype=np.int64)
UPPER_BODY_POLICY_INDEX = np.array([2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int64)
WRIST_POLICY_INDEX = np.array([23, 24, 25, 26, 27, 28], dtype=np.int64)

# Motor constants (policy_parameters.hpp:39-65; gear_sonic envs/manager_env/robots/g1.py:10-27).
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0

_MOTOR = {
    "5020": (ARMATURE_5020, EFFORT_5020),
    "7520_14": (ARMATURE_7520_14, EFFORT_7520_14),
    "7520_22": (ARMATURE_7520_22, EFFORT_7520_22),
    "4010": (ARMATURE_4010, EFFORT_4010),
}
# (motor type, gain multiplier) per MuJoCo joint; the multiplier doubles kp/kd/armature for the dual-motor
# ankles and waist roll/pitch (policy_parameters.hpp:143-207, g1.py:277-291).
_MUJOCO_MOTORS = (
    [("7520_22", 1), ("7520_22", 1), ("7520_14", 1), ("7520_22", 1), ("5020", 2), ("5020", 2)] * 2
    + [("7520_14", 1), ("5020", 2), ("5020", 2)]
    + ([("5020", 1)] * 5 + [("4010", 1)] * 2) * 2
)


def _per_joint(fn) -> np.ndarray:
    return np.array([fn(_MOTOR[m], k) for m, k in _MUJOCO_MOTORS], dtype=np.float64)


ARMATURE_MUJOCO = _per_joint(lambda mo, k: k * mo[0])
"""Joint armature used in SONIC training (g1.py actuators), MuJoCo order."""
KP_MUJOCO = _per_joint(lambda mo, k: k * mo[0] * NATURAL_FREQ**2)
"""PD stiffness (policy_parameters.hpp ``kps``), MuJoCo order."""
KD_MUJOCO = _per_joint(lambda mo, k: k * 2.0 * DAMPING_RATIO * mo[0] * NATURAL_FREQ)
"""PD damping (policy_parameters.hpp ``kds``), MuJoCo order."""
ACTION_SCALE_MUJOCO = _per_joint(lambda mo, k: 0.25 * mo[1] / (mo[0] * NATURAL_FREQ**2))
"""``0.25 * effort / stiffness`` without the gain multiplier (policy_parameters.hpp ``g1_action_scale``)."""

# Training effort / velocity limits (g1.py:239-357). Ankles and waist roll/pitch use 50 N m.
EFFORT_LIMIT_MUJOCO = np.array(
    [139.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2 + [88.0, 50.0, 50.0] + ([25.0] * 5 + [5.0] * 2) * 2,
    dtype=np.float64,
)
VELOCITY_LIMIT_MUJOCO = np.array(
    [20.0, 20.0, 32.0, 20.0, 37.0, 37.0] * 2 + [32.0, 37.0, 37.0] + ([37.0] * 5 + [22.0] * 2) * 2,
    dtype=np.float64,
)

DEFAULT_ANGLES_MUJOCO = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2 + [0.0] * 3 + [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
    + [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
"""Standing pose the action is offset from (policy_parameters.hpp:210-240), MuJoCo order."""

DEFAULT_ROOT_HEIGHT = 0.788740
"""Planner context root height (localmotion_kplanner.hpp ``PlannerConfig::default_height``)."""


def mujoco_to_policy(x, dim: int = -1):
    """Reorder a MuJoCo-order joint axis to policy order (numpy or torch)."""
    return _take(x, MUJOCO_TO_ISAACLAB, dim)


def policy_to_mujoco(x, dim: int = -1):
    """Reorder a policy-order joint axis to MuJoCo order (numpy or torch)."""
    return _take(x, ISAACLAB_TO_MUJOCO, dim)


def _take(x, index: np.ndarray, dim: int):
    if isinstance(x, torch.Tensor):
        return torch.index_select(x, dim, torch.as_tensor(index, device=x.device))
    return np.take(np.asarray(x), index, axis=dim)


DEFAULT_ANGLES_POLICY = mujoco_to_policy(DEFAULT_ANGLES_MUJOCO)
ACTION_SCALE_POLICY = mujoco_to_policy(ACTION_SCALE_MUJOCO)
KP_POLICY = mujoco_to_policy(KP_MUJOCO)
KD_POLICY = mujoco_to_policy(KD_MUJOCO)


def gains_by_name() -> dict[str, dict[str, float]]:
    """Per-joint SONIC gains and limits keyed by joint name (for actuator configs)."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        out[name] = {
            "stiffness": float(KP_MUJOCO[i]),
            "damping": float(KD_MUJOCO[i]),
            "armature": float(ARMATURE_MUJOCO[i]),
            "effort_limit": float(EFFORT_LIMIT_MUJOCO[i]),
            "velocity_limit": float(VELOCITY_LIMIT_MUJOCO[i]),
            "default_angle": float(DEFAULT_ANGLES_MUJOCO[i]),
        }
    return out


class JointMap:
    """Name-based map between a simulator joint list and the SONIC policy order.

    Args:
        sim_joint_names: Joint names of the articulation in simulator order (may contain hand joints).
        device: Torch device of the index tensors.

    Raises:
        KeyError: If one of the 29 SONIC body joints is missing from ``sim_joint_names``.
    """

    def __init__(self, sim_joint_names: Sequence[str], device: str | torch.device = "cpu"):
        names = list(sim_joint_names)
        missing = [n for n in POLICY_JOINT_NAMES if n not in names]
        if missing:
            raise KeyError(f"SONIC body joints missing from the articulation: {missing}")
        self.sim_joint_names = names
        self.num_sim_joints = len(names)
        self.sim_ids = torch.tensor([names.index(n) for n in POLICY_JOINT_NAMES], dtype=torch.long, device=device)
        """Simulator index of each policy joint, shape (29,)."""
        self.sim_ids_mujoco = torch.tensor(
            [names.index(n) for n in MUJOCO_JOINT_NAMES], dtype=torch.long, device=device
        )
        """Simulator index of each MuJoCo-order joint, shape (29,)."""

    def sim_to_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Gather ``(..., num_sim_joints)`` into ``(..., 29)`` policy order."""
        return x.index_select(-1, self.sim_ids)

    def policy_to_sim(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Scatter ``(..., 29)`` policy-order values into a ``(..., num_sim_joints)`` tensor (other joints kept)."""
        if out is None:
            out = torch.zeros(*x.shape[:-1], self.num_sim_joints, dtype=x.dtype, device=x.device)
        out[..., self.sim_ids] = x
        return out
