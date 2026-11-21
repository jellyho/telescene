# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test-only transcription of SONIC's PICO body -> POSE-frame path (GR00T-WholeBodyControl 1fa5f7fa3966).

Transcribed, single frame, numpy/scipy, from ``gear_sonic/scripts/pico_manager_thread_server.py``
``compute_from_body_poses`` (561-586), ``process_smpl_joints`` (450-489), ``PoseStreamer.parent_indices``
(1270-1296) and the wrist block of ``PoseStreamer.run_once`` (1420-1478); ``gear_sonic/trl/utils/
torch_transform.py`` ``compute_human_joints`` (595-656); ``gear_sonic/isaac_utils/rotations.py``
``remove_smpl_base_rot`` / ``smpl_root_ytoz_up`` (704-719); ``gear_sonic/trl/utils/rotation_conversion.py``
``decompose_rotation_aa`` / ``quaternion_multiply_np`` (595-618). Changes: torch and kornia calls are
replaced by numpy/scipy equivalents, the rest skeleton is an argument, and the 50 Hz resampling is left out
(one sample in, one frame out). It is the reference ``core/pico_body.py`` is checked against; nothing here
is imported from the third-party checkout.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26,
                 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
                 21, 52, 53]
PARENT_INDICES = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23][:24]


def synthetic_joints55(seed=0):
    """A made-up SMPL-X rest skeleton with the real tree and a realistic pelvis rest position."""
    rng = np.random.default_rng(seed)
    j = np.zeros((55, 3))
    j[0] = [0.0031, -0.3514, 0.0120]
    for i in range(1, 55):
        j[i] = j[SMPLX_PARENTS[i]] + rng.normal(0.0, 0.12, 3)
    return j


def _quat_mul(a, b):                       # wxyz Hamilton product (quat_mul, w_last=False)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def _aa_to_quat(aa):                       # kornia angle_axis_to_quaternion, wxyz
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * aa / angle])


def _wxyz_to_rot(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])


def compute_human_joints(body_pose, global_orient, joints55):
    full_pose = np.concatenate([global_orient, body_pose, np.zeros(99)]).reshape(55, 3)
    rot_mats = R.from_rotvec(full_pose).as_matrix()
    rel = joints55.copy()
    rel[1:] -= joints55[SMPLX_PARENTS[1:]]
    tf = np.zeros((55, 4, 4))
    tf[:, :3, :3] = rot_mats
    tf[:, :3, 3] = rel
    tf[:, 3, 3] = 1.0
    chain = [tf[0]]
    for i in range(1, 55):
        chain.append(chain[SMPLX_PARENTS[i]] @ tf[i])
    joints = np.stack(chain)[:, :3, 3]
    return joints[np.concatenate([np.arange(22), [39, 54]])]


def _decompose_rotation_aa(rotation_aa, v2):
    angle = np.linalg.norm(rotation_aa, axis=1)[:, None]
    w = np.cos(angle / 2)
    v = np.sin(angle / 2) * rotation_aa / angle
    q = np.concatenate([w, v], axis=1)
    v_twist = np.dot(v, v2)[:, None] * v2
    q_twist = np.concatenate([w, v_twist], axis=1)
    q_twist = q_twist / np.linalg.norm(q_twist, axis=1)[:, None]
    q_twist_inv = q_twist * np.array([1, -1, -1, -1])
    q_swing = np.stack([_quat_mul(a, b) for a, b in zip(q_twist_inv, q)])
    return q_twist, q_swing


def reference_frame(body_poses_np, joints55):
    """``body_poses_np`` (24, 7) ``[x, y, z, qx, qy, qz, qw]``; returns the POSE frame as ``pico_body`` does."""
    global_rots = R.from_quat(body_poses_np[:, 3:7]) * R.from_euler("y", 180, degrees=True)
    local_rots = []
    for i in range(24):
        if PARENT_INDICES[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rots.append(global_rots[PARENT_INDICES[i]].inv() * global_rots[i])
    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])
    body_pose = pose_aa[1:].flatten()
    global_orient = pose_aa[0]

    q = _aa_to_quat(global_orient)
    q = _quat_mul(_aa_to_quat(np.array([np.pi / 2, 0.0, 0.0])), q)                     # smpl_root_ytoz_up
    joints = compute_human_joints(body_pose[:63], _wxyz_to_rot(q).as_rotvec(), joints55)
    q = _quat_mul(q, np.array([0.5, -0.5, -0.5, -0.5]))                                # remove_smpl_base_rot
    smpl_joints_local = _wxyz_to_rot(q).inv().apply(joints)

    body_pose21 = body_pose[:63].reshape(-1, 21, 3)
    l_elbow, r_elbow, l_wrist, r_wrist = (body_pose21[:, k] for k in (17, 18, 19, 20))
    _, l_swing = _decompose_rotation_aa(l_elbow, np.array([0, 1, 0]))
    _, r_swing = _decompose_rotation_aa(r_elbow, np.array([0, 1, 0]))
    l_swing_euler = R.from_quat(l_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    r_swing_euler = R.from_quat(r_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    l_wrist_euler = R.from_rotvec(l_wrist).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(r_wrist).as_euler("XYZ", degrees=False)
    joint_pos = np.zeros(29)
    joint_pos[23] = l_swing_euler[0, 0] + l_wrist_euler[0, 0]
    joint_pos[25] = -(-l_wrist_euler[0, 1])
    joint_pos[27] = l_swing_euler[0, 2] + l_wrist_euler[0, 2]
    joint_pos[24] = -(r_swing_euler[0, 0] + r_wrist_euler[0, 0])
    joint_pos[26] = -r_wrist_euler[0, 1]
    joint_pos[28] = r_swing_euler[0, 2] + r_wrist_euler[0, 2]
    return {"smpl_joints": smpl_joints_local, "root_quat": q, "smpl_pose": body_pose21[0],
            "wrist_joint_pos": joint_pos[23:29]}
