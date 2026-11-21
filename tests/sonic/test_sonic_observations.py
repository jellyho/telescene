# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Encoder/decoder observation shapes for every mode, zero-filling, VR 3-point and SMPL inputs."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from telescene.sonic.core import rotations as rot
from telescene.sonic.core.commands import PlannerCommand
from telescene.sonic.core.joint_order import DEFAULT_ANGLES_POLICY, UPPER_BODY_POLICY_INDEX
from telescene.sonic.core.obs_config import ObsLayout
from telescene.sonic.core.observations import build_decoder_obs
from telescene.sonic.core.policy import RobotState, SonicInput, SonicPolicyCore
from telescene.sonic.core.smpl import SmplStream
from telescene.sonic.core.vr3pt import HMD_TO_HEAD_WXYZ, Vr3ptBuilder, Vr3ptCfg


def _state(n, yaw=0.0):
    q = rot.yaw_to_quat(torch.full((n,), yaw))
    jp = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32).repeat(n, 1)
    return RobotState(jp, torch.zeros(n, 29), q, torch.zeros(n, 3))


def _wxyz(r: Rotation) -> torch.Tensor:
    return torch.from_numpy(r.as_quat()[..., [3, 0, 1, 2]]).float()


def _slot(core, name, obs):
    s = core.layout.encoder_slot(name)
    return obs[:, s.offset : s.end]


def test_encoder_obs_per_mode(dummy_dir):
    core = SonicPolicyCore.from_dir(dummy_dir, num_envs=3)
    n = 3
    vr_pos = torch.arange(9.0).repeat(n, 1) + 1
    vr_orn = torch.tensor([1.0, 0, 0, 0] * 3).repeat(n, 1)
    inputs = SonicInput(
        encoder_mode=torch.tensor([0, 1, 2]),
        planner=PlannerCommand.idle(n),
        vr_position=vr_pos,
        vr_orientation=vr_orn,
        smpl_joints=torch.randn(n, 24, 3),
        smpl_root_quat=rot.yaw_to_quat(torch.zeros(n)),
        smpl_wrist_joint_pos=torch.full((n, 6), 0.3),
    )
    out = core.step(_state(n), inputs)
    obs = core.last_encoder_obs
    assert obs.shape == (3, 1762)
    assert core.last_decoder_obs.shape == (3, 994)
    assert out.joint_pos_target.shape == (3, 29) and out.tokens.shape == (3, 64)
    np.testing.assert_array_equal(obs[:, 0].numpy(), [0, 1, 2])
    assert torch.all(obs[:, 1:4] == 0)
    for name in core.layout.encoder:
        rows = [m for m in (0, 1, 2) if name.spec.name in core.layout.modes[m].required]
        block = obs[:, name.offset : name.end]
        for m in (0, 1, 2):
            if m not in rows:
                assert torch.all(block[m] == 0), f"{name.spec.name} should be zero for mode {m}"
    # g1 row: planner joint targets present; teleop row: VR inputs passed through; smpl row: wrist joints
    assert _slot(core, "motion_joint_positions_10frame_step5", obs)[0].abs().sum() > 0
    torch.testing.assert_close(_slot(core, "vr_3point_local_target", obs)[1], vr_pos[1])
    torch.testing.assert_close(_slot(core, "vr_3point_local_orn_target", obs)[1], vr_orn[1])
    wrists = _slot(core, "motion_joint_positions_wrists_10frame_step1", obs)[2]
    torch.testing.assert_close(wrists, torch.full((60,), 0.3))
    assert _slot(core, "smpl_joints_10frame_step1", obs)[2].abs().sum() > 0
    # anchor orientation at start: robot and reference aligned -> identity 6D
    anchor = _slot(core, "motion_anchor_orientation_10frame_step5", obs)[0].view(10, 6)
    torch.testing.assert_close(anchor[0], torch.tensor([1.0, 0, 0, 1, 0, 0]), atol=1e-5, rtol=0)


def test_heading_alignment_with_rotated_robot(dummy_dir):
    """A robot spawned at any yaw sees an identity anchor orientation (planner frame starts at zero yaw)."""
    core = SonicPolicyCore.from_dir(dummy_dir, num_envs=2)
    state = _state(2)
    state.base_quat = rot.yaw_to_quat(torch.tensor([1.2, -2.5]))
    core.step(state, SonicInput(torch.zeros(2, dtype=torch.long), PlannerCommand.idle(2)))
    anchor = _slot(core, "motion_anchor_orientation", core.last_encoder_obs)
    assert torch.all(anchor == 0)  # not required by g1 mode
    anchor = _slot(core, "motion_anchor_orientation_10frame_step5", core.last_encoder_obs).view(2, 10, 6)
    torch.testing.assert_close(anchor[:, 0], torch.tensor([[1.0, 0, 0, 1, 0, 0]] * 2), atol=1e-5, rtol=0)


def test_upper_body_override(dummy_dir):
    core = SonicPolicyCore.from_dir(dummy_dir, num_envs=2)
    ub = torch.linspace(-0.5, 0.5, 17).repeat(2, 1)
    inputs = SonicInput(
        torch.zeros(2, dtype=torch.long), PlannerCommand.idle(2), upper_body_joint_pos=ub,
        upper_body_mask=torch.tensor([True, False]),
    )
    core.step(_state(2), inputs)
    jp = _slot(core, "motion_joint_positions_10frame_step5", core.last_encoder_obs).view(2, 10, 29)
    jv = _slot(core, "motion_joint_velocities_10frame_step5", core.last_encoder_obs).view(2, 10, 29)
    cols = torch.as_tensor(UPPER_BODY_POLICY_INDEX)
    torch.testing.assert_close(jp[0][:, cols], ub[0].expand(10, -1))
    assert torch.all(jv[0][:, cols] == 0)
    assert not torch.allclose(jp[1][:, cols], ub[1].expand(10, -1))


def test_decoder_obs_layout():
    layout = ObsLayout.builtin("release")
    tokens = torch.randn(2, 64)
    hist = torch.arange(2 * 10 * 93, dtype=torch.float32).view(2, 10, 93)
    obs = build_decoder_obs(layout, tokens, hist)
    terms = np.split(hist[0].numpy(), [3, 32, 61, 90], axis=1)
    expected = np.concatenate([tokens[0].numpy(), *(t.ravel() for t in terms)])
    np.testing.assert_array_equal(obs[0].numpy(), expected)


def test_low_latency_layout_end_to_end(tmp_path):
    from telescene.sonic.core.dummy import make_dummy_sonic_dir

    d = make_dummy_sonic_dir(str(tmp_path / "ll"), encoder_dim=1247)
    core = SonicPolicyCore.from_dir(d, num_envs=2, observation_config="low_latency")
    assert core.smpl.window == 4
    inputs = SonicInput(
        torch.tensor([2, 2]), PlannerCommand.idle(2), smpl_joints=torch.randn(2, 24, 3),
        smpl_root_quat=rot.yaw_to_quat(torch.zeros(2)), smpl_wrist_joint_pos=torch.zeros(2, 6),
    )
    out = core.step(_state(2), inputs)
    assert core.last_encoder_obs.shape == (2, 1247) and torch.isfinite(out.action).all()


# ---------------------------------------------------------------------------- VR 3-point


def _pose(p, r: Rotation) -> torch.Tensor:
    return torch.cat((torch.as_tensor(p, dtype=torch.float32).view(-1, 3), _wxyz(r).view(-1, 4)), dim=-1)


def test_hmd_axis_convention():
    # a USD/OpenXR camera looking along world +x with +z up has columns x_c=-y, y_c=+z, z_c=-x
    cam = Rotation.from_matrix(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    q = rot.quat_mul(_wxyz(cam).view(1, 4), torch.tensor([HMD_TO_HEAD_WXYZ]))
    torch.testing.assert_close(rot.quat_to_matrix(q)[0], torch.eye(3), atol=1e-6, rtol=0)


def test_vr3pt_targets_and_frame_invariance():
    b = Vr3ptBuilder(1, cfg=Vr3ptCfg(calibrate_head=False))
    cam = Rotation.from_matrix(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    wrist_rot = Rotation.from_euler("y", 0.5)
    wrists = torch.stack((_pose([0.3, 0.2, 0.9], wrist_rot), _pose([0.3, -0.2, 0.9], wrist_rot)), dim=1)
    head = _pose([0.0, 0.0, 1.2], cam)
    root_p = torch.tensor([[0.0, 0.0, 0.79]])
    root_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pos, orn = b(wrists, head, root_p, root_q)
    assert pos.shape == (1, 9) and orn.shape == (1, 12)
    off = wrist_rot.apply([0.18, -0.025, 0.0])
    np.testing.assert_allclose(pos[0, :3].numpy(), np.array([0.3, 0.2, 0.11]) + off, atol=1e-5)
    np.testing.assert_allclose(pos[0, 6:].numpy(), [0.0, 0.0, 0.4], atol=1e-5)
    np.testing.assert_allclose(orn[0, 8:].numpy(), [1.0, 0, 0, 0], atol=1e-5)

    # same scene moved and yawed in the world gives the same pelvis-frame targets
    T = Rotation.from_euler("z", 2.0)
    t = np.array([3.0, -1.0, 0.0])

    def move(pose):
        p = T.apply(pose[..., :3].numpy()) + t
        r = T * Rotation.from_quat(pose[..., 3:].numpy()[..., [1, 2, 3, 0]])
        return _pose(p, r)

    wrists2 = torch.stack((move(wrists[:, 0]), move(wrists[:, 1])), dim=1)
    root2 = move(torch.cat((root_p, root_q), -1))
    pos2, orn2 = b(wrists2, move(head), root2[:, :3], root2[:, 3:])
    torch.testing.assert_close(pos2, pos, atol=1e-5, rtol=0)
    torch.testing.assert_close((orn2.view(3, 4) * orn.view(3, 4)).sum(-1).abs(), torch.ones(3), atol=1e-5, rtol=0)


def test_vr3pt_calibration_and_fallback():
    b = Vr3ptBuilder(2, cfg=Vr3ptCfg(calibrate_head=True, calibrate_wrists=True))
    root_p = torch.zeros(2, 3)
    root_q = torch.tensor([[1.0, 0, 0, 0]] * 2)
    tilted = Rotation.from_euler("xyz", [0.3, -0.2, 0.7])  # arbitrary HMD frame
    head = _pose([[0, 0, 1.2]] * 2, Rotation.concatenate([tilted, tilted]))
    ident = Rotation.identity()
    target_w = torch.stack((_pose([0.4, 0.3, 0.1], ident), _pose([0.4, -0.3, 0.1], ident)), 1)
    target_w = target_w.expand(2, -1, -1).clone()
    robot_w = torch.stack((_pose([0.2, 0.2, -0.2], Rotation.from_euler("z", 0.4)),
                           _pose([0.2, -0.2, -0.2], Rotation.from_euler("z", -0.4))), 1).expand(2, -1, -1).clone()
    b.calibrate(torch.tensor([0]), target_w, head, root_p, root_q, robot_w)
    pos, orn = b(target_w, head, root_p, root_q)
    robot_pts, robot_q = b.wrist_points(robot_w, root_p, root_q)
    # env 0 calibrated: targets equal the robot's current wrist points, head upright
    torch.testing.assert_close(pos[0, :6], robot_pts[0].reshape(6), atol=1e-5, rtol=0)
    torch.testing.assert_close(orn[0, 8:], torch.tensor([1.0, 0, 0, 0]), atol=1e-5, rtol=0)
    # env 1 not calibrated: tilted head target
    assert not torch.allclose(orn[1, 8:], torch.tensor([1.0, 0, 0, 0]), atol=1e-3)
    # zero quaternions (no tracking) fall back to the deploy defaults
    pos, orn = b(torch.zeros(2, 2, 7), torch.zeros(2, 7), root_p, root_q)
    torch.testing.assert_close(pos[1], torch.tensor([0.0903, 0.1615, -0.2411, 0.1280, -0.1522, -0.2461, 0.0241,
                                                     -0.0081, 0.4028]))
    with pytest.raises(ValueError):
        Vr3ptBuilder(1, cfg=Vr3ptCfg(root_frame="torso"))


# ---------------------------------------------------------------------------- SMPL


def test_smpl_stream_window():
    s = SmplStream(2, window=4)
    q = rot.yaw_to_quat(torch.zeros(2))
    s.push(torch.full((2, 24, 3), 1.0), q, torch.full((2, 6), 1.0))
    assert torch.all(s.motion.smpl_joints == 1.0)  # first frame fills the window
    for k in (2.0, 3.0):
        s.push(torch.full((2, 24, 3), k), q, torch.full((2, 6), k), mask=torch.tensor([True, False]))
    idx = s.motion.frame_indices(4, 1)
    window = s.motion.gather(s.motion.smpl_joints, idx)[:, :, 0, 0]
    assert window[0].tolist() == [1.0, 1.0, 2.0, 3.0]
    assert window[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert s.count.tolist() == [3, 1]
    assert s.motion.joint_pos[0, -1, 23:].tolist() == [3.0] * 6
