# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""End-to-end batched core on CPU: cross-check against Isaac Lab's SonicPolicy, determinism, env independence."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from telescene.sonic.core import rotations as rot
from telescene.sonic.core.commands import PlannerCommand, VelocityCommandMapper
from telescene.sonic.core.joint_order import (
    DEFAULT_ANGLES_MUJOCO,
    mujoco_to_policy,
    policy_to_mujoco,
)
from telescene.sonic.core.policy import RobotState, SonicInput, SonicPolicyCore

import sonic_reference_policy as ref


def command_speed(step: int) -> float:
    """sonic.py:44-53 scripted speed profile, time-compressed (updates every 100 ms)."""
    tick = step // 5
    if tick < 6:
        return 0.0
    if tick < 11:
        return round(0.1 * (tick - 5), 3)
    if tick < 30:
        return 0.6
    return round(max(0.0, 0.592 - (tick - 30) * (0.6 / 6.0)), 3)


def _speed_command(speed: float) -> PlannerCommand:
    c = PlannerCommand.idle(1, speed=0.0)
    c.mode[:] = int(speed > 0.0)
    c.speed[:] = speed
    c.movement_direction[:, 0] = float(speed > 0.0)
    return c


class FakeRobot:
    """First-order joint response, slowly yawing and tilting base: deterministic closed-loop inputs."""

    def __init__(self, yaw0: float):
        self.q = DEFAULT_ANGLES_MUJOCO.astype(np.float64).copy() + 0.02
        self.dq = np.zeros(29)
        self.yaw0 = yaw0
        self.t = 0

    def observe(self):
        tilt = 0.0 if self.t == 0 else 0.05 * np.sin(0.1 * self.t)
        r = Rotation.from_euler("zyx", [self.yaw0 + 0.002 * self.t, tilt, -0.5 * tilt])
        gyro = np.array([0.01 * np.cos(0.1 * self.t), 0.02, -0.01])
        return self.q.astype(np.float32), self.dq.astype(np.float32), r.as_quat().astype(np.float32), gyro

    def apply(self, target_mujoco):
        new = 0.6 * self.q + 0.4 * np.asarray(target_mujoco, dtype=np.float64)
        self.dq = (new - self.q) / 0.02
        self.q = new
        self.t += 1


def test_matches_isaaclab_sonic_policy(dummy_dir):
    """Single env, g1 mode: batched core reproduces sonic_policy.py tick by tick (planner, blend, heading)."""
    robot_a, robot_b = FakeRobot(0.7), FakeRobot(0.7)
    q, _, _, _ = robot_a.observe()
    reference = ref.SonicPolicy(dummy_dir, q)
    core = SonicPolicyCore.from_dir(dummy_dir, num_envs=1, init_command_speed=0.0)
    max_err = 0.0
    for step in range(330):
        speed = command_speed(step)
        q, dq, quat_xyzw, gyro = robot_a.observe()
        target_ref, enc_ref = reference.step(q, dq, quat_xyzw, gyro, speed)
        q, dq, quat_xyzw, gyro = robot_b.observe()
        state = RobotState(
            joint_pos=mujoco_to_policy(torch.from_numpy(q)[None]),
            joint_vel=mujoco_to_policy(torch.from_numpy(dq)[None]),
            base_quat=rot.xyzw_to_wxyz(torch.from_numpy(quat_xyzw)[None]),
            base_ang_vel=torch.from_numpy(gyro).float()[None],
        )
        out = core.step(state, SonicInput(torch.zeros(1, dtype=torch.long), _speed_command(speed)))
        np.testing.assert_allclose(core.last_encoder_obs[0].numpy(), enc_ref[0], atol=2e-4, err_msg=f"step {step}")
        target = policy_to_mujoco(out.joint_pos_target[0]).numpy()
        max_err = max(max_err, float(np.abs(target - target_ref).max()))
        np.testing.assert_allclose(target, target_ref, atol=2e-4, err_msg=f"step {step}")
        robot_a.apply(target_ref)
        robot_b.apply(target)
    assert reference.frame == int(core.planner.motion.frame[0])
    assert reference.idle_target is not None and bool(core.planner.idle_stored[0])  # idle re-adaptation ran
    assert max_err < 2e-4


def _batched_inputs(n):
    cmds = torch.tensor(
        [[0.0, 0.0, 0.0, 0.72], [0.4, 0.0, 0.0, 0.72], [0.2, -0.2, 0.6, 0.72], [0.0, 0.0, 0.0, 0.55]]
    )[:n]
    return cmds


def _run_core(core, cmds, ticks, mode, yaw):
    n = cmds.shape[0]
    mapper = VelocityCommandMapper(n)
    robots = [FakeRobot(y) for y in yaw]
    outs = []
    for _ in range(ticks):
        obs = [r.observe() for r in robots]
        state = RobotState(
            joint_pos=mujoco_to_policy(torch.from_numpy(np.stack([o[0] for o in obs]))),
            joint_vel=mujoco_to_policy(torch.from_numpy(np.stack([o[1] for o in obs]))),
            base_quat=rot.xyzw_to_wxyz(torch.from_numpy(np.stack([o[2] for o in obs]))),
            base_ang_vel=torch.from_numpy(np.stack([o[3] for o in obs])).float(),
        )
        vr_pos = torch.tensor([0.25, 0.2, 0.0, 0.25, -0.2, 0.0, 0.0, 0.0, 0.4]).repeat(n, 1)
        vr_orn = torch.tensor([1.0, 0, 0, 0] * 3).repeat(n, 1)
        out = core.step(state, SonicInput(mode, mapper(cmds, 0.02), vr_position=vr_pos, vr_orientation=vr_orn))
        for i, r in enumerate(robots):
            r.apply(policy_to_mujoco(out.joint_pos_target[i]).numpy())
        outs.append(out.joint_pos_target.clone())
    return torch.stack(outs)


def test_batched_end_to_end_and_determinism(dummy_dir, dummy_dir_dynamic):
    cmds = _batched_inputs(4)
    mode = torch.tensor([0, 0, 1, 0])
    yaw = [0.0, 0.5, -1.0, 2.0]
    a = _run_core(SonicPolicyCore.from_dir(dummy_dir, num_envs=4), cmds, 150, mode, yaw)
    b = _run_core(SonicPolicyCore.from_dir(dummy_dir, num_envs=4), cmds, 150, mode, yaw)
    assert a.shape == (150, 4, 29) and torch.isfinite(a).all()
    assert torch.equal(a, b)  # bit-identical across instances
    c = _run_core(SonicPolicyCore.from_dir(dummy_dir_dynamic, num_envs=4), cmds, 150, mode, yaw)
    torch.testing.assert_close(c, a, atol=1e-5, rtol=0)  # fixed-batch row loop == true batch
    # every env behaves as if run alone
    for i in range(4):
        single = _run_core(SonicPolicyCore.from_dir(dummy_dir, num_envs=1), cmds[i : i + 1], 150, mode[i : i + 1],
                           [yaw[i]])
        torch.testing.assert_close(single[:, 0], a[:, i], atol=1e-5, rtol=0)


def test_reset_one_env(dummy_dir):
    core = SonicPolicyCore.from_dir(dummy_dir, num_envs=2)
    cmds = _batched_inputs(2)
    mode = torch.zeros(2, dtype=torch.long)
    _run_core(core, cmds, 30, mode, [0.0, 0.0])
    length0 = int(core.planner.motion.length[0])
    core.reset(torch.tensor([1]))
    assert core.planner.initialized.tolist() == [True, False]
    assert torch.all(core.history.data[1] == 0) and torch.any(core.history.data[0] != 0)
    assert torch.all(core.last_action[1] == 0) and torch.any(core.last_action[0] != 0)
    _run_core(core, cmds, 1, mode, [0.0, 0.0])
    # re-initialised from the measured pose; walking command replans at its first tick -> frame 0 + 1
    assert bool(core.planner.initialized[1]) and int(core.planner.tick[1]) == 1
    assert int(core.planner.motion.frame[1]) == 1
    assert int(core.planner.motion.length[0]) == length0  # env 0 untouched
