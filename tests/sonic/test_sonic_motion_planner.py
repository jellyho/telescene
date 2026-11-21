# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Planner resampling, context sampling, cross-fade and replan rules."""

import threading
import time

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from telescene.sonic.core import motion as mo
from telescene.sonic.core.commands import (
    LocomotionMode,
    PlannerCommand,
    VelocityCommandCfg,
    VelocityCommandMapper,
)
from telescene.sonic.core.joint_order import (
    DEFAULT_ANGLES_POLICY,
    LOWER_BODY_POLICY_INDEX,
    MUJOCO_TO_ISAACLAB,
)
from telescene.sonic.core.planner import SonicPlanner

import sonic_reference_policy as ref


def _random_qpos(frames, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((frames, 36)).astype(np.float32) * 0.3
    quats = Rotation.from_euler("z", np.linspace(0, 1.0, frames)[:, None]).as_quat()[:, [3, 0, 1, 2]]
    q[:, 3:7] = quats
    return q


def _ref_plan(qpos, count):
    poses = qpos[:count].astype(np.float64)
    motion = ref._sample_motion(poses, np.arange(int(count * 50 / 30)) * (30 / 50))
    velocity = np.diff(motion[:, 7:], axis=0) * 50
    return motion, np.concatenate((velocity, velocity[-1:]))


@pytest.mark.parametrize("count", [48, 37, 64, 3])
def test_resample_matches_reference(count):
    qpos = _random_qpos(64, count)
    pos, quat, jp, jv, length = mo.resample_planner_output(torch.from_numpy(qpos)[None], torch.tensor([count]))
    motion, velocity = _ref_plan(qpos, count)
    n = len(motion)
    assert int(length) == n == (count * 50) // 30
    np.testing.assert_allclose(pos[0, :n].numpy(), motion[:, :3], atol=1e-5)
    np.testing.assert_allclose(np.abs((quat[0, :n].numpy() * motion[:, 3:7]).sum(-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(jp[0, :n].numpy(), motion[:, 7:][:, MUJOCO_TO_ISAACLAB], atol=1e-5)
    np.testing.assert_allclose(jv[0, :n].numpy(), velocity[:, MUJOCO_TO_ISAACLAB], atol=1e-3)


def _buffer_from(motion_np, velocity_np, frame):
    buf = mo.MotionBuffer(1, 160)
    t = len(motion_np)
    m = torch.from_numpy(motion_np).float()
    buf.write(
        torch.tensor([0]),
        m[None, :, :3],
        m[None, :, 3:7],
        m[None, :, 7:][..., MUJOCO_TO_ISAACLAB],
        torch.from_numpy(velocity_np).float()[None][..., MUJOCO_TO_ISAACLAB],
        torch.tensor([t]),
    )
    buf.frame[0] = frame
    return buf


@pytest.mark.parametrize("frame", [1, 17, 78, 79])
def test_context_and_blend_match_reference(frame):
    motion, velocity = _ref_plan(_random_qpos(64, 1), 48)
    buf = _buffer_from(motion, velocity, frame)
    ctx = mo.sample_context(buf, torch.tensor([0]), look_ahead=2)[0].numpy()
    expected = ref._sample_motion(motion, frame + 2 + np.arange(4) * (50 / 30))
    np.testing.assert_allclose(ctx[:, :3], expected[:, :3], atol=1e-5)
    np.testing.assert_allclose(np.abs((ctx[:, 3:7] * expected[:, 3:7]).sum(-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(ctx[:, 7:], expected[:, 7:], atol=1e-5)

    new_motion, new_velocity = _ref_plan(_random_qpos(64, 2), 40)
    frames = np.arange(len(new_motion) + 2)
    old = np.minimum(frame + frames, len(motion) - 1)
    new = np.clip(frames - 2, 0, len(new_motion) - 1)
    w = np.clip((frames - 2) / 8, 0.0, 1.0)[:, None]
    blended = (1 - w) * motion[old] + w * new_motion[new]
    blended_vel = (1 - w) * velocity[old] + w * new_velocity[new]
    plan = mo.resample_planner_output(torch.from_numpy(_random_qpos(64, 2))[None], torch.tensor([40]))
    mo.blend_into(buf, torch.tensor([0]), plan, gen_offset=2)
    n = len(blended)
    assert int(buf.length[0]) == n and int(buf.frame[0]) == 0
    np.testing.assert_allclose(buf.root_pos[0, :n].numpy(), blended[:, :3], atol=1e-5)
    np.testing.assert_allclose(buf.joint_pos[0, :n].numpy(), blended[:, 7:][:, MUJOCO_TO_ISAACLAB], atol=1e-5)
    np.testing.assert_allclose(buf.joint_vel[0, :n].numpy(), blended_vel[:, MUJOCO_TO_ISAACLAB], atol=1e-3)


def test_frame_indices_clamp():
    buf = mo.MotionBuffer(2, 20)
    buf.length[:] = torch.tensor([12, 20])
    buf.frame[:] = torch.tensor([5, 0])
    idx = buf.frame_indices(4, 3)
    assert idx.tolist() == [[5, 8, 11, 11], [0, 3, 6, 9]]


def _cmd(n, mode, speed, facing_yaw=0.0, height=-1.0):
    c = PlannerCommand.idle(n)
    c.mode[:] = mode
    c.speed[:] = speed
    c.height[:] = height
    if mode > 0 and mode not in (4, 5, 6):
        c.movement_direction[:, 0] = 1.0
    c.facing_direction[:, 0] = np.cos(facing_yaw)
    c.facing_direction[:, 1] = np.sin(facing_yaw)
    return c


def _run(planner, schedule, ticks):
    """schedule(tick) -> PlannerCommand; returns ticks at which a replan happened."""
    replans = []
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    for t in range(ticks):
        cmd = schedule(t)
        planner.plan(cmd)
        if bool(planner.replanned[0]):
            replans.append(t)
        planner.commit()
        planner.advance(cmd, joint)
    return replans


def test_replan_rules(dummy_dir):
    p = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1)
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    walk = lambda t: _cmd(1, 1, 0.4) if t >= 10 else _cmd(1, 0, -1.0)  # noqa: E731
    # mode change at the first planner tick >= 10; the 1.0 s timer counts planner ticks since init
    assert _run(p, walk, 120) == [10, 45, 95]

    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    squat_speeds = lambda t: _cmd(1, 4, -1.0 if t < 20 else 0.5, height=0.6)  # noqa: E731
    # entering squat replans; speed changes in a static mode do not
    assert _run(p, squat_speeds, 60) == [0]

    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    turning = lambda t: _cmd(1, 0, -1.0, facing_yaw=0.01 * t)  # noqa: E731
    # facing changes replan at every planner tick, even when idle (t = 0 equals the initial facing)
    assert _run(p, turning, 21) == [5, 10, 15, 20]

    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    run = lambda t: _cmd(1, int(LocomotionMode.RUN), 2.0)  # noqa: E731
    assert _run(p, run, 16) == [0, 5, 10, 15]


def test_idle_readaptation(dummy_dir):
    p = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1)
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    idle = _cmd(1, 0, -1.0)
    p.initialize(torch.tensor([0]), joint, idle)
    length = int(p.motion.length[0])
    lower = torch.as_tensor(LOWER_BODY_POLICY_INDEX)
    measured = joint.clone()
    measured[0, lower] += 0.3  # robot far from the planner target -> ADAPTING
    for _ in range(length + 5):
        p.plan(idle)
        p.commit()
        p.advance(idle, measured)
    assert int(p.motion.frame[0]) == length - 1
    assert int(p.idle_state[0]) == 1
    held = p.motion.joint_pos[0, length - 1, lower]
    start = p.idle_target[0]
    # each adapting tick moves the held target 2 % towards the measured pose
    assert torch.all((held - start).abs() > 1e-3)
    assert torch.all((measured[0, lower] - held).abs() < (measured[0, lower] - start).abs())


def test_velocity_mapper():
    mapper = VelocityCommandMapper(4, cfg=VelocityCommandCfg())
    cmd = torch.tensor([[0.0, 0.0, 0.0, 0.72], [0.4, 0.0, 0.0, 0.72], [0.0, 0.3, 1.0, 0.72], [0.4, 0.0, 0.0, 0.5]])
    out = mapper(cmd, 0.02)
    assert out.mode.tolist() == [0, 1, 1, 4]
    torch.testing.assert_close(out.speed, torch.tensor([-1.0, 0.4, 0.3, -1.0]))
    torch.testing.assert_close(out.height, torch.tensor([-1.0, -1.0, -1.0, 0.5]))
    torch.testing.assert_close(out.movement_direction[1], torch.tensor([1.0, 0.0, 0.0]))
    # lateral command expressed in the (just rotated) facing frame
    yaw = 0.02
    torch.testing.assert_close(out.movement_direction[2], torch.tensor([-np.sin(yaw), np.cos(yaw), 0.0]).float())
    torch.testing.assert_close(out.facing_direction[2], torch.tensor([np.cos(yaw), np.sin(yaw), 0.0]).float())
    torch.testing.assert_close(out.movement_direction[3], torch.zeros(3))
    for _ in range(49):
        out = mapper(cmd, 0.02)
    assert mapper.yaw[2].item() == pytest.approx(1.0, abs=1e-5)
    mapper.reset(torch.tensor([2]))
    assert mapper.yaw[2].item() == 0.0


def _wait(planner):
    while planner._future is not None and not planner._future[1].done():
        time.sleep(0.001)


def test_background_plan_matches_the_synchronous_one(dummy_dir):
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    walk = _cmd(1, 1, 0.4)
    sync = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1)
    bg = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1, background=True)
    for p in (sync, bg):
        p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
        p.plan(walk)
        assert bool(p.replanned[0])
    _wait(bg)
    sync.commit()
    bg.commit()
    assert bg._future is None
    assert int(bg.motion.length[0]) == int(sync.motion.length[0])
    assert torch.allclose(bg.motion.joint_pos, sync.motion.joint_pos)


def test_background_plan_is_aligned_by_the_ticks_it_took(dummy_dir):
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    walk = _cmd(1, 1, 0.4)
    idle = _cmd(1, 0, -1.0)
    p = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1, background=True, look_ahead=5)
    p.initialize(torch.tensor([0]), joint, idle)
    p.plan(walk)
    plan = p.infer(mo.sample_context(p.motion, torch.tensor([0]), 5), PlannerCommand(*(t[[0]] for t in walk.as_tuple())))
    _wait(p)
    for _ in range(3):
        p.advance(walk, joint)
    expected_tail = plan[2][0, 8:20].clone()
    p.commit()
    # three ticks passed, so the plan starts two frames ahead and is fully faded in eight frames later
    assert int(p.motion.frame[0]) == 0
    assert int(p.motion.length[0]) == int(plan[4][0]) + 2
    assert torch.allclose(p.motion.joint_pos[0, 10:22], expected_tail)
    assert p.late_plans == 0


def test_background_plan_arriving_late_skips_what_has_passed(dummy_dir):
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    walk = _cmd(1, 1, 0.4)
    p = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1, background=True, look_ahead=2)
    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    p.plan(walk)
    plan = p.infer(mo.sample_context(p.motion, torch.tensor([0]), 2), PlannerCommand(*(t[[0]] for t in walk.as_tuple())))
    _wait(p)
    for _ in range(5):
        p.advance(walk, joint)
    p.commit()
    assert p.late_plans == 1
    assert int(p.motion.length[0]) == int(plan[4][0]) - 3
    assert torch.allclose(p.motion.joint_pos[0, 8:20], plan[2][0, 11:23])


def test_only_one_background_plan_in_flight_and_reset_discards_it(dummy_dir):
    joint = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32)[None]
    p = SonicPlanner.from_file(f"{dummy_dir}/planner_sonic.onnx", num_envs=1, background=True)
    p.initialize(torch.tensor([0]), joint, _cmd(1, 0, -1.0))
    gate = threading.Event()
    p._worker.submit(gate.wait)  # hold the worker so that the plan cannot finish
    p.plan(_cmd(1, 1, 0.4))
    first = p._future
    assert first is not None
    p.commit()
    assert p._future is first  # not finished: nothing blended, nothing waited for
    for _ in range(5):
        p.advance(_cmd(1, 1, 0.4), joint)
    p.plan(_cmd(1, 1, 0.6))
    assert p._future is first and not bool(p.replanned[0])
    p.reset()
    assert p._future is None
    gate.set()
