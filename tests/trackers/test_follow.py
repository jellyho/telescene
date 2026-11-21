"""The follow policy's goal edit and its place in the SuperMimic controller, with no simulator. The edit is checked
against the training code's own (``follow/edits.py``) when that is present, and the network against a checkpoint."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from telescene import ControlInput
from telescene.trackers import follow as fl
from telescene.trackers import supermimic as sm

FOLLOW = Path(os.environ["FOLLOW_ROOT"]) if os.environ.get("FOLLOW_ROOT") else None  # the unreleased training code
CHECKPOINT = Path(os.environ.get("FOLLOW_CHECKPOINT", sm.FOLLOW_CHECKPOINT))
CHECKPOINT_V2 = Path(os.environ["FOLLOW_CHECKPOINT_V2"]) if os.environ.get("FOLLOW_CHECKPOINT_V2") else None


def goal_of(rng, yaws):
    q = rng.normal(size=(10, 29))
    qd = rng.normal(size=(10, 29))
    z = 0.7 + 0.05 * rng.normal(size=10)
    quats = np.stack([sm.yaw_quat(y) for y in yaws])
    return sm.goal_obs(sm.yaw_quat(0.0), q, qd, z, quats)


def test_zero_action_changes_nothing():
    goal = goal_of(np.random.default_rng(0), np.linspace(0, 1, 10))
    legs = np.array([sm.JOINT_NAMES.index(n) for n in fl.LEG_JOINTS])
    np.testing.assert_allclose(fl.edit_goal(goal, np.zeros(16), np.zeros(16), legs), goal, atol=1e-12)


def test_edit_moves_legs_height_and_yaw_only():
    goal = goal_of(np.random.default_rng(1), np.zeros(10))
    legs = np.array([sm.JOINT_NAMES.index(n) for n in fl.LEG_JOINTS])
    action = np.zeros(16)
    action[:12], action[12], action[15] = 1.0, -1.0, 1.0
    edited = fl.edit_goal(goal, action, np.zeros(16), legs).reshape(-1)
    frames, before = edited[:590].reshape(10, 59), goal[:590].reshape(10, 59)
    np.testing.assert_allclose(frames[:, legs] - before[:, legs], fl.LEG_SCALE)
    arms = np.setdiff1d(np.arange(29), legs)
    np.testing.assert_allclose(frames[:, arms], before[:, arms])
    np.testing.assert_allclose(frames[:, 58] - before[:, 58], -fl.HEIGHT_SCALE)
    yaw = 0.5 * 1.0
    np.testing.assert_allclose(edited[590:596], [math.cos(yaw), -math.sin(yaw), math.sin(yaw), math.cos(yaw), 0, 0], atol=1e-9)


def test_edit_matches_training_code():
    torch = pytest.importorskip("torch")
    if FOLLOW is None or not (FOLLOW / "follow" / "edits.py").exists():
        pytest.skip("no follow training code")
    sys.path.insert(0, str(FOLLOW))
    from follow import edits

    rng = np.random.default_rng(2)
    goal = goal_of(rng, rng.uniform(-1, 1, 10))
    action, rate = rng.uniform(-1, 1, 16), rng.normal(size=16)
    legs = edits.leg_indices(sm.JOINT_NAMES)
    theirs = edits.edit_goal(torch.tensor(goal[None]), torch.tensor(action[None]), torch.tensor(rate[None]), legs)[0].numpy()
    ours = fl.edit_goal(goal, action, rate, legs.numpy())
    np.testing.assert_allclose(ours, theirs, atol=1e-9)
    assert edits.FRAMES == fl.FRAMES and edits.LEG_JOINTS == fl.LEG_JOINTS and edits.ROT_SCALE == tuple(fl.ROT_SCALE)
    assert (edits.LEG_SCALE, edits.HEIGHT_SCALE, edits.INTEGRAL_DECAY, edits.INTEGRAL_LIMIT) == (
        fl.LEG_SCALE, fl.HEIGHT_SCALE, fl.INTEGRAL_DECAY, fl.INTEGRAL_LIMIT)


def test_full_edit_matches_training_code():
    torch = pytest.importorskip("torch")
    if FOLLOW is None or not (FOLLOW / "follow" / "edits.py").exists():
        pytest.skip("no follow training code")
    sys.path.insert(0, str(FOLLOW))
    from follow import edits

    rng = np.random.default_rng(3)
    goal = goal_of(rng, rng.uniform(-1, 1, 10))
    action, rate = rng.uniform(-1, 1, 33), rng.normal(size=33)
    ours_scales = fl.joint_scales(sm.JOINT_NAMES)
    np.testing.assert_allclose(edits.joint_scales(sm.JOINT_NAMES).numpy(), ours_scales, atol=1e-7)  # theirs are float32
    theirs = edits.edit_goal_full(torch.tensor(goal[None]), torch.tensor(action[None]), torch.tensor(rate[None]),
                                  torch.tensor(ours_scales))[0].numpy()
    ours = fl.edit_goal_full(goal, action, rate, ours_scales)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)
    assert edits.FULL_EDIT_DIM == fl.FULL_EDIT_DIM


def test_full_edit_scales_each_joint():
    goal = goal_of(np.random.default_rng(4), np.zeros(10))
    scales = fl.joint_scales(sm.JOINT_NAMES)
    action = np.zeros(33)
    action[:29] = 1.0
    frames = fl.edit_goal_full(goal, action, np.zeros(33), scales)[:590].reshape(10, 59)
    np.testing.assert_allclose(frames[:, :29] - goal[:590].reshape(10, 59)[:, :29], np.broadcast_to(scales, (10, 29)))
    waist = [i for i, n in enumerate(sm.JOINT_NAMES) if "waist" in n]
    assert waist and np.all(scales[waist] == 0.2) and scales[sm.JOINT_NAMES.index("left_knee_joint")] == 0.35


class NetV2:
    outputs = 33

    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.copy())
        out = np.zeros(33)
        out[sm.JOINT_NAMES.index("left_elbow_joint")] = 1.0
        return out


def test_v2_follower_reads_the_whole_goal_and_edits_every_joint():
    net = NetV2()
    follower = fl.Follower(net, sm.JOINT_NAMES)
    assert follower.version == 2
    goal = goal_of(np.random.default_rng(5), np.zeros(10))
    history = np.zeros(930)
    edited = follower(history, goal, (1.0, 0.0), (0.0, 0.0), 0.0, 0.0)
    x = net.inputs[-1]
    assert x.shape == (93 + 650 + 2 + 1 + 33,)
    np.testing.assert_allclose(x[93:743], goal)
    np.testing.assert_allclose(x[743:745], [0.8, 0.0])  # 1 m away, shortened to max_error
    elbow = sm.JOINT_NAMES.index("left_elbow_joint")
    np.testing.assert_allclose(edited[:590].reshape(10, 59)[:, elbow] - goal[:590].reshape(10, 59)[:, elbow], 0.35)
    assert follower.action.shape == (33,) and not follower.integral.any()


def test_action_filter_smooths_what_edits_the_goal_and_what_the_network_sees():
    net = NetV2()
    follower = fl.Follower(net, sm.JOINT_NAMES, action_filter=0.3)
    goal = goal_of(np.random.default_rng(7), np.zeros(10))
    elbow = sm.JOINT_NAMES.index("left_elbow_joint")
    follower(np.zeros(930), goal, (0.0, 0.0), (0.0, 0.0), 0.0, 0.0)
    assert follower.action[elbow] == pytest.approx(0.3)
    follower(np.zeros(930), goal, (0.0, 0.0), (0.0, 0.0), 0.0, 0.0)
    assert follower.action[elbow] == pytest.approx(0.51)
    assert net.inputs[-1][-33:][elbow] == pytest.approx(0.3)  # its last action, filtered
    with pytest.raises(ValueError):
        fl.Follower(net, sm.JOINT_NAMES, action_filter=0.0)


def test_v2_checkpoint_loads_and_answers():
    pytest.importorskip("torch")
    if CHECKPOINT_V2 is None or not CHECKPOINT_V2.exists():
        pytest.skip("no v2 follow checkpoint")
    net = fl.FollowNet(CHECKPOINT_V2)
    assert net.outputs == 33 and net.inputs == 93 + 650 + 2 + 1 + 33
    follower = fl.Follower(net, sm.JOINT_NAMES)
    goal = goal_of(np.random.default_rng(6), np.zeros(10))
    edited = follower(np.zeros(930), goal, (0.3, 0.0), (0.0, 0.0), 0.0, 0.0)
    assert edited.shape == (650,) and np.all(np.abs(follower.action) <= 1)


def test_error_is_in_the_robot_heading():
    np.testing.assert_allclose(fl.error_in_heading((1.0, 1.0), (0.0, 1.0), math.pi / 2), [0.0, -1.0], atol=1e-12)


def test_checkpoint_loads_and_answers():
    pytest.importorskip("torch")
    if not CHECKPOINT.exists():
        pytest.skip("no follow checkpoint")
    net = fl.FollowNet(CHECKPOINT)
    assert net.inputs == 93 + 59 + 6 + 2 + 1 + 2 + 16
    out = net(np.zeros(net.inputs))
    assert out.shape == (16,) and np.all(np.abs(out) <= 1)


class Still:
    def __init__(self, pos=(0.0, 0.0, 0.75), yaw=0.0):
        self.pos, self.yaw = np.array(pos), yaw

    def __call__(self):
        return sm.RobotState(sm.yaw_quat(self.yaw), np.zeros(3), sm.DEFAULT_JOINT_POS.copy(), np.zeros(29), self.pos)


class Net:
    """Remembers what it was given and always pushes the yaw."""

    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.copy())
        out = np.zeros(16)
        out[15] = 1.0
        return out


def body_at(x, z):
    """A body in the room (OpenXR local floor: -Z ahead, +X right) with its pelvis at (x, 1, z), facing -Z."""
    body = np.zeros((24, 7))
    body[:, :3] = (x, 1.0, z)
    body[1, 0], body[2, 0] = x - 0.1, x + 0.1  # left hip on the left (-X), right hip on the right
    return body


def test_controller_edits_towards_the_operator_and_records_the_command():
    net, goals = Net(), []
    policy = lambda history, goal: (goals.append(goal.copy()), np.zeros(29))[1]
    controller = sm.SuperMimicController(Still((2.0, 3.0, 0.75)), retarget=lambda body: (np.zeros(29), sm.yaw_quat(0.0), 0.7), policy=policy,
                                         lookahead=0, blend_seconds=0.02, follow=fl.Follower(net, sm.JOINT_NAMES))
    body = body_at(0.0, -0.5)  # half a metre ahead of the room origin
    control = ControlInput(None, body, True, (0.0, 0.0, 0.0), (0.0,) * 4, (0.0, 0.0), (0.0, 0.0), 0.02, False,
                           anchor=(2.0, 3.0, 0.0), anchor_yaw=0.0, world_scale=0.8)
    controller.act(control)
    controller.act(control)
    target = controller.last_target
    np.testing.assert_allclose(target, [2.0 + 0.4, 3.0], atol=1e-9)
    x = net.inputs[-1]
    np.testing.assert_allclose(x[93 + 59 + 6: 93 + 59 + 8], [0.4, 0.0], atol=1e-9)  # robot at the anchor facing +x: 0.4 m ahead
    assert sm.yaw_of(controller._operator[1]) + controller._yaw_offset == pytest.approx(0.0, abs=1e-9)  # faces -Z: sim +x
    rec = controller.record()
    assert set(rec) == {"command", "goal", "latent", "follow_action", "follow_error", "follow_target", "robot_joint_pos", "robot_quat",
                        "ref_joint_pos", "ref_root_z", "ref_root_quat", "blend", "tracking_error"}
    assert rec["command"].shape == (650,) and rec["goal"].shape == (650,)
    np.testing.assert_allclose(rec["command"], goals[-1], atol=1e-6)
    assert not np.allclose(rec["command"], rec["goal"])  # the yaw push made it into what SuperMimic got
    np.testing.assert_allclose(rec["command"][:590], rec["goal"][:590], atol=1e-6)


def test_no_body_leaves_the_goal_alone():
    net, goals = Net(), []
    policy = lambda history, goal: (goals.append(goal.copy()), np.zeros(29))[1]
    controller = sm.SuperMimicController(Still(), retarget=lambda body: 1 / 0, policy=policy, lookahead=0,
                                         follow=fl.Follower(net, sm.JOINT_NAMES))
    controller.act(ControlInput(None, None, True, (0.0, 0.0, 0.0), (0.0,) * 4, (0.0, 0.0), (0.0, 0.0), 0.02, False))
    assert not net.inputs
    np.testing.assert_allclose(controller.record()["command"], controller.record()["goal"])
