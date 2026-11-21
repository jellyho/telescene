"""The SuperMimic adapter, with no simulator. The numbers in ``fixtures/supermimic_vectors.npz`` come from realmimic's own
functions: see ``fixtures/make_supermimic_vectors.py``. Tests that need weights or GMR skip without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import fake_body  # noqa: E402

from telescene import ControlInput, make_controller, registered  # noqa: E402
from telescene.trackers import supermimic as sm  # noqa: E402

UPSTREAM = np.load(Path(__file__).parent / "fixtures" / "supermimic_vectors.npz")
CHECKPOINT = sm.CHECKPOINT


def state_at(sample: int, tick: int = -1) -> sm.RobotState:
    row = UPSTREAM["state_hist"][sample, tick].astype(np.float64)
    return sm.RobotState(row[3:7], row[39:42], row[7:36], row[42:71])


def upstream_inputs(sample: int):
    states = [state_at(sample, t) for t in range(10)]
    actions = [UPSTREAM["action_hist"][sample, t].astype(np.float64) for t in range(10)]
    ref = UPSTREAM["ref_window"][sample].astype(np.float64)
    return states, actions, ref


@pytest.mark.parametrize("ours, theirs", [(sm.DEFAULT_JOINT_POS, "default_joint_pos"), (sm.ACTION_SCALE, "action_scale"),
                                          (sm.STIFFNESS, "kp"), (sm.DAMPING, "kd"), (sm.ARMATURE, "armature"), (sm.EFFORT_LIMIT, "effort")])
def test_constants_match_upstream(ours, theirs):
    np.testing.assert_allclose(ours, UPSTREAM[theirs], rtol=1e-6, atol=1e-9)


def test_order_and_timing_match_upstream():
    assert tuple(UPSTREAM["joint_names"].tolist()) == sm.JOINT_NAMES
    assert float(UPSTREAM["stand_root_height"]) == pytest.approx(sm.STAND_ROOT_HEIGHT)


def test_history_matches_upstream():
    for i in range(len(UPSTREAM["history"])):
        states, actions, _ = upstream_inputs(i)
        np.testing.assert_allclose(sm.history_obs(states, actions), UPSTREAM["history"][i], atol=2e-5)


def test_goal_matches_upstream():
    for i in range(len(UPSTREAM["goal"])):
        states, _, ref = upstream_inputs(i)
        goal = sm.goal_obs(states[-1].quat, ref[:, 7:36], ref[:, 42:71], ref[:, 2], ref[:, 3:7])
        np.testing.assert_allclose(goal, UPSTREAM["goal"][i], atol=2e-5)


@pytest.fixture(scope="module")
def policy():
    pytest.importorskip("torch")
    if not CHECKPOINT.exists():
        pytest.skip("no checkpoint")
    return sm.SuperMimicPolicy(CHECKPOINT, threads=1)


def test_action_matches_upstream_network(policy):
    assert policy.iteration == int(UPSTREAM["iteration"])
    for i in range(len(UPSTREAM["action"])):
        action = policy(UPSTREAM["history"][i], UPSTREAM["goal"][i])
        np.testing.assert_allclose(action, UPSTREAM["action"][i], atol=1e-5)
        np.testing.assert_allclose(action * sm.ACTION_SCALE + sm.DEFAULT_JOINT_POS, UPSTREAM["joint_target"][i], atol=1e-5)


def test_whole_path_matches_the_exported_graph(policy):
    """Our observations then our network, against upstream's ONNX, which builds its observations inside the graph.
    Its weights are float16, so it is only as close as upstream's own ``encoding_vs_trained_weights`` figure."""
    ort = pytest.importorskip("onnxruntime")
    path = CHECKPOINT.parent / "web" / "policy.onnx"
    if not path.exists():
        pytest.skip("no ONNX export")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for i in range(len(UPSTREAM["action"])):
        states, actions, ref = upstream_inputs(i)
        ours = policy(sm.history_obs(states, actions), sm.goal_obs(states[-1].quat, ref[:, 7:36], ref[:, 42:71], ref[:, 2], ref[:, 3:7]))
        theirs = session.run(["action"], {k: UPSTREAM[k][i][None] for k in ("state_hist", "action_hist", "ref_window")})[0][0]
        np.testing.assert_allclose(ours, theirs, atol=0.004)


def test_gravity_of_upright_robot():
    np.testing.assert_allclose(sm.projected_gravity(np.array([1.0, 0, 0, 0])), [0, 0, -1])
    np.testing.assert_allclose(sm.projected_gravity(sm.yaw_quat(1.0)), [0, 0, -1])


def pose(value):
    return np.full(29, float(value)), np.array([1.0, 0, 0, 0]), 0.7


def test_window_holds_the_newest_still_when_lookahead_is_zero():
    stream = sm.ReferenceStream(lookahead=0)
    stream.push(*pose(0.0))
    stream.push(*pose(0.1))
    q, qd, z, quat = stream.window()
    assert q.shape == (10, 29) and qd.shape == (10, 29) and z.shape == (10,) and quat.shape == (10, 4)
    np.testing.assert_allclose(q, 0.1)
    np.testing.assert_allclose(qd[0], 0.1 / 0.02)
    np.testing.assert_allclose(qd[1:], 0.0)


def test_window_uses_real_frames_with_lookahead():
    stream = sm.ReferenceStream(lookahead=9)
    for i in range(30):
        stream.push(*pose(i * 0.01))
    q, qd, _, _ = stream.window()
    np.testing.assert_allclose(q[:, 0], [0.01 * i for i in range(20, 30)])  # oldest first, ending now
    np.testing.assert_allclose(qd[:, 0], 0.5)


def test_window_is_defined_from_the_first_pose():
    stream = sm.ReferenceStream(lookahead=3)
    stream.push(*pose(0.2))
    q, qd, _, _ = stream.window()
    np.testing.assert_allclose(q, 0.2)
    np.testing.assert_allclose(qd, 0.0)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, history, goal):
        self.calls.append((history.copy(), goal.copy()))
        return np.full(29, 0.5)


def still(quat=(1.0, 0.0, 0.0, 0.0), q=None):
    return lambda: sm.RobotState(np.array(quat), np.zeros(3), sm.DEFAULT_JOINT_POS.copy() if q is None else q, np.zeros(29))


def control(tracking=False, body=None):
    return ControlInput(None, body, tracking, (0.0, 0.0, 0.0), (0.0,) * 4, (0.0, 0.0), (0.0, 0.0), 0.02, False)


def test_first_history_is_all_the_reset_state_and_later_ticks_see_the_last_action():
    recorder = Recorder()
    controller = sm.SuperMimicController(still(), retarget=None, policy=recorder, lookahead=0)
    target = controller.act(control())
    np.testing.assert_allclose(target, 0.5 * sm.ACTION_SCALE + sm.DEFAULT_JOINT_POS)
    history = recorder.calls[0][0]
    assert history.shape == (930,) and recorder.calls[0][1].shape == (650,)
    np.testing.assert_allclose(history[-290:], 0.0)
    controller.act(control())
    second = recorder.calls[1][0]
    np.testing.assert_allclose(second[-29:], 0.5)
    np.testing.assert_allclose(second[-58:-29], 0.0)
    controller.reset()
    controller.act(control())
    np.testing.assert_allclose(recorder.calls[2][0][-290:], 0.0)


def test_standing_reference_keeps_the_robot_heading():
    recorder = Recorder()
    controller = sm.SuperMimicController(still(sm.yaw_quat(1.0)), policy=recorder, lookahead=0)
    controller.act(control())
    goal = recorder.calls[0][1]
    np.testing.assert_allclose(goal[:29], sm.DEFAULT_JOINT_POS)
    assert goal[58] == pytest.approx(sm.STAND_ROOT_HEIGHT)
    np.testing.assert_allclose(goal[590:596], [1, 0, 0, 1, 0, 0], atol=1e-9)  # no orientation error


def test_tracking_blends_in_and_aligns_the_operator_heading():
    recorder = Recorder()
    target = np.full(29, 0.3)
    retarget = lambda body: (target, sm.yaw_quat(2.0), 0.7)  # the operator faces elsewhere than the robot
    controller = sm.SuperMimicController(still(sm.yaw_quat(0.5)), retarget=retarget, policy=recorder, lookahead=0, blend_seconds=0.1)
    body = np.ones((24, 7))
    controller.act(control(True, body))
    assert 0.0 < controller._blend < 1.0
    for _ in range(6):
        controller.act(control(True, body))
    assert controller._blend == 1.0
    goal = recorder.calls[-1][1]
    np.testing.assert_allclose(goal[:29], 0.3)
    assert goal[58] == pytest.approx(0.7)
    np.testing.assert_allclose(goal[590:596], [1, 0, 0, 1, 0, 0], atol=1e-9)
    for _ in range(6):
        controller.act(control(False, body))
    assert controller._blend == 0.0
    np.testing.assert_allclose(recorder.calls[-1][1][:29], sm.DEFAULT_JOINT_POS)


def test_operator_turn_is_passed_on_relative_to_the_start():
    recorder = Recorder()
    quats = [sm.yaw_quat(2.0)]
    controller = sm.SuperMimicController(still(), retarget=lambda body: (np.zeros(29), quats[0], 0.7), policy=recorder, lookahead=0, blend_seconds=0.02)
    controller.act(control(True, np.ones((24, 7))))
    quats[0] = sm.yaw_quat(2.3)
    controller.act(control(True, np.ones((24, 7))))
    relative = recorder.calls[-1][1][590:596]
    assert relative[0] == pytest.approx(np.cos(0.3)) and relative[2] == pytest.approx(np.sin(0.3))  # matrix: m00, m01, m10


def test_no_body_stands():
    recorder = Recorder()
    controller = sm.SuperMimicController(still(), retarget=lambda body: 1 / 0, policy=recorder, lookahead=0)
    controller.act(control(True, None))
    np.testing.assert_allclose(recorder.calls[0][1][:29], sm.DEFAULT_JOINT_POS)


def test_registered():
    assert "supermimic" in registered()
    with pytest.raises(TypeError):
        make_controller("supermimic")  # it needs an environment


def test_reader_converts_quaternion_and_joint_order():
    torch = pytest.importorskip("torch")

    class Data:
        root_link_quat_w = torch.tensor([[0.1, 0.2, 0.3, 0.9]])  # xyzw
        root_ang_vel_b = torch.tensor([[1.0, 2.0, 3.0]])
        joint_pos = torch.arange(29.0)[None]
        joint_vel = -torch.arange(29.0)[None]
        root_link_pos_w = torch.tensor([[1.0, 2.0, 0.7]])

    class Robot:
        joint_names = list(reversed(sm.JOINT_NAMES))
        body_names = ["pelvis"]
        data = Data()

    class Env:
        scene = {"robot": Robot()}

    state = sm.isaac_state_reader(Env())()
    np.testing.assert_allclose(state.quat, [0.9, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state.joint_pos, np.arange(28, -1, -1))
    np.testing.assert_allclose(state.joint_vel, -np.arange(28, -1, -1))
    np.testing.assert_allclose(state.pos, [1.0, 2.0, 0.7])


@pytest.fixture(scope="module")
def gmr():
    if not sm.GMR_ROOT.exists():
        pytest.skip("no GMR")
    pytest.importorskip("mink")
    pytest.importorskip("mink")
    return sm.GmrRetargeter(sm.GMR_ROOT)


def settled(gmr, body, ticks=10):
    gmr.reset()
    for _ in range(ticks):
        out = gmr(body.astype(np.float32))
    return out


def index(name):
    return sm.JOINT_NAMES.index(name)


def test_retarget_t_pose_holds_the_arms_out(gmr):
    q, quat, z = settled(gmr, fake_body.pose(arms_down=0.0, elbow=0.0))
    assert q[index("left_shoulder_roll_joint")] == pytest.approx(np.pi / 2, abs=0.05)
    assert q[index("right_shoulder_roll_joint")] == pytest.approx(-np.pi / 2, abs=0.05)
    assert q[index("left_elbow_joint")] == pytest.approx(np.pi / 2, abs=0.05)
    assert sm.yaw_of(quat) == pytest.approx(np.pi / 2, abs=0.05)  # facing -z in the room is +y in GMR's axes
    assert 0.7 < z < 0.85


def test_retarget_each_motion_moves_the_right_joints(gmr):
    rest, _, stand_z = settled(gmr, fake_body.pose())
    raised, _, _ = settled(gmr, fake_body.pose(left_raise=1.0))
    assert raised[index("left_shoulder_pitch_joint")] < rest[index("left_shoulder_pitch_joint")] - 0.7  # forward is negative
    assert raised[index("right_shoulder_pitch_joint")] == pytest.approx(rest[index("right_shoulder_pitch_joint")], abs=0.05)
    squat, _, squat_z = settled(gmr, fake_body.pose(squat=0.5))
    assert squat[index("left_knee_joint")] > rest[index("left_knee_joint")] + 0.5
    assert squat[index("right_knee_joint")] > rest[index("right_knee_joint")] + 0.5
    assert squat_z < stand_z - 0.01
    lean, _, _ = settled(gmr, fake_body.pose(lean=0.3))
    assert lean[index("waist_pitch_joint")] == pytest.approx(rest[index("waist_pitch_joint")] + 0.3, abs=0.05)


def test_retarget_turn_follows_yaw(gmr):
    _, front, _ = settled(gmr, fake_body.pose())
    _, turned, _ = settled(gmr, fake_body.pose(turn=0.5))
    assert sm.yaw_of(turned) - sm.yaw_of(front) == pytest.approx(0.5, abs=0.05)


def test_retarget_waits_for_every_joint_then_holds_a_lost_one(gmr):
    gmr.reset()
    body = fake_body.pose().astype(np.float32)
    missing = body.copy()
    missing[20, 3:7] = 0.0
    assert gmr(missing) is None
    good, _, _ = settled(gmr, body)
    q, _, _ = gmr(missing)
    np.testing.assert_allclose(q, good, atol=0.02)


def test_scripted_motion_stays_finite_and_in_range(gmr):
    gmr.reset()
    model = gmr.gmr.model
    limits = model.jnt_range[1:]
    for t in np.arange(0.0, 30.0, 0.1):
        q, quat, z = gmr(fake_body.scripted(float(t)).astype(np.float32))
        assert np.isfinite(q).all() and np.all(q >= limits[:, 0] - 1e-3) and np.all(q <= limits[:, 1] + 1e-3)
        assert 0.6 < z < 0.9 and np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)


def _wrist_error(retargeter, body, height):
    """Left wrist relative to the pelvis, robot against the operator scaled by robot / operator height (metres)."""
    import mujoco

    settled(retargeter, body)
    model, data = retargeter.gmr.model, retargeter.gmr.configuration.data
    mujoco.mj_kinematics(model, data)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    robot = data.xmat[pid].reshape(3, 3).T @ (data.xpos[wid] - data.xpos[pid])
    human = retargeter._human_data(body.astype(np.float64))
    pelvis_rot = sm.quat_matrix(np.asarray(retargeter.gmr.scaled_human_data["Pelvis"][1], float))
    wanted = pelvis_rot.T @ (human["Left_Wrist"][0] - human["Pelvis"][0]) * (1.32 / height)
    return float(np.linalg.norm(robot - wanted))


def test_teleop_preset_scales_the_legs_to_the_g1_and_puts_the_hand_nearer(gmr):
    plain = sm.GmrRetargeter(sm.GMR_ROOT, operator_height=1.82)
    teleop = sm.GmrRetargeter(sm.GMR_ROOT, operator_height=1.82, preset="teleop")
    assert teleop.gmr.human_scale_table["Left_Foot"] == pytest.approx(1.365 / 1.82)
    assert len(teleop.gmr.tasks2) == len(plain.gmr.tasks2) + 2
    body = fake_body.pose(left_raise=1.2, elbow=0.6)
    assert _wrist_error(teleop, body, 1.82) < _wrist_error(plain, body, 1.82)
    with pytest.raises(ValueError):
        sm.GmrRetargeter(sm.GMR_ROOT, preset="nope")
