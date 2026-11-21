"""The spec-driven ONNX tracker, with no simulator. SuperMimic's own observation, written as a spec, has to come out
exactly as ``history_obs`` and ``goal_obs`` build it; a tiny generated ONNX graph stands in for a real checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from test_supermimic import UPSTREAM, control, still, upstream_inputs  # noqa: E402

from telescene import registered  # noqa: E402
from telescene.trackers import onnx_tracker as ot  # noqa: E402
from telescene.trackers import supermimic as sm  # noqa: E402

SUPERMIMIC_RUNTIME = {
    "slug": "supermimic_as_spec", "onnx": "none.onnx", "joint_names": list(sm.JOINT_NAMES),
    "default_joint_pos": sm.DEFAULT_JOINT_POS.tolist(), "action_scale": sm.ACTION_SCALE.tolist(), "lookahead": 9,
    "stand_root_height": sm.STAND_ROOT_HEIGHT,
    "inputs": {
        "history": [{"terms": ["joint_pos_rel", "joint_vel", "base_ang_vel", "projected_gravity", "last_action"],
                     "steps": list(range(-9, 1)), "major": "term"}],
        "goal": [{"terms": ["ref_joint_pos", "ref_joint_vel", "ref_root_height"], "steps": list(range(10)), "major": "time"},
                 {"terms": ["ref_root_ori_6d"], "steps": list(range(10))}],
    },
}


def spec_file(tmp_path, runtime, wrap=True):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"runtime": runtime} if wrap else runtime))
    return path


def test_supermimic_written_as_a_spec_builds_the_same_observation(tmp_path):
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, SUPERMIMIC_RUNTIME))
    assert spec.history == 10 and spec.frames == 10
    for i in range(len(UPSTREAM["action"])):
        states, actions, ref = upstream_inputs(i)
        ticks = [ot.Tick(s, a) for s, a in zip(states, actions)]
        zero = np.zeros(3)

        def frame(k):
            return ot.Frame(ref[k, 7:36], ref[k, 42:71], np.array([0.0, 0.0, ref[k, 2]]), ref[k, 3:7], zero, zero)

        history = np.concatenate([ot.build_block(b, spec, ticks, frame) for b in spec.inputs["history"]])
        goal = np.concatenate([ot.build_block(b, spec, ticks, frame) for b in spec.inputs["goal"]])
        np.testing.assert_array_equal(history, sm.history_obs(states, actions))
        np.testing.assert_array_equal(goal, sm.goal_obs(states[-1].quat, ref[:, 7:36], ref[:, 42:71], ref[:, 2], ref[:, 3:7]))
        np.testing.assert_allclose(history, UPSTREAM["history"][i], atol=1e-5)
        np.testing.assert_allclose(goal, UPSTREAM["goal"][i], atol=1e-5)


def test_supermimic_as_a_spec_drives_the_controller_like_supermimic(tmp_path):
    """Same policy stub, same operator-less ticks: the two controllers see the same inputs and give the same targets."""
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, SUPERMIMIC_RUNTIME))
    seen_sm, seen_ot = [], []

    def sm_policy(history, goal):
        seen_sm.append((history, goal))
        return np.sin(np.arange(29) + len(seen_sm))

    def ot_policy(inputs):
        seen_ot.append((inputs["history"], inputs["goal"]))
        return np.sin(np.arange(29) + len(seen_ot))

    a = sm.SuperMimicController(still(), policy=sm_policy, lookahead=9)
    b = ot.OnnxTrackerController(still(), spec, policy=ot_policy)
    for _ in range(5):
        np.testing.assert_allclose(b.act(control()), a.act(control()), atol=1e-12)
    for (h1, g1), (h2, g2) in zip(seen_sm, seen_ot):
        np.testing.assert_allclose(h2, h1, atol=1e-12)
        np.testing.assert_allclose(g2, g1, atol=1e-12)


def shuffled_runtime():
    rng = np.random.default_rng(0)
    order = rng.permutation(29)
    return order, {
        "slug": "toy", "onnx": "toy.onnx", "joint_names": [sm.JOINT_NAMES[i] for i in order],
        "default_joint_pos": sm.DEFAULT_JOINT_POS[order].tolist(), "action_scale": 0.5, "action_clip": 2.0,
        "last_action": "target",
        "inputs": {"obs": [{"terms": ["joint_pos_rel", "last_action"], "steps": [-2, -1, 0], "major": "time", "scale": 2.0},
                           {"terms": ["ref_joint_pos"], "steps": [1, 2, 4, 8]}]},
        "constant_inputs": {"time_step": [[0.0]]},
    }


def toy_onnx(path: Path, obs_dim: int):
    """``action = obs[:29] * 10``, with a second input it ignores, the way some exports carry a time step."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    take = helper.make_node("Slice", ["obs", "starts", "ends", "axes"], ["head"])
    times = helper.make_node("Mul", ["head", "ten"], ["actions"])
    graph = helper.make_graph(
        [take, times], "toy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim]),
         helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        [helper.make_tensor("starts", TensorProto.INT64, [1], [0]), helper.make_tensor("ends", TensorProto.INT64, [1], [29]),
         helper.make_tensor("axes", TensorProto.INT64, [1], [1]), helper.make_tensor("ten", TensorProto.FLOAT, [], [10.0])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def test_onnx_graph_joint_order_scale_clip_and_last_target(tmp_path):
    pytest.importorskip("onnxruntime")
    order, runtime = shuffled_runtime()
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, runtime, wrap=False))
    assert spec.history == 3 and spec.frames == 9
    toy_onnx(tmp_path / "toy.onnx", 3 * 58 + 4 * 29)
    q = sm.DEFAULT_JOINT_POS.copy()
    q[order[0]] += 0.05  # the network's first joint is off its default by 0.05
    controller = ot.OnnxTrackerController(still(q=q), spec)
    target = controller.act(control())
    # obs[0] = 2 * 0.05 (scaled joint_pos_rel of the oldest tick, which is the reset state) -> raw 1.0, clip 2 -> 1.0
    raw = np.zeros(29)
    raw[0] = 1.0
    raw[1:] = np.clip(20 * (q[order] - sm.DEFAULT_JOINT_POS[order])[1:], -2, 2)
    np.testing.assert_allclose(target, (raw * 0.5 + sm.DEFAULT_JOINT_POS[order])[np.argsort(order)], atol=1e-6)
    obs = controller.inputs()["obs"]
    np.testing.assert_allclose(obs[29:58], 2.0 * sm.DEFAULT_JOINT_POS[order], atol=1e-6)  # before any action: the default
    controller.act(control())
    obs = controller.inputs()["obs"]  # now, oldest first: [-2] reset, [-1] reset, [0] the first target in force
    np.testing.assert_allclose(obs[-29 * 4 - 29:-29 * 4], 2.0 * (raw * 0.5 + sm.DEFAULT_JOINT_POS[order]), atol=1e-6)


def test_stands_in_its_own_default_pose(tmp_path):
    order, runtime = shuffled_runtime()
    runtime["default_joint_pos"] = (sm.DEFAULT_JOINT_POS[order] + 0.1).tolist()
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, runtime))
    controller = ot.OnnxTrackerController(still(), spec, policy=lambda inputs: np.zeros(29))
    controller.act(control())
    np.testing.assert_allclose(controller.last_reference[0], sm.DEFAULT_JOINT_POS + 0.1)


def test_residual_output_is_added_to_the_reference_now(tmp_path):
    order, runtime = shuffled_runtime()
    runtime["action_offset"] = "ref_joint_pos"
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, runtime))
    assert spec.residual
    controller = ot.OnnxTrackerController(still(), spec, policy=lambda inputs: np.full(29, 0.2))
    target = controller.act(control())
    np.testing.assert_allclose(target, controller.last_reference[0] + 0.5 * 0.2, atol=1e-9)


def test_zeroed_reference_joints_are_zero_in_the_reference_and_the_residual(tmp_path):
    _, runtime = shuffled_runtime()
    runtime.update(action_offset="ref_joint_pos", ref_zero_joints=["left_knee_joint"])
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, runtime))
    controller = ot.OnnxTrackerController(still(), spec, policy=lambda inputs: np.full(29, 0.2))
    target = controller.act(control())
    knee = list(sm.JOINT_NAMES).index("left_knee_joint")
    assert sm.DEFAULT_JOINT_POS[knee] != 0 and controller.last_reference[0][knee] == 0
    assert target[knee] == pytest.approx(0.5 * 0.2)
    with pytest.raises(ValueError, match="ref_zero_joints"):
        ot.TrackerSpec.from_json(spec_file(tmp_path, {**runtime, "ref_zero_joints": ["left_wrist"]}))


def test_yahmp_names_are_the_root_terms():
    q = np.array([np.cos(0.1), np.sin(0.1), 0.05, 0.02])
    q /= np.linalg.norm(q)
    f = ot.Frame(np.zeros(29), np.zeros(29), np.zeros(3), q, np.array([0.3, -0.2, 0.1]), np.array([0.1, 0.2, 0.5]))
    t = ot.REFERENCE_TERMS
    def term(name):
        return t[name](f, f, None, None)
    np.testing.assert_allclose(term("ref_anchor_lin_vel_xy_b"), term("ref_root_lin_vel_xy_local"))
    np.testing.assert_allclose(term("ref_anchor_yaw_rate_b"), term("ref_root_yaw_rate_local"))
    np.testing.assert_allclose(np.concatenate((term("ref_anchor_roll"), term("ref_anchor_pitch"))), term("ref_root_roll_pitch_world"))


@pytest.mark.parametrize("change, message", [
    ({"joint_names": list(sm.JOINT_NAMES[:-1]) + ["left_hand_joint"]}, "29"),
    ({"inputs": {"obs": [{"terms": ["keypoints"]}]}}, "unknown observation terms"),
    ({"inputs": {"obs": [{"terms": ["joint_vel", "ref_joint_pos"], "steps": [0, 1]}]}}, "history ticks"),
    ({"inputs": {"obs": [{"terms": ["joint_vel"], "major": "frame"}]}}, "major"),
    ({"lookahead": 40}, "lookahead"),
])
def test_bad_specs_say_what_is_wrong(tmp_path, change, message):
    _, runtime = shuffled_runtime()
    with pytest.raises(ValueError, match=message):
        ot.TrackerSpec.from_json(spec_file(tmp_path, {**runtime, **change}))


def test_registered():
    assert "onnx_tracker" in registered()


def test_takes_record_the_same_tracking_columns_as_supermimic(tmp_path):
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, SUPERMIMIC_RUNTIME))
    a = sm.SuperMimicController(still(), policy=lambda h, g: np.zeros(29), lookahead=9)
    b = ot.OnnxTrackerController(still(), spec, policy=lambda inputs: np.zeros(29))
    before = b.record()
    assert before["tracking_error"] == 0.0 and before["ref_root_quat"].shape == (4,)
    a.act(control())
    b.act(control())
    ra, rb = a.record(), b.record()
    assert set(rb) <= set(ra)
    for key in rb:
        np.testing.assert_allclose(rb[key], ra[key], atol=1e-6)


def test_observation_noise_is_seeded_reaches_only_the_policy_and_both_controllers_take_it(tmp_path):
    spec = ot.TrackerSpec.from_json(spec_file(tmp_path, SUPERMIMIC_RUNTIME))

    def run(make, seed):
        seen = []
        controller = make(seen, sm.ObservationNoise(seed) if seed is not None else None)
        for _ in range(3):
            controller.act(control())
        return seen, controller

    native = lambda seen, noise: sm.SuperMimicController(still(), policy=lambda h, g: seen.append(h) or np.zeros(29),
                                                         lookahead=9, obs_noise=noise)
    generic = lambda seen, noise: ot.OnnxTrackerController(still(), spec, policy=lambda i: seen.append(i["history"]) or np.zeros(29),
                                                           obs_noise=noise)
    for make in (native, generic):
        clean, _ = run(make, None)
        a, controller = run(make, 1)
        b, _ = run(make, 1)
        c, _ = run(make, 2)
        np.testing.assert_array_equal(a[-1], b[-1])
        assert 0 < np.abs(a[-1] - clean[-1]).max() < 0.5 and not np.allclose(a[-1], c[-1])
        np.testing.assert_array_equal(controller.record()["robot_joint_pos"], sm.DEFAULT_JOINT_POS.astype(np.float32))
        assert controller.last_tracking_error == pytest.approx(0.0, abs=1e-9)


def test_past_frames_hold_the_first_pose_then_follow_the_stream():
    stream = ot.PoseStream(lookahead=0, past=3, dt=0.02)
    order = np.arange(29)
    for i in range(6):
        stream.push(np.full(29, 0.1 * i), np.array([1.0, 0, 0, 0]), np.array([0.01 * i, 0.0, 0.7]))
    assert stream.frame(0, order).joint_pos[0] == pytest.approx(0.5)
    assert stream.frame(-3, order).joint_pos[0] == pytest.approx(0.2)
    assert stream.frame(-1, order).joint_vel[0] == pytest.approx(5.0)  # central
    assert stream.frame(0, order).root_lin_vel[0] == pytest.approx(0.5)  # backward at the newest
    np.testing.assert_allclose(stream.frame(2, order).joint_vel, 0.0)  # beyond the newest: held still
    fresh = ot.PoseStream(lookahead=0, past=3, dt=0.02)
    fresh.push(np.zeros(29), np.array([1.0, 0, 0, 0]), np.zeros(3))
    assert fresh.frame(-3, order).joint_pos[0] == 0.0


def test_terms_against_hand_worked_values():
    q = sm.yaw_quat(0.3)
    roll = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
    np.testing.assert_allclose(ot.roll_pitch(roll), [0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(ot.roll_pitch(q), [0.0, 0.0], atol=1e-12)
    w = ot.angular_velocity(sm.yaw_quat(0.02), np.array([1.0, 0, 0, 0]), 0.02)
    np.testing.assert_allclose(w, [0, 0, 1.0], atol=1e-9)
    now = ot.Frame(np.zeros(29), np.zeros(29), np.array([1.0, 2.0, 0.7]), q, np.zeros(3), np.zeros(3))
    ahead = ot.Frame(np.zeros(29), np.zeros(29), np.array([1.0 + np.cos(0.3), 2.0 + np.sin(0.3), 0.7]), q, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(ot._heading_local(ahead, now), [1.0, 0.0, 0.7], atol=1e-12)
    np.testing.assert_allclose(ot.quat_xyzw(np.array([1.0, 2, 3, 4])), [2, 3, 4, 1])


def test_torso_is_the_pelvis_turned_by_the_waist():
    pytest.importorskip("mujoco")
    if not ot.G1_XML.exists():
        pytest.skip("no GMR G1 model")
    kin = ot.Kinematics()
    q = np.zeros(29)
    _, upright = kin.torso_pose(q, np.zeros(3), np.array([1.0, 0, 0, 0]))
    q[sm.JOINT_NAMES.index("waist_yaw_joint")] = 0.4
    _, turned = kin.torso_pose(q, np.zeros(3), sm.yaw_quat(0.2))
    assert abs(abs(upright[0]) - 1.0) < 1e-6
    assert sm.yaw_of(turned) == pytest.approx(0.6, abs=1e-6)
    assert kin.stand_height(sm.DEFAULT_JOINT_POS) == pytest.approx(sm.STAND_ROOT_HEIGHT)
    straight = sm.DEFAULT_JOINT_POS.copy()
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -0.2), ("knee", 0.4), ("ankle_pitch", -0.2)):
            straight[sm.JOINT_NAMES.index(f"{side}_{joint}_joint")] = value
    assert 0.77 < kin.stand_height(straight) < 0.81  # TWIST2's default; its own sim starts at 0.793


def test_humanoid_gpt_terms_against_hand_worked_values():
    yaw = 0.5
    f = ot.Frame(np.zeros(29), np.zeros(29), np.array([2.0, 1.0, 0.75]), sm.yaw_quat(yaw), np.array([np.cos(yaw), np.sin(yaw), 0.0]),
                 np.array([0.0, 0.0, 0.3]))
    robot = sm.RobotState(sm.yaw_quat(0.2), np.zeros(3), np.zeros(29), np.zeros(29), np.array([1.0, 1.0, 0.75]))
    t = ot.Tick(robot, np.zeros(29), None, None)
    terms = ot.REFERENCE_TERMS
    # moving forward along its own heading at 1 m/s, turning at 0.3 rad/s
    np.testing.assert_allclose(terms["ref_pelvis_cvel_gv"](f, f, t, None), [0, 0, 0.3, 1.0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(terms["heading_err_cos_sin"](f, f, t, None), [np.cos(0.3), np.sin(0.3)], atol=1e-12)
    # the reference is 1 m along world x; the robot faces 0.2 rad left of x
    np.testing.assert_allclose(terms["xy_err_heading"](f, f, t, None), [np.cos(0.2), -np.sin(0.2)], atol=1e-12)
