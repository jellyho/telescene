"""The loop against a fake simulator and controller: no Isaac, no headset, real behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from telescene import AutoScale, ControlInput, haptic_amplitude, make_controller, register, registered
from telescene.session import CommandHold, TeleopSession, WalkDecision, scale_for_height, skeleton_height
from telescene.wire import InputFrame


def frame(seq=1, a=False, b=False, tracker=True, not_worn=False, left=(0.0, 0.0), head=(0.0, 1.6, 0.0), body=1.0):
    return InputFrame(seq, a, tracker, True, (0.5, 0, 0, 0), left, (0.0, 0.0),
                      np.array([*head, 0, 0, 0, 1], np.float32), np.full((24, 7), body, np.float32), not_worn, b)


class FakeServer:
    packets_received = packets_sent = 0

    def __init__(self):
        self.next = None

    def latest_input(self):
        return self.next

    def send_poses(self, *args, status=0):
        self.packets_sent += 1
        self.haptics = args[4] if len(args) > 4 else (0.0, 0.0)
        self.status = args[5] if len(args) > 5 else status

    def take_input_gap_ms(self):
        return 0.0


class FakeSim:
    def __init__(self):
        self.resets, self.fall, self.actions, self.forces = 0, False, [], (0.0, 0.0)

    def reset(self):
        self.resets += 1

    def step(self, action):
        self.actions.append(action)
        fell, self.fall = self.fall, False
        return fell

    def root_pose(self):
        return (0.0, 0.0, 0.0)

    def link_poses(self):
        return np.zeros((1, 7), np.float32)

    def anchor(self):
        return (0.0, 0.0, 0.0), 0.0

    def hand_forces(self):
        return self.forces


class FakeController:
    def __init__(self):
        self.last: ControlInput | None = None
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, control):
        self.last = control
        return "action"

    def status(self):
        return ""


@pytest.fixture
def rig():
    sim, controller, server, lines = FakeSim(), FakeController(), FakeServer(), []
    return TeleopSession(sim, controller, server, log=lines.append), sim, controller, server, lines


def test_runs_with_no_headset_and_sends_poses_anyway(rig):
    session, sim, controller, server, _ = rig
    session.tick()
    assert controller.last.frame is None and controller.last.body is None and not controller.last.tracking
    assert controller.last.velocity == (0.0, 0.0, 0.0)
    assert sim.actions == ["action"] and server.packets_sent == 1


def test_a_toggles_tracking_and_holding_it_does_not_flicker(rig):
    session, _, controller, server, _ = rig
    states = []
    for a in (True, True, False, True):
        server.next = frame(a=a)
        session.tick()
        states.append(controller.last.tracking)
    assert states == [True, True, True, False]


def test_quiet_headset_holds_posture_and_tracking_but_stops_the_sticks(rig):
    session, _, controller, server, _ = rig
    server.next = frame(a=True, left=(0.0, 0.9), body=2.0)
    session.tick()
    assert controller.last.left_axis == (0.0, 0.9)
    server.next = None
    session.tick()
    control = controller.last
    assert control.quiet and control.tracking and control.left_axis == (0.0, 0.0)
    assert control.triggers[0] == 0.5 and float(control.body[0, 0]) == 2.0


def test_tracker_dropout_keeps_the_last_body(rig):
    session, _, controller, server, _ = rig
    server.next = frame(body=1.0)
    session.tick()
    server.next = frame(tracker=False, body=9.0)
    session.tick()
    assert float(controller.last.body[0, 0]) == 1.0 and not controller.last.quiet


def test_taking_the_headset_off_ends_tracking_until_a_new_press(rig):
    session, _, controller, server, lines = rig
    server.next = frame(a=True)
    session.tick()
    assert controller.last.tracking
    server.next = frame(a=True, not_worn=True)
    session.tick()
    assert not controller.last.tracking and any("came off" in line for line in lines)
    server.next = frame(a=False)
    session.tick()
    server.next = frame(a=True)
    session.tick()
    assert controller.last.tracking


def test_a_fall_resets_the_controller_and_ends_tracking(rig):
    session, sim, controller, server, lines = rig
    server.next = frame(a=True)
    session.tick()
    sim.fall = True
    session.tick()
    assert controller.resets == 1 and not session.tracking.on and any("reset" in line for line in lines)


def test_b_resets_once_per_press_and_only_when_live(rig):
    session, sim, controller, server, _ = rig
    server.next = frame(a=True)
    session.tick()
    for b in (True, True):
        server.next = frame(a=True, b=b)
        session.tick()
    assert sim.resets == 1 and controller.resets == 1 and not session.tracking.on
    server.next = frame(b=False)
    session.tick()
    server.next = None
    session.tick()
    assert sim.resets == 1


def test_a_stale_b_press_does_not_reset(rig):
    session, sim, _, server, _ = rig
    session.tick()
    server.next = frame(b=False)
    session.tick()
    server.next = None
    session.frame = frame(b=True)
    session.tick()
    assert sim.resets == 0


def test_robot_walks_to_the_operator_only_with_tracking_off(rig):
    session, _, controller, server, _ = rig
    server.next = frame(head=(0.0, 1.6, -2.0))  # 2 m ahead
    session.tick()
    assert controller.last.velocity[0] > 0.3
    server.next = frame(a=True, head=(0.0, 1.6, -2.0))
    session.tick()
    assert controller.last.velocity == (0.0, 0.0, 0.0)


def test_robot_does_not_walk_to_a_headset_that_came_off(rig):
    session, _, controller, server, _ = rig
    server.next = frame(not_worn=True, head=(0.0, 1.6, -2.0))
    session.tick()
    assert controller.last.velocity == (0.0, 0.0, 0.0)


def test_steering_stops_following_until_the_operator_walks(rig):
    session, _, controller, server, _ = rig
    server.next = frame(left=(0.0, 0.9), head=(0.0, 1.6, -2.0))
    session.tick()
    assert controller.last.velocity == (0.0, 0.0, 0.0) and controller.last.left_axis == (0.0, 0.9)
    server.next = frame(head=(0.0, 1.6, -2.0))
    session.tick()
    assert controller.last.velocity == (0.0, 0.0, 0.0)
    server.next = frame(head=(0.0, 1.6, -2.5))
    session.tick()
    assert controller.last.velocity[0] > 0.3


def test_report_shows_what_an_operator_needs(rig):
    session, _, _, server, lines = rig
    server.next = frame()
    session.tick()
    session._report()
    report = next(line for line in lines if "tick p50" in line)
    assert "tracking off" in report and "root 0.00 0.00" in report and any("measured:" in line for line in lines)


def test_run_holds_the_rate_and_stops(rig):
    session, sim, *_ = rig
    session.run(seconds=0.2)
    assert 6 <= len(sim.actions) <= 12


def test_registry_builds_by_name_and_rejects_mistakes():
    name = "test-fake"
    register(name, FakeController)
    assert name in registered() and isinstance(make_controller(name), FakeController)
    with pytest.raises(ValueError):
        register(name, FakeController)
    with pytest.raises(KeyError, match="Registered"):
        make_controller("no-such")


def _person(height_scale=1.0):
    """A body in the SMPL layout."""
    body = np.zeros((24, 7), np.float32)
    body[:, 6] = 1
    joints = {0: (0, 0.95, 0), 3: (0, 1.05, 0), 6: (0, 1.17, 0), 9: (0, 1.30, 0), 12: (0, 1.50, 0), 15: (0, 1.60, 0),
              1: (-0.1, 0.87, 0), 2: (0.1, 0.87, 0), 4: (-0.1, 0.47, 0), 5: (0.1, 0.47, 0), 7: (-0.1, 0.08, 0), 8: (0.1, 0.08, 0)}
    for index, xyz in joints.items():
        body[index, :3] = np.array(xyz) * height_scale
    return body


def test_skeleton_height_is_about_the_person_and_scales_with_them():
    tall = skeleton_height(_person())
    assert 1.6 < tall < 1.9
    assert skeleton_height(_person(1.1)) > tall


def test_skeleton_height_does_not_change_when_the_person_moves():
    moved = _person()
    moved[:, :3] += np.array([1.0, -0.5, 2.0], np.float32)
    assert skeleton_height(moved) == pytest.approx(skeleton_height(_person()), abs=1e-4)


def _scale_frame(body, not_worn=False, tracked=True):
    return InputFrame(1, False, True, tracked, (0, 0, 0, 0), (0, 0), (0, 0), np.zeros(7, np.float32), body, not_worn)


def test_auto_scale_decides_once_from_a_worn_tracked_body():
    auto = AutoScale(1.32, samples=3)
    body = _person()
    assert auto(_scale_frame(body, not_worn=True)) is None
    assert auto(_scale_frame(body, tracked=False)) is None and auto(None) is None
    assert auto(_scale_frame(body)) is None and auto(_scale_frame(body)) is None
    assert auto(_scale_frame(body)) == pytest.approx(1.32 / skeleton_height(body))
    assert auto(_scale_frame(_person(1.1))) is None


def test_auto_scale_rejects_a_nonsense_skeleton():
    auto = AutoScale(1.32, samples=1)
    assert auto(_scale_frame(np.zeros((24, 7), np.float32))) is None
    assert auto.done and auto.operator_height is None


def test_session_applies_auto_scale_and_keeps_the_fallback_on_failure():
    sim, controller, server, lines = FakeSim(), FakeController(), FakeServer(), []
    session = TeleopSession(sim, controller, server, world_scale=0.7, auto_scale=AutoScale(1.32, samples=1),
                            log=lines.append)
    server.next = _scale_frame(_person())
    session.tick()
    assert session.world_scale == pytest.approx(1.32 / skeleton_height(_person()))
    sim, controller, server, lines = FakeSim(), FakeController(), FakeServer(), []
    session = TeleopSession(sim, controller, server, world_scale=0.7, auto_scale=AutoScale(1.32, samples=1),
                            log=lines.append)
    server.next = _scale_frame(np.zeros((24, 7), np.float32))
    session.tick()
    session.tick()
    assert session.world_scale == 0.7 and any("could not measure" in line for line in lines)


def test_scale_for_height_is_the_ratio():
    assert scale_for_height(1.365, 1.82) == pytest.approx(0.75)


def test_haptics_tell_a_touch_from_a_bump():
    assert haptic_amplitude(0.0) == 0.0 and haptic_amplitude(2.9) == 0.0
    assert haptic_amplitude(3.0) == pytest.approx(0.15)
    assert haptic_amplitude(50.0) < 0.4 < haptic_amplitude(150.0) < haptic_amplitude(200.0) == 1.0
    assert haptic_amplitude(694.0) == 1.0


def test_haptic_gain_scales_and_turns_off():
    assert haptic_amplitude(50.0, gain=0.5) == pytest.approx(haptic_amplitude(50.0) / 2)
    assert haptic_amplitude(694.0, gain=0.0) == 0.0


def test_session_sends_each_hands_vibration_and_reports_the_peak(rig):
    session, sim, _, server, lines = rig
    sim.forces = (200.0, 0.0)
    session.tick()
    assert server.haptics == (1.0, 0.0)
    sim.forces = (0.0, 0.0)
    session.tick()
    assert server.haptics == (0.0, 0.0)
    session._report()
    assert "hand force peak 200 0 N" in lines[-1]


class FakeMirror:
    def __init__(self):
        self.seen = 0

    def act(self, control):
        self.seen += 1

    def anchor(self):
        return (0.0, 0.0, 0.0), 0.0

    def link_poses(self):
        return np.zeros((3, 7), np.float32)

    def status(self):
        return ""


def _mode_rig():
    sim, controller, server, mirror = FakeSim(), FakeController(), FakeServer(), FakeMirror()
    server_calls = []
    original = server.send_poses

    def send(*args, **kwargs):
        server_calls.append(args)
        original(*args, **kwargs)

    server.send_poses = send
    return TeleopSession(sim, controller, server, mirror=mirror, log=lambda line: None), sim, controller, server, mirror, server_calls


def _mode_frame(**kw):
    f = frame(**{k: v for k, v in kw.items() if k in ("a", "b", "left", "head")})
    f.mirror, f.menu = kw.get("mirror", False), kw.get("menu", False)
    return f


def test_mirror_mode_sends_mirror_poses_and_keeps_the_robot_still():
    session, sim, controller, server, mirror, calls = _mode_rig()
    server.next = _mode_frame(mirror=True, a=True, b=True, left=(0.0, 0.9), head=(0.0, 1.6, -2.0))
    session.tick()
    assert mirror.seen == 1 and len(calls[-1][0]) == 3 and calls[-1][3] == 1.0
    assert not controller.last.tracking and controller.last.velocity == (0.0, 0.0, 0.0)
    assert controller.last.left_axis == (0.0, 0.0) and sim.resets == 0 and len(sim.actions) == 1


def test_menu_keeps_the_robot_still_and_shows_no_mirror():
    session, sim, controller, server, mirror, calls = _mode_rig()
    server.next = _mode_frame(menu=True, a=True, head=(0.0, 1.6, -2.0))
    session.tick()
    assert mirror.seen == 0 and len(calls[-1][0]) == 1
    assert not controller.last.tracking and controller.last.velocity == (0.0, 0.0, 0.0)


def test_leaving_mirror_mode_needs_a_new_press_to_track():
    session, _, controller, server, *_ = _mode_rig()
    server.next = _mode_frame(mirror=True, a=True)
    session.tick()
    server.next = _mode_frame(a=True)
    session.tick()
    assert not controller.last.tracking
    server.next = _mode_frame()
    session.tick()
    server.next = _mode_frame(a=True)
    session.tick()
    assert controller.last.tracking


def test_a_controller_with_a_planner_keeps_walking_while_tracking(rig):
    session, _, controller, server, _ = rig
    controller.walks_while_tracking = True
    server.next = frame(a=True, head=(0.0, 1.6, -2.0))
    session.tick()
    assert controller.last.tracking and controller.last.velocity[0] > 0.3
    assert controller.last.world_scale == 1.0


def test_hold_keeps_small_changes_out_until_they_matter():
    hold = CommandHold(speed=0.1, turn=0.25, direction=0.5, hold=0.6)
    assert hold((0.2, 0.0, 0.0), 0.0) == (0.2, 0.0, 0.0)  # starting goes straight through
    assert hold((0.21, 0.01, 0.05), 0.1) == (0.2, 0.0, 0.0)  # noise, too soon
    assert hold((0.21, 0.01, 0.05), 5.0) == (0.2, 0.0, 0.0)  # noise, however late
    assert hold((0.45, 0.0, 0.0), 5.1) == (0.45, 0.0, 0.0)  # a real change after the hold
    assert hold((0.1, 0.0, 0.0), 5.2) == (0.45, 0.0, 0.0)  # a real change, too soon
    assert hold((0.1, 0.0, 0.0), 5.8) == (0.1, 0.0, 0.0)


def test_hold_stops_at_once():
    hold = CommandHold()
    hold((0.3, 0.0, 0.0), 0.0)
    assert hold((0.0, 0.0, 0.0), 0.01) == (0.0, 0.0, 0.0)


def test_hold_sees_a_change_of_direction_and_of_turning():
    hold = CommandHold(direction=0.5, turn=0.25, hold=0.0)
    hold((0.3, 0.0, 0.0), 0.0)
    assert hold((0.0, 0.3, 0.0), 1.0) == (0.0, 0.3, 0.0)
    assert hold((0.0, 0.3, 0.5), 2.0) == (0.0, 0.3, 0.5)


def test_hold_turns_on_the_spot_only_when_it_matters():
    hold = CommandHold(turn=0.25, hold=0.0)
    assert hold((0.0, 0.0, 0.05), 0.0) == (0.0, 0.0, 0.0)
    assert hold((0.0, 0.0, 0.5), 1.0) == (0.0, 0.0, 0.5)


def test_walk_decision_starts_far_and_stops_close():
    walk = WalkDecision(start=0.25, stop=0.1)
    assert [walk(d) for d in (0.2, 0.3, 0.2, 0.11, 0.05, 0.2)] == [False, True, True, True, False, False]


def _rec_frame(y=False, a=False):
    f = frame(a=a, body=2.0)
    f.y_button = y
    return f


def test_y_starts_and_stops_a_take_and_nothing_is_recorded_before(tmp_path):
    from telescene.recording import Recorder

    sim, controller, server = FakeSim(), FakeController(), FakeServer()
    sim.snapshot = lambda: {"art/robot/joints": np.arange(3, dtype=np.float32)}
    takes = iter(range(10))
    session = TeleopSession(sim, controller, server, recorder=lambda: Recorder(tmp_path / f"take{next(takes)}.npz"),
                            log=lambda line: None)
    for y in (False, False, True, True, False, False, False, True, False, True, False):
        server.next = _rec_frame(y=y, a=True)
        session.tick()
    assert [p.name for p in session.saved] == ["take0.npz"]
    data = np.load(tmp_path / "take0.npz")
    assert len(data["time"]) == 5 and data["art/robot/joints"].shape == (5, 3)
    assert data["tracking"].all() and float(data["body"][0, 0, 0]) == 2.0
    session.stop_recording()  # the session ending saves the open take
    assert [p.name for p in session.saved] == ["take0.npz", "take1.npz"]
    assert len(np.load(tmp_path / "take1.npz")["time"]) == 2


def test_starting_a_take_buzzes_and_a_held_or_stale_y_does_nothing(tmp_path):
    from telescene.recording import Recorder

    sim, controller, server = FakeSim(), FakeController(), FakeServer()
    session = TeleopSession(sim, controller, server, recorder=lambda: Recorder(tmp_path / "t.npz"), log=lambda line: None)
    server.next = _rec_frame(y=True)
    session.tick()
    assert session.recorder is not None and server.haptics == (1.0, 1.0) and server.status == 1
    server.next = None  # quiet: the last frame, Y down, is held but must not count
    session.tick()
    assert session.recorder is not None


def test_recorder_grows_past_a_block(tmp_path):
    from telescene import recording

    r = recording.Recorder(tmp_path / "long.npz")
    for i in range(recording.BLOCK + 3):
        r.add(x=float(i))
    r.save()
    assert np.load(tmp_path / "long.npz")["x"][-1] == recording.BLOCK + 2
