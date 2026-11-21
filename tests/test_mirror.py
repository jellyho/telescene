"""The mirror scene against a fake server: geometry, and a whole session with no simulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from telescene import wire
from telescene.mirror import MirrorView, bone_pose, joint_errors, reflect, xr_to_sim
from telescene.session import TeleopSession
from telescene.wire import InputFrame

PARENTS = [-1, 0, 1]
LENGTHS = [0.0, 0.5, 0.3]


def body():
    b = np.zeros((3, 7), np.float32)
    b[:, 6] = 1
    b[:, :3] = [(0, 1.0, 0), (0, 1.5, 0), (0.3, 1.5, 0)]
    return b


class FakeServer:
    packets_received = packets_sent = 0

    def __init__(self, frame):
        self.frame = frame
        self.poses = None

    def latest_input(self):
        return self.frame

    def send_poses(self, poses, *args):
        self.poses = poses.copy()

    def take_input_gap_ms(self):
        return 0.0


def test_axes_match_the_app_and_the_mirror_flips_only_depth():
    assert xr_to_sim(np.array([1.0, 2.0, 3.0])).tolist() == [-3.0, -1.0, 2.0]
    out = reflect(np.array([[0.3, 1.5, -0.2]]), 1.0)
    assert out[0].tolist() == pytest.approx([0.3, 1.5, -1.8])


@pytest.mark.parametrize("end", [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0.3, -0.4, 0.5), (0, 0, -1)])
def test_bone_pose_puts_a_z_capsule_between_the_points(end):
    start, end = np.zeros(3), np.array(end, float)
    pose = bone_pose(start, end)
    assert pose[:3] == pytest.approx((start + end) / 2)
    axis = Rotation.from_quat(pose[3:7]).apply([0, 0, 1])
    assert axis == pytest.approx(end / np.linalg.norm(end), abs=1e-5)


def test_joint_errors_ignore_where_the_body_is():
    tracked = body()[:, :3].astype(float)
    assert joint_errors(tracked, tracked + 5.0) == pytest.approx(np.zeros(3))
    moved = tracked.copy()
    moved[2, 0] += 0.1
    assert joint_errors(tracked, moved)[2] == pytest.approx(0.1)


def _session(convert):
    view = MirrorView(PARENTS, LENGTHS, convert=convert, distance=1.0, names=["pelvis", "neck", "hand"])
    frame = InputFrame(1, False, True, True, (0, 0, 0, 0), (0, 0), (0, 0), np.array([0, 1.6, 0, 0, 0, 0, 1], np.float32), body())
    server = FakeServer(frame)
    return view, server, TeleopSession(view, view, server, log=lambda line: None)


def test_scene_has_a_part_per_dot_figure_dot_and_bone_and_unpacks():
    view, *_ = _session(lambda b: b[:, :3])
    nodes, poses = wire.unpack_scene(view.scene_blob)
    assert poses == 3 + 3 + 2
    assert sorted({n.pose_index for n in nodes}) == [-1, *range(8)]


def test_dots_only_scene_without_a_conversion():
    view, *_ = _session(None)
    assert wire.unpack_scene(view.scene_blob)[1] == 3


def test_body_appears_in_the_mirror_about_twice_the_distance_away():
    view, server, session = _session(lambda b: b[:, :3])
    session.tick()
    pelvis = server.poses[0]
    assert pelvis[:3] == pytest.approx([2.0, 0.0, 1.0])  # 2 m ahead, height unchanged
    assert view.status().startswith("SMPL vs tracked: mean 0.0 cm")


def test_figure_sits_on_the_tracked_pelvis_and_reports_what_differs():
    def shifted(b):
        out = b[:, :3].astype(float) + 7.0
        out[2, 0] += 0.2
        return out

    view, server, session = _session(shifted)
    session.tick()
    assert server.poses[3][:3] == pytest.approx(server.poses[0][:3])
    assert "hand 20" in view.status()


def test_waits_below_the_floor_until_a_body_arrives():
    view, server, session = _session(lambda b: b[:, :3])
    server.frame = None
    session.tick()
    assert (server.poses[:, 2] < 0).all() and view.status() == ""
