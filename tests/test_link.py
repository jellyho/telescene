"""The headset link: packet formats and the server over real sockets."""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from telescene import wire as sf
from telescene.isaac.scene import tessellate_capsule, tessellate_cylinder, tessellate_sphere, triangulate
from telescene.server import HeadsetServer
from telescene.session import FollowGate, RisingEdge, Toggle, follow_command, head_in_sim


def _node(pose=-1, rgb=(0.5, 0.5, 0.5)):
    return sf.Node(pose, rgb, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), np.array([0, 1, 2], np.uint32))


def test_scene_round_trips():
    blob = sf.pack_scene([_node(3, (1, 0, 0)), _node()], 7)
    nodes, poses = sf.unpack_scene(blob)
    assert poses == 7 and nodes[0].pose_index == 3 and nodes[0].rgb == (1, 0, 0) and nodes[1].pose_index == -1
    assert nodes[0].vertices.shape == (3, 3) and list(nodes[1].indices) == [0, 1, 2]


def test_merge_keeps_one_draw_per_pose_and_colour_and_rebases_indices():
    merged = sf.merge_nodes([_node(1), _node(1), _node(2)])
    assert len(merged) == 2
    assert list(merged[0].indices) == [0, 1, 2, 3, 4, 5] and len(merged[0].vertices) == 6


def test_input_round_trips_and_has_a_fixed_size_the_c_side_must_match():
    body = np.arange(24 * 7, dtype=np.float32).reshape(24, 7)
    head = np.arange(7, dtype=np.float32)
    frame = sf.InputFrame(9, True, True, False, (0.1, 0.2, 0.3, 0.4), (0.5, 0.6), (0.7, 0.8), head, body)
    packet = sf.pack_input(frame)
    assert len(packet) == 4 + 4 + 4 + 15 * 4 + 24 * 7 * 4 + 14 * 4 == sf.INPUT_SIZE
    out = sf.unpack_input(packet)
    assert out.seq == 9 and out.a_button and out.tracker_ok and not out.all_tracked
    assert out.right_axis == pytest.approx((0.7, 0.8)) and (out.body == body).all()


def test_poses_round_trip():
    poses = np.arange(14, dtype=np.float32).reshape(2, 7)
    seq, got, anchor, yaw, scale, haptics, status = sf.unpack_poses(
        sf.pack_poses(5, poses, (1, 2, 0), 3.0, 0.7, (0.5, 0.25), sf.STATUS_RECORDING))
    assert haptics == (0.5, 0.25) and status == sf.STATUS_RECORDING
    assert seq == 5 and (got == poses).all() and anchor == (1, 2, 0) and yaw == 3.0 and scale == pytest.approx(0.7)


@pytest.mark.parametrize("make,extent", [
    (lambda: tessellate_cylinder(0.1, 1.0), (0.1, 0.1, 0.5)),
    (lambda: tessellate_cylinder(0.1, 1.0, "X"), (0.5, 0.1, 0.1)),
    (lambda: tessellate_capsule(0.1, 1.0, "Y"), (0.1, 0.6, 0.1)),
    (lambda: tessellate_sphere(0.2), (0.2, 0.2, 0.2)),
])
def test_primitives_have_usd_extents_and_valid_indices(make, extent):
    vertices, indices = make()
    assert np.abs(vertices).max(axis=0) == pytest.approx(extent, abs=0.01)
    assert indices.max() < len(vertices) and len(indices) % 3 == 0


def test_triangulate_fans_mixed_faces():
    assert list(triangulate(None, [3, 4], [0, 1, 2, 0, 2, 3, 4])) == [0, 1, 2, 0, 2, 3, 0, 3, 4]


def test_server_sends_scene_then_poses_to_whoever_sent_input():
    blob = sf.pack_scene([_node()], 1)
    server = HeadsetServer(blob, port=0, input_port=0, pose_port=0, beacon_port=None)
    pose_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose_rx.bind(("127.0.0.1", 0))
    pose_rx.settimeout(2)
    server._pose_port = pose_rx.getsockname()[1]
    try:
        with socket.create_connection(("127.0.0.1", server.port)) as conn:
            got = b""
            while len(got) < len(blob):
                got += conn.recv(65536)
            assert got == blob
        poses = np.zeros((1, 7), np.float32)
        server.send_poses(poses, (0, 0, 0), 0.0, 1.0)
        assert server.latest_input() is None and server.packets_sent == 0
        frame = sf.InputFrame(1, False, True, False, (0, 0, 0, 0), (0, 0), (0, 0), np.zeros(7), np.zeros((24, 7)))
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(sf.pack_input(frame), ("127.0.0.1", server.input_port))
        time.sleep(0.05)
        assert server.latest_input().seq == 1
        t = time.perf_counter()
        server.send_poses(poses, (0, 0, 0), 0.0, 1.0)
        assert time.perf_counter() - t < 0.005
        assert sf.unpack_poses(pose_rx.recv(4096))[0] == 1
    finally:
        server.close()


def test_decimate_merges_close_vertices_and_drops_collapsed_triangles():
    v = np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    node = sf.Node(0, (1, 1, 1), v, np.array([0, 1, 2, 0, 2, 3], np.uint32))
    out = sf.decimate(node, 0.01)
    assert len(out.vertices) == 3 and len(out.indices) == 3


def test_srgb_material_names_decode_to_linear():
    from telescene.isaac.scene import srgb_hex_to_linear

    assert srgb_hex_to_linear("ffffff") == pytest.approx((1, 1, 1))
    assert srgb_hex_to_linear("000000") == pytest.approx((0, 0, 0))
    assert srgb_hex_to_linear("808080")[0] == pytest.approx(0.2158, abs=1e-3)


def test_head_in_sim_inverts_the_app_mapping_at_the_robot_heading():
    
    # Anchor at (1.5, 0.2) facing -X. One real metre ahead (-Z) at scale 0.5 is 0.5 m further along -X.
    x, y, yaw = head_in_sim([0, 1.6, -1, 0, 0, 0, 1], (1.5, 0.2, 0), np.pi, 0.5)
    assert (x, y) == pytest.approx((1.0, 0.2)) and abs(abs(yaw) - np.pi) < 1e-6
    # One real metre to the right (+X) is the robot's right: at heading -X that is +Y in the world.
    x, y, _ = head_in_sim([1, 1.6, 0, 0, 0, 0, 1], (1.5, 0.2, 0), np.pi, 0.5)
    assert (x, y) == pytest.approx((1.5, 0.7))


def test_head_yaw_turns_with_a_left_turn_of_the_head():
    
    s = np.sqrt(0.5)  # +90 deg about OpenXR +Y is a turn to the left
    _, _, yaw = head_in_sim([0, 1.6, 0, 0, s, 0, s], (0, 0, 0), 0.0, 1.0)
    assert yaw == pytest.approx(np.pi / 2)


def test_follow_command_walks_at_a_speed_the_planner_can_use():
    assert follow_command((0.1, 0, 0), (0, 0), 0.0, walking=False)[:2] == (0.0, 0.0)
    vx, vy, _ = follow_command((0.1, 0, 0), (0, 0), 0.0)
    assert vx == pytest.approx(0.35) and vy == pytest.approx(0), "close by, but not so slow that it marches on the spot"
    vx, vy, _ = follow_command((2, 0, 0), (0, 0), 0.0)
    assert vx == pytest.approx(0.6) and vy == pytest.approx(0)
    vx, vy, _ = follow_command((-2, 0, np.pi), (0, 0), np.pi)  # target ahead of a robot facing -X
    assert vx == pytest.approx(0.6) and vy == pytest.approx(0, abs=1e-6)
    assert follow_command((0, 0, 3.0), (0, 0), -3.0)[2] < 0, "must turn the short way across +-pi"
    assert follow_command((0, 0, 0.1), (0, 0), 0.0, walking=False)[2] == 0.0, "a small error is left alone"
    assert follow_command((0, 0, 1.0), (0, 0), 0.0, walking=False)[2] > 0


def _input(seq, a=False, tracker=True, stick=(0.0, 0.0), body_value=0.0):
    return sf.InputFrame(seq, a, tracker, False, (0.5, 0, 0, 0), stick, (0, 0), np.zeros(7),
                         np.full((24, 7), body_value, np.float32))


def _poll(server, seq, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        frame = server.latest_input()
        if frame is not None and frame.seq == seq:
            return frame
        time.sleep(0.002)
    raise AssertionError(f"frame {seq} never arrived")


def test_tcp_input_returns_the_newest_frame_and_handles_a_split_one():
    server = HeadsetServer(sf.pack_scene([_node()], 1), port=0, input_port=0, pose_port=0, beacon_port=None)
    try:
        with socket.create_connection(("127.0.0.1", server.input_port)) as tx:
            tx.sendall(b"".join(sf.pack_input(_input(i)) for i in range(1, 6)))
            assert _poll(server, 5).seq == 5
            assert server.packets_received == 5
            packet = sf.pack_input(_input(6))
            tx.sendall(packet[:100])
            time.sleep(0.02)
            assert server.latest_input().seq == 5
            tx.sendall(packet[100:])
            assert _poll(server, 6).seq == 6
    finally:
        server.close()


def test_tcp_input_survives_reconnect_and_garbage_and_never_blocks():
    server = HeadsetServer(sf.pack_scene([_node()], 1), port=0, input_port=0, pose_port=0, beacon_port=None)
    try:
        t = time.perf_counter()
        assert server.latest_input() is None
        assert time.perf_counter() - t < 0.005
        with socket.create_connection(("127.0.0.1", server.input_port)) as tx:
            tx.sendall(sf.pack_input(_input(1)))
            _poll(server, 1)
        end = time.monotonic() + 1
        while server._input_conn is not None and time.monotonic() < end:
            server.latest_input()
        assert server._input_conn is None
        with socket.create_connection(("127.0.0.1", server.input_port)) as tx:
            tx.sendall(b"x" * sf.INPUT_SIZE)
            end = time.monotonic() + 1
            while server._input_conn is None and time.monotonic() < end:
                server.latest_input()
            while server._input_conn is not None and time.monotonic() < end:
                server.latest_input()
            assert server._input_conn is None
        with socket.create_connection(("127.0.0.1", server.input_port)) as tx:
            tx.sendall(sf.pack_input(_input(9)))
            assert _poll(server, 9).seq == 9
    finally:
        server.close()


def test_rising_edge_fires_once_per_press():
    edge = RisingEdge()
    assert [edge(p) for p in (False, True, True, False, True)] == [False, True, False, False, True]


def test_sticks_pause_following_until_the_operator_walks():
    gate = FollowGate(stick=0.1, walk=0.3)
    still, head = (0.0, 0.0), (0.0, 1.6, 0.0)
    assert gate(still, still, head)
    assert not gate((0.0, 0.8), still, head)  # steering
    assert not gate(still, still, head)  # let go: stays where the stick left it
    assert not gate(still, still, (0.2, 1.6, 0.0))  # shifting on the spot is not walking
    assert gate(still, still, (0.4, 1.6, 0.0))  # walked away: follows again
    assert gate(still, still, (0.0, 1.6, 0.0))


def test_turning_pauses_following_but_hip_height_does_not():
    gate = FollowGate()
    head = (0.0, 1.6, 0.0)
    assert gate((0.0, 0.0), (0.0, 0.9), head)
    assert not gate((0.0, 0.0), (0.9, 0.0), head)


def test_follow_gate_reset_hands_control_back():
    gate = FollowGate()
    head = (0.0, 1.6, 0.0)
    gate((0.5, 0.0), (0.0, 0.0), head)
    gate.reset()
    assert gate((0.0, 0.0), (0.0, 0.0), head)


def test_beacon_is_broadcast_for_an_app_with_no_address():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", 0))
    rx.settimeout(3)
    server = HeadsetServer(sf.pack_scene([_node()], 1), port=0, input_port=0, pose_port=0,
                           beacon_port=rx.getsockname()[1])
    try:
        assert rx.recv(16) == sf.BEACON
    finally:
        server.close()
        rx.close()


def test_mode_flags_survive_the_wire():
    frame = _input(1)
    frame.mirror = True
    out = sf.unpack_input(sf.pack_input(frame))
    assert out.mirror is True and out.menu is False
    frame = _input(2)
    frame.menu = True
    out = sf.unpack_input(sf.pack_input(frame))
    assert out.menu is True and out.mirror is False


def _read_scene(conn, size):
    got = b""
    conn.settimeout(3)
    while len(got) < size:
        got += conn.recv(65536)
    return got


def test_app_asks_for_a_scene_and_a_silent_client_gets_the_main_one():
    main, mirror = sf.pack_scene([_node()], 1), sf.pack_scene([_node(), _node(1)], 2)
    server = HeadsetServer(main, port=0, input_port=0, pose_port=0, beacon_port=None, mirror_blob=mirror)
    try:
        with socket.create_connection(("127.0.0.1", server.port)) as conn:
            conn.sendall(sf.SCENE_REQUEST_MIRROR)
            assert _read_scene(conn, len(mirror)) == mirror
        with socket.create_connection(("127.0.0.1", server.port)) as conn:
            conn.sendall(b"S")
            assert _read_scene(conn, len(main)) == main
        with socket.create_connection(("127.0.0.1", server.port)) as conn:
            assert _read_scene(conn, len(main)) == main
    finally:
        server.close()


def test_grips_survive_the_wire_and_default_to_invalid():
    frame = _input(1)
    frame.grips = np.arange(14, dtype=np.float32).reshape(2, 7)
    assert (sf.unpack_input(sf.pack_input(frame)).grips == frame.grips).all()
    assert (sf.unpack_input(sf.pack_input(_input(2))).grips == 0).all()


def _ask(port, request: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
        s.sendall(request)
        out = b""
        while chunk := s.recv(4096):
            out += chunk
        return out


def test_headset_lists_scenes_and_a_pick_stops_serving():
    import struct

    server = HeadsetServer(sf.pack_scene([_node()], 1), port=0, input_port=0, pose_port=0, beacon_port=None,
                           scenes=["Full: palette + cart + stool", "Stool only"], current_scene=1)
    try:
        listing = _ask(server.port, b"L")
        assert listing[:4] == b"XLST" and struct.unpack("<II", listing[4:12]) == (2, 1)
        first = listing[12]
        assert listing[13:13 + first] == b"Full: palette + cart + stool"
        assert _ask(server.port, b"C\x05") == b"NO" and server.requested_scene is None
        assert _ask(server.port, b"C\x01") == b"OK" and server.requested_scene == 1
    finally:
        server.close()


def test_scene_groups_go_out_before_each_name_and_the_list_is_capped():
    import struct

    from telescene.wire import MAX_SCENES, SCENE_GROUP_SEPARATOR

    server = HeadsetServer(sf.pack_scene([_node()], 1), port=0, input_port=0, pose_port=0, beacon_port=None,
                           scenes=["open_top_drawer", "box_to_cart"], scene_groups=["drawers", "carry"], current_scene=0)
    try:
        listing = _ask(server.port, b"L")
        assert struct.unpack("<II", listing[4:12]) == (2, 0)
        first = listing[12]
        assert listing[13:13 + first].decode() == "drawers" + SCENE_GROUP_SEPARATOR + "open_top_drawer"
    finally:
        server.close()
    with pytest.raises(ValueError):
        HeadsetServer(b"", port=0, input_port=0, pose_port=0, beacon_port=None, scenes=["s"] * (MAX_SCENES + 1))
    with pytest.raises(ValueError):
        HeadsetServer(b"", port=0, input_port=0, pose_port=0, beacon_port=None, scenes=["a", "b"], scene_groups=["g"])


def test_a_client_that_asks_for_nothing_still_gets_the_scene():
    blob = sf.pack_scene([_node()], 1)
    server = HeadsetServer(blob, port=0, input_port=0, pose_port=0, beacon_port=None)
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as s:
            got = b""
            while len(got) < len(blob):
                got += s.recv(65536)
        assert got == blob
    finally:
        server.close()


@pytest.mark.parametrize("yaw_room", [0.0, 0.7, -2.5])
def test_recenter_puts_the_operator_at_the_spawn_facing_its_way(yaw_room):
    from telescene.session import recenter

    # a head at room (0.4, 1.6, -0.9) looking yaw_room to the left of -Z
    q = (0.0, np.sin(yaw_room / 2), 0.0, np.cos(yaw_room / 2))
    head = np.array([0.4, 1.6, -0.9, *q])
    anchor, anchor_yaw = recenter(head, None, (1.5, 0.2), np.pi, 0.78)
    x, y, yaw = head_in_sim(head, anchor, anchor_yaw, 0.78)
    assert (x, y) == pytest.approx((1.5, 0.2)) and np.cos(yaw - np.pi) == pytest.approx(1.0)
    pelvis = np.array([0.5, 0.9, -1.0])
    anchor, anchor_yaw = recenter(head, pelvis, (1.5, 0.2), np.pi, 0.78)
    x, y, _ = head_in_sim(np.array([*pelvis, *q]), anchor, anchor_yaw, 0.78)
    assert (x, y) == pytest.approx((1.5, 0.2))
