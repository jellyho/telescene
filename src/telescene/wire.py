# SPDX-License-Identifier: MIT
"""Wire formats between the PC and the headset app (apps/pico/main.c). Stdlib and numpy only.

The headset renders the scene itself, so head motion never crosses the network and the PC renders
nothing. The PC sends geometry once over TCP and then only poses of what moves.

Everything is little-endian. Scene coordinates are the simulator's: metres, Z up.

    TCP  scene   "XSCN" u32 version, u32 num_poses, u32 num_nodes, then per node:
                 i32 pose_index (-1 static), 3f rgb, u32 num_vertices, u32 num_indices,
                 3f * num_vertices, u32 * num_indices
    UDP  poses   "XPOS" u32 seq, u32 n, 3f anchor, f anchor_yaw, f world_scale, 2f haptics (left, right, 0 to 1),
                 u32 status (STATUS_RECORDING), n * (3f pos, 4f xyzw)
    TCP  input   (or UDP, chosen on the headset) "XINP" u32 seq, u32 flags, 8f (lt, lg, rt, rg, lx, ly, rx, ry), 7f head,
                 24 * 7f body (xyz, xyzw; quaternion all zero when that joint is not valid),
                 2 * 7f controller grips (left, right; the same)

``world_scale`` is simulator metres per real metre: below 1 the operator is smaller than life.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SCENE_PORT, POSE_PORT, INPUT_PORT, BEACON_PORT = 47800, 47801, 47802, 47803
PROGRESS_PORT = 47804
"""While the PC starts up: ``progress.StartupProgress`` answers each connection with how far it has got."""
BEACON = b"XBCN"
"""Broadcast by the PC so that the app can find it without being told an address."""
VERSION = 1
BODY_JOINTS = 24

FLAG_A_BUTTON, FLAG_TRACKER, FLAG_ALL_TRACKED, FLAG_NOT_WORN, FLAG_B_BUTTON = 1, 2, 4, 8, 16
FLAG_MIRROR, FLAG_MENU, FLAG_Y_BUTTON = 32, 64, 128
STATUS_RECORDING = 1
"""In the pose packet's status: a take is being recorded, so the app shows it."""
SCENE_REQUEST_MIRROR = b"M"
SCENE_REQUEST_LIST = b"L"
"""Asks for the scenes the PC offers: ``pack_scene_list``."""
SCENE_REQUEST_CHOOSE = b"C"
"""Followed by one byte, the index of the scene to restart with. The PC answers ``OK``."""


SCENE_GROUP_SEPARATOR = "\x1f"
"""Between a scene's group and its name in ``pack_scene_list``; the headset menu shows one tab per group."""
MAX_SCENES = 96
"""The most scenes the headset menu takes. A pick is one byte, so never more than 256."""


def pack_scene_list(names: list[str], current: int, groups: list[str] | None = None) -> bytes:
    """``XLST``, count, current (u32 each), then per scene one length byte and its ASCII name, cut to 47 bytes.

    With ``groups``, each name goes out as ``group + SCENE_GROUP_SEPARATOR + name``.
    """
    import struct

    if len(names) > MAX_SCENES:
        raise ValueError(f"{len(names)} scenes, the headset menu takes {MAX_SCENES}")
    if groups is not None and len(groups) != len(names):
        raise ValueError("one group per scene")
    out = [b"XLST", struct.pack("<II", len(names), current)]
    for i, name in enumerate(names):
        if groups is not None:
            name = groups[i][:16] + SCENE_GROUP_SEPARATOR + name
        raw = name.encode("ascii", "replace")[:47]
        out += [bytes([len(raw)]), raw]
    return b"".join(out)
"""The app's first byte on the scene connection in mirror mode. Anything else, or nothing, is the main scene."""

_SCENE_HEADER = struct.Struct("<4sIII")
_NODE_HEADER = struct.Struct("<i3fII")
_POSE_HEADER = struct.Struct("<4sII7fI")
_INPUT = struct.Struct("<4sII" + "8f" + "7f" + f"{BODY_JOINTS * 7}f" + "14f")


@dataclass
class Node:
    pose_index: int
    rgb: tuple[float, float, float]
    vertices: np.ndarray
    indices: np.ndarray


def merge_nodes(nodes: list[Node]) -> list[Node]:
    """One node per (pose, colour): one draw call each, which is what a mobile GPU wants."""
    groups: dict[tuple, list[Node]] = {}
    for node in nodes:
        key = (node.pose_index, tuple(round(float(c), 3) for c in node.rgb))
        groups.setdefault(key, []).append(node)
    merged = []
    for (pose_index, rgb), members in groups.items():
        offsets = np.cumsum([0] + [len(m.vertices) for m in members[:-1]])
        merged.append(Node(pose_index, rgb, np.concatenate([m.vertices for m in members]),
                           np.concatenate([m.indices + o for m, o in zip(members, offsets)])))
    return merged


def decimate(node: Node, cell: float) -> Node:
    """Merge vertices into a ``cell``-metre grid and drop collapsed triangles (vertex clustering).

    Crude, and fine here: the headset shades flat, so there are no normals to keep consistent, and
    a robot link seen from a metre away does not need sub-centimetre detail.
    """
    if cell <= 0 or len(node.indices) == 0:
        return node
    keys = np.floor(node.vertices / cell).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(first), 3), np.float64)
    np.add.at(sums, inverse, node.vertices)
    vertices = (sums / np.bincount(inverse, minlength=len(first))[:, None]).astype(np.float32)
    tris = inverse[node.indices].reshape(-1, 3)
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return Node(node.pose_index, node.rgb, vertices, tris[keep].ravel().astype(np.uint32))


def pack_scene(nodes: list[Node], num_poses: int) -> bytes:
    parts = [_SCENE_HEADER.pack(b"XSCN", VERSION, num_poses, len(nodes))]
    for n in nodes:
        parts.append(_NODE_HEADER.pack(n.pose_index, *n.rgb, len(n.vertices), len(n.indices)))
        parts.append(np.asarray(n.vertices, "<f4").tobytes())
        parts.append(np.asarray(n.indices, "<u4").tobytes())
    return b"".join(parts)


def unpack_scene(blob: bytes) -> tuple[list[Node], int]:
    magic, version, num_poses, count = _SCENE_HEADER.unpack_from(blob)
    if magic != b"XSCN" or version != VERSION:
        raise ValueError("not a scene blob of this version")
    offset, nodes = _SCENE_HEADER.size, []
    for _ in range(count):
        pose_index, r, g, b, nv, ni = _NODE_HEADER.unpack_from(blob, offset)
        offset += _NODE_HEADER.size
        vertices = np.frombuffer(blob, "<f4", nv * 3, offset).reshape(-1, 3)
        offset += nv * 12
        indices = np.frombuffer(blob, "<u4", ni, offset)
        offset += ni * 4
        nodes.append(Node(pose_index, (r, g, b), vertices, indices))
    return nodes, num_poses


def pack_poses(seq: int, poses: np.ndarray, anchor, anchor_yaw: float, world_scale: float, haptics=(0.0, 0.0),
               status: int = 0) -> bytes:
    """``poses`` is ``(n, 7)``: xyz, xyzw."""
    return _POSE_HEADER.pack(b"XPOS", seq & 0xFFFFFFFF, len(poses), *anchor, anchor_yaw, world_scale, *haptics, status) + \
        np.asarray(poses, "<f4").tobytes()


def unpack_poses(packet: bytes):
    magic, seq, n, x, y, z, yaw, scale, left, right, status = _POSE_HEADER.unpack_from(packet)
    if magic != b"XPOS":
        raise ValueError("not a pose packet")
    return seq, np.frombuffer(packet, "<f4", n * 7, _POSE_HEADER.size).reshape(-1, 7), (x, y, z), yaw, scale, (left, right), status


@dataclass
class InputFrame:
    seq: int
    a_button: bool
    tracker_ok: bool
    all_tracked: bool
    triggers: tuple[float, float, float, float]  # lt, lg, rt, rg
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    head: np.ndarray
    body: np.ndarray
    not_worn: bool = False
    b_button: bool = False
    mirror: bool = False
    menu: bool = False
    y_button: bool = False
    grips: np.ndarray | None = None
    """``(2, 7)``: where the controllers really are, unlike the body joints, which are estimated."""


def pack_input(frame: InputFrame) -> bytes:
    flags = (FLAG_A_BUTTON if frame.a_button else 0) | (FLAG_TRACKER if frame.tracker_ok else 0) | \
        (FLAG_ALL_TRACKED if frame.all_tracked else 0) | (FLAG_NOT_WORN if frame.not_worn else 0) | \
        (FLAG_B_BUTTON if frame.b_button else 0) | \
        (FLAG_MIRROR if frame.mirror else 0) | (FLAG_MENU if frame.menu else 0) | \
        (FLAG_Y_BUTTON if frame.y_button else 0)
    return _INPUT.pack(b"XINP", frame.seq, flags, *frame.triggers, *frame.left_axis, *frame.right_axis,
                       *np.asarray(frame.head).ravel(), *np.asarray(frame.body).ravel(),
                       *(np.zeros(14, np.float32) if frame.grips is None else np.asarray(frame.grips).ravel()))


def unpack_input(packet: bytes) -> InputFrame:
    values = _INPUT.unpack(packet)
    if values[0] != b"XINP":
        raise ValueError("not an input packet")
    flags, v = values[2], np.array(values[3:], np.float32)
    return InputFrame(values[1], bool(flags & FLAG_A_BUTTON), bool(flags & FLAG_TRACKER), bool(flags & FLAG_ALL_TRACKED),
                      tuple(v[:4]), tuple(v[4:6]), tuple(v[6:8]), v[8:15], v[15:15 + BODY_JOINTS * 7].reshape(BODY_JOINTS, 7), bool(flags & FLAG_NOT_WORN), bool(flags & FLAG_B_BUTTON),
                      bool(flags & FLAG_MIRROR), bool(flags & FLAG_MENU),
                      bool(flags & FLAG_Y_BUTTON), v[15 + BODY_JOINTS * 7:].reshape(2, 7))


INPUT_SIZE = _INPUT.size
