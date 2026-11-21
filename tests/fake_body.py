"""A synthetic operator: a rough human skeleton in the format of ``XR_BD_body_tracking``. Numpy only.

Joints in OpenXR room axes, x right, y up, z back, with the operator facing -z, and every joint's orientation the identity
in a T-pose, which is what SONIC's PICO path assumes. Bone lengths are rough SMPL values typed in by hand, for a person of
about 1.75 m. Nobody real moved like this, so passing with it says nothing about a real tracker's noise.

Run it to stream the motion to a telescene server:  python tests/fake_body.py --seconds 45 --press 2
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
_OFFSETS_FACING_Z = np.array([
    (0, 0, 0), (0.058, -0.082, -0.018), (-0.060, -0.091, -0.014), (0.004, 0.109, -0.023), (0.043, -0.386, 0.008),
    (-0.043, -0.383, -0.005), (0.005, 0.136, 0.003), (-0.015, -0.427, -0.037), (0.019, -0.420, -0.035),
    (0.0, 0.056, 0.027), (0.041, -0.060, 0.122), (-0.035, -0.063, 0.130), (-0.013, 0.212, -0.034),
    (0.072, 0.120, -0.019), (-0.083, 0.117, -0.023), (0.010, 0.089, 0.050), (0.123, 0.045, -0.009),
    (-0.113, 0.048, -0.009), (0.255, -0.015, -0.027), (-0.260, -0.014, -0.031), (0.265, 0.013, -0.001),
    (-0.269, 0.007, -0.006), (0.087, -0.011, -0.016), (-0.089, -0.009, -0.010)])
OFFSETS = _OFFSETS_FACING_Z * np.array([-1.0, 1.0, -1.0])
"""Turned to face -z: the operator's left is -x."""
HIP, KNEE, ANKLE = (1, 2), (4, 5), (7, 8)
SHOULDER, ELBOW = (16, 17), (18, 19)
SPINE1, SPINE3, PELVIS = 3, 9, 0


def _axis_angle(axis, angle: float) -> np.ndarray:
    """xyzw."""
    axis = np.asarray(axis, np.float64)
    return np.array([*(axis * math.sin(angle / 2)), math.cos(angle / 2)])


def _mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rotate(q, v):
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def forward_kinematics(local: dict[int, np.ndarray], root_pos=(0.0, 0.0, 0.0), root=None) -> np.ndarray:
    """``(24, 7)`` from rotations of each joint against its parent, given in the room's axes at the T-pose."""
    body = np.zeros((24, 7))
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    body[0, :3], body[0, 3:7] = root_pos, identity if root is None else root
    for joint in range(1, 24):
        parent = body[PARENTS[joint]]
        body[joint, :3] = parent[:3] + _rotate(parent[3:7], OFFSETS[joint])
        body[joint, 3:7] = _mul(parent[3:7], local.get(joint, identity))
    return body


def stand_on_floor(body: np.ndarray) -> np.ndarray:
    """Move a body so that the lower ankle is 8 cm above the floor, in the middle of the room."""
    body = body.copy()
    body[:, 1] += 0.08 - min(body[ANKLE[0], 1], body[ANKLE[1], 1])
    return body


def pose(arms_down: float = 1.3, left_raise: float = 0.0, right_raise: float = 0.0, elbow: float = 0.3,
         squat: float = 0.0, lean: float = 0.0, turn: float = 0.0) -> np.ndarray:
    """Radians. ``arms_down`` from a T-pose, each raise lifts that arm forward, ``squat`` bends both hips, ``lean`` forward."""
    x, y, z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)  # the same rotation about x takes any limb forward
    local = {
        HIP[0]: _axis_angle(x, squat), HIP[1]: _axis_angle(x, squat),  # thigh forward is +x at the hip: forward is -z
        KNEE[0]: _axis_angle(x, -2 * squat), KNEE[1]: _axis_angle(x, -2 * squat),
        ANKLE[0]: _axis_angle(x, squat), ANKLE[1]: _axis_angle(x, squat),
        SPINE1: _axis_angle(x, -lean),
        SHOULDER[0]: _mul(_axis_angle(x, left_raise), _axis_angle(z, arms_down)),
        SHOULDER[1]: _mul(_axis_angle(x, right_raise), _axis_angle(z, -arms_down)),
        ELBOW[0]: _axis_angle(y, -elbow), ELBOW[1]: _axis_angle(y, elbow),
    }
    return stand_on_floor(forward_kinematics(local, root=_axis_angle(y, turn)))


def scripted(t: float) -> np.ndarray:
    """Stand, wave each arm, squat a little, lean, swing both arms, turn a little. Repeats every 30 s."""
    t = t % 30.0
    def bump(a, b):  # 0 outside, smoothly 1 in the middle
        return math.sin(math.pi * (t - a) / (b - a)) ** 2 if a < t < b else 0.0
    return pose(left_raise=1.2 * bump(5, 10) * (1 + 0.3 * math.sin(6 * t)),
                right_raise=1.2 * bump(10, 15) * (1 + 0.3 * math.sin(6 * t)),
                elbow=0.3 + 0.8 * bump(5, 15),
                squat=0.45 * bump(15, 19),
                lean=0.25 * bump(19, 22),
                arms_down=1.3 - 0.8 * bump(22, 25) * math.sin(4 * t) ** 2,
                turn=0.5 * bump(25, 30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--press", type=float, default=2.0, help="press A at this time. Negative never presses")
    parser.add_argument("--still", action="store_true", help="stand still instead of the scripted motion")
    parser.add_argument("--walk", type=float, default=0.0, help="slide the whole body forward at this speed, m/s, without stepping")
    parser.add_argument("--b-at", type=float, default=-1.0, help="press B, reset, at this time")
    parser.add_argument("--off-at", type=float, default=-1.0, help="press A again at this time, to end tracking")
    parser.add_argument("--noise", type=float, default=0.0, help="std of tracker noise: this many metres on positions, ten times this many radians on orientations")
    parser.add_argument("--fall-at", type=float, default=-1.0, help="from this time show a body lying down, which no robot can copy")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from telescene.wire import INPUT_PORT, InputFrame, pack_input

    for _ in range(600):
        try:
            link = socket.create_connection((args.host, INPUT_PORT))
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise SystemExit("no server")
    link.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    start, seq = time.time(), 0
    rng = np.random.default_rng(0)
    while (t := time.time() - start) < args.seconds:
        body = (pose() if args.still else scripted(t)).astype(np.float32)
        if 0 <= args.fall_at < t:
            body = stand_on_floor(forward_kinematics({}, root=_axis_angle((1.0, 0.0, 0.0), 1.5))).astype(np.float32)
        body[:, 2] -= args.walk * t
        if args.noise:
            body[:, :3] += rng.normal(0.0, args.noise, (24, 3))
            body[:, 3:7] += rng.normal(0.0, 5.0 * args.noise, (24, 4))  # half the angle, in a quaternion
            body[:, 3:7] /= np.linalg.norm(body[:, 3:7], axis=1, keepdims=True)
        seq += 1
        head = np.concatenate((body[15, :3], body[15, 3:7])).astype(np.float32)
        frame = InputFrame(seq, (0 <= args.press < t < args.press + 0.5) or (0 <= args.off_at < t < args.off_at + 0.5), True, True,
                           (0, 0, 0, 0), (0, 0), (0, 0), head, body, False, 0 <= args.b_at < t < args.b_at + 0.5)
        link.sendall(pack_input(frame))
        time.sleep(1 / 90)
    print("fake body done", seq, flush=True)


if __name__ == "__main__":
    main()
