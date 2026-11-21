# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Any whole-body G1 tracker that ships as ONNX, described by a spec instead of by code.

Everything a tracker shares with SuperMimic is SuperMimic's: the GMR retarget, blending between standing and the
operator, reading the robot. What differs is written down in a ``TrackerSpec``: the joint order the network uses, its
default pose and action scale, and how its observation is laid out. So adding a tracker is a spec and, for an observation
term nobody has needed yet, one function in ``ROBOT_TERMS`` or ``REFERENCE_TERMS``.

An observation is a list of blocks. A block is some terms over some steps:

    {"terms": ["joint_pos_rel", "joint_vel"], "steps": [-4, -3, -2, -1, 0], "major": "term"}

Robot terms take history ticks as steps, 0 being now and -k k ticks ago. Reference terms take frames: 0 is the frame the
robot should be at now, k > 0 is k control ticks ahead, k < 0 is k ticks back. A block may mix the two over steps 0 and
down, as Isaac Lab's history of a whole observation group does. Steps are laid out in the order listed,
so a newest-first policy lists them newest first. ``major: "term"`` puts each term's steps together (Isaac Lab's history
layout); ``"time"`` puts each step's terms together. ``scale`` is one number or one per element of a step.

A reference frame k <= 0 is paired with the robot as it was at that tick, for terms relative to the robot (``ref_*_rel``,
``*_in_robot_torso``); later frames with the robot now.

The reference is the operator through GMR: joints, pelvis orientation and height, and the pelvis's horizontal path, which
starts under the robot when tracking engages and turns with the same yaw offset as the orientation. Torso terms come from
forward kinematics of GMR's G1 model (``mujoco``, only loaded when a term needs it).

``tests/trackers/test_onnx_tracker.py`` writes SuperMimic's observation as a spec and checks it matches ``history_obs`` and
``goal_obs`` exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..controller import ControlInput, register
from .supermimic import (CONTROL_DT, GMR_ROOT, REPO_ROOT, JOINT_NAMES, NUM_JOINTS, RobotState, SuperMimicController, isaac_state_reader,
                         projected_gravity, quat_conjugate, quat_matrix, quat_mul, yaw_of)

WEIGHTS_DIR = Path(os.environ.get("TELESCENE_TRACKER_WEIGHTS") or REPO_ROOT / "checkpoints/trackers")
"""Where ``scripts/download_trackers.sh`` puts each spec's ONNX, as ``<slug>.onnx``."""
G1_XML = Path(GMR_ROOT) / "assets/unitree_g1/g1_mocap_29dof.xml"
GRAVITY = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Block:
    terms: tuple[str, ...]
    steps: tuple[int, ...] = (0,)
    major: str = "term"
    scale: float | tuple[float, ...] = 1.0
    clip: float | None = None

    def __post_init__(self):
        if self.major not in ("term", "time"):
            raise ValueError(f"major must be 'term' or 'time', not {self.major!r}")
        unknown = [t for t in self.terms if t not in ROBOT_TERMS and t not in REFERENCE_TERMS]
        if unknown:
            raise ValueError(f"unknown observation terms {unknown}. Robot: {sorted(ROBOT_TERMS)}; reference: {sorted(REFERENCE_TERMS)}")
        if self.robot and max(self.steps) > 0:
            raise ValueError(f"a block with robot terms takes history ticks, 0 and down: {self.steps}")

    @property
    def reference(self) -> bool:
        """Has reference terms, so needs reference frames."""
        return any(t in REFERENCE_TERMS for t in self.terms)

    @property
    def robot(self) -> bool:
        """Has robot terms, so needs history ticks."""
        return any(t in ROBOT_TERMS for t in self.terms)


@dataclass(frozen=True)
class TrackerSpec:
    """What the network expects. Arrays are in ``joint_names`` order, the network's own.

    Attributes:
        slug: The tracker's name, with the variant for a file of several (``mosaic_pico_teleop``).
        onnx: The checkpoint.
        joint_names: The 29 joints in the order the network reads and writes them.
        default_joint_pos: ``joint_pos_rel``'s zero, and the pose the robot stands in.
        action_scale: Radians per unit of output, one or 29.
        action_offset: Added after scaling. Usually the default pose; with ``residual``, used only before the first tick.
        inputs: ONNX input name to its blocks, concatenated in order.
        output: The ONNX output holding the action, or None for the first.
        constant_inputs: Inputs fed the same value every tick, e.g. a time step a graph wants but a live stream lacks.
        action_clip: Clip the raw output to this, before scaling, or None.
        control_dt: Seconds per policy tick. The runner's decimation has to match.
        lookahead: Real future frames: the reference is this many ticks late, so frames up to it are real.
        stand_root_height: Pelvis height of ``default_joint_pos`` with the feet on the ground. None: worked out by
            ``Kinematics.stand_height`` when a controller is made.
        last_action: What ``last_action`` observes: the raw output or the joint target.
        sha256: The checkpoint's, checked on load when given.
        residual: The output is a residual over the reference: the offset is frame 0's ``ref_joint_pos`` each tick
            (``action_offset: "ref_joint_pos"`` in the spec, as yahmp's).
        ref_zero_joints: Joints whose reference is held at 0 (so their reference velocity is 0 too, and a residual adds to
            0): those the policy only ever saw at 0 in training, as yahmp's wrists.
    """

    slug: str
    onnx: Path
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    inputs: dict[str, tuple[Block, ...]]
    output: str | None = None
    constant_inputs: dict[str, np.ndarray] = field(default_factory=dict)
    action_clip: float | None = None
    control_dt: float = CONTROL_DT
    lookahead: int = 0
    stand_root_height: float | None = None
    last_action: str = "raw"
    sha256: str | None = None
    residual: bool = False
    ref_zero_joints: tuple[str, ...] = ()

    def __post_init__(self):
        if sorted(self.joint_names) != sorted(JOINT_NAMES):
            raise ValueError(f"joint_names must be the G1's 29: missing {set(JOINT_NAMES) - set(self.joint_names)}, "
                             f"extra {set(self.joint_names) - set(JOINT_NAMES)}")
        for name in ("default_joint_pos", "action_scale", "action_offset"):
            if getattr(self, name).shape != (NUM_JOINTS,):
                raise ValueError(f"{name} must have 29 numbers, has {getattr(self, name).shape}")
        if set(self.ref_zero_joints) - set(JOINT_NAMES):
            raise ValueError(f"ref_zero_joints names no G1 joint: {set(self.ref_zero_joints) - set(JOINT_NAMES)}")
        if self.last_action not in ("raw", "target"):
            raise ValueError("last_action is 'raw' or 'target'")
        if not 0 <= self.lookahead < self.frames + 16:
            raise ValueError(f"lookahead {self.lookahead} is out of range")

    @property
    def order(self) -> np.ndarray:
        """``x[order]`` takes a ``JOINT_NAMES`` array to the network's order."""
        return np.array([JOINT_NAMES.index(n) for n in self.joint_names])

    def _steps(self, reference: bool) -> list[int]:
        return [s for blocks in self.inputs.values() for b in blocks if (b.reference if reference else b.robot) for s in b.steps]

    @property
    def history(self) -> int:
        """Ticks of robot state kept: robot terms' and those reference frames paired with the robot back then."""
        return 1 - min(self._steps(False) + self._steps(True) + [0])

    @property
    def frames(self) -> int:
        return 1 + max(self._steps(True) + [0])

    @property
    def past(self) -> int:
        return -min(self._steps(True) + [0])

    @property
    def needs_kinematics(self) -> bool:
        return any(t in KINEMATIC_TERMS for blocks in self.inputs.values() for b in blocks for t in b.terms)

    @classmethod
    def from_json(cls, path: str | Path, weights_dir: str | Path = WEIGHTS_DIR) -> "TrackerSpec":
        """Reads a spec's ``runtime`` block, or the whole file if it has none. ``file.json#variant`` picks one of several
        runtime blocks. The checkpoint is ``runtime.onnx`` relative to the file, else ``weights_dir/<slug>.onnx``.

        ``action_scale`` may be one number, ``action_offset`` may be ``"default"`` or ``"ref_joint_pos"`` (a residual). The lookahead is ``runtime.lookahead``,
        else ``reference.lookahead_s`` in ticks.
        """
        path, _, variant = str(path).partition("#")
        path = Path(path)
        doc = json.loads(path.read_text())
        raw = doc.get("runtime", doc)
        if "inputs" not in raw:
            blocks = sorted(k for k, v in raw.items() if isinstance(v, dict) and "inputs" in v)
            if variant not in blocks:
                raise ValueError(f"{path.name} has runtime blocks {blocks}: pick one with {path.name}#<name>")
            raw = raw[variant]
        elif variant:
            raise ValueError(f"{path.name} has one runtime block, not '{variant}'")
        slug = raw.get("slug") or path.stem + (f"_{variant}" if variant else "")
        joint_names = tuple(raw.get("joint_names") or doc.get("joint_order"))
        default = np.asarray(raw.get("default_joint_pos") or doc["default_joint_pos"], np.float64)
        offset = raw.get("action_offset", "default")
        residual = offset == "ref_joint_pos"
        if residual:
            offset = "default"
        if isinstance(offset, str) and offset != "default":
            raise ValueError(f"{slug}: action_offset {offset!r} is not supported yet")
        control_dt = float(raw.get("control_dt") or CONTROL_DT)
        reference = raw.get("reference") or {}
        lookahead = raw.get("lookahead")
        if lookahead is None:
            lookahead = round(float(reference.get("lookahead_s") or 0.0) / control_dt)
        onnx = (path.parent / raw["onnx"]) if raw.get("onnx") else Path(weights_dir) / f"{slug}.onnx"

        def block(b):
            scale = b.get("scale", 1.0)
            return Block(tuple(b["terms"]), tuple(b.get("steps", (0,))), b.get("major", "term"),
                         tuple(float(x) for x in scale) if isinstance(scale, list) else float(1.0 if scale is None else scale), b.get("clip"))

        return cls(
            slug=slug, onnx=onnx.resolve(), joint_names=joint_names, default_joint_pos=default,
            action_scale=np.broadcast_to(np.asarray(raw["action_scale"], np.float64), (NUM_JOINTS,)).copy(),
            action_offset=default.copy() if offset == "default" else np.asarray(offset, np.float64),
            inputs={name: tuple(block(b) for b in blocks) for name, blocks in raw["inputs"].items()},
            output=raw.get("output"), constant_inputs={k: np.asarray(v, np.float32) for k, v in (raw.get("constant_inputs") or {}).items()},
            action_clip=raw.get("action_clip"), control_dt=control_dt, lookahead=int(lookahead),
            stand_root_height=raw.get("stand_root_height"), last_action=raw.get("last_action") or "raw",
            sha256=raw.get("sha256"), residual=residual, ref_zero_joints=tuple(raw.get("ref_zero_joints") or ()))


# ------------------------------------------------------------------------------------------------ kinematics
def quat_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def roll_pitch(q: np.ndarray) -> np.ndarray:
    """ZYX Euler roll and pitch of a wxyz quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))])


def ori_6d_cols(r: np.ndarray) -> np.ndarray:
    """First two columns, row-major: ``[r00, r01, r10, r11, r20, r21]``."""
    return r[:, :2].reshape(-1)


def angular_velocity(q_now: np.ndarray, q_before: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity taking ``q_before`` to ``q_now`` in ``dt``, both wxyz."""
    d = quat_mul(q_now, quat_conjugate(q_before))
    if d[0] < 0:
        d = -d
    s = np.linalg.norm(d[1:])
    if s < 1e-12:
        return np.zeros(3)
    return d[1:] / s * (2.0 * math.atan2(s, d[0])) / dt


class Kinematics:
    """The torso_link of GMR's G1 model, for any joints and pelvis pose. Joints in ``JOINT_NAMES`` order, quats wxyz."""

    def __init__(self, xml: str | Path = G1_XML):
        import mujoco

        self.mujoco, self.model = mujoco, mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.torso = self.model.body("torso_link").id

    def stand_height(self, joint_pos: np.ndarray) -> float:
        """Pelvis height that puts the lower ankle of ``joint_pos`` where SuperMimic's default pose puts it. So a
        straighter default stands taller, as it does on the robot."""
        def ankle(q):
            self.torso_pose(q, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
            return min(self.data.xpos[self.model.body(n).id][2] for n in ("left_ankle_roll_link", "right_ankle_roll_link"))

        from .supermimic import DEFAULT_JOINT_POS, STAND_ROOT_HEIGHT

        return float(STAND_ROOT_HEIGHT + ankle(DEFAULT_JOINT_POS) - ankle(joint_pos))

    def torso_pose(self, joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = 0.0
        self.data.qpos[:3], self.data.qpos[3:7] = root_pos, root_quat
        self.data.qpos[self.qadr] = joint_pos
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.torso].copy(), self.data.xquat[self.torso].copy()


# ------------------------------------------------------------------------------------------- the reference
@dataclass
class Frame:
    """One reference frame. Joints in the network's order; positions in the simulator's world; velocities world-frame."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    torso_pos: np.ndarray | None = None
    torso_quat: np.ndarray | None = None
    torso_lin_vel: np.ndarray | None = None
    torso_ang_vel: np.ndarray | None = None


class PoseStream:
    """Poses one per tick, read back as frames around now: ``lookahead`` ticks behind the newest, with ``past`` more kept.

    Velocities follow ``supermimic.ReferenceStream``: central differences inside the real frames, a backward difference at
    the newest, zero beyond it, where the newest is held. Before the first pose, the first is held. So frames 0 and on are
    exactly ``ReferenceStream.window``'s.
    """

    def __init__(self, lookahead: int, past: int, dt: float, kinematics: Kinematics | None = None):
        self.lookahead, self.past, self.dt, self.kinematics = lookahead, past, dt, kinematics
        self._poses: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=past + lookahead + 2)
        self._torso: deque = deque(maxlen=past + lookahead + 2)

    def reset(self) -> None:
        self._poses.clear()
        self._torso.clear()

    def push(self, joint_pos: np.ndarray, root_quat: np.ndarray, root_pos: np.ndarray) -> None:
        """``joint_pos`` in ``JOINT_NAMES`` order, ``root_quat`` wxyz, ``root_pos`` in the simulator's world."""
        pose = (np.asarray(joint_pos, np.float64), np.asarray(root_quat, np.float64), np.asarray(root_pos, np.float64))
        self._poses.append(pose)
        self._torso.append(self.kinematics.torso_pose(pose[0], pose[2], pose[1]) if self.kinematics is not None else None)

    def frame(self, k: int, order: np.ndarray) -> Frame:
        n = len(self._poses) - 1
        i = n - self.lookahead + k

        def at(j):
            return self._poses[min(max(j, 0), n)]

        def torso(j):
            return self._torso[min(max(j, 0), n)]

        q, quat, pos = at(i)
        zero = np.zeros(3)
        if i > n:
            a = None
        elif i == n:
            a, b, span = i, i - 1, self.dt
        else:
            a, b, span = i + 1, i - 1, 2 * self.dt
        if a is None:
            qd, v, w = np.zeros(NUM_JOINTS), zero, zero
        else:
            qd = (at(a)[0] - at(b)[0]) / span
            v = (at(a)[2] - at(b)[2]) / span
            w = angular_velocity(at(a)[1], at(b)[1], span)
        out = Frame(q[order], qd[order], pos, quat, v, w)
        if self.kinematics is not None:
            out.torso_pos, out.torso_quat = torso(i)
            if a is None:
                out.torso_lin_vel = out.torso_ang_vel = zero
            else:
                out.torso_lin_vel = (torso(a)[0] - torso(b)[0]) / span
                out.torso_ang_vel = angular_velocity(torso(a)[1], torso(b)[1], span)
        return out


# ---------------------------------------------------------------------------------------- observation terms
@dataclass
class Tick:
    """The robot at one tick as the policy saw it, the action in force then, and its torso if a term needs it."""

    state: RobotState
    action: np.ndarray
    target: np.ndarray | None = None
    torso_quat: np.ndarray | None = None


# (tick, spec) -> vector. State joints are in JOINT_NAMES order; actions and targets in the network's.
ROBOT_TERMS: dict[str, Callable[[Tick, TrackerSpec], np.ndarray]] = {
    "base_ang_vel": lambda t, spec: t.state.ang_vel_b,
    "base_lin_vel": lambda t, spec: t.state.lin_vel_b,
    "projected_gravity": lambda t, spec: projected_gravity(t.state.quat),
    "base_roll_pitch": lambda t, spec: roll_pitch(t.state.quat),
    "joint_pos": lambda t, spec: t.state.joint_pos[spec.order],
    "joint_pos_rel": lambda t, spec: t.state.joint_pos[spec.order] - spec.default_joint_pos,
    "joint_vel": lambda t, spec: t.state.joint_vel[spec.order],
    "last_action": lambda t, spec: t.action,
    "prev_joint_pos_target": lambda t, spec: np.zeros(NUM_JOINTS) if t.target is None else t.target,
    "anchor_quat_world_xyzw": lambda t, spec: quat_xyzw(t.torso_quat),
}


def _rel(robot_quat: np.ndarray, frame_quat: np.ndarray) -> np.ndarray:
    return quat_mul(quat_conjugate(robot_quat / np.linalg.norm(robot_quat)), frame_quat)


def _torso_rel(t: Tick, f: Frame) -> np.ndarray:
    return quat_matrix(t.torso_quat).T @ quat_matrix(f.torso_quat)


def _heading_local(f: Frame, now: Frame) -> np.ndarray:
    d = f.root_pos - np.array([now.root_pos[0], now.root_pos[1], 0.0])
    yaw = yaw_of(now.root_quat)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])


def _yaw_frame(q: np.ndarray) -> np.ndarray:
    """World to the heading frame of a wxyz quaternion: rotation by minus its yaw."""
    c, s = math.cos(yaw_of(q)), math.sin(yaw_of(q))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _cos_sin(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def _robot_xy(t: "Tick") -> np.ndarray:
    return np.zeros(2) if t.state.pos is None else np.asarray(t.state.pos[:2], np.float64)


# (frame, frame 0, the robot's tick paired with the frame, spec) -> vector.
REFERENCE_TERMS: dict[str, Callable[[Frame, Frame, Tick, TrackerSpec], np.ndarray]] = {
    "ref_joint_pos": lambda f, now, t, spec: f.joint_pos,
    "ref_joint_pos_rel": lambda f, now, t, spec: f.joint_pos - spec.default_joint_pos,
    "ref_joint_vel": lambda f, now, t, spec: f.joint_vel,
    "ref_root_height": lambda f, now, t, spec: f.root_pos[2:3],
    "ref_root_ori_6d": lambda f, now, t, spec: ori_6d_cols(quat_matrix(_rel(t.state.quat, f.root_quat))),
    "ref_root_ori_6d_rows": lambda f, now, t, spec: quat_matrix(_rel(t.state.quat, f.root_quat))[:2, :].reshape(-1),
    "ref_root_quat_rel": lambda f, now, t, spec: _rel(t.state.quat, f.root_quat),
    "ref_root_gravity": lambda f, now, t, spec: projected_gravity(f.root_quat),
    "ref_root_roll_pitch_world": lambda f, now, t, spec: roll_pitch(f.root_quat),
    "ref_root_lin_vel_xy_local": lambda f, now, t, spec: (quat_matrix(f.root_quat).T @ f.root_lin_vel)[:2],
    "ref_root_yaw_rate_local": lambda f, now, t, spec: (quat_matrix(f.root_quat).T @ f.root_ang_vel)[2:3],
    # yahmp's names: the same, and roll and pitch one at a time
    "ref_anchor_lin_vel_xy_b": lambda f, now, t, spec: (quat_matrix(f.root_quat).T @ f.root_lin_vel)[:2],
    "ref_anchor_yaw_rate_b": lambda f, now, t, spec: (quat_matrix(f.root_quat).T @ f.root_ang_vel)[2:3],
    "ref_anchor_roll": lambda f, now, t, spec: roll_pitch(f.root_quat)[:1],
    "ref_anchor_pitch": lambda f, now, t, spec: roll_pitch(f.root_quat)[1:],
    # humanoid_gpt's: the reference pelvis's world twist [angular, linear] in its own yaw-only frame (the online path, a
    # body-origin twist); heading and xy error of the robot against the reference, in the robot's heading frame
    "ref_pelvis_cvel_gv": lambda f, now, t, spec: np.concatenate((_yaw_frame(f.root_quat) @ f.root_ang_vel,
                                                                  _yaw_frame(f.root_quat) @ f.root_lin_vel)),
    "heading_err_cos_sin": lambda f, now, t, spec: _cos_sin(yaw_of(f.root_quat) - yaw_of(t.state.quat)),
    "xy_err_heading": lambda f, now, t, spec: (_yaw_frame(t.state.quat) @ np.append(f.root_pos[:2] - _robot_xy(t), 0.0))[:2],
    "ref_pelvis_pos_heading_local": lambda f, now, t, spec: _heading_local(f, now),
    "textop_ref_anchor_pos_b": lambda f, now, t, spec: quat_matrix(t.state.quat).T @ (f.root_pos - t.state.pos),
    "ref_torso_ori_6d": lambda f, now, t, spec: ori_6d_cols(_torso_rel(t, f)),
    "ref_torso_ori_6d_rel": lambda f, now, t, spec: ori_6d_cols(_torso_rel(t, f)),
    "ref_torso_lin_vel_in_robot_torso": lambda f, now, t, spec: quat_matrix(t.torso_quat).T @ f.torso_lin_vel,
    "ref_torso_ang_vel_in_robot_torso": lambda f, now, t, spec: quat_matrix(t.torso_quat).T @ f.torso_ang_vel,
    "ref_torso_gravity": lambda f, now, t, spec: GRAVITY @ quat_matrix(f.torso_quat),
    "ref_torso_height": lambda f, now, t, spec: f.torso_pos[2:3],
    "ref_anchor_quat_world_xyzw": lambda f, now, t, spec: quat_xyzw(f.torso_quat),
}
KINEMATIC_TERMS = {"anchor_quat_world_xyzw", "ref_torso_ori_6d", "ref_torso_ori_6d_rel", "ref_torso_lin_vel_in_robot_torso",
                   "ref_torso_ang_vel_in_robot_torso", "ref_torso_gravity", "ref_torso_height", "ref_anchor_quat_world_xyzw"}
"""Terms that need the torso, robot's or reference's: they load ``Kinematics``."""


def build_block(block: Block, spec: TrackerSpec, ticks: list[Tick], frame: Callable[[int], Frame]) -> np.ndarray:
    """One block's numbers. ``ticks`` are oldest first and end now; ``frame(k)`` is reference frame ``k``."""
    now = frame(0) if block.reference else None

    def one(term: str, step: int) -> np.ndarray:
        tick = ticks[len(ticks) - 1 + min(step, 0)]
        if term in REFERENCE_TERMS:
            return REFERENCE_TERMS[term](frame(step), now, tick, spec)
        return ROBOT_TERMS[term](tick, spec)

    if block.major == "term":
        parts = [one(t, s) for t in block.terms for s in block.steps]
    else:
        parts = [one(t, s) for s in block.steps for t in block.terms]
    out = np.concatenate(parts)
    if isinstance(block.scale, tuple):
        if len(out) % len(block.scale):
            raise ValueError(f"block {block.terms}: {len(block.scale)} scales do not divide {len(out)} numbers")
        out = out * np.tile(block.scale, len(out) // len(block.scale))
    else:
        out = out * block.scale
    return np.clip(out, -block.clip, block.clip) if block.clip is not None else out


# -------------------------------------------------------------------------------------------- the network
class OnnxPolicy:
    """Runs the checkpoint on the CPU. ``__call__`` takes the built inputs and returns the raw action, 29 numbers."""

    def __init__(self, spec: TrackerSpec, threads: int = 2):
        import onnxruntime as ort

        if not spec.onnx.exists():
            raise FileNotFoundError(f"{spec.slug}: no checkpoint at {spec.onnx}; download it (see the spec's onnx_url)")
        if spec.sha256 and hashlib.sha256(spec.onnx.read_bytes()).hexdigest() != spec.sha256:
            raise ValueError(f"{spec.slug}: {spec.onnx} does not match the spec's sha256")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(spec.onnx), options, providers=["CPUExecutionProvider"])
        self.input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
        expected = set(spec.inputs) | set(spec.constant_inputs)
        if expected != set(self.input_shapes):
            raise ValueError(f"{spec.onnx.name} wants inputs {sorted(self.input_shapes)}, the spec gives {sorted(expected)}")
        outputs = [o.name for o in self.session.get_outputs()]
        self.output = spec.output or outputs[0]
        if self.output not in outputs:
            raise ValueError(f"{spec.onnx.name} has outputs {outputs}, not '{self.output}'")
        self.constants = spec.constant_inputs
        self._checked = False

    def _shape(self, name: str) -> list[int]:
        return [d if isinstance(d, int) else 1 for d in self.input_shapes[name]]

    def check(self, inputs: dict[str, np.ndarray]) -> None:
        """Every input's size against the graph's. A mismatch means the spec's layout is wrong: refuse to run."""
        for name, value in inputs.items():
            want = int(np.prod(self._shape(name)))
            if want != value.size:
                raise ValueError(f"input '{name}': the spec builds {value.size} numbers, the graph wants {self.input_shapes[name]}")

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if not self._checked:
            self.check(inputs)
            self._checked = True
        feed = {k: np.asarray(v, np.float32).reshape(self._shape(k)) for k, v in inputs.items()}
        feed.update({k: v.astype(np.float32).reshape(self._shape(k)) for k, v in self.constants.items()})
        return np.asarray(self.session.run([self.output], feed)[0], np.float64).reshape(-1)[:NUM_JOINTS]


# ----------------------------------------------------------------------------------------- the controller
class OnnxTrackerController(SuperMimicController):
    """SuperMimic's controller with the observation, network and action of ``spec``. Takes record ``tracking_columns``.

    Args:
        state: Reads the robot, joints in ``JOINT_NAMES`` order.
        spec: The tracker.
        retarget: See ``GmrRetargeter``. Its ``last_root_xy`` gives the reference its horizontal path.
        emit: Makes one simulator action from the 29 targets in ``JOINT_NAMES`` order and the ``ControlInput``.
        policy: ``inputs -> raw action``. Replaces the ONNX session, for tests.
        obs_noise: See ``supermimic.ObservationNoise``.
        lookahead: Replaces the spec's.
        kinematics: Replaces the default ``Kinematics``, for terms that need the torso.
    """

    def __init__(self, state: Callable[[], RobotState], spec: TrackerSpec, retarget=None,
                 emit: Callable[[np.ndarray, ControlInput], object] | None = None, blend_seconds: float = 1.0, policy=None,
                 obs_noise=None, lookahead: int | None = None, kinematics: Kinematics | None = None):
        self.spec = spec
        self.history = spec.history
        self._back = np.argsort(spec.order)  # network order -> JOINT_NAMES order
        self.kinematics = kinematics if kinematics is not None or not spec.needs_kinematics else Kinematics()
        self.poses = PoseStream(spec.lookahead if lookahead is None else lookahead, spec.past, spec.control_dt, self.kinematics)
        super().__init__(state, retarget=retarget, emit=emit, lookahead=0, blend_seconds=blend_seconds,
                         policy=policy if policy is not None else OnnxPolicy(spec), obs_noise=obs_noise)
        self.lookahead = self.poses.lookahead
        self.stand_joint_pos = spec.default_joint_pos[self._back]
        self.stand_root_height = spec.stand_root_height
        if self.stand_root_height is None:
            try:
                self.stand_root_height = (self.kinematics or Kinematics()).stand_height(self.stand_joint_pos)
            except (ImportError, OSError, ValueError):  # no mujoco or no model: SuperMimic's
                from .supermimic import STAND_ROOT_HEIGHT

                self.stand_root_height = STAND_ROOT_HEIGHT
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._last_action = self.spec.action_offset.copy() if self.spec.last_action == "target" else np.zeros(NUM_JOINTS)
        self._last_target: np.ndarray | None = None
        self._ticks: deque[Tick] = deque(maxlen=self.history)
        self._engaged_at: tuple[np.ndarray, np.ndarray] | None = None  # (robot xy, operator xy) when tracking began
        self._stand_xy: np.ndarray | None = None
        self.poses.reset()

    def _observe(self) -> RobotState:
        now = super()._observe()
        torso = self.kinematics.torso_pose(now.joint_pos, np.zeros(3), now.quat)[1] if self.kinematics is not None else None
        tick = Tick(now, self._last_action, self._last_target, torso)
        if not self._ticks:
            self._ticks.extend([tick] * self.history)
        else:
            self._ticks.append(tick)
        return now

    def _root_xy(self, now: RobotState) -> np.ndarray:
        """Where the reference pelvis is over the floor: the robot's spot when tracking began, plus the operator's way
        since then turned by the yaw offset and faded with the blend. Standing, where the robot stood."""
        here = np.zeros(2) if now.pos is None else np.asarray(now.pos[:2], np.float64)
        operator_xy = getattr(self.retarget, "last_root_xy", None)
        if self._yaw_offset is None or operator_xy is None:
            self._engaged_at = None
            if self._stand_xy is None or self._blend <= 0.0:
                self._stand_xy = here
            return self._stand_xy
        if self._engaged_at is None:
            self._engaged_at = (self._stand_xy if self._stand_xy is not None else here, np.asarray(operator_xy, np.float64).copy())
        c, s = math.cos(self._yaw_offset), math.sin(self._yaw_offset)
        d = np.asarray(operator_xy, np.float64) - self._engaged_at[1]
        return self._engaged_at[0] + self._blend * np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def inputs(self) -> dict[str, np.ndarray]:
        cache: dict[int, Frame] = {}

        def frame(k):
            if k not in cache:
                cache[k] = self.poses.frame(k, self.spec.order)
            return cache[k]

        ticks = list(self._ticks)
        return {name: np.concatenate([build_block(b, self.spec, ticks, frame) for b in blocks]) for name, blocks in self.spec.inputs.items()}

    def act(self, control: ControlInput):
        now = self._observe()
        q, quat, z = self._next_pose(control, now)
        if self.spec.ref_zero_joints:
            q = np.array(q, np.float64)
            q[[JOINT_NAMES.index(n) for n in self.spec.ref_zero_joints]] = 0.0
        xy = self._root_xy(now)
        self.poses.push(q, quat, np.array([xy[0], xy[1], z]))
        inputs = self.inputs()
        t0 = time.perf_counter()
        raw = self.policy(inputs)
        self._policy_ms.append((time.perf_counter() - t0) * 1000)
        if self.spec.action_clip is not None:
            raw = np.clip(raw, -self.spec.action_clip, self.spec.action_clip)
        offset = self.poses.frame(0, self.spec.order).joint_pos if self.spec.residual else self.spec.action_offset
        target = raw * self.spec.action_scale + offset  # network order
        self._last_action = raw if self.spec.last_action == "raw" else target
        self._last_target = target
        current = self.poses.frame(0, np.arange(NUM_JOINTS))
        self.last_reference = current.joint_pos, current.root_pos[2], current.root_quat
        self.last_tracking_error = float(np.sqrt(np.mean((self.last_state.joint_pos - current.joint_pos) ** 2)))
        return self.emit(target[self._back], control)

    def record(self) -> dict[str, np.ndarray]:
        return self.tracking_columns()


def from_env(env, spec: str | Path | TrackerSpec, retarget=None,
             hands: Callable[[tuple[float, float, float, float]], np.ndarray] | None = None, robot: str = "robot",
             **options) -> OnnxTrackerController:
    """Like ``supermimic.from_env``: the action is ``hands`` columns, if any, then the 29 targets in ``JOINT_NAMES`` order."""
    import torch

    spec = spec if isinstance(spec, TrackerSpec) else TrackerSpec.from_json(spec)
    action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    offset = action.shape[1] - NUM_JOINTS

    def emit(target: np.ndarray, control: ControlInput):
        if hands is not None:
            action[0, :offset] = torch.as_tensor(hands(control.triggers), device=action.device)
        action[0, offset:] = torch.as_tensor(target, dtype=action.dtype, device=action.device)
        return action

    return OnnxTrackerController(isaac_state_reader(env, robot), spec, retarget, emit, **options)


register("onnx_tracker", from_env)
