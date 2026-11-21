# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""SuperMimic whole-body tracking of the operator, in simulation, on a stock Unitree G1.

The weights are realmimic's, an independent reproduction of SuperMimic (arXiv 2610.28480). The paper's authors have
released no code or checkpoint, so nothing here says anything about the paper's own results. See ``SuperMimicPolicy``.

Every 20 ms the policy reads ten ticks of the robot's joints, pelvis gyro and gravity direction, and ten frames of a G1
reference starting now, and returns 29 joint position targets. So an operator needs turning into G1 joints first:

    body (24, 7) -> GmrRetargeter -> ReferenceStream -> window of 10 frames -> SuperMimicPolicy -> joint targets

The reference has no root position in it, so on its own **the robot's position does not follow the operator**: it only
goes where copying the legs carries it. Its yaw does follow, relative to where operator and robot faced when tracking
began. With a ``follow`` policy (``follow.py``) the goal is edited every tick so that the robot also walks to where the
operator stands in the simulator, and the yaw follows the operator's real heading in the room instead.

A live stream has no future. ``lookahead`` chooses how to fill the window: real frames, ``lookahead`` ticks late, then
the newest held still with zero velocity, which is how upstream fills a window that runs past the end of a clip.
"""

from __future__ import annotations

import math
import os
import sys
import time
import types
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..controller import ControlInput, register
from ..hands import operator_yaw, room_position_to_sim

REPO_ROOT = Path(__file__).resolve().parents[3]
GMR_ROOT = Path(os.environ.get("TELESCENE_GMR") or REPO_ROOT / "third_party/GMR")
"""A GMR checkout (github.com/YanjieZe/GMR, MIT), as ``scripts/download_trackers.sh`` clones it."""
CHECKPOINT = Path(os.environ.get("TELESCENE_SUPERMIMIC") or REPO_ROOT / "checkpoints/supermimic/realmimic_g1_86k.pt")
"""realmimic's released policy, as ``scripts/download_trackers.sh`` fetches it."""
FOLLOW_CHECKPOINT = REPO_ROOT / "checkpoints/supermimic/follow_v1.pt"
"""The follow policy v1 (``follow.py``), from this repo's GitHub release."""

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
"""MuJoCo order: what the policy's inputs and outputs are in."""

NUM_JOINTS, HISTORY, FRAMES = 29, 10, 10
CONTROL_DT, PHYSICS_DT = 0.02, 0.005
STAND_ROOT_HEIGHT = 0.7570215776517653
"""Pelvis height of the default pose with the feet on the ground."""

_POS_STAND = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6, "shoulder_pitch": 0.2}
DEFAULT_JOINT_POS = np.array(
    [_POS_STAND.get(n.removeprefix("left_").removeprefix("right_").removesuffix("_joint"),
                    {"left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2}.get(n, 0.0))
     for n in JOINT_NAMES], np.float64)
"""mjlab's ``KNEES_BENT_KEYFRAME``."""

# mjlab's G1 motors at the commit realmimic pins: kp = armature * (2 pi 10 Hz)^2, kd = 4 * armature * 2 pi 10 Hz, doubled
# for the two-motor ankles and waist. Checked against upstream's own functions in tests/test_supermimic.py.
_MOTORS = {  # armature, effort limit
    "5020": (0.003609725, 25.0),
    "7520_14": (0.010177520, 88.0),
    "7520_22": (0.025101925, 139.0),
    "4010": (0.00425, 5.0),
    "double_5020": (0.00721945, 50.0),
}
_JOINT_MOTOR = {"elbow": "5020", "shoulder_pitch": "5020", "shoulder_roll": "5020", "shoulder_yaw": "5020",
                "wrist_roll": "5020", "hip_pitch": "7520_14", "hip_yaw": "7520_14", "waist_yaw": "7520_14",
                "hip_roll": "7520_22", "knee": "7520_22", "wrist_pitch": "4010", "wrist_yaw": "4010",
                "waist_pitch": "double_5020", "waist_roll": "double_5020", "ankle_pitch": "double_5020",
                "ankle_roll": "double_5020"}
_MOTOR_OF = [_JOINT_MOTOR[n.removeprefix("left_").removeprefix("right_").removesuffix("_joint")] for n in JOINT_NAMES]
ARMATURE = np.array([_MOTORS[m][0] for m in _MOTOR_OF])
EFFORT_LIMIT = np.array([_MOTORS[m][1] for m in _MOTOR_OF])
_NATURAL = 2.0 * 3.1415926535 * 10.0
STIFFNESS = ARMATURE * _NATURAL ** 2
DAMPING = 4.0 * ARMATURE * _NATURAL
ACTION_SCALE = 0.25 * EFFORT_LIMIT / STIFFNESS
"""Radians per unit of policy output."""


# ------------------------------------------------------------------------------------------ quaternions, wxyz
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    two = 2.0 / np.sum(q * q, axis=-1)
    return np.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
                     two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
                     two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)),
                    axis=-1).reshape(*q.shape[:-1], 3, 3)


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(a @ b)
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    angle = math.acos(dot)
    return (math.sin((1 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


# ------------------------------------------------------------------------------------------- the observation
@dataclass(frozen=True)
class RobotState:
    """One tick of what the policy may see. Joints in ``JOINT_NAMES`` order, ``ang_vel_b`` in the pelvis frame."""

    quat: np.ndarray
    ang_vel_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    pos: np.ndarray | None = None
    """Pelvis position in the simulator's world. Only the follow policy needs it."""
    lin_vel_b: np.ndarray | None = None
    """Pelvis linear velocity in the pelvis frame. Simulation only; some trackers observe it."""


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """(0, 0, -1) in the body frame."""
    return np.array([0.0, 0.0, -1.0]) @ quat_matrix(quat)


def history_obs(states: list[RobotState], actions: list[np.ndarray]) -> np.ndarray:
    """``(930,)``, term by term and oldest first within a term: joint_pos - q0, joint_vel, ang_vel, gravity, last action."""
    return np.concatenate((
        np.concatenate([s.joint_pos - DEFAULT_JOINT_POS for s in states]),
        np.concatenate([s.joint_vel for s in states]),
        np.concatenate([s.ang_vel_b for s in states]),
        np.concatenate([projected_gravity(s.quat) for s in states]),
        np.concatenate(actions),
    ))


def goal_obs(robot_quat: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray, root_z: np.ndarray,
             root_quat: np.ndarray) -> np.ndarray:
    """``(650,)`` from a window of ``FRAMES``: frame by frame absolute joints, joint velocities, pelvis height, then the
    orientation of each frame's pelvis relative to the robot's, as the first two columns of its matrix."""
    frames = np.concatenate((joint_pos, joint_vel, root_z[:, None]), axis=1).ravel()
    robot = robot_quat / np.linalg.norm(robot_quat)
    relative = quat_mul(quat_conjugate(robot)[None], root_quat)
    return np.concatenate((frames, quat_matrix(relative)[:, :, :2].reshape(-1)))


class ObservationNoise:
    """Gaussian noise on what a policy reads of the robot, from a seed. The simulator is deterministic, so replaying a
    take twice gives the same run; seeding this is how one take gives several, for comparing trackers fairly. The robot
    as recorded and scored stays the true one. ``scale`` multiplies every sigma.

    Sigmas are a fraction of the noise trackers usually train with (Isaac Lab's joint_pos 0.01, joint_vel 1.5,
    ang_vel 0.2), small enough not to change how a tracker does on average."""

    SIGMA = {"joint_pos": 0.005, "joint_vel": 0.05, "ang_vel_b": 0.02}

    def __init__(self, seed: int, scale: float = 1.0):
        self.seed, self.scale, self.rng = seed, scale, np.random.default_rng(seed)

    def __call__(self, s: RobotState) -> RobotState:
        def noisy(name):
            value = getattr(s, name)
            return value + self.rng.normal(0.0, self.SIGMA[name] * self.scale, value.shape)

        return RobotState(s.quat, noisy("ang_vel_b"), noisy("joint_pos"), noisy("joint_vel"), s.pos, s.lin_vel_b)


# ---------------------------------------------------------------------------------------------- the reference
class ReferenceStream:
    """Turns one pose per tick into the policy's window. Joint velocities come from differences.

    Args:
        lookahead: Ticks of real future the window holds, 0 to ``frames - 1``, which is also how late the robot is.
        dt: Seconds per tick.
        frames: Frames in the window. SuperMimic's is ``FRAMES``; other trackers want other lengths.
    """

    def __init__(self, lookahead: int = FRAMES - 1, dt: float = CONTROL_DT, frames: int = FRAMES):
        if not 0 <= lookahead < frames:
            raise ValueError(f"lookahead must be 0 to {frames - 1}")
        self.lookahead, self.dt, self.frames = lookahead, dt, frames
        self._poses: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=lookahead + 2)

    def reset(self) -> None:
        self._poses.clear()

    def push(self, joint_pos: np.ndarray, root_quat: np.ndarray, root_z: float) -> None:
        self._poses.append((joint_pos, root_quat, root_z))

    def window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``joint_pos (F, 29), joint_vel (F, 29), root_z (F,), root_quat (F, 4)``, F = ``frames``. Frame 0 is now."""
        real = self.lookahead + 1
        poses = list(self._poses)
        poses = [poses[0]] * (real + 1 - len(poses)) + poses  # the oldest is only there to difference against
        q = np.stack([p[0] for p in poses])
        joint_vel = np.zeros((self.frames, NUM_JOINTS))
        if real > 1:
            joint_vel[:real - 1] = (q[2:] - q[:-2]) / (2 * self.dt)
        joint_vel[real - 1] = (q[-1] - q[-2]) / self.dt
        hold = self.frames - real
        pad = [poses[-1]] * hold
        kept = poses[1:] + pad
        return (np.concatenate((q[1:], np.repeat(q[-1:], hold, axis=0))), joint_vel,
                np.array([p[2] for p in kept]), np.stack([p[1] for p in kept]))


# ------------------------------------------------------------------------------------------------- the policy
class SuperMimicPolicy:
    """The released network, rebuilt from weights alone: no upstream code and no pickled objects run.

    Encoder ``goal -> 2048, 1024, 512, 256 -> tanh(32)``, decoder ``latent + history -> 2048, 1024, 512, 256 -> 29``, both
    ELU, each input divided by a stored running mean and variance.

    Args:
        checkpoint: ``realmimic_g1_86k.pt`` from huggingface.co/kevinhenryz/realmimic. CC BY-NC 4.0, and subject to the
            BONES-SEED licence. ``scripts/download_trackers.sh`` fetches it to ``CHECKPOINT``; it is not in this repo.
        threads: CPU threads for inference. This runs in the control loop next to the simulator.
    """

    EPS = 1e-2  # RunningNorm: std = sqrt(var + eps^2)

    def __init__(self, checkpoint: str | Path, threads: int = 2):
        import torch

        torch.set_num_threads(threads)
        self._torch = torch
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        env, cfg = ck["config"]["env"], ck["config"]["policy"]
        if env.get("goal_features") != "sonic" or env.get("anchor_obs") != "pelvis_ori10":
            raise ValueError(f"expected a 'sonic' / 'pelvis_ori10' policy, got {env.get('goal_features')} / {env.get('anchor_obs')}")
        if cfg["activation"] != "elu" or cfg["input_norm"] != "running" or cfg["history"] != HISTORY or cfg["future_frames"] != FRAMES:
            raise ValueError(f"unexpected policy config {cfg}")
        state = ck["policy"]
        self.iteration = ck.get("iteration")
        self.encoder = self._mlp(state, "encoders.motion.mlp.net")
        self.decoder = self._mlp(state, "decoder.mlp.net")
        self.goal_mean, self.goal_std = self._norm(state, "encoders.motion.norm")
        self.obs_mean, self.obs_std = self._norm(state, "decoder.norm")

    def _norm(self, state, prefix):
        return state[f"{prefix}.mean"], self._torch.sqrt(state[f"{prefix}.var"] + self.EPS ** 2)

    def _mlp(self, state, prefix):
        torch = self._torch
        layers, index = [], 0
        while f"{prefix}.{index}.weight" in state:
            linear = torch.nn.Linear(*reversed(state[f"{prefix}.{index}.weight"].shape))
            linear.load_state_dict({"weight": state[f"{prefix}.{index}.weight"], "bias": state[f"{prefix}.{index}.bias"]})
            layers += [linear, torch.nn.ELU()]
            index += 2
        return torch.nn.Sequential(*layers[:-1]).eval().requires_grad_(False)

    def __call__(self, history: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Action ``(29,)`` in the policy's units: the target is ``action * ACTION_SCALE + DEFAULT_JOINT_POS``."""
        torch = self._torch
        with torch.inference_mode():
            obs = torch.from_numpy(history.astype(np.float32))
            target = torch.from_numpy(goal.astype(np.float32))
            latent = torch.tanh(self.encoder((target - self.goal_mean) / self.goal_std))
            self.last_latent = latent.numpy()
            return self.decoder(torch.cat((latent, (obs - self.obs_mean) / self.obs_std))).numpy().astype(np.float64)


# ------------------------------------------------------------------------------------------------ the operator
_XR_TO_GMR = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
"""OpenXR room axes (x right, y up, z back) to GMR's (x right, y forward, z up), as in its ``XRobotStreamer``."""
_XR_TO_GMR_QUAT = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
GMR_BODY_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2", "Left_Ankle", "Right_Ankle",
    "Spine3", "Left_Foot", "Right_Foot", "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
    "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)


def valid_joints(body: np.ndarray) -> np.ndarray:
    """A joint that is not valid arrives with an all-zero quaternion."""
    return np.any(body[:, 3:7] != 0.0, axis=1)


def _load_gmr(root: Path):
    """``GeneralMotionRetargeting`` without running GMR's ``__init__``, which pulls in its viewer and PICO SDK."""
    if "general_motion_retargeting" not in sys.modules:
        package = types.ModuleType("general_motion_retargeting")
        package.__path__ = [str(root / "general_motion_retargeting")]
        sys.modules["general_motion_retargeting"] = package
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

    return GeneralMotionRetargeting


RETARGET_PRESETS = {
    "gmr": {},
    "teleop": {"leg_height": 1.365, "robot_height": 1.32, "wrist_cost": 200.0},
}
"""``leg_height``: operator height at which the knees and feet scale by 0.75 (G1 legs over a 1.82 m operator's).
``robot_height``: the wrists' scale is this over the operator's height. ``wrist_cost``: weight of the wrist positions."""


class GmrRetargeter:
    """PICO's 24 joints to G1 joints with GMR (github.com/YanjieZe/GMR, MIT) and its ``xrobot_to_g1`` table.

    Holds the last good position and orientation of a joint that stops being valid, and returns None until every joint
    has been seen once.

    Args:
        gmr_root: A GMR checkout. ``mink`` must be importable: GMR's IK solves with it.
        operator_height: Metres. Scales the operator to the robot, by GMR's own rule.
        preset: "gmr", GMR's table as it is. "teleop": the knees and feet scaled to the G1's legs (GMR's 0.9 asks
            0.60 m legs to reach about 0.72 m), and each wrist put where the operator's is relative to the pelvis,
            scaled by height, which GMR's arms (orientation only) do not do. On recorded takes (10,500 frames) it
            cut the hand error from 8.6 to 4.4 cm and the foot placement error from 8.3 to 6.1 cm, and narrowed the
            stance from 0.25 to 0.20 m; the thigh direction got 3 degrees worse. See ``RETARGET_PRESETS``.
    """

    def __init__(self, gmr_root: str | Path, operator_height: float | None = None, preset: str = "gmr"):
        if preset not in RETARGET_PRESETS:
            raise ValueError(f"preset {preset!r}: one of {sorted(RETARGET_PRESETS)}")
        cls = _load_gmr(Path(gmr_root))
        self.gmr = cls("xrobot", "unitree_g1", actual_human_height=operator_height, verbose=False)
        self.preset = preset
        if preset == "teleop":
            self._install_teleop_preset(operator_height or 1.8)
        import mujoco

        self._mujoco = mujoco
        model = self.gmr.model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"GMR's G1 has another joint order: {names}")
        self._ankles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
        self._qpos0 = self.gmr.configuration.data.qpos.copy()
        self._ankle_at_stand = self._stand_ankle_height()
        self.reset()

    def _install_teleop_preset(self, operator_height: float) -> None:
        import mink

        g, cfg = self.gmr, RETARGET_PRESETS["teleop"]
        legs = cfg["leg_height"] / operator_height
        for name in ("Left_Knee", "Right_Knee", "Left_Foot", "Right_Foot"):
            g.human_scale_table[name] = legs
        arm_scale = cfg["robot_height"] / operator_height
        wrists = {}
        for side, name in (("left", "Left_Wrist"), ("right", "Right_Wrist")):
            task = mink.RelativeFrameTask(f"{side}_wrist_yaw_link", "body", "pelvis", "body",
                                          position_cost=cfg["wrist_cost"], orientation_cost=0.0, lm_damping=1.0)
            g.tasks2.append(task)
            wrists[name] = task
        original = g.update_targets

        def update_targets(human_data, offset_to_ground=False):
            original(human_data, offset_to_ground)
            pelvis_rot = quat_matrix(np.asarray(g.scaled_human_data["Pelvis"][1], float))
            pelvis = np.asarray(human_data["Pelvis"][0], float)
            for name, task in wrists.items():
                relative = pelvis_rot.T @ (np.asarray(human_data[name][0], float) - pelvis) * arm_scale
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.identity(), relative))

        g.update_targets = update_targets

    def _lowest_ankle(self, qpos: np.ndarray) -> float:
        data = self.gmr.configuration.data
        data.qpos[:] = qpos
        self._mujoco.mj_kinematics(self.gmr.model, data)
        return float(min(data.xpos[i, 2] for i in self._ankles))

    def _stand_ankle_height(self) -> float:
        qpos = self._qpos0.copy()
        qpos[2], qpos[3:7], qpos[7:] = STAND_ROOT_HEIGHT, (1.0, 0.0, 0.0, 0.0), DEFAULT_JOINT_POS
        return self._lowest_ankle(qpos)

    def reset(self) -> None:
        qpos = self._qpos0.copy()
        qpos[7:] = DEFAULT_JOINT_POS
        self.gmr.configuration.update(qpos)
        self._held: np.ndarray | None = None
        self._seen = np.zeros(len(GMR_BODY_NAMES), bool)

    def _human_data(self, body: np.ndarray) -> dict:
        positions = body[:, :3].astype(np.float64) @ _XR_TO_GMR.T
        xyzw = body[:, 3:7].astype(np.float64)
        wxyz = quat_mul(_XR_TO_GMR_QUAT, np.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=1))
        return {name: [positions[i], wxyz[i]] for i, name in enumerate(GMR_BODY_NAMES)}

    def __call__(self, body: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """``joint_pos (29,), pelvis quat wxyz (4,), pelvis height``: the height with the lower foot on the ground."""
        valid = valid_joints(body)
        self._seen |= valid
        if self._held is None:
            self._held = body.astype(np.float64).copy()
        self._held[valid] = body[valid]
        if not self._seen.all():
            return None
        qpos = self.gmr.retarget(self._human_data(self._held))
        lowest = self._lowest_ankle(qpos)
        self.last_root_xy = qpos[:2].copy()  # GMR's world, z up; for trackers that also want where the pelvis goes
        return qpos[7:].copy(), qpos[3:7].copy(), float(qpos[2]) - lowest + self._ankle_at_stand


# ---------------------------------------------------------------------------------------------- the controller
def isaac_state_reader(env, robot: str = "robot") -> Callable[[], RobotState]:
    """Reads an Isaac Lab articulation, env 0. Its quaternions are xyzw and its joints in its own order."""
    articulation = env.scene[robot]
    order = [articulation.joint_names.index(n) for n in JOINT_NAMES]
    if articulation.body_names[0] != "pelvis":
        raise ValueError(f"the root must be the pelvis, not {articulation.body_names[0]}")

    def numpy_of(value):
        return (value.torch if hasattr(value, "torch") else value)[0].detach().cpu().numpy().astype(np.float64)

    def read() -> RobotState:
        data = articulation.data
        x, y, z, w = numpy_of(data.root_link_quat_w)
        return RobotState(np.array([w, x, y, z]), numpy_of(data.root_ang_vel_b),
                          numpy_of(data.joint_pos)[order], numpy_of(data.joint_vel)[order], numpy_of(data.root_link_pos_w),
                          numpy_of(data.root_lin_vel_b) if hasattr(data, "root_lin_vel_b") else None)

    return read


class SuperMimicController:
    """A telescene controller: the operator's whole body while tracking, standing in the default pose otherwise.

    It does not walk, so the session's velocity and the sticks do nothing: with tracking off the robot stands where it is.

    Args:
        state: Reads the robot.
        checkpoint: See ``SuperMimicPolicy``.
        retarget: ``body (24, 7) -> (joint_pos, pelvis quat wxyz, pelvis height)`` or None. See ``GmrRetargeter``.
        emit: Makes one action for the simulator from the 29 absolute joint targets and the ``ControlInput``.
        lookahead: See ``ReferenceStream``. Each tick of it delays the robot by 20 ms.
        blend_seconds: How long the reference takes to move between standing and the operator.
        policy: Replaces ``checkpoint``, for tests.
        follow: A ``follow.Follower``, or None to leave the goal as it is. Needs ``RobotState.pos``.
        obs_noise: An ``ObservationNoise``, or None to read the robot exactly.
    """

    history = HISTORY
    """Ticks of robot state kept for the observation."""

    def __init__(self, state: Callable[[], RobotState], checkpoint: str | Path | None = None, retarget=None,
                 emit: Callable[[np.ndarray, ControlInput], object] | None = None, lookahead: int = FRAMES - 1,
                 blend_seconds: float = 1.0, policy=None, follow=None, obs_noise: ObservationNoise | None = None):
        self.state, self.retarget, self.lookahead, self.obs_noise = state, retarget, lookahead, obs_noise
        self.emit = emit or (lambda target, control: target)
        self.policy = policy if policy is not None else SuperMimicPolicy(checkpoint)
        self.reference = ReferenceStream(lookahead)
        self.follow = follow
        self.blend_step = CONTROL_DT / blend_seconds if blend_seconds > 0 else 1.0
        self.stand_joint_pos, self.stand_root_height = DEFAULT_JOINT_POS, STAND_ROOT_HEIGHT  # what ``blend`` 0 asks for
        self._retarget_ms: list[float] = []
        self._policy_ms: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._states: deque[RobotState] = deque(maxlen=self.history)
        self._actions: deque[np.ndarray] = deque(maxlen=self.history)
        self._last_action = np.zeros(NUM_JOINTS)
        self.reference.reset()
        self._blend = 0.0
        self._operator: tuple[np.ndarray, np.ndarray, float] | None = None
        self._yaw_offset: float | None = None
        self._stand_yaw: float | None = None
        self.last_goal = self.last_command = np.zeros(650)
        self.last_target = np.zeros(2)
        self.last_reference = (self.stand_joint_pos, self.stand_root_height, np.array([1.0, 0.0, 0.0, 0.0]))
        self.last_state: RobotState | None = None
        self.last_tracking_error = 0.0
        if self.follow is not None:
            self.follow.reset()
        if self.retarget is not None and hasattr(self.retarget, "reset"):
            self.retarget.reset()

    def _observe(self) -> RobotState:
        """The robot as the policy sees it. ``last_state`` keeps it as it is."""
        self.last_state = self.state()
        now = self.obs_noise(self.last_state) if self.obs_noise is not None else self.last_state
        if not self._states:
            self._states.extend([now] * self.history)  # upstream: every slot holds the reset state and a zero action
            self._actions.extend([self._last_action.copy()] * self.history)  # zeros, unless a subclass resets it
        else:
            self._states.append(now)
            self._actions.append(self._last_action)
        return now

    def _next_pose(self, control: ControlInput, now: RobotState) -> tuple[np.ndarray, np.ndarray, float]:
        robot_yaw = yaw_of(now.quat)
        if self._stand_yaw is None:
            self._stand_yaw = robot_yaw
        if control.tracking and control.body is not None and self.retarget is not None:
            t0 = time.perf_counter()
            pose = self.retarget(control.body)
            self._retarget_ms.append((time.perf_counter() - t0) * 1000)
            if pose is not None:
                self._operator = pose
        engaged = control.tracking and self._operator is not None
        if engaged and self._yaw_offset is None:
            if self.follow is not None:  # face where the operator really faces in the room, as the position does
                self._yaw_offset = operator_yaw(control.body, control.anchor_yaw) - yaw_of(self._operator[1])
            else:
                self._yaw_offset = robot_yaw - yaw_of(self._operator[1])
            self._stand_yaw = robot_yaw
        elif self._blend <= 0.0 and not engaged and self._yaw_offset is not None:
            self._yaw_offset, self._operator = None, None
            self._stand_yaw = robot_yaw
        self._blend = float(np.clip(self._blend + (self.blend_step if engaged else -self.blend_step), 0.0, 1.0))
        stand = yaw_quat(self._stand_yaw)
        if self._operator is None:
            return self.stand_joint_pos, stand, self.stand_root_height
        q, quat, z = self._operator
        aligned = quat_mul(yaw_quat(self._yaw_offset), quat)
        b = self._blend
        return (1 - b) * self.stand_joint_pos + b * q, slerp(stand, aligned, b), (1 - b) * self.stand_root_height + b * z

    def act(self, control: ControlInput):
        now = self._observe()
        self.reference.push(*self._next_pose(control, now))
        joint_pos, joint_vel, root_z, root_quat = self.reference.window()
        history = history_obs(list(self._states), list(self._actions))
        goal = goal_obs(now.quat, joint_pos, joint_vel, root_z, root_quat)
        self.last_goal = goal
        t0 = time.perf_counter()
        if self.follow is not None:
            goal = self._follow(control, now, history, goal, root_quat[0])
        self.last_command = goal
        action = self.policy(history, goal)
        self._policy_ms.append((time.perf_counter() - t0) * 1000)
        self._last_action = action
        self.last_reference = joint_pos[0], root_z[0], root_quat[0]
        self.last_tracking_error = float(np.sqrt(np.mean((self.last_state.joint_pos - joint_pos[0]) ** 2)))
        return self.emit(action * ACTION_SCALE + DEFAULT_JOINT_POS, control)

    def _follow(self, control: ControlInput, now: RobotState, history: np.ndarray, goal: np.ndarray,
                ref_quat: np.ndarray) -> np.ndarray:
        """The goal edited by the follow policy towards the operator's pelvis in the simulator, faded with the blend."""
        if control.body is None or now.pos is None or self._blend <= 0.0 or self._operator is None:
            self.follow.reset()
            return goal
        pelvis = room_position_to_sim(control.body[0, :3], control.anchor_yaw)
        self.last_target = np.array([control.anchor[0] + pelvis[0] * control.world_scale,
                                     control.anchor[1] + pelvis[1] * control.world_scale])
        return self.follow(history, goal, self.last_target, now.pos[:2], yaw_of(now.quat), yaw_of(ref_quat), self._blend)

    def record(self) -> dict[str, np.ndarray]:
        """Columns for a recording, the same every tick. ``command`` is the goal SuperMimic was given (the follow
        policy's output when there is one); ``goal`` is the operator's own, before any edit; ``latent`` is the encoder's
        32 numbers for ``command``, the compact form, but only meaningful to this checkpoint's decoder."""
        f = self.follow
        latent = getattr(self.policy, "last_latent", None)
        return {**self.tracking_columns(), "command": self.last_command.astype(np.float32), "goal": self.last_goal.astype(np.float32),
                "latent": np.zeros(32, np.float32) if latent is None else np.asarray(latent, np.float32),
                "follow_action": (f.action if f is not None else np.zeros(16)).astype(np.float32),  # 16 v1, 33 v2
                "follow_error": (f.error if f is not None else np.zeros(2)).astype(np.float32),
                "follow_target": self.last_target.astype(np.float32)}

    def tracking_columns(self) -> dict[str, np.ndarray]:
        """What every tracker records, so that trackers can be compared from their takes alone. Joints in
        ``JOINT_NAMES`` order, quaternions wxyz. ``ref_*`` is the reference frame for now, the one the robot should be at;
        ``tracking_error`` is the joint RMS between the two [rad]. Standing values before the first tick."""
        state = self.last_state
        joint_pos, root_z, root_quat = self.last_reference
        return {"robot_joint_pos": (self.stand_joint_pos if state is None else state.joint_pos).astype(np.float32),
                "robot_quat": (np.array([1.0, 0, 0, 0]) if state is None else state.quat).astype(np.float32),
                "ref_joint_pos": np.asarray(joint_pos, np.float32), "ref_root_z": np.float32(root_z),
                "ref_root_quat": np.asarray(root_quat, np.float32), "blend": np.float32(self._blend),
                "tracking_error": np.float32(self.last_tracking_error)}

    def status(self) -> str:
        ik = f"{np.mean(self._retarget_ms):.1f}" if self._retarget_ms else "-"
        net = f"{np.mean(self._policy_ms):.1f}" if self._policy_ms else "-"
        self._retarget_ms.clear()
        self._policy_ms.clear()
        follow = "" if self.follow is None else f", follow err {np.linalg.norm(self.follow.error):.2f} m edit {np.abs(self.follow.action).mean():.2f}"
        return f"blend {self._blend:.2f}, joint rms {getattr(self, 'last_tracking_error', 0.0):.3f} rad{follow}, ik {ik} ms, net {net} ms"


def from_env(env, checkpoint: str | Path, retarget=None, hands: Callable[[tuple[float, float, float, float]], np.ndarray] | None = None,
             robot: str = "robot", follow_checkpoint: str | Path | None = None, follow_filter: float = 1.0,
             **options) -> SuperMimicController:
    """A controller for an Isaac Lab environment whose action is ``hands`` columns, if any, then the 29 targets in
    ``JOINT_NAMES`` order. ``follow_checkpoint`` adds the follow policy and ``follow_filter`` smooths its output
    (``Follower``'s ``action_filter``; see ``follow.py``)."""
    import torch

    action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    offset = action.shape[1] - NUM_JOINTS

    def emit(target: np.ndarray, control: ControlInput):
        if hands is not None:
            action[0, :offset] = torch.as_tensor(hands(control.triggers), device=action.device)
        action[0, offset:] = torch.as_tensor(target, dtype=action.dtype, device=action.device)
        return action

    if follow_checkpoint is not None:
        from .follow import Follower, FollowNet

        options["follow"] = Follower(FollowNet(follow_checkpoint), JOINT_NAMES, action_filter=follow_filter)
    return SuperMimicController(isaac_state_reader(env, robot), checkpoint, retarget, emit, **options)


register("supermimic", from_env)
