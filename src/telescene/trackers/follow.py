# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""The follow policy: a small network that edits SuperMimic's goal so that the robot also goes where the operator is.

SuperMimic's goal has no root position in it, so on its own the robot only goes where copying the legs carries it, and
drifts. The follow network reads the robot, the first goal frame and where the operator is relative to the robot, and
returns 16 numbers in [-1, 1]: offsets on the 12 leg joints, on the pelvis height and on the pelvis roll, pitch and
yaw. They are added to every frame of the goal, and SuperMimic, unchanged, turns the edited goal into joint targets:

    goal (650) --[+ follow edit]--> edited goal (650) --> SuperMimic --> 29 joint targets

The edited goal is what the robot was really asked to do, so it is what a recording should keep as the command.

Two versions exist, told apart by the checkpoint's output size:

- v1, 16 outputs (above). Its input is the robot, the first goal frame, the error, a leaky integral of the error and
  its last action.
- v2, 33 outputs: offsets on all 29 joints, each scaled by ``joint_scales``, then the pelvis height and the pelvis roll,
  pitch and yaw. Its input is the robot, the whole goal window, the error, the yaw error and its last action. It has no
  integral. It was also trained to keep the ankles and wrists on the reference pose.

Trained in realmimic's mjlab environment by back-propagation through SuperMimic's world model, on BONES-SEED clips whose
pelvis path was moved by random offsets. The training code is not released; the v1 weights are on this repo's GitHub
release (CC BY-NC 4.0, as the SuperMimic weights it was trained through).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FRAMES, FRAME_DIM, NJ = 10, 59, 29
FRAMES_END = FRAMES * FRAME_DIM
GOAL_DIM = FRAMES_END + FRAMES * 6
LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
EDIT_DIM = len(LEG_JOINTS) + 1 + 3
FULL_EDIT_DIM = NJ + 1 + 3
LEG_SCALE, HEIGHT_SCALE = 0.35, 0.06
ROT_SCALE = np.array((0.12, 0.12, 0.5))
OBS_TERMS = (29, 29, 3, 3, 29)
INTEGRAL_DECAY, INTEGRAL_LIMIT = 0.98, 1.0
CONTROL_DT = 0.02


def latest_frame(history: np.ndarray, frames: int = 10) -> np.ndarray:
    """``(930,)`` term-major history, oldest first within a term -> ``(93,)`` newest tick."""
    out, at = [], 0
    for size in OBS_TERMS:
        out.append(history[at + (frames - 1) * size: at + frames * size])
        at += frames * size
    return np.concatenate(out)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """``(..., 6)`` first two columns row-major ``[m00, m01, m10, m11, m20, m21]`` -> ``(..., 3, 3)``, re-orthonormalised."""
    m = r6.reshape(*r6.shape[:-1], 3, 2)
    a = m[..., 0] / np.linalg.norm(m[..., 0], axis=-1, keepdims=True)
    b = m[..., 1] - (a * m[..., 1]).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def euler_xyz_matrix(rpy: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw -> ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def edit_goal(goal: np.ndarray, action: np.ndarray, action_rate: np.ndarray, legs: np.ndarray) -> np.ndarray:
    """Apply a follow action in [-1, 1] to every frame of a ``(650,)`` goal. ``action_rate`` is its change per second;
    it goes into the joint velocities. ``legs`` are the leg joints' indices in the goal's joint order."""
    frames = goal[:FRAMES_END].reshape(FRAMES, FRAME_DIM).copy()
    nl = len(legs)
    frames[:, legs] += action[:nl] * LEG_SCALE
    frames[:, NJ + legs] += action_rate[:nl] * LEG_SCALE
    frames[:, 2 * NJ] += action[nl] * HEIGHT_SCALE
    rel = rot6d_to_matrix(goal[FRAMES_END:].reshape(FRAMES, 6))
    rel = euler_xyz_matrix(action[nl + 1:] * ROT_SCALE)[None] @ rel
    return np.concatenate((frames.ravel(), rel[:, :, :2].reshape(-1)))


def joint_scales(joint_names, legs: float = 0.35, waist: float = 0.2, arms: float = 0.35) -> np.ndarray:
    """v2: radians a full-scale action moves each joint."""
    return np.array([legs if ("hip" in n or "knee" in n or "ankle" in n) else waist if "waist" in n else arms
                     for n in joint_names])


def edit_goal_full(goal: np.ndarray, action: np.ndarray, action_rate: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """v2: like ``edit_goal`` but on every joint, each by its own scale."""
    frames = goal[:FRAMES_END].reshape(FRAMES, FRAME_DIM).copy()
    frames[:, :NJ] += action[:NJ] * scales
    frames[:, NJ:2 * NJ] += action_rate[:NJ] * scales
    frames[:, 2 * NJ] += action[NJ] * HEIGHT_SCALE
    rel = rot6d_to_matrix(goal[FRAMES_END:].reshape(FRAMES, 6))
    rel = euler_xyz_matrix(action[NJ + 1:] * ROT_SCALE)[None] @ rel
    return np.concatenate((frames.ravel(), rel[:, :, :2].reshape(-1)))


def error_in_heading(target_xy, root_xy, root_yaw: float) -> np.ndarray:
    """Target minus robot in the robot's heading frame: x ahead, y left."""
    dx, dy = float(target_xy[0] - root_xy[0]), float(target_xy[1] - root_xy[1])
    c, s = math.cos(root_yaw), math.sin(root_yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class FollowNet:
    """The trained network, rebuilt from its weights: ``LayerNorm (no affine) -> [Linear, ELU] * k -> Linear -> tanh``."""

    def __init__(self, checkpoint: str | Path, threads: int = 1):
        import torch

        torch.set_num_threads(max(torch.get_num_threads(), threads))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("hl", state)
        weights = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.") and k.endswith(".weight")})
        layers = []
        for i, index in enumerate(weights):
            w, b = state[f"net.{index}.weight"], state[f"net.{index}.bias"]
            linear = torch.nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data.copy_(w)
            linear.bias.data.copy_(b)
            layers.append(linear)
            if i < len(weights) - 1:
                layers.append(torch.nn.ELU())
        self.inputs = layers[0].in_features
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(self.inputs, elementwise_affine=False), *layers).eval()
        self.outputs = layers[-1].out_features
        if self.outputs not in (EDIT_DIM, FULL_EDIT_DIM):
            raise ValueError(f"{checkpoint}: {self.outputs} outputs, expected {EDIT_DIM} (v1) or {FULL_EDIT_DIM} (v2)")
        self._torch = torch

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            return self._torch.tanh(self.net(self._torch.from_numpy(x.astype(np.float32))[None]))[0].numpy().astype(np.float64)


class Follower:
    """Keeps the follow network's memory (its last action and, for v1, the leaky integral of the error) and edits goals.

    Args:
        net: ``FollowNet``, or any ``(inputs,) -> (outputs,)`` callable; its ``outputs`` attribute (16 when it has
            none) picks v1 or v2.
        joint_names: The goal's joint order.
        max_error: Errors longer than this, in metres, are shortened before the network sees them. Training put the
            target up to 0.6 m away and moved it by up to 1.2 m.
        action_filter: Weight of the network's new output each tick, an exponential moving average; 1 is none. The
            filtered action is what edits the goal and what the network sees as its last. v2 alone shook the robot at
            twice SuperMimic's high-frequency joint acceleration; 0.3 brought that to v1's and kept its accuracy
            (follow/eval.py, offset 0 and 0.5 m).
    """

    def __init__(self, net, joint_names, max_error: float = 0.8, action_filter: float = 1.0):
        if not 0.0 < action_filter <= 1.0:
            raise ValueError("action_filter must be in (0, 1]")
        self.net, self.max_error, self.action_filter = net, max_error, action_filter
        names = list(joint_names)
        self.dim = getattr(net, "outputs", EDIT_DIM)
        self.version = 2 if self.dim == FULL_EDIT_DIM else 1
        self.legs = np.array([names.index(n) for n in LEG_JOINTS])
        self.scales = joint_scales(names)
        self.reset()

    def reset(self) -> None:
        self.prev = np.zeros(self.dim)
        self.filtered = np.zeros(self.dim)
        self.integral = np.zeros(2)
        self.action = np.zeros(self.dim)
        self.error = np.zeros(2)

    def __call__(self, history: np.ndarray, goal: np.ndarray, target_xy, root_xy, root_yaw: float, ref_yaw: float,
                 strength: float = 1.0) -> np.ndarray:
        """The edited goal. ``strength`` in [0, 1] fades the edit in and out; 0 returns ``goal`` itself."""
        err = error_in_heading(target_xy, root_xy, root_yaw)
        self.error = err
        length = float(np.linalg.norm(err))
        if length > self.max_error:
            err = err * (self.max_error / length)
        yaw_err = [wrap(ref_yaw - root_yaw)]
        if self.version == 2:
            x = np.concatenate((latest_frame(history), goal, err, yaw_err, self.prev))
        else:
            self.integral = np.clip(INTEGRAL_DECAY * self.integral + err * CONTROL_DT, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
            x = np.concatenate((latest_frame(history), goal[:FRAME_DIM], goal[FRAMES_END:FRAMES_END + 6], err,
                                yaw_err, self.integral, self.prev))
        self.filtered = self.filtered + self.action_filter * (self.net(x) - self.filtered)
        action = self.filtered * strength
        rate = (action - self.prev) / CONTROL_DT
        self.prev, self.action = action, action
        if strength <= 0.0:
            return goal
        if self.version == 2:
            return edit_goal_full(goal, action, rate, self.scales)
        return edit_goal(goal, action, rate, self.legs)
