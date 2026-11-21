# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""SONIC as a telescene controller: headset input to the action of a G1 env with ``hands`` then ``sonic`` terms.

* :class:`SonicSwitchController`, SONIC ``switch`` mode (122-D): standing, hybrid (head and controllers drive the upper
  body, the planner walks the legs) or whole-body POSE, chosen each tick.
* :class:`SonicPoseController`, SONIC ``pose`` mode (101-D): POSE or standing.

Hands 14 come first in both. The planner does not run in POSE, so velocity is ignored then and the robot only moves as
far as copying the legs carries it.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .inputs import Locomotion, body_arrays, hand_columns


class SonicPoseController:
    """Args:
        env: The unwrapped environment, whose action term must be SONIC POSE.
        skeleton: Path of ``human_joints_info.pkl``.
    """

    def __init__(self, env, skeleton: str | Path):
        from .core.pico_body import load_smpl_skeleton
        from .pose_retargeter import SonicPoseRetargeter

        self.node = SonicPoseRetargeter(load_smpl_skeleton(Path(skeleton)), device="cpu")
        self.locomotion = Locomotion()
        self.action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
        self.planner = next(t._core.planner for t in env.action_manager._terms.values() if hasattr(t, "_core"))

    def reset(self) -> None:
        self.node.reset()
        self.locomotion.reset()

    def act(self, control):
        action = self.action
        action[0, :14] = torch.as_tensor(hand_columns(*control.triggers), device=action.device)
        action[0, 14:18] = torch.as_tensor(
            self.locomotion(control.left_axis, control.right_axis, control.dt), device=action.device)
        action[0, 14] += control.velocity[0]
        action[0, 15] += control.velocity[1]
        action[0, 16] += control.velocity[2]
        if control.body is None:
            action[0, 18] = 0.0
        else:
            quat, valid = body_arrays(control.body)
            action[0, 18:101] = torch.as_tensor(self.node.retarget(quat, valid, control.tracking).copy(),
                                                device=action.device)
        return action

    def status(self) -> str:
        return f"late plans {self.planner.late_plans}"


class SonicSwitchController:
    """Hybrid by default, whole-body POSE while the operator has it toggled on with A. Both are SONIC, one session.

    Hybrid: head and controllers drive the upper body and the planner walks the legs to the operator. The hands come
    from the controllers, which are measured, so this is the more exact of the two. POSE: the whole body copies the
    operator, as SMPL, with nothing keeping the robot where the operator is.

    With the headset off, in the app's menu or mirror, or with no controllers yet, the planner stands the robot still.
    """

    PLANNER, HYBRID, POSE = 0.0, 1.0, 2.0
    MODE_NAMES = {0.0: "standing", 1.0: "hybrid", 2.0: "pose"}
    HANDS, LOCO, MODE, VR, SMPL = 0, 14, 18, 19, 40

    def __init__(self, env, skeleton: str | Path):
        from .core.pico_body import load_smpl_skeleton
        from .pose_retargeter import SonicPoseRetargeter

        self.node = SonicPoseRetargeter(load_smpl_skeleton(Path(skeleton)), device="cpu")
        self.locomotion = Locomotion()
        self.robot = env.scene["robot"]
        self.action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
        self.action[0, self.SMPL + 72 + 3] = 1.0  # a valid root quaternion before the first body
        self.planner = next(t._core.planner for t in env.action_manager._terms.values() if hasattr(t, "_core"))
        self._grips = None
        self._mode = self.PLANNER

    def reset(self) -> None:
        self.node.reset()
        self.locomotion.reset()

    def _root(self):
        data = self.robot.data
        root, q = data.root_pos_w, data.root_quat_w
        root = root.torch if hasattr(root, "torch") else root
        q = q.torch if hasattr(q, "torch") else q
        qx, qy, qz, qw = q[0].tolist()
        return root[0][:2].tolist(), math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def _hybrid_targets(self, control):
        """Left wrist, right wrist, head; or None until there is a body and both controllers have been seen."""
        from telescene.hands import head_target, is_valid, wrist_target

        frame = control.frame
        if frame.grips is not None and is_valid(frame.grips[0]) and is_valid(frame.grips[1]):
            self._grips = frame.grips.copy()
        if self._grips is None or control.body is None:
            return None
        xy, yaw = self._root()
        wrists = [wrist_target(g, control.body, xy, yaw, control.anchor_yaw, control.world_scale) for g in self._grips]
        return (*wrists, head_target(frame.head, control.body, yaw, control.anchor_yaw))

    def act(self, control):
        action = self.action
        device = action.device
        action[0, self.HANDS:self.LOCO] = torch.as_tensor(hand_columns(*control.triggers), device=device)
        action[0, self.LOCO:self.MODE] = torch.as_tensor(
            self.locomotion(control.left_axis, control.right_axis, control.dt), device=device)
        for i in range(3):
            action[0, self.LOCO + i] += control.velocity[i]
        frame = control.frame
        mode = self.PLANNER
        if frame is not None and not (frame.menu or frame.mirror or frame.not_worn):
            targets = self._hybrid_targets(control)
            if targets is not None:
                mode = self.HYBRID
                for k, pose in enumerate(targets):
                    start = self.VR + 7 * k
                    action[0, start:start + 7] = torch.as_tensor(pose, dtype=action.dtype, device=device)
            if control.body is not None:
                quat, valid = body_arrays(control.body)
                out = self.node.retarget(quat, valid, control.tracking)
                action[0, self.SMPL:] = torch.as_tensor(out[1:].copy(), device=device)
                if out[0] > 0.5:
                    mode = self.POSE
        action[0, self.MODE] = mode
        self._mode = mode
        return action

    def status(self) -> str:
        return f"{self.MODE_NAMES[self._mode]} | late plans {self.planner.late_plans}"
