# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Command structs fed to the SONIC core, and the AGILE-style velocity command -> planner command mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class LocomotionMode(IntEnum):
    """Planner locomotion modes (localmotion_kplanner.hpp:78-106)."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19
    LEDGE_WALKING = 20
    OBJECT_CARRYING = 21
    STEALTH_WALK_2 = 22
    HAPPY_DANCE_WALK = 23
    ZOMBIE_WALK = 24
    GUN_WALK = 25
    SCARE_WALK = 26


STATIC_MODES = (0, 4, 5, 6, 7, 9)
"""``is_static_motion_mode`` (localmotion_kplanner.hpp:108-115)."""
BOXING_REPLAN_MODES = (11, 12, 13, 15, 16)
"""Modes replanned every 1.0 s as "boxing" (g1_deploy_onnx_ref.cpp:3681-3686)."""


@dataclass
class PlannerCommand:
    """Batched planner inputs (``MovementState``, localmotion_kplanner.hpp:60-73), all in the planner frame.

    Attributes:
        mode: ``(E,)`` long, :class:`LocomotionMode`.
        movement_direction: ``(E, 3)`` desired movement direction (unit or scaled, z = 0).
        facing_direction: ``(E, 3)`` desired facing direction (unit, z = 0).
        speed: ``(E,)`` target speed in m/s; ``-1`` = mode default, ``0`` = stationary.
        height: ``(E,)`` target body height for squat/kneel modes; ``-1`` = mode default.
    """

    mode: torch.Tensor
    movement_direction: torch.Tensor
    facing_direction: torch.Tensor
    speed: torch.Tensor
    height: torch.Tensor

    @classmethod
    def idle(cls, num_envs: int, device: str | torch.device = "cpu", speed: float = -1.0) -> PlannerCommand:
        facing = torch.zeros(num_envs, 3, device=device)
        facing[:, 0] = 1.0
        return cls(
            mode=torch.zeros(num_envs, dtype=torch.long, device=device),
            movement_direction=torch.zeros(num_envs, 3, device=device),
            facing_direction=facing,
            speed=torch.full((num_envs,), speed, device=device),
            height=torch.full((num_envs,), -1.0, device=device),
        )

    def clone(self) -> PlannerCommand:
        return PlannerCommand(*(t.clone() for t in self.as_tuple()))

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (self.mode, self.movement_direction, self.facing_direction, self.speed, self.height)

    def assign(self, other: PlannerCommand, mask: torch.Tensor):
        """In-place ``self[mask] = other[mask]``."""
        for dst, src in zip(self.as_tuple(), other.as_tuple()):
            dst[mask] = src[mask]


@dataclass
class VelocityCommandCfg:
    """Mapping from ``[vx, vy, wz, hip_height]`` (AGILE / ``LocomotionRootCmdRetargeter``) to planner commands.

    Speed bands follow gamepad.hpp:348-385 (slow walk 0.2-0.8 m/s, walk 0.8-1.5, run 1.5-3.0); the minimum
    slow-walk speed is 0.1 m/s as in the PICO manager (pico_manager_thread_server.py:1814-1815).
    """

    linear_deadband: float = 0.05
    """Planar speed [m/s] below which the robot idles."""
    yaw_deadband: float = 0.05
    """Yaw rate [rad/s] below which the facing direction is held (avoids replans on stick noise)."""
    slow_walk_speed: tuple[float, float] = (0.1, 0.8)
    walk_speed: tuple[float, float] = (0.8, 1.5)
    run_speed: tuple[float, float] = (1.5, 3.0)
    allow_run: bool = False
    squat_hip_height: float | None = 0.65
    """``hip_height`` below this switches to IDLE_SQUAT; ``None`` disables squatting."""
    squat_height_range: tuple[float, float] = (0.1, 0.8)
    """Planner squat height clamp (gamepad.hpp:369-372)."""


class VelocityCommandMapper:
    """Integrates yaw rate into a facing direction and maps body-frame velocity to planner commands.

    The facing yaw lives in the planner frame, which the core aligns with the robot's heading when the planner
    is initialised. ``(vx, vy)`` are in the commanded facing frame (robot frame when tracking is good), as in
    the PICO manager (pico_manager_thread_server.py:1823-1830).
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", cfg: VelocityCommandCfg | None = None):
        self.cfg = cfg or VelocityCommandCfg()
        self.device = torch.device(device)
        self.yaw = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None, yaw: torch.Tensor | float = 0.0):
        idx = slice(None) if env_ids is None else env_ids
        self.yaw[idx] = yaw

    def __call__(self, command: torch.Tensor, dt: float) -> PlannerCommand:
        """``command``: ``(E, 4)`` ``[vx, vy, wz, hip_height]``; ``dt``: policy period [s]."""
        cfg = self.cfg
        vx, vy, wz, hip = command.unbind(-1)
        turn = wz.abs() >= cfg.yaw_deadband
        self.yaw = torch.where(turn, self.yaw + wz * dt, self.yaw)
        cos, sin = torch.cos(self.yaw), torch.sin(self.yaw)
        zeros = torch.zeros_like(cos)
        facing = torch.stack((cos, sin, zeros), dim=-1)

        speed = torch.hypot(vx, vy)
        moving = speed >= cfg.linear_deadband
        inv = torch.where(moving, 1.0 / speed.clamp_min(1e-6), zeros)
        lx, ly = vx * inv, vy * inv
        movement = torch.stack((cos * lx - sin * ly, sin * lx + cos * ly, zeros), dim=-1)

        slow = speed <= cfg.slow_walk_speed[1]
        run = (speed > cfg.walk_speed[1]) if cfg.allow_run else torch.zeros_like(moving)
        mode = torch.full_like(vx, int(LocomotionMode.WALK), dtype=torch.long)
        mode = torch.where(slow, int(LocomotionMode.SLOW_WALK), mode)
        mode = torch.where(run, int(LocomotionMode.RUN), mode)
        target = torch.where(
            slow,
            speed.clamp(*cfg.slow_walk_speed),
            torch.where(run, speed.clamp(*cfg.run_speed), speed.clamp(*cfg.walk_speed)),
        )

        mode = torch.where(moving, mode, int(LocomotionMode.IDLE))
        target = torch.where(moving, target, torch.full_like(target, -1.0))
        movement = torch.where(moving.unsqueeze(-1), movement, torch.zeros_like(movement))
        height = torch.full_like(target, -1.0)
        if cfg.squat_hip_height is not None:
            squat = hip < cfg.squat_hip_height
            mode = torch.where(squat, int(LocomotionMode.IDLE_SQUAT), mode)
            target = torch.where(squat, torch.full_like(target, -1.0), target)
            movement = torch.where(squat.unsqueeze(-1), torch.zeros_like(movement), movement)
            height = torch.where(squat, hip.clamp(*cfg.squat_height_range), height)
        return PlannerCommand(mode, movement, facing, target, height)
