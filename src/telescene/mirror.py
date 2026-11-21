"""A mirror in the operator's room: see your own tracked body. Needs no simulator.

The tracked joints appear as dots. If a conversion is given, such as the SMPL a controller receives, it appears as a
stick figure over them, so what the conversion loses can be seen, and the log gives the error per joint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .isaac.scene import _cube, tessellate_capsule, tessellate_sphere
from .wire import Node, pack_scene

_DOT_RGB, _FIGURE_RGB, _FRAME_RGB = (0.1, 0.8, 0.3), (0.9, 0.45, 0.1), (0.25, 0.25, 0.28)
_NEAREST = 1e-6


def xr_to_sim(points: np.ndarray) -> np.ndarray:
    """OpenXR room axes to the scene's, with the anchor at the origin: forward -Z is +X, left -X is +Y, up +Y is +Z."""
    return np.stack((-points[..., 2], -points[..., 0], points[..., 1]), axis=-1)


def reflect(points: np.ndarray, distance: float) -> np.ndarray:
    """Across the mirror, a plane ``distance`` ahead of the room origin. OpenXR axes."""
    out = points.copy()
    out[..., 2] = -2.0 * distance - out[..., 2]
    return out


def bone_pose(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """``(7,)`` xyz, xyzw that places a capsule built along Z between two points."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    pose = np.array([*((start + end) / 2), 0.0, 0.0, 0.0, 1.0], np.float32)
    if length < _NEAREST:
        return pose
    d = direction / length
    w = 1.0 + float(d[2])
    if w < _NEAREST:  # pointing straight down
        pose[3:7] = (1.0, 0.0, 0.0, 0.0)
        return pose
    q = np.array([-d[1], d[0], 0.0, w])
    pose[3:7] = q / np.linalg.norm(q)
    return pose


def joint_errors(tracked: np.ndarray, converted: np.ndarray) -> np.ndarray:
    """Per joint distance with the pelvises (joint 0) together. Includes any difference in body proportions."""
    return np.linalg.norm((tracked - tracked[0]) - (converted - converted[0]), axis=-1)


class MirrorView:
    """Both the ``Simulator`` and the ``Controller`` of a session: the operator's input is all that moves.

    Args:
        parents: Parent of each joint, -1 for the root. 24 joints.
        bone_lengths: Length of the bone from each joint to its parent. Unused for the root.
        convert: ``convert(body) -> (24, 3)`` joints in OpenXR axes, positioned anywhere. None shows only the dots.
        distance: Metres from the room origin to the mirror. You see yourself at about twice this.
        names: Joint names for the log.
    """

    JOINT_DOT, FIGURE_DOT = 0.025, 0.02

    def __init__(self, parents: Sequence[int], bone_lengths: Sequence[float],
                 convert: Callable[[np.ndarray], np.ndarray] | None = None, distance: float = 1.0,
                 names: Sequence[str] | None = None):
        self.parents, self.convert, self.distance = list(parents), convert, distance
        self.names = list(names) if names else [str(i) for i in range(len(self.parents))]
        self.bones = [(i, p) for i, p in enumerate(self.parents) if p >= 0]
        n = len(self.parents)
        self._num_poses = n + (n + len(self.bones) if convert else 0)
        self._poses = np.zeros((self._num_poses, 7), np.float32)
        self._poses[:, 6] = 1.0
        self._poses[:, 2] = -10.0  # below the floor until the first body arrives
        self._errors: list[np.ndarray] = []
        self.scene_blob = pack_scene(self._nodes(bone_lengths), self._num_poses)

    def _nodes(self, bone_lengths) -> list[Node]:
        n = len(self.parents)
        nodes = [Node(i, _DOT_RGB, *tessellate_sphere(self.JOINT_DOT)) for i in range(n)]
        if self.convert is not None:
            nodes += [Node(n + i, _FIGURE_RGB, *tessellate_sphere(self.FIGURE_DOT)) for i in range(n)]
            for k, (child, _) in enumerate(self.bones):
                radius = 0.018
                height = max(float(bone_lengths[child]) - 2 * radius, 0.001)
                nodes.append(Node(2 * n + k, _FIGURE_RGB, *tessellate_capsule(radius, height)))
        return nodes + self._frame()

    def _frame(self, width: float = 1.2, height: float = 2.0, bar: float = 0.02) -> list[Node]:
        """An outline where the mirror is, so that the reflection has somewhere to be."""
        unit, faces = _cube(1.0)
        at = self.distance
        boxes = (((at, -width / 2, height / 2), (bar, bar, height)), ((at, width / 2, height / 2), (bar, bar, height)),
                 ((at, 0.0, height), (bar, width, bar)), ((at, 0.0, 0.0), (bar, width, bar)))
        return [Node(-1, _FRAME_RGB, unit * np.array(size, np.float32) + np.array(centre, np.float32), faces)
                for centre, size in boxes]

    # Controller
    def reset(self) -> None:
        self._errors.clear()

    def act(self, control):
        if control.body is None:
            return None
        tracked = control.body[:, :3].astype(np.float64)
        n = len(self.parents)
        self._poses[:n, :3] = xr_to_sim(reflect(tracked, self.distance))
        self._poses[:n, 3:6], self._poses[:n, 6] = 0.0, 1.0
        if self.convert is not None:
            joints = np.asarray(self.convert(control.body), np.float64)
            self._errors.append(joint_errors(tracked, joints))
            aligned = joints - joints[0] + tracked[0]
            placed = xr_to_sim(reflect(aligned, self.distance))
            self._poses[n:2 * n, :3] = placed
            self._poses[n:2 * n, 3:6], self._poses[n:2 * n, 6] = 0.0, 1.0
            for k, (child, parent) in enumerate(self.bones):
                self._poses[2 * n + k] = bone_pose(placed[parent], placed[child])
        return None

    def status(self) -> str:
        if not self._errors:
            return ""
        mean = np.mean(self._errors, axis=0) * 100
        self._errors.clear()
        worst = np.argsort(mean)[::-1][:3]
        names = ", ".join(f"{self.names[i]} {mean[i]:.0f}" for i in worst)
        return f"SMPL vs tracked: mean {mean.mean():.1f} cm, worst {names} cm"

    # Simulator
    def step(self, action) -> bool:
        return False

    def root_pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def link_poses(self) -> np.ndarray:
        return self._poses

    def anchor(self) -> tuple[tuple[float, float, float], float]:
        return (0.0, 0.0, 0.0), 0.0

    def hand_forces(self) -> tuple[float, float]:
        return (0.0, 0.0)
