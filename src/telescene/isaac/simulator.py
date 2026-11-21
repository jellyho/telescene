"""An Isaac Lab manager-based environment as a telescene ``Simulator``."""

from __future__ import annotations

import math

import numpy as np

from ..wire import pack_scene
from .scene import _t, extract_scene


HAND_SENSORS = ("telescene_left_hand", "telescene_right_hand")


def prepare_env_cfg(env_cfg, hand_bodies: tuple[str, str] | None = None, robot: str = "robot") -> None:
    """Settings a live teleop run needs. Call before ``gym.make``.

    ``hand_bodies`` is a left and a right body-name regex, such as ``G1_HAND_BODIES``. It adds contact sensors, which
    make the controllers vibrate when the hands touch something.

    Demo recorders keep about 60 tensors per tick in memory until an episode is exported, and live teleop never
    exports. On a G1 scene that took garbage collections from 40 ms up to 375 ms over 10 minutes. Time-outs would reset
    the scene under the operator.
    """
    env_cfg.scene.num_envs = 1
    env_cfg.recorders = None
    if getattr(env_cfg, "xr", None) is not None:
        env_cfg.xr = None
    if getattr(env_cfg, "isaac_teleop", None) is not None:
        env_cfg.isaac_teleop = None
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out = None
    if hand_bodies is not None:
        from isaaclab.sensors import ContactSensorCfg

        robot_cfg = getattr(env_cfg.scene, robot)
        robot_cfg.spawn.activate_contact_sensors = True
        for name, bodies in zip(HAND_SENSORS, hand_bodies):
            setattr(env_cfg.scene, name, ContactSensorCfg(prim_path=f"{robot_cfg.prim_path}/{bodies}"))


class IsaacLabSimulator:
    """Wraps one environment, env 0, for one operator.

    Args:
        env: A gymnasium-wrapped ``ManagerBasedEnv``, already made and reset.
        robot: Name of the operated robot in ``env.scene``.
        **scene_options: Passed to :func:`extract_scene`.
    """

    def __init__(self, env, robot: str = "robot", **scene_options):
        self.env, self.unwrapped = env, env.unwrapped
        self.robot = self.unwrapped.scene[robot]
        nodes, self.parts, self.stats = extract_scene(self.unwrapped, **scene_options)
        self.scene_blob = pack_scene(nodes, len(self.parts.entries))
        self.triangles = sum(len(n.indices) for n in nodes) // 3
        self.nodes = len(nodes)
        sensors = self.unwrapped.scene.sensors
        self._hands = [sensors[n] for n in HAND_SENSORS] if all(n in sensors for n in HAND_SENSORS) else None

    def reset(self) -> None:
        self.env.reset()

    def step(self, action) -> bool:
        _, _, terminated, truncated, _ = self.env.step(action)
        return bool(terminated.any() or truncated.any())

    def root_pose(self) -> tuple[float, float, float]:
        data = self.robot.data
        x, y = _t(data.root_pos_w)[0][:2].tolist()
        qx, qy, qz, qw = _t(data.root_quat_w)[0].tolist()
        return x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def link_poses(self) -> np.ndarray:
        return self.parts.poses()

    def snapshot(self) -> dict[str, np.ndarray]:
        """Everything needed to put the scene back as it was this tick: root states and joint positions of every
        articulation, and root states of every rigid object, by scene name."""
        out = {}
        scene = self.unwrapped.scene
        for name, asset in scene.articulations.items():
            out[f"art/{name}/root"] = _t(asset.data.root_state_w)[0].cpu().numpy()
            out[f"art/{name}/joints"] = _t(asset.data.joint_pos)[0].cpu().numpy()
        for name, asset in scene.rigid_objects.items():
            out[f"rigid/{name}/root"] = _t(asset.data.root_state_w)[0].cpu().numpy()
        return out

    def hand_forces(self) -> tuple[float, float]:
        if self._hands is None:
            return (0.0, 0.0)
        left, right = (float(_t(s.data.net_forces_w)[0].norm(dim=-1).max()) for s in self._hands)
        return left, right

    def anchor(self) -> tuple[tuple[float, float, float], float]:
        """The robot's spawn. Yaw comes from the configured spawn, so that a fall or a step cannot skew the room."""
        data = _t(self.robot.data.root_pos_w)[0].tolist()
        rot = self.robot.cfg.init_state.rot
        return (data[0], data[1], 0.0), 2.0 * math.atan2(rot[2], rot[3])
