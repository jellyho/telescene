# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Isaac Lab configs for the busy box cell, built from the exporter's USD and joint map.

The cell USD (room, props, mechanisms) is spawned once per environment by an asset config
(:func:`busybox_cell_cfg`). Its articulations (:func:`busybox_articulation_cfg`) and free props
(:func:`busybox_prop_cfgs`) are views on prims inside it (``spawn=None``). Isaac Lab treats an
articulation's ``init_state`` as its root *link* pose and the default reset event writes it back,
while the busy box root body is not at the cell origin (``o10_busybox`` sits at (-0.1, 0.2, 0)
with yaw -90 deg). Spawning the cell on its own keeps the USD at the env origin and lets each view
carry the body's real layout pose.

Every mapped joint is passive: an implicit drive with zero stiffness whose damping, Coulomb
friction and armature come from ``joint_map.json`` (MJCF ``damping``, ``frictionloss``,
``armature``). With the cell at the env origin (the default, and what the station success term
assumes) cell coordinates (garage_layout world: Z up, metres) are env-local coordinates.

Fixed bases stay as authored. Each fixed-base root carries the ``ArticulationRootAPI`` on a
``PhysicsFixedJoint`` whose ``body0`` is the cell's default prim (a static Xform), which PhysX
treats as a world joint. ``fix_root_link`` is left None: Isaac Lab only recognises world joints
with an empty body target, so it would try to create one on the (non-rigid) joint prim and raise.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

from .layout import IDENTITY_POSE, Pose, compose_pose
from .joint_map import BusyBoxJointMap, JointMapError, load_joint_map, resolve_joint_map_path, resolve_usd_path

BUSYBOX_PRIM_PATH = "{ENV_REGEX_NS}/BusyBox"
"""Default prim path of the cell in each environment."""

PASSIVE_ACTUATOR_NAME = "busybox_passive"
"""Key of the actuator group that holds every mapped joint of an articulation."""


def _load(joint_map: BusyBoxJointMap | str | os.PathLike[str] | None, usd_path=None) -> BusyBoxJointMap:
    if isinstance(joint_map, BusyBoxJointMap):
        return joint_map
    return load_joint_map(resolve_joint_map_path(joint_map, usd_path))


def busybox_cell_cfg(
    *,
    prim_path: str = BUSYBOX_PRIM_PATH,
    usd_path: str | os.PathLike[str] | None = None,
    joint_map: BusyBoxJointMap | str | os.PathLike[str] | None = None,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    self_collisions: bool = True,
    solver_position_iterations: int = 8,
    solver_velocity_iterations: int = 4,
) -> AssetBaseCfg:
    """The cell USD, spawned once per environment.

    The articulation-root properties of the busy box (the map's articulation) are authored here,
    since the articulation itself is a view without a spawner.

    Args:
        prim_path: Prim path (expression) to spawn the cell at.
        usd_path: Cell USD. Defaults to ``$TELESCENE_BUSYBOX_USD``, then the file the joint map names, then the
            single USD file next to the map (see :func:`.joint_map.resolve_usd_path`).
        joint_map: A loaded map or its path. Defaults to ``$TELESCENE_BUSYBOX_JOINT_MAP``, then
            ``joint_map.json`` next to the USD, then ``assets/busybox/full/``.
        pos: Cell position in the environment frame [m]. Pass the same pose to the views.
        rot: Cell orientation (x, y, z, w) in the environment frame.
        self_collisions: Let non-adjacent busy box links collide, as MJCF sibling bodies do. The MJCF
            contact excludes arrive in the USD as filtered pairs.
        solver_position_iterations: PhysX position iterations of the busy box articulation.
        solver_velocity_iterations: PhysX velocity iterations of the busy box articulation.

    Raises:
        FileNotFoundError: If the joint map or the USD is missing (the message names the env vars).
        JointMapError: If the joint map is malformed.
    """
    joint_map = _load(joint_map, usd_path)
    usd = resolve_usd_path(usd_path, joint_map)
    articulation_props = None
    if joint_map.articulation_root is not None:
        # keys are regex suffixes of the spawn prim; only the busy box root, not the other articulations
        articulation_props = {
            re.escape(joint_map.articulation_root): [
                PhysxArticulationCfg(
                    enabled_self_collisions=self_collisions,
                    solver_position_iteration_count=solver_position_iterations,
                    solver_velocity_iteration_count=solver_velocity_iterations,
                ),
                NewtonArticulationCfg(self_collision_enabled=self_collisions),
            ]
        }
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=False,
            articulation_props=articulation_props,
        ),
    )


ROLLING_RESISTANCE: dict[str, float] = {}
"""Damping [N m s/rad] on joint names matching each pattern, replacing the map's. Empty: the map is right.

It held ``bror_cart_roll_.*: 0.01`` until palette r6 put that value in the MJCF itself. r6 with no override gives the
same as r5 with it, measured with ``scripts/dev/check_cart.py --coast``: a 20 N push for 0.3 s coasts 16 cm and stops.
Joint friction is no substitute: it does nothing until about 0.5, where it locks the wheels."""


def apply_damping_overrides(params: dict, joint_map: BusyBoxJointMap, overrides: dict[str, float]) -> dict:
    """``actuator_params`` with the damping of matching joints replaced."""
    for pattern, damping in overrides.items():
        for joint in joint_map.joints:
            if re.fullmatch(pattern, joint.name):
                params["damping"][joint.regex] = damping
    return params


def busybox_articulation_cfg(
    joint_map: BusyBoxJointMap | str | os.PathLike[str] | None = None,
    *,
    cell_prim_path: str = BUSYBOX_PRIM_PATH,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    damping_overrides: dict[str, float] | None = None,
) -> ArticulationCfg:
    """A view of one articulation of the spawned cell (the map's articulation, ``o10_busybox`` by default).

    Load other articulations with ``load_joint_map(path, articulation="o34_step_platform")``.

    Args:
        joint_map: A loaded map or its path (see :func:`busybox_cell_cfg`).
        cell_prim_path: Prim path (expression) of the cell, as passed to :func:`busybox_cell_cfg`.
        pos: Cell position in the environment frame [m], as passed to :func:`busybox_cell_cfg`.
        rot: Cell orientation (x, y, z, w), as passed to :func:`busybox_cell_cfg`.
        damping_overrides: Joint name patterns to damping. None uses :data:`ROLLING_RESISTANCE`, ``{}`` none.

    Raises:
        JointMapError: If the map does not place the articulation's root body in the cell.
    """
    joint_map = _load(joint_map)
    body_path, root_rel = joint_map.view_paths()
    if joint_map.root_pose is None:
        raise JointMapError(f"{joint_map.path}: no layout pose for the root body {joint_map.articulation!r}")
    root_pos, root_rot = compose_pose((pos, rot), joint_map.root_pose)
    return ArticulationCfg(
        prim_path=cell_prim_path + body_path,
        spawn=None,
        articulation_root_prim_path=root_rel,
        init_state=ArticulationCfg.InitialStateCfg(pos=root_pos, rot=root_rot, joint_pos=joint_map.init_joint_pos()),
        actuators={PASSIVE_ACTUATOR_NAME: ImplicitActuatorCfg(**apply_damping_overrides(
            joint_map.actuator_params(), joint_map, ROLLING_RESISTANCE if damping_overrides is None else damping_overrides))},
    )


def busybox_prop_cfgs(
    joint_map: BusyBoxJointMap,
    bodies: Iterable[str] | None = None,
    prim_path: str = BUSYBOX_PRIM_PATH,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, RigidObjectCfg]:
    """Rigid-object views on the free props of the cell, keyed by MJCF body name.

    The cell USD already contains the props, so nothing is spawned. Registering them lets the
    default reset event put them back at their layout pose and lets the station success term
    read their poses (it finds a rigid object whose scene name is the MJCF body name).

    Args:
        joint_map: The loaded joint map (needs ``rigid_objects``, ``body_paths`` and ``body_init``).
        bodies: MJCF bodies to register. Defaults to every rigid object of the map; names that
            are not rigid objects are ignored.
        prim_path: Prim path (expression) of the cell, as passed to :func:`busybox_cell_cfg`.
        pos: Cell position in the environment frame [m], as passed to :func:`busybox_cell_cfg`.
        rot: Cell orientation (x, y, z, w), as passed to :func:`busybox_cell_cfg`.
    """
    names = joint_map.rigid_objects if bodies is None else [b for b in bodies if b in joint_map.rigid_objects]
    cfgs = {}
    for name in names:
        if name not in joint_map.body_paths:
            raise JointMapError(f"{joint_map.path}: rigid object {name!r} has no 'bodies.{name}.usd_rel_path'")
        layout: Pose = joint_map.body_init.get(name, IDENTITY_POSE)
        body_pos, body_rot = compose_pose((pos, rot), layout)
        cfgs[name] = RigidObjectCfg(
            prim_path=prim_path + joint_map.body_paths[name],
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=body_pos, rot=body_rot),
        )
    return cfgs
