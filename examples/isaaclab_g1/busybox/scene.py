# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""The BusyBox scenes: a pegboard work bench (the "palette"), a trolley and a step stool, with drawers, a slide shelf,
knobs, bins and small objects. ``assets/busybox/scenes.json`` lists them; the headset menu offers each as a button.

A scene is one converted cell: a USD of primitives with its physics, and a ``joint_map.json`` saying which USD joint
each modelled joint became and how stiff, damped and rubbed it is. ``use_busybox`` swaps a G1 env's table for one:

    entry = scene_entry("sort_two_bins")
    use_busybox(env_cfg, entry)          # before gym.make

The env must have ``scene.robot`` and may have ``scene.table`` and ``scene.box``, which are removed. The cell is
spawned at the env origin, so cell coordinates are env coordinates (Z up, metres).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .joint_map import (DEFAULT_ARTICULATION, JointMapError, NoMovableJointsError, articulation_names, load_joint_map,
                        resolve_joint_map_path)
from .layout import yaw_quat_xyzw

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES_JSON = REPO_ROOT / "assets" / "busybox" / "scenes.json"
DEFAULT_SPAWN = (1.5, 0.2, 180.0)
"""x, y [m] and yaw [deg]: in front of the bench, facing it (-x)."""

COLOUR_NAME = re.compile(r"col_([0-9a-fA-F]{6})[0-9a-fA-F]{2}")
"""The converter names each material after its sRGB colour, ``col_rrggbbaa``, and sets no diffuse colour."""

PICK_STAND_AT, PICK_STAND_HEIGHT, PICK_STAND_SIZE = (1.3, 0.65), 0.70, 0.25
"""A 70 cm stand at the robot's right as it faces the bench from ``DEFAULT_SPAWN``, clear of the stool and the cart."""


def scenes(path: str | Path = SCENES_JSON) -> list[dict]:
    """The manifest's entries whose USD is present, grouped as the manifest first names each group."""
    entries = [e for e in json.loads(Path(path).read_text())["scenes"] if (REPO_ROOT / e["usd"]).is_file()]
    first_seen: dict[str, int] = {}
    for e in entries:
        first_seen.setdefault(e.get("group", ""), len(first_seen))
    return sorted(entries, key=lambda e: first_seen[e.get("group", "")])


def scene_entry(name: str, path: str | Path = SCENES_JSON) -> dict:
    for e in json.loads(Path(path).read_text())["scenes"]:
        if e["name"] == name:
            return e
    raise KeyError(f"no BusyBox scene '{name}' in {path}: one of {[e['name'] for e in scenes(path)]}")


def busybox_colour(prim) -> tuple[float, float, float] | None:
    """The colour a cell prim's material name gives, linear RGB, or None."""
    from pxr import UsdShade

    from telescene.isaac.scene import srgb_hex_to_linear

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if material:
        match = COLOUR_NAME.fullmatch(material.GetPrim().GetName())
        if match:
            return srgb_hex_to_linear(match.group(1))
    return None


def add_pick_box(env_cfg, size=(0.35, 0.05, 0.05)) -> None:
    """A 1 kg box on a stand at the robot's right, to carry to a drawer. Its long side runs across the robot's view, so
    both hands can take it from the front."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg

    at, stand = PICK_STAND_AT, PICK_STAND_HEIGHT
    env_cfg.scene.pick_stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PickStand",
        spawn=sim_utils.CuboidCfg(size=(PICK_STAND_SIZE, PICK_STAND_SIZE, stand), collision_props=CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.30, 0.33))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*at, stand / 2)))
    env_cfg.scene.pick_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickBox",
        spawn=sim_utils.CuboidCfg(
            size=(size[1], size[0], size[2]), rigid_props=RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), collision_props=CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.40, 0.08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*at, stand + size[2] / 2 + 0.002)))


def use_busybox(env_cfg, entry: dict, pick_box: bool | None = None) -> None:
    """Put the G1 env in the scene ``entry`` (one of ``scenes()``), instead of its table.

    Registers the cell's articulations (drawers, shelf, trolley) and its free objects as scene entities named after
    their bodies, so a reset puts them back. ``pick_box``: add the box on a stand; None follows the entry.
    """
    from isaaclab.utils import replace

    from .cell import busybox_articulation_cfg, busybox_cell_cfg, busybox_prop_cfgs

    usd = REPO_ROOT / entry["usd"]
    path = resolve_joint_map_path(None, usd)
    names = articulation_names(path)
    main = DEFAULT_ARTICULATION if DEFAULT_ARTICULATION in names else (names[0] if names else None)
    try:
        joint_map = load_joint_map(path, articulation=main)
    except JointMapError:
        if names:
            raise
        joint_map = load_joint_map(path, articulation=None, allow_empty=True)  # a lone stool, or the bare floor

    scene = env_cfg.scene
    for name in ("table", "box"):
        if hasattr(scene, name):
            setattr(scene, name, None)
    scene.busybox_cell = cell = busybox_cell_cfg(usd_path=usd, joint_map=joint_map)
    if joint_map.joints:
        scene.busybox = busybox_articulation_cfg(joint_map, cell_prim_path=cell.prim_path)
    for name in names:
        if name == joint_map.articulation:
            continue
        try:
            setattr(scene, name, busybox_articulation_cfg(load_joint_map(path, articulation=name), cell_prim_path=cell.prim_path))
        except NoMovableJointsError:
            continue
    for name, prop in busybox_prop_cfgs(joint_map, prim_path=cell.prim_path).items():
        setattr(scene, name, prop)

    x, y, yaw = DEFAULT_SPAWN
    if entry.get("spawn"):
        (x, y), yaw = entry["spawn"]["xy"], entry["spawn"].get("yaw_deg", 180.0)
    robot = scene.robot
    scene.robot = replace(robot, init_state=replace(robot.init_state, pos=(float(x), float(y), robot.init_state.pos[2]),
                                                    rot=yaw_quat_xyzw(math.radians(yaw))))
    if pick_box if pick_box is not None else entry.get("pick_box", True) is not False:
        add_pick_box(env_cfg)
    print(f"[busybox] scene {entry['name']}: {usd.relative_to(REPO_ROOT)}, robot at ({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg"
          + (f". Task: {entry['task_en']}" if entry.get("task_en") else ""), flush=True)
