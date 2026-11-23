# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Black hands, feet and head on the G1, as the real robot has, for rendering. Visual only: no physics changes.

The G1 USD gives every link the same silver. The headset already draws these parts black (teleop.py's ``colour``);
this does the same for render_take.py's RTX video.
"""

from __future__ import annotations

import re

BLACK_LINKS = re.compile(r"(left|right)_hand_.*|.*ankle_roll_link|head_link")
"""Link names drawn black: the Dex3 hands, the feet and the head."""


def paint_g1(robot_prim_path: str, material_path: str = "/World/Looks/G1Black") -> int:
    """Bind a black material to the black links under ``robot_prim_path``, over whatever their meshes carry.

    Binding on the link prims works on an instanced asset, since they are not instance proxies, and
    ``strongerThanDescendants`` wins over the meshes' own bindings. The head is the exception, see below. Returns how
    many prims were painted.
    """
    from pxr import Usd, UsdShade

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.025, 0.025, 0.03), roughness=0.45, metallic=0.1)
    cfg.func(material_path, cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    painted = 0
    root = stage.GetPrimAtPath(robot_prim_path)
    # The head's mesh is not under head_link but inside torso_link's instanced visuals, where nothing can be bound.
    # Un-instance only such groups, the ones holding a black part of a link that is not itself black.
    for prim in Usd.PrimRange(root):
        if prim.IsInstanceable() and not BLACK_LINKS.fullmatch(prim.GetParent().GetName()):
            inside = Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            if any(BLACK_LINKS.fullmatch(p.GetName()) for p in inside if p != prim):
                prim.SetInstanceable(False)
    for prim in Usd.PrimRange(root):
        if prim.IsInstanceProxy() or not BLACK_LINKS.fullmatch(prim.GetName()):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        painted += 1
    return painted


def paint_when_spawned(robot_cfg) -> None:
    """Make ``robot_cfg``'s spawner paint the G1 as it creates it, before physics starts.

    Painting afterwards cannot un-instance the head's group: that changes the stage's structure, which invalidates
    PhysX's view of the running simulation."""
    spawn = robot_cfg.spawn
    original = spawn.func

    def spawn_and_paint(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        paint_g1(str(prim.GetPath()))
        return prim

    spawn.func = spawn_and_paint
