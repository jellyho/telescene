# SPDX-License-Identifier: MIT
"""Turn the live USD stage into scene nodes, plus the list of what moves. Needs Kit's pxr."""

from __future__ import annotations

import math
import re

import numpy as np

from ..wire import Node, decimate, merge_nodes

DEFAULT_RGB = (0.72, 0.73, 0.75)

G1_HIDDEN_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "head_link",
                    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                    "left_shoulder_roll_link", "right_shoulder_roll_link")
"""For the Unitree G1: the operator stands inside these and they block the view. Hiding a body changes only the
headset's scene. The simulation and any camera recording still have the whole robot."""


G1_HAND_BODIES = ("left_hand_.*", "right_hand_.*")
"""For a Unitree G1 with Dex3 hands: what counts as touching something, for the controllers' vibration."""


def _cube(size: float):
    h = size / 2
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], np.float32)
    f = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return v, np.array(f, np.uint32)


def _axis(points: np.ndarray, axis: str) -> np.ndarray:
    """Tessellators build along Z; move Z to the USD ``axis``."""
    return {"Z": points, "X": points[:, [2, 1, 0]], "Y": points[:, [0, 2, 1]]}[axis]


def _rings(profile: list[tuple[float, float]], segments: int = 14):
    """Revolve ``(radius, z)`` around Z. Closed at both ends by the profile's first and last rings."""
    rows = len(profile)
    verts = [[r * math.cos(2 * math.pi * s / segments), r * math.sin(2 * math.pi * s / segments), z]
             for r, z in profile for s in range(segments)]
    faces = []
    for row in range(rows - 1):
        for s in range(segments):
            a, b = row * segments + s, row * segments + (s + 1) % segments
            c, d = a + segments, b + segments
            faces += [a, b, d, a, d, c]
    for s in range(1, segments - 1):  # caps
        faces += [0, s + 1, s]
        last = (rows - 1) * segments
        faces += [last, last + s, last + s + 1]
    return np.array(verts, np.float32), np.array(faces, np.uint32)


def tessellate_sphere(radius: float):
    profile = [(radius * math.sin(t), -radius * math.cos(t)) for t in np.linspace(0.05, math.pi - 0.05, 8)]
    return _rings(profile)


def tessellate_cylinder(radius: float, height: float, axis: str = "Z"):
    v, f = _rings([(radius, -height / 2), (radius, height / 2)])
    return _axis(v, axis), f


def tessellate_capsule(radius: float, height: float, axis: str = "Z"):
    arc = np.linspace(0.05, math.pi / 2, 4)
    profile = [(radius * math.sin(t), -height / 2 - radius * math.cos(t)) for t in arc]
    profile += [(r, -z) for r, z in reversed(profile)]
    v, f = _rings(profile)
    return _axis(v, axis), f


def triangulate(points: np.ndarray, counts, indices) -> np.ndarray:
    out, cursor = [], 0
    for count in counts:
        out += [v for i in range(1, count - 1) for v in (indices[cursor], indices[cursor + i], indices[cursor + i + 1])]
        cursor += count
    return np.array(out, np.uint32)


def _geometry(prim):
    from pxr import UsdGeom

    kind = prim.GetTypeName()
    if kind == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            return None
        return np.array(points, np.float32), triangulate(points, list(counts), list(indices))
    if kind == "Cube":
        return _cube(UsdGeom.Cube(prim).GetSizeAttr().Get())
    if kind == "Sphere":
        return tessellate_sphere(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    if kind == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return tessellate_cylinder(c.GetRadiusAttr().Get(), c.GetHeightAttr().Get(), c.GetAxisAttr().Get())
    if kind == "Capsule":
        c = UsdGeom.Capsule(prim)
        return tessellate_capsule(c.GetRadiusAttr().Get(), c.GetHeightAttr().Get(), c.GetAxisAttr().Get())
    return None


def srgb_hex_to_linear(hex_rgb: str) -> tuple[float, float, float]:
    """``a19485`` to linear RGB. The headset's swapchain is sRGB, so the shader works in linear."""
    out = []
    for i in (0, 2, 4):
        c = int(hex_rgb[i : i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def material_colour(prim) -> tuple[float, float, float] | None:
    """The bound material's ``diffuseColor`` or ``diffuse_color_constant``, in linear RGB, if it has one set."""
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        return None
    for child in material.GetPrim().GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        for name in ("diffuseColor", "diffuse_color_constant"):
            shader_input = shader.GetInput(name)
            value = shader_input.Get() if shader_input else None
            if value is not None:
                return tuple(float(c) for c in value)[:3]
    return None


def _t(value):
    return value.torch if hasattr(value, "torch") else value


class MovingParts:
    """Everything that moves, in a fixed order: the pose index of a node is an index into this."""

    def __init__(self, env):
        self.entries: list[tuple[object, int | None]] = []
        self.by_prim: dict[str, int] = {}
        scene = env.scene
        for asset in list(scene.articulations.values()):
            for index, name in enumerate(asset.body_names):
                self.entries.append((asset, index))
            self._register(asset, list(asset.body_names), len(self.entries) - len(asset.body_names))
        for asset in list(scene.rigid_objects.values()):
            self.entries.append((asset, None))
            root = _env0(asset.cfg.prim_path)
            self.by_prim[root] = len(self.entries) - 1
        self._articulations = [a for a in scene.articulations.values()]
        self._rigid = [a for a in scene.rigid_objects.values()]

    def _register(self, asset, names: list[str], first: int) -> None:
        from pxr import Usd

        import isaaclab.sim as sim_utils

        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(_env0(asset.cfg.prim_path))
        if not root:
            return
        wanted = {name: first + i for i, name in enumerate(names)}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            index = wanted.get(prim.GetName())
            if index is not None:
                self.by_prim.setdefault(str(prim.GetPath()), index)

    def poses(self) -> np.ndarray:
        """``(n, 7)`` xyz, xyzw in the env frame. Allocates one array per tick and copies nothing else."""
        import torch

        parts = []
        for asset in self._articulations:
            data = asset.data
            pos = _t(getattr(data, "body_link_pos_w", data.body_pos_w))[0]
            quat = _t(getattr(data, "body_link_quat_w", data.body_quat_w))[0]
            parts.append(torch.cat((pos, quat), dim=-1))
        for asset in self._rigid:
            parts.append(torch.cat((_t(asset.data.root_pos_w)[0], _t(asset.data.root_quat_w)[0]), dim=-1).unsqueeze(0))
        return torch.cat(parts, dim=0).detach().cpu().numpy() if parts else np.zeros((0, 7), np.float32)


def _env0(prim_path: str) -> str:
    path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    return re.sub(r"/env_(\[\^/\]\+|\.\*)", "/env_0", path)


def extract_scene(env, skip_names=("floor",), decimate_cell=0.008, hidden_bodies=(), colour=None) -> tuple[list[Node], MovingParts, dict]:
    """Walk ``/World/envs/env_0``: each visible render geometry, baked into its moving part's frame.

    Args:
        skip_names: Prim names left out of the headset's scene.
        decimate_cell: Vertex clustering cell in metres, so smaller detail is merged.
        hidden_bodies: Body names left out of the headset's scene, usually what the operator stands inside.
        colour: ``colour(prim, owner_name) -> rgb or None``, tried before the material. Linear RGB.
    """
    from pxr import Gf, Usd, UsdGeom

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    parts = MovingParts(env)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = stage.GetPrimAtPath("/World/envs/env_0")
    nodes: list[Node] = []
    stats = {"geoms": 0, "dynamic": 0, "skipped": 0}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        image = UsdGeom.Imageable(prim)

        lowered = path.lower()
        hidden_by_user = any(f"/{b}/" in path for b in hidden_bodies)
        if ((image.ComputeVisibility() == UsdGeom.Tokens.invisible and not hidden_by_user)
                or image.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy)
                or "/collision" in lowered or prim.GetName() in skip_names):
            stats["skipped"] += 1
            continue
        geometry = _geometry(prim)
        if geometry is None:
            stats["skipped"] += 1
            continue
        points, indices = geometry
        pose_index, owner = -1, None
        ancestor = prim
        while ancestor and ancestor.GetPath() != root.GetPath():
            hit = parts.by_prim.get(str(ancestor.GetPath()))
            if hit is not None:
                pose_index, owner = hit, ancestor
                break
            ancestor = ancestor.GetParent()
        if owner is not None and owner.GetName() in hidden_bodies:
            stats["skipped"] += 1
            continue
        world = np.array(cache.GetLocalToWorldTransform(prim), np.float64)
        if owner is not None:
            link = cache.GetLocalToWorldTransform(owner)
            link = Gf.Matrix4d(link).RemoveScaleShear()
            world = world @ np.array(link.GetInverse(), np.float64)
            stats["dynamic"] += 1
        homogeneous = np.c_[points, np.ones(len(points))]
        rgb = colour(prim, owner.GetName() if owner else "") if colour is not None else None
        rgb = rgb or material_colour(prim) or DEFAULT_RGB
        nodes.append(Node(pose_index, rgb, (homogeneous @ world)[:, :3].astype(np.float32),
                          indices.astype(np.uint32)))
        stats["geoms"] += 1
    return [decimate(n, decimate_cell) for n in merge_nodes(nodes)], parts, stats
