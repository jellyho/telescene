"""Reading colours from USD. Skipped if pxr is not installed."""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from telescene.isaac.scene import material_colour  # noqa: E402


def _bound(stage, name, set_colour=True):
    material = UsdShade.Material.Define(stage, f"/M/{name}")
    shader = UsdShade.Shader.Define(stage, f"/M/{name}/S")
    shader.CreateIdAttr("UsdPreviewSurface")
    colour = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if set_colour:
        colour.Set(Gf.Vec3f(0.1, 0.2, 0.3))
    prim = UsdGeom.Cube.Define(stage, f"/G/{name}").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return prim


def test_reads_the_diffuse_colour():
    stage = Usd.Stage.CreateInMemory()
    assert material_colour(_bound(stage, "a")) == pytest.approx((0.1, 0.2, 0.3))


def test_no_material_or_no_value_gives_none():
    stage = Usd.Stage.CreateInMemory()
    assert material_colour(UsdGeom.Cube.Define(stage, "/G/none").GetPrim()) is None
    assert material_colour(_bound(stage, "unset", set_colour=False)) is None
