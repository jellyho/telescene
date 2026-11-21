"""The BusyBox scenes in ``assets/busybox``: every manifest entry has its USD and a joint map that loads, with no
Isaac Lab. The headset menu must take them all."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "examples" / "isaaclab_g1" / "busybox"))

import joint_map as jm  # noqa: E402

MANIFEST = json.loads((REPO / "assets" / "busybox" / "scenes.json").read_text())


@pytest.mark.parametrize("entry", MANIFEST["scenes"], ids=lambda e: e["name"])
def test_scene_files_and_joint_map(entry):
    usd = REPO / entry["usd"]
    assert usd.is_file()
    assert (usd.parent / "Payload").is_dir()
    path = jm.resolve_joint_map_path(None, usd)
    names = jm.articulation_names(path)
    assert set(entry.get("articulations", [])) <= set(names)
    for name in names:
        try:
            joint_map = jm.load_joint_map(path, articulation=name)
        except jm.NoMovableJointsError:
            continue
        assert joint_map.joints
    props = jm.load_joint_map(path, articulation=None, allow_empty=True).rigid_objects
    assert set(entry.get("rigid_objects", [])) <= set(props)


def test_menu_takes_every_scene():
    from telescene.wire import MAX_SCENES, pack_scene_list

    names = [e.get("label_en") or e["name"] for e in MANIFEST["scenes"]]
    assert len(names) <= MAX_SCENES
    pack_scene_list(names, 0, [e.get("group") or "other" for e in MANIFEST["scenes"]])


def test_no_local_paths_in_assets():
    for path in (REPO / "assets" / "busybox").rglob("*"):
        if path.suffix in (".json", ".usda"):
            text = path.read_text(errors="ignore")
            assert "C:\\" not in text and "/home/" not in text and "Users/" not in text, path
