# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Busy box joint map: which USD joint each MJCF joint became, and its passive properties.

``tools/usd_export/convert_cell.py`` writes ``joint_map.json`` next to the converted cell
(schema: ``tools/usd_export/joint_map.schema.json``, version 1). The parts read here::

    {
      "source": {"init_key_equals_qpos0": true, ...},
      "usd": {"file": "factory.usda", "default_prim": "/factory", ...},
      "joints": {
        "o11_bb_drawer_joint": {
          "type": "slide",                          # hinge | slide (ball / free are skipped)
          "usd_joint_name": "o11_bb_drawer_joint",  # joint name as Isaac Lab reports it
          "articulation": "o10_busybox",            # root body of its articulation
          "range": [0.0, 0.3],                      # rad (hinge) or m (slide)
          "damping": 8.0, "frictionloss": 4.0, "armature": 0.01,   # SI
          "init": 0.0                               # qpos at the init key
        }
      },
      "articulations": {"o10_busybox": {"articulation_root_rel_path": "/Geometry/o10_busybox/PhysicsFixedJoint"}},
      "bodies": {"o43_crate": {"usd_name": "o43_crate", "usd_rel_path": "/Geometry/o43_crate",
                               "init_pos": [...], "init_quat_xyzw": [...]}},
      "rigid_objects": ["o43_crate", ...]
    }

A minimal map with only ``joints`` (and optionally ``usd`` as a file name) is accepted too.

Position conventions: the USD joint reads 0 at the authored pose, MuJoCo reads ``ref`` there
(``qpos0``). ``ref`` is the joint's own ``ref`` key when present, else its ``init`` when the
exporter reports ``init_key_equals_qpos0``, else 0. Source position = USD position + ``ref``.

This module does not import Isaac Lab, so it can be tested anywhere.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
"""This repository (``examples/isaaclab_g1/busybox`` is three levels down)."""

DEFAULT_CELL_DIR = REPO_ROOT / "assets" / "busybox" / "full"
"""The whole cell: the palette, the cart and the stool."""

DEFAULT_ARTICULATION = "o10_busybox"
"""Root body of the busy box articulation in the cell."""

JOINT_MAP_FILENAME = "joint_map.json"
USD_ENV_VAR = "TELESCENE_BUSYBOX_USD"
JOINT_MAP_ENV_VAR = "TELESCENE_BUSYBOX_JOINT_MAP"
USD_SUFFIXES = (".usd", ".usda", ".usdc")
PREFERRED_USD_STEMS = ("factory", "busybox_cell", "busybox", "scene")

_TYPE_ALIASES = {"hinge": "revolute", "revolute": "revolute", "slide": "prismatic", "prismatic": "prismatic"}


class JointMapError(ValueError):
    """The joint map exists but cannot be used."""


class NoMovableJointsError(JointMapError):
    """The selected articulation has no revolute or prismatic joint (e.g. a ball-jointed cable)."""


@dataclass(frozen=True)
class JointMapEntry:
    """One articulated busy box joint, in SI units."""

    name: str
    """MJCF joint name, as used by the station table."""
    usd_joint_name: str
    """Joint name as Isaac Lab reports it (the USD joint prim name)."""
    joint_type: str
    """``revolute`` or ``prismatic``."""
    lower: float | None = None
    """Lower limit, source convention [rad or m]; None when unlimited."""
    upper: float | None = None
    damping: float = 0.0
    frictionloss: float = 0.0
    armature: float = 0.0
    stiffness: float = 0.0
    ref: float = 0.0
    """Source position at which the USD joint reads zero [rad or m]."""
    init: float | None = None
    """Start position, source convention [rad or m]; None keeps the authored pose."""
    articulation: str | None = None
    usd_rel_path: str | None = None

    @property
    def usd_range(self) -> tuple[float, float] | None:
        """Limits in the USD convention (source minus ``ref``), or None when unlimited."""
        if self.lower is None or self.upper is None:
            return None
        return (self.lower - self.ref, self.upper - self.ref)

    @property
    def usd_init(self) -> float | None:
        """Start position in the USD convention, or None."""
        return None if self.init is None else self.init - self.ref

    @property
    def regex(self) -> str:
        """Exact-match regular expression for Isaac Lab's joint-name matching."""
        return re.escape(self.usd_joint_name)


@dataclass(frozen=True)
class BusyBoxJointMap:
    """The parsed joint map, restricted to one articulation of the cell."""

    path: Path
    joints: tuple[JointMapEntry, ...]
    usd: Path | None = None
    """USD file named by the map, resolved against the map's folder."""
    articulation: str | None = None
    """Root body name of the selected articulation (None for a minimal map)."""
    articulation_root: str | None = None
    """Prim with the ArticulationRootAPI, relative to the cell's default prim (``/Geometry/...``)."""
    skipped: tuple[str, ...] = ()
    """MJCF joints left out: other types (ball, free) or other articulations."""
    bodies: Mapping[str, str] = field(default_factory=dict)
    """MJCF body name -> USD body (prim) name."""
    body_paths: Mapping[str, str] = field(default_factory=dict)
    """MJCF body name -> prim path relative to the cell's default prim."""
    rigid_objects: tuple[str, ...] = ()
    """MJCF bodies that are free rigid bodies (the props)."""
    body_init: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    """MJCF body name -> layout pose in the cell frame: position [m], quaternion (x, y, z, w)."""

    def __len__(self) -> int:
        return len(self.joints)

    def __getitem__(self, name: str) -> JointMapEntry:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"joint {name!r} is not in the busy box joint map {self.path}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def usd_joint_names(self) -> tuple[str, ...]:
        return tuple(j.usd_joint_name for j in self.joints)

    def source_to_usd(self) -> dict[str, str]:
        """MJCF joint name -> USD joint name."""
        return {j.name: j.usd_joint_name for j in self.joints}

    def refs(self) -> dict[str, float]:
        """MJCF joint name -> ``ref`` for the joints with a non-zero ref."""
        return {j.name: j.ref for j in self.joints if j.ref != 0.0}

    @property
    def root_body_path(self) -> str | None:
        """Prim path of the articulation's root body, relative to the cell's default prim."""
        return None if self.articulation is None else self.body_paths.get(self.articulation)

    @property
    def root_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        """Layout pose of the root body in the cell frame: position [m], quaternion (x, y, z, w)."""
        return None if self.articulation is None else self.body_init.get(self.articulation)

    def view_paths(self) -> tuple[str, str | None]:
        """Where an Isaac Lab articulation view of this articulation attaches inside the cell.

        Returns:
            The root body path (relative to the cell's default prim), to append to the cell prim
            path, and the articulation-root prim relative to that body (None when the root body
            carries the ``ArticulationRootAPI`` itself, as floating-base roots do).

        Raises:
            JointMapError: If the map does not name the root body's prim or the articulation root.
        """
        body, root = self.root_body_path, self.articulation_root
        if body is None or root is None:
            raise JointMapError(
                f"{self.path}: articulation {self.articulation!r} needs 'bodies.<root>.usd_rel_path' and"
                " 'articulations.<root>.articulation_root_rel_path' to be placed in the cell"
            )
        body = "/" + body.strip("/")
        if root == body:
            return body, None
        if root.startswith(body + "/"):
            return body, root[len(body) :]
        raise JointMapError(f"{self.path}: articulation root {root} is not under its root body {body}")

    def actuator_params(self) -> dict[str, Any]:
        """Keyword arguments for one ``ImplicitActuatorCfg`` driving every mapped joint.

        Passive joints get zero stiffness, so the drive only adds viscous damping. MuJoCo's
        ``frictionloss`` acts both at rest and in motion, so it sets the static (``friction``) and
        the dynamic (``dynamic_friction``) Coulomb friction effort [N or N m].
        """
        return {
            "joint_names_expr": [j.regex for j in self.joints],
            "stiffness": {j.regex: j.stiffness for j in self.joints},
            "damping": {j.regex: j.damping for j in self.joints},
            "friction": {j.regex: j.frictionloss for j in self.joints},
            "dynamic_friction": {j.regex: j.frictionloss for j in self.joints},
            "armature": {j.regex: j.armature for j in self.joints},
        }

    def init_joint_pos(self) -> dict[str, float]:
        """``ArticulationCfg.InitialStateCfg.joint_pos`` in the USD convention."""
        init = {j.regex: j.usd_init for j in self.joints if j.usd_init not in (None, 0.0)}
        return init if init else {".*": 0.0}


def _float(entry: Mapping[str, Any], key: str, name: str, default: float | None) -> float | None:
    value = entry.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointMapError(f"joint {name!r}: '{key}' must be a number, got {value!r}")
    return float(value)


def _parse_entry(name: str, entry: Any, init_is_ref: bool) -> JointMapEntry | None:
    """One joint, or None for joint types without an actuator (ball, free)."""
    if not isinstance(entry, Mapping):
        raise JointMapError(f"joint {name!r}: expected an object, got {type(entry).__name__}")
    if entry.get("type") is None:
        raise JointMapError(f"joint {name!r}: missing 'type'")
    joint_type = _TYPE_ALIASES.get(str(entry["type"]).lower())
    if joint_type is None:
        return None
    usd_name = entry.get("usd_joint_name")
    if not usd_name or not isinstance(usd_name, str):
        raise JointMapError(f"joint {name!r}: missing 'usd_joint_name'")

    lower = upper = None
    rng = entry.get("range")
    if rng is not None:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise JointMapError(f"joint {name!r}: 'range' must be [lower, upper], got {rng!r}")
        lower, upper = float(rng[0]), float(rng[1])
        if upper < lower:
            raise JointMapError(f"joint {name!r}: range {list(rng)} runs backwards")

    init = _float(entry, "init", name, None)
    ref = _float(entry, "ref", name, init if (init_is_ref and init is not None) else 0.0)
    values = {key: _float(entry, key, name, 0.0) for key in ("damping", "frictionloss", "armature", "stiffness")}
    for key, value in values.items():
        if value < 0.0:
            raise JointMapError(f"joint {name!r}: '{key}' must be >= 0, got {value}")
    return JointMapEntry(
        name=name,
        usd_joint_name=usd_name.rstrip("/").rsplit("/", 1)[-1],
        joint_type=joint_type,
        lower=lower,
        upper=upper,
        ref=ref,
        init=init,
        articulation=entry.get("articulation"),
        usd_rel_path=entry.get("usd_rel_path"),
        **values,
    )


def load_joint_map(
    path: str | os.PathLike[str] | None = None, articulation: str | None = DEFAULT_ARTICULATION, allow_empty: bool = False
) -> BusyBoxJointMap:
    """Load and validate the busy box joint map.

    Args:
        path: The map file. Defaults to :func:`resolve_joint_map_path`.
        articulation: Root body of the articulation to keep joints of, when the map lists
            ``articulations``. None keeps every revolute and prismatic joint.
        allow_empty: Return a map with no joints instead of raising :class:`NoMovableJointsError`, for a
            cell with nothing to articulate (a lone stool, the bare floor) whose props are still wanted.

    Raises:
        FileNotFoundError: If the map does not exist; the message says how to produce or point to it.
        JointMapError: If the map is malformed.
    """
    path = resolve_joint_map_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"BusyBox joint map not found: {path}\n"
            f"  The USD exporter (tools/usd_export/convert_cell.py) writes it as {JOINT_MAP_FILENAME} next to the"
            f" cell USD in {DEFAULT_CELL_DIR}.\n"
            f"  Run the exporter, or set {JOINT_MAP_ENV_VAR}=/path/to/{JOINT_MAP_FILENAME}"
            f" (or {USD_ENV_VAR}=/path/to/cell.usda with the map next to it)."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JointMapError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(doc, Mapping) or not isinstance(doc.get("joints"), Mapping):
        raise JointMapError(f"{path}: expected an object with a 'joints' object keyed by MJCF joint name")

    articulations = doc.get("articulations")
    root = None
    if isinstance(articulations, Mapping) and articulation is not None:
        if articulation not in articulations:
            raise JointMapError(f"{path}: no articulation {articulation!r}; the map has {sorted(articulations)}")
        root = articulations[articulation].get("articulation_root_rel_path")
    else:
        articulation = None
        root = doc.get("articulation_root")
    if root:
        root = "/" + str(root).strip("/")

    init_is_ref = bool((doc.get("source") or {}).get("init_key_equals_qpos0", False))
    joints: list[JointMapEntry] = []
    skipped: list[str] = []
    for name, entry in doc["joints"].items():
        try:
            parsed = _parse_entry(str(name), entry, init_is_ref)
        except JointMapError as exc:
            raise JointMapError(f"{path}: {exc}") from exc
        if parsed is None or (articulation is not None and parsed.articulation != articulation):
            skipped.append(str(name))
        else:
            joints.append(parsed)
    if not joints and not allow_empty:
        where = f" in articulation {articulation!r}" if articulation else ""
        raise NoMovableJointsError(f"{path}: no revolute or prismatic joints{where}")
    for attr in ("name", "usd_joint_name"):
        values = [getattr(j, attr) for j in joints]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise JointMapError(f"{path}: duplicate {attr} {dupes}")

    usd = doc.get("usd")
    if isinstance(usd, Mapping):
        usd = usd.get("file")
    bodies = {k: v for k, v in (doc.get("bodies") or {}).items() if isinstance(v, Mapping)}
    body_init = {
        k: (tuple(float(x) for x in v["init_pos"]), tuple(float(x) for x in v["init_quat_xyzw"]))
        for k, v in bodies.items()
        if v.get("init_pos") is not None and v.get("init_quat_xyzw") is not None
    }
    return BusyBoxJointMap(
        path=path,
        joints=tuple(joints),
        usd=None if not usd else (path.parent / str(usd)).resolve(),
        articulation=articulation,
        articulation_root=root,
        skipped=tuple(skipped),
        bodies={k: v.get("usd_name") or k for k, v in bodies.items()},
        body_paths={k: v["usd_rel_path"] for k, v in bodies.items() if v.get("usd_rel_path")},
        rigid_objects=tuple(doc.get("rigid_objects") or ()),
        body_init=body_init,
    )


def articulation_names(path: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    """Root bodies of the articulations the map lists (empty for a minimal map).

    Raises:
        FileNotFoundError: If the map does not exist.
        JointMapError: If the map is not valid JSON.
    """
    path = resolve_joint_map_path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JointMapError(f"{path}: not valid JSON ({exc})") from exc
    articulations = doc.get("articulations") if isinstance(doc, Mapping) else None
    return tuple(articulations) if isinstance(articulations, Mapping) else ()


def resolve_joint_map_path(
    path: str | os.PathLike[str] | None = None, usd_path: str | os.PathLike[str] | None = None
) -> Path:
    """The joint map to load: ``path``, else ``$TELESCENE_BUSYBOX_JOINT_MAP``, else next to ``usd_path``, else
    next to ``$TELESCENE_BUSYBOX_USD``, else in :data:`DEFAULT_CELL_DIR`."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if os.environ.get(JOINT_MAP_ENV_VAR):
        return Path(os.environ[JOINT_MAP_ENV_VAR]).expanduser().resolve()
    usd_path = usd_path if usd_path is not None else os.environ.get(USD_ENV_VAR) or None
    if usd_path is not None:
        return Path(usd_path).expanduser().resolve().parent / JOINT_MAP_FILENAME
    return DEFAULT_CELL_DIR / JOINT_MAP_FILENAME


def resolve_usd_path(usd_path: str | os.PathLike[str] | None = None, joint_map: BusyBoxJointMap | None = None) -> Path:
    """The cell USD: ``usd_path``, else ``$TELESCENE_BUSYBOX_USD``, else the file the joint map names, else the
    single USD file in the map's folder (``factory.*`` / ``busybox_cell.*`` / ``busybox.*`` / ``scene.*``
    win a tie).

    Raises:
        FileNotFoundError: If no USD file can be found.
    """
    if usd_path is None and os.environ.get(USD_ENV_VAR):
        usd_path = os.environ[USD_ENV_VAR]
    if usd_path is None and joint_map is not None and joint_map.usd is not None:
        usd_path = joint_map.usd
    if usd_path is not None:
        path = Path(usd_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BusyBox cell USD not found: {path} (set {USD_ENV_VAR} to the converted cell)")
        return path
    folder = joint_map.path.parent if joint_map is not None else DEFAULT_CELL_DIR
    candidates = sorted(p for p in folder.glob("*") if p.suffix.lower() in USD_SUFFIXES) if folder.is_dir() else []
    if len(candidates) > 1:
        for stem in PREFERRED_USD_STEMS:
            preferred = [p for p in candidates if p.stem == stem]
            if preferred:
                candidates = preferred[:1]
                break
    if len(candidates) != 1:
        found = ", ".join(p.name for p in candidates) or "none"
        raise FileNotFoundError(
            f"BusyBox cell USD: expected one {'/'.join(USD_SUFFIXES)} file in {folder}, found {found}."
            f" Set {USD_ENV_VAR}=/path/to/cell.usda or add \"usd\" to {JOINT_MAP_FILENAME}."
        )
    return candidates[0].resolve()
