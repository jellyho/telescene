# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Encoder/decoder input layouts of the released SONIC models.

A layout is the ordered list of observation names from a deploy ``observation_config.yaml``
(gear_sonic_deploy ``policy/release/*.yaml``; the HF model ships the same file). Names are resolved
with the same registry as the C++ deploy (``g1_deploy_onnx_ref.cpp:1708-1814``): concatenation follows
the listed order and observations not required by the active encoder mode are zero-filled
(``g1_deploy_onnx_ref.cpp:2022-2075``).

Built-in layouts (copied name by name from the YAML files at GR00T-WholeBodyControl 1fa5f7fa3966):

=================  ===============================================  ========  ========
name               file                                             encoder   decoder
=================  ===============================================  ========  ========
``release``        policy/release/observation_config.yaml           1762      994
``low_latency``    policy/release/observation_config_low_latency    1247      994
``v1_1``           policy/sonic_v1_1/observation_config.yaml        1751      994
``sonic_release``  policy/release/observation_config_sonic_release  1751      994
=================  ===============================================  ========  ========
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml

ENCODER_MODE_IDS = {"g1": 0, "teleop": 1, "smpl": 2}
"""Encoder mode ids (docs/source/tutorials/zmq.md "Encode Mode Logic")."""

NUM_SMPL_JOINTS = 24
NUM_SMPL_POSES = 21
SMPL_LOWER_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
SMPL_ELBOW_WRIST_POSES = (17, 18, 19, 20)


@dataclass(frozen=True)
class ObsSpec:
    """One registry entry.

    ``kind`` selects the gatherer. ``subset`` names a joint/body subset, ``variant`` the anchor-orientation
    mode (``base`` / ``heading`` / ``refheading``) and ``parts`` the members of a composite alias.
    """

    name: str
    kind: str
    per_frame: int = 0
    num_frames: int = 1
    step: int = 1
    subset: str | None = None
    variant: str = "base"
    parts: tuple[ObsSpec, ...] = ()

    @property
    def dim(self) -> int:
        if self.parts:
            return sum(p.dim for p in self.parts)
        return self.per_frame * self.num_frames


_MULTI = r"(?:_(\d+)frame_step(\d+))?"
_JOINT_SUBSETS = {None: 29, "lowerbody": 12, "wrists": 6}


def _frames(m: re.Match, first_group: int) -> tuple[int, int]:
    n, s = m.group(first_group), m.group(first_group + 1)
    return (int(n), int(s)) if n else (1, 1)


def resolve_obs(name: str, token_dim: int = 64) -> ObsSpec:
    """Resolve an observation name to its spec (raises ``KeyError`` for unknown names)."""
    if name == "token_state":
        return ObsSpec(name, "token", token_dim)
    if name in ("encoder_mode", "encoder_mode_4", "encoder_index"):
        return ObsSpec(name, "encoder_mode", 3 if name == "encoder_mode" else 4)

    # training-name aliases used by observation_config_sonic_release.yaml (g1_deploy_onnx_ref.cpp:1721-1728)
    aliases = {
        "command_multi_future_nonflat": (
            "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5",
        ),
        "motion_anchor_ori_b_mf_nonflat": ("motion_anchor_orientation_10frame_step5",),
        "command_multi_future_lower_body": (
            "motion_joint_positions_lowerbody_10frame_step5",
            "motion_joint_velocities_lowerbody_10frame_step5",
        ),
        "motion_anchor_ori_b": ("motion_anchor_orientation",),
        "smpl_joints_multi_future_local_nonflat": ("smpl_joints_10frame_step1",),
        "smpl_root_ori_b_multi_future": ("smpl_anchor_orientation_10frame_step1",),
        "joint_pos_multi_future_wrist_for_smpl": ("motion_joint_positions_wrists_10frame_step1",),
    }
    if name in aliases:
        return ObsSpec(name, "composite", parts=tuple(resolve_obs(p, token_dim) for p in aliases[name]))

    m = re.fullmatch(r"motion_joint_(positions|velocities)(?:_(lowerbody|wrists))?" + _MULTI, name)
    if m:
        n, s = _frames(m, 3)
        kind = "motion_joint_pos" if m.group(1) == "positions" else "motion_joint_vel"
        return ObsSpec(name, kind, _JOINT_SUBSETS[m.group(2)], n, s, subset=m.group(2))
    m = re.fullmatch(r"(?:motion_root_z_position|smpl_root_z)" + _MULTI, name)
    if m:
        n, s = _frames(m, 1)
        return ObsSpec(name, "motion_root_z", 1, n, s)
    m = re.fullmatch(r"(motion|smpl)_anchor_orientation(?:_(heading|refheading))?" + _MULTI, name)
    if m:
        n, s = _frames(m, 3)
        return ObsSpec(name, "anchor_orientation", 6, n, s, variant=m.group(2) or "base")
    m = re.fullmatch(r"smpl_joints(?:_(lower))?" + _MULTI, name)
    if m:
        n, s = _frames(m, 2)
        joints = len(SMPL_LOWER_JOINTS) if m.group(1) else NUM_SMPL_JOINTS
        return ObsSpec(name, "smpl_joints", 3 * joints, n, s, subset=m.group(1))
    m = re.fullmatch(r"smpl_(pose|elbow_wrist_poses)" + _MULTI, name)
    if m:
        n, s = _frames(m, 2)
        poses = NUM_SMPL_POSES if m.group(1) == "pose" else len(SMPL_ELBOW_WRIST_POSES)
        return ObsSpec(name, "smpl_pose", 3 * poses, n, s, subset=None if m.group(1) == "pose" else "elbow_wrist")
    if name in ("vr_3point_local_target", "vr_3point_local_target_compliant"):
        return ObsSpec(name, "vr_3point_pos", 9)
    if name == "vr_3point_local_orn_target":
        return ObsSpec(name, "vr_3point_orn", 12)
    if name == "vr_3point_compliance":
        return ObsSpec(name, "vr_3point_compliance", 3)
    if name == "heading_diff_robot_ref":
        return ObsSpec(name, "heading_diff", 6)

    proprio = r"(base_angular_velocity|body_joint_positions|body_joint_velocities|last_actions|gravity_dir)"
    m = re.fullmatch(r"(his_)?" + proprio + _MULTI, name)
    if m and (m.group(1) is not None) == (m.group(3) is not None):
        n, s = _frames(m, 3)
        per = 3 if m.group(2) in ("base_angular_velocity", "gravity_dir") else 29
        return ObsSpec(name, "proprio", per, n, s, subset=m.group(2))
    raise KeyError(f"Unknown SONIC observation name: {name!r}")


@dataclass(frozen=True)
class ObsSlot:
    spec: ObsSpec
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.spec.dim


@dataclass(frozen=True)
class EncoderMode:
    name: str
    mode_id: int
    required: tuple[str, ...]


@dataclass
class ObsLayout:
    """Resolved decoder and encoder layouts plus encoder modes."""

    decoder: list[ObsSlot]
    encoder: list[ObsSlot]
    modes: dict[int, EncoderMode]
    token_dim: int = 64
    name: str = ""
    _by_name: dict[str, ObsSlot] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._by_name = {s.spec.name: s for s in self.encoder}

    @property
    def decoder_dim(self) -> int:
        return self.decoder[-1].end if self.decoder else 0

    @property
    def encoder_dim(self) -> int:
        return self.encoder[-1].end if self.encoder else 0

    def encoder_slot(self, name: str) -> ObsSlot:
        return self._by_name[name]

    def required(self, mode_id: int) -> tuple[str, ...]:
        if not self.modes:
            return tuple(self._by_name)
        return self.modes[mode_id].required

    @property
    def history_length(self) -> int:
        """Proprio ring length needed by the decoder terms."""
        lengths = [(s.spec.num_frames - 1) * s.spec.step + 1 for s in self.decoder if s.spec.kind == "proprio"]
        return max(lengths, default=1)

    @classmethod
    def from_dict(cls, cfg: dict, name: str = "") -> ObsLayout:
        """Build from a parsed deploy ``observation_config.yaml``."""
        encoder_cfg = cfg.get("encoder") or {}
        token_dim = int(encoder_cfg.get("dimension", 64) or 64)
        decoder = _slots([o["name"] for o in cfg.get("observations", []) if o.get("enabled", True)], token_dim)
        encoder = _slots(
            [o["name"] for o in encoder_cfg.get("encoder_observations", []) if o.get("enabled", True)], token_dim
        )
        modes = {}
        for mode in encoder_cfg.get("encoder_modes", []) or []:
            modes[int(mode["mode_id"])] = EncoderMode(
                mode["name"], int(mode["mode_id"]), tuple(mode.get("required_observations", []))
            )
        names = {s.spec.name for s in encoder}
        for mode in modes.values():
            unknown = [r for r in mode.required if r not in names]
            if unknown:
                raise ValueError(f"Encoder mode '{mode.name}' requires observations not in the layout: {unknown}")
        return cls(decoder=decoder, encoder=encoder, modes=modes, token_dim=token_dim, name=name)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> ObsLayout:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f), name=os.fspath(path))

    @classmethod
    def builtin(cls, name: str = "release") -> ObsLayout:
        if name not in BUILTIN_CONFIGS:
            raise KeyError(f"Unknown built-in SONIC layout {name!r}; choose from {sorted(BUILTIN_CONFIGS)}")
        return cls.from_dict(BUILTIN_CONFIGS[name], name=name)


def _slots(names: list[str], token_dim: int) -> list[ObsSlot]:
    slots, offset = [], 0
    for n in names:
        spec = resolve_obs(n, token_dim)
        slots.append(ObsSlot(spec, offset))
        offset += spec.dim
    return slots


def _cfg(decoder: list[str], encoder: list[str], modes: dict[str, list[str]]) -> dict:
    return {
        "observations": [{"name": n, "enabled": True} for n in decoder],
        "encoder": {
            "dimension": 64,
            "encoder_observations": [{"name": n, "enabled": True} for n in encoder],
            "encoder_modes": [
                {"name": k, "mode_id": ENCODER_MODE_IDS[k], "required_observations": v} for k, v in modes.items()
            ],
        },
    }


_DECODER_10 = [
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
]

BUILTIN_CONFIGS: dict[str, dict] = {
    "release": _cfg(
        _DECODER_10,
        [
            "encoder_mode_4",
            "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5",
            "motion_root_z_position_10frame_step5",
            "motion_root_z_position",
            "motion_anchor_orientation",
            "motion_anchor_orientation_10frame_step5",
            "motion_joint_positions_lowerbody_10frame_step5",
            "motion_joint_velocities_lowerbody_10frame_step5",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "smpl_joints_10frame_step1",
            "smpl_anchor_orientation_10frame_step1",
            "motion_joint_positions_wrists_10frame_step1",
        ],
        {
            "g1": [
                "encoder_mode_4",
                "motion_joint_positions_10frame_step5",
                "motion_joint_velocities_10frame_step5",
                "motion_anchor_orientation_10frame_step5",
            ],
            "teleop": [
                "encoder_mode_4",
                "motion_joint_positions_lowerbody_10frame_step5",
                "motion_joint_velocities_lowerbody_10frame_step5",
                "vr_3point_local_target",
                "vr_3point_local_orn_target",
                "motion_anchor_orientation",
            ],
            "smpl": [
                "encoder_mode_4",
                "smpl_joints_10frame_step1",
                "smpl_anchor_orientation_10frame_step1",
                "motion_joint_positions_wrists_10frame_step1",
            ],
        },
    ),
    "low_latency": _cfg(
        _DECODER_10,
        [
            "encoder_mode_4",
            "motion_joint_positions_10frame_step1",
            "motion_joint_velocities_10frame_step1",
            "motion_anchor_orientation_10frame_step1",
            "motion_anchor_orientation",
            "motion_joint_positions_lowerbody_10frame_step1",
            "motion_joint_velocities_lowerbody_10frame_step1",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "smpl_joints_4frame_step1",
            "smpl_anchor_orientation_4frame_step1",
            "motion_joint_positions_wrists_4frame_step1",
        ],
        {
            "g1": [
                "encoder_mode_4",
                "motion_joint_positions_10frame_step1",
                "motion_joint_velocities_10frame_step1",
                "motion_anchor_orientation_10frame_step1",
            ],
            "teleop": [
                "encoder_mode_4",
                "motion_joint_positions_lowerbody_10frame_step1",
                "motion_joint_velocities_lowerbody_10frame_step1",
                "vr_3point_local_target",
                "vr_3point_local_orn_target",
                "motion_anchor_orientation",
            ],
            "smpl": [
                "encoder_mode_4",
                "smpl_joints_4frame_step1",
                "smpl_anchor_orientation_4frame_step1",
                "motion_joint_positions_wrists_4frame_step1",
            ],
        },
    ),
    "v1_1": _cfg(
        _DECODER_10,
        [
            "encoder_mode_4",
            "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5",
            "motion_anchor_orientation_heading_10frame_step5",
            "motion_anchor_orientation_heading",
            "motion_joint_positions_lowerbody_10frame_step5",
            "motion_joint_velocities_lowerbody_10frame_step5",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "smpl_joints_10frame_step1",
            "smpl_anchor_orientation_heading_10frame_step1",
            "motion_joint_positions_wrists_10frame_step1",
        ],
        {
            "g1": [
                "encoder_mode_4",
                "motion_joint_positions_10frame_step5",
                "motion_joint_velocities_10frame_step5",
                "motion_anchor_orientation_heading_10frame_step5",
            ],
            "teleop": [
                "encoder_mode_4",
                "motion_joint_positions_lowerbody_10frame_step5",
                "motion_joint_velocities_lowerbody_10frame_step5",
                "vr_3point_local_target",
                "vr_3point_local_orn_target",
                "motion_anchor_orientation_heading",
            ],
            "smpl": [
                "encoder_mode_4",
                "smpl_joints_10frame_step1",
                "smpl_anchor_orientation_heading_10frame_step1",
                "motion_joint_positions_wrists_10frame_step1",
            ],
        },
    ),
    "sonic_release": _cfg(
        _DECODER_10,
        [
            "encoder_index",
            "command_multi_future_nonflat",
            "motion_anchor_ori_b_mf_nonflat",
            "command_multi_future_lower_body",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "motion_anchor_ori_b",
            "smpl_joints_multi_future_local_nonflat",
            "smpl_root_ori_b_multi_future",
            "joint_pos_multi_future_wrist_for_smpl",
        ],
        {
            "g1": ["encoder_index", "command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"],
            "teleop": [
                "encoder_index",
                "command_multi_future_lower_body",
                "vr_3point_local_target",
                "vr_3point_local_orn_target",
                "motion_anchor_ori_b",
            ],
            "smpl": [
                "encoder_index",
                "smpl_joints_multi_future_local_nonflat",
                "smpl_root_ori_b_multi_future",
                "joint_pos_multi_future_wrist_for_smpl",
            ],
        },
    ),
}
