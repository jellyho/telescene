# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""A telescene mirror that shows the SMPL SONIC receives, over the tracked joints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from telescene.mirror import MirrorView

from .core.pico_body import SMPL_PARENTS, THUMB_PARENTS, PicoBodyRetargeter, load_smpl_skeleton
from .core.rotations import quat_apply
from .inputs import body_arrays

NAMES = ("pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
         "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
         "l_hand", "r_hand")


def smpl_converter(skeleton):
    """Tracked body to the joints SONIC is given, put back in the world and in OpenXR axes."""
    retargeter = PicoBodyRetargeter(1, skeleton)

    def convert(body: np.ndarray) -> np.ndarray:
        quat, valid = body_arrays(body)
        with torch.no_grad():
            frame = retargeter(torch.as_tensor(quat.copy()).unsqueeze(0), torch.as_tensor(valid.copy()).unsqueeze(0))
            joints = quat_apply(frame.root_quat.unsqueeze(1).expand(1, 24, 4), frame.smpl_joints)[0]
        x, y, z = joints.numpy().T  # Z up to OpenXR: the inverse of smpl_root_ytoz_up
        return np.stack((x, z, -y), axis=-1)

    return convert


def make_mirror_view(skeleton_path: str | Path, distance: float = 1.0) -> MirrorView:
    skeleton = load_smpl_skeleton(Path(skeleton_path))
    parents = list(SMPL_PARENTS) + list(THUMB_PARENTS)
    rest = skeleton.rest.numpy()
    lengths = [0.0 if p < 0 else float(np.linalg.norm(rest[i] - rest[p])) for i, p in enumerate(parents)]
    return MirrorView(parents, lengths, convert=smpl_converter(skeleton), distance=distance, names=NAMES)
