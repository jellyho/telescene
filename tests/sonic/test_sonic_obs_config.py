# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Encoder/decoder layouts against the deploy YAML files and the Isaac Lab demo offsets."""

import os

import pytest

from telescene.sonic.core.obs_config import BUILTIN_CONFIGS, ObsLayout, resolve_obs

from sonic_testutil import SONIC_REPO

YAMLS = {
    "release": "gear_sonic_deploy/policy/release/observation_config.yaml",
    "low_latency": "gear_sonic_deploy/policy/release/observation_config_low_latency.yaml",
    "v1_1": "gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml",
    "sonic_release": "gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml",
}
DIMS = {"release": 1762, "low_latency": 1247, "v1_1": 1751, "sonic_release": 1751}


@pytest.mark.parametrize("name", sorted(BUILTIN_CONFIGS))
def test_builtin_dims(name):
    layout = ObsLayout.builtin(name)
    assert layout.encoder_dim == DIMS[name]
    assert layout.decoder_dim == 994
    assert layout.token_dim == 64
    assert layout.history_length == 10
    assert sorted(layout.modes) == [0, 1, 2]


@pytest.mark.parametrize("name", sorted(YAMLS))
def test_builtin_matches_upstream_yaml(name):
    path = os.path.join(SONIC_REPO, YAMLS[name])
    if not os.path.isfile(path):
        pytest.skip(f"third_party SONIC repo not present: {path}")
    upstream = ObsLayout.from_yaml(path)
    builtin = ObsLayout.builtin(name)
    assert [(s.spec.name, s.offset, s.spec.dim) for s in upstream.encoder] == [
        (s.spec.name, s.offset, s.spec.dim) for s in builtin.encoder
    ]
    assert [(s.spec.name, s.offset) for s in upstream.decoder] == [(s.spec.name, s.offset) for s in builtin.decoder]
    assert {k: v.required for k, v in upstream.modes.items()} == {k: v.required for k, v in builtin.modes.items()}
    assert upstream.encoder_dim == DIMS[name]


def test_release_offsets_match_isaaclab_sonic_policy():
    """sonic_policy.py:151-155 writes the g1 terms at 4:294, 294:584 and 601:661."""
    layout = ObsLayout.builtin("release")
    for name, (lo, hi) in {
        "encoder_mode_4": (0, 4),
        "motion_joint_positions_10frame_step5": (4, 294),
        "motion_joint_velocities_10frame_step5": (294, 584),
        "motion_anchor_orientation_10frame_step5": (601, 661),
    }.items():
        slot = layout.encoder_slot(name)
        assert (slot.offset, slot.end) == (lo, hi)
    decoder = [(s.spec.name, s.offset, s.end) for s in layout.decoder]
    assert decoder[0] == ("token_state", 0, 64)
    assert decoder[-1] == ("his_gravity_dir_10frame_step1", 964, 994)


def test_resolve_names():
    assert resolve_obs("motion_joint_positions_lowerbody_10frame_step5").dim == 120
    assert resolve_obs("motion_joint_positions_wrists_4frame_step1").dim == 24
    assert resolve_obs("smpl_joints_4frame_step1").dim == 288
    assert resolve_obs("smpl_joints_lower_10frame_step1").dim == 270
    assert resolve_obs("smpl_elbow_wrist_poses_10frame_step1").dim == 120
    assert resolve_obs("command_multi_future_nonflat").dim == 580
    assert resolve_obs("motion_anchor_orientation_refheading_10frame_step1").variant == "refheading"
    assert resolve_obs("his_body_joint_positions_4frame_step1").dim == 116
    assert resolve_obs("body_joint_positions").dim == 29
    assert resolve_obs("encoder_mode").dim == 3
    with pytest.raises(KeyError):
        resolve_obs("motion_joint_torques_10frame_step5")
    with pytest.raises(KeyError):
        resolve_obs("his_body_joint_positions")


def test_mode_must_reference_layout():
    cfg = {
        "observations": [{"name": "token_state"}],
        "encoder": {
            "encoder_observations": [{"name": "encoder_mode_4"}],
            "encoder_modes": [{"name": "g1", "mode_id": 0, "required_observations": ["vr_3point_local_target"]}],
        },
    }
    with pytest.raises(ValueError, match="vr_3point_local_target"):
        ObsLayout.from_dict(cfg)
