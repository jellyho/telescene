# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Joint orders, index tables and gains against the upstream name lists and formulas."""

import random

import numpy as np
import pytest
import torch

from telescene.sonic.core import joint_order as jo

import sonic_reference_policy as ref


def test_name_lists_match_upstream():
    assert list(jo.MUJOCO_JOINT_NAMES) == ref.JOINT_NAMES
    assert list(jo.POLICY_JOINT_NAMES) == ref.POLICY_JOINT_NAMES
    assert len(set(jo.MUJOCO_JOINT_NAMES)) == 29
    assert set(jo.MUJOCO_JOINT_NAMES) == set(jo.POLICY_JOINT_NAMES)


def test_index_tables_follow_names():
    for m, name in enumerate(jo.MUJOCO_JOINT_NAMES):
        assert jo.POLICY_JOINT_NAMES[jo.ISAACLAB_TO_MUJOCO[m]] == name
    for p, name in enumerate(jo.POLICY_JOINT_NAMES):
        assert jo.MUJOCO_JOINT_NAMES[jo.MUJOCO_TO_ISAACLAB[p]] == name
    np.testing.assert_array_equal(jo.MUJOCO_TO_ISAACLAB, ref.POLICY_ORDER)
    np.testing.assert_array_equal(jo.ISAACLAB_TO_MUJOCO[jo.MUJOCO_TO_ISAACLAB], np.arange(29))


def test_subsets():
    lower = [jo.POLICY_JOINT_NAMES[i] for i in jo.LOWER_BODY_MUJOCO_ORDER_IN_POLICY_INDEX]
    assert lower == list(jo.MUJOCO_JOINT_NAMES[:12])
    assert sorted(jo.LOWER_BODY_POLICY_INDEX) == sorted(jo.LOWER_BODY_MUJOCO_ORDER_IN_POLICY_INDEX)
    wrists = [jo.POLICY_JOINT_NAMES[i] for i in jo.WRIST_POLICY_INDEX]
    assert wrists == [f"{s}_wrist_{a}_joint" for a in ("roll", "pitch", "yaw") for s in ("left", "right")]
    upper = {jo.POLICY_JOINT_NAMES[i] for i in jo.UPPER_BODY_POLICY_INDEX}
    assert upper == set(jo.MUJOCO_JOINT_NAMES[12:])
    assert list(jo.UPPER_BODY_POLICY_INDEX) == sorted(jo.UPPER_BODY_POLICY_INDEX)


def test_gains_match_isaaclab_demo_parameters():
    np.testing.assert_allclose(jo.ACTION_SCALE_MUJOCO, ref.ACTION_SCALE, rtol=1e-6)
    np.testing.assert_allclose(jo.KP_MUJOCO, ref.STIFFNESS, rtol=1e-6)
    np.testing.assert_allclose(jo.KD_MUJOCO, ref.DAMPING, rtol=1e-6)
    np.testing.assert_allclose(jo.DEFAULT_ANGLES_MUJOCO, ref.DEFAULT_POS, atol=1e-7)
    # deploy: kp of a 5020 joint = armature * (20 pi)^2, ankles doubled
    assert jo.KP_MUJOCO[19] == pytest.approx(0.003609725 * (20 * 3.1415926535) ** 2)
    assert jo.KP_MUJOCO[4] == pytest.approx(2 * 0.003609725 * (20 * 3.1415926535) ** 2)
    assert jo.ARMATURE_MUJOCO[4] == pytest.approx(2 * 0.003609725)
    g = jo.gains_by_name()
    assert g["left_knee_joint"]["effort_limit"] == 139.0
    assert g["right_wrist_yaw_joint"]["velocity_limit"] == 22.0


def test_reorder_round_trip_numpy_and_torch():
    x = np.random.default_rng(0).standard_normal((5, 29))
    np.testing.assert_array_equal(jo.policy_to_mujoco(jo.mujoco_to_policy(x)), x)
    t = torch.from_numpy(x)
    torch.testing.assert_close(jo.mujoco_to_policy(jo.policy_to_mujoco(t)), t)
    np.testing.assert_array_equal(jo.mujoco_to_policy(x), x[:, ref.POLICY_ORDER])


def _sim_names(seed: int = 3) -> list[str]:
    hands = [f"{s}_hand_{f}_joint" for s in ("left", "right") for f in
             ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")]
    names = list(jo.MUJOCO_JOINT_NAMES) + hands
    random.Random(seed).shuffle(names)
    return names


def test_joint_map_round_trip_with_hands():
    names = _sim_names()
    jm = jo.JointMap(names)
    sim = torch.randn(4, len(names))
    pol = jm.sim_to_policy(sim)
    for p, name in enumerate(jo.POLICY_JOINT_NAMES):
        torch.testing.assert_close(pol[:, p], sim[:, names.index(name)])
    back = jm.policy_to_sim(pol, out=torch.full_like(sim, 7.0))
    for i, name in enumerate(names):
        expected = sim[:, i] if name in jo.POLICY_JOINT_NAMES else torch.full((4,), 7.0)
        torch.testing.assert_close(back[:, i], expected)
    np.testing.assert_array_equal(jm.sim_ids_mujoco.numpy(), [names.index(n) for n in jo.MUJOCO_JOINT_NAMES])


def test_joint_map_missing_joint():
    names = _sim_names()
    names.remove("waist_roll_joint")
    with pytest.raises(KeyError, match="waist_roll_joint"):
        jo.JointMap(names)
