# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""PICO body -> SONIC POSE frame (core/pico_body.py) against a transcription of SONIC's own path."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

import sonic_reference_pico as ref
from telescene.sonic.core import pico_body as pb
from telescene.sonic.core.rotations import quat_mul, wxyz_to_xyzw, yaw_to_quat
from telescene.sonic.core.smpl import SmplStream

RY180_INV = R.from_euler("y", -180, degrees=True)


def _skeleton(joints55):
    return pb.SmplSkeleton.from_joints_info({"J": torch.from_numpy(joints55), "parents_list": ref.SMPLX_PARENTS})


def _body_from_globals(g):
    """PICO joint orientations (xyzw, OpenXR frame) whose SONIC-corrected globals are the rotations ``g``."""
    return (g * RY180_INV).as_quat()


def _random_body(n, seed, spread=0.6):
    """Bodies with plausible local rotations (a random root, joints up to ~``spread`` rad from their parent)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = [R.random(random_state=rng.integers(1 << 30))]
        for i in range(1, 24):
            local = R.from_rotvec(rng.normal(0.0, spread / 2, 3))
            g.append(g[ref.PARENT_INDICES[i]] * local)
        out.append(_body_from_globals(R.concatenate(g)))
    return np.stack(out)                                            # (n, 24, 4) xyzw


def _ours(body_xyzw, skeleton, dtype=torch.float64):
    q = torch.from_numpy(body_xyzw).to(dtype)
    return pb.body_to_smpl(torch.cat((q[..., 3:4], q[..., :3]), dim=-1), skeleton)


@pytest.fixture(scope="module")
def joints55():
    return ref.synthetic_joints55()


def test_matches_sonic_reference(joints55):
    body = _random_body(12, seed=1)
    got = _ours(body, _skeleton(joints55))
    for k in range(len(body)):
        want = ref.reference_frame(np.concatenate([np.zeros((24, 3)), body[k]], axis=1), joints55)
        np.testing.assert_allclose(got["smpl_joints"][k].numpy(), want["smpl_joints"], atol=1e-9)
        np.testing.assert_allclose(got["root_quat"][k].numpy(), want["root_quat"], atol=1e-9)   # same sign too
        np.testing.assert_allclose(got["smpl_pose"][k].numpy(), want["smpl_pose"], atol=1e-9)
        np.testing.assert_allclose(got["wrist_joint_pos"][k].numpy(), want["wrist_joint_pos"], atol=1e-7)


def test_float32_close_to_reference(joints55):
    body = _random_body(4, seed=2)
    got = _ours(body, _skeleton(joints55), dtype=torch.float32)
    want = ref.reference_frame(np.concatenate([np.zeros((24, 3)), body[0]], axis=1), joints55)
    np.testing.assert_allclose(got["smpl_joints"][0].numpy(), want["smpl_joints"], atol=1e-5)
    np.testing.assert_allclose(got["wrist_joint_pos"][0].numpy(), want["wrist_joint_pos"], atol=1e-4)


def test_rest_pose(joints55):
    """Corrected globals all identity: zero local pose, zero wrists, the rest skeleton turned Y-up -> Z-up."""
    body = _body_from_globals(R.identity(24))[None]
    sk = _skeleton(joints55)
    got = _ours(body, sk)
    np.testing.assert_allclose(got["smpl_pose"].numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(got["wrist_joint_pos"].numpy(), 0.0, atol=1e-12)
    rest = sk.rest.double().numpy()
    rx90 = R.from_euler("x", 90, degrees=True)
    joints = rest[0] + rx90.apply(rest - rest[0])
    root = rx90 * R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()         # xyzw (0.5, ...) is the same base rotation
    np.testing.assert_allclose(got["smpl_joints"][0].numpy(), root.inv().apply(joints), atol=1e-12)


def test_input_yaw_turns_only_the_root(joints55):
    """Turning the whole body about OpenXR's up axis (y) is a yaw of the Z-up root; local pose is unchanged."""
    body = _random_body(3, seed=3)
    sk = _skeleton(joints55)
    base = _ours(body, sk)
    yaw = 0.7
    turned = np.stack([(R.from_euler("y", yaw) * R.from_quat(b)).as_quat() for b in body])
    got = _ours(turned, sk)
    torch.testing.assert_close(got["smpl_pose"], base["smpl_pose"])
    torch.testing.assert_close(got["wrist_joint_pos"], base["wrist_joint_pos"])
    want_root = quat_mul(yaw_to_quat(torch.full((3,), yaw, dtype=torch.float64)), base["root_quat"])
    torch.testing.assert_close((got["root_quat"] * want_root).sum(-1).abs(), torch.ones(3, dtype=torch.float64))
    # As in SONIC the joints keep the rest pelvis offset J[0]; relative to joint 0 they do not move.
    rel = lambda f: f["smpl_joints"] - f["smpl_joints"][:, :1]  # noqa: E731
    torch.testing.assert_close(rel(got), rel(base))


def test_batch_equals_single(joints55):
    body = _random_body(5, seed=4)
    sk = _skeleton(joints55)
    batch = _ours(body, sk)
    for k in range(5):
        one = _ours(body[k : k + 1], sk)
        for name in batch:
            torch.testing.assert_close(batch[name][k : k + 1], one[name])


@pytest.mark.parametrize(
    "joint, axis, want",
    [(20, 0, (1, 0, 0, 0, 0, 0)), (20, 1, (0, 0, 1, 0, 0, 0)), (20, 2, (0, 0, 0, 0, 1, 0)),
     (21, 0, (0, -1, 0, 0, 0, 0)), (21, 1, (0, 0, 0, -1, 0, 0)), (21, 2, (0, 0, 0, 0, 0, 1)),
     (18, 1, (0, 0, 0, 0, 0, 0)), (18, 0, (1, 0, 0, 0, 0, 0)), (19, 2, (0, 0, 0, 0, 0, 1))],
)
def test_wrist_single_axis(joints55, joint, axis, want):
    """Order [l_roll, r_roll, l_pitch, r_pitch, l_yaw, r_yaw]; an elbow's twist about y does not reach the wrist."""
    angle = 0.4
    local = [R.identity()] * 24
    local[joint] = R.from_rotvec(np.eye(3)[axis] * angle)
    g = [local[0]]
    for i in range(1, 24):
        g.append(g[ref.PARENT_INDICES[i]] * local[i])
    got = _ours(_body_from_globals(R.concatenate(g))[None], _skeleton(joints55))
    np.testing.assert_allclose(got["wrist_joint_pos"][0].numpy(), angle * np.array(want), atol=1e-9)


def test_invalid_joints(joints55):
    sk = _skeleton(joints55)
    body = torch.from_numpy(_random_body(2, seed=5)).float()
    rt = pb.PicoBodyRetargeter(2, sk)

    valid = torch.ones(2, 24, dtype=torch.bool)
    valid[:, 0] = False                                              # pelvis never seen yet
    valid[1, 20] = False                                             # env 1: left wrist never seen
    first = rt(body, valid)
    assert not first.ok.any()
    assert torch.isfinite(first.smpl_joints).all()
    local_l_wrist = pb.aa_to_quat(first.smpl_pose[1, 19])
    torch.testing.assert_close(local_l_wrist, torch.tensor([1.0, 0.0, 0.0, 0.0]))   # parent's rotation copied

    second = rt(body, torch.ones(2, 24, dtype=torch.bool))
    assert second.ok.all()
    full = _ours(body.double().numpy(), sk, dtype=torch.float32)
    torch.testing.assert_close(second.smpl_joints, full["smpl_joints"], atol=1e-5, rtol=1e-5)

    garbage = body.clone()
    garbage[:, 5] = 0.0                                              # zero quaternion: treated as invalid
    garbage[:, 7] = torch.tensor([0.3, -0.2, 0.1, 0.9])
    held = rt(garbage, torch.tensor([[True] * 7 + [False] + [True] * 16] * 2))
    torch.testing.assert_close(held.smpl_joints, second.smpl_joints)  # both bad joints keep their last pose

    rt.reset(torch.tensor([0]))
    assert not rt.seen[0].any() and rt.seen[1].all()


def test_into_smpl_stream(joints55):
    sk = _skeleton(joints55)
    rt = pb.PicoBodyRetargeter(3, sk)
    stream = SmplStream(3, window=4)
    body = torch.from_numpy(_random_body(3, seed=6)).float()
    valid = torch.ones(3, 24, dtype=torch.bool)
    valid[2, 0] = False                                              # env 2 has no pelvis yet
    frame = rt(body, valid)
    stream.push(frame.smpl_joints, frame.root_quat, frame.wrist_joint_pos, frame.smpl_pose, mask=frame.ok)
    assert stream.started.tolist() == [True, True, False]
    torch.testing.assert_close(stream.motion.smpl_joints[0, -1], frame.smpl_joints[0])
    torch.testing.assert_close(stream.motion.root_quat[1, -1], frame.root_quat[1])


def test_helpers_against_scipy():
    rng = np.random.default_rng(7)
    rots = R.random(64, random_state=8)
    q = torch.from_numpy(rots.as_quat()[:, [3, 0, 1, 2]])
    e = rots.as_euler("XYZ")
    keep = np.abs(np.cos(e[:, 1])) > 1e-3                            # away from gimbal lock
    np.testing.assert_allclose(pb.quat_to_euler_xyz(q).numpy()[keep], e[keep], atol=1e-9)
    aa = torch.from_numpy(rots.as_rotvec())
    np.testing.assert_allclose(pb.quat_to_aa(q).numpy(), rots.as_rotvec(), atol=1e-9)
    np.testing.assert_allclose(np.abs((pb.aa_to_quat(aa) * q).sum(-1).numpy()), 1.0, atol=1e-12)
    small = torch.from_numpy(rng.normal(0, 1e-10, (4, 3)))
    assert torch.isfinite(pb.quat_to_aa(pb.aa_to_quat(small))).all()
    assert torch.isfinite(pb.swing_about(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), 1)).all()   # SONIC gives NaN here
    torch.testing.assert_close(wxyz_to_xyzw(torch.tensor([[1.0, 0.0, 0.0, 0.0]])), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_skeleton_checks(tmp_path):
    bad = ref.synthetic_joints55()
    with pytest.raises(ValueError):
        pb.SmplSkeleton.from_joints_info({"J": torch.from_numpy(bad[:24]), "parents_list": ref.SMPLX_PARENTS})
    pointer = tmp_path / "human_joints_info.pkl"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 4302\n")
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        pb.load_smpl_skeleton(pointer)
    with pytest.raises(FileNotFoundError, match="not found"):
        pb.load_smpl_skeleton(tmp_path / "missing.pkl")


def _real_skeleton_or_skip():
    path = pb.find_human_joints_info()
    if not path.is_file() or path.read_bytes()[:40].startswith(b"version https://git-lfs"):
        pytest.skip(f"SONIC rest skeleton not fetched at {path}")
    return path


def test_real_skeleton_loads_weights_only():
    path = _real_skeleton_or_skip()
    sk = pb.load_smpl_skeleton(path)
    info = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(sk.rest.shape) == (24, 3)
    torch.testing.assert_close(sk.rest[:22], info["J"][:22])
    torch.testing.assert_close(sk.rest[22:], info["J"][[39, 54]])
    assert list(info["parents_list"]) == ref.SMPLX_PARENTS


def test_real_skeleton_matches_reference():
    path = _real_skeleton_or_skip()
    joints55 = torch.load(path, map_location="cpu", weights_only=True)["J"].double().numpy()
    body = _random_body(4, seed=9)
    got = _ours(body, pb.load_smpl_skeleton(path))
    for k in range(4):
        want = ref.reference_frame(np.concatenate([np.zeros((24, 3)), body[k]], axis=1), joints55)
        np.testing.assert_allclose(got["smpl_joints"][k].numpy(), want["smpl_joints"], atol=1e-6)
        np.testing.assert_allclose(got["wrist_joint_pos"][k].numpy(), want["wrist_joint_pos"], atol=1e-7)
