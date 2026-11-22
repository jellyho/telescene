# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""GigaBrain-WBC-0.5 (open-gigaai/giga-brain-wbc, Apache-2.0): a reference encoder and a Transformer action decoder.

The encoder turns ten reference frames into a 64-number token. The decoder reads the robot (93 numbers, the spec's
``obs_dict`` block) and the token, and attends over its last 32 ticks through a KV cache that its graph takes in and
hands back each tick. This follows the authors' G1 deployment (``deploy/unitree_g1/cpp``), so the encoder input is:

    mode one-hot (4) | per frame k = 0..9:
        joint positions, absolute, the network's order          10 x 29
        root rotation relative to the robot now, 6D (rows of R[:, :2])  10 x 6
        root position relative to the reference at frame -1, in that frame   10 x 3
        root position relative to the robot now, in the robot's frame        10 x 3
        world down in the reference root's frame                 10 x 3

Frames come from GMR through ``PoseStream`` as for every spec tracker: with ``lookahead`` 9 the window ends at the
newest pose, which is what the authors do with a streamed reference (their window is the newest ten frames).
"""

from __future__ import annotations

import time

import numpy as np

from .onnx_tracker import OnnxTrackerController, PoseStream, TrackerSpec, onnx_session
from .supermimic import NUM_JOINTS, quat_conjugate, quat_matrix, quat_mul

WINDOW = 10
MODES = 4
DOWN = np.array([0.0, 0.0, -1.0])


def encoder_input(frames: list, base, robot_quat: np.ndarray, robot_pos: np.ndarray | None, mode: int = 0) -> np.ndarray:
    """The encoder's 444 numbers. ``frames``: reference frames 0..9 (``onnx_tracker.Frame``, joints in the network's
    order); ``base``: frame -1; the robot's root now, wxyz and world position (None: zeros, as without localisation)."""
    one_hot = np.zeros(MODES)
    one_hot[mode] = 1.0
    robot_inv = quat_conjugate(robot_quat / np.linalg.norm(robot_quat))
    robot_r = quat_matrix(robot_quat)
    base_r = quat_matrix(base.root_quat)
    joints = np.concatenate([f.joint_pos for f in frames])
    rot = np.concatenate([quat_matrix(quat_mul(robot_inv, f.root_quat))[:, :2].reshape(-1) for f in frames])
    pos_ref = np.concatenate([base_r.T @ (f.root_pos - base.root_pos) for f in frames])
    if robot_pos is None:
        pos_robot = np.zeros(3 * len(frames))
    else:
        pos_robot = np.concatenate([robot_r.T @ (f.root_pos - robot_pos) for f in frames])
    gravity = np.concatenate([quat_matrix(f.root_quat).T @ DOWN for f in frames])
    return np.concatenate((one_hot, joints, rot, pos_ref, pos_robot, gravity))


class GigaBrainPolicy:
    """Both graphs on the CPU, and the decoder's KV cache between calls. ``__call__`` takes ``{"obs_dict": 93 numbers,
    "encoder": 444}`` and returns the raw action."""

    def __init__(self, spec: TrackerSpec, threads: int = 2):
        if spec.encoder is None:
            raise ValueError(f"{spec.slug}: a gigabrain spec needs an encoder")
        self.encoder = onnx_session(spec.encoder, spec.encoder_sha256, spec.slug, threads)
        self.decoder = onnx_session(spec.onnx, spec.sha256, spec.slug, threads)
        shapes = {i.name: [d if isinstance(d, int) else 1 for d in i.shape] for i in self.decoder.get_inputs()}
        want = {"obs_dict", "past_key_values", "cache_valid", "abs_position"}
        if set(shapes) != want:
            raise ValueError(f"{spec.onnx.name} wants inputs {sorted(shapes)}, a gigabrain decoder takes {sorted(want)}")
        self.shapes = shapes
        self.token_size = int(np.prod(self.encoder.get_outputs()[0].shape[1:]))
        self.encoder_size = int(np.prod(self.encoder.get_inputs()[0].shape[1:]))
        self.encoder_input = self.encoder.get_inputs()[0].name
        meta = self.decoder.get_modelmeta().custom_metadata_map
        self.rope_max = int(meta.get("rope_max_seq_len", 0))
        self.rope_margin = 1024
        self.encoder_ms: list[float] = []
        self.reset()

    def reset(self) -> None:
        """Forget the past: what the authors' controller does at start and when the RoPE position nears its limit."""
        self.cache = {k: np.zeros(self.shapes[k], np.float32) for k in ("past_key_values", "cache_valid", "abs_position")}

    def token(self, encoder_obs: np.ndarray) -> np.ndarray:
        if encoder_obs.size != self.encoder_size:
            raise ValueError(f"encoder input: built {encoder_obs.size} numbers, the graph wants {self.encoder_size}")
        t0 = time.perf_counter()
        out = self.encoder.run(None, {self.encoder_input: encoder_obs.astype(np.float32).reshape(1, -1)})[0]
        self.encoder_ms.append((time.perf_counter() - t0) * 1000)
        return np.asarray(out, np.float64).reshape(-1)

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        obs = np.concatenate((inputs["obs_dict"], self.token(inputs["encoder"])))
        if obs.size != int(np.prod(self.shapes["obs_dict"])):
            raise ValueError(f"decoder input: built {obs.size} numbers, the graph wants {self.shapes['obs_dict']}")
        if self.rope_max and float(self.cache["abs_position"].max()) >= self.rope_max - self.rope_margin:
            self.reset()
        feed = {"obs_dict": obs.astype(np.float32).reshape(self.shapes["obs_dict"]), **self.cache}
        action, keys, valid, position = self.decoder.run(
            ["action", "present_key_values", "present_cache_valid", "present_abs_position"], feed)
        self.cache = {"past_key_values": keys, "cache_valid": valid, "abs_position": position}
        return np.asarray(action, np.float64).reshape(-1)[:NUM_JOINTS]


class GigaBrainController(OnnxTrackerController):
    """``OnnxTrackerController`` with GigaBrain's encoder input and its stateful decoder. ``policy`` may replace
    ``GigaBrainPolicy`` (tests): it gets the same two inputs."""

    def __init__(self, state, spec: TrackerSpec, retarget=None, emit=None, policy=None, **options):
        super().__init__(state, spec, retarget, emit, policy=policy if policy is not None else GigaBrainPolicy(spec),
                         **options)
        # frame -1 is the encoder's base frame
        self.poses = PoseStream(self.poses.lookahead, max(1, spec.past), spec.control_dt, self.kinematics)
        self.reset()

    def reset(self) -> None:
        super().reset()
        if hasattr(getattr(self, "policy", None), "reset"):
            self.policy.reset()

    def inputs(self) -> dict[str, np.ndarray]:
        out = super().inputs()
        order = self.spec.order
        now = self._ticks[-1].state
        pos = None if now.pos is None else np.asarray(now.pos, np.float64)
        frames = [self.poses.frame(k, order) for k in range(WINDOW)]
        out["encoder"] = encoder_input(frames, self.poses.frame(-1, order), np.asarray(now.quat, np.float64), pos,
                                       self.spec.encoder_mode)
        return out
