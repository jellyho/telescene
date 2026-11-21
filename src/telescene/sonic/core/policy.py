# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched SONIC controller: planner + encoder + decoder, one call per 50 Hz control tick.

Order inside :meth:`SonicPolicyCore.step` follows IsaacLab-demos ``sonic_policy.py:131-170`` and the deploy
control loop (g1_deploy_onnx_ref.cpp ``Control`` 3833-4000): planner tick -> history append -> encoder ->
decoder -> blend new plan -> advance frame. Joint targets are ``default + action_scale * action``
(g1_deploy_onnx_ref.cpp:3154-3163). No wall clock is used; one call is one tick of simulated time.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from . import rotations as rot
from .commands import PlannerCommand
from .history import HistoryBuffer
from .joint_order import ACTION_SCALE_POLICY, DEFAULT_ANGLES_POLICY, NUM_BODY_JOINTS
from .obs_config import ENCODER_MODE_IDS, ObsLayout
from .observations import PROPRIO_DIM, SMPL_MODE, EncoderContext, build_decoder_obs, build_encoder_obs, proprio_sample
from .onnx_model import DECODER_INPUTS, DECODER_OUTPUTS, ENCODER_INPUTS, ENCODER_OUTPUTS, OnnxModel
from .planner import SonicPlanner
from .checkpoint import DECODER_FILE, ENCODER_FILE, PLANNER_FILE, check_sonic_dir
from .smpl import SmplStream

CONTROL_DT = 0.02
OBS_CONFIG_FILE = "observation_config.yaml"


@dataclass
class RobotState:
    """Measured state at the start of a tick (policy joint order, Isaac Lab sim units).

    Attributes:
        joint_pos: ``(E, 29)`` absolute joint positions [rad].
        joint_vel: ``(E, 29)`` joint velocities [rad/s].
        base_quat: ``(E, 4)`` pelvis orientation in world, **wxyz**.
        base_ang_vel: ``(E, 3)`` pelvis angular velocity in the pelvis frame [rad/s].
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_quat: torch.Tensor
    base_ang_vel: torch.Tensor


@dataclass
class SonicInput:
    """Per-tick commands.

    Attributes:
        encoder_mode: ``(E,)`` long: 0 = g1 (planner whole body), 1 = teleop (planner legs + VR 3-point),
            2 = smpl (POSE).
        planner: Planner command (used in modes 0 and 1).
        vr_position / vr_orientation: VR 3-point targets ``(E, 9)`` / ``(E, 12)`` (mode 1).
        upper_body_joint_pos / upper_body_joint_vel / upper_body_mask: optional external upper-body joint
            reference replacing the planner's 17 upper-body joints in mode 0.
        smpl_joints / smpl_root_quat / smpl_wrist_joint_pos: one streamed SMPL frame (mode 2).
    """

    encoder_mode: torch.Tensor
    planner: PlannerCommand
    vr_position: torch.Tensor | None = None
    vr_orientation: torch.Tensor | None = None
    vr_compliance: torch.Tensor | None = None
    upper_body_joint_pos: torch.Tensor | None = None
    upper_body_joint_vel: torch.Tensor | None = None
    upper_body_mask: torch.Tensor | None = None
    smpl_joints: torch.Tensor | None = None
    smpl_root_quat: torch.Tensor | None = None
    smpl_wrist_joint_pos: torch.Tensor | None = None


@dataclass
class SonicOutput:
    joint_pos_target: torch.Tensor
    """``(E, 29)`` PD position targets, policy order [rad]."""
    action: torch.Tensor
    """``(E, 29)`` raw decoder output, policy order."""
    tokens: torch.Tensor
    """``(E, token_dim)`` encoder tokens."""


def resolve_layout(onnx_dir: str | None, observation_config: str | None) -> ObsLayout:
    """``observation_config``: a built-in name, a YAML path, or ``None`` (``<onnx_dir>/observation_config.yaml``
    if present, else the built-in ``release`` layout)."""
    if observation_config is None:
        candidate = os.path.join(onnx_dir, OBS_CONFIG_FILE) if onnx_dir else None
        if candidate and os.path.isfile(candidate):
            return ObsLayout.from_yaml(candidate)
        return ObsLayout.builtin("release")
    if os.path.isfile(observation_config):
        return ObsLayout.from_yaml(observation_config)
    return ObsLayout.builtin(observation_config)


class SonicPolicyCore:
    """Batched SONIC inference state machine.

    Args:
        encoder: ``model_encoder.onnx`` session.
        decoder: ``model_decoder.onnx`` session.
        planner: Planner (owns the planner session and the reference motion).
        layout: Encoder/decoder input layout.
        num_envs: Batch size.
        device: Torch device of all state tensors.
        history_init: ``"zeros"`` (released references) or ``"repeat"`` (training CircularBuffer).
        init_command_speed: ``target_vel`` of the planner's initial IDLE plan (-1 deploy, 0 sonic_policy.py).
    """

    def __init__(
        self,
        encoder: OnnxModel,
        decoder: OnnxModel,
        planner: SonicPlanner,
        layout: ObsLayout,
        num_envs: int,
        device: str | torch.device = "cpu",
        history_init: str = "zeros",
        init_command_speed: float = -1.0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.planner = planner
        self.layout = layout
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.init_command_speed = init_command_speed
        self._check_widths()
        self.history = HistoryBuffer(num_envs, layout.history_length, PROPRIO_DIM, device, init=history_init)
        smpl_window = max(
            (s.spec.num_frames for s in layout.encoder if s.spec.kind == "smpl_joints"), default=1
        )
        self.smpl = SmplStream(num_envs, smpl_window, device)
        self.last_action = torch.zeros(num_envs, NUM_BODY_JOINTS, device=device)
        self.tokens = torch.zeros(num_envs, layout.token_dim, device=device)
        self.planner_heading = torch.zeros(num_envs, 4, device=device)
        self.planner_heading[:, 0] = 1.0
        self.smpl_heading = self.planner_heading.clone()
        self.prev_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.default = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32, device=device)
        self.action_scale = torch.as_tensor(ACTION_SCALE_POLICY, dtype=torch.float32, device=device)
        self.last_encoder_obs: torch.Tensor | None = None
        self.last_decoder_obs: torch.Tensor | None = None

    @classmethod
    def from_dir(
        cls,
        onnx_dir: str,
        num_envs: int,
        device: str | torch.device = "cpu",
        observation_config: str | None = None,
        providers: list | None = None,
        encoder_file: str = ENCODER_FILE,
        decoder_file: str = DECODER_FILE,
        planner_file: str = PLANNER_FILE,
        planner_kwargs: dict | None = None,
        **kwargs,
    ) -> SonicPolicyCore:
        """Load the three released ONNX files from ``onnx_dir``."""
        check_sonic_dir(onnx_dir, (encoder_file, decoder_file, planner_file))
        layout = resolve_layout(onnx_dir, observation_config)
        encoder = OnnxModel(os.path.join(onnx_dir, encoder_file), ENCODER_INPUTS, ENCODER_OUTPUTS, providers, device)
        decoder = OnnxModel(os.path.join(onnx_dir, decoder_file), DECODER_INPUTS, DECODER_OUTPUTS, providers, device)
        planner = SonicPlanner.from_file(
            os.path.join(onnx_dir, planner_file), num_envs, device, providers, **(planner_kwargs or {})
        )
        return cls(encoder, decoder, planner, layout, num_envs, device, **kwargs)

    def _check_widths(self):
        enc = self.encoder.input_shape("obs_dict")[-1]
        dec = self.decoder.input_shape("obs_dict")[-1]
        tok = self.encoder.output_shape("encoded_tokens")[-1]
        for got, want, what in (
            (enc, self.layout.encoder_dim, "encoder input"),
            (dec, self.layout.decoder_dim, "decoder input"),
            (tok, self.layout.token_dim, "token"),
        ):
            if isinstance(got, int) and got != want:
                raise ValueError(
                    f"{what} width {got} does not match observation layout '{self.layout.name}' ({want}); "
                    "pass the observation_config.yaml that ships with the checkpoint"
                )

    @property
    def mode_ids(self) -> dict[str, int]:
        return dict(ENCODER_MODE_IDS)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Forget all state; the planner re-initialises from the measured pose on the next :meth:`step`."""
        idx = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.planner.reset(idx)
        self.history.reset(idx)
        self.smpl.reset(idx)
        self.last_action[idx] = 0.0
        self.tokens[idx] = 0.0
        self.prev_mode[idx] = 0

    def _initialize(self, state: RobotState):
        ids = (~self.planner.initialized).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        cmd = PlannerCommand.idle(ids.numel(), self.device, speed=self.init_command_speed)
        self.planner.initialize(ids, state.joint_pos[ids], cmd)
        motion = self.planner.motion
        # first tick reads frame 1, as sonic_policy.py:77 and the deploy's first CurrentFrameAdvancement
        motion.frame[ids] = torch.minimum(torch.ones_like(ids), motion.length[ids] - 1)
        ref = motion.current(motion.root_quat)[ids]
        self.planner_heading[ids] = rot.quat_mul(rot.heading_quat(state.base_quat[ids]), rot.heading_quat_inv(ref))

    def _update_smpl(self, state: RobotState, inputs: SonicInput, mode: torch.Tensor):
        entering = (mode == SMPL_MODE) & (self.prev_mode != SMPL_MODE)
        if bool(entering.any()):
            self.smpl.reset(entering.nonzero(as_tuple=False).squeeze(-1))
        if inputs.smpl_joints is None or inputs.smpl_root_quat is None:
            return
        first = (mode == SMPL_MODE) & ~self.smpl.started
        self.smpl.push(
            inputs.smpl_joints, inputs.smpl_root_quat, inputs.smpl_wrist_joint_pos, mask=mode == SMPL_MODE
        )
        if bool(first.any()):
            ref = self.smpl.motion.root_quat[first, 0]
            self.smpl_heading[first] = rot.quat_mul(
                rot.heading_quat(state.base_quat[first]), rot.heading_quat_inv(ref)
            )

    @torch.no_grad()
    def step(self, state: RobotState, inputs: SonicInput) -> SonicOutput:
        """Run one 50 Hz control tick for all environments."""
        mode = inputs.encoder_mode.to(self.device).long()
        self._initialize(state)
        self.planner.plan(inputs.planner)
        self._update_smpl(state, inputs, mode)

        self.history.append(
            proprio_sample(state.joint_pos, state.joint_vel, state.base_quat, state.base_ang_vel, self.last_action)
        )
        ctx = EncoderContext(
            mode=mode,
            base_quat=state.base_quat,
            planner_motion=self.planner.motion,
            planner_heading=self.planner_heading,
            smpl_motion=self.smpl.motion,
            smpl_heading=self.smpl_heading,
            vr_position=inputs.vr_position,
            vr_orientation=inputs.vr_orientation,
            vr_compliance=inputs.vr_compliance,
            upper_body_joint_pos=inputs.upper_body_joint_pos,
            upper_body_joint_vel=inputs.upper_body_joint_vel,
            upper_body_mask=inputs.upper_body_mask,
        )
        encoder_obs = build_encoder_obs(self.layout, ctx)
        tokens = self.encoder.run_torch({"obs_dict": encoder_obs}, device=self.device)["encoded_tokens"].float()
        decoder_obs = build_decoder_obs(self.layout, tokens, self.history.data)
        action = self.decoder.run_torch({"obs_dict": decoder_obs}, device=self.device)["action"].float()

        self.planner.commit()
        self.planner.advance(inputs.planner, state.joint_pos)
        self.prev_mode = mode
        self.last_action = action
        self.tokens = tokens
        self.last_encoder_obs = encoder_obs
        self.last_decoder_obs = decoder_obs
        return SonicOutput(self.default + self.action_scale * action, action, tokens)
