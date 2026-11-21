# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Configuration of the SONIC action term (Isaac Lab v3.0.0)."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from .core.vr3pt import HMD_TO_HEAD_WXYZ

if TYPE_CHECKING:
    from .action import SonicAction


@configclass
class SonicActionCfg(ActionTermCfg):
    """SONIC whole-body controller driving the 29 G1 body joints (hand joints are left to other terms).

    Action vector layout per :attr:`mode` (quaternions are Isaac Lab **xyzw**):

    * ``"planner"`` (4): ``[vx, vy, wz, hip_height]``, identical to ``AgileBasedLowerBodyActionCfg`` and the
      output of Isaac Teleop's ``LocomotionRootCmdRetargeter``.
    * ``"vr_3pt"`` (26): ``[vx, vy, wz, hip_height, vr_active,
      left_wrist(x, y, z, qx, qy, qz, qw), right_wrist(7), head(7)]``. Wrists are ``*_wrist_yaw_link`` target
      poses (what the stock ``Se3AbsRetargeter`` produces for Pink IK), head is the HMD pose from
      ``HeadSource.transformed(world_T_anchor)``, all in :attr:`pose_frame`. ``vr_active > 0.5`` selects the
      teleop encoder (planner legs + 3-point upper body); otherwise the planner drives the whole body.
      A rising edge of ``vr_active`` calibrates (see :attr:`vr_calibrate_head`).
    * ``"pose"`` (87): ``[vx, vy, wz, hip_height, pose_active, smpl_joints(24 x 3), smpl_root_quat(4),
      wrist_joint_pos(6)]``; ``pose_active > 0.5`` selects the SMPL encoder.
    """

    class_type: type[SonicAction] | str = "{DIR}.action:SonicAction"

    asset_name: str = "robot"

    onnx_dir: str = MISSING
    """Directory holding ``model_encoder.onnx``, ``model_decoder.onnx``, ``planner_sonic.onnx``
    (HF ``nvidia/GEAR-SONIC`` layout, or :func:`~telescene.sonic.core.dummy.make_dummy_sonic_dir`)."""

    observation_config: str | None = None
    """Built-in layout name (``release``, ``low_latency``, ``v1_1``, ``sonic_release``), a YAML path, or ``None``
    for ``<onnx_dir>/observation_config.yaml`` falling back to ``release``."""

    encoder_file: str = "model_encoder.onnx"
    decoder_file: str = "model_decoder.onnx"
    planner_file: str = "planner_sonic.onnx"

    mode: str = "planner"
    """``"planner"``, ``"vr_3pt"``, ``"pose"`` or ``"switch"``; fixes the action layout. ``"switch"`` carries both the
    3-point and the POSE inputs and takes the encoder mode each tick (108-D): locomotion 4, encoder mode 1 (0 planner,
    1 hybrid, 2 POSE), left wrist, right wrist and head 7 each, SMPL joints 72, root quaternion 4, wrists 6."""

    policy_dt: float = 0.02
    """SONIC control period [s]; must be an integer multiple of the env step (``sim.dt * decimation``)."""

    history_init: str = "zeros"
    """Decoder history padding after reset: ``"zeros"`` (released inference) or ``"repeat"`` (training)."""

    init_command_speed: float = -1.0
    """``target_vel`` of the initial IDLE plan (-1 as the deploy, 0 as sonic_policy.py)."""

    background_planner: bool = False
    """Run planner inference on a worker thread, so that a replan does not stall the control step. Needed for
    real-time teleop, where it costs about 70 ms on CPU. Off by default to keep runs deterministic."""

    background_planner_look_ahead: int = 5
    """Control ticks between the current frame and the plan's start when in the background. It must cover the
    inference time, or plans arrive late and fade in part way through."""

    # velocity command -> planner command
    linear_deadband: float = 0.05
    yaw_deadband: float = 0.05
    slow_walk_speed: tuple[float, float] = (0.1, 0.8)
    walk_speed: tuple[float, float] = (0.8, 1.5)
    allow_run: bool = False
    squat_hip_height: float | None = 0.65
    """``hip_height`` below this enters the planner's IDLE_SQUAT mode (planner height = hip_height, clamped to
    [0.1, 0.8]). ``None`` ignores the hip height channel."""

    # upper body reference in the g1 (planner) encoder mode
    upper_body_term: str | None = None
    """Name of another action term (e.g. the Pink IK term) whose processed joint targets replace the planner's
    17 upper-body joints in the g1 encoder input, as the deploy's PLANNER_FROZEN_UPPER mode does. Its targets
    are then overwritten on the robot by this term, so list this term **after** it in the actions cfg. Pink IK
    solves in ``apply_actions``, so the reference lags one env step (and the planner upper body is used until
    the IK term has run once)."""

    zero_body_effort_targets: bool = True
    """Write zero feed-forward efforts to the 29 body joints on every physics step. This cancels the gravity
    compensation that ``PinkInverseKinematicsActionCfg.enable_gravity_compensation`` (default ``True``) puts on
    the arm and waist joints SONIC drives; SONIC was trained with plain implicit PD. Needs this term listed after
    the IK term."""

    # VR 3-point
    pose_frame: str = "world"
    """Frame of the wrist/head poses: ``"world"`` (sim world relative to the env origin, as Pink IK reads the same
    ``Se3AbsRetargeter`` outputs; Isaac Teleop default) or ``"root"`` (already in the pelvis frame, e.g. with
    ``IsaacTeleopCfg.target_frame_prim_path`` set to the pelvis prim, which rebases *every* pipeline output and so
    does not suit a pipeline that also feeds Pink IK)."""
    vr_root_frame: str = "full"
    """``"full"`` pelvis frame (training definition) or ``"heading"`` (pelvis position + yaw only)."""
    vr_head_mode: str = "hmd"
    """``"hmd"``: torso orientation follows the calibrated HMD. ``"neutral"``: upright torso target."""
    vr_hmd_axes_wxyz: tuple[float, float, float, float] = HMD_TO_HEAD_WXYZ
    vr_calibrate_head: bool = True
    vr_calibrate_wrists: bool = False
    wrist_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    """Robot bodies the wrist targets refer to (used for wrist calibration)."""
