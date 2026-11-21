# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""SONIC as an Isaac Lab action term, running the ONNX models inside the env step on simulated time.

Structure follows Isaac Lab v3.0.0 ``AgileBasedLowerBodyAction``
(``isaaclab_tasks/contrib/locomanip_pick_place/mdp/actions.py``): the policy runs in :meth:`process_actions`
(once per env step) and :meth:`apply_actions` writes the held joint position targets every physics step.

Poses in the ``"world"`` pose frame are read relative to each env origin (``scene.env_origins``), the convention
of the stock ``PinkInverseKinematicsAction`` that consumes the same ``Se3AbsRetargeter`` outputs
(``isaaclab/envs/mdp/actions/pink_task_space_actions.py`` ``_get_base_link_frame_transform``). For the single
teleop env the origin is zero, so this equals the sim world frame of Isaac Teleop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from .core import rotations as rot
from .core.commands import VelocityCommandCfg, VelocityCommandMapper
from .core.joint_order import DEFAULT_ANGLES_POLICY, POLICY_JOINT_NAMES, UPPER_BODY_POLICY_INDEX
from .core.policy import RobotState, SonicInput, SonicPolicyCore
from .core.vr3pt import Vr3ptBuilder, Vr3ptCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import SonicActionCfg

ACTION_DIMS = {"planner": 4, "vr_3pt": 26, "pose": 87, "switch": 108}
LOCO = slice(0, 4)
ACTIVE = 4
VR_LEFT, VR_RIGHT, VR_HEAD = slice(5, 12), slice(12, 19), slice(19, 26)
SMPL_JOINTS, SMPL_ROOT_QUAT, SMPL_WRISTS = slice(5, 77), slice(77, 81), slice(81, 87)
# "switch" keeps both inputs, so that the encoder can change between ticks: ACTIVE holds the encoder mode.
SWITCH_SMPL_JOINTS, SWITCH_SMPL_ROOT_QUAT, SWITCH_SMPL_WRISTS = slice(26, 98), slice(98, 102), slice(102, 108)


def _pose_xyzw_to_wxyz(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat((pose[..., 0:3], rot.xyzw_to_wxyz(pose[..., 3:7])), dim=-1)


class SonicAction(ActionTerm):
    """SONIC planner + encoder + decoder producing PD targets for the 29 G1 body joints."""

    cfg: SonicActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: SonicActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.mode not in ACTION_DIMS:
            raise ValueError(f"SonicActionCfg.mode must be one of {sorted(ACTION_DIMS)}, got {cfg.mode!r}")
        if cfg.pose_frame not in ("world", "root"):
            raise ValueError(f"pose_frame must be 'world' or 'root', got {cfg.pose_frame!r}")
        self._env = env

        joint_ids, joint_names = self._asset.find_joints(list(POLICY_JOINT_NAMES), preserve_order=True, as_proxy=True)
        if list(joint_names) != list(POLICY_JOINT_NAMES):
            raise ValueError(f"SONIC body joints not found in order on '{cfg.asset_name}': {joint_names}")
        self._joint_ids = joint_ids.torch
        self._joint_names = list(joint_names)

        ratio = cfg.policy_dt / env.step_dt
        self._policy_every = max(1, round(ratio))
        if abs(self._policy_every * env.step_dt - cfg.policy_dt) > 1e-6:
            raise ValueError(
                f"SONIC needs a {cfg.policy_dt} s control period; env.step_dt = {env.step_dt} s is not a divisor"
            )
        self._step_count = 0

        self._core = SonicPolicyCore.from_dir(
            cfg.onnx_dir,
            self.num_envs,
            self.device,
            observation_config=cfg.observation_config,
            encoder_file=cfg.encoder_file,
            decoder_file=cfg.decoder_file,
            planner_file=cfg.planner_file,
            history_init=cfg.history_init,
            init_command_speed=cfg.init_command_speed,
            planner_kwargs=(
                {"background": True, "look_ahead": cfg.background_planner_look_ahead}
                if cfg.background_planner
                else None
            ),
        )
        self._mapper = VelocityCommandMapper(
            self.num_envs,
            self.device,
            VelocityCommandCfg(
                linear_deadband=cfg.linear_deadband,
                yaw_deadband=cfg.yaw_deadband,
                slow_walk_speed=tuple(cfg.slow_walk_speed),
                walk_speed=tuple(cfg.walk_speed),
                allow_run=cfg.allow_run,
                squat_hip_height=cfg.squat_hip_height,
            ),
        )
        self._vr = None
        self._wrist_body_ids: list[int] = []
        if cfg.mode in ("vr_3pt", "switch"):
            self._vr = Vr3ptBuilder(
                self.num_envs,
                self.device,
                Vr3ptCfg(
                    root_frame=cfg.vr_root_frame,
                    head_mode=cfg.vr_head_mode,
                    hmd_axes_wxyz=tuple(cfg.vr_hmd_axes_wxyz),
                    calibrate_head=cfg.vr_calibrate_head,
                    calibrate_wrists=cfg.vr_calibrate_wrists,
                ),
            )
            self._wrist_body_ids, _ = self._asset.find_bodies(list(cfg.wrist_body_names), preserve_order=True)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        default = torch.as_tensor(DEFAULT_ANGLES_POLICY, dtype=torch.float32, device=self.device)
        self._default_targets = default.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._processed_actions = self._default_targets.clone()
        self._zero_efforts = torch.zeros_like(self._processed_actions)
        self._active_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._encoder_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._upper_cols: torch.Tensor | None = None
        self._upper_width = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return ACTION_DIMS[self.cfg.mode]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """PD position targets of the 29 body joints, SONIC policy order (see :attr:`joint_names`)."""
        return self._processed_actions

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def tokens(self) -> torch.Tensor:
        """Latest 64-D SONIC tokens ``(num_envs, 64)``."""
        return self._core.tokens

    @property
    def policy_action(self) -> torch.Tensor:
        """Latest raw decoder output ``(num_envs, 29)``, policy order."""
        return self._core.last_action

    @property
    def encoder_mode(self) -> torch.Tensor:
        """Encoder mode used at the latest policy step: 0 g1, 1 teleop, 2 smpl."""
        return self._encoder_mode

    @property
    def core(self) -> SonicPolicyCore:
        return self._core

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self._step_count % self._policy_every == 0:
            self._policy_step()
        self._step_count += 1

    def apply_actions(self):
        command = self._asset.actuators.target_command
        command.set_position_index(value=self._processed_actions, joint_ids=self._joint_ids)
        if self.cfg.zero_body_effort_targets:
            command.set_effort_index(value=self._zero_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = self._resolve_env_ids(env_ids)
        self._core.reset(ids)
        idx = slice(None) if ids is None else ids
        self._mapper.reset(idx)
        if self._vr is not None:
            self._vr.reset(idx)
        self._raw_actions[idx] = 0.0
        self._processed_actions[idx] = self._default_targets[idx]
        self._active_prev[idx] = False
        self._encoder_mode[idx] = 0

    """
    Internals.
    """

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor | None:
        """``None`` (all envs) or a long index tensor.

        Isaac Lab v3.0.0 passes ``slice(None)`` from ``ManagerBasedEnv.reset`` and an int32 tensor from
        ``ManagerBasedRLEnv.step`` (``manager_based_env.py:403-440``, ``manager_based_rl_env.py:242``).
        """
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return None
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            return ids.nonzero(as_tuple=False).reshape(-1)
        return ids.to(torch.long).reshape(-1)

    def _robot_state(self) -> RobotState:
        data = self._asset.data
        return RobotState(
            joint_pos=data.joint_pos.torch[:, self._joint_ids],
            joint_vel=data.joint_vel.torch[:, self._joint_ids],
            base_quat=rot.xyzw_to_wxyz(data.root_quat_w.torch),
            base_ang_vel=data.root_ang_vel_b.torch,
        )

    def _root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pelvis pose in :attr:`SonicActionCfg.pose_frame` (wxyz)."""
        if self.cfg.pose_frame == "root":
            pos = torch.zeros(self.num_envs, 3, device=self.device)
            quat = torch.zeros(self.num_envs, 4, device=self.device)
            quat[:, 0] = 1.0
            return pos, quat
        data = self._asset.data
        return data.root_pos_w.torch - self._env.scene.env_origins, rot.xyzw_to_wxyz(data.root_quat_w.torch)

    def _robot_wrist_pose(self) -> torch.Tensor:
        """Measured wrist-link poses ``(E, 2, 7)`` wxyz in :attr:`SonicActionCfg.pose_frame`."""
        data = self._asset.data
        pos = data.body_pos_w.torch[:, self._wrist_body_ids]
        quat = rot.xyzw_to_wxyz(data.body_quat_w.torch[:, self._wrist_body_ids])
        if self.cfg.pose_frame == "root":
            root_p = data.root_pos_w.torch.unsqueeze(1).expand_as(pos)
            root_q = rot.xyzw_to_wxyz(data.root_quat_w.torch).unsqueeze(1).expand_as(quat)
            pos = rot.quat_apply_inverse(root_q, pos - root_p)
            quat = rot.quat_mul(rot.quat_conjugate(root_q), quat)
        else:
            pos = pos - self._env.scene.env_origins.unsqueeze(1)
        return torch.cat((pos, quat), dim=-1)

    def _upper_body(self) -> torch.Tensor | None:
        """Upper-body joint targets ``(E, 17)`` from :attr:`SonicActionCfg.upper_body_term`, or ``None``.

        ``PinkInverseKinematicsAction`` fills ``processed_actions`` in ``apply_actions`` (one row of
        ``_controlled_joint_names`` per joint); until its first physics step the buffer still has the raw-action
        width, so ``None`` (planner upper body) is returned then. Because this term runs in ``process_actions``,
        the targets read are the IK solution of the previous env step.
        """
        term = self._env.action_manager.get_term(self.cfg.upper_body_term)
        if self._upper_cols is None:
            names = getattr(term, "_controlled_joint_names", None) or getattr(term, "_joint_names", None)
            if names is None:
                raise AttributeError(f"Cannot read joint names of action term '{self.cfg.upper_body_term}'")
            wanted = [POLICY_JOINT_NAMES[i] for i in UPPER_BODY_POLICY_INDEX]
            missing = [n for n in wanted if n not in names]
            if missing:
                raise KeyError(f"Action term '{self.cfg.upper_body_term}' does not drive {missing}")
            self._upper_cols = torch.tensor([list(names).index(n) for n in wanted], device=self.device)
            self._upper_width = len(names)
        targets = term.processed_actions
        if targets.shape[-1] != self._upper_width:
            return None
        return targets[:, self._upper_cols].to(device=self.device, dtype=torch.float32)

    def _policy_step(self):
        cfg = self.cfg
        actions = self._raw_actions
        state = self._robot_state()
        planner_cmd = self._mapper(actions[:, LOCO], cfg.policy_dt)
        mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        inputs = SonicInput(encoder_mode=mode, planner=planner_cmd)
        active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if cfg.mode in ("vr_3pt", "switch"):
            if cfg.mode == "vr_3pt":
                active = actions[:, ACTIVE] > 0.5
                mode = torch.where(active, 1, mode)
            else:
                mode = actions[:, ACTIVE].round().long().clamp(0, 2)
                active = mode == 1
            wrists = torch.stack(
                (_pose_xyzw_to_wxyz(actions[:, VR_LEFT]), _pose_xyzw_to_wxyz(actions[:, VR_RIGHT])), dim=1
            )
            head = _pose_xyzw_to_wxyz(actions[:, VR_HEAD])
            root_pos, root_quat = self._root_pose()
            rising = (active & ~self._active_prev).nonzero(as_tuple=False).squeeze(-1)
            if rising.numel() > 0:
                robot_wrists = self._robot_wrist_pose() if cfg.vr_calibrate_wrists else None
                self._vr.calibrate(rising, wrists, head, root_pos, root_quat, robot_wrists)
            inputs.vr_position, inputs.vr_orientation = self._vr(wrists, head, root_pos, root_quat)
        elif cfg.mode == "pose":
            active = actions[:, ACTIVE] > 0.5
            mode = torch.where(active, 2, mode)
            inputs.smpl_joints = actions[:, SMPL_JOINTS].reshape(self.num_envs, 24, 3)
            inputs.smpl_root_quat = rot.quat_normalize(rot.xyzw_to_wxyz(actions[:, SMPL_ROOT_QUAT]))
            inputs.smpl_wrist_joint_pos = actions[:, SMPL_WRISTS]
        if cfg.mode == "switch":
            inputs.smpl_joints = actions[:, SWITCH_SMPL_JOINTS].reshape(self.num_envs, 24, 3)
            inputs.smpl_root_quat = rot.quat_normalize(rot.xyzw_to_wxyz(actions[:, SWITCH_SMPL_ROOT_QUAT]))
            inputs.smpl_wrist_joint_pos = actions[:, SWITCH_SMPL_WRISTS]

        if cfg.upper_body_term is not None:
            upper = self._upper_body()
            if upper is not None:
                inputs.upper_body_joint_pos = upper
                inputs.upper_body_mask = mode == 0
        inputs.encoder_mode = mode
        out = self._core.step(state, inputs)
        self._processed_actions[:] = out.joint_pos_target
        self._active_prev = active
        self._encoder_mode = mode
