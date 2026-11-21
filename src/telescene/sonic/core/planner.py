# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched, synchronous (sim-time) port of SONIC's locomotion planner loop.

The deploy runs the planner in its own 10 Hz thread on the wall clock (g1_deploy_onnx_ref.cpp ``Planner``
3595-3790). Here it runs inside the control step every ``planner_period`` ticks, plans from the motion at
``frame + look_ahead`` and the plan is blended in at the end of the same tick, as IsaacLab-demos
``sonic_policy.py`` does (``step`` 131-170). Replan rules and intervals are the deploy's.

With ``background=True`` inference leaves the control step, as on the deploy: the released planner takes
about 70 ms on one CPU thread, which is 3.5 control ticks. ``look_ahead`` must then cover that latency, and the
plan is blended in at whichever tick it finishes, aligned by the ticks that have passed.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch

from .commands import BOXING_REPLAN_MODES, STATIC_MODES, LocomotionMode, PlannerCommand
from .joint_order import DEFAULT_ROOT_HEIGHT, LOWER_BODY_POLICY_INDEX, policy_to_mujoco
from .motion import MotionBuffer, blend_into, resample_planner_output, sample_context
from .onnx_model import PLANNER_INPUTS, PLANNER_OUTPUTS, OnnxModel

ALLOWED_PRED_NUM_TOKENS = (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
"""Default token mask (localmotion_kplanner_tensorrt.hpp:96-102, sonic_policy.py:70)."""

# Idle re-adaptation thresholds (g1_deploy_onnx_ref.cpp:244-246).
ADAPT_TRIGGER = 0.10
ADAPT_STOP = 0.05
RECOVER_TRIGGER = 0.045
_IDLE, _ADAPTING, _RECOVERING = 0, 1, 2


class SonicPlanner:
    """Planner session plus per-environment reference motion.

    Args:
        model: Loaded ``planner_sonic.onnx``.
        num_envs: Batch size.
        device: Torch device of the motion buffers.
        capacity: Motion buffer length in 50 Hz frames (a plan is at most 64 planner frames = 106 frames).
        look_ahead: Control ticks between the current frame and the first context frame (``PlannerConfig``
            ``motion_look_ahead_steps`` = 2).
        planner_period: Control ticks per planner tick (50 Hz / 10 Hz = 5).
        replan_ticks: Periodic replan interval in planner ticks: run 0.1 s, crawl 0.2 s, boxing and default
            1.0 s (g1_deploy_onnx_ref.cpp:231-234).
        random_seed: Planner ``random_seed`` input.
        idle_readapt: Enable the idle re-adaptation state machine (g1_deploy_onnx_ref.cpp:3336-3400).
        background: Run inference on a worker thread so a replan never stalls the control step. For real-time
            use; off by default so that a run stays deterministic. At most one plan is in flight.
    """

    def __init__(
        self,
        model: OnnxModel,
        num_envs: int,
        device: str | torch.device = "cpu",
        capacity: int = 128,
        look_ahead: int = 2,
        planner_period: int = 5,
        replan_ticks: dict[str, int] | None = None,
        random_seed: int = 1234,
        default_height: float = DEFAULT_ROOT_HEIGHT,
        allowed_pred_num_tokens: tuple[int, ...] = ALLOWED_PRED_NUM_TOKENS,
        idle_readapt: bool = True,
        background: bool = False,
    ):
        self.model = model
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.look_ahead = look_ahead
        self.planner_period = planner_period
        self.replan_ticks = {"run": 1, "crawl": 2, "boxing": 10, "default": 10, **(replan_ticks or {})}
        self.random_seed = random_seed
        self.default_height = default_height
        self.allowed_pred_num_tokens = torch.tensor(allowed_pred_num_tokens, dtype=torch.long)
        self.idle_readapt = idle_readapt
        self.motion = MotionBuffer(num_envs, capacity, device)
        self.last = PlannerCommand.idle(num_envs, device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.replan_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.replanned = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.idle_state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.idle_stored = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.idle_target = torch.zeros(num_envs, len(LOWER_BODY_POLICY_INDEX), device=device)
        self._lower = torch.as_tensor(LOWER_BODY_POLICY_INDEX, device=device)
        self._pending: tuple[torch.Tensor, tuple] | None = None
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="sonic_planner") if background else None
        self._future: tuple[torch.Tensor, Future, int] | None = None
        self._clock = 0
        self.late_plans = 0
        """Background plans that finished after their first frame had already passed."""

    @classmethod
    def from_file(cls, path, num_envs: int, device="cpu", providers=None, **kwargs) -> SonicPlanner:
        return cls(OnnxModel(path, PLANNER_INPUTS, PLANNER_OUTPUTS, providers, device), num_envs, device, **kwargs)

    def infer(self, context: torch.Tensor, command: PlannerCommand) -> tuple:
        """Run the planner on ``(n, 4, 36)`` MuJoCo-qpos context; returns the resampled 50 Hz plan."""
        n = context.shape[0]
        feeds = {
            "context_mujoco_qpos": context.float(),
            "target_vel": command.speed.float(),
            "mode": command.mode.long(),
            "movement_direction": command.movement_direction.float(),
            "facing_direction": command.facing_direction.float(),
            "random_seed": torch.full((n,), self.random_seed, dtype=torch.long),
            "height": command.height.float(),
            "has_specific_target": torch.zeros(n, 1, dtype=torch.long),
            "specific_target_positions": torch.zeros(n, 4, 3),
            "specific_target_headings": torch.zeros(n, 4),
            "allowed_pred_num_tokens": self.allowed_pred_num_tokens.unsqueeze(0).expand(n, -1),
        }
        feeds = {k: v for k, v in feeds.items() if self.model.has_input(k)}
        out = self.model.run_torch(feeds, device=self.device)
        qpos = out["mujoco_qpos"].float()
        count = out["num_pred_frames"].reshape(n).long()
        if bool((count < 2).any()) or bool((count > qpos.shape[1]).any()):
            raise RuntimeError(f"planner returned an invalid frame count {count.tolist()} (max {qpos.shape[1]})")
        if not torch.isfinite(qpos).all():
            raise RuntimeError("planner returned non-finite qpos")
        return resample_planner_output(qpos, count)

    def initialize(self, env_ids: torch.Tensor, joint_pos: torch.Tensor, command: PlannerCommand | None = None):
        """Start a fresh plan from the measured joints (policy order, absolute rad), zero yaw, default height.

        Mirrors ``LocalMotionPlannerBase::Initialize`` / ``InitializeContext`` (localmotion_kplanner.hpp:332, 591):
        all four context frames equal ``[0, 0, default_height, 1, 0, 0, 0, joints]``.
        """
        n = env_ids.numel()
        if command is None:
            command = PlannerCommand.idle(n, self.device)
        context = torch.zeros(n, 4, 36, device=self.device)
        context[..., 2] = self.default_height
        context[..., 3] = 1.0
        context[..., 7:] = policy_to_mujoco(joint_pos.to(self.device)).unsqueeze(1)
        plan = self.infer(context, command)
        self.motion.write(env_ids, *plan)
        self.motion.frame[env_ids] = 0
        for dst, src in zip(self.last.as_tuple(), command.as_tuple()):
            dst[env_ids] = src.to(self.device)
        self.initialized[env_ids] = True
        self.tick[env_ids] = 0
        self.replan_counter[env_ids] = 0
        self.idle_state[env_ids] = _IDLE
        self.idle_stored[env_ids] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self.initialized[idx] = False
        self._pending = None
        self._future = None  # whatever is still running is ignored when it finishes

    def plan(self, command: PlannerCommand):
        """Planner tick: decide which environments replan and compute their plans (applied by :meth:`commit`)."""
        self._pending = None
        self.replanned[:] = False
        if self._future is not None:
            return  # one at a time; the change is still seen at the next planner tick, against self.last
        due = self.initialized & (self.tick % self.planner_period == 0)
        if not bool(due.any()):
            return
        mode = command.mode
        interval = torch.full_like(mode, self.replan_ticks["default"])
        interval = torch.where(mode == int(LocomotionMode.RUN), self.replan_ticks["run"], interval)
        interval = torch.where(mode == int(LocomotionMode.CRAWLING), self.replan_ticks["crawl"], interval)
        boxing = torch.isin(mode, torch.tensor(BOXING_REPLAN_MODES, device=mode.device))
        interval = torch.where(boxing, self.replan_ticks["boxing"], interval)
        self.replan_counter = torch.where(due, self.replan_counter + 1, self.replan_counter)
        time_to = due & (self.replan_counter >= interval)
        self.replan_counter = torch.where(time_to, torch.zeros_like(self.replan_counter), self.replan_counter)

        last = self.last
        mode_changed = mode != last.mode
        facing_changed = (command.facing_direction != last.facing_direction).any(-1)
        height_changed = command.height != last.height
        speed_changed = command.speed != last.speed
        dir_changed = (command.movement_direction != last.movement_direction).any(-1)
        static = torch.isin(mode, torch.tensor(STATIC_MODES, device=mode.device))
        moving_update = ~static & (speed_changed | dir_changed | (time_to & (command.speed != 0)))
        need = due & (mode_changed | facing_changed | height_changed | moving_update)
        if not bool(need.any()):
            return
        ids = need.nonzero(as_tuple=False).squeeze(-1)
        context = sample_context(self.motion, ids, self.look_ahead)
        sub = PlannerCommand(*(t[ids] for t in command.as_tuple()))
        if self._worker is not None:
            self._future = (ids, self._worker.submit(self.infer, context, sub), self._clock)
        else:
            self._pending = (ids, self.infer(context, sub))
        self.last.assign(command, need)
        self.replanned[ids] = True

    def commit(self):
        """Blend pending plans into the motion (current frame becomes frame 0)."""
        if self._future is not None and self._future[1].done():
            ids, future, submitted = self._future
            self._future = None
            gen_offset = self.look_ahead - (self._clock - submitted)
            self.late_plans += gen_offset < 0
            blend_into(self.motion, ids, future.result(), gen_offset=gen_offset)
            self.idle_stored[ids] = False
            return
        if self._pending is None:
            return
        ids, plan = self._pending
        blend_into(self.motion, ids, plan, gen_offset=self.look_ahead)
        self.idle_stored[ids] = False
        self._pending = None

    def advance(self, command: PlannerCommand, joint_pos: torch.Tensor):
        """Advance the cursor one tick; at the motion end in IDLE, re-adapt the held lower-body targets."""
        self._clock += 1
        active = self.initialized
        motion = self.motion
        nxt = motion.frame + active.long()
        at_end = active & (nxt >= motion.length)
        motion.frame = torch.where(at_end, (motion.length - 1).clamp_min(0), nxt)
        self.tick = self.tick + active.long()
        if not self.idle_readapt:
            return
        adapt = at_end & (command.mode == int(LocomotionMode.IDLE))
        if not bool(adapt.any()):
            return
        rows = adapt.nonzero(as_tuple=False).squeeze(-1)
        frame = motion.frame[rows]
        target = motion.joint_pos[rows, frame][:, self._lower]
        fresh = ~self.idle_stored[rows]
        self.idle_target[rows[fresh]] = target[fresh]
        self.idle_state[rows[fresh]] = _IDLE
        self.idle_stored[rows] = True
        measured = joint_pos[rows][:, self._lower]
        error = (target - measured).abs().mean(-1)
        state = self.idle_state[rows]
        state = torch.where((state == _IDLE) & (error > ADAPT_TRIGGER), _ADAPTING,
                            torch.where((state == _IDLE) & (error < RECOVER_TRIGGER), _RECOVERING, state))
        state = torch.where((state == _ADAPTING) & (error < ADAPT_STOP), _IDLE, state)
        state = torch.where((state == _RECOVERING) & (error > ADAPT_TRIGGER), _ADAPTING, state)
        self.idle_state[rows] = state
        new = torch.where(
            (state == _ADAPTING).unsqueeze(-1),
            0.98 * target + 0.02 * measured,
            torch.where((state == _RECOVERING).unsqueeze(-1), 0.98 * target + 0.02 * self.idle_target[rows], target),
        )
        jp = motion.joint_pos[rows, frame]
        jp[:, self._lower] = new
        motion.joint_pos[rows, frame] = jp
