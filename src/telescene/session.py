"""The teleop loop: operator input, controller, simulator, poses out. Knows no simulator or controller.

The headset is never waited for. Input is read without blocking, poses are sent without blocking, and the scene goes
out on another thread, so a slow or absent headset cannot delay physics.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from .controller import ControlInput, Controller
from .server import HeadsetServer


class Simulator(Protocol):
    def reset(self) -> None:
        """Put the scene back to its start."""

    def step(self, action: Any) -> bool:
        """Advance one control tick. True if the episode ended and the simulator reset itself."""

    def root_pose(self) -> tuple[float, float, float]:
        """Robot ``(x, y, yaw)`` in the simulator's frame."""

    def link_poses(self) -> np.ndarray:
        """``(n, 7)`` xyz, xyzw for everything that moves, in the order the scene was built."""

    def anchor(self) -> tuple[tuple[float, float, float], float]:
        """``((x, y, z), yaw)``: where the operator's room origin sits in the simulator."""

    def hand_forces(self) -> tuple[float, float]:
        """Contact force on the left and right hand in newtons. ``(0.0, 0.0)`` if the simulator cannot tell."""


class RisingEdge:
    """True once per press."""

    def __init__(self) -> None:
        self._held = False

    def __call__(self, pressed: bool) -> bool:
        rising = pressed and not self._held
        self._held = pressed
        return rising


class Toggle:
    """Flip on a rising edge, so holding a button does not flicker."""

    def __init__(self, on: bool = False) -> None:
        self.on, self._edge = on, RisingEdge()

    def __call__(self, pressed: bool) -> bool:
        if self._edge(pressed):
            self.on = not self.on
        return self.on

    def force_off(self) -> None:
        """Off until the next press. A button still held does not count as one."""
        self.on = False


class FollowGate:
    """Whether the robot should walk to the operator. Sticks take over, walking takes back.

    Without this, following pulls against the sticks and undoes them when they are let go. Hip height on the
    right stick's y axis does not move the robot, so it does not count.

    Args:
        stick: Stick deflection that counts as steering.
        walk: Real metres the operator's head must move from where it was at the last steering.
    """

    def __init__(self, stick: float = 0.1, walk: float = 0.3) -> None:
        self.stick, self.walk = stick, walk
        self._paused_at: tuple[float, float] | None = None

    def reset(self) -> None:
        self._paused_at = None

    def __call__(self, left_axis, right_axis, head) -> bool:
        x, z = float(head[0]), float(head[2])
        if max(abs(left_axis[0]), abs(left_axis[1]), abs(right_axis[0])) > self.stick:
            self._paused_at = (x, z)
            return False
        if self._paused_at is None:
            return True
        if np.hypot(x - self._paused_at[0], z - self._paused_at[1]) > self.walk:
            self._paused_at = None
            return True
        return False


def head_in_sim(head, anchor, anchor_yaw: float, world_scale: float) -> tuple[float, float, float]:
    """Head pose (OpenXR local floor, xyz xyzw) to simulator ``(x, y, yaw)``. Inverse of the app's mapping."""
    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)

    def to_world(rx: float, rz: float) -> tuple[float, float]:
        forward, left = -rz, -rx  # -Z is forward and -X is left at the anchor's heading
        return c * forward - s * left, s * forward + c * left

    dx, dy = to_world(float(head[0]), float(head[2]))
    qx, qy, qz, qw = (float(v) for v in head[3:7])
    # The head's forward, -Z rotated by the quaternion, flattened to the floor.
    fx, fz = -2 * (qx * qz + qw * qy), -(1 - 2 * (qx * qx + qy * qy))
    wx, wy = to_world(fx, fz)
    return anchor[0] + dx * world_scale, anchor[1] + dy * world_scale, float(np.arctan2(wy, wx))


def recenter(head, pelvis, spawn_xy, spawn_yaw: float, world_scale: float) -> tuple[tuple[float, float, float], float]:
    """A room anchor that puts the operator where the robot spawns, facing the way it faces: the anchor and yaw for
    which ``head_in_sim`` of this head gives ``spawn_xy`` and ``spawn_yaw``. ``pelvis`` (room xyz), when tracked, places
    the operator better than the head, which leans; the heading is the head's, the way the operator looks."""
    _, _, room_yaw = head_in_sim(head, (0.0, 0.0, 0.0), 0.0, 1.0)
    anchor_yaw = float(spawn_yaw - room_yaw)
    at = pelvis if pelvis is not None else head
    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
    forward, left = -float(at[2]), -float(at[0])
    dx, dy = c * forward - s * left, s * forward + c * left
    return (float(spawn_xy[0] - dx * world_scale), float(spawn_xy[1] - dy * world_scale), 0.0), anchor_yaw


def follow_command(target, root_xy, root_yaw: float, walking: bool = True, gain: float = 1.2, min_speed: float = 0.35,
                   max_speed: float = 0.6, yaw_gain: float = 1.2, max_turn: float = 1.0,
                   yaw_deadband: float = 0.25) -> tuple[float, float, float]:
    """Body frame ``(vx, vy, wz)`` that walks the robot to ``target = (x, y, yaw)``.

    ``min_speed``: SONIC's planner marches on the spot below about 0.2 m/s, so easing off near the target would leave
    the robot stuck short of it. ``FollowGate`` is not involved: ``walking`` says whether to move at all.
    """
    ex, ey = target[0] - root_xy[0], target[1] - root_xy[1]
    distance = float(np.hypot(ex, ey))
    vx = vy = 0.0
    if walking and distance > 1e-6:
        speed = float(np.clip(gain * distance, min_speed, max_speed))
        c, s = np.cos(root_yaw), np.sin(root_yaw)
        vx, vy = speed * (c * ex + s * ey) / distance, speed * (-s * ex + c * ey) / distance
    error = (target[2] - root_yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(error) < yaw_deadband:  # otherwise it dithers either side of the target, and each turn makes a planner replan
        return float(vx), float(vy), 0.0
    return float(vx), float(vy), float(np.clip(yaw_gain * error, -max_turn, max_turn))


# XR_BD_body_tracking joints, which follow the SMPL order.
PELVIS, LEFT_HIP, RIGHT_HIP, SPINE1, LEFT_KNEE, RIGHT_KNEE, SPINE2, LEFT_ANKLE, RIGHT_ANKLE, SPINE3 = range(10)
NECK, HEAD = 12, 15
_SPINE = (PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD)
_LEG = ((LEFT_ANKLE, LEFT_KNEE, LEFT_HIP), (RIGHT_ANKLE, RIGHT_KNEE, RIGHT_HIP))
BELOW_ANKLE, ABOVE_HEAD_JOINT = 0.08, 0.14
"""Metres of a standing person that the skeleton's joints do not span."""


def skeleton_height(body: np.ndarray) -> float:
    """Standing height from bone lengths, which do not change with posture, so the operator need not stand straight."""
    def chain(joints) -> float:
        return float(sum(np.linalg.norm(body[a, :3] - body[b, :3]) for a, b in zip(joints, joints[1:])))

    return float(np.mean([chain(joints) for joints in _LEG]) + chain(_SPINE) + BELOW_ANKLE + ABOVE_HEAD_JOINT)


def haptic_amplitude(force: float, gain: float = 1.0, touch: float = 3.0, full: float = 200.0,
                     faintest: float = 0.15) -> float:
    """Controller vibration for a contact force in newtons.

    A G1 hand read 6 to 50 N touching things and up to 694 N when bumping into them, in a worn test, so ``full`` is high
    enough that those two feel different. ``gain`` scales the result: 0 turns vibration off.
    """
    if force < touch:
        return 0.0
    return float(min(1.0, gain * (faintest + (1.0 - faintest) * (force - touch) / (full - touch))))


def scale_for_height(robot_height: float, operator_height: float) -> float:
    """World scale that makes a robot of ``robot_height`` look as tall as the operator."""
    return robot_height / operator_height


class AutoScale:
    """World scale from the operator's skeleton: the robot's height over the operator's. **Not yet validated.**

    No call that reads the height set in PICO's settings was found, so this measures the tracked skeleton. Whether
    that skeleton follows the real person is not known, and ``BELOW_ANKLE`` and ``ABOVE_HEAD_JOINT`` are estimates. It
    decides once, from a worn and fully tracked headset, and never changes the scale under the operator after that.

    Args:
        robot_height: Metres. 1.32 for a Unitree G1.
        samples: Ticks to take the median over.
        limits: Heights outside this are a bad measurement, and the fallback stays.
    """

    def __init__(self, robot_height: float, samples: int = 100, limits: tuple[float, float] = (1.3, 2.2)):
        self.robot_height, self.samples, self.limits = robot_height, samples, limits
        self.operator_height: float | None = None
        self.done = False
        self._seen: list[float] = []

    def __call__(self, frame) -> float | None:
        """The new scale at the tick it is decided, otherwise None."""
        if self.done or frame is None or frame.not_worn or not frame.tracker_ok or not frame.all_tracked:
            return None
        self._seen.append(skeleton_height(frame.body))
        if len(self._seen) < self.samples:
            return None
        self.done = True
        height = float(np.median(self._seen))
        if not self.limits[0] <= height <= self.limits[1]:
            return None
        self.operator_height = height
        return scale_for_height(self.robot_height, height)


class WalkDecision:
    """Whether the robot should be walking to the operator: start when far, carry on until close.

    The gap between the two keeps it still while the operator only shifts their weight, and stops it dithering at the
    edge.
    """

    def __init__(self, start: float = 0.25, stop: float = 0.1) -> None:
        self.start, self.stop = start, stop
        self.walking = False

    def __call__(self, distance: float) -> bool:
        if distance > self.start:
            self.walking = True
        elif distance < self.stop:
            self.walking = False
        return self.walking


class CommandHold:
    """Keeps a walking command steady so that a planner under the legs can get on with it.

    A command worked out from the operator's position changes a little every tick. A planner that replans on any change
    then restarts its motion ten times a second and the robot shuffles on the spot, which SONIC's did. So a new command
    goes through only if it differs enough and the last one has had time to act, except for starting and stopping.

    Args:
        speed: Change in m/s that counts.
        turn: Change in rad/s that counts.
        direction: Change in heading of travel, radians, that counts.
        hold: Seconds a command stays at least.
        moving: Speed below this is standing still.
    """

    def __init__(self, speed: float = 0.1, turn: float = 0.25, direction: float = 0.5, hold: float = 0.6, moving: float = 0.05):
        self.speed, self.turn, self.direction, self.hold, self.moving = speed, turn, direction, hold, moving
        self._held = (0.0, 0.0, 0.0)
        self._since = -math.inf

    def reset(self) -> None:
        self._held, self._since = (0.0, 0.0, 0.0), -math.inf

    def _differs(self, new) -> bool:
        old_speed, new_speed = math.hypot(*self._held[:2]), math.hypot(*new[:2])
        if abs(new_speed - old_speed) > self.speed or abs(new[2] - self._held[2]) > self.turn:
            return True
        if new_speed < self.moving or old_speed < self.moving:
            return False
        change = math.atan2(new[1], new[0]) - math.atan2(self._held[1], self._held[0])
        return abs((change + math.pi) % (2 * math.pi) - math.pi) > self.direction

    def __call__(self, command, now: float) -> tuple[float, float, float]:
        moving_now = math.hypot(*command[:2]) >= self.moving
        moving_before = math.hypot(*self._held[:2]) >= self.moving
        stopping = moving_before and not moving_now
        starting = moving_now and not moving_before
        settled = now - self._since >= self.hold
        if stopping or starting or (settled and self._differs(command)):
            self._held, self._since = tuple(command), now
        return self._held


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))] if ordered else float("nan")


class TeleopSession:
    """Runs one simulator and one controller against one headset.

    Right A toggles tracking. Right B resets the scene and, with ``recenter_on_reset``, re-anchors the room so that
    the operator stands where the robot spawns and looks the way it faces. Tracking turns off when the headset comes off or the robot
    falls: copying a headset on a desk, or whatever just made the robot fall, would only make it fall again.

    Args:
        simulator: The simulator, already reset.
        controller: Makes the actions.
        server: The headset link.
        world_scale: Simulator metres per real metre. Smaller makes the world look bigger.
        auto_scale: Replaces ``world_scale`` once it has measured the operator. None keeps ``world_scale``.
        haptic_gain: Strength of the controllers' vibration. 0 turns it off.
        recorder: ``recorder() -> Recorder``, called when the operator presses left Y to start a take. Each take gets a
            row per tick, the simulator's ``snapshot()`` if it has one and the operator's input, and is saved when Y is
            pressed again or the session ends. The controllers buzz once, long, to start and twice, short, to stop.
        mirror: Serves the app's mirror mode: a ``MirrorView`` also given to the server as ``mirror_blob``. While the
            operator is in it or in the menu the robot stands still with tracking off.
        follow: With tracking off, walk the robot to the operator.
        recenter_on_reset: On right B, align the room with where the operator is and looks (see ``recenter``).
        rate: Control rate to hold, Hz.
        realtime: Hold the rate. False runs each tick as soon as the last is done, on a clock that advances one period per
            tick, for replays: the same run, faster.
        log_every: Seconds between log lines. Zero for none.
        log: Where log lines go.
    """

    def __init__(self, simulator: Simulator, controller: Controller, server: HeadsetServer, world_scale: float = 1.0,
                 auto_scale: AutoScale | None = None, haptic_gain: float = 1.0, mirror=None, recorder=None, follow: bool = True, rate: float = 50.0, log_every: float = 3.0, log: Callable[[str], None] = print,
                 recenter_on_reset: bool = True, realtime: bool = True):
        self.simulator, self.controller, self.server = simulator, controller, server
        self.realtime, self._ticks = realtime, 0
        self.world_scale, self.follow, self.period, self.log_every = world_scale, follow, 1.0 / rate, log_every
        self._log = log
        self.auto_scale, self.haptic_gain, self.mirror = auto_scale, haptic_gain, mirror
        self.new_recording, self.recorder, self.saved = recorder, None, []
        self._record_press = RisingEdge()
        self._buzz: tuple[float, list[tuple[float, float]]] | None = None  # start time, (from, to) seconds after it
        self._t0 = time.perf_counter()
        self.tracking = Toggle()
        self.gate, self._reset_press = FollowGate(), RisingEdge()
        self.hold, self.walk = CommandHold(), WalkDecision()
        self.frame = None
        self.body = None
        self.quiet = True
        self.anchor, self.anchor_yaw = simulator.anchor()
        self.spawn, self.recenter_on_reset = (tuple(self.anchor), self.anchor_yaw), recenter_on_reset
        self.resets = 0
        self._last = self._clock()
        self._peak_force = (0.0, 0.0)
        self._tick_ms: list[float] = []
        self._step_ms: list[float] = []

    def _read(self):
        fresh = self.server.latest_input()
        self.quiet = fresh is None
        if fresh is not None:
            self.frame = fresh
            if fresh.tracker_ok:
                self.body = fresh.body

    def _worn(self) -> bool:
        return self.frame is None or not self.frame.not_worn

    def _velocity(self, tracking: bool, left, right) -> tuple[float, float, float]:
        if tracking and not getattr(self.controller, "walks_while_tracking", False):
            self.gate.reset()  # back in charge when tracking ends: the robot comes to the operator
            return (0.0, 0.0, 0.0)
        if self.frame is None or not self._worn() or not self.follow or not self.gate(left, right, self.frame.head):
            self.hold.reset()
            return (0.0, 0.0, 0.0)
        target = head_in_sim(self.frame.head, self.anchor, self.anchor_yaw, self.world_scale)
        x, y, yaw = self.simulator.root_pose()
        walking = self.walk(math.hypot(target[0] - x, target[1] - y))
        return self.hold(follow_command(target, (x, y), yaw, walking), self._clock())

    def tick(self) -> None:
        """One control tick. Takes as long as the controller and simulator take; ``run`` does the pacing."""
        t0, clock = time.perf_counter(), self._clock()
        self._ticks += 1
        self._read()
        frame = self.frame
        if self.auto_scale is not None:
            scale = self.auto_scale(frame if not self.quiet else None)
            if scale is not None:
                self.world_scale = scale
                self._log(f"[telescene] operator about {self.auto_scale.operator_height:.2f} m from the skeleton: "
                          f"world scale {scale:.3f}")
            elif self.auto_scale.done and self.auto_scale.operator_height is None:
                self.auto_scale = None
                self._log(f"[telescene] could not measure the operator: world scale stays {self.world_scale:.3f}")
        pressed = frame is not None and not self.quiet
        tracking = self.tracking(frame.a_button if frame is not None else False)
        in_mirror = frame is not None and frame.mirror and self.mirror is not None
        away = frame is not None and (frame.menu or in_mirror)  # the operator is not driving the robot
        if away:
            self.tracking.force_off()
            tracking = False
        if tracking and not self._worn():
            self.tracking.force_off()
            tracking = False
            self._log("[telescene] tracking off: the headset came off. Press A to resume.")
        if self._reset_press(pressed and frame.b_button and not away):
            self.simulator.reset()
            self.controller.reset()
            self.tracking.force_off()
            self.gate.reset()
            tracking = False
            if self.recenter_on_reset:
                pelvis = self.body[0, :3] if self.body is not None and frame.tracker_ok else None
                self.anchor, self.anchor_yaw = recenter(frame.head, pelvis, self.spawn[0], self.spawn[1], self.world_scale)
            self._log("[telescene] scene reset by the operator" + (", room re-aligned to where you look" if self.recenter_on_reset else ""))

        still = (0.0, 0.0)
        left, right = (frame.left_axis, frame.right_axis) if pressed and not away else (still, still)
        control = ControlInput(
            frame=frame,
            body=self.body,
            tracking=tracking,
            velocity=(0.0, 0.0, 0.0) if away else self._velocity(tracking, left, right),
            triggers=frame.triggers if frame is not None else (0.0, 0.0, 0.0, 0.0),
            left_axis=left,
            right_axis=right,
            dt=clock - self._last,
            quiet=self.quiet,
            anchor=tuple(self.anchor),
            anchor_yaw=self.anchor_yaw,
            world_scale=self.world_scale,
        )
        self._last = clock
        action = self.controller.act(control)
        t1 = time.perf_counter()
        fell = self.simulator.step(action)
        t2 = time.perf_counter()
        if fell:
            self.controller.reset()
            self.resets += 1
            if tracking:
                self.tracking.force_off()
                self._log("[telescene] tracking off: the robot reset. Press A to resume.")
                if hasattr(self.server, "tracking_lost"):  # a replay presses A again by itself
                    self.server.tracking_lost()
        if self.new_recording is not None and self._record_press(pressed and frame.y_button):
            self._toggle_recording()
        if self.recorder is not None:
            self._record(tracking, in_mirror, fell)
        forces = self.simulator.hand_forces()
        self._peak_force = (max(self._peak_force[0], forces[0]), max(self._peak_force[1], forces[1]))
        haptics = (haptic_amplitude(forces[0], self.haptic_gain), haptic_amplitude(forces[1], self.haptic_gain))
        if self._buzz is not None:
            since = time.perf_counter() - self._buzz[0]
            if since > self._buzz[1][-1][1]:
                self._buzz = None
            elif any(a <= since < b for a, b in self._buzz[1]):
                haptics = (1.0, 1.0)
        if in_mirror:
            self.mirror.act(control)
            origin, yaw = self.mirror.anchor()
            self.server.send_poses(self.mirror.link_poses(), origin, yaw, 1.0, status=self._status())
        else:
            self.server.send_poses(self.simulator.link_poses(), self.anchor, self.anchor_yaw, self.world_scale, haptics,
                                   self._status())
        self._step_ms.append((t2 - t1) * 1000)
        self._tick_ms.append((time.perf_counter() - t0) * 1000)

    def _status(self) -> int:
        from .wire import STATUS_RECORDING

        return STATUS_RECORDING if self.recorder is not None else 0

    def _toggle_recording(self) -> None:
        if self.recorder is None:
            self.recorder = self.new_recording()
            self._buzz = (time.perf_counter(), [(0.0, 0.4)])
            self._log(f"[telescene] recording to {self.recorder.path}")
        else:
            self.stop_recording()
            self._buzz = (time.perf_counter(), [(0.0, 0.12), (0.25, 0.37)])

    def stop_recording(self) -> None:
        """Save the take, if one is running."""
        if self.recorder is None:
            return
        path, rows = self.recorder.save(), self.recorder.rows
        self.recorder = None
        if path is not None:
            self.saved.append(path)
            self._log(f"[telescene] recorded {rows} ticks ({rows / 50:.0f} s) to {path}")

    def _record(self, tracking: bool, in_mirror: bool, fell: bool = False) -> None:
        frame = self.frame
        snapshot = getattr(self.simulator, "snapshot", None)
        no_pose = np.zeros(7, np.float32)
        self.recorder.add(
            time=self._clock() - self._t0,
            tracking=tracking,
            mirror=in_mirror,
            fell=bool(fell),
            quiet=self.quiet,
            worn=self._worn(),
            world_scale=self.world_scale,
            head=frame.head if frame is not None else no_pose,
            body=self.body if self.body is not None else np.zeros((24, 7), np.float32),
            grips=frame.grips if frame is not None and frame.grips is not None else np.zeros((2, 7), np.float32),
            triggers=frame.triggers if frame is not None else (0.0, 0.0, 0.0, 0.0),
            **(snapshot() if snapshot is not None else {}),
            **(self.controller.record() if hasattr(self.controller, "record") else {}),
        )

    def _report(self) -> None:
        tick, step = self._tick_ms, self._step_ms
        held = "HOLDING LAST INPUT " if self.quiet and self.frame is not None else ""
        state = "no input" if self.frame is None else (
            f"{held}head {np.round(self.frame.head[:3], 2).tolist()} all_tracked {self.frame.all_tracked}")
        x, y, _ = self.simulator.root_pose()
        status = (self.mirror.status() if self.mirror is not None else "") or self.controller.status()
        self._log(f"[telescene] tick p50 {_percentile(tick, 0.5):.1f} p99 {_percentile(tick, 0.99):.1f} "
                  f"max {max(tick, default=math.nan):.0f} ms against {self.period * 1000:.0f} | "
                  f"step p50 {_percentile(step, 0.5):.1f} max {max(step, default=math.nan):.0f} | "
                  f"tracking {'on' if self.tracking.on else 'off'} | resets {self.resets} | {state} | "
                  f"link {self.server.packets_received} in {self.server.packets_sent} out, "
                  f"input gap max {self.server.take_input_gap_ms():.0f} ms | root {x:.2f} {y:.2f} | "
                  f"hand force peak {self._peak_force[0]:.0f} {self._peak_force[1]:.0f} N"
                  + (" | REC" if self.recorder is not None else "")
                  + (f" | {status}" if status else ""))
        self._tick_ms, self._step_ms, self.resets, self._peak_force = [], [], 0, (0.0, 0.0)
        if self.frame is not None and self.frame.tracker_ok and self.frame.all_tracked and self._worn():
            # To compare against the operator's real height when validating AutoScale.
            self._log(f"[telescene] measured: skeleton {skeleton_height(self.frame.body):.2f} m, "
                      f"head {float(self.frame.head[1]):.2f} m, scale {self.world_scale:.3f}")

    def _clock(self) -> float:
        """Seconds: the wall clock, or with ``realtime`` False one period per tick since the session began."""
        return time.perf_counter() if self.realtime else self._t0 + self._ticks * self.period

    def run(self, seconds: float = 0.0) -> None:
        """Until ``seconds`` have passed, or forever if zero, or a replay has run out. Hold the rate, and skip rather than
        catch up."""
        # A full collection walks everything the simulator made at startup, which stalled the loop for about 150 ms
        # under Isaac. All of that lives for the whole run, so leave it out. Later garbage is still collected.
        gc.collect()
        gc.freeze()
        start = next_tick = time.perf_counter()
        next_log = start + self.log_every
        try:
            while not seconds or time.perf_counter() - start < seconds:
                if getattr(self.server, "requested_scene", None) is not None:
                    self._log("[telescene] stopping for the scene the headset picked")
                    break
                if getattr(self.server, "done", False) is True:  # a replay that has played every row
                    break
                self.tick()
                now = time.perf_counter()
                if self.log_every and now >= next_log:
                    self._report()
                    next_log = now + self.log_every
                if self.realtime:
                    next_tick = max(next_tick + self.period, now)
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
        finally:
            self.stop_recording()
