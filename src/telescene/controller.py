"""The seam between the headset and whatever makes the robot move.

A controller turns one tick of operator input into one simulator action. Everything an operator expects of every
controller stays outside it: buttons, walking to the operator, reset, timing and rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .wire import InputFrame


@dataclass(frozen=True)
class ControlInput:
    """One tick of operator input.

    Attributes:
        frame: Newest headset frame, or None before the first one. It stays the last frame if the headset goes quiet.
        body: ``(24, 7)`` xyz, xyzw from ``XR_BD_body_tracking``, or None before the tracker has reported. Holds the
            last good body if tracking drops, so the robot pauses instead of dropping out.
        tracking: The operator wants the whole body tracked. Toggled with A.
        velocity: Body frame ``(vx, vy, wz)`` that walks the robot to the operator. Zero while tracking.
        triggers: Left trigger, left grip, right trigger, right grip, from 0 to 1.
        left_axis: Left stick. Zero while the headset is quiet, so the robot does not walk off unsteered.
        right_axis: Right stick. Same.
        dt: Seconds since the previous tick.
        quiet: No input has arrived recently.
        anchor: Simulator position of the operator's room origin.
        anchor_yaw: Simulator yaw of the operator's room.
        world_scale: Simulator metres per real metre.
    """

    frame: InputFrame | None
    body: np.ndarray | None
    tracking: bool
    velocity: tuple[float, float, float]
    triggers: tuple[float, float, float, float]
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    dt: float
    quiet: bool
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_yaw: float = 0.0
    world_scale: float = 1.0


class Controller(Protocol):
    """Implement this to drive a robot with your own policy. See ``telescene.mirror`` and ``register``.

    Set ``walks_while_tracking = True`` if the legs keep following ``velocity`` while the operator is tracked, as with a
    planner under the legs and only the upper body tracked. Without it the session sends zero velocity while tracking.
    """

    def reset(self) -> None:
        """The scene was just reset or the robot fell. Forget any state."""

    def act(self, control: ControlInput) -> Any:
        """Return one action for ``Simulator.step``. This runs in the control loop, so it must not block."""

    def status(self) -> str:
        """A few words for the log. May be empty."""


_REGISTRY: dict[str, Callable[..., Controller]] = {}


def register(name: str, factory: Callable[..., Controller]) -> None:
    """Make a controller available by name. Registering a name twice is a mistake and raises."""
    if name in _REGISTRY:
        raise ValueError(f"controller '{name}' is already registered")
    _REGISTRY[name] = factory


def make_controller(name: str, **kwargs) -> Controller:
    if name not in _REGISTRY:
        raise KeyError(f"no controller named '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered() -> list[str]:
    return sorted(_REGISTRY)
