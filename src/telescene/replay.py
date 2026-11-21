"""Play a recorded take's operator input back into a session, in place of the headset.

A take (see ``recording.py``) keeps the operator's head, body, controllers and triggers every tick, and whether tracking
was on. ``ReplayServer`` hands one row per tick to ``TeleopSession`` as if it had just arrived, so a new controller, a
follow policy say, can be tried on real operator motion with no headset. Tracking is switched by pressing A for one tick
wherever the recorded tracking changes, and left Y is pressed on the second tick so that the replay records itself.

If the robot falls, the session turns tracking off. The replay then presses A again a second later, as an operator
would, so a fall costs the rest of the take nothing more than the fall itself and trackers stay comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .wire import InputFrame


class ReplayServer:
    """Duck-types ``HeadsetServer`` for ``TeleopSession``. ``start`` and ``stop`` are rows of the take."""

    def __init__(self, take: str | Path, start: int = 0, stop: int | None = None, record: bool = True):
        data = np.load(take)
        self.rows = {k: data[k] for k in ("tracking", "quiet", "worn", "mirror", "head", "body", "grips", "triggers")}
        self.start, self.stop = start, len(self.rows["tracking"]) if stop is None else stop
        self.record = record
        self.row = start
        self.packets_received = self.packets_sent = 0
        self._tracking = False
        self._resume_at = 0

    def tracking_lost(self, wait_ticks: int = 50) -> None:
        """The session turned tracking off. Press A again after ``wait_ticks`` if the take is still tracking."""
        self._tracking = False
        self._resume_at = self.row + wait_ticks

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def done(self) -> bool:
        return self.row >= self.stop

    def latest_input(self, max_age_s: float = 0.5) -> InputFrame | None:
        if self.done:
            return None
        i, r = self.row, self.rows
        self.row += 1
        if r["quiet"][i]:
            return None
        self.packets_received += 1
        tracking = bool(r["tracking"][i]) and i >= self._resume_at
        press = tracking != self._tracking
        self._tracking = tracking
        return InputFrame(seq=i, a_button=press, tracker_ok=True, all_tracked=True, triggers=tuple(r["triggers"][i]),
                          left_axis=(0.0, 0.0), right_axis=(0.0, 0.0), head=r["head"][i], body=r["body"][i],
                          not_worn=not bool(r["worn"][i]), mirror=bool(r["mirror"][i]),
                          y_button=self.record and i == self.start + 1, grips=r["grips"][i])

    def send_poses(self, *args, **kwargs) -> None:
        self.packets_sent += 1

    def take_input_gap_ms(self) -> float:
        return 0.0

    def close(self) -> None:
        pass
