"""How far a starting PC has got, for the headset to show while it waits for a scene.

Starting Isaac takes about a minute and the scene port opens only at the end of it. ``StartupProgress`` opens a small
port at once and answers every connection with one packet: ``XPRG``, the fraction done (f32), and the stage's name
(a length byte and ASCII). The fraction moves with time inside each stage, from how long each stage took last time, so
the bar keeps moving through the long ones; it stops short of a stage's end until that stage really ends.

    progress = StartupProgress(["starting Isaac", "building the scene", ...])
    progress.stage("building the scene")
    ...
    progress.done()  # remembers the durations for next time and closes the port
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path

from .wire import PROGRESS_PORT


def pack_progress(fraction: float, text: str) -> bytes:
    raw = text.upper().encode("ascii", "replace")[:47]
    return b"XPRG" + struct.pack("<f", fraction) + bytes([len(raw)]) + raw


class StartupProgress:
    """Args:
        stages: In order. The first starts now.
        history: Where the stage durations of the last start are kept. None keeps nothing.
        guess: Seconds per stage when there is no history.
    """

    def __init__(self, stages: list[str], port: int = PROGRESS_PORT, history: str | Path | None = None, guess: float = 10.0):
        self.stages = list(stages)
        self.history = Path(history) if history else None
        expected = {}
        if self.history is not None and self.history.is_file():
            try:
                expected = {k: float(v) for k, v in json.loads(self.history.read_text()).items()}
            except (ValueError, OSError):
                expected = {}
        self.expected = [max(0.5, expected.get(s, guess)) for s in self.stages]
        self.took: dict[str, float] = {}
        self.current, self.since = 0, time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = None
        try:
            self._sock = socket.create_server(("0.0.0.0", port), reuse_port=False)
            self._sock.settimeout(0.5)
            threading.Thread(target=self._serve, name="startup-progress", daemon=True).start()
        except OSError:  # the port is taken, by a server still closing say: the headset just shows less
            self._sock = None

    def fraction(self, now: float | None = None) -> float:
        with self._lock:
            now = time.monotonic() if now is None else now
            total = sum(self.expected)
            done = sum(self.expected[: self.current])
            if self.current < len(self.stages):
                done += self.expected[self.current] * min(0.95, (now - self.since) / self.expected[self.current])
            return min(1.0, done / total)

    def text(self) -> str:
        with self._lock:
            return self.stages[self.current] if self.current < len(self.stages) else "ready"

    def stage(self, name: str) -> None:
        """Begin ``name``, ending every stage before it."""
        with self._lock:
            now = time.monotonic()
            index = self.stages.index(name)
            if index > self.current:
                self.took[self.stages[self.current]] = now - self.since
                self.current, self.since = index, now

    def done(self) -> None:
        with self._lock:
            if self.current < len(self.stages):
                self.took[self.stages[self.current]] = time.monotonic() - self.since
                self.current = len(self.stages)
        self._stop.set()
        if self.history is not None and self.took:
            try:
                self.history.parent.mkdir(parents=True, exist_ok=True)
                self.history.write_text(json.dumps({k: round(v, 2) for k, v in self.took.items()}))
            except OSError:
                pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                conn.settimeout(1.0)
                conn.sendall(pack_progress(self.fraction(), self.text()))
            except OSError:
                pass
            finally:
                conn.close()
        self._sock.close()
