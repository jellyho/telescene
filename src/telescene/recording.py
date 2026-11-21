"""Record a teleop session: every tick's simulator state and the operator's input, for replay and rendering.

Rows go into arrays allocated ahead in large blocks, so recording adds no per-tick Python objects for the garbage
collector to walk (that stalled the loop before) and no allocation on most ticks. ``save`` writes one ``.npz``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BLOCK = 50 * 60  # one minute at 50 Hz


class Recorder:
    """Args:
        path: Where to write the ``.npz``. A ``.json`` with the metadata goes beside it.
        meta: Anything worth keeping about the run, such as task and controller.
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.columns: dict[str, np.ndarray] = {}
        self.rows = 0
        self.started = time.time()

    def _grow(self, name: str, value: np.ndarray) -> np.ndarray:
        column = self.columns.get(name)
        if column is None:
            column = np.zeros((BLOCK, *value.shape), value.dtype)
        elif self.rows >= len(column):
            column = np.concatenate((column, np.zeros_like(column[:BLOCK])))
        self.columns[name] = column
        return column

    def add(self, **values) -> None:
        """One row. Every call should give the same names with the same shapes."""
        for name, value in values.items():
            value = np.asarray(value)
            self._grow(name, value)[self.rows] = value
        self.rows += 1

    def save(self) -> Path | None:
        if not self.rows:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **{k: v[:self.rows] for k, v in self.columns.items()})
        meta = {**self.meta, "rows": self.rows, "seconds": round(time.time() - self.started, 1)}
        self.path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        return self.path
