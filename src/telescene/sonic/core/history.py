# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Batched fixed-length history ring (oldest first), as consumed by the SONIC decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class HistoryBuffer:
    """History of the last ``length`` samples per environment.

    ``init="zeros"`` pads missing history with zeros, as both released inference references do
    (sonic_policy.py:80 ``np.zeros((10, 93))`` and the deploy StateLogger, state_logger.cpp:189-195).
    ``init="repeat"`` fills the whole window with the first sample after a reset, as Isaac Lab's
    ``CircularBuffer`` does during training.

    Args:
        num_envs: Batch size.
        length: Number of stored samples.
        dim: Feature size of one sample.
        device: Torch device.
        init: ``"zeros"`` or ``"repeat"``.
    """

    def __init__(self, num_envs: int, length: int, dim: int, device: str | torch.device = "cpu", init: str = "zeros"):
        if init not in ("zeros", "repeat"):
            raise ValueError(f"init must be 'zeros' or 'repeat', got {init!r}")
        self.length = length
        self.dim = dim
        self.init = init
        self._buf = torch.zeros(num_envs, length, dim, device=device)
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    @property
    def data(self) -> torch.Tensor:
        """``(num_envs, length, dim)``, oldest sample first."""
        return self._buf

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self._buf[idx] = 0.0
        self._fresh[idx] = True

    def append(self, sample: torch.Tensor):
        """Push ``(num_envs, dim)`` as the newest sample."""
        self._buf = torch.roll(self._buf, shifts=-1, dims=1)
        self._buf[:, -1] = sample
        if self.init == "repeat" and bool(self._fresh.any()):
            self._buf[self._fresh] = sample[self._fresh].unsqueeze(1).expand(-1, self.length, -1)
        self._fresh[:] = False

    def window(self, num_frames: int, step: int = 1) -> torch.Tensor:
        """The newest ``num_frames`` samples ``step`` apart, oldest first: ``(num_envs, num_frames, dim)``."""
        span = (num_frames - 1) * step + 1
        if span > self.length:
            raise ValueError(f"window of {num_frames} frames x step {step} exceeds history length {self.length}")
        return self._buf[:, self.length - span :: step]
