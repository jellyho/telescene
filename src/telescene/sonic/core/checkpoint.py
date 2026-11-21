# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Where the SONIC ONNX files are. Nothing is downloaded here: ``scripts/download_sonic.sh`` does that.

Lookup order of :func:`find_sonic_dir`:

1. an explicit path;
2. ``$TELESCENE_SONIC_DIR``;
3. ``<repo>/checkpoints/gear_sonic/`` itself if it holds the three models, else its single
   sub-folder that does (``checkpoints/gear_sonic/<hf revision>/``, the HF ``nvidia/GEAR-SONIC`` layout).

For a smoke test without the licence, point ``TELESCENE_SONIC_DIR`` at a folder written by
``python -m telescene.sonic.core.dummy <dir>`` (same names, shapes and dtypes; meaningless as control).
"""

from __future__ import annotations

import os
from pathlib import Path

ENCODER_FILE = "model_encoder.onnx"
DECODER_FILE = "model_decoder.onnx"
PLANNER_FILE = "planner_sonic.onnx"
MODEL_FILES = (ENCODER_FILE, DECODER_FILE, PLANNER_FILE)

SONIC_DIR_ENV_VAR = "TELESCENE_SONIC_DIR"
REPO_ROOT = Path(__file__).resolve().parents[4]
"""Repository root for a source checkout (``src/telescene/sonic/core``)."""
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "gear_sonic"


def _has_models(folder: Path, files: tuple[str, ...] = MODEL_FILES) -> bool:
    return all((folder / f).is_file() for f in files)


def find_sonic_dir(path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None) -> Path:
    """The folder to load SONIC from. It is not checked here; :func:`check_sonic_dir` raises with help.

    Args:
        path: Explicit folder; wins over everything.
        root: Default search root (``checkpoints/gear_sonic`` of the repository).
    """
    if path is not None:
        return Path(path).expanduser().resolve()
    if os.environ.get(SONIC_DIR_ENV_VAR):
        return Path(os.environ[SONIC_DIR_ENV_VAR]).expanduser().resolve()
    root = Path(root) if root is not None else DEFAULT_CHECKPOINT_ROOT
    if _has_models(root):
        return root
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and _has_models(p)) if root.is_dir() else []
    return candidates[0] if len(candidates) == 1 else root


def check_sonic_dir(onnx_dir: str | os.PathLike[str], files: tuple[str, ...] = MODEL_FILES) -> Path:
    """Raise a FileNotFoundError naming what is missing and how to provide it; returns the folder."""
    folder = Path(onnx_dir)
    missing = [f for f in files if not (folder / f).is_file()]
    if not missing:
        return folder
    hint = ""
    if folder.is_dir():
        subdirs = sorted(p.name for p in folder.iterdir() if p.is_dir() and _has_models(p, files))
        if subdirs:
            hint = f"\n  Several checkpoints found under it ({', '.join(subdirs)}); pick one with {SONIC_DIR_ENV_VAR}."
    raise FileNotFoundError(
        f"SONIC ONNX files missing in {folder}: {', '.join(missing)}.{hint}\n"
        f"  Run scripts/download_sonic.sh (HF nvidia/GEAR-SONIC, NVIDIA Open Model License) or set"
        f" {SONIC_DIR_ENV_VAR}=/path/to/GEAR-SONIC. For a smoke test, stand-in models: python -m telescene.sonic.core.dummy <dir>"
    )
