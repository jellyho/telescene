# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Finding the SONIC ONNX folder, the missing-file error, and the token recorder term (stubbed Isaac Lab)."""


import pytest
import torch

from telescene.sonic.core import SonicPolicyCore
from telescene.sonic.core import checkpoint as ck


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(ck.SONIC_DIR_ENV_VAR, raising=False)


def _fake_models(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ck.MODEL_FILES:
        (folder / name).write_bytes(b"")
    return folder


def test_default_root_is_the_repo_checkpoints_folder():
    assert ck.DEFAULT_CHECKPOINT_ROOT == ck.REPO_ROOT / "checkpoints" / "gear_sonic"
    assert (ck.REPO_ROOT / "pyproject.toml").is_file()


def test_lookup_order(tmp_path, monkeypatch):
    root = tmp_path / "gear_sonic"
    assert ck.find_sonic_dir(root=root) == root  # nothing there: the root, check_sonic_dir explains
    rev = _fake_models(root / "6733128a3d8a")
    assert ck.find_sonic_dir(root=root) == rev  # single HF revision folder
    _fake_models(root / "other_rev")
    assert ck.find_sonic_dir(root=root) == root  # ambiguous
    _fake_models(root)
    assert ck.find_sonic_dir(root=root) == root  # files directly in the root
    monkeypatch.setenv(ck.SONIC_DIR_ENV_VAR, str(rev))
    assert ck.find_sonic_dir(root=root) == rev.resolve()
    assert ck.find_sonic_dir(tmp_path / "explicit", root=root) == (tmp_path / "explicit").resolve()


def test_missing_files_raise_with_help(tmp_path):
    root = tmp_path / "gear_sonic"
    _fake_models(root / "a")
    _fake_models(root / "b")
    with pytest.raises(FileNotFoundError, match="TELESCENE_SONIC_DIR") as err:
        ck.check_sonic_dir(root)
    assert "a, b" in str(err.value) and "telescene.sonic.core.dummy" in str(err.value)
    assert ck.check_sonic_dir(root / "a") == root / "a"
    with pytest.raises(FileNotFoundError, match="planner_sonic.onnx"):
        SonicPolicyCore.from_dir(str(tmp_path / "nothing"), num_envs=1)


def test_dummy_dir_is_found(dummy_dir, monkeypatch):
    monkeypatch.setenv(ck.SONIC_DIR_ENV_VAR, dummy_dir)
    folder = ck.check_sonic_dir(ck.find_sonic_dir())
    core = SonicPolicyCore.from_dir(str(folder), num_envs=1)
    assert core.tokens.shape == (1, 64)
