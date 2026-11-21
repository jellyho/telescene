# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Fixtures for the SONIC tests: dummy ONNX models with the released signatures."""

import pytest


@pytest.fixture(scope="session")
def dummy_dir(tmp_path_factory):
    """Dummy encoder (1762), decoder (994) and planner with batch fixed to 1, like the release export."""
    from telescene.sonic.core.dummy import make_dummy_sonic_dir

    return make_dummy_sonic_dir(str(tmp_path_factory.mktemp("sonic_fixed")), batch=1)


@pytest.fixture(scope="session")
def dummy_dir_dynamic(tmp_path_factory):
    """Same weights as :func:`dummy_dir` but with a symbolic batch dimension."""
    from telescene.sonic.core.dummy import make_dummy_sonic_dir

    return make_dummy_sonic_dir(str(tmp_path_factory.mktemp("sonic_dynamic")), batch=None)
