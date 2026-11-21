# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Shared paths and helpers for the SONIC tests (unique module name: tests/ has no packages)."""

import os

THIRD_PARTY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
SONIC_REPO = os.path.join(THIRD_PARTY, "GR00T-WholeBodyControl")
