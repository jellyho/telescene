# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Whole-body motion trackers for the G1 that copy the operator's pose, joint for joint.

- ``supermimic``: realmimic's SuperMimic policy, with GMR retargeting the operator to G1 joints.
- ``follow``: an optional network that edits SuperMimic's goal so the robot also walks to where the operator stands.
- ``onnx_tracker``: any tracker released as ONNX, described by a JSON spec (``specs/trackers/``).

Weights are not in the repo: ``scripts/download_trackers.sh`` fetches them. Importing a module registers its controller
(``telescene.make_controller("supermimic", ...)``).
"""
