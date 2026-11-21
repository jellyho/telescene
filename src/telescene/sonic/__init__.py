# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""SONIC (NVIDIA GEAR-SONIC) as the robot's whole-body controller, for G1 in Isaac Lab.

* :mod:`.core` is numpy / torch / onnxruntime only and imports on any machine.
* :mod:`.controllers` turns headset input into SONIC's action: :class:`~.controllers.SonicSwitchController`
  (hybrid head and hands with planner legs, and whole-body POSE on A) and :class:`~.controllers.SonicPoseController`.
* :mod:`.mirror` shows the SMPL body SONIC is given, over your tracked joints, with no simulator.
* :mod:`.action`, :mod:`.action_cfg` and :mod:`.actuators` need Isaac Lab. They are not imported here;
  ``SonicAction`` and ``SonicActionCfg`` resolve lazily on first access.
"""

_LAZY = {
    "SonicAction": ".action",
    "SonicActionCfg": ".action_cfg",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
