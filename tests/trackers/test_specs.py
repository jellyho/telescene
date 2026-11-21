"""Every tracker spec in ``specs/trackers/`` loads, and, where ``scripts/download_trackers.sh`` has fetched its checkpoint,
builds inputs of the sizes its ONNX graph wants and returns finite targets, standing and then tracking a body."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))

import fake_body  # noqa: E402
from test_supermimic import control  # noqa: E402

from telescene.trackers import onnx_tracker as ot  # noqa: E402
from telescene.trackers import supermimic as sm  # noqa: E402

SPECS = Path(__file__).parents[2] / "specs" / "trackers"


def shipped() -> list[str]:
    """``file.json`` or ``file.json#variant`` for every runtime block whose checkpoint the download script fetches."""
    out = []
    for path in sorted(SPECS.glob("*.json")):
        runtime = json.loads(path.read_text())["runtime"]
        if "inputs" in runtime:
            out.append(path.name)
        else:
            out += [f"{path.name}#{v}" for v in sorted(runtime) if isinstance(runtime[v], dict) and "inputs" in runtime[v]
                    and (ot.WEIGHTS_DIR / f"{path.stem}_{v}.onnx").exists()]
    return out


@pytest.mark.parametrize("name", shipped())
def test_spec_loads(name):
    spec = ot.TrackerSpec.from_json(str(SPECS / name))
    assert sorted(spec.joint_names) == sorted(sm.JOINT_NAMES)
    assert spec.onnx.parent == ot.WEIGHTS_DIR.resolve()


@pytest.mark.parametrize("name", shipped())
def test_checkpoint_runs(name):
    pytest.importorskip("onnxruntime")
    spec = ot.TrackerSpec.from_json(str(SPECS / name))
    if not spec.onnx.exists():
        pytest.skip(f"no {spec.onnx.name}: scripts/download_trackers.sh")
    retarget = None
    if sm.GMR_ROOT.exists():
        pytest.importorskip("mink")
        retarget = sm.GmrRetargeter(sm.GMR_ROOT, operator_height=1.75)

    def standing():  # with a position and velocity, which some trackers observe and a simulator always gives
        return sm.RobotState(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), sm.DEFAULT_JOINT_POS.copy(), np.zeros(29),
                             pos=np.array([0.0, 0.0, sm.STAND_ROOT_HEIGHT]), lin_vel_b=np.zeros(3))

    controller = ot.OnnxTrackerController(standing, spec, retarget=retarget)
    body = fake_body.pose().astype(np.float32)
    for i in range(30):
        tracking = retarget is not None and i >= 10
        target = controller.act(control(tracking=tracking, body=body if tracking else None))
        assert target.shape == (29,) and np.all(np.isfinite(target))
