"""Write ``supermimic_vectors.npz``: inputs and outputs of realmimic's own functions, to test our copy against.

Run with realmimic's own environment, and with its commit recorded in the output, from outside its checkout:

    $C/venv/bin/python tests/fixtures/make_supermimic_vectors.py $C/realmimic $C/weights/realmimic_g1_86k.pt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

repo, checkpoint = Path(sys.argv[1]), sys.argv[2]

from realmimic.export.web_policy import build, sample_inputs  # noqa: E402
from realmimic.robots import get_robot_spec  # noqa: E402
from realmimic.robots.g1 import articulation, joint_gains  # noqa: E402
from supermimic.envs.mdp.motion_library import sonic_goal_features  # noqa: E402
from supermimic.kinematics.rotations import anchor_window_obs  # noqa: E402
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major  # noqa: E402

web, constants, ck = build(checkpoint)
samples = sample_inputs(constants, 32, 10, 10, seed=7)
out = {k: [] for k in ("history", "goal", "action", "joint_target", "state_hist", "action_hist", "ref_window")}
with torch.no_grad():
    for sample in samples:
        state, last, ref = (torch.from_numpy(sample[k]) for k in ("state_hist", "action_hist", "ref_window"))
        obs = analytic_observation(state, last, web.xi, web.q0)
        history = flatten_history_term_major(obs.transpose(0, 1))
        goal = torch.cat([sonic_goal_features(ref), anchor_window_obs(state[:, -1, 3:7], ref[..., 3:7])], dim=-1)
        action, target = web(state, last, ref)
        for name, value in (("history", history), ("goal", goal), ("action", action), ("joint_target", target),
                            ("state_hist", state), ("action_hist", last), ("ref_window", ref)):
            out[name].append(value.numpy()[0])
spec = get_robot_spec("g1")
kp, kd, scale = joint_gains(spec, constants.joint_names)
armature, effort = np.zeros(29), np.zeros(29)
import re  # noqa: E402

for actuator in articulation(spec).actuators:
    for pattern in actuator.target_names_expr:
        for i, name in enumerate(constants.joint_names):
            if re.fullmatch(pattern, name):
                armature[i], effort[i] = actuator.armature, actuator.effort_limit
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
np.savez_compressed(Path(__file__).with_name("supermimic_vectors.npz"),
                    **{k: np.stack(v) for k, v in out.items()},
                    joint_names=np.array(constants.joint_names), default_joint_pos=constants.default_joint_pos,
                    action_scale=constants.action_scale, stand_root_height=constants.stand_root_height, kp=kp, kd=kd,
                    armature=armature, effort=effort, realmimic_commit=commit, iteration=ck.get("iteration"))
print("written", commit, "scale agrees with the env term:", np.allclose(scale, constants.action_scale))
print("kp", np.round(kp, 3)); print("kd", np.round(kd, 3)); print("armature", armature); print("effort", effort)
