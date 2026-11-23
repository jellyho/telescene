#!/usr/bin/env python3
# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Render a take (teleop.py --record) to an mp4 with Isaac's RTX renderer, from a fixed camera.

Nothing is simulated: the recorded robot, scene and objects are written back frame by frame. The take's .json next to
the .npz names the scene and the controller, so the scene is rebuilt as it was.

    examples/isaaclab_g1/render_take.py TAKE.npz [--out VIDEO.mp4] [--eye X Y Z --target X Y Z] [--start S --seconds S]

Needs ffmpeg on the PATH and an RTX GPU. Use a disk directory for TMPDIR if /tmp is in RAM: Isaac writes temporary
files there.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("take", help="a .npz from teleop.py --record")
parser.add_argument("--out", default=None, help="default: the take's name with .mp4")
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument("--eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="camera position, scene frame. Default: beside the robot's mean position, looking at it")
parser.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="what the camera looks at")
parser.add_argument("--start", type=float, default=0.0, help="seconds into the take")
parser.add_argument("--seconds", type=float, default=0.0, help="how much to render; 0 renders to the end")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
TAKE = Path(args_cli.take).expanduser()
META = json.loads(TAKE.with_suffix(".json").read_text())

sys.path.insert(0, HERE)


def main() -> None:
    import g1_env
    import numpy as np
    from isaaclab_tasks.utils import resolve_task_config

    from telescene.isaac import prepare_env_cfg

    data = np.load(TAKE)
    times = data["time"]
    g1_env.register()
    sonic = META.get("controller") in ("switch", "pose")
    task = g1_env.TASK_ID if sonic else g1_env.TRACKER_TASK_ID
    env_cfg, _ = resolve_task_config(task, "", overrides=[])
    scene = META.get("scene")
    if isinstance(scene, dict):
        import busybox

        busybox.use_busybox(env_cfg, scene)
    elif "rigid/box/root" not in data.files:
        env_cfg.scene.box = None
    env_cfg.sim.device = args_cli.device
    prepare_env_cfg(env_cfg)

    import isaaclab.sim as sim_utils
    from g1_paint import paint_when_spawned
    from isaaclab.sensors import CameraCfg

    paint_when_spawned(env_cfg.scene.robot)  # black hands, feet and head, as on the real robot

    width, height = args_cli.size
    env_cfg.scene.render_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCam", update_period=0.0, width=width, height=height, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)))

    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym
        import torch

        env = gym.make(task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped
        sc, sim = unwrapped.scene, unwrapped.sim
        origin = sc.env_origins[0].cpu()

        start = int(np.searchsorted(times, times[0] + args_cli.start))
        end_time = times[-1] if not args_cli.seconds else min(times[-1], times[start] + args_cli.seconds)
        frames = [int(np.searchsorted(times, t)) for t in np.arange(times[start], end_time, 1.0 / args_cli.fps)]

        robot_xy = data["art/robot/root"][frames, :2].mean(axis=0)
        target = np.array(args_cli.target if args_cli.target else (*robot_xy, 0.6))
        eye = np.array(args_cli.eye if args_cli.eye else (*(robot_xy + (2.1, -2.1)), 1.7))
        camera = sc["render_cam"]
        camera.set_world_poses_from_view(torch.tensor(eye, dtype=torch.float32).unsqueeze(0) + origin,
                                         torch.tensor(target, dtype=torch.float32).unsqueeze(0) + origin)

        assets = [(sc.articulations[k.split("/")[1]], k) for k in data.files
                  if k.startswith("art/") and k.endswith("/root") and k.split("/")[1] in sc.articulations]
        rigid = [(sc.rigid_objects[k.split("/")[1]], k) for k in data.files
                 if k.startswith("rigid/") and k.split("/")[1] in sc.rigid_objects]
        dt = sim.get_physics_dt()

        def root_state(key: str, i: int):
            root = torch.as_tensor(data[key][i], dtype=torch.float32).unsqueeze(0).clone()
            root[:, :3] += origin
            root[:, 7:] = 0.0
            return root

        def show(i: int) -> None:
            for asset, key in assets:
                asset.write_root_state_to_sim(root_state(key, i))
                joints = torch.as_tensor(data[key.replace("/root", "/joints")][i], dtype=torch.float32).unsqueeze(0)
                asset.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            for asset, key in rigid:
                asset.write_root_state_to_sim(root_state(key, i))
            sim.forward()
            sim.render()
            sc.update(dt)

        out = Path(args_cli.out or TAKE.with_suffix(".mp4"))
        for _ in range(30):  # RTX settles over the first frames
            show(frames[0])
        encoder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(args_cli.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            stdin=subprocess.PIPE)
        try:
            for n, i in enumerate(frames):
                show(i)
                rgb = camera.data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
                encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
                if n % 150 == 0:
                    print(f"[render] {n}/{len(frames)} frames", flush=True)
        finally:
            encoder.stdin.close()
            encoder.wait()
            env.close()
        print(f"[render] {len(frames)} frames at {args_cli.fps:g} fps to {out}", flush=True)


if __name__ == "__main__":
    main()
