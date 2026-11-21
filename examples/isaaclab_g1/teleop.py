#!/usr/bin/env python3
# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Whole-body teleop of a simulated G1 from a PICO headset: Isaac Lab physics, SONIC, and the telescene app.

Use run.sh next to this file, which also sets the headset up over USB. In the headset: the robot stands and walks to
you. Right A toggles whole-body POSE, right B resets the scene, left Y starts and stops a take (with --record).
Left stick walks, right stick turns and sets the hip height, triggers and grips close the fingers.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--controller", choices=("switch", "pose"), default="switch",
                    help="switch: hybrid (head and hands, planner legs), and A toggles whole-body POSE. "
                         "pose: A toggles whole-body POSE and standing")
parser.add_argument("--operator-height", type=float, default=1.82, help="your height in metres")
parser.add_argument("--robot-height", type=float, default=1.365,
                    help="the height the robot is scaled to you by. The G1 is 1.32 m; 1.365 looked right to a 1.82 m operator")
parser.add_argument("--world-scale", type=float, default=None, help="sim metres per real metre. Overrides the heights")
parser.add_argument("--haptic-gain", type=float, default=1.0, help="strength of the controllers' vibration. 0 turns it off")
parser.add_argument("--no-follow", action="store_true", help="with tracking off, do not walk the robot to the operator")
parser.add_argument("--record", default=None, metavar="DIR", help="save takes here (left Y starts and stops one)")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 runs until Ctrl-C")
parser.add_argument("--sonic-dir", default=None, help="SONIC ONNX folder. Default $TELESCENE_SONIC_DIR or checkpoints/gear_sonic")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
args_cli.headless = True          # the headset draws the scene; the PC renders nothing
args_cli.enable_cameras = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telescene.progress import StartupProgress  # noqa: E402

# Isaac takes about a minute to start; the headset shows this while it waits for the scene.
PROGRESS = StartupProgress(["starting Isaac Lab", "loading the task", "starting the simulator", "building the scene",
                            "preparing the headset scene"],
                           history=os.path.join(os.path.expanduser("~/.cache"), "telescene_startup.json"))

SILVER, BLACK = (0.72, 0.73, 0.75), (0.05, 0.05, 0.06)


def g1_colour(prim, owner_name: str):
    """G1: black hands and feet, silver elsewhere. Anything else keeps its material's colour."""
    if "Robot" not in str(prim.GetPath()):
        return None
    return BLACK if any(k in owner_name for k in ("hand", "ankle_roll")) else SILVER


def main() -> None:
    PROGRESS.stage("loading the task")
    import g1_sonic_env
    from isaaclab_tasks.utils import resolve_task_config

    from telescene.isaac import G1_HAND_BODIES, prepare_env_cfg

    g1_sonic_env.register()
    env_cfg, _ = resolve_task_config(g1_sonic_env.TASK_ID, "", overrides=[])
    if args_cli.sonic_dir:
        env_cfg.actions.sonic.onnx_dir = os.path.abspath(args_cli.sonic_dir)
    if args_cli.controller == "pose":
        env_cfg.actions.sonic.mode = "pose"
    env_cfg.sim.device = args_cli.device
    prepare_env_cfg(env_cfg, hand_bodies=G1_HAND_BODIES)

    PROGRESS.stage("starting the simulator")
    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        from telescene import HeadsetServer, TeleopSession, scale_for_height
        from telescene.isaac import G1_HIDDEN_BODIES, IsaacLabSimulator
        from telescene.sonic.controllers import SonicPoseController, SonicSwitchController
        from telescene.sonic.core.pico_body import find_human_joints_info
        from telescene.sonic.mirror import make_mirror_view

        PROGRESS.stage("building the scene")
        env = gym.make(g1_sonic_env.TASK_ID, cfg=env_cfg)
        server = session = None
        try:
            env.reset()
            PROGRESS.stage("preparing the headset scene")
            sim = IsaacLabSimulator(env, hidden_bodies=G1_HIDDEN_BODIES, colour=g1_colour)
            print(f"[telescene] scene: {sim.stats}, {sim.nodes} draw groups, {sim.triangles} triangles, "
                  f"{len(sim.scene_blob) / 1e6:.2f} MB", flush=True)
            skeleton = find_human_joints_info()
            mirror = make_mirror_view(skeleton)
            server = HeadsetServer(sim.scene_blob, mirror_blob=mirror.scene_blob)
            PROGRESS.done()
            scale = args_cli.world_scale or scale_for_height(args_cli.robot_height, args_cli.operator_height)
            controller = (SonicSwitchController if args_cli.controller == "switch" else SonicPoseController)(
                env.unwrapped, skeleton)
            print(f"[telescene] controller {args_cli.controller}, world scale {scale:.3f}. Open the app in the headset.",
                  flush=True)
            recorder = None
            if args_cli.record:
                import time

                from telescene.recording import Recorder

                meta = {"task": g1_sonic_env.TASK_ID, "controller": args_cli.controller, "world_scale": scale,
                        "physics_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation}

                def recorder():
                    path = os.path.join(args_cli.record, time.strftime("%Y%m%d-%H%M%S") + f"_{args_cli.controller}.npz")
                    return Recorder(path, meta)

            import signal

            def stop(signum, frame):  # still save what was recorded
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, stop)
            session = TeleopSession(sim, controller, server, world_scale=scale, haptic_gain=args_cli.haptic_gain,
                                    mirror=mirror, recorder=recorder, follow=not args_cli.no_follow,
                                    log=lambda line: print(line, flush=True))
            try:
                session.run(args_cli.seconds)
            except KeyboardInterrupt:
                pass
        finally:
            if session is not None and session.saved:
                print(f"[telescene] {len(session.saved)} take(s) saved", flush=True)
            if server is not None:
                server.close()
            env.close()


if __name__ == "__main__":
    main()
