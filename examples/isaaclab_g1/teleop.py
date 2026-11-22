#!/usr/bin/env python3
# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""Whole-body teleop of a simulated G1 from a PICO headset: Isaac Lab physics, a whole-body controller, and the app.

Use run.sh next to this file, which also sets the headset up over USB. In the headset: the robot stands and walks to
you. Right A toggles whole-body tracking, right B resets the scene, left Y starts and stops a take (with --record).
Left stick walks, right stick turns and sets the hip height, triggers and grips close the fingers. With a BusyBox scene,
the menu's SCENE tab switches scenes.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPECS = os.path.join(REPO, "specs", "trackers")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--controller", choices=("switch", "pose", "supermimic", "tracker"), default="switch",
                    help="switch: SONIC hybrid (head and hands, planner legs), and A toggles whole-body POSE. "
                         "pose: A toggles SONIC's whole-body POSE and standing. "
                         "supermimic: A toggles SuperMimic's whole-body tracking and standing. "
                         "tracker: the same, with the ONNX tracker --tracker names")
parser.add_argument("--tracker", default=None, metavar="NAME",
                    help="tracker: a spec in specs/trackers (twist2, teleopit, textop, mimiclite, humanoid_gpt, yahmp, "
                         "protomotions_g1, gigabrain_wbc, mosaic#gmt, mosaic#pico_teleop), or a path to one")
parser.add_argument("--follow", nargs="?", const="v1", default=None, metavar="CHECKPOINT",
                    help="supermimic: the follow policy, so the robot also walks to where you stand. Alone, v1")
parser.add_argument("--retarget", choices=("gmr", "teleop"), default="gmr",
                    help="supermimic/tracker: gmr is GMR's table as it is. teleop scales the legs to the G1 and puts the "
                         "wrists where yours are relative to the pelvis: about half the hand error on recorded takes")
parser.add_argument("--lookahead", type=int, default=9, help="supermimic: real future frames, each 20 ms of delay")
parser.add_argument("--scene", default="table",
                    help="table (Isaac Lab's packing table), or a BusyBox scene from assets/busybox/scenes.json: full, "
                         "palette, sort_two_bins, drawer_open, ... ('busybox' is full)")
parser.add_argument("--scene-choice-file", default=None,
                    help="BusyBox: where to write the scene picked in the headset menu. The run then ends, and run.sh "
                         "starts again in that scene")
parser.add_argument("--no-box", action="store_true", help="leave out the 1 kg box")
parser.add_argument("--hand-torsion-patch", type=float, default=0.01, metavar="M",
                    help="PhysX torsional patch radius on the hands and the objects, so a pinched object does not spin. 0: off")
parser.add_argument("--operator-height", type=float, default=1.82, help="your height in metres")
parser.add_argument("--robot-height", type=float, default=1.365,
                    help="the height the robot is scaled to you by. The G1 is 1.32 m; 1.365 looked right to a 1.82 m operator")
parser.add_argument("--world-scale", type=float, default=None, help="sim metres per real metre. Overrides the heights")
parser.add_argument("--haptic-gain", type=float, default=1.0, help="strength of the controllers' vibration. 0 turns it off")
parser.add_argument("--no-follow", action="store_true", help="with tracking off, do not walk the robot to the operator")
parser.add_argument("--record", default=None, metavar="DIR", help="save takes here (left Y starts and stops one)")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 runs until Ctrl-C")
parser.add_argument("--replay", default=None, metavar="TAKE.npz",
                    help="no headset: play a take's operator input back (head, body, controllers, buttons). With --record, "
                         "the result is saved as a new take")
parser.add_argument("--sonic-dir", default=None, help="SONIC ONNX folder. Default $TELESCENE_SONIC_DIR or checkpoints/gear_sonic")
add_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
args_cli.headless = True          # the headset draws the scene; the PC renders nothing
args_cli.enable_cameras = False
if args_cli.controller == "tracker" and not args_cli.tracker:
    parser.error("--controller tracker needs --tracker")
if args_cli.follow and args_cli.controller != "supermimic":
    parser.error("--follow works with --controller supermimic")
if args_cli.scene == "busybox":
    args_cli.scene = "full"


def spec_path(name: str) -> str:
    """``twist2`` -> specs/trackers/twist2.json; ``mosaic#gmt`` -> specs/trackers/mosaic.json#gmt; a path stays."""
    file, _, variant = name.partition("#")
    if not os.path.isfile(file):
        file = os.path.join(SPECS, file if file.endswith(".json") else file + ".json")
    if not os.path.isfile(file):
        parser.error(f"no tracker spec '{name}': see {SPECS}")
    return os.path.abspath(file) + (f"#{variant}" if variant else "")


SPEC = spec_path(args_cli.tracker) if args_cli.controller == "tracker" else None

sys.path.insert(0, HERE)

from telescene.progress import StartupProgress  # noqa: E402

# Isaac takes about a minute to start; the headset shows this while it waits for the scene.
PROGRESS = StartupProgress(["starting Isaac Lab", "loading the task", "starting the simulator", "building the scene",
                            "preparing the headset scene"],
                           history=os.path.join(os.path.expanduser("~/.cache"), "telescene_startup.json"))

SILVER, BLACK = (0.72, 0.73, 0.75), (0.05, 0.05, 0.06)


def colour(prim, owner_name: str):
    """G1: black hands and feet, silver elsewhere. BusyBox: the colour its material is named after. Anything else keeps
    its material's colour."""
    if "Robot" in str(prim.GetPath()):
        return BLACK if any(k in owner_name for k in ("hand", "ankle_roll")) else SILVER
    if "/BusyBox" in str(prim.GetPath()):
        from busybox import busybox_colour

        return busybox_colour(prim)
    return None


def make_controller(env, skeleton):
    if args_cli.controller in ("switch", "pose"):
        from telescene.sonic.controllers import SonicPoseController, SonicSwitchController

        return (SonicSwitchController if args_cli.controller == "switch" else SonicPoseController)(env, skeleton)

    from telescene.sonic.inputs import hand_columns
    from telescene.trackers import supermimic as sm

    if not sm.GMR_ROOT.exists():
        raise SystemExit(f"no GMR at {sm.GMR_ROOT}: run scripts/download_trackers.sh")
    retarget = sm.GmrRetargeter(sm.GMR_ROOT, operator_height=args_cli.operator_height, preset=args_cli.retarget)

    def hands(triggers):
        return hand_columns(*triggers)

    if args_cli.controller == "tracker":
        from telescene.trackers import onnx_tracker

        controller = onnx_tracker.from_env(env, SPEC, retarget=retarget, hands=hands)
        print(f"[telescene] tracker {controller.spec.slug}: {controller.spec.onnx}", flush=True)
        return controller
    if not sm.CHECKPOINT.exists():
        raise SystemExit(f"no SuperMimic weights at {sm.CHECKPOINT}: run scripts/download_trackers.sh")
    follow = None
    if args_cli.follow:
        follow = sm.FOLLOW_CHECKPOINT if args_cli.follow == "v1" else args_cli.follow
    return sm.from_env(env, sm.CHECKPOINT, retarget=retarget, hands=hands, lookahead=args_cli.lookahead,
                       follow_checkpoint=follow)


def main() -> None:
    PROGRESS.stage("loading the task")
    import g1_env
    from isaaclab_tasks.utils import resolve_task_config

    from telescene.isaac import G1_HAND_BODIES, prepare_env_cfg

    g1_env.register()
    task = g1_env.TASK_ID if args_cli.controller in ("switch", "pose") else g1_env.TRACKER_TASK_ID
    env_cfg, _ = resolve_task_config(task, "", overrides=[])
    if args_cli.controller in ("switch", "pose"):
        if args_cli.sonic_dir:
            env_cfg.actions.sonic.onnx_dir = os.path.abspath(args_cli.sonic_dir)
        if args_cli.controller == "pose":
            env_cfg.actions.sonic.mode = "pose"
    if SPEC:
        g1_env.use_tracker_motors(env_cfg, SPEC)
    scene_names, scene_labels, scene_groups, entry = [], [], None, None
    if args_cli.scene != "table":
        import busybox

        entry = busybox.scene_entry(args_cli.scene)
        busybox.use_busybox(env_cfg, entry, pick_box=False if args_cli.no_box else None)
        if args_cli.scene_choice_file:
            from telescene.wire import MAX_SCENES

            entries = busybox.scenes()[:MAX_SCENES]
            scene_names = [e["name"] for e in entries]
            scene_labels = [e.get("label_en") or e["name"] for e in entries]
            groups = [e.get("group", "") or "other" for e in entries]
            scene_groups = groups if len(set(groups)) > 1 else None
    elif args_cli.no_box:
        env_cfg.scene.box = None
    g1_env.add_torsion_patch(env_cfg, args_cli.hand_torsion_patch)
    env_cfg.sim.device = args_cli.device
    prepare_env_cfg(env_cfg, hand_bodies=G1_HAND_BODIES)

    PROGRESS.stage("starting the simulator")
    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        from telescene import HeadsetServer, TeleopSession, scale_for_height
        from telescene.isaac import G1_HIDDEN_BODIES, IsaacLabSimulator
        from telescene.sonic.core.pico_body import find_human_joints_info
        from telescene.sonic.mirror import make_mirror_view

        PROGRESS.stage("building the scene")
        env = gym.make(task, cfg=env_cfg)
        server = session = None
        try:
            env.reset()
            PROGRESS.stage("preparing the headset scene")
            sim = IsaacLabSimulator(env, hidden_bodies=G1_HIDDEN_BODIES, colour=colour)
            print(f"[telescene] scene: {sim.stats}, {sim.nodes} draw groups, {sim.triangles} triangles, "
                  f"{len(sim.scene_blob) / 1e6:.2f} MB", flush=True)
            skeleton = find_human_joints_info()
            mirror = make_mirror_view(skeleton)
            if args_cli.replay:
                from telescene.replay import ReplayServer

                server = ReplayServer(args_cli.replay, record=bool(args_cli.record))
                args_cli.seconds = args_cli.seconds or len(server) / 50.0 + 0.5
                print(f"[telescene] replaying {len(server)} ticks of {args_cli.replay}", flush=True)
            else:
                current = scene_names.index(args_cli.scene) if args_cli.scene in scene_names else 255
                server = HeadsetServer(sim.scene_blob, mirror_blob=mirror.scene_blob, scenes=scene_labels,
                                       scene_groups=scene_groups, current_scene=current)
            PROGRESS.done()
            scale = args_cli.world_scale or scale_for_height(args_cli.robot_height, args_cli.operator_height)
            controller = make_controller(env.unwrapped, skeleton)
            print(f"[telescene] controller {args_cli.controller}, scene {args_cli.scene}, world scale {scale:.3f}. "
                  "Open the app in the headset.", flush=True)
            recorder = None
            if args_cli.record:
                import time

                from telescene.recording import Recorder

                tag = controller.spec.slug if SPEC else args_cli.controller + ("_follow" if args_cli.follow else "")
                meta = {"task": task, "controller": args_cli.controller, "tracker": SPEC, "follow": args_cli.follow,
                        "retarget": args_cli.retarget if args_cli.controller in ("supermimic", "tracker") else None,
                        "scene": entry or "table", "world_scale": scale, "physics_dt": env_cfg.sim.dt,
                        "decimation": env_cfg.decimation}

                def recorder():
                    path = os.path.join(args_cli.record, time.strftime("%Y%m%d-%H%M%S") + f"_{tag}"
                                        + ("_replay" if args_cli.replay else "") + ".npz")
                    return Recorder(path, {**meta, "replay_of": args_cli.replay})

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
            picked = getattr(server, "requested_scene", None)
            if picked is not None and args_cli.scene_choice_file and picked < len(scene_names):
                with open(args_cli.scene_choice_file, "w") as f:
                    f.write(scene_names[picked])
        finally:
            if session is not None and session.saved:
                print(f"[telescene] {len(session.saved)} take(s) saved", flush=True)
            if server is not None:
                server.close()
            env.close()


if __name__ == "__main__":
    main()
