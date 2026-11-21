#!/usr/bin/env python3
# Copyright (c) 2026, telescene contributors.
# SPDX-License-Identifier: MIT

"""A mirror in the headset, with no simulator: green dots are your tracked joints, orange is the SMPL body SONIC
would be given. A quick check that the app, the network and body tracking work. Needs no GPU.

    python examples/mirror.py [--distance 1.0]
"""

from __future__ import annotations

import argparse

from telescene import HeadsetServer, TeleopSession
from telescene.sonic.core.pico_body import find_human_joints_info
from telescene.sonic.mirror import make_mirror_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl-C")
    parser.add_argument("--distance", type=float, default=1.0, help="metres to the mirror; you see yourself at twice this")
    parser.add_argument("--skeleton", default=None, help="human_joints_info.pkl; default checkpoints/ or $TELESCENE_SMPL_SKELETON")
    args = parser.parse_args()

    view = make_mirror_view(find_human_joints_info(args.skeleton), args.distance)
    server = HeadsetServer(view.scene_blob)
    print(f"[telescene] mirror at {args.distance} m. Open the app in the headset and pick MIRROR.", flush=True)
    try:
        TeleopSession(view, view, server, log=lambda line: print(line, flush=True)).run(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
