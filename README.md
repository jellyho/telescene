# telescene

Whole-body humanoid teleoperation from a VR headset: send the scene, not pixels.

The PC sends the meshes once and then only the poses of what moves. The headset draws everything itself over
passthrough, so head motion never crosses the network, and the PC renders nothing, so rendering cannot delay physics.
Body tracking and controllers go back to the PC, and a controller of your choice drives the robot: NVIDIA's SONIC,
SuperMimic, or any of nine released whole-body trackers, at a table or in the BusyBox work-bench scenes.

The repo holds two halves, which are installed separately:

| Half | What | Where | Guide |
|---|---|---|---|
| Headset | A native OpenXR app for the PICO 4 Ultra (C, no engine) | `apps/pico/` | [docs/headset.md](docs/headset.md) |
| PC | The link, the teleop loop, an Isaac Lab adapter, whole-body controllers for the G1, and a G1 example with the BusyBox scenes | `src/telescene/`, `examples/`, `specs/`, `assets/` | [docs/simulation.md](docs/simulation.md) |

## Quick start

You need a PICO 4 Ultra, ideally with two Motion Trackers for the legs, and an Ubuntu PC with an NVIDIA GPU. Both must be on the same network.

```bash
# PC: Isaac Lab 3.0 + Isaac Sim 6.2 into .venv, then SONIC's weights (about 865 MB)
scripts/install_isaaclab.sh
scripts/download_sonic.sh

# Headset app: the Android toolchain once (no root), then build and install over USB
apps/pico/setup_toolchain.sh
apps/pico/install.sh

# Start the simulation. Then, in the headset, open Telescene and pick CONNECT.
examples/isaaclab_g1/run.sh --operator-height 1.75
```

More controllers and scenes:

```bash
scripts/download_trackers.sh                                          # GMR, SuperMimic, follow policy, 9 ONNX trackers
examples/isaaclab_g1/run.sh --scene busybox                           # the BusyBox bench; the menu's SCENE tab switches
examples/isaaclab_g1/run.sh --controller supermimic --follow          # copies your whole body and walks to you
examples/isaaclab_g1/run.sh --controller tracker --tracker teleopit   # twist2, mimiclite, humanoid_gpt, ...
```

To check the app and body tracking first, without a simulator or a GPU, run `examples/isaaclab_g1/run.sh --mirror`. Then pick MIRROR in the headset.

## In the headset

The app opens on a menu. Point a controller at CONNECT or MIRROR and pull its trigger. Left X brings you back to the menu.

| Input | Does |
|---|---|
| Right A | Turns whole-body POSE on and off. Off is hybrid: your head and hands drive the upper body, and the legs walk to you. With the other controllers, off is standing |
| Right B | Resets the scene |
| Left Y | Starts and stops a take (with `--record DIR`): one long buzz to start, two short to stop |
| Left stick | Walks |
| Right stick | Turns; up and down set the hip height |
| Triggers, grips | Close the fingers |

The robot follows you while POSE is off. Steering with a stick pauses that until you walk 0.3 m. POSE turns off when
the headset comes off, after a reset and when the robot falls. The controllers vibrate when the hands touch something.

## Use it with your own robot or simulator

`TeleopSession` knows no simulator and no controller. A controller has three methods. `act` gets a `ControlInput`
(head, 24-joint body, controllers, triggers, sticks) and returns one action for the simulator:

```python
class MyController:
    def reset(self): ...
    def act(self, control): ...
    def status(self): return ""
```

For Isaac Lab, `telescene.isaac.IsaacLabSimulator` wraps any `ManagerBasedEnv`:

```python
from telescene import HeadsetServer, TeleopSession, scale_for_height
from telescene.isaac import G1_HIDDEN_BODIES, IsaacLabSimulator, prepare_env_cfg

prepare_env_cfg(env_cfg)
env = gym.make(task, cfg=env_cfg)
env.reset()
sim = IsaacLabSimulator(env, hidden_bodies=G1_HIDDEN_BODIES)
TeleopSession(sim, MyController(env), HeadsetServer(sim.scene_blob), world_scale=scale_for_height(1.365, 1.82)).run()
```

`examples/isaaclab_g1/teleop.py` is the full version. `telescene.mirror.MirrorView` is the smallest `Simulator` and
`Controller` in the repo.

## Layout

| Path | What |
|---|---|
| `apps/pico/` | The headset app: `main.c` (OpenXR, GLES 3, body tracking, networking), the build and install scripts |
| `src/telescene/wire.py`, `server.py` | Scene and pose formats, numpy only; the PC end of the link |
| `src/telescene/session.py` | The teleop loop, for any simulator and controller |
| `src/telescene/isaac/` | Isaac Lab: the scene sent to the headset, and the simulator adapter |
| `src/telescene/sonic/` | SONIC (NVIDIA GEAR-SONIC) on G1: inference in numpy / torch / onnxruntime, an Isaac Lab action term and the headset controllers |
| `src/telescene/trackers/` | Joint-for-joint whole-body trackers: SuperMimic with GMR retargeting, the follow policy, and a spec-driven ONNX tracker |
| `specs/trackers/` | One JSON per released G1 tracker: observation layout, motors, timing, sources and licences |
| `assets/busybox/` | The BusyBox scenes (USD) and their manifest, `scenes.json` |
| `src/telescene/recording.py`, `replay.py` | Takes: record a session, and play a take's operator input back with no headset |
| `examples/isaaclab_g1/` | A G1 at a table or in a BusyBox scene, driven by any of the controllers |
| `examples/mirror.py` | Your tracked body in a mirror, no simulator |
| `tests/` | CPU only: `pip install -e .[sonic,trackers,dev] && pytest` |

## Limits

- Tested on one PICO 4 Ultra with two Motion Trackers. Whole-body tracking needs `XR_BD_body_tracking`.
- The headset draws USD `Mesh`, `Cube`, `Sphere`, `Cylinder` and `Capsule` prims, with flat colours. It draws no textures, transparency,
  deformables or particles. Objects added after startup need the app to reconnect.
- In POSE, SONIC's planner does not run, so the robot moves only as far as copying your legs carries it, and it can drift
  from where you stand. To bring it back, press A, let it walk to you, and press A again.

## Licence

MIT, see [LICENSE](LICENSE): the code, the tracker specs and the BusyBox scenes. Some things are not covered by it and
are not in this repo:
- SONIC's weights, under the NVIDIA Open Model License, which `scripts/download_sonic.sh` downloads;
- the tracker weights, each under its own project's terms, which `scripts/download_trackers.sh` downloads. SuperMimic's
  (realmimic) and the follow policy's are CC BY-NC 4.0: **non-commercial use only**;
- GMR (MIT), cloned into `third_party/`;
- Isaac Sim and its assets, under NVIDIA's licences;
- the Khronos OpenXR loader (Apache-2.0), which the build bundles into the APK.

See [THIRD_PARTY.md](THIRD_PARTY.md).
