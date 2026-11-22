# The simulation (PC)

The PC runs Isaac Lab physics on the CPU and a whole-body controller (SONIC by default) on a 50 Hz loop. It renders nothing: the headset draws the scene.
It sends the scene's meshes once when the app connects, then link poses every tick.

Tested on Ubuntu 26.04 with an RTX 4080 SUPER (16 GB), driver 595, and 32 GB of RAM. Isaac Sim 6.2 asks for a recent
NVIDIA driver and Linux x86_64. A run takes about 6–8 GB of RAM.

## 1. Install Isaac Lab and telescene

```bash
scripts/install_isaaclab.sh
```

The script does four things:
1. installs `uv` if it is missing;
2. clones Isaac Lab v3.0.0 into `third_party/IsaacLab`;
3. syncs Isaac Lab with Isaac Sim 6.2 into `.venv` in this repo, from Isaac Lab's own lock file;
4. installs telescene there in editable mode, with `onnxruntime`, `pyyaml`, `mujoco` and `mink`.

The first run downloads about 60 GB into uv's cache. Set `UV_CACHE_DIR` to put it elsewhere. Isaac Sim asks you to
accept NVIDIA's licence on its first start, or set `OMNI_KIT_ACCEPT_EULA=YES`.

**Already have Isaac Lab 3.0?** Install into its Python instead: `uv pip install --python /path/to/python -e .[sonic,trackers]`,
and run the example with `PYTHON=/path/to/python`.

## 2. Download SONIC

```bash
scripts/download_sonic.sh
```

This writes `checkpoints/gear_sonic/` (encoder, decoder, planner and observation layout from
`huggingface.co/nvidia/GEAR-SONIC`, about 865 MB) and `checkpoints/human_joints_info.pkl` (SONIC's SMPL rest skeleton,
from GR00T-WholeBodyControl). It checks every file's sha256. The weights are under the NVIDIA Open Model License: read
`checkpoints/gear_sonic/LICENSE`.

Elsewhere on disk? Set `TELESCENE_SONIC_DIR` (the folder with the three ONNX files) and `TELESCENE_SMPL_SKELETON`
(the `.pkl`).

## 2b. Download the other trackers (optional)

```bash
scripts/download_trackers.sh
```

Only needed for `--controller supermimic` or `--controller tracker`. It clones GMR (the retargeter they share) into
`third_party/GMR`, and fetches SuperMimic, the follow policy and ten ONNX trackers into `checkpoints/` (about 295 MB
with GMR), checking every sha256. Each set of weights keeps its own licence; see [Controllers](#controllers) and
[THIRD_PARTY.md](../THIRD_PARTY.md). `SKIP_MIMICLITE=1` skips the one whose weights state no licence.

## 3. Run

```bash
examples/isaaclab_g1/run.sh --operator-height 1.75
```

When the log says `Open the app in the headset`, do that and pick CONNECT (see [headset.md](headset.md)). The robot is
a G1 with three-finger hands, at Isaac Lab's packing table with a 1 kg box on it. Stop with Ctrl-C.

```bash
examples/isaaclab_g1/run.sh --scene busybox                          # the BusyBox work bench; SCENE in the menu switches
examples/isaaclab_g1/run.sh --controller supermimic --follow         # SuperMimic, walking to where you stand
examples/isaaclab_g1/run.sh --controller tracker --tracker twist2    # any tracker in specs/trackers
```

With a headset on USB, `run.sh` first installs or updates the app and sets the PC's address. It never starts the app.

| Option | |
|---|---|
| `--controller C` | `switch` (default), `pose`, `supermimic` or `tracker`: see [Controllers](#controllers) |
| `--tracker NAME` | With `tracker`: `twist2`, `teleopit`, `textop`, `mimiclite`, `humanoid_gpt`, `yahmp`, `protomotions_g1`, `gigabrain_wbc`, `mosaic#gmt`, `mosaic#pico_teleop`, or a spec's path |
| `--follow` | With `supermimic`: the follow policy, so the robot also walks to you |
| `--retarget teleop` | With `supermimic`/`tracker`: legs scaled to the G1 and wrists placed relative to the pelvis. About half the hand error of GMR's own table on recorded takes |
| `--scene NAME` | `table` (default), or a BusyBox scene: `busybox` (= `full`), `palette`, `sort_two_bins`, ... See [Scenes](#scenes) |
| `--operator-height M` | Your height. The robot is scaled to you as `--robot-height` (1.365) over this |
| `--world-scale S` | Sim metres per real metre, instead of the heights |
| `--controller pose` | A toggles whole-body POSE and standing, with no hybrid mode |
| `--record DIR` | Left Y starts and stops a take, saved as `DIR/<time>_<controller>.npz` plus a `.json` |
| `--no-follow` | Do not walk the robot to you while tracking is off |
| `--haptic-gain G` | Controller vibration; 0 turns it off |
| `--seconds N` | Stop after N seconds |
| `--replay TAKE.npz` | No headset: play a take's operator input back. With `--record`, the result is saved as a new take |
| `--mirror` | No simulator: the mirror only (first argument) |

The log prints one line every few seconds. Read it like this:

```
tick p50 9.8 p99 10.4 max 10 ms against 20 | ... | tracking off | resets 0 | ... | hybrid | late plans 0
```

The tick must stay under 20 ms for 50 Hz. On the test PC it is 10–12 ms. `late plans` counts SONIC planner results
that arrived after they were due. The planner runs on a worker thread, about 70 ms per plan on the CPU.

## Controllers

| `--controller` | What | Moves the robot to you | Weights, licence |
|---|---|---|---|
| `switch` (default) | SONIC: hybrid, and A toggles whole-body POSE | Yes, with the planner | NVIDIA Open Model License |
| `pose` | SONIC: A toggles whole-body POSE and standing | No | the same |
| `supermimic` | [realmimic](https://huggingface.co/kevinhenryz/realmimic)'s SuperMimic: A toggles whole-body tracking and standing | Only with `--follow` | CC BY-NC 4.0 (non-commercial) |
| `tracker` | Any ONNX tracker that a JSON spec in `specs/trackers/` describes | No | Per tracker, below |

### SONIC
, and the default controller (`SonicSwitchController`) moves between them:

| Mode | When | What drives the robot |
|---|---|---|
| Standing | In the menu or mirror, headset off, no controllers seen yet | SONIC's planner: stands still |
| Hybrid | Default once you connect | Head and both controllers drive the upper body; the planner walks the legs to you |
| POSE | Right A toggles it | Your whole body, as SMPL from the 24 tracked joints |

Hybrid is the more exact for the hands, because controllers are measured and body joints are estimated. POSE copies
your legs and torso, but the planner does not run, so the robot does not follow your position.

### SuperMimic, the follow policy, and the ONNX trackers

These copy your whole body, joint for joint, from the 24 tracked joints. GMR (`third_party/GMR`) turns your body into G1
joints each tick; the tracker turns that reference into joint targets. They do not walk on their own: the robot goes
where copying your legs carries it. With A off, the robot stands. `telescene.trackers` holds the code.

- **SuperMimic** (`telescene/trackers/supermimic.py`): realmimic, an independent reproduction of SuperMimic
  (arXiv 2610.28480). Rebuilt from the weights alone, and checked against realmimic's own functions
  (`tests/trackers`). It reads 9 frames ahead by default (`--lookahead`, 20 ms each), so it lags you by 180 ms.
- **The follow policy** (`--follow`, `telescene/trackers/follow.py`): a small network trained for this repo that edits
  SuperMimic's goal every tick so the robot also walks to where you stand. It is trained through SuperMimic's world
  model; its weights are on this repo's GitHub release under CC BY-NC 4.0.
- **ONNX trackers** (`telescene/trackers/onnx_tracker.py`): one controller for any G1 tracker released as ONNX. A spec
  (`specs/trackers/<name>.json`) says the joint order, default pose, action scale, motors and timing the network was
  trained with, and lays out its observation term by term, each with a `file:line` reference to the project's own code.
  The env then uses the spec's motors and physics rate. Adding a tracker is a spec, and maybe one observation term.
- **GigaBrain-WBC-0.5** (`--tracker gigabrain_wbc`, `telescene/trackers/gigabrain.py`): a reference encoder and a
  Transformer action decoder that keeps a KV cache of its last 32 ticks, so it has its own small controller on top of
  the ONNX tracker's. Its inputs follow the authors' G1 deployment and evaluation code; a test checks the encoder
  input against an independent computation. It reads 9 frames ahead (180 ms), as they do with a streamed reference.

| `--tracker` | Project | Weights licence | Hand error, still (cm) |
|---|---|---|---|
| `mimiclite` | [roboparty/mimiclite](https://github.com/roboparty/mimiclite) | not stated (code GPL-3.0) | 2.5 |
| `teleopit` | [BotRunner64/Teleopit](https://github.com/BotRunner64/Teleopit) | MIT | 3.3 |
| `humanoid_gpt` | [GalaxyGeneralRobotics/Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) | Apache-2.0 | 3.5 |
| `twist2` | [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2) | MIT (in the repo) | 3.7 |
| `yahmp` | [hucebot/yahmp](https://github.com/hucebot/yahmp) | Apache-2.0 (in the repo) | 5.2 |
| `textop` | [TeleHuman/TextOp](https://github.com/TeleHuman/TextOp) | MIT (in the repo) | 7.0 |
| `mosaic#gmt`, `mosaic#pico_teleop` | [BAAI-Humanoid/MOSAIC](https://github.com/BAAI-Humanoid/MOSAIC) | Apache-2.0 | about 11 |
| `protomotions_g1` | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) | NVIDIA Open Model License | 11.9 |
| `gigabrain_wbc` | [open-gigaai/giga-brain-wbc](https://github.com/open-gigaai/giga-brain-wbc) | Apache-2.0 | not yet measured |

For comparison, SuperMimic scored 6.8 and SONIC POSE 6.6 on the same test. The last column is from our own replay of
seven recorded teleop takes (standing manipulation, crouching, reaching), five seeds each: the hand's position error
against the GMR reference while the operator stood still, with the heading aligned. It measures how each tracker
follows *this* reference in *this* simulator, not the tracker's quality in its own setting. `humanoid_gpt` needs 1 kHz
physics, which puts the tick near the 20 ms budget on our PC. `twist2` drifts in yaw, as it sees only the yaw rate.
`gigabrain_wbc` was added after that test; on one replayed BusyBox take its joint error was 0.11 rad rms, about
SuperMimic's, with no falls.

## Scenes

`--scene table` is Isaac Lab's packing table. The BusyBox scenes are a pegboard work bench (the "palette": drawers, a
slide shelf, bins, hooks, a lamp socket), a trolley on casters and a step stool, all built from primitives.
`assets/busybox/scenes.json` lists them, and with a BusyBox scene running the headset menu's SCENE tab offers them all;
picking one restarts the simulator in it, and the app reconnects.

| Group | Scenes |
|---|---|
| pieces | `full`, `palette`, `cart_stool`, `cart`, `stool`, `floor` |
| tabletop | `slide_shelf_pull`, `sort_two_bins`, `ring_hook` |
| floor | `drawer_open`, `low_shelf_fetch`, `floor_pick` |
| overhead | `bulb_fit`, `bulb_remove`, `upper_shelf_take` |
| whole_body | `drawer_foot_close`, `cart_load_park`, `heavy_box_lift` |

A task scene puts the robot where the task starts and prints the task. Its entry also carries a success check
(`success`), which the teleop does not evaluate yet. `examples/isaaclab_g1/busybox/scene.py` loads a scene into any G1
env config: `use_busybox(env_cfg, scene_entry("drawer_open"))`.

## Takes

A take records each tick's operator input (head, body, controllers, buttons) and the simulator's state, as one `.npz`
with a `.json` of metadata beside it. `--replay TAKE.npz` (`telescene.replay.ReplayServer`) plays a take's operator input back into a session with no headset.
For example, you can compare two controllers on the same motion.

## Without the real weights

For a smoke test of the whole loop, use stand-in models with SONIC's signatures. They have no weights, need no
licence, and are meaningless as control:

```bash
.venv/bin/python -m telescene.sonic.core.dummy /tmp/sonic_dummy
TELESCENE_SONIC_DIR=/tmp/sonic_dummy examples/isaaclab_g1/run.sh --seconds 20
```

## Tests

CPU only, no Isaac Sim:

```bash
pip install -e .[sonic,trackers,dev] torch
pytest
```

Tests that need the SMPL skeleton skip until `scripts/download_sonic.sh` has run, and those that need GMR or tracker
weights until `scripts/download_trackers.sh` has. Tests comparing against the upstream YAML files skip unless
GR00T-WholeBodyControl is cloned into `third_party/`.

## Your own scene

`examples/isaaclab_g1/g1_env.py` is an ordinary Isaac Lab `ManagerBasedRLEnvCfg`. Add objects to
`G1SonicSceneCfg`, and the headset draws them. It draws USD meshes and primitives with flat colours, from each prim's bound
material (`diffuseColor` or `diffuse_color_constant`); anything else is grey. Keep the two action terms as they are (`hands` then `sonic`, or `hands` then `body`
for the trackers): the controller fills their columns in that order.
