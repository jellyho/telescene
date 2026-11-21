# The simulation (PC)

The PC runs Isaac Lab physics on the CPU and SONIC on a 50 Hz loop. It renders nothing: the headset draws the scene.
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
4. installs telescene there in editable mode, with `onnxruntime` and `pyyaml`.

The first run downloads about 60 GB into uv's cache. Set `UV_CACHE_DIR` to put it elsewhere. Isaac Sim asks you to
accept NVIDIA's licence on its first start, or set `OMNI_KIT_ACCEPT_EULA=YES`.

**Already have Isaac Lab 3.0?** Install into its Python instead: `uv pip install --python /path/to/python -e .[sonic]`,
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

## 3. Run

```bash
examples/isaaclab_g1/run.sh --operator-height 1.75
```

When the log says `Open the app in the headset`, do that and pick CONNECT (see [headset.md](headset.md)). The robot is
a G1 with three-finger hands, at Isaac Lab's packing table with a 1 kg box on it. Stop with Ctrl-C.

With a headset on USB, `run.sh` first installs or updates the app and sets the PC's address. It never starts the app.

| Option | |
|---|---|
| `--operator-height M` | Your height. The robot is scaled to you as `--robot-height` (1.365) over this |
| `--world-scale S` | Sim metres per real metre, instead of the heights |
| `--controller pose` | A toggles whole-body POSE and standing, with no hybrid mode |
| `--record DIR` | Left Y starts and stops a take, saved as `DIR/<time>_<controller>.npz` plus a `.json` |
| `--no-follow` | Do not walk the robot to you while tracking is off |
| `--haptic-gain G` | Controller vibration; 0 turns it off |
| `--seconds N` | Stop after N seconds |
| `--mirror` | No simulator: the mirror only (first argument) |

The log prints one line every few seconds. Read it like this:

```
tick p50 9.8 p99 10.4 max 10 ms against 20 | ... | tracking off | resets 0 | ... | hybrid | late plans 0
```

The tick must stay under 20 ms for 50 Hz. On the test PC it is 10–12 ms. `late plans` counts SONIC planner results
that arrived after they were due. The planner runs on a worker thread, about 70 ms per plan on the CPU.

## The modes

SONIC has three encoders, and the default controller (`SonicSwitchController`) moves between them:

| Mode | When | What drives the robot |
|---|---|---|
| Standing | In the menu or mirror, headset off, no controllers seen yet | SONIC's planner: stands still |
| Hybrid | Default once you connect | Head and both controllers drive the upper body; the planner walks the legs to you |
| POSE | Right A toggles it | Your whole body, as SMPL from the 24 tracked joints |

Hybrid is the more exact for the hands, because controllers are measured and body joints are estimated. POSE copies
your legs and torso, but the planner does not run, so the robot does not follow your position.

## Takes

A take records each tick's operator input (head, body, controllers, buttons) and the simulator's state, as one `.npz`
with a `.json` of metadata beside it. `telescene.replay.ReplayServer` plays a take's operator input back into a session with no headset.
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
pip install -e .[sonic,dev] torch
pytest
```

Tests that need the SMPL skeleton skip until `scripts/download_sonic.sh` has run. Tests comparing against the upstream
YAML files skip unless GR00T-WholeBodyControl is cloned into `third_party/`.

## Your own scene

`examples/isaaclab_g1/g1_sonic_env.py` is an ordinary Isaac Lab `ManagerBasedRLEnvCfg`. Add objects to
`G1SonicSceneCfg`, and the headset draws them. It draws USD meshes and primitives with flat colours, from each prim's bound
material (`diffuseColor` or `diffuse_color_constant`); anything else is grey. Keep the two action terms (`hands` then `sonic`) as they are: the controller fills
their columns in that order.
