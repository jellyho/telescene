# Third-party code, models and assets

Nothing in this first table is in the repository: the scripts fetch it, pinned and hash-checked where they can be.

| What | From | Licence | Fetched by |
|---|---|---|---|
| SONIC weights: `model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`, `observation_config.yaml` | [huggingface.co/nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC), revision `6733128a3d8a` | NVIDIA Open Model License (copied next to the weights as `LICENSE`) | `scripts/download_sonic.sh` |
| SMPL rest skeleton `human_joints_info.pkl` | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), commit `1fa5f7fa3966` | In an Apache-2.0 repo; SMPL-X-derived data whose own terms are not stated, so it is fetched, not copied here | `scripts/download_sonic.sh` |
| GMR (General Motion Retargeting) | [YanjieZe/GMR](https://github.com/YanjieZe/GMR), commit `25f56db79ac4` | MIT | `scripts/download_trackers.sh` (sparse clone into `third_party/GMR`) |
| SuperMimic weights `realmimic_g1_86k.pt` and model card | [huggingface.co/kevinhenryz/realmimic](https://huggingface.co/kevinhenryz/realmimic), revision `56322aa74f74` | CC BY-NC 4.0, trained on BONES-SEED | `scripts/download_trackers.sh` |
| Follow policy v1 `follow_v1.pt` | This repo's GitHub release v0.2.0; trained for telescene through realmimic's world model | CC BY-NC 4.0, as the weights it was trained through | `scripts/download_trackers.sh` |
| ONNX trackers: TeleopIt, TWIST2, TextOp, MOSAIC (2), Humanoid-GPT, YAHMP, ProtoMotions, MimicLite | Pinned URLs and sha256 in `scripts/download_trackers.sh`; each project in `specs/trackers/<name>.json` (`repo`, `license_code`, `license_weights`) | Per project: MIT, Apache-2.0, NVIDIA Open Model License (ProtoMotions); MimicLite's weights state none | `scripts/download_trackers.sh` |
| Isaac Lab 3.0.0 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | `scripts/install_isaaclab.sh` |
| Isaac Sim 6.2, the G1 and table USD | NVIDIA, via pypi.nvidia.com and the Isaac Nucleus | NVIDIA Omniverse licence and asset terms | `scripts/install_isaaclab.sh`, Isaac at first run |
| OpenXR loader for Android 1.1.49 | [KhronosGroup/OpenXR-SDK](https://github.com/KhronosGroup/OpenXR-SDK), Maven Central | Apache-2.0 | `apps/pico/setup_toolchain.sh`; the build copies `libopenxr_loader.so` into the APK |
| JDK 17, Android SDK, NDK | Eclipse Temurin, Google | GPL-2.0 with Classpath Exception; Android SDK licence | `apps/pico/setup_toolchain.sh` |

## Ours, in this repository

- `specs/trackers/*.json`: written for telescene by reading each tracker's own code; each cites `file:line`. They hold
  numbers copied from those projects (joint orders, default poses, gains), not their code.
- `assets/busybox/`: the BusyBox scenes, modelled for this project from primitives (boxes, cylinders, capsules) and
  converted to USD. Their furniture is sized after real IKEA pieces (BROR trolley, BÅSINGEN stool); no IKEA model or
  mesh is used. MIT, as the rest.

## Code derived from other projects

`src/telescene/sonic/` is our own implementation of SONIC's inference. Its joint tables, observation layouts, PICO body
conversion and planner resampling follow GR00T-WholeBodyControl (Apache-2.0) and Isaac Lab's `sonic_policy.py`
(BSD-3-Clause), with file and line references in the docstrings.

`src/telescene/trackers/supermimic.py` rebuilds realmimic's network from its weights and follows realmimic's
observation layout, checked against vectors from realmimic's own functions (`tests/trackers/fixtures`, made by
`make_supermimic_vectors.py`); it runs none of its code. `GmrRetargeter` imports GMR from its clone.

Two test files are copies, kept under their own licences:

| File | From | Licence |
|---|---|---|
| `tests/sonic/sonic_reference_policy.py` | Isaac Lab `contrib/g1_terrain/agents/sonic_policy.py`, tag `interactive-demos` | BSD-3-Clause, [licenses/LICENSE-IsaacLab](licenses/LICENSE-IsaacLab) |
| `tests/sonic/sonic_reference_pico.py` | A numpy transcription of GR00T-WholeBodyControl's PICO body path | Apache-2.0, [licenses/LICENSE-GR00T-WholeBodyControl](licenses/LICENSE-GR00T-WholeBodyControl) |
