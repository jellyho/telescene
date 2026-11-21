# Third-party code, models and assets

Nothing in this first table is in the repository: the scripts fetch it, pinned and hash-checked where they can be.

| What | From | Licence | Fetched by |
|---|---|---|---|
| SONIC weights: `model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`, `observation_config.yaml` | [huggingface.co/nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC), revision `6733128a3d8a` | NVIDIA Open Model License (copied next to the weights as `LICENSE`) | `scripts/download_sonic.sh` |
| SMPL rest skeleton `human_joints_info.pkl` | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), commit `1fa5f7fa3966` | In an Apache-2.0 repo; SMPL-X-derived data whose own terms are not stated, so it is fetched, not copied here | `scripts/download_sonic.sh` |
| Isaac Lab 3.0.0 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | `scripts/install_isaaclab.sh` |
| Isaac Sim 6.2, the G1 and table USD | NVIDIA, via pypi.nvidia.com and the Isaac Nucleus | NVIDIA Omniverse licence and asset terms | `scripts/install_isaaclab.sh`, Isaac at first run |
| OpenXR loader for Android 1.1.49 | [KhronosGroup/OpenXR-SDK](https://github.com/KhronosGroup/OpenXR-SDK), Maven Central | Apache-2.0 | `apps/pico/setup_toolchain.sh`; the build copies `libopenxr_loader.so` into the APK |
| JDK 17, Android SDK, NDK | Eclipse Temurin, Google | GPL-2.0 with Classpath Exception; Android SDK licence | `apps/pico/setup_toolchain.sh` |

## Code derived from other projects

`src/telescene/sonic/` is our own implementation of SONIC's inference. Its joint tables, observation layouts, PICO body
conversion and planner resampling follow GR00T-WholeBodyControl (Apache-2.0) and Isaac Lab's `sonic_policy.py`
(BSD-3-Clause), with file and line references in the docstrings.

Two test files are copies, kept under their own licences:

| File | From | Licence |
|---|---|---|
| `tests/sonic/sonic_reference_policy.py` | Isaac Lab `contrib/g1_terrain/agents/sonic_policy.py`, tag `interactive-demos` | BSD-3-Clause, [licenses/LICENSE-IsaacLab](licenses/LICENSE-IsaacLab) |
| `tests/sonic/sonic_reference_pico.py` | A numpy transcription of GR00T-WholeBodyControl's PICO body path | Apache-2.0, [licenses/LICENSE-GR00T-WholeBodyControl](licenses/LICENSE-GR00T-WholeBodyControl) |
