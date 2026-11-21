#!/usr/bin/env bash
# Download SONIC (NVIDIA GEAR-SONIC) for the G1 example, and check every file's sha256.
#
#   scripts/download_sonic.sh [DIR]      default: checkpoints/ in this repo
#
# Writes DIR/gear_sonic/{model_encoder,model_decoder,planner_sonic}.onnx, observation_config.yaml and the licence
# (about 865 MB), and DIR/human_joints_info.pkl, SONIC's SMPL rest skeleton (4 kB).
#
# The weights are under the NVIDIA Open Model License, not this repo's MIT licence: read
# DIR/gear_sonic/LICENSE. The skeleton comes from NVlabs/GR00T-WholeBodyControl (Apache-2.0).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIR=${1:-$REPO/checkpoints}
HF_REV=6733128a3d8a523b1418b06bca3cdf61c8b0987f
GR00T_REV=1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b
mkdir -p "$DIR/gear_sonic"

fetch() {  # url, file, sha256
    if [[ -f $2 ]] && echo "$3  $2" | sha256sum -c --status; then echo "have $2"; return; fi
    echo "get  $2"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --status || { echo "sha256 mismatch for $1" >&2; rm -f "$2.part"; exit 1; }
    mv "$2.part" "$2"
}

HF=https://huggingface.co/nvidia/GEAR-SONIC/resolve/$HF_REV
fetch "$HF/LICENSE" "$DIR/gear_sonic/LICENSE" 24ab66be50d1aca4fc5e029ef76ce4ceaac6557ea21665caf4b140695a76ffee
fetch "$HF/observation_config.yaml" "$DIR/gear_sonic/observation_config.yaml" 466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b
fetch "$HF/model_encoder.onnx" "$DIR/gear_sonic/model_encoder.onnx" 013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3
fetch "$HF/model_decoder.onnx" "$DIR/gear_sonic/model_decoder.onnx" c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed
fetch "$HF/planner_sonic.onnx" "$DIR/gear_sonic/planner_sonic.onnx" 39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea
# A Git LFS file: GitHub's media host serves the content.
fetch "https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/$GR00T_REV/gear_sonic/data/human/human_joints_info.pkl" \
    "$DIR/human_joints_info.pkl" 4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975
echo "done: $DIR"
