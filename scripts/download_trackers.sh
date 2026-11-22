#!/usr/bin/env bash
# Download the whole-body trackers: GMR (the retargeter they share), SuperMimic, the follow policy and the ONNX trackers
# that specs/trackers/ describes. Every file's sha256 is checked.
#
#   scripts/download_trackers.sh [DIR]      default: checkpoints/ in this repo; GMR goes to third_party/GMR
#
# Writes:
#   third_party/GMR                     GMR (github.com/YanjieZe/GMR, MIT), only its code and its G1 model (about 55 MB)
#   DIR/supermimic/realmimic_g1_86k.pt  realmimic's SuperMimic policy (110 MB), CC BY-NC 4.0, with its model card
#   DIR/supermimic/follow_v1.pt         the follow policy (1 MB), from this repo's release, CC BY-NC 4.0
#   DIR/trackers/<slug>.onnx            ten ONNX trackers (about 135 MB), each under its own project's licence
#
# None of these weights are under this repo's MIT licence: see THIRD_PARTY.md. realmimic and the follow policy are for
# non-commercial use only.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIR=${1:-$REPO/checkpoints}
GMR_DIR=${TELESCENE_GMR:-$REPO/third_party/GMR}
GMR_REV=25f56db79ac411258b377b6d4ccba0c8ff3372ac
REALMIMIC_REV=56322aa74f744a462ae416e03140f220e11a4865
RELEASE=https://github.com/jellyho/telescene/releases/download/v0.2.0
mkdir -p "$DIR/supermimic" "$DIR/trackers"

fetch() {  # url, file, sha256
    if [[ -f $2 ]] && echo "$3  $2" | sha256sum -c --status; then echo "have $2"; return; fi
    echo "get  $2"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --status || { echo "sha256 mismatch for $1" >&2; rm -f "$2.part"; exit 1; }
    mv "$2.part" "$2"
}

# GMR: a sparse, blobless clone, pinned. Its IK needs mink (pip install -e .[trackers]).
if [[ ! -d $GMR_DIR/.git ]]; then
    echo "get  $GMR_DIR"
    git clone -q --filter=blob:none --no-checkout https://github.com/YanjieZe/GMR "$GMR_DIR"
    git -C "$GMR_DIR" sparse-checkout set general_motion_retargeting assets/unitree_g1
fi
git -C "$GMR_DIR" -c advice.detachedHead=false checkout -q "$GMR_REV"
echo "have $GMR_DIR at ${GMR_REV:0:7}"

HF=https://huggingface.co/kevinhenryz/realmimic/resolve/$REALMIMIC_REV
fetch "$HF/README.md" "$DIR/supermimic/realmimic_README.md" c29d8721bae5dee34977c7a2485436955dcd15faa1d9f38bc3d0443bd96cbc11
fetch "$HF/realmimic_g1_86k.pt" "$DIR/supermimic/realmimic_g1_86k.pt" b38d474df238198e8cc28e0e6286ae69b88d91bdd10d2abf8f73df35afea7ae2
fetch "$RELEASE/follow_v1.pt" "$DIR/supermimic/follow_v1.pt" 930065bdc2b62fb9f6a83350b9966fb3b4430b3ecf6fe90e9c74f25529fced8b

# The ONNX trackers, at the commits the specs were written against. Licences: specs/trackers/<name>.json.
T=$DIR/trackers
fetch https://huggingface.co/12e21/Teleopit-models/resolve/94cf996444fea6894b87c28e86606cd4c2f1408f/checkpoints/track_g1.onnx \
    "$T/teleopit.onnx" 1ebd341d9193e1c49a986450f6043ba1a9473ad46636ce0bcb1c7755c856e0de
fetch https://github.com/amazon-far/TWIST2/raw/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/ckpts/twist2_1017_20k.onnx \
    "$T/twist2.onnx" 2d1fb3a31e4e967f70ecfefc3ad1e7b2ac491677068b89f60b565a94e7735061
fetch https://raw.githubusercontent.com/TeleHuman/TextOp/ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22/TextOpDeploy/src/textop_ctrl/models/policy.onnx \
    "$T/textop.onnx" 8df6371cce677a6ef8fd7124d082fbf98df11735b9e1aa5808f1bf9d3152b582
fetch https://huggingface.co/BAAI-Humanoid/MOSAIC_Model/resolve/main/gmt.onnx \
    "$T/mosaic_gmt.onnx" d6e124126a1b3e4fd3cc75e820fc0a4e170aa66ab297290fa1a3f1495fa07f3d
fetch https://huggingface.co/BAAI-Humanoid/MOSAIC_Model/resolve/main/pico_teleop.onnx \
    "$T/mosaic_pico_teleop.onnx" b18d873b701559fff42d81e1a1755b618de14a7cc41a538c9998bbebe61cef4f
fetch https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/raw/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv264.onnx \
    "$T/humanoid_gpt.onnx" bbaac94ba08d30bd96879e0c5878d60a38180f1cb819a9336cc6b56d49f333f9
fetch https://raw.githubusercontent.com/hucebot/yahmp/2111a998663e9f787b3ac36e8f8a7c17745ca83f/assets/models/g1_yahmp.onnx \
    "$T/yahmp.onnx" 59f02e2aa6408a006f6af672b78b71858f32c234756e3fb47a98c1e1a77861f6
# A Git LFS file: GitHub's media host serves the content. NVIDIA Open Model License.
fetch https://media.githubusercontent.com/media/NVlabs/ProtoMotions/6a1887f65b8a619c77198431d0ec00d5c88bea2c/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    "$T/protomotions_g1.onnx" fbbcf33a3fbcc939a5765161947f390b7911800011ca535ab896acf9f40e12a3
# GigaBrain-WBC-0.5, Apache-2.0: a reference encoder and an action decoder (35 MB).
GIGA=https://huggingface.co/open-gigaai/GigaBrain-WBC-0.5/resolve/2aba857fc8bb0415aee2433d292c5b7cd8055cdd/checkpoints/unitree_g1_model/exported
fetch "$GIGA/policy_encoder.onnx" "$T/gigabrain_wbc_encoder.onnx" 68a31c2dfe720649e0113ec45f6e6f77166ea6bee1cf5f2f5bb1846ade42e378
fetch "$GIGA/policy_decoder.onnx" "$T/gigabrain_wbc.onnx" 1ce6a68754a748c00d8f2104b9552f4af384a749d33af5363211f98e8e9185c0
# On Google Drive, with no stated licence (its code is GPL-3.0). Skip it with SKIP_MIMICLITE=1.
if [[ -z ${SKIP_MIMICLITE:-} ]]; then
    fetch "https://drive.usercontent.google.com/download?id=1TR5L_eAMzT8OWg4Afes4OpTU8YjS0HyA&export=download&confirm=t" \
        "$T/mimiclite.onnx" f647b2674a593cdc9e3b661f653b3f0f120238523a79f3ec7765bc480af22f48
fi
echo "done: $DIR, $GMR_DIR"
