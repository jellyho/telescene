#!/usr/bin/env bash
# Install Isaac Lab 3.0.0 with Isaac Sim 6.2 into .venv in this repo, then telescene with SONIC, in editable mode.
# No root needed. The first run downloads tens of GB into uv's cache (about 60 GB; set UV_CACHE_DIR to move it).
#
#   scripts/install_isaaclab.sh [--isaaclab-dir DIR] [--venv DIR]
#
# Isaac Sim asks you to accept the NVIDIA Omniverse licence on its first start (or set OMNI_KIT_ACCEPT_EULA=YES).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ISAACLAB_DIR=$REPO/third_party/IsaacLab; VENV=$REPO/.venv
while (($#)); do case $1 in
    --isaaclab-dir) ISAACLAB_DIR=$(realpath -m "$2"); shift 2 ;;
    --venv) VENV=$(realpath -m "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
esac; done
[[ $(uname -sm) == "Linux x86_64" ]] || { echo "Isaac Sim 6.2 needs Linux x86_64" >&2; exit 1; }
command -v nvidia-smi >/dev/null || echo "warning: no nvidia-smi; Isaac Sim needs an NVIDIA GPU and driver" >&2

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d $ISAACLAB_DIR/.git ]]; then
    git clone --depth 1 --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
(cd "$ISAACLAB_DIR" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --inexact --extra isaacsim)
uv pip install --python "$VENV/bin/python" -e "$REPO[sonic,trackers]"
"$VENV/bin/python" -c "import isaaclab, telescene, onnxruntime; print('isaaclab', isaaclab.__version__ if hasattr(isaaclab, '__version__') else '', 'telescene ok')"
echo "done. Next: scripts/download_sonic.sh, then examples/isaaclab_g1/run.sh"
