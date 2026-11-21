#!/usr/bin/env bash
# Start the G1 teleop server. Then open "Telescene" in the headset and pick CONNECT; the app finds this PC.
#
#   examples/isaaclab_g1/run.sh [--operator-height 1.82] [--record takes/] [--controller pose] [--seconds N]
#   examples/isaaclab_g1/run.sh --mirror        no simulator: see your tracked body in a mirror
#
# Uses .venv in the repo (scripts/install_isaaclab.sh), or $PYTHON. With a headset on USB, the app is installed
# or updated first and told this PC's address; the app itself is never started from here.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
XRT=${TELESCENE_ANDROID:-${XRT:-$HOME/.local/share/telescene-android}}
ADB=${ADB:-$(command -v adb || echo "$XRT/sdk/platform-tools/adb")}
if [[ -x $ADB || -n $(command -v "$ADB" 2>/dev/null) ]] && "$ADB" get-state >/dev/null 2>&1; then
    APK=$REPO/apps/pico/build/telescene.apk
    if [[ -f $APK || -d $XRT/sdk/ndk ]]; then "$REPO/apps/pico/install.sh"; fi
fi
if [[ ${1:-} == --mirror ]]; then
    shift; exec "$PYTHON" "$REPO/examples/mirror.py" "$@"
fi
# Isaac leaves about 86 MB of temporary files per run; keep them off a RAM-backed /tmp if you have one.
export TMPDIR=${TMPDIR:-/tmp}
exec "$PYTHON" "$REPO/examples/isaaclab_g1/teleop.py" "$@"
