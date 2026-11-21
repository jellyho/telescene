#!/usr/bin/env bash
# Start the G1 teleop server. Then open "Telescene" in the headset and pick CONNECT; the app finds this PC.
#
#   examples/isaaclab_g1/run.sh [--operator-height 1.82] [--record takes/] [--controller pose] [--seconds N]
#   examples/isaaclab_g1/run.sh --scene busybox                         a BusyBox scene; the headset menu switches them
#   examples/isaaclab_g1/run.sh --controller supermimic [--follow]      SuperMimic (scripts/download_trackers.sh)
#   examples/isaaclab_g1/run.sh --controller tracker --tracker twist2   one of specs/trackers
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
SCENE=table; ARGS=()
while (($#)); do case $1 in --scene) SCENE=$2; shift 2 ;; *) ARGS+=("$1"); shift ;; esac; done
if [[ $SCENE == table ]]; then
    exec "$PYTHON" "$REPO/examples/isaaclab_g1/teleop.py" "${ARGS[@]}"
fi
# A BusyBox scene picked in the headset menu ends the run with its name in $CHOICE; start again in it. The app reconnects.
CHOICE=$(mktemp "${TMPDIR%/}/telescene_scene.XXXXXX")
trap 'rm -f "$CHOICE"' EXIT
while true; do
    echo "[telescene] scene $SCENE"
    : > "$CHOICE"
    "$PYTHON" "$REPO/examples/isaaclab_g1/teleop.py" "${ARGS[@]}" --scene "$SCENE" --scene-choice-file "$CHOICE" || exit $?
    [[ -s $CHOICE ]] || break
    SCENE=$(cat "$CHOICE")
done
