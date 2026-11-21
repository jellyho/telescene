#!/usr/bin/env bash
# Install the app on a PICO connected over USB (developer mode on, USB debugging allowed), and tell it this PC's
# address. Then put the headset on and open "Telescene" from the app library yourself.
#
#   apps/pico/install.sh [APK]       default: build/telescene.apk, built first if missing
#   PC_IP=192.168.0.10 apps/pico/install.sh     when the PC has several networks
#
# Without the address the app finds the PC by a UDP broadcast, which some networks block.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
XRT=${TELESCENE_ANDROID:-${XRT:-$HOME/.local/share/telescene-android}}
ADB=${ADB:-$(command -v adb || echo "$XRT/sdk/platform-tools/adb")}
APK=${1:-build/telescene.apk}
[[ -f $APK ]] || ./build.sh
"$ADB" get-state >/dev/null || { echo "no headset on adb: plug it in and allow USB debugging in the headset" >&2; exit 1; }
"$ADB" install -r "$APK" | tail -1
PC_IP=${PC_IP:-$(ip -4 -o route get 1.1.1.1 | sed -E 's/.* src ([0-9.]+).*/\1/')}
"$ADB" shell setprop debug.telescene.pc "$PC_IP"
echo "installed; the app will connect to $PC_IP. setprop does not survive a headset reboot: run this again after one."
