#!/usr/bin/env bash
# Install what build.sh needs, without root, into $TELESCENE_ANDROID (default ~/.local/share/telescene-android):
# JDK 17 (Temurin), Android command-line tools, platform 34, build-tools 34.0.0, NDK 26.3, platform-tools (adb),
# and the Khronos OpenXR loader for Android 1.1.49. About 2.5 GB. Re-running skips what is there.
#
# sdkmanager asks you to accept the Android SDK licences.
set -euo pipefail
XRT=${TELESCENE_ANDROID:-${XRT:-$HOME/.local/share/telescene-android}}
mkdir -p "$XRT"; cd "$XRT"
JDK_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.21%2B9/OpenJDK17U-jdk_x64_linux_hotspot_17.0.21_9.tar.gz"
TOOLS_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
OXR_URL="https://repo1.maven.org/maven2/org/khronos/openxr/openxr_loader_for_android/1.1.49/openxr_loader_for_android-1.1.49.aar"
OXR_SHA=9a731db8a21bfc5c4be47c8862a508a2e4db3d7167062dc65683cedf7563f0cd

if ! ls -d jdk-17* >/dev/null 2>&1; then
    echo "[toolchain] JDK 17"; curl -fL "$JDK_URL" | tar -xz
fi
JAVA_HOME=$(echo "$XRT"/jdk-17*); export JAVA_HOME PATH="$JAVA_HOME/bin:$PATH"
if [[ ! -x sdk/cmdline-tools/latest/bin/sdkmanager ]]; then
    echo "[toolchain] Android command-line tools"
    curl -fL -o tools.zip "$TOOLS_URL"; rm -rf sdk/cmdline-tools; mkdir -p sdk/cmdline-tools
    unzip -q tools.zip -d sdk/cmdline-tools && mv sdk/cmdline-tools/cmdline-tools sdk/cmdline-tools/latest && rm tools.zip
fi
echo "[toolchain] SDK packages"
yes | sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$XRT/sdk" --licenses >/dev/null || true
sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$XRT/sdk" \
    "platforms;android-34" "build-tools;34.0.0" "ndk;26.3.11579264" "platform-tools"
if [[ ! -f oxr/l.aar ]]; then
    echo "[toolchain] OpenXR loader"
    mkdir -p oxr; curl -fL -o oxr/l.aar.part "$OXR_URL"
    echo "$OXR_SHA  oxr/l.aar.part" | sha256sum -c --status || { echo "OpenXR loader sha256 mismatch" >&2; exit 1; }
    mv oxr/l.aar.part oxr/l.aar
fi
[[ -d oxr/aar ]] || (mkdir -p oxr/aar && cd oxr/aar && unzip -q ../l.aar)
echo "[toolchain] done: $XRT. adb is $XRT/sdk/platform-tools/adb"
