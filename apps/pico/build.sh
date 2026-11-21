#!/usr/bin/env bash
# Build build/telescene.apk without Gradle, from main.c and two small Java classes. Needs the JDK, SDK, NDK and
# OpenXR loader that setup_toolchain.sh installs into $TELESCENE_ANDROID (default ~/.local/share/telescene-android).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
XRT=${TELESCENE_ANDROID:-${XRT:-$HOME/.local/share/telescene-android}}; SDK=$XRT/sdk; BT=$SDK/build-tools/34.0.0
[[ -d $SDK/ndk ]] || { echo "no Android toolchain in $XRT: run ./setup_toolchain.sh first" >&2; exit 1; }
NDK=$(echo "$SDK"/ndk/*); TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
export JAVA_HOME=$(echo "$XRT"/jdk-*) PATH="$(echo "$XRT"/jdk-*)/bin:$PATH"
OXR=$XRT/oxr/aar; INC=$(dirname "$(dirname "$(find "$OXR" -name openxr.h | head -1)")"); LOADER=$(find "$OXR" -path '*arm64*' -name libopenxr_loader.so | head -1)
OUT=build; rm -rf $OUT; mkdir -p $OUT/lib/arm64-v8a
GLUE=$NDK/sources/android/native_app_glue
"$TC/aarch64-linux-android29-clang" -shared -fPIC -O2 -Wall -Wno-unused-result -fno-strict-aliasing -I"$INC" -I"$GLUE" main.c "$GLUE/android_native_app_glue.c" \
  -u ANativeActivity_onCreate -L"$(dirname "$LOADER")" -lopenxr_loader -landroid -llog -lEGL -lGLESv3 -lm -o $OUT/lib/arm64-v8a/libtelescene.so
cp "$LOADER" $OUT/lib/arm64-v8a/
JAR=$SDK/platforms/android-34/android.jar
# MainActivity and CaptureService: screen capture of each take (MediaProjection has no native API).
mkdir -p $OUT/classes
javac -source 8 -target 8 -nowarn -Xlint:-options -bootclasspath "$JAR" -d $OUT/classes $(find java -name '*.java')
"$BT/d8" --lib "$JAR" --min-api 29 --output $OUT $(find $OUT/classes -name '*.class')
"$BT/aapt2" compile --dir res -o $OUT/res.zip  # the launcher icon; tools/make_icon.py draws it
"$BT/aapt2" link -o $OUT/u.apk -I "$JAR" --manifest AndroidManifest.xml -R $OUT/res.zip
(cd $OUT && zip -q -r -0 u.apk lib classes.dex)
"$BT/zipalign" -f -p 4 $OUT/u.apk $OUT/a.apk
KS=$XRT/debug.keystore; [ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias d -dname CN=debug -keyalg RSA -keysize 2048 -validity 10000
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --out $OUT/telescene.apk $OUT/a.apk
echo "built $OUT/telescene.apk"
